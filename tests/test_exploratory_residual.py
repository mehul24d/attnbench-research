"""EXPLORATORY residual split (the adviser's task of 22 Nov 2026, item 2):
every function behind a reported number, on fixtures worked out by hand in
the comments. Costs are small integers, so the comparisons are exact.
"""

from __future__ import annotations

import pytest

from attnbench.exploratory import residual as RS
from attnbench.predict.recursion import MechanismCost, close_prefill, kernel_op, price

COSTS = {"synchronize": MechanismCost(s_fixed=1)}


def priced(ops):
    return price(close_prefill(ops), COSTS)


# A: kernels (h, d) = (1, 5), (1, 5), (3, 1), then the closing synchronise
# (s_0 = 1), no queue term. H: 1, 2, 5; D: 6, 11, 12. The close waits from
# H = 5 to D = 12 (7), then adds s_0: 13. Host 5 + s_0 1 + wait 7 = 13.
A = [kernel_op(1, 5), kernel_op(1, 5), kernel_op(3, 1)]


def test_terms_add_to_the_prediction():
    t = RS.terms(priced(A), None)
    assert (t["host"], t["sync_cost"], t["queue_stall"], t["sync_wait"]) == (5, 1, 0, 7)
    assert t["predicted"] == 13
    assert t["device_chain"] == 11 and t["predicted_idle"] == 2
    assert t["n_sync"] == 1


def test_terms_count_the_queue_stall():
    # Q = 1, kernels (1, 5) twice. k0: H 1, D 6. k1: H 2 waits for D0 = 6
    # (stall 4), D 11. Close: H 6 waits for D1 = 11 (stall 5), no sync wait,
    # + s_0: 12. Host 2 + s_0 1 + stall 9 + wait 0 = 12.
    t = RS.terms(priced([kernel_op(1, 5), kernel_op(1, 5)]), 1)
    assert (t["host"], t["sync_cost"], t["queue_stall"], t["sync_wait"]) == (2, 1, 9, 0)
    assert t["predicted"] == 12


def test_terms_refuse_a_sequence_without_its_closing_synchronise():
    # (1, 5) alone: D_N = 6, H_N = 1, so the four terms cannot add up.
    with pytest.raises(ValueError, match="closing synchronise"):
        RS.terms(price([kernel_op(1, 5)], COSTS), None)


def test_busy_is_the_union_of_intervals():
    assert RS.busy([(0, 2), (1, 3), (5, 6)]) == 4      # [0, 3] and [5, 6]
    assert RS.busy([(0, 1), (0.5, 0.75)]) == 1          # nested
    assert RS.busy([(5, 6), (0, 1)]) == 2               # unsorted, disjoint
    assert RS.busy([(0, 1), (1.5, 2)]) == 1.5           # a gap shorter than 1 is a gap
    assert RS.busy([]) == 0
    with pytest.raises(ValueError):
        RS.busy([(2, 1)])


# B: kernels (4, 1) twice. With every h times k (k >= 0.25): H 4k, 8k;
# D 4k + 1, 8k + 1; the close waits 1 and adds s_0: 8k + 2. At k = 0:
# D 1, 2; the close at H 0 waits 2, + 1: 3.
B = [kernel_op(4, 1), kernel_op(4, 1)]


def test_host_scale_to_match_solves_the_recursion():
    assert RS.host_scale_to_match(priced(B), None, 6.0) == pytest.approx(0.5, abs=1e-12)
    assert RS.host_scale_to_match(priced(B), None, 10.0) == pytest.approx(1.0, abs=1e-12)


def test_host_scale_to_match_is_none_out_of_reach():
    assert RS.host_scale_to_match(priced(B), None, 2.0) is None      # below 3, at k = 0
    assert RS.host_scale_to_match(priced(B), None, 100.0) is None    # above 34, at k = 4


def test_counterfactuals_remove_one_term_each():
    # A, measured 12: as predicted 13 (+1/12); s_0 = 0 gives 12 (0); the
    # no-wait ablation keeps the closing synchronise's wait, so 13.
    c = RS.counterfactuals(priced(A), None, 12.0)
    assert c["as_predicted"] == pytest.approx(1 / 12)
    assert c["sync_cost_zero"] == pytest.approx(0.0)
    assert c["no_sync_wait"] == pytest.approx(1 / 12)
    assert c["no_queue_term"] == pytest.approx(1 / 12)
    # With Q = 1 on the two (1, 5) kernels the stall is in the wait's place:
    # 12 either way, so removing the queue term changes nothing.
    two = priced([kernel_op(1, 5), kernel_op(1, 5)])
    q = RS.counterfactuals(two, 1, 12.0)
    assert q["as_predicted"] == pytest.approx(0.0) and q["no_queue_term"] == pytest.approx(0.0)
    # Q = 1 on (1, 5), (1, 1), (6, 1): k1 stalls from H 2 to D0 = 6, so k2's
    # host work starts late: H 12, D 13; the close stalls to 13, + 1: 14.
    # Without the queue: H 1, 2, 8; D 6, 7, 9; the close waits 1, + 1: 10.
    late = priced([kernel_op(1, 5), kernel_op(1, 1), kernel_op(6, 1)])
    t = RS.terms(late, 1)
    assert (t["host"], t["queue_stall"], t["sync_wait"], t["predicted"]) == (8, 5, 0, 14)
    q = RS.counterfactuals(late, 1, 14.0)
    assert q["as_predicted"] == pytest.approx(0.0)
    assert q["no_queue_term"] == pytest.approx(10 / 14 - 1)


def test_split_cell_error_is_the_idle_time_error():
    # A, measured 12, traced busy 11: error 1; idle predicted 2, measured 1.
    s = RS.split_cell(priced(A), None, 12.0, 11.0)
    assert s["error_s"] == 1 and s["measured_idle"] == 1 and s["idle_error_s"] == 1
    assert s["chain_minus_busy"] == 0
    assert s["host_over_measured"] == pytest.approx(5 / 12)
    # A chain longer than the busy time shifts the identity by that gap.
    s = RS.split_cell(priced(A), None, 12.0, 10.0)
    assert s["idle_error_s"] == s["error_s"] - s["chain_minus_busy"] == 0
    with pytest.raises(ValueError):
        RS.split_cell(priced(A), None, 0.0, 1.0)


def test_per_kernel_excess_pools_gemm_and_other_device_only():
    p = priced([kernel_op(1, 4), kernel_op(1, 2), kernel_op(1, 8)])
    fam = ["kernel:aten::mm", "kernel:aten::add", "kernel:flash_attn_3::fwd", "sync:closing"]
    # mm 4 -> 6 (+2), add 2 -> 6 (+4); attention and the sync are left out.
    r = RS.per_kernel_excess(p, [6.0, 6.0, 1.0, None], fam)
    assert r == {"positions": 2, "excess_s_per_kernel": 3.0}
    # An unmeasured position is skipped, not counted as zero.
    r = RS.per_kernel_excess(p, [6.0, None, 1.0, None], fam)
    assert r == {"positions": 1, "excess_s_per_kernel": 2.0}
    assert RS.per_kernel_excess(p, [None] * 4, fam)["excess_s_per_kernel"] is None


def test_share_of_error_counts_only_overpredicted_cells():
    cells = [{"error_s": 2.0, "host": 4.0, "sync_cost": 0.5, "queue_stall": 0.0, "sync_wait": 1.0},
             {"error_s": -1.0, "host": 9.0, "sync_cost": 9.0, "queue_stall": 9.0, "sync_wait": 9.0},
             {"error_s": 1.0, "host": 8.0, "sync_cost": 0.5, "queue_stall": 1.0, "sync_wait": 0.0}]
    s = RS.share_of_error(cells)
    assert (s["n_overpredicted"], s["n"]) == (2, 3)
    assert s["error_s"] == 3.0 and s["host"] == 12.0 and s["sync_cost"] == 1.0
    assert s["queue_stall"] == 1.0 and s["error_over_host"] == 0.25
