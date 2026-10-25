"""T2.25: the delay-injection slope check of Gate A, on CPU.

Variant times are worked by hand from Section 5.4 with the costs of
test_predict_recursion (synchronize s_fixed 1; item 1 + payload / 4). The
idle slope is checked against s_0's own formula, mechanism by mechanism.
"""

from __future__ import annotations

import pytest

from attnbench.predict.delay_injection import (
    BUSY, IDLE, INJECTED_LABEL, START, busy_sites, gate, inject, ols_slope,
    plan_variants, slope_for, spread)
from attnbench.predict.recursion import (
    SYNC_MECHANISMS, CostModelError, MechanismCost, OpKind, close_prefill,
    kernel_op, s0, sync_op)

from test_predict_recursion import COSTS

# Dense arm: three K(h1, d10), then the closing synchronise.
# H 1, 2, 3; D 11, 21, 31; close: max(3, 31) + 1 = 32.
DENSE = close_prefill([kernel_op(1, 10)] * 3)


def test_busy_sites_are_where_the_device_is_still_working():
    assert busy_sites(DENSE, COSTS, None) == (0, 1, 2)
    # Host-bound: K(h10, d1) x3: H 10, 20, 30; D 11, 21, 31. Right after
    # each launch the device still holds that kernel, so every site is busy,
    # and a synchronisation there waits 1, not 10.
    host_bound = close_prefill([kernel_op(10, 1)] * 3)
    assert busy_sites(host_bound, COSTS, None) == (0, 1, 2)
    (_, one) = plan_variants(host_bound, "synchronize", BUSY, (0, 1), COSTS, None)
    assert one.predicted_waits == (1,)
    # A kernel with no device time leaves the device idle behind it.
    assert busy_sites(close_prefill([kernel_op(10, 0), kernel_op(1, 1)]),
                      COSTS, None) == (1,)


def test_spread_picks_evenly_spaced_distinct_sites():
    assert spread(tuple(range(10)), 3) == (1, 5, 8)
    assert spread((0, 1, 2), 1) == (1,) and spread((0, 1, 2), 2) == (0, 2)
    assert spread((4, 7), 0) == ()
    with pytest.raises(CostModelError, match="distinct busy sites"):
        spread((0, 1), 3)


def test_inject_places_synchronisations_after_their_sites():
    ops, pos = inject(DENSE, (START, 1, 1), "item", payload_bytes=4)
    kinds = [o.kind for o in ops]
    assert kinds == [OpKind.SYNC, OpKind.KERNEL, OpKind.KERNEL, OpKind.SYNC,
                     OpKind.SYNC, OpKind.KERNEL, OpKind.SYNC]
    assert pos == (0, 3, 4)
    assert all(ops[p].label == INJECTED_LABEL and ops[p].payload_bytes == 4
               for p in pos)
    assert ops[-1] == DENSE[-1]            # the closing synchronise stays last


def test_busy_variants_hand_computed():
    vs = plan_variants(DENSE, "synchronize", BUSY, (0, 1, 2, 3), COSTS, None)
    # c = 1, after op 1: K K S K close.  S waits 21 - 2 = 19, ends at 22;
    #   K: H 23, D 33; close: 33 + 1 = 34.
    # c = 2, after ops 0 and 2: S waits 10 (to 11), ends 12; K H 13 D 23;
    #   K H 14 D 33; S waits 19, ends 34; close 35.
    # c = 3, after 0, 1, 2: 12; K D 23; S at 13 waits 10, 24; K H 25 D 35;
    #   S 36; close 37.
    assert [v.predicted for v in vs] == [32, 34, 35, 37]
    assert [v.sites for v in vs] == [(), (1,), (0, 2), (0, 1, 2)]
    assert vs[1].predicted_waits == (19,)
    assert vs[2].predicted_waits == (10, 19)


def test_idle_variants_cost_s0_each():
    # c syncs at the start, each at an idle device: + 1 each.
    vs = plan_variants(DENSE, "synchronize", IDLE, (0, 1, 2, 4), COSTS, None)
    assert [v.predicted for v in vs] == [32, 33, 34, 36]
    assert all(w == 0 for v in vs for w in v.predicted_waits)


@pytest.mark.parametrize("mechanism", SYNC_MECHANISMS)
def test_the_idle_slope_is_s0_for_every_mechanism(mechanism):
    payload = 0 if mechanism == "synchronize" else 8
    vs = plan_variants(DENSE, mechanism, IDLE, (0, 1, 3), COSTS, None,
                       payload_bytes=payload)
    per_sync = s0(COSTS[mechanism], payload, False)
    assert ols_slope([v.count for v in vs], [v.predicted for v in vs]) \
        == pytest.approx(per_sync)


def test_a_perfect_model_has_slope_one_everywhere():
    vb = plan_variants(DENSE, "item", BUSY, (0, 1, 2, 3), COSTS, None,
                       payload_bytes=4)
    vi = plan_variants(DENSE, "item", IDLE, (0, 1, 2), COSTS, None,
                       payload_bytes=4)
    measured = {(v.placement, v.count): [v.predicted] * 3 for v in vb + vi}
    s = slope_for("item", vb + vi, measured)
    assert s.slope == pytest.approx(1.0)
    assert {p: x.ratio for p, x in s.placements.items()} == \
        pytest.approx({BUSY: 1.0, IDLE: 1.0})
    assert gate({"item": s}, bound=0.05, mechanisms=["item"]) == {"item": True}


def test_double_counting_shows_as_a_slope_of_two():
    # The model charges s_fixed 2 where the device pays 1: the "measured"
    # times come from the true costs, the predictions from the doubled ones.
    doubled = {**COSTS, "synchronize": MechanismCost(s_fixed=2)}
    pred = plan_variants(DENSE, "synchronize", IDLE, (0, 1, 2, 4), doubled, None)
    true = plan_variants(DENSE, "synchronize", IDLE, (0, 1, 2, 4), COSTS, None)
    measured = {(IDLE, v.count): [v.predicted] for v in true}
    s = slope_for("synchronize", pred, measured)
    assert s.slope == pytest.approx(2.0)
    assert s.placements[IDLE].ratio == pytest.approx(2.0)
    assert gate({"synchronize": s}, bound=0.5,
                mechanisms=["synchronize"]) == {"synchronize": False}


def test_the_pooled_slope_regresses_increments_through_the_origin():
    vb = plan_variants(DENSE, "synchronize", BUSY, (0, 1, 2, 3), COSTS, None)
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1, 2, 4), COSTS, None)
    # Measured: busy 32, 33, 34, 35 (increments 1, 2, 3 against predicted
    # 2, 3, 5); idle 32, 33, 34, 36 (increments 1, 2, 4, predicted the same).
    measured = {(BUSY, 0): [32], (BUSY, 1): [33], (BUSY, 2): [34], (BUSY, 3): [35],
                (IDLE, 0): [32], (IDLE, 1): [33], (IDLE, 2): [34], (IDLE, 4): [36]}
    s = slope_for("synchronize", vb + vi, measured)
    p, m = [2, 3, 5, 1, 2, 4], [1, 2, 3, 1, 2, 4]
    assert s.slope == pytest.approx(sum(a * b for a, b in zip(p, m))
                                    / sum(b * b for b in m))
    # Busy: predicted slope of (32, 34, 35, 37) on (0..3) is 1.6; measured 1.
    assert s.placements[BUSY].predicted_slope == pytest.approx(1.6)
    assert s.placements[BUSY].ratio == pytest.approx(1.6)
    assert s.placements[IDLE].ratio == pytest.approx(1.0)


def test_the_statistic_of_the_repeats_is_the_median():
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1), COSTS, None)
    s = slope_for("synchronize", vi, {(IDLE, 0): [32, 32, 99],
                                      (IDLE, 1): [33, 1, 33]})
    assert s.placements[IDLE].measured == (32, 33)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

def test_the_base_must_be_a_closed_prefill():
    with pytest.raises(CostModelError, match="closing synchronise"):
        busy_sites([kernel_op(1, 1)], COSTS, None)
    with pytest.raises(CostModelError, match="outside the arm"):
        inject(DENSE, (3,), "synchronize")     # 3 is the closing synchronise
    with pytest.raises(CostModelError, match="outside the arm"):
        inject(DENSE, (-2,), "synchronize")


@pytest.mark.parametrize("counts", [(1, 2), (0,), (0, 0, 1), (2, 0), (0, -1),
                                    (0, True)])
def test_counts_must_include_zero_and_ascend(counts):
    with pytest.raises(CostModelError, match="counts"):
        plan_variants(DENSE, "synchronize", IDLE, counts, COSTS, None)


def test_too_many_busy_injections_are_refused():
    with pytest.raises(CostModelError, match="distinct busy sites"):
        plan_variants(DENSE, "synchronize", BUSY, (0, 4), COSTS, None)


def test_a_busy_site_that_turns_idle_is_refused():
    # Base K(h1,d5) K(h1,d0) K(h1,d0): H 1, 2, 3; D 6, 6, 6. Busy after all
    # three, the last two only because of the first kernel's backlog.
    base = close_prefill([kernel_op(1, 5), kernel_op(1, 0), kernel_op(1, 0)])
    assert busy_sites(base, COSTS, None) == (0, 1, 2)
    # One injection, at spread((0, 1, 2), 1) = site 1: S at 2 waits to 6.
    plan_variants(base, "synchronize", BUSY, (0, 1), COSTS, None)
    # Two, at sites 0 and 2: the first drains the backlog (waits to 6, ends
    # 7); then K: H 8, D 8; K: H 9, D 9; the second S, at position 4 of the
    # injected sequence, is reached at 9 with D = 9: no wait.
    with pytest.raises(CostModelError, match=r"at \[4\] find the device idle"):
        plan_variants(base, "synchronize", BUSY, (0, 2), COSTS, None)


def test_a_queue_stall_counts_as_the_wait():
    # Q = 1, K(h1,d10) x3, one injection after op 1. Op 1 is issued at 11
    # (the queue held it until op 0 finished at 11) and done at 21. The
    # queue term then holds the sync from 11 to 21, so its sync wait is 0
    # and its queue stall, 10, is the whole wait.
    (_, one) = plan_variants(DENSE, "synchronize", BUSY, (0, 1), COSTS, 1)
    assert one.sites == (1,) and one.predicted_waits == (10,)


def test_an_idle_site_after_an_operation_with_no_device_time():
    # K(h1,d0) K(h1,d5): H 1, 2; D 1, 7. Idle after op 0, busy after op 1.
    z = close_prefill([kernel_op(1, 0), kernel_op(1, 5)])
    assert busy_sites(z, COSTS, None) == (1,)
    # Two syncs after op 0: 1 + 1 + 1 = 3; K(h1,d5): H 4, D 9; close 10.
    vs = plan_variants(z, "synchronize", IDLE, (0, 2), COSTS, None, idle_site=0)
    assert [v.predicted for v in vs] == [8, 10]
    with pytest.raises(CostModelError, match="still busy"):
        plan_variants(z, "synchronize", IDLE, (0, 1), COSTS, None, idle_site=1)


def test_idle_placement_refuses_a_busy_site():
    with pytest.raises(CostModelError, match="still busy"):
        plan_variants(DENSE, "synchronize", IDLE, (0, 1), COSTS, None,
                      idle_site=0)


def test_unknown_placement_is_refused():
    with pytest.raises(CostModelError, match="placement"):
        plan_variants(DENSE, "synchronize", "warm", (0, 1), COSTS, None)


def test_measurements_must_match_the_plan():
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1, 2), COSTS, None)
    with pytest.raises(CostModelError, match="no measurement for counts \\[2\\]"):
        slope_for("synchronize", vi, {(IDLE, 0): [32], (IDLE, 1): [33]})
    with pytest.raises(CostModelError, match="unplanned"):
        slope_for("synchronize", vi, {(IDLE, 0): [32], (IDLE, 1): [33],
                                      (IDLE, 2): [34], (BUSY, 1): [1]})
    with pytest.raises(CostModelError, match="among synchronize's"):
        slope_for("synchronize", vi + plan_variants(DENSE, "item", IDLE, (0, 1),
                                                    COSTS, None), {})
    with pytest.raises(CostModelError, match="no variants"):
        slope_for("item", (), {})


@pytest.mark.parametrize("samples", [[], [0.0], [float("nan")], [-1.0]])
def test_bad_repeats_are_refused(samples):
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1), COSTS, None)
    with pytest.raises(CostModelError, match="timed repeats"):
        slope_for("synchronize", vi, {(IDLE, 0): [32], (IDLE, 1): samples})


@pytest.mark.parametrize("times", [(32, 32, 31), (32, 32, 32)])
def test_a_measured_slope_that_does_not_rise_is_refused(times):
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1, 2), COSTS, None)
    with pytest.raises(CostModelError, match="does not rise"):
        slope_for("synchronize", vi, {(IDLE, c): [t] for c, t in
                                      zip((0, 1, 2), times)})


def test_a_sync_inside_the_base_is_never_a_busy_site():
    base = close_prefill([kernel_op(1, 10), sync_op("synchronize"),
                          kernel_op(1, 10)])
    # K: H 1, D 11; S: H = D = 12; K: H 13, D 23.
    assert busy_sites(base, COSTS, None) == (0, 2)


def _gate_at(pred_step, meas_step, bound):
    # Idle, counts (0, 1): predicted increment pred_step, measured meas_step,
    # so the pooled slope is pred_step / meas_step exactly.
    costs = {**COSTS, "synchronize": MechanismCost(s_fixed=pred_step)}
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1), costs, None)
    p0 = vi[0].predicted
    s = slope_for("synchronize", vi, {(IDLE, 0): [p0], (IDLE, 1): [p0 + meas_step]})
    return s.slope, gate({"synchronize": s}, bound=bound,
                         mechanisms=["synchronize"])["synchronize"]


def test_the_gate_bound_is_inclusive_and_two_sided():
    assert _gate_at(3, 2, 0.5) == (1.5, True)     # exactly on the bound
    assert _gate_at(3, 2, 0.25) == (1.5, False)
    assert _gate_at(1, 2, 0.25) == (0.5, False)   # under-prediction fails too
    assert _gate_at(1, 2, 0.5) == (0.5, True)


def test_gate_needs_every_mechanism_and_a_bound():
    vi = plan_variants(DENSE, "synchronize", IDLE, (0, 1), COSTS, None)
    s = slope_for("synchronize", vi, {(IDLE, 0): [32], (IDLE, 1): [33]})
    with pytest.raises(CostModelError, match="no slope for \\['item'\\]"):
        gate({"synchronize": s}, bound=0.1, mechanisms=["synchronize", "item"])
    for bad in (0, -0.1, float("nan"), True):
        with pytest.raises(CostModelError, match="slope bound"):
            gate({"synchronize": s}, bound=bad, mechanisms=["synchronize"])
    # The bound is inclusive: slope 1 + 0.1 passes at 0.1.
    s2 = slope_for("synchronize", vi, {(IDLE, 0): [32], (IDLE, 1): [32 + 1 / 1.1]})
    assert s2.slope == pytest.approx(1.1)
    assert gate({"synchronize": s2}, bound=0.1 + 1e-12,
                mechanisms=["synchronize"]) == {"synchronize": True}
    assert gate({"synchronize": s2}, bound=0.09,
                mechanisms=["synchronize"]) == {"synchronize": False}


def test_ols_slope_needs_two_counts():
    with pytest.raises(CostModelError, match="two distinct"):
        ols_slope([1, 1], [2, 3])
    assert ols_slope([0, 1, 2], [1, 3, 5]) == 2
