"""T2.9 baselines and the training-set rule they share with T2.10.

Hand-computed numbers are worked in the comments from Section 5.4. The
fitted sum is checked against a vertex enumeration of its linear program
(every choice of three tight conditions, solved and kept if feasible),
which shares no code with scipy's linprog.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from attnbench.predict.baselines import (
    LABELS, fit_fitted_sum, step_form, steps, sum_as_written,
    sum_plus_s0)
from attnbench.predict.recursion import (
    CostModelError, close_prefill, copy_op, kernel_op, price, recurse,
    sync_op)
from attnbench.predict.training import (
    COMMITTED, GROUND_TRUTH_STATISTIC, RETROSPECTIVE, STATISTICS, Cell,
    GroundTruthRow, HardwareFeatures,
    committed_training_set, retrospective_training_set)

from test_predict_recursion import COSTS

HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)


def P(ops):
    return price(ops, COSTS)


# layer 0: K(h2,d3), K(h1,d1), S(item, 4 bytes, s0 = 2)
# layer 1: C(h1,l4), K(h5,d1)
# closing synchronise (s0 = 1, no layer)
LAYERED = close_prefill([
    kernel_op(2, 3, layer=0), kernel_op(1, 1, layer=0),
    sync_op("item", payload_bytes=4, layer=0),
    copy_op(1, 4, layer=1), kernel_op(5, 1, layer=1),
])


def test_sum_as_written_leaves_out_s0():
    # h 2+1+1+5 = 9, l 4, d 3+1+1 = 5: 18.  s0 (2 + 1) is not in it.
    assert sum_as_written(P(LAYERED)) == 18


def test_sum_plus_s0_adds_every_synchronisation_cost():
    assert sum_plus_s0(P(LAYERED)) == 21


def test_step_form_takes_the_larger_side_of_each_layer_plus_the_tail():
    # layer 0: sum h 3, sum l + d 4 -> 4.  layer 1: sum h 6, sum l + d 5 -> 6.
    # tail: s0 2 + 1 = 3.  Step form = 4 + 6 + 3 = 13.
    p = P(LAYERED)
    assert steps(p) == [(0, 3, 4), (1, 6, 5)]
    assert step_form(p) == 13


def test_a_chunked_prefill_has_one_step_per_visit_to_a_layer():
    # Layers 0, 1, 0, 1 (two chunks): four steps, not two.
    # K(h3,d1) -> 3; K(h1,d4) -> 4; K(h2,d2) -> 2; K(h1,d5) -> 5.  Sum 14.
    ops = [kernel_op(3, 1, layer=0), kernel_op(1, 4, layer=1),
           kernel_op(2, 2, layer=0), kernel_op(1, 5, layer=1)]
    assert [s[0] for s in steps(P(ops))] == [0, 1, 0, 1]
    assert step_form(P(ops)) == 14
    # Grouping by label instead would give max(5, 3) + max(2, 9) = 14 here
    # too, so a case where the two differ:
    # layer 0 visits: (h4,d0), (h0,d4) -> 4 + 4 = 8; grouped: max(4,4) = 4.
    ops = [kernel_op(4, 0, layer=0), kernel_op(1, 1, layer=1),
           kernel_op(0, 4, layer=0)]
    assert step_form(P(ops)) == 4 + 1 + 4


def test_a_synchronisation_never_splits_a_step():
    ops = [kernel_op(1, 2, layer=0), sync_op("synchronize", layer=None),
           kernel_op(3, 1, layer=0)]
    # One step: sum h 4, sum d 3 -> 4; tail 1.
    assert steps(P(ops)) == [(0, 4, 3)]
    assert step_form(P(ops)) == 5


def test_step_form_needs_a_layer_on_every_kernel_and_copy():
    with pytest.raises(CostModelError, match="no layer label"):
        step_form(P([kernel_op(1, 1, layer=0), copy_op(1, 1)]))


def test_labels_keep_the_two_sums_apart():
    assert "no s_0" in LABELS["sum"] and "not in the plan" in LABELS["sum_plus_s0"]
    assert "no s_0" in LABELS["fitted_sum"]


def random_layered(rng, n):
    ops, layer = [], 0
    for _ in range(n):
        if rng.random() < 0.2:
            layer += 1
        r = rng.random()
        if r < 0.6:
            ops.append(kernel_op(rng.randint(0, 6), rng.randint(0, 9), layer=layer))
        elif r < 0.8:
            ops.append(copy_op(rng.randint(0, 3), rng.randint(0, 5), layer=layer))
        else:
            ops.append(sync_op("item", payload_bytes=rng.choice([0, 4]),
                               layer=layer))
    return close_prefill(ops)


@pytest.mark.parametrize("seed", range(50))
def test_baseline_orderings(seed):
    rng = random.Random(3000 + seed)
    p = P(random_layered(rng, rng.randint(1, 30)))
    H, LD, S = sum(p.host), sum(p.link) + sum(p.device), sum(p.sync_cost)
    sf = step_form(p)
    # Per-step maxima lie between the max of the totals and their sum.
    assert max(H, LD) + S <= sf <= sum_plus_s0(p)
    assert sum_as_written(p) <= sum_plus_s0(p)
    # The recursion overlaps host and device, so it never exceeds Sum + s_0.
    assert recurse(p, None).end_to_end <= sum_plus_s0(p)


# --------------------------------------------------------------------------
# Training sets
# --------------------------------------------------------------------------

def row(pairing, ops, measured, arm="dense", L=8192, statistic="median"):
    cell = Cell(pairing=pairing, model_name="m-small", code_path="cp-a", arm=arm,
                context_length=L, density=1.0, batch=1, priced=P(ops),
                hardware=HW)
    return GroundTruthRow(cell=cell, measured=measured, statistic=statistic)


ROWS = [row("P1", [kernel_op(1, 1)], 2.0), row("P1", [kernel_op(2, 1)], 3.0),
        row("P2", [kernel_op(1, 2)], 3.0), row("P4", [kernel_op(1, 3)], 4.0)]


def test_committed_set_uses_only_the_banked_pairings():
    ts = committed_training_set(ROWS, target="P4", banked=["P2", "P1"])
    assert ts.kind == COMMITTED
    assert ts.trained_on == ("P1", "P2")
    assert {r.pairing for r in ts.rows} == {"P1", "P2"}
    # P2 from P1 alone, as in the planned run order.
    ts = committed_training_set(ROWS, target="P2", banked=["P1"])
    assert ts.trained_on == ("P1",) and len(ts.rows) == 2


def test_committed_set_refusals():
    with pytest.raises(CostModelError, match="its own prediction"):
        committed_training_set(ROWS, target="P2", banked=["P1", "P2"])
    with pytest.raises(CostModelError, match="nothing to fit"):
        committed_training_set(ROWS, target="P1", banked=[])
    with pytest.raises(CostModelError, match="have no ground-truth rows"):
        committed_training_set(ROWS, target="P4", banked=["P1", "P3"])
    with pytest.raises(CostModelError, match="positive finite"):
        committed_training_set(ROWS + [row("P1", [kernel_op(1, 1)], 0.0)],
                               target="P2", banked=["P1"])


def test_retrospective_set_is_every_other_pairing_and_says_so():
    ts = retrospective_training_set(ROWS, target="P2")
    assert ts.kind == RETROSPECTIVE
    assert ts.trained_on == ("P1", "P4")
    with pytest.raises(CostModelError, match="no pairing other than"):
        retrospective_training_set([ROWS[0]], target="P1")


def test_the_statistic_is_the_median_unless_a_row_says_otherwise():
    # Section 5.3's ground truth reports median, p90 and p99 and names none.
    assert GROUND_TRUTH_STATISTIC == "median" and STATISTICS == ("median", "p90", "p99")
    assert ROWS[0].statistic == "median"
    assert committed_training_set(ROWS, "P4", ["P1"]).statistic == "median"
    p90 = [row("P1", [kernel_op(1, 1)], 2.5, statistic="p90")]
    assert committed_training_set(p90, "P2", ["P1"]).statistic == "p90"


def test_rows_that_mix_statistics_or_name_another_are_refused():
    mixed = ROWS + [row("P1", [kernel_op(1, 1)], 2.5, statistic="p99")]
    with pytest.raises(CostModelError, match="mix ground-truth statistics"):
        committed_training_set(mixed, "P4", ["P1"])
    with pytest.raises(CostModelError, match="mix ground-truth statistics"):
        retrospective_training_set(mixed, "P2")
    with pytest.raises(CostModelError, match="statistic must be one of"):
        committed_training_set([row("P1", [kernel_op(1, 1)], 2.0,
                                    statistic="mean")], "P2", ["P1"])


# --------------------------------------------------------------------------
# Fitted sum (loss: sum of absolute relative errors)
# --------------------------------------------------------------------------

def terms_row(pairing, H, L, D, y, n_sync=0):
    ops = [kernel_op(H, D, layer=0)]
    if L:
        ops.append(copy_op(0, L, layer=0))
    ops += [sync_op("synchronize") for _ in range(n_sync)]   # s0 1 each
    return row(pairing, ops, y)


def committed(rows):
    return committed_training_set(rows, target="P2", banked=["P1"])


def test_fitted_sum_recovers_exact_coefficients():
    # y = 2 sum h + 0.5 sum l + 3 sum d on four rows with independent terms:
    # zero loss, and the exact fit is the only one.
    data = [(1, 2, 1), (2, 1, 3), (3, 4, 1), (1, 1, 1)]
    rows = [terms_row("P1", H, L, D, 2 * H + 0.5 * L + 3 * D) for H, L, D in data]
    f = fit_fitted_sum(committed(rows))
    assert (f.a, f.b, f.c) == pytest.approx((2, 0.5, 3), abs=1e-9)
    assert f.training_loss == pytest.approx(0, abs=1e-12)
    assert f.trained_on == ("P1",) and f.kind == COMMITTED
    assert f.statistic == "median" and f.label == "fitted_sum"
    assert f.unidentified == () and f.condition_number < 1e3
    assert f.predict(P([kernel_op(1, 1), copy_op(0, 2)])) == pytest.approx(6)


def test_fitted_sum_minimises_absolute_relative_error_not_its_square():
    # Host only: H = (1, 1, 1), y = (1, 2, 4).
    # sum |a - y_i| / y_i = |a-1| + |a-2|/2 + |a-4|/4: slope -1.75 below 1,
    # +0.25 between 1 and 2, so a = 1 with loss 0 + 0.5 + 0.75 = 1.25.
    # (Least squares on relative error would give
    #  sum(1/y)/sum(1/y^2) = 1.75/1.3125 = 4/3 instead.)
    rows = [terms_row("P1", 1, 0, 0, y) for y in (1.0, 2.0, 4.0)]
    f = fit_fitted_sum(committed(rows))
    assert f.a == pytest.approx(1) and f.training_loss == pytest.approx(1.25)
    assert f.unidentified == ("link", "device")
    assert f.condition_number == pytest.approx(1.0)


def test_fitted_sum_clamps_a_negative_coefficient():
    # (H, D, y) = (1, 0, 1), (1, 1, 0.5).  The exact fit is a = 1, c = -0.5.
    # With c = 0: |a - 1| + |a - 0.5| / 0.5, slope -3 below 0.5 and +1 above,
    # so a = 0.5 with loss 0.5; raising c from 0 only adds to the second term.
    rows = [terms_row("P1", 1, 0, 0, 1.0), terms_row("P1", 1, 0, 1, 0.5)]
    f = fit_fitted_sum(committed(rows))
    assert (f.a, f.b, f.c) == pytest.approx((0.5, 0, 0), abs=1e-9)
    assert f.training_loss == pytest.approx(0.5)
    assert f.unidentified == ("link",)


def test_fitted_sum_plus_s0_fits_the_remainder_and_adds_s0_back():
    # Host only, one synchronise each (s0 = 1): H = (1, 1), y = (3, 5).
    # As written: |a - 3|/3 + |a - 5|/5, slope +2/15 above 3 -> a = 3.
    # Plus s0:    |a + 1 - 3|/3 + |a + 1 - 5|/5 -> a = 2, and a cell with
    #             H = 1, S = 1 predicts 2 + 1 = 3.
    rows = [terms_row("P1", 1, 0, 0, 3.0, n_sync=1),
            terms_row("P1", 1, 0, 0, 5.0, n_sync=1)]
    plain = fit_fitted_sum(committed(rows))
    plus = fit_fitted_sum(committed(rows), add_s0=True)
    assert plain.a == pytest.approx(3) and plain.label == "fitted_sum"
    assert plus.a == pytest.approx(2) and plus.label == "fitted_sum_plus_s0"
    cell = P([kernel_op(1, 0), sync_op("synchronize")])
    assert plain.predict(cell) == pytest.approx(3)
    assert plus.predict(cell) == pytest.approx(3)
    assert plus.predict(P([kernel_op(1, 0), sync_op("synchronize"),
                           sync_op("synchronize")])) == pytest.approx(4)


def test_condition_number_hand_computed():
    # Rows (H, D, y) = (1, 0, 1) and (0, 4, 2): weighted design
    # [[1, 0], [0, 2]], singular values 2 and 1, condition number 2.
    rows = [terms_row("P1", 1, 0, 0, 1.0), terms_row("P1", 0, 0, 4, 2.0)]
    f = fit_fitted_sum(committed(rows))
    assert f.condition_number == pytest.approx(2.0)
    assert (f.a, f.c) == pytest.approx((1.0, 0.5))
    # sum h equal to sum d on every row: the two cannot be told apart.
    rows = [terms_row("P1", 1, 0, 1, 2.0), terms_row("P1", 2, 0, 2, 3.0)]
    assert fit_fitted_sum(committed(rows)).condition_number == float("inf")


def test_an_unidentified_term_is_fixed_at_zero_not_left_to_the_solver(monkeypatch):
    # No copies anywhere, so sum l is zero on every row and b is fixed at
    # 0 by its bounds; a solver free to pick b would make it arbitrary.
    import attnbench.predict.baselines as bl
    seen = {}
    real = bl.linprog

    def spy(*args, **kw):
        seen["bounds"] = kw["bounds"]
        return real(*args, **kw)

    monkeypatch.setattr(bl, "linprog", spy)
    rows = [terms_row("P1", 1, 0, 1, 2.0), terms_row("P1", 2, 0, 1, 3.0)]
    f = bl.fit_fitted_sum(committed(rows))
    assert seen["bounds"][:3] == [(0, None), (0, 0), (0, None)]
    assert f.b == 0 and f.unidentified == ("link",)


def test_labels_name_every_baseline():
    assert set(LABELS) == {"sum", "sum_plus_s0", "step_form", "fitted_sum",
                           "fitted_sum_plus_s0"}
    assert "not fitted" in LABELS["fitted_sum_plus_s0"]


def exhaustive_l1(X, r):
    """Least absolute deviations over beta >= 0 by enumerating vertices:
    each choice of three tight conditions from {row i fitted exactly} and
    {beta_j = 0}, solved, kept if feasible. Needs X of full column rank."""
    n = X.shape[0]
    conds = [("row", i) for i in range(n)] + [("zero", j) for j in range(3)]
    best = None
    for trio in itertools.combinations(conds, 3):
        M = np.zeros((3, 3)); v = np.zeros(3)
        for k, (kind, idx) in enumerate(trio):
            if kind == "row":
                M[k] = X[idx]; v[k] = r[idx]
            else:
                M[k, idx] = 1.0
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        beta = np.linalg.solve(M, v)
        if np.any(beta < -1e-12):
            continue
        obj = float(np.abs(X @ np.clip(beta, 0, None) - r).sum())
        best = obj if best is None else min(best, obj)
    return best


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("add_s0", [False, True])
def test_fitted_sum_matches_the_vertex_enumeration(seed, add_s0):
    rng = random.Random(4000 + seed)
    rows = []
    for _ in range(rng.randint(4, 9)):
        H, L, D = rng.randint(0, 8), rng.randint(0, 5), rng.randint(1, 8)
        y = abs(rng.gauss(1.0, 0.6)) * H + abs(rng.gauss(0.5, 1.0)) * L \
            + rng.gauss(1.0, 1.5) * D + rng.uniform(0.5, 3)
        rows.append(terms_row("P1", H, L, D, max(y, 0.25),
                              n_sync=rng.randint(0, 2)))
    f = fit_fitted_sum(committed(rows), add_s0=add_s0)
    A = np.array([[sum(r.cell.priced.host), sum(r.cell.priced.link),
                   sum(r.cell.priced.device)] for r in rows], dtype=float)
    y = np.array([r.measured for r in rows])
    S = np.array([sum(r.cell.priced.sync_cost) if add_s0 else 0.0 for r in rows])
    X, rhs = A / y[:, None], (y - S) / y
    assert min(f.a, f.b, f.c) >= 0
    if np.linalg.matrix_rank(X) < 3:
        pytest.skip("rank-deficient draw; the vertex enumeration needs full rank")
    assert f.training_loss == pytest.approx(exhaustive_l1(X, rhs), rel=1e-7, abs=1e-9)
