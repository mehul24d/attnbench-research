"""T2.9 baselines and the training-set rule they share with T2.10.

Hand-computed numbers are worked in the comments from Section 5.4. The
fitted sum is checked against an exhaustive solver (least squares on every
subset of active terms, keep the best non-negative one), which shares no
code with scipy's nnls.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from attnbench.predict.baselines import (
    LABELS, FittedSum, fit_fitted_sum, step_form, steps, sum_as_written,
    sum_plus_s0)
from attnbench.predict.recursion import (
    CostModelError, close_prefill, copy_op, kernel_op, price, recurse,
    sync_op)
from attnbench.predict.training import (
    COMMITTED, RETROSPECTIVE, Cell, GroundTruthRow, HardwareFeatures,
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
    assert set(LABELS) == {"sum", "sum_plus_s0", "step_form", "fitted_sum"}
    assert "no s_0" in LABELS["sum"] and "not in the plan" in LABELS["sum_plus_s0"]


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

def row(pairing, ops, measured, arm="dense", L=8192):
    cell = Cell(pairing=pairing, arm=arm, context_length=L, density=1.0,
                batch=1, priced=P(ops), hardware=HW)
    return GroundTruthRow(cell=cell, measured=measured)


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


# --------------------------------------------------------------------------
# Fitted sum
# --------------------------------------------------------------------------

def terms_row(pairing, H, L, D, y):
    ops = [kernel_op(H, D, layer=0)]
    if L:
        ops.append(copy_op(0, L, layer=0))
    return row(pairing, ops, y)


def test_fitted_sum_recovers_exact_coefficients():
    # y = 2 sum h + 0.5 sum l + 3 sum d on four rows with independent terms.
    data = [(1, 2, 1), (2, 1, 3), (3, 4, 1), (1, 1, 1)]
    rows = [terms_row("P1", H, L, D, 2 * H + 0.5 * L + 3 * D)
            for H, L, D in data]
    ts = committed_training_set(rows, target="P2", banked=["P1"])
    for loss in ("absolute", "relative"):
        f = fit_fitted_sum(ts, loss=loss)
        assert (f.a, f.b, f.c) == pytest.approx((2, 0.5, 3), abs=1e-9)
        assert f.trained_on == ("P1",) and f.kind == COMMITTED
        assert f.unidentified == ()
        assert f.predict(P([kernel_op(1, 1), copy_op(0, 2)])) == pytest.approx(6)


def test_fitted_sum_clamps_a_negative_coefficient_and_refits():
    # (H, D, y) = (1,1,1), (2,1,3), (1,2,0.5); no copies, so sum l = 0.
    # Unconstrained least squares: [6 5; 5 6][a c] = [7.5 5], a = 20/11,
    # c = -7.5/11 < 0.  With c = 0: a = sum(Hy)/sum(H^2) = 7.5/6 = 1.25.
    # KKT: residual (-0.25, 0.5, -0.75), D . r = -1.25 <= 0, so c stays 0.
    rows = [terms_row("P1", 1, 0, 1, 1), terms_row("P1", 2, 0, 1, 3),
            terms_row("P1", 1, 0, 2, 0.5)]
    f = fit_fitted_sum(retrospective_training_set(rows + [ROWS[2]], "P2"),
                       loss="absolute")
    assert (f.a, f.b, f.c) == pytest.approx((1.25, 0, 0), abs=1e-12)
    assert f.unidentified == ("link",)
    assert f.kind == RETROSPECTIVE


def test_relative_and_absolute_losses_differ_as_hand_computed():
    # Host only: H = (1, 2), y = (1, 4).
    # absolute: a = sum(Hy)/sum(H^2) = (1 + 8)/5 = 1.8.
    # relative: x = H/y = (1, 0.5); a = sum(x)/sum(x^2) = 1.5/1.25 = 1.2.
    rows = [row("P1", [kernel_op(1, 0, layer=0)], 1.0),
            row("P1", [kernel_op(2, 0, layer=0)], 4.0)]
    ts = committed_training_set(rows, target="P2", banked=["P1"])
    assert fit_fitted_sum(ts, loss="absolute").a == pytest.approx(1.8)
    assert fit_fitted_sum(ts, loss="relative").a == pytest.approx(1.2)
    assert fit_fitted_sum(ts, loss="relative").unidentified == ("link", "device")


@pytest.mark.parametrize("loss", [None, "l2", "rel"])
def test_fitted_sum_requires_a_stated_loss(loss):
    ts = committed_training_set(ROWS, target="P4", banked=["P1"])
    with pytest.raises(CostModelError, match="loss must be"):
        fit_fitted_sum(ts, loss=loss)


def exhaustive_nnls(A, y):
    best, best_x = None, None
    for k in range(4):
        for active in itertools.combinations(range(3), k):
            x = np.zeros(3)
            if active:
                sol, *_ = np.linalg.lstsq(A[:, active], y, rcond=None)
                if np.any(sol < 0):
                    continue
                x[list(active)] = sol
            obj = float(np.sum((A @ x - y) ** 2))
            if best is None or obj < best - 1e-12:
                best, best_x = obj, x
    return best, best_x


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("loss", ["absolute", "relative"])
def test_fitted_sum_matches_the_exhaustive_solver(seed, loss):
    rng = random.Random(4000 + seed)
    rows = []
    for _ in range(rng.randint(3, 9)):
        H, L, D = rng.randint(0, 8), rng.randint(0, 5), rng.randint(1, 8)
        y = abs(rng.gauss(1.0, 0.6)) * H + abs(rng.gauss(0.5, 1.0)) * L \
            + rng.gauss(1.0, 1.5) * D + rng.uniform(0.5, 3)
        rows.append(terms_row("P1", H, L, D, max(y, 0.25)))
    ts = committed_training_set(rows, target="P2", banked=["P1"])
    f = fit_fitted_sum(ts, loss=loss)
    A = np.array([[sum(r.cell.priced.host), sum(r.cell.priced.link),
                   sum(r.cell.priced.device)] for r in rows], dtype=float)
    y = np.array([r.measured for r in rows])
    if loss == "relative":
        A, y = A / y[:, None], np.ones_like(y)
    best, _ = exhaustive_nnls(A, y)
    ours = float(np.sum((A @ np.array([f.a, f.b, f.c]) - y) ** 2))
    assert min(f.a, f.b, f.c) >= 0
    assert ours == pytest.approx(best, rel=1e-9, abs=1e-12)
