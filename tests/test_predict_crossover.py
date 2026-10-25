"""T2.19: the crossover bracket rule, the quadratic fit, crossover error,
and the per-prediction errors (scoring.py).

Brackets are worked by hand on the plan's grid (2048 to 32768). The
quadratic is checked against numpy.polyfit on the raw lengths, which shares
no code with the scaled least-squares fit.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from attnbench.predict.crossover import (
    CROSSOVER, NONE_IN_RANGE, SPARSE_BECOMES_FASTER, SPARSE_BECOMES_SLOWER,
    UNRESOLVED, Bracket, CrossoverResult, GridDifference, bracket_of,
    crossover_brackets, crossover_error, differences_from_intervals,
    fit_quadratic, quadratic_crossovers, sign_at)
from attnbench.predict.intervals import Interval
from attnbench.predict.recursion import CostModelError
from attnbench.predict.scoring import (
    CORRECT, FALSE_PROFITABLE, MISSED_PROFITABLE, NOT_INTERPRETED,
    absolute_relative_error, decision_error, measured_profitable,
    predicted_profitable, sign_agreement)

GRID = (2048, 4096, 8192, 16384, 32768)


def diffs(*triples, grid=GRID):
    return [GridDifference(L, p, lo, hi) for L, (p, lo, hi) in zip(grid, triples)]


POS = (2.0, 1.0, 3.0)      # sparse slower, interval clear of 0
NEG = (-2.0, -3.0, -1.0)   # sparse faster
UND = (-0.2, -1.0, 0.5)    # interval straddles 0


# --------------------------------------------------------------------------
# The bracket rule
# --------------------------------------------------------------------------

def test_an_adjacent_sign_change_is_a_one_step_bracket():
    r = crossover_brackets(diffs(POS, POS, NEG, NEG, NEG), floor=0)
    assert r.status == CROSSOVER and r.signs == (1, 1, -1, -1, -1)
    assert r.brackets == (Bracket(4096, 8192, 1, 2, SPARSE_BECOMES_FASTER),)


def test_an_undetermined_length_widens_the_bracket():
    # Signs +, +, 0, -, -: the bracket runs from 4096 to 16384.
    r = crossover_brackets(diffs(POS, POS, UND, NEG, NEG), floor=0)
    assert r.brackets == (Bracket(4096, 16384, 1, 3, SPARSE_BECOMES_FASTER),)


def test_a_point_inside_the_floor_is_undetermined_even_with_a_clear_interval():
    small = (0.4, 0.1, 0.7)            # interval clear of 0, point below 0.5
    assert sign_at(GridDifference(8192, *small), 0.5) == 0
    assert sign_at(GridDifference(8192, *small), 0.3) == 1
    r = crossover_brackets(diffs(NEG, small, POS, POS, POS), floor=0.5)
    assert r.brackets == (Bracket(2048, 8192, 0, 2, SPARSE_BECOMES_SLOWER),)


def test_the_floor_is_strict_at_its_edge():
    assert sign_at(GridDifference(1, 0.5, 0.1, 0.9), 0.5) == 0
    assert sign_at(GridDifference(1, -0.5, -0.9, -0.1), 0.5) == 0


def test_two_crossovers_are_two_brackets():
    r = crossover_brackets(diffs(NEG, POS, POS, UND, NEG), floor=0)
    assert r.status == CROSSOVER
    assert r.brackets == (Bracket(2048, 4096, 0, 1, SPARSE_BECOMES_SLOWER),
                          Bracket(8192, 32768, 2, 4, SPARSE_BECOMES_FASTER))
    with pytest.raises(CostModelError, match="two crossovers"):
        bracket_of(r)


def test_none_in_range_needs_every_length_determined():
    r = crossover_brackets(diffs(POS, POS, POS, POS, POS), floor=0)
    assert r.status == NONE_IN_RANGE and r.brackets == ()
    assert bracket_of(r) is None


@pytest.mark.parametrize("signs", [
    (POS, POS, POS, POS, UND), (UND,) * 5, (POS, UND, POS, POS, POS),
    (UND, NEG, NEG, NEG, NEG)])
def test_without_a_determined_change_any_undetermined_length_is_unresolved(signs):
    assert crossover_brackets(diffs(*signs), floor=0).status == UNRESOLVED


@pytest.mark.parametrize("seed", range(40))
def test_brackets_match_a_scan_written_another_way(seed):
    rng = random.Random(5500 + seed)
    ds = diffs(*[rng.choice([POS, NEG, UND]) for _ in GRID])
    r = crossover_brackets(ds, floor=0)
    # Independent scan: walk the grid carrying the last determined sign.
    last, found = None, []
    for i, d in enumerate(ds):
        s = 1 if d.lo > 0 else -1 if d.hi < 0 else 0
        if s and last and s != last[1]:
            found.append((last[0], i))
        if s:
            last = (i, s)
    assert [(b.lower_index, b.upper_index) for b in r.brackets] == found


def test_bracket_inputs_are_checked():
    with pytest.raises(CostModelError, match="at least two"):
        crossover_brackets(diffs(POS, grid=(2048,)), floor=0)
    with pytest.raises(CostModelError, match="must increase"):
        crossover_brackets(diffs(POS, NEG, grid=(4096, 2048)), floor=0)
    with pytest.raises(CostModelError, match="outside its interval"):
        crossover_brackets(diffs((5.0, 1.0, 3.0), POS), floor=0)
    with pytest.raises(CostModelError, match="resolution floor"):
        crossover_brackets(diffs(POS, NEG), floor=-1)


def test_differences_from_intervals():
    ivs = [Interval(point=p, lo=lo, hi=hi, level=0.9, n_boot=10, seed=0)
           for p, lo, hi in (POS, NEG)]
    assert differences_from_intervals((2048, 4096), ivs) == diffs(POS, NEG)
    with pytest.raises(CostModelError, match="one interval per"):
        differences_from_intervals((2048,), ivs)


# --------------------------------------------------------------------------
# The quadratic
# --------------------------------------------------------------------------

def test_an_exact_quadratic_is_recovered_with_zero_residual():
    a, b, c = 1e-9, 1e-5, 0.01
    t = [a * L * L + b * L + c for L in GRID]
    q = fit_quadratic(GRID, t)
    assert q.a == pytest.approx(a, rel=1e-9) and q.b == pytest.approx(b, rel=1e-9)
    assert q.c == pytest.approx(c, rel=1e-9)
    assert q.rms_residual < 1e-12 and q.max_abs_relative_residual < 1e-10
    assert q(10000) == pytest.approx(a * 1e8 + b * 1e4 + c)


def test_the_fit_agrees_with_polyfit_and_reports_its_residual():
    t = [0.05, 0.09, 0.30, 0.55, 2.20]     # not a quadratic
    q = fit_quadratic(GRID, t)
    A, B, Cc = np.polyfit(np.array(GRID, float), np.array(t), 2)
    assert (q.a, q.b, q.c) == pytest.approx((A, B, Cc), rel=1e-6)
    resid = np.array(t) - np.polyval([A, B, Cc], np.array(GRID, float))
    assert q.residuals == pytest.approx(tuple(resid), abs=1e-9)
    assert q.rms_residual == pytest.approx(math.sqrt(np.mean(resid ** 2)))
    assert q.max_abs_relative_residual == pytest.approx(
        np.max(np.abs(resid) / np.array(t)))


def test_quadratic_crossovers_are_the_roots_inside_the_grid():
    dense = [2e-5 * L + 0.05 for L in GRID]
    k = 1e-10
    sparse = [d + k * (L - 6000) * (L - 24000) for d, L in zip(dense, GRID)]
    qs, qd = fit_quadratic(GRID, sparse), fit_quadratic(GRID, dense)
    assert quadratic_crossovers(qs, qd) == pytest.approx((6000, 24000), rel=1e-6)
    # A root below the grid is left out: (L - 1000)(L - 24000).
    sparse = [d + k * (L - 1000) * (L - 24000) for d, L in zip(dense, GRID)]
    assert quadratic_crossovers(fit_quadratic(GRID, sparse), qd) \
        == pytest.approx((24000,), rel=1e-6)
    # No real root: sparse always slower.
    sparse = [d + 0.01 + k * L * L for d, L in zip(dense, GRID)]
    assert quadratic_crossovers(fit_quadratic(GRID, sparse), qd) == ()


def test_parallel_fits_have_no_crossover_to_report():
    dense = [2e-5 * L + 0.05 for L in GRID]
    sparse = [d + 0.01 for d in dense]
    assert quadratic_crossovers(fit_quadratic(GRID, sparse),
                                fit_quadratic(GRID, dense)) == ()


def test_a_linear_difference_has_one_root():
    dense = [0.1 for _ in GRID]
    sparse = [0.1 + 1e-6 * (L - 10000) + 0.2 for L in GRID]  # root at -190000
    assert quadratic_crossovers(fit_quadratic(GRID, sparse),
                                fit_quadratic(GRID, dense)) == ()


def test_quadratic_inputs_are_checked():
    with pytest.raises(CostModelError, match="three grid lengths"):
        fit_quadratic((2048, 4096), (1.0, 2.0))
    with pytest.raises(CostModelError, match="positive, finite"):
        fit_quadratic(GRID, (1, 2, 3, 4))
    with pytest.raises(CostModelError, match="positive, finite"):
        fit_quadratic(GRID, (1, 2, 3, 4, -5))
    with pytest.raises(CostModelError, match="different grids"):
        quadratic_crossovers(fit_quadratic(GRID, (1, 2, 3, 4, 5)),
                             fit_quadratic(GRID[:4], (1, 2, 3, 4)))


# --------------------------------------------------------------------------
# Crossover error
# --------------------------------------------------------------------------

def res(*signs):
    m = {1: POS, -1: NEG, 0: UND}
    return crossover_brackets(diffs(*[m[s] for s in signs]), floor=0)


def test_a_measured_bracket_inside_the_predicted_interval():
    # Predicted +,0,0,-,- : bracket indices 0..3. Measured +,+,-,-,- : 1..2.
    e = crossover_error(res(1, 0, 0, -1, -1), res(1, 1, -1, -1, -1))
    assert e.status == "compared"
    (c,) = e.comparisons
    assert c.contains and c.overlaps and c.direction_agrees
    assert c.shift_grid_steps == ((1 + 2) - (0 + 3)) / 2 == 0


def test_a_measured_bracket_outside_the_predicted_one():
    # Predicted 0..1, measured 3..4: no overlap, shifted 3 steps.
    e = crossover_error(res(1, -1, -1, -1, -1), res(1, 1, 1, 1, -1))
    (c,) = e.comparisons
    assert not c.contains and not c.overlaps and c.shift_grid_steps == 3


def test_brackets_sharing_only_an_endpoint_do_not_overlap():
    (c,) = crossover_error(res(1, -1, -1, -1, -1), res(1, 0, -1, -1, -1)).comparisons
    # Predicted 0..1, measured 0..2: the predicted is inside the measured.
    assert c.overlaps and not c.contains
    (c,) = crossover_error(res(1, 1, -1, -1, -1), res(1, 1, 1, -1, -1)).comparisons
    assert not c.overlaps     # 1..2 and 2..3


def test_crossover_error_statuses():
    none_ = res(1, 1, 1, 1, 1)
    one = res(1, 1, -1, -1, -1)
    two = res(-1, 1, 1, -1, -1)
    assert crossover_error(none_, none_).status == "both none in range"
    assert crossover_error(none_, one).status == "missed"
    assert crossover_error(one, none_).status == "spurious"
    assert crossover_error(one, two).status == "count differs"
    assert crossover_error(res(1, 1, 1, 1, 0), one).status == "unresolved"
    assert crossover_error(one, res(0, 0, 0, 0, 0)).status == "unresolved"
    (c,) = crossover_error(one, res(-1, -1, 1, 1, 1)).comparisons
    assert not c.direction_agrees
    with pytest.raises(CostModelError, match="different grids"):
        crossover_error(one, crossover_brackets(diffs(POS, NEG, grid=(1, 2)),
                                                floor=0))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_absolute_relative_error():
    assert absolute_relative_error(12, 10) == pytest.approx(0.2)
    assert absolute_relative_error(8, 10) == pytest.approx(0.2)
    with pytest.raises(CostModelError):
        absolute_relative_error(1, 0)


@pytest.mark.parametrize("pred, meas, admissible, want", [
    # (sparse, dense) predicted, measured; floor 0.1 throughout.
    ((1.0, 2.0), (1.0, 2.0), True, CORRECT),            # both profitable
    ((1.0, 2.0), (2.0, 1.0), True, FALSE_PROFITABLE),
    ((2.0, 1.0), (1.0, 2.0), True, MISSED_PROFITABLE),
    ((2.0, 1.0), (2.0, 1.0), True, CORRECT),
    ((1.0, 2.0), (1.95, 2.0), True, NOT_INTERPRETED),   # 0.05 inside 0.1
    ((1.0, 2.0), (2.0, 1.95), True, NOT_INTERPRETED),
    ((1.0, 2.0), (1.0, 2.0), False, CORRECT),           # inadmissible: never
    ((1.0, 2.0), (1.95, 2.0), False, CORRECT),          # floor moot
])
def test_decision_error(pred, meas, admissible, want):
    assert decision_error(*pred, *meas, admissible=admissible, floor=0.1) == want


def test_profitability_rules():
    assert predicted_profitable(1, 2, True) and not predicted_profitable(1, 2, False)
    assert not predicted_profitable(2, 2, True)
    # The floor is the noise in what was measured; it is not applied to a
    # prediction, so a predicted gain of 0.05 s still counts.
    assert predicted_profitable(1.95, 2.0, True)
    assert decision_error(1.95, 2.0, 1.0, 2.0, admissible=True, floor=0.1) == CORRECT
    assert measured_profitable(1, 2, True, floor=1) is None     # exactly the floor
    assert measured_profitable(1, 2, True, floor=0.99) is True
    assert measured_profitable(3, 2, True, floor=0.99) is False
    assert measured_profitable(1, 2, False, floor=0.5) is False
    with pytest.raises(CostModelError, match="admissible must be a bool"):
        decision_error(1, 2, 1, 2, admissible=1, floor=0)


def test_sign_agreement():
    assert sign_agreement(1, 2, 1, 2, floor=0.5) is True
    assert sign_agreement(1, 2, 3, 2, floor=0.5) is False
    assert sign_agreement(1, 2, 2.5, 2, floor=0.5) is None   # |0.5| <= 0.5
    assert sign_agreement(3, 2, 3, 2, floor=0) is True
    with pytest.raises(CostModelError, match="resolution floor"):
        sign_agreement(1, 2, 1, 2, floor=float("nan"))
