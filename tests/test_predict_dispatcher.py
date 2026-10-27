"""T2.22: the dispatcher, its comparators, regret and decision errors, and the
margin frontier, on a hand-worked grid.

The grid (seconds; model "m", batch 1; floor 5). Dense is `dense`; the sparse
configurations are a@0.5 and b@0.25.

    L = 16384        pred   meas   kernel
      dense          100    100     40
      a@0.5           90     80     20     (kernel speedup 2)
      b@0.25         105    110     10     (kernel speedup 4)
    L = 32768
      dense          400    400    200
      a@0.5          395    410    160     (1.25)
      b@0.25         420    430    220     (0.909)
    L = 8192 (not an H6 length)
      dense           10     10     10
      a@0.5            8      9      8     (1.25)
      b@0.25          11     12      5     (2)

Dispatcher, both admissible: 16384: best predicted a (90), 100 - 90 = 10 > 5:
a. 32768: best predicted a (395), 400 - 395 = 5 is not more than 5: dense.
Kernel-speedup rule: 16384: both beat dense, b by more (4 > 2): b. 32768: a
only (1.25 > 1): a. Measured best admissible: a (80) and dense (400).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from attnbench.analysis.exact_noninferiority import paired_exact_ni
from attnbench.predict import dispatcher as D
from attnbench.predict.dispatcher import (
    Config, DispatchError, Point, admissible_at_margin, best_kernel_config,
    best_measured_choice, build_points, choices_of, dispatcher, evaluate,
    h6_line, kernel_speedup_rule, margin_frontier, score_policy)
from attnbench.predict.scoring import (CORRECT, FALSE_PROFITABLE,
                                       MISSED_PROFITABLE)
from attnbench.predict.training import CellKey

M = "m"
A, B = Config("a", 0.5), Config("b", 0.25)
DENSE = Config("dense", 1.0)

PRED = {  # point -> config -> seconds
    16384: {DENSE: 100, A: 90, B: 105},
    32768: {DENSE: 400, A: 395, B: 420},
    8192: {DENSE: 10, A: 8, B: 11},
}
MEAS = {
    16384: {DENSE: 100, A: 80, B: 110},
    32768: {DENSE: 400, A: 410, B: 430},
    8192: {DENSE: 10, A: 9, B: 12},
}
KERN = {
    16384: {DENSE: 40, A: 20, B: 10},
    32768: {DENSE: 200, A: 160, B: 220},
    8192: {DENSE: 10, A: 8, B: 5},
}
FLOOR = 5.0


def key(L, c, code_path="cp"):
    return CellKey(M, code_path, c.arm, c.density, L, 1)


def predicted():
    return {key(L, c): v for L, row in PRED.items() for c, v in row.items()}


def kernel():
    return {key(L, c): v for L, row in KERN.items() for c, v in row.items()}


def measured():
    return {Point(M, L, 1): dict(row) for L, row in MEAS.items()}


def points(**kw):
    return build_points(predicted(), dense_arms={"dense"}, kernel=kernel(), **kw)


def P(L):
    return Point(M, L, 1)


def by_point(pts, L):
    (p,) = [p for p in pts if p.point == P(L)]
    return p


BOTH = frozenset({A, B})


# --------------------------------------------------------------------------
# Building the points
# --------------------------------------------------------------------------

def test_points_are_model_length_and_batch_with_one_dense_each():
    pts = points()
    assert [p.point for p in pts] == [P(8192), P(16384), P(32768)]
    p = by_point(pts, 16384)
    assert p.dense.config == DENSE and p.dense.predicted == 100
    assert [c.config for c in p.sparse] == [B, A] or [c.config for c in p.sparse] == [A, B]
    assert p.candidate(A).kernel == 20 and p.candidate(Config("z", 1.0)) is None


def test_a_point_needs_exactly_one_dense_cell():
    pred = {k: v for k, v in predicted().items() if not (k.arm == "dense" and k.context_length == 8192)}
    with pytest.raises(DispatchError, match="0 dense cells"):
        build_points(pred, dense_arms={"dense"})
    pred = predicted()
    pred[key(8192, DENSE, "cp2")] = 12
    with pytest.raises(DispatchError, match="2 dense cells"):
        build_points(pred, dense_arms={"dense"})
    with pytest.raises(DispatchError, match="name the dense"):
        build_points(predicted(), dense_arms=set())


def test_one_configuration_on_two_code_paths_must_be_chosen_between():
    pred = predicted()
    pred[key(16384, A, "cp-triton")] = 70
    with pytest.raises(DispatchError, match="use_code_path"):
        build_points(pred, dense_arms={"dense"})
    pts = build_points(pred, dense_arms={"dense"}, use_code_path={"a": "cp-triton"})
    assert by_point(pts, 16384).candidate(A).predicted == 70
    pts = build_points(pred, dense_arms={"dense"}, use_code_path={"a": "cp"})
    assert by_point(pts, 16384).candidate(A).predicted == 90


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, None])
def test_predictions_must_be_positive_finite_times(bad):
    pred = predicted()
    pred[key(16384, A)] = bad
    with pytest.raises(DispatchError, match="positive finite"):
        build_points(pred, dense_arms={"dense"})


# --------------------------------------------------------------------------
# The dispatcher
# --------------------------------------------------------------------------

def test_the_dispatcher_picks_the_best_predicted_admissible_configuration():
    pts = points()
    assert dispatcher(by_point(pts, 16384), BOTH, FLOOR) == A
    # b predicted 105 is slower than dense, so it never wins on prediction.
    assert dispatcher(by_point(pts, 16384), frozenset({B}), FLOOR) == DENSE


def test_faster_by_exactly_the_floor_is_dense_and_by_more_is_sparse():
    pts = points()
    p = by_point(pts, 32768)                     # 400 - 395 = 5
    assert dispatcher(p, BOTH, 5.0) == DENSE
    assert dispatcher(p, BOTH, 4.999) == A
    assert dispatcher(p, BOTH, 0.0) == A


def test_a_predicted_tie_with_dense_is_dense():
    pred = predicted()
    pred[key(16384, A)] = 100
    pred[key(16384, B)] = 100
    p = by_point(build_points(pred, dense_arms={"dense"}), 16384)
    assert dispatcher(p, BOTH, 0.0) == DENSE


def test_equal_predicted_sparse_times_go_to_the_smaller_configuration():
    pred = predicted()
    pred[key(16384, A)] = 80
    pred[key(16384, B)] = 80
    p = by_point(build_points(pred, dense_arms={"dense"}), 16384)
    assert dispatcher(p, BOTH, FLOOR) == A            # ("a", 0.5) < ("b", 0.25)


def test_nothing_admissible_is_dense_and_the_floor_is_checked():
    p = by_point(points(), 16384)
    assert dispatcher(p, frozenset(), FLOOR) == DENSE
    for bad in (-1, float("nan"), True, None):
        with pytest.raises(DispatchError, match="resolution floor"):
            dispatcher(p, BOTH, bad)


def test_a_configuration_outside_the_admissible_set_is_never_chosen():
    p = by_point(points(), 16384)
    assert dispatcher(p, frozenset({A}), FLOOR) == A
    assert dispatcher(p, frozenset({Config("a", 0.75)}), FLOOR) == DENSE


# --------------------------------------------------------------------------
# The comparators
# --------------------------------------------------------------------------

def test_the_kernel_speedup_rule_takes_the_best_kernel_that_beats_dense():
    pts = points()
    assert kernel_speedup_rule(by_point(pts, 16384), BOTH) == B      # 4 > 2
    assert kernel_speedup_rule(by_point(pts, 32768), BOTH) == A      # b 0.909 < 1
    assert kernel_speedup_rule(by_point(pts, 32768), frozenset({B})) == DENSE
    assert kernel_speedup_rule(by_point(pts, 16384), frozenset()) == DENSE


def test_the_kernel_rule_ignores_the_predicted_times():
    pred = {k: (1000 if k.arm == "b" else v) for k, v in predicted().items()}
    p = by_point(build_points(pred, dense_arms={"dense"}, kernel=kernel()), 16384)
    assert kernel_speedup_rule(p, BOTH) == B


def test_a_missing_kernel_time_is_refused_not_skipped():
    kern = kernel()
    del kern[key(16384, B)]
    p = by_point(build_points(predicted(), dense_arms={"dense"}, kernel=kern), 16384)
    with pytest.raises(DispatchError, match="no kernel time"):
        kernel_speedup_rule(p, BOTH)
    # With b not admissible the rule never needs it.
    assert kernel_speedup_rule(p, frozenset({A})) == A


def test_the_best_kernel_configuration_is_one_fixed_choice_by_geometric_mean():
    # a: (2 * 1.25 * 1.25) ** (1/3) = 1.462; b: (4 * 0.909 * 2) ** (1/3) = 1.937.
    pts = points()
    assert best_kernel_config(pts, BOTH) == B
    assert best_kernel_config(pts, frozenset({A})) == A
    assert best_kernel_config(pts, frozenset()) is None
    ch = choices_of("best_kernel_config", pts, BOTH, FLOOR)
    assert set(ch.values()) == {B}
    assert set(choices_of("best_kernel_config", pts, frozenset(), FLOOR).values()) == {None}


def test_a_kernel_that_only_ties_dense_does_not_beat_it():
    kern = kernel()
    kern[key(16384, A)] = 40                     # equal to dense: speedup 1
    kern[key(16384, B)] = 40
    p = by_point(build_points(predicted(), dense_arms={"dense"}, kernel=kern), 16384)
    assert kernel_speedup_rule(p, BOTH) == DENSE


def test_the_fixed_configuration_is_chosen_by_geometric_not_arithmetic_mean():
    # a: speedups 4 and 0.5 (geometric mean 1.414, arithmetic 2.25);
    # b: 1.5 and 1.5 (both 1.5). The geometric mean prefers b.
    kern = {key(16384, DENSE): 60, key(16384, A): 15, key(16384, B): 40,
            key(32768, DENSE): 60, key(32768, A): 120, key(32768, B): 40}
    pred = {k: v for k, v in predicted().items() if k.context_length != 8192}
    pts = build_points(pred, dense_arms={"dense"}, kernel=kern)
    assert best_kernel_config(pts, BOTH) == B


def test_the_fixed_configuration_is_unscored_where_the_point_lacks_it():
    pred = {k: v for k, v in predicted().items()
            if not (k.arm == "b" and k.context_length == 32768)}
    pts = build_points(pred, dense_arms={"dense"}, kernel=kernel())
    ch = choices_of("best_kernel_config", pts, BOTH, FLOOR)
    assert ch[P(32768)] is None and ch[P(16384)] == B
    s = score_policy(pts, ch, measured(), BOTH, FLOOR)
    assert s.n_scored == 1
    assert [u.point for u in s.unscored] == [P(32768)]
    assert "no choice" in s.unscored[0].unscored


def test_always_dense_and_the_policy_names():
    ch = choices_of("always_dense", points(), BOTH, FLOOR)
    assert set(ch.values()) == {DENSE}
    with pytest.raises(DispatchError, match="policy must be one of"):
        choices_of("oracle", points(), BOTH, FLOOR)
    with pytest.raises(DispatchError, match="nearest pairing's best"):
        choices_of("nearest_pairing", points(), BOTH, FLOOR)


def test_the_nearest_pairings_choice_is_applied_where_it_exists():
    near = {P(16384): A, P(32768): DENSE, P(8192): Config("z", 0.1)}
    ch = choices_of("nearest_pairing", points(), BOTH, FLOOR, nearest=near)
    assert ch == {P(8192): None, P(16384): A, P(32768): DENSE}


# --------------------------------------------------------------------------
# Regret, decision errors and time gained
# --------------------------------------------------------------------------

def score(name, adm=BOTH, **kw):
    pts = points()
    ch = choices_of(name, pts, adm, FLOOR, nearest=kw.pop("nearest", None))
    return score_policy(pts, ch, measured(), adm, FLOOR, **kw)


def test_the_dispatcher_scored_on_the_h6_lengths():
    s = score("dispatcher")
    # 16384: a (80) is the best measured: regret 0, correct. 32768: dense
    # (400) is the best: regret 0; the best admissible sparse is 410, slower:
    # correct. Gain 20 + 0; speedup 500 / 480.
    assert s.n_scored == 2 and s.mean_regret == 0 and s.worst_regret == 0
    assert s.decisions == {CORRECT: 2, FALSE_PROFITABLE: 0, MISSED_PROFITABLE: 0}
    assert s.gain_s == 20 and s.speedup_vs_dense == pytest.approx(500 / 480)
    assert s.decision_errors == 0 and not s.dense_best_everywhere


def test_the_kernel_speedup_rule_loses_on_both_points():
    s = score("kernel_speedup_rule")
    # 16384: b 110 over best 80: regret 0.375; 110 - 100 = 10 > 5: false.
    # 32768: a 410 over best 400: regret 0.025; 410 - 400 = 10 > 5: false.
    assert s.mean_regret == pytest.approx(0.2) and s.worst_regret == pytest.approx(0.375)
    assert s.decisions[FALSE_PROFITABLE] == 2
    assert s.gain_s == -20 and s.speedup_vs_dense == pytest.approx(500 / 520)


def test_always_dense_misses_the_profitable_point():
    s = score("always_dense")
    # 16384: regret 100/80 - 1 = 0.25, and a was faster by 20 > 5: missed.
    # 32768: regret 0, best sparse slower: correct.
    assert s.mean_regret == pytest.approx(0.125) and s.worst_regret == pytest.approx(0.25)
    assert s.decisions == {CORRECT: 1, FALSE_PROFITABLE: 0, MISSED_PROFITABLE: 1}
    assert s.gain_s == 0 and s.speedup_vs_dense == 1


def test_the_fixed_kernel_configuration_loses_everywhere():
    s = score("best_kernel_config")
    # b at 16384: 110 / 80 - 1 = 0.375; at 32768: 430 / 400 - 1 = 0.075.
    assert s.mean_regret == pytest.approx(0.225) and s.worst_regret == pytest.approx(0.375)
    assert s.decisions[FALSE_PROFITABLE] == 2


def test_the_nearest_pairings_choice_is_scored_like_any_other():
    s = score("nearest_pairing", nearest={P(16384): A, P(32768): DENSE})
    assert s.mean_regret == 0 and s.decisions[CORRECT] == 2


def test_every_point_is_scored_when_lengths_is_none():
    s = score("dispatcher", lengths=None)
    # 8192: best predicted a (8), 10 - 8 = 2 is not more than 5: dense. The
    # best measured admissible is a (9) vs dense 10: regret 10/9 - 1; dense
    # chosen, a faster by 1 <= 5: inside the floor, not interpreted.
    assert s.n_scored == 3 and s.inside_floor == 1
    assert s.mean_regret == pytest.approx((0 + 0 + (10 / 9 - 1)) / 3)
    assert sum(s.decisions.values()) == 2


def test_a_sparse_choice_inside_the_floor_is_not_interpreted_but_has_regret():
    meas = measured()
    meas[P(16384)][A] = 97                       # 3 below dense: inside the floor
    pts = points()
    ch = choices_of("dispatcher", pts, BOTH, FLOOR)
    s = score_policy(pts, ch, meas, BOTH, FLOOR)
    assert s.inside_floor == 1 and sum(s.decisions.values()) == 1
    (ps,) = [x for x in s.points if x.point == P(16384)]
    assert ps.outcome is None and ps.regret == pytest.approx(0.0)   # a is still the best


def test_a_dense_choice_where_sparse_is_faster_by_more_than_the_floor_is_missed():
    meas = measured()
    meas[P(32768)][A] = 380                      # 20 faster than dense 400
    pts = points()
    ch = {p.point: DENSE for p in pts}
    s = score_policy(pts, ch, meas, BOTH, FLOOR)
    assert s.decisions[MISSED_PROFITABLE] == 2   # 16384 (80) and 32768 (380)


def test_unmeasured_points_and_configurations_are_listed_not_scored():
    meas = measured()
    del meas[P(32768)][DENSE]
    del meas[P(16384)][A]
    pts = points()
    ch = choices_of("dispatcher", pts, BOTH, FLOOR)
    s = score_policy(pts, ch, meas, BOTH, FLOOR)
    assert s.n_scored == 0 and s.mean_regret is None and s.speedup_vs_dense is None
    reasons = {u.point: u.unscored for u in s.unscored}
    assert "no measured dense" in reasons[P(32768)]
    assert "was not measured" in reasons[P(16384)]


def test_when_dense_is_best_everywhere_the_dispatcher_can_only_tie_always_dense():
    meas = {p: {c: (t if c == DENSE else t * 2) for c, t in row.items()}
            for p, row in measured().items()}
    pts = points()
    for name in ("dispatcher", "always_dense"):
        s = score_policy(pts, choices_of(name, pts, BOTH, FLOOR), meas, BOTH, FLOOR)
        assert s.dense_best_everywhere
    s = score_policy(pts, choices_of("always_dense", pts, BOTH, FLOOR), meas, BOTH, FLOOR)
    assert s.mean_regret == 0
    # The kernel rule's losses are what the dispatcher can still avoid.
    k = score_policy(pts, choices_of("kernel_speedup_rule", pts, BOTH, FLOOR),
                     meas, BOTH, FLOOR)
    assert k.mean_regret > 0 and k.gain_s < 0
    # A property of the grid and the set, so the same for every policy.
    assert k.dense_best_everywhere


def test_best_measured_choice_prefers_dense_on_a_tie_then_the_smaller_config():
    p = by_point(points(), 16384)
    assert best_measured_choice(p, {DENSE: 5, A: 5, B: 5}, BOTH) == DENSE
    assert best_measured_choice(p, {DENSE: 9, A: 5, B: 5}, BOTH) == A
    assert best_measured_choice(p, {DENSE: 9, A: 5, B: 1}, frozenset({A})) == A
    with pytest.raises(DispatchError, match="no measured dense"):
        best_measured_choice(p, {A: 5}, BOTH)


def test_removing_an_arm_removes_it_from_the_choice_and_from_the_reference():
    # Per-arm rule for a failed mask gate (Section 5.1): with a not admissible
    # the best measured admissible configuration at 16384 is dense (100), not
    # a (80), so a dense choice has regret 0 instead of 0.25.
    pts = points()
    only_b = frozenset({B})
    ch = choices_of("dispatcher", pts, only_b, FLOOR)
    assert ch[P(16384)] == DENSE
    s = score_policy(pts, ch, measured(), only_b, FLOOR)
    assert s.mean_regret == 0
    with_a = score_policy(pts, ch, measured(), BOTH, FLOOR)
    assert with_a.worst_regret == pytest.approx(0.25)


# --------------------------------------------------------------------------
# With and without the Holm-corrected set; H6
# --------------------------------------------------------------------------

def test_every_admissible_set_is_evaluated_and_labelled():
    out = evaluate(points(), measured(), {"uncorrected": BOTH, "holm": [("a", 0.5)]},
                   FLOOR)
    assert set(out) == {"uncorrected", "holm"}
    assert set(out["holm"]) == set(D.POLICIES) - {"nearest_pairing"}
    # With only a admissible the kernel rule picks a at 16384 (regret 0) and
    # at 32768 (regret 0.025).
    assert out["holm"]["kernel_speedup_rule"].mean_regret == pytest.approx(0.0125)
    assert out["uncorrected"]["kernel_speedup_rule"].mean_regret == pytest.approx(0.2)
    out = evaluate(points(), measured(), {"uncorrected": BOTH}, FLOOR,
                   nearest={P(16384): A, P(32768): DENSE})
    assert "nearest_pairing" in out["uncorrected"]
    with pytest.raises(DispatchError, match="at least one admissible set"):
        evaluate(points(), measured(), {}, FLOOR)


def test_h6_line_compares_with_the_primary_comparator():
    out = evaluate(points(), measured(), {"uncorrected": BOTH}, FLOOR)["uncorrected"]
    line = h6_line(out["dispatcher"], out["kernel_speedup_rule"])
    assert line.lower_mean_regret and line.lower_worst_regret
    assert line.no_more_decision_errors and not line.dense_best_everywhere
    assert (line.dispatcher_errors, line.rule_errors) == (0, 2)
    # The reverse comparison fails on every count.
    rev = h6_line(out["kernel_speedup_rule"], out["dispatcher"])
    assert not rev.lower_mean_regret and not rev.no_more_decision_errors
    # An unscored side gives no verdict on regret.
    none = h6_line(score_policy(points(), {}, measured(), BOTH, FLOOR),
                   out["kernel_speedup_rule"])
    assert none.lower_mean_regret is None and none.lower_worst_regret is None


# --------------------------------------------------------------------------
# Margin against speedup
# --------------------------------------------------------------------------

def test_a_margin_admits_what_the_exact_test_would_call_non_inferior():
    # lower > -margin, strictly, as paired_exact_ni.
    lower = {("a", 0.5): -3.0, ("b", 0.25): -8.0, ("c", 0.1): -5.0}
    assert admissible_at_margin(lower, 2) == frozenset()
    assert admissible_at_margin(lower, 5) == frozenset({A})        # -5 is not > -5
    assert admissible_at_margin(lower, 10) == frozenset({A, B, Config("c", 0.1)})
    for bad in (0, -1, float("nan"), True):
        with pytest.raises(DispatchError, match="positive number of points"):
            admissible_at_margin(lower, bad)


@pytest.mark.parametrize("b,c,n", [(0, 0, 200), (3, 9, 200), (10, 12, 100),
                                   (0, 40, 400), (7, 7, 50)])
@pytest.mark.parametrize("margin", [2.0, 5.0, 10.0])
def test_the_margin_rule_is_the_one_the_accuracy_test_applies(b, c, n, margin):
    dense = np.zeros(n, dtype=bool)
    sparse = np.zeros(n, dtype=bool)
    dense[:c + 20] = True                      # c dense-only pairs, 20 both-right
    sparse[c:c + 20 + b + c] = False
    sparse[:] = False
    sparse[c:c + 20] = True                    # both-right: indices c..c+19
    sparse[c + 20:c + 20 + b] = True           # b sparse-only
    r = paired_exact_ni(dense, sparse, margin_pts=margin)
    assert (r.b_sparse_only, r.c_dense_only) == (b, c)
    assert (A in admissible_at_margin({tuple(A): r.lower_pts}, margin)) == r.non_inferior


def test_the_frontier_sweeps_the_margin_and_reports_what_each_set_buys():
    lower = {tuple(A): -3.0, tuple(B): -8.0}
    rows = margin_frontier(points(), measured(), lower, [10, 2, 5], FLOOR)
    got = {(r.margin_pts, r.policy): r for r in rows}
    assert [r.margin_pts for r in rows] == [2, 2, 5, 5, 10, 10]    # sorted, once each
    # Margin 2: nothing admissible, both policies are dense.
    for pol in ("dispatcher", "kernel_speedup_rule"):
        r = got[(2, pol)]
        assert r.n_admissible == 0 and r.speedup_vs_dense == 1 and r.gain_s == 0
    # Margin 5: a admissible. Dispatcher a at 16384, dense at 32768: 500/480;
    # the rule picks a at both: 16384 (80), 32768 (410): 500/490.
    assert got[(5, "dispatcher")].n_admissible == 1
    assert got[(5, "dispatcher")].speedup_vs_dense == pytest.approx(500 / 480)
    assert got[(5, "kernel_speedup_rule")].speedup_vs_dense == pytest.approx(500 / 490)
    # Margin 10: b joins; the rule takes b at 16384 (110) and a at 32768 (410).
    assert got[(10, "dispatcher")].speedup_vs_dense == pytest.approx(500 / 480)
    assert got[(10, "kernel_speedup_rule")].speedup_vs_dense == pytest.approx(500 / 520)
    assert got[(10, "kernel_speedup_rule")].decision_errors == 2
    with pytest.raises(DispatchError, match="not swept"):
        margin_frontier(points(), measured(), lower, [5], FLOOR,
                        policies=("nearest_pairing",))


# --------------------------------------------------------------------------
# What the committed file carries
# --------------------------------------------------------------------------

def spec(**kw):
    args = dict(predictor="recursion", floor_s=FLOOR, dense_arms=("dense",),
                admissible={"uncorrected": (A, B), "holm": (A,)},
                kernel_times=kernel(), use_code_path={})
    args.update(kw)
    return D.DispatchSpec(**args)


def test_the_file_records_every_policys_choice_per_set_and_point():
    rows = D.make_choices(spec(), predicted())
    assert len(rows) == 2 * 4 * 3                      # sets x policies x points
    get = {(r["set"], r["policy"], r["context_length"]): r["choice"] for r in rows}
    assert get[("uncorrected", "dispatcher", 16384)] == {
        "arm": "a", "density": 0.5, "code_path": "cp"}
    assert get[("uncorrected", "dispatcher", 32768)]["arm"] == "dense"
    assert get[("uncorrected", "kernel_speedup_rule", 16384)]["arm"] == "b"
    assert get[("holm", "kernel_speedup_rule", 16384)]["arm"] == "a"
    assert get[("uncorrected", "best_kernel_config", 8192)]["arm"] == "b"
    assert all(r["policy"] != "nearest_pairing" for r in rows)


def test_the_nearest_pairings_choice_is_echoed_into_the_file():
    near = {P(16384): A, P(32768): DENSE}
    rows = D.make_choices(spec(nearest=near), predicted())
    got = {(r["set"], r["context_length"]): r["choice"] for r in rows
           if r["policy"] == "nearest_pairing"}
    assert got[("uncorrected", 16384)]["arm"] == "a"
    assert got[("uncorrected", 8192)] is None


def test_the_spec_round_trips_through_json():
    s = spec(nearest={P(16384): A}, use_code_path={"a": "cp"})
    back = D.spec_from_json(s.to_json())
    assert back.to_json() == s.to_json()
    assert back.kernel_times == s.kernel_times and back.nearest == s.nearest
    with pytest.raises(DispatchError, match="unreadable"):
        D.spec_from_json({"predictor": "recursion"})
    with pytest.raises(DispatchError, match="at least one admissible set"):
        D.make_choices(spec(admissible={}), predicted())
