"""The confirmatory analysis, H0 to H5 and the marginal label (T4.3 wave 2a,
adviser's ruling 2a of 11 Nov 2026; `attnbench/analysis/hypotheses.py`).

H1 is hand-worked on `test_predict_commit`'s P2 record, 20 s (xattn) and
10 s (dense) measured: recursion 16 and 8 (-0.20, -0.20), so its median
absolute relative error is 0.20; the predicted and measured signs of
xattn - dense are both positive.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from types import SimpleNamespace

import pytest

from attnbench.analysis import hypotheses as HY
from attnbench.lock.floors import FloorTable
from attnbench.lock.holm import Counts
from attnbench.predict import crossover as X
from attnbench.predict import intervals as IV
from attnbench.predict.commit import load_committed
from attnbench.predict.recursion import recurse
from attnbench.predict.training import CellKey

import lock_fixture as LF
from test_predict_commit import (CP, MODEL, NO_SKEW, T0, Clock, MemoryStore, committed,
                                 locked, p2_record, trained)  # noqa: F401

FLOORS = FloorTable({(4, 1): 0.0, (16384, 1): 0.01377, (16384, 4): 0.02759, (32768, 1): 0.10116})


def lk(**over):
    base = dict(tolerances={"H0": 0.1, "H1": 0.25, "H2": 0.25}, floors=FLOORS,
                bootstrap=IV.Settings(0.95, 200, 0), h4_bound_pts=10.0,
                h4_tests=("t1", "t2"), h4_alpha=0.05, h4_n=100,
                scored=LF.SCORED, sensitivity_s={(16384, 1): 0.11105, (16384, 4): 0.44154},
                analysis_commit="c" * 40)
    base.update(over)
    return HY.Locked(**base)


@pytest.fixture
def p2(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    f, receipt = committed(tmp_path, p2_record(), store, T0 + timedelta(seconds=5))
    return load_committed(store, receipt.remote, ground_truth_started_at=T0 + timedelta(seconds=6),
                          clock_skew=NO_SKEW, sync_gap=NO_SKEW,
                          banked_at={"P1": T0 - timedelta(days=1)}, local=f.path, **trained())


def seconds_of(cp, xattn=20.0, dense=10.0):
    return {c.key: (xattn if c.arm == "xattn" else dense) for c in cp.cells}


# --- the lock ----------------------------------------------------------------------

def test_the_analysis_reads_every_threshold_from_the_lock():
    from attnbench.lock import checklist as CL
    lock = LF.complete_lock(CL.PLAN.read_text(encoding="utf-8"))
    lock["scored_pairings"] = LF.SCORED
    lock["values"]["h4_confirmatory_family"]["value"] = {"m": 2, "tests": ["a", "b"]}
    got = HY.Locked.from_lock(lock)
    assert got.tolerances == {"H0": lock["values"]["h0_to_h3_tolerances"]["value"]["H0"],
                              "H1": lock["values"]["h0_to_h3_tolerances"]["value"]["H1"],
                              "H2": lock["values"]["h0_to_h3_tolerances"]["value"]["H2"]}
    assert got.sensitivity_s == {(16384, 1): pytest.approx(0.11105),
                                 (16384, 4): pytest.approx(0.44154)}
    assert got.bootstrap == IV.settings_from_lock(lock) and got.h4_tests == ("a", "b")
    lock["values"]["marginal_label_s_sensitivity_floors"]["value"] = None
    with pytest.raises(HY.HypothesisError, match="marginal_label_s_sensitivity_floors"):
        HY.Locked.from_lock(lock)
    lock["values"]["marginal_label_s_sensitivity_floors"]["value"] = {"16384x1": 111.05}
    lock["scored_pairings"] = {**LF.SCORED, "H1": ["P1", "P2"]}
    with pytest.raises(HY.HypothesisError, match="H1 names P1"):
        HY.Locked.from_lock(lock)


# --- the marginal label ------------------------------------------------------------

def test_a_comparison_above_the_floor_and_inside_the_sensitivity_floor_is_marginal():
    L = lk()
    kw = dict(floors=L.floors, sensitivity_s=L.sensitivity_s)
    assert not HY.marginal(0.010, 16370, 1, **kw)           # inside the floor: not interpreted
    assert HY.marginal(0.050, 16370, 1, **kw)               # 13.77 < 50 <= 111.05 ms
    assert HY.marginal(0.11105, 16370, 1, **kw)             # at the sensitivity floor
    assert not HY.marginal(0.112, 16370, 1, **kw)
    assert not HY.marginal(0.2, 32700, 1, **kw)             # 32768 has no sensitivity floor
    # P2's or P4's own within-session spread labels too, at any length
    assert HY.marginal(0.2, 32700, 1, **kw, spread_s={(32768, 1): 0.25})


def test_the_within_session_spread_is_pooled_per_budget_like_p1s_floor():
    k = lambda L, b=1, arm="a": CellKey("m", "cp", arm, 1.0, L, b)
    spread = HY.within_session_spread({k(16370): [1.0, 1.1, 1.3], k(16380, arm="b"): [2.0, 2.0]},
                                      quantile=1.0, budget_of=FLOORS.budget_of)
    assert spread == {(16384, 1): pytest.approx(0.2)}      # |1.3 - 1.1|
    with pytest.raises(HY.HypothesisError, match="two repeats"):
        HY.within_session_spread({k(16370): [1.0]}, quantile=0.95, budget_of=FLOORS.budget_of)


# --- H1 ---------------------------------------------------------------------------------

def test_h1_holds_on_the_hand_worked_record_and_fails_outside_its_tolerance(p2):
    out = HY.h1(p2, seconds_of(p2), locked=lk(), dense_arm="dense")
    assert out["recursion_median_abs_relative"] == pytest.approx(0.20)
    assert out["parts"]["tolerance"] == HY.HOLDS and out["sign"]["agree"] == 1
    assert out["primary_comparison"] == "fitted_sum" and out["rivals_not_in_file"] == []
    assert set(out["rivals_median_abs_relative"]) == set(HY.H1_RIVALS)
    assert out["verdict"] == (HY.HOLDS if all(out["recursion_below"].values()) else HY.FAILS)
    tight = HY.h1(p2, seconds_of(p2), locked=lk(tolerances={"H0": .1, "H1": 0.1, "H2": .1}),
                  dense_arm="dense")
    assert tight["parts"]["tolerance"] == HY.FAILS and tight["verdict"] == HY.FAILS


def test_h1_fails_on_a_wrong_sign_and_reads_p1_only_as_a_labelled_dry_run(p2):
    # xattn measured faster than dense: the recursion's + sign is wrong
    out = HY.h1(p2, seconds_of(p2, xattn=8.0, dense=10.0), locked=lk(), dense_arm="dense")
    assert out["sign"]["verdict"] == HY.FAILS and out["verdict"] == HY.FAILS
    with pytest.raises(HY.HypothesisError, match="development pairing"):
        HY.h1(p2, seconds_of(p2), locked=lk(), dense_arm="dense", development=True)
    p1 = dataclasses.replace(p2, pairing="P1")
    with pytest.raises(HY.HypothesisError, match="development pairing"):
        HY.h1(p1, seconds_of(p1), locked=lk(), dense_arm="dense")


def test_h1_refuses_a_file_without_its_rivals(p2):
    cells = tuple(dataclasses.replace(c, predictions={k: v for k, v in c.predictions.items()
                                                      if k != "learned"}) for c in p2.cells)
    with pytest.raises(HY.HypothesisError, match="lacks \\['learned'\\]"):
        HY.h1(dataclasses.replace(p2, cells=cells), seconds_of(p2), locked=lk(),
              dense_arm="dense")


# --- H0 and H2 (light stand-ins: they read the cells' priced sequences only) -----------

def _file(pairing, cells, q=8):
    return SimpleNamespace(pairing=pairing, cells=cells, queue_depth=q)


def _cell(p2, scale_host=1.0, scale_device=1.0):
    c = p2.cells[0]
    pr = dataclasses.replace(c.priced, host=tuple(h * scale_host for h in c.priced.host),
                             device=tuple(d * scale_device for d in c.priced.device))
    return dataclasses.replace(c, priced=pr)


def test_h0_is_the_median_over_operations_above_the_floor(p2):
    a, b = _file("P1", [_cell(p2)]), _file("P2", [_cell(p2, scale_host=1.05)])
    out = HY.h0(a, b, side="host", floors=FLOORS, tolerance=0.1)
    assert out["verdict"] == HY.HOLDS and out["median_abs_relative_difference"] == pytest.approx(0.05)
    assert out["coverage_share"] == pytest.approx(
        sum(h for h in a.cells[0].priced.host if h > 0) / sum(a.cells[0].priced.host))
    far = HY.h0(a, _file("P2", [_cell(p2, scale_host=1.2)]), side="host", floors=FLOORS,
                tolerance=0.1)
    assert far["verdict"] == HY.FAILS and far["max_abs_relative_difference"] == pytest.approx(0.2)
    high = FloorTable({(4, 1): 1e9})
    assert HY.h0(a, b, side="host", floors=high, tolerance=0.1)["verdict"] == HY.NOT_SCORED
    other = dataclasses.replace(_cell(p2), priced=dataclasses.replace(
        _cell(p2).priced, mechanisms=tuple("x" for _ in _cell(p2).priced.mechanisms)))
    sep = HY.h0(a, _file("P2", [other]), side="host", floors=FLOORS, tolerance=0.1)
    assert sep["cells_with_a_different_sequence"] and sep["verdict"] == HY.NOT_SCORED


def test_h2_reruns_the_recursion_on_the_sources_host_times(p2):
    src = _file("P1", [_cell(p2, scale_host=2.0)])
    pred, left = HY.transferred(p2, src)
    c = p2.cells[0]
    assert left == [list(HY._match_key(p2.cells[1]))]
    want = recurse(dataclasses.replace(c.priced, host=src.cells[0].priced.host),
                   p2.queue_depth).end_to_end
    assert pred == {c.key: want}
    out = HY.h2(p2, src, seconds_of(p2), locked=lk(), dense_arm="dense", h0_verdict=HY.HOLDS)
    assert out["n_cells"] == 1 and out["median_abs_relative"] == pytest.approx(abs(want - 20) / 20)
    assert HY.h2(p2, src, seconds_of(p2), locked=lk(), dense_arm="dense",
                 h0_verdict=HY.FAILS)["verdict"] == HY.NOT_TESTED


# --- H3 and H5 ------------------------------------------------------------------------

GRID = (2048, 4096, 8192, 16384, 32768)
GF = FloorTable({(L, 1): 0.0 for L in GRID})


def diffs(*signs):
    m = {1: (1.0, 0.5, 1.5), -1: (-1.0, -1.5, -0.5), 0: (0.0, -0.5, 0.5)}
    return [X.GridDifference(L, *m[s]) for L, s in zip(GRID, signs)]


def test_h3_holds_when_the_measured_bracket_is_inside_the_predicted_one():
    out = HY.h3({"xa": diffs(1, 0, 0, -1, -1)}, {"xa": diffs(1, 1, -1, -1, -1)},
                floors=GF, pairing="P4")
    assert out["verdict"] == HY.HOLDS
    out = HY.h3({"xa": diffs(1, -1, -1, -1, -1)}, {"xa": diffs(1, 1, 1, 1, -1)},
                floors=GF, pairing="P4")
    assert out["verdict"] == HY.FAILS
    none = HY.h3({"xa": diffs(1, 1, 1, 1, 1)}, {"xa": diffs(1, 1, 1, 1, 1)}, floors=GF,
                 pairing="P4")
    assert none["verdict"] == HY.NOT_SCORED and "reported as such" in none["estimators"]["xa"]["why"]
    with pytest.raises(HY.HypothesisError, match="different estimators"):
        HY.h3({"xa": diffs(1, 1, 1, 1, 1)}, {"mp": diffs(1, 1, 1, 1, 1)}, floors=GF, pairing="P4")


def test_measured_differences_carry_the_locks_bootstrap(monkeypatch):
    import statistics
    k = lambda arm, L: CellKey("m", "cp", arm, 1.0, L, 1)
    s, d = [k("xa", L) for L in GRID[:2]], [k("dense", L) for L in GRID[:2]]
    samples = {s[0]: [2.0, 2.1, 2.2], d[0]: [1.0, 1.1, 1.2], s[1]: [3.0, 3.1], d[1]: [3.0, 3.2]}
    out = HY.measured_differences(samples, s, d, statistic=statistics.median,
                                  settings=IV.Settings(0.95, 100, 0))
    assert out[0].point == pytest.approx(1.0) and out[0].lo <= out[0].point <= out[0].hi
    monkeypatch.setattr(IV, "LOCKED", IV.Settings(0.95, 1000, 0))
    with pytest.raises(Exception, match="below the lock's 1000"):
        HY.measured_differences(samples, s, d, statistic=statistics.median,
                                settings=IV.Settings(0.95, 100, 0))


def test_h5_compares_the_predicted_shift_sign_and_the_gpu_only_estimators_size():
    # first pairing: crossover between 2048 and 8192; second: between 8192 and 32768
    first = {"host": diffs(1, 0, -1, -1, -1), "gpu": diffs(1, 1, 0, -1, -1)}
    second = {"host": diffs(1, 1, 1, 0, -1), "gpu": diffs(1, 1, 1, -1, -1)}
    out = HY.h5({"host": 1, "gpu": 1}, first, second, floors=GF, gpu_only="gpu")
    assert out["a"]["host"]["verdict"] == HY.HOLDS and out["a"]["host"]["shift_grid_steps"] == 2
    assert out["a"]["gpu"]["shift_grid_steps"] == 0.5 and out["b"] == HY.HOLDS
    assert out["c"]["verdict"] == HY.NOT_SCORED
    wrong = HY.h5({"host": -1, "gpu": 1}, first, second, floors=GF, gpu_only="gpu")
    assert wrong["a"]["host"]["verdict"] == HY.FAILS


# --- H4 -------------------------------------------------------------------------------

def test_h4_holm_over_the_locks_family_and_inconclusive_below_its_n():
    L = lk(h4_tests=("t1", "t2"), h4_n=200, h4_alpha=0.05)
    close = Counts(b=2, c=2, n=400)            # equal accuracy, tight bounds
    far = Counts(b=0, c=80, n=400)             # 20 points apart
    out = HY.h4({"t1": close, "t2": close}, locked=L)
    assert out["verdict"] == HY.HOLDS and out["m"] == 2
    out = HY.h4({"t1": close, "t2": far}, locked=L)
    assert out["tests"]["t2"]["verdict"] == HY.FAILS and out["verdict"] == HY.FAILS
    short = HY.h4({"t1": close, "t2": Counts(b=1, c=1, n=150)}, locked=L)
    assert short["verdict"] == HY.INCONCLUSIVE and short["below_power"] == ["t2"]
    with pytest.raises(HY.HypothesisError, match="the lock's 2"):
        HY.h4({"t1": close}, locked=L)


# --- pins from the break-test (T4.3 wave 2a) ---------------------------------------------

def test_h0_scores_an_operation_only_above_the_floor_on_both_pairings(p2):
    a = _file("P1", [_cell(p2)])
    hosts = sorted(h for h in a.cells[0].priced.host if h > 0)
    f = 0.75 * hosts[-1]                    # the largest is above it, its half below
    floors = FloorTable({(4, 1): f})
    b = _file("P2", [_cell(p2, scale_host=0.5)])
    out = HY.h0(a, b, side="host", floors=floors, tolerance=0.9)
    assert out["operations_scored"] == sum(h > f and 0.5 * h > f for h in a.cells[0].priced.host)
    assert out["operations_scored"] < sum(h > f for h in a.cells[0].priced.host)


def test_h0_takes_the_median_not_the_mean(p2):
    c = _cell(p2)
    pr = c.priced                                    # three copies: enough scored operations
    c = dataclasses.replace(c, priced=dataclasses.replace(
        pr, **{f.name: getattr(pr, f.name) * 3 for f in dataclasses.fields(pr)}))
    host = list(c.priced.host)
    assert sum(h > 0 for h in host) >= 3
    i = max(range(len(host)), key=lambda j: host[j])
    host[i] *= 3.0                                   # one operation 200% off, the rest equal
    other = dataclasses.replace(c, priced=dataclasses.replace(c.priced, host=tuple(host)))
    out = HY.h0(_file("P1", [c]), _file("P2", [other]), side="host", floors=FLOORS,
                tolerance=0.1)
    assert out["median_abs_relative_difference"] == 0.0 and out["verdict"] == HY.HOLDS
    assert out["max_abs_relative_difference"] == pytest.approx(2.0)


def test_h1s_recursion_must_be_strictly_below_each_rival(p2):
    cells = tuple(dataclasses.replace(c, predictions={**c.predictions,
                                                      "sum": c.predictions["recursion"]})
                  for c in p2.cells)
    out = HY.h1(dataclasses.replace(p2, cells=cells), seconds_of(p2), locked=lk(),
                dense_arm="dense")
    assert out["recursion_below"]["sum"] is False and out["verdict"] == HY.FAILS


def _counts_with_p_between(lo, hi):
    for n in range(40, 2000, 2):
        cnt = Counts(b=0, c=0, n=n)
        if lo < HY._tost_p(cnt, 10.0) <= hi:
            return cnt
    raise AssertionError("no such counts")


def test_h4_steps_down_holms_levels():
    """With m = 2 at alpha 0.05 the smallest p must clear 0.025: two tests
    with p in (0.025, 0.05] are not equivalent, though each would be alone."""
    cnt = _counts_with_p_between(0.025, 0.05)
    out = HY.h4({"t1": cnt, "t2": cnt}, locked=lk(h4_n=1, h4_alpha=0.05))
    assert out["verdict"] == HY.FAILS
    one = HY.h4({"t1": cnt}, locked=lk(h4_tests=("t1",), h4_n=1, h4_alpha=0.05))
    assert one["verdict"] == HY.HOLDS


def test_h4_is_two_sided_equivalence_not_non_inferiority():
    better = Counts(b=80, c=0, n=400)          # the second 20 points better: not equivalent
    out = HY.h4({"t1": better, "t2": Counts(b=2, c=2, n=400)}, locked=lk(h4_n=1))
    assert out["tests"]["t1"]["verdict"] == HY.FAILS


def test_h5b_fails_when_the_gpu_only_shift_is_not_smaller():
    first = {"host": diffs(1, 1, 0, -1, -1), "gpu": diffs(1, 0, -1, -1, -1)}
    second = {"host": diffs(1, 1, 1, -1, -1), "gpu": diffs(1, 1, 1, 0, -1)}
    out = HY.h5({"host": 1, "gpu": 1}, first, second, floors=GF, gpu_only="gpu")
    assert out["a"]["gpu"]["shift_grid_steps"] > out["a"]["host"]["shift_grid_steps"]
    assert out["b"] == HY.FAILS
