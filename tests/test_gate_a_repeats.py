"""Gate A over the repeat sessions (`attnbench/analysis/gate_a_repeats.py`):
the floors from two sessions' ground truth, and the plan's thresholds as
written, which never pass a leg that cannot be applied."""

from __future__ import annotations

import pytest

from attnbench.analysis import gate_a_repeats as GR
from attnbench.analysis.gate_a import GateAError, NOT_MEASURED
from attnbench.lock.floors import FloorsError


def row(arm, budget, batch, xs):
    return {"arm": arm, "budget": budget, "batch": batch, "prefill_ms_samples": list(xs)}


def result(*rows, traced=False):
    return {"traced": traced, "rows": list(rows)}


A = [("gt_L2048_b1.json", result(row("dense", 2048, 1, [100, 102, 98]),
                                 row("xattention", 2048, 1, [90, 91, 89])))]
B = [("gt_L2048_b1.json", result(row("dense", 2048, 1, [110, 112, 108]),
                                 row("xattention", 2048, 1, [90, 90, 90]))),
     ("gt_L4096_b1.json", result(row("dense", 4096, 1, [200, 200])))]


def test_the_floors_come_from_both_sessions_cell_by_cell():
    fl = GR.floors({"a": A, "b": B}, quantile=1.0, provisional=True, pairing="P1")
    cells = {tuple(c["cell"]): c for c in fl["cells"]}
    d = cells[("dense", 2048, 1)]
    assert d["median_ms"] == {"a": 100, "b": 110}
    assert d["between_session_spread"] == pytest.approx(10 / 105)
    assert d["within_session_spread"] == {"a": pytest.approx(4 / 100), "b": pytest.approx(4 / 110)}
    assert cells[("xattention", 2048, 1)]["between_session_spread"] == pytest.approx(0.0)
    # the noise floor at quantile 1 is the largest between-session spread
    assert fl["noise_floor_relative"] == pytest.approx(10 / 105)
    assert fl["n_cells_both"] == 2
    assert fl["only_in_one_session"] == [["dense", 4096, 1]]
    # |x - median| pooled: dense a 2,0,2 ; xatt a 1,0,1 ; dense b 2,0,2 ; xatt b 0,0,0 ; 4096 0,0
    assert fl["resolution_floor_ms"] == pytest.approx(2.0)
    assert fl["n_repeats"] == 14 and fl["provisional"] is True


def test_the_floors_refuse_one_session_a_traced_result_and_a_cell_timed_twice():
    with pytest.raises(FloorsError, match="two sessions"):
        GR.floors({"a": A}, quantile=0.95, provisional=True, pairing="P1")
    with pytest.raises(FloorsError, match="traced"):
        GR.floors({"a": A, "b": [("t", result(row("dense", 2048, 1, [1, 2]), traced=True))]},
                  quantile=0.95, provisional=True, pairing="P1")
    twice = [("x", result(row("dense", 2048, 1, [1, 2]))), ("y", result(row("dense", 2048, 1, [1, 2])))]
    with pytest.raises(GateAError, match="timed twice"):
        GR.floors({"a": A, "b": twice}, quantile=0.95, provisional=True, pairing="P1")
    with pytest.raises(FloorsError, match="P1"):
        GR.floors({"a": A, "b": B}, quantile=0.95, provisional=True, pairing="P2")


def rep(below=(True, True), bound=None, provisional=False, failing=None, slopes=None):
    fam = {"bound": bound, "bound_provisional": provisional}
    if bound is not None:
        fam["failing_cells"] = failing or {}
    return {"prediction_sha256": "f" * 64,
            "predictors": {"all": {"recursion_below": dict(zip(GR.LEG1_RIVALS, below)),
                                   "by_predictor": {"recursion": {"n": 35}}}},
            "families": fam, "slopes": slopes or {"status": NOT_MEASURED}}


def measured(ok):
    return {"status": "measured", "within_bound": ok}


def test_every_leg_holding_in_both_sessions_passes():
    r = {"a": rep(bound=0.1, slopes=measured(True)), "b": rep(bound=0.1, slopes=measured(True))}
    aw = GR.as_written(r)
    assert aw["gate"] == "passes" and aw["not_holding"] == []


@pytest.mark.parametrize("b_below", [(False, True), (True, False)])
def test_leg_one_fails_when_either_rival_beats_the_recursion_in_either_session(b_below):
    r = {"a": rep(bound=0.1, slopes=measured(True)),
         "b": rep(below=b_below, bound=0.1, slopes=measured(True))}
    aw = GR.as_written(r)
    assert aw["legs"]["leg1"]["verdict"] == GR.FAILS
    assert aw["gate"] == "does not pass" and aw["not_holding"] == ["leg1"]


def test_a_leg_with_no_bound_a_provisional_bound_or_no_slopes_is_not_applied_and_does_not_pass():
    aw = GR.as_written({"a": rep(), "b": rep()})
    assert aw["legs"]["leg2"]["verdict"] == GR.NOT_APPLIED
    assert aw["legs"]["leg3"]["verdict"] == GR.NOT_APPLIED
    assert aw["gate"] == "does not pass" and aw["not_holding"] == ["leg2", "leg3"]
    prov = GR.as_written({"a": rep(bound=0.1, provisional=True), "b": rep(bound=0.1, provisional=True)})
    assert prov["legs"]["leg2"]["verdict"] == GR.NOT_APPLIED
    one = GR.as_written({"a": rep(bound=0.1, slopes=measured(True)), "b": rep(bound=0.1)})
    assert one["legs"]["leg3"] == {"verdict": GR.NOT_APPLIED, "why": "the slopes are not measured in b"}


def test_leg_two_fails_on_a_failing_cell_in_either_session_and_refuses_mixed_bounds():
    r = {"a": rep(bound=0.1, slopes=measured(True)),
         "b": rep(bound=0.1, failing={"cell": "why"}, slopes=measured(True))}
    assert GR.as_written(r)["legs"]["leg2"]["verdict"] == GR.FAILS
    with pytest.raises(GateAError, match="different family bounds"):
        GR.as_written({"a": rep(bound=0.1), "b": rep(bound=0.2)})


def test_leg_three_fails_on_a_slope_outside_its_bound_and_gate_a_needs_two_sessions():
    r = {"a": rep(bound=0.1, slopes=measured(True)), "b": rep(bound=0.1, slopes=measured(False))}
    assert GR.as_written(r)["legs"]["leg3"]["verdict"] == GR.FAILS
    with pytest.raises(GateAError, match="both repeat sessions"):
        GR.as_written({"a": rep()})


def test_the_markdown_names_the_verdicts_and_the_per_cell_spread():
    fl = GR.floors({"a": A, "b": B}, quantile=0.95, provisional=True, pairing="P1")
    md = GR.to_markdown(GR.report(label="x", reports={"a": rep(), "b": rep()}, floors_=fl))
    assert "Gate A does not pass" in md and "provisional quantile" in md
    assert "| dense L2048 b1 | 100.0 | 110.0 | 0.0952 | 0.0400 | 0.0364 |" in md


def test_the_cell_floors_take_the_larger_of_within_and_between_per_length():
    fl = GR.cell_floors({"a": A, "b": B}, quantile=1.0, pairing="P1")
    assert set(fl) == {"2048x1", "4096x1"}
    c = fl["2048x1"]
    # within: |x - median| over the four 2048 series is at most 2; between:
    # dense's medians 100 and 110 differ by 10, xattention's by 0
    assert c["within_ms"] == pytest.approx(2.0)
    assert c["between_ms"] == pytest.approx(10.0)
    assert c["floor_ms"] == pytest.approx(10.0) and c["source"] == "between"
    assert c["n_between_series"] == 2
    # 4096 ran in one session: its within-session floor, labelled
    one = fl["4096x1"]
    assert one["between_ms"] is None and one["floor_ms"] == one["within_ms"] == 0.0
    assert one["source"] == "within" and one["n_between_series"] == 0


def test_a_group_short_of_clean_rounds_leaves_both_floors():
    """Appendix A, 9 Nov 2026 (ninth paste): T3.2c's 16384 groups (8 and 2
    clean rounds) leave the floors, so the cell's floor is within-session only."""
    def untraced(*rows, clean, complete=True):
        return {**result(*rows), "clean_rounds": clean, "complete": complete}
    other = ("gt_L4096_b1.json", untraced(row("dense", 4096, 1, [200, 200]), clean=10))
    a = [("gt_L2048_b1.json", untraced(row("dense", 2048, 1, [100, 130, 70]), clean=8,
                                       complete=False)), other]
    b = [("gt_L2048_b1.json", untraced(row("dense", 2048, 1, [110, 112, 108]), clean=10))]
    kept, excluded = GR.drop_short_groups({"a": a, "b": b}, min_clean_rounds=10)
    assert kept == {"a": [other], "b": b}
    assert excluded == [{"session": "a", "result": "gt_L2048_b1.json", "complete": False,
                         "clean_rounds": 8, "min_clean_rounds": 10}]
    c = GR.cell_floors(kept, quantile=1.0, pairing="P1")["2048x1"]
    assert c["between_ms"] is None and c["source"] == "within"
    assert c["floor_ms"] == pytest.approx(2.0)          # b's |x - median| only, not a's 30
    # a complete group at the minimum stays
    full = [("gt_L2048_b1.json", untraced(row("dense", 2048, 1, [100, 102, 98]), clean=10))]
    kept, excluded = GR.drop_short_groups({"a": full, "b": b}, min_clean_rounds=10)
    assert excluded == [] and GR.cell_floors(kept, quantile=1.0, pairing="P1")["2048x1"][
        "between_ms"] == pytest.approx(10.0)


def test_v2_leg3_reads_the_range_and_refuses_an_empty_one():
    """Adviser's ruling 2d of 11 Nov 2026 (T4.3 read A, finding 7a, 7f): a
    mechanism outside the range or with no slope fails; an empty v2 leg is
    refused, never vacuously held."""
    def v2(within, missing=()):
        return {"status": "measured", "range": [0.707, 1.414], "within_range": within,
                "missing": list(missing), "slopes": {}}
    both = lambda x, y: {"a": rep(slopes=x), "b": rep(slopes=y)}
    assert GR.as_written(both(v2({"m": True}), v2({"m": True})))["legs"]["leg3"]["verdict"] == GR.HOLDS
    assert GR.as_written(both(v2({"m": True}), v2({"m": False}, ["m"])))["legs"]["leg3"]["verdict"] \
        == GR.FAILS
    assert GR.as_written(both(v2({"m": True}), v2({"m": True}, ["m"])))["legs"]["leg3"]["verdict"] \
        == GR.FAILS
    with pytest.raises(GateAError, match="vacuously"):
        GR.as_written(both(v2({"m": True}), v2({})))
    with pytest.raises(GateAError, match="slope range"):
        GR.as_written(both(v2({"m": True}), measured(True)))


def test_leg1_carries_its_cell_counts_and_is_not_applied_without_them():
    r = {"a": rep(), "b": rep()}
    l1 = GR.as_written(r)["legs"]["leg1"]
    assert l1["verdict"] == GR.HOLDS and l1["n_cells"] == {"a": 35, "b": 35}
    r["b"]["predictors"]["all"]["by_predictor"]["recursion"]["n"] = 0
    assert GR.as_written(r)["legs"]["leg1"]["verdict"] == GR.NOT_APPLIED
    del r["b"]["predictors"]["all"]["by_predictor"]
    assert GR.as_written(r)["legs"]["leg1"]["verdict"] == GR.NOT_APPLIED
