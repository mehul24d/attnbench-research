"""Gate A's inputs (T3.3; `attnbench/analysis/gate_a.py`) on the committed P1
fixture, and the baselines' left-to-right sums.

Hand-worked from `test_predict_commit`'s predictions, 20 s (xattn) and 10 s
(dense) measured:

    xattn: recursion 16 -0.20, sum 13 -0.35, step 14 -0.30
    dense: recursion  8 -0.20, sum  8 -0.20, step  7 -0.30

so over both cells the medians of the absolute relative errors are 0.20
(recursion), 0.275 (sum) and 0.30 (step form), and on the dense cell alone
the recursion ties the sum.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

from attnbench.analysis import gate_a as G
from attnbench.components import op_sequence as ops
from attnbench.predict import baselines as B

import _figure_fixtures as FX
from test_predict_commit import MODEL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture
def cp(tmp_path):
    return FX.loaded_p1(tmp_path)[0]


def gt(**over):
    """The fixture's ground truth with rows labelled as `time_arms` labels
    them: the xattn arm at its nominal density, dense by name."""
    r = FX.ground_truth(**over)
    for row in r["rows"]:
        row["model"] = MODEL
    r["rows"][0]["arm"] = "xattn@0.5"
    return r


def seconds(cp):
    return G.measured(cp, [("gt.json", gt())])["seconds"]


# --- the baselines' sums --------------------------------------------------------------

def test_plain_sum_adds_left_to_right_on_every_python():
    xs = [1e16, 1.0, -1e16]
    assert B.plain_sum(xs) == 0.0                # (1e16 + 1) rounds to 1e16
    assert math.fsum(xs) == 1.0                  # a compensated sum (3.12's `sum`) keeps the 1
    assert B.plain_sum([]) == 0
    assert B.plain_sum(x for x in (0.5, 0.25)) == 0.75


def test_the_sum_baselines_use_the_left_to_right_order(monkeypatch):
    seen = []
    real = B.plain_sum
    monkeypatch.setattr(B, "plain_sum", lambda xs: seen.append(1) or real(xs))
    from test_predict_commit import COSTS, OPS
    from attnbench.predict.recursion import price
    p = price(OPS, COSTS)
    B.sum_as_written(p), B.sum_plus_s0(p), B.step_form(p), B.fitted_terms(p)
    assert len(seen) == 3 + 4 + 2 + 3


# --- placing rows on cells ------------------------------------------------------------

def test_rows_are_placed_on_the_committed_cells_by_label(cp):
    m = G.measured(cp, [("gt.json", gt())])
    assert sorted(k.arm for k in m["seconds"]) == ["dense", "xattn"]
    assert sorted(m["seconds"].values()) == [10.0, 20.0]
    assert m["unmeasured"] == []


def test_a_row_without_a_nominal_density_is_placed_by_its_executed_density(cp):
    r = gt()
    r["rows"][0]["arm"] = "xattn"
    r["rows"][0]["executed_density"] = 0.5 + 0.5 * G.DENSITY_TOL
    assert len(G.measured(cp, [("gt.json", r)])["seconds"]) == 2
    r["rows"][0]["executed_density"] = 0.55
    with pytest.raises(G.GateAError, match="'xattn' at 4 x 1 fits 0 committed cells"):
        G.measured(cp, [("gt.json", r)])


@pytest.mark.parametrize("change, match", [
    (lambda r: r["rows"][0].update(arm="xattn@0.25"), "fits 0 committed cells"),
    (lambda r: r["rows"][0].update(model="other"), "fits 0 committed cells"),
    (lambda r: r["rows"][0].update(context_length=8), "fits 0 committed cells"),
    (lambda r: r["rows"][0].update(arm="xattn@half"), "'half' is not a density"),
    (lambda r: r["rows"][0].pop("arm"), "no arm label"),
])
def test_a_row_that_fits_no_cell_is_refused(cp, change, match):
    r = gt()
    change(r)
    with pytest.raises(G.GateAError, match=match):
        G.measured(cp, [("gt.json", r)])


def test_a_cell_measured_twice_is_refused_and_an_unmeasured_one_is_listed(cp):
    with pytest.raises(G.GateAError, match="measured twice"):
        G.measured(cp, [("a.json", gt()), ("b.json", gt())])
    r = gt()
    r["rows"] = r["rows"][1:]
    m = G.measured(cp, [("gt.json", r)])
    assert [k.arm for k in m["unmeasured"]] == ["xattn"]


def test_traced_or_incomplete_ground_truth_is_refused(cp):
    from attnbench.figures.results import ResultsError
    for over, match in ((dict(traced=True), "traced"), (dict(complete=False), "complete")):
        with pytest.raises(ResultsError, match=match):
            G.measured(cp, [("gt.json", gt(**over))])


# --- leg 1 ----------------------------------------------------------------------------

def test_leg_one_is_the_hand_worked_comparison(cp):
    out = G.predictors(cp, seconds(cp), dense_arm="dense", relative_floor=0.25)
    a = out["all"]["by_predictor"]
    assert a["recursion"]["median_abs_relative"] == pytest.approx(0.20)
    assert a["sum"]["median_abs_relative"] == pytest.approx(0.275)
    assert a["step_form"]["median_abs_relative"] == pytest.approx(0.30)
    assert a["recursion"]["n_inside_floor"] == 2 and a["sum"]["n_inside_floor"] == 1
    assert out["all"]["recursion_below"] == {"sum": True, "step_form": True}
    assert out["dense"]["recursion_below"] == {"sum": False, "step_form": True}   # a tie
    assert out["synchronising"]["by_predictor"]["recursion"]["n"] == 1
    assert out["synchronisation-free"] is None


def test_leg_one_needs_the_recursion(cp):
    with pytest.raises(G.GateAError, match="name it"):
        G.predictors(cp, seconds(cp), dense_arm="dense", relative_floor=0.0,
                     predictors=("sum", "step_form"))


# --- the descriptive per-arm table ----------------------------------------------------

def test_sanity_is_labelled_descriptive_and_scores_the_sign_against_dense(cp):
    s = G.sanity(cp, seconds(cp), dense_arm="dense", resolution_floor_s=0.0)
    assert "descriptive" in s["label"] and "not a hypothesis test" in s["label"]
    x = s["arms"]["xattn"]["recursion"]
    assert x["median_abs_relative"] == pytest.approx(0.20)
    # predicted 16 against dense's 8, measured 20 against 10: sparse slower both ways
    assert (x["sign_agree"], x["sign_disagree"], x["inside_floor"]) == (1, 0, 0)
    assert "sign_agree" not in s["arms"]["dense"]["recursion"]
    s = G.sanity(cp, seconds(cp), dense_arm="dense", resolution_floor_s=10.0)
    assert s["arms"]["xattn"]["sum"]["inside_floor"] == 1         # |20 - 10| <= 10


def test_a_sign_disagreement_is_counted(cp):
    secs = G.measured(cp, [("gt.json", gt(xattn_ms=5000.0))])["seconds"]
    x = G.sanity(cp, secs, dense_arm="dense", resolution_floor_s=0.0)["arms"]["xattn"]
    assert all((v["sign_agree"], v["sign_disagree"]) == (0, 1) for v in x.values())


# --- leg 2 ----------------------------------------------------------------------------

def _seq_and_trace():
    doc = FX.trace()
    seq = FX.parse(doc)
    return seq, ops.device_durations(doc, seq)


def test_a_cells_report_compares_its_priced_device_times_with_the_trace():
    seq, trace_s = _seq_and_trace()
    priced = [{"device": trace_s.get(i, 0.0)} for i in range(len(seq.ops))]
    rep = G.replay_report(seq, trace_s, priced)
    assert rep["positions_compared"] == len(trace_s)
    assert all(f["relative_difference"] == 0.0 for f in rep["families"].values())
    i = sorted(trace_s)[0]
    priced[i] = {"device": trace_s[i] * 1.5}
    rep = G.replay_report(seq, trace_s, priced)
    assert max(abs(f["relative_difference"]) for f in rep["families"].values()) > 0
    with pytest.raises(G.GateAError, match="priced operations for a sequence of"):
        G.replay_report(seq, trace_s, priced[:-1])


def _reports():
    seq, trace_s = _seq_and_trace()
    good = G.replay_report(seq, trace_s, [{"device": trace_s.get(i, 0.0)}
                                          for i in range(len(seq.ops))])
    bad = copy.deepcopy(good)
    fam = sorted(bad["families"])[0]
    bad["families"][fam]["replay_s"] *= 1.2
    bad["families"][fam]["relative_difference"] = 0.2
    return {"a": good, "b": bad}, fam


def test_leg_two_gives_the_spread_and_at_a_bound_the_failing_cells():
    reps, fam = _reports()
    out = G.families(reps, bound=None, bound_provisional=False)
    assert "failing_cells" not in out and out["cells"] == 2
    e = out["families"][fam]
    assert e["cells"] == 2
    assert e["abs_relative_difference"]["max"] == pytest.approx(0.2)
    assert e["abs_relative_difference"]["median"] == pytest.approx(0.1)
    assert e["pooled_relative_difference"] == pytest.approx(
        (e["replay_s"] - e["trace_s"]) / e["trace_s"])
    out = G.families(reps, bound=0.1, bound_provisional=True)
    assert list(out["failing_cells"]) == ["b"] and out["bound_provisional"] is True
    assert list(G.families(reps, bound=0.25, bound_provisional=True)["failing_cells"]) == []
    with pytest.raises(G.GateAError, match="no cell"):
        G.families({}, bound=None, bound_provisional=False)


# --- leg 3 and the report -------------------------------------------------------------

def test_leg_three_is_not_measured_without_slopes():
    out = G.slopes(None, bound=0.1, mechanisms=("item",))
    assert out["status"] == G.NOT_MEASURED and "no delay injection" in out["why"]


def test_the_report_holds_every_leg_and_leaves_the_decision_to_the_researcher(cp, tmp_path):
    reps, _ = _reports()
    rep = G.report(label="fixture", cp=cp, results=[("gt.json", gt())], dense_arm="dense",
                   relative_floor=0.25, resolution_floor_s=0.0, family_reports=reps,
                   family_bound=0.1, family_bound_provisional=True)
    assert rep["decision"].startswith("the researcher's")
    assert rep["slopes"]["status"] == G.NOT_MEASURED
    assert rep["measured_cells"] == 2 and rep["prediction_sha256"] == cp.sha256
    j, md = G.write(rep, tmp_path / "out")
    assert json.loads(j.read_text())["label"] == "fixture"
    text = md.read_text()
    for needle in ("# Gate A inputs: fixture", "Leg 1", "| all | recursion | 2 | 0.2000",
                   "bound 0.1 (provisional)", "Cells outside the bound: 1.",
                   "Leg 3", "not measured", "descriptive"):
        assert needle in text, needle


def test_the_script_refuses_a_provisional_flag_without_a_bound(capsys):
    import gate_a as script
    argv = ["--label", "x", "--remote", "gs://b/p.json", "--local", "p.json",
            "--ground-truth", "g", "--cells", "c", "--recordings", "r",
            "--relative-floor", "0", "--resolution-floor-ms", "0", "--out", "o",
            "--family-bound-provisional"]
    assert script.main(argv) == 2
    assert "needs --family-bound" in capsys.readouterr().err


def test_the_script_reads_a_step_nine_directory(tmp_path):
    import gate_a as script
    d = tmp_path / "09_ground_truth" / "eager"
    d.mkdir(parents=True)
    (d / "stamps.json").write_text(json.dumps({"clock_not_before": 5.0,
                                               "first_clocked_block_started_at": [6.0, 7.0]}))
    (d / "gt_L4_b1.json").write_text(json.dumps(gt()))
    results, starts = script.ground_truth(tmp_path / "09_ground_truth")
    assert [n for n, _ in results] == ["eager/gt_L4_b1.json"] and starts == [5.0, 6.0, 7.0]
    with pytest.raises(G.GateAError, match="no ground truth"):
        script.ground_truth(tmp_path / "empty")


def test_the_recorded_sequence_is_the_banked_one_parsed_again(tmp_path):
    t, s = FX.write_pair(tmp_path, FX.trace())
    seq = G.recorded_sequence(t, s)
    assert all(op.correlation is not None for _, op in seq.device_ops())
    assert ops.device_durations(t, seq)                       # what the banked record cannot do
    banked = ops.from_record({k: v for k, v in json.loads(s.read_text()).items() if k != "sha256"})
    assert all(op.correlation is None for _, op in banked.device_ops())


def test_a_recorded_sequence_that_differs_from_the_banked_one_is_refused(tmp_path):
    doc = FX.trace()
    t, s = FX.write_pair(tmp_path, doc)
    rec = json.loads(s.read_text())
    rec["context_length"] += 1
    s.write_text(json.dumps(rec))
    with pytest.raises(G.GateAError, match="does not match its own sha256"):
        G.recorded_sequence(t, s)
    other = FX.with_event(doc, FX._x("cpu_op", "aten::relu", 360, 5))
    other["traceEvents"] += [FX._call("cudaLaunchKernel", 361, 2, 99),
                             FX._dev("kernel", "relu", 560, 5, 99)]
    t2, _ = FX.write_pair(tmp_path, other, name="b")
    _, s1 = FX.write_pair(tmp_path, doc, name="c")
    with pytest.raises(G.GateAError, match="does not parse to the banked"):
        G.recorded_sequence(t2, s1)
