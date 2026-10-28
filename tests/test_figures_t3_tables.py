"""T3.1 and T3.3 tables (C5) on the committed P1 fixture.

Hand-worked from `test_predict_commit`'s predictions and 20 s (xattn), 10 s
(dense) measured:

    xattn: recursion 16 -0.20, sum 13 -0.35, sum+s0 16 -0.20, step 14 -0.30
    dense: recursion  8 -0.20, sum  8 -0.20, sum+s0  9 -0.10, step  7 -0.30
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from attnbench.figures import results as RS
from attnbench.figures import t3_tables as T
from attnbench.predict.commit import cell_key, verify_bytes

import _figure_fixtures as FX

FLOOR = 0.25
KW = dict(dense_arms=["dense"], relative_floor=FLOOR)


@pytest.fixture
def cp(tmp_path):
    return FX.loaded_p1(tmp_path)[0]


def measured(**over):
    gt = FX.ground_truth(**over)
    return RS.measured_cells([("gt.json", gt, {"xattn": FX.XATTN, "dense": FX.DENSE})],
                             statistic="median")


def test_measured_cells_are_seconds_keyed_by_the_stated_cell():
    m = measured()
    assert m == {cell_key(*FX.XATTN): 20.0, cell_key(*FX.DENSE): 10.0}


@pytest.mark.parametrize("arms, match", [
    ({"xattn": FX.DENSE}, "stated to measure a cell of arm 'dense'"),
    ({"other": (FX.MODEL, FX.CP, "other", 0.5, 4, 1)}, "0 rows for arm 'other'"),
])
def test_a_row_stated_for_the_wrong_cell_is_refused(arms, match):
    with pytest.raises(RS.ResultsError, match=match):
        RS.measured_cells([("gt.json", FX.ground_truth(), arms)], statistic="median")


def test_a_cell_measured_twice_or_from_bad_ground_truth_is_refused():
    gt = FX.ground_truth()
    with pytest.raises(RS.ResultsError, match="measured twice"):
        RS.measured_cells([("a", gt, {"xattn": FX.XATTN}), ("b", gt, {"xattn": FX.XATTN})],
                          statistic="median")
    for over, match in ((dict(traced=True), "traced"), (dict(complete=False), "complete")):
        with pytest.raises(RS.ResultsError, match=match):
            RS.measured_cells([("a", FX.ground_truth(**over), {"xattn": FX.XATTN})],
                              statistic="median")
    with pytest.raises(RS.ResultsError, match="positive median"):
        RS.measured_cells([("a", FX.ground_truth(xattn_ms=None), {"xattn": FX.XATTN})],
                          statistic="median")


def test_the_cell_table_is_the_hand_worked_one(cp):
    cols, rows = T.cell_table(cp, measured(), **KW)
    assert cols[:9] == T.CELL_FIELDS
    assert [r["arm"] for r in rows] == ["dense", "xattn"]           # sorted by key
    d, x = rows
    assert (x["measured_s"], x["n_synchronisations"], x["arm_class"]) == (
        20.0, 1, "synchronising")
    assert (d["measured_s"], d["n_synchronisations"], d["arm_class"]) == (10.0, 0, "dense")
    want = {"xattn": {"recursion": (16, -0.20), "sum": (13, -0.35),
                      "sum_plus_s0": (16, -0.20), "step_form": (14, -0.30)},
            "dense": {"recursion": (8, -0.20), "sum": (8, -0.20),
                      "sum_plus_s0": (9, -0.10), "step_form": (7, -0.30)}}
    for r in rows:
        for p, (pred, rel) in want[r["arm"]].items():
            assert r[f"{p}_s"] == pred
            assert r[f"{p}_relative"] == pytest.approx(rel)
            assert r[f"{p}_inside_floor"] == (abs(rel) <= FLOOR)


def test_the_predictor_summary_is_the_hand_worked_one(cp):
    cols, rows = T.predictor_summary(cp, measured(), by_arm_class=False, **KW)
    assert cols == ("predictor",) + T.SUMMARY_FIELDS
    got = {r["predictor"]: r for r in rows}
    assert [r["predictor"] for r in rows] == ["recursion", "sum", "sum_plus_s0", "step_form"]
    for p, (med, mean, mx, inside) in {
            "recursion": (0.20, -0.20, 0.20, 2), "sum": (0.275, -0.275, 0.35, 1),
            "sum_plus_s0": (0.15, -0.15, 0.20, 2), "step_form": (0.30, -0.30, 0.30, 0)}.items():
        r = got[p]
        assert r["n"] == 2 and r["n_inside_floor"] == inside
        assert r["median_abs_relative"] == pytest.approx(med)
        assert r["mean_relative"] == pytest.approx(mean)
        assert r["max_abs_relative"] == pytest.approx(mx)
    _, by_class = T.predictor_summary(cp, measured(), by_arm_class=True, **KW)
    assert len(by_class) == 8
    assert {(r["predictor"], r["arm_class"]) for r in by_class} >= {
        ("recursion", "dense"), ("recursion", "synchronising")}


def test_unmatched_cells_are_listed_not_dropped(cp):
    only_dense = RS.measured_cells([("gt", FX.ground_truth(), {"dense": FX.DENSE})],
                                   statistic="median")
    extra = dict(measured())
    extra[cell_key(FX.MODEL, FX.CP, "xattn", 0.25, 4, 1)] = 5.0
    assert T.unmatched(cp, only_dense, **KW)["unmeasured"] == [list(cell_key(*FX.XATTN))]
    assert T.unmatched(cp, extra, **KW)["unpredicted"] == [
        list(cell_key(FX.MODEL, FX.CP, "xattn", 0.25, 4, 1))]
    _, rows = T.cell_table(cp, only_dense, **KW)
    assert [r["arm"] for r in rows] == ["dense"]


def test_a_prediction_checked_by_hash_alone_is_refused(tmp_path):
    cp, path, _, _ = FX.loaded_p1(tmp_path)
    side = path.with_name(path.name + ".sha256").read_text()
    unsynced = verify_bytes(path.read_bytes(), side, path.name)
    for f in (lambda: T.cell_table(unsynced, measured(), **KW),
              lambda: T.predictor_summary(unsynced, measured(), by_arm_class=False, **KW)):
        with pytest.raises(RS.ResultsError, match="load_committed"):
            f()


def test_the_floor_has_no_default_and_predictors_are_named_once(cp):
    with pytest.raises(TypeError):
        T.cell_table(cp, measured(), dense_arms=["dense"])
    with pytest.raises(T.TableError, match="once"):
        T.cell_table(cp, measured(), predictors=("recursion", "recursion"), **KW)


def test_csv_keeps_precision_and_markdown_rounds(cp):
    cols, rows = T.cell_table(cp, measured(), **KW)
    csv_text = T.to_csv(cols, rows)
    assert csv_text.splitlines()[0] == ",".join(cols)
    assert "-0.35" in csv_text
    md = T.to_markdown(cols, rows, caption="P1")
    assert md.startswith("**P1**")
    assert "| -0.3500 |" in md and "| 20.000000 |" in md and "| True |" in md
    with pytest.raises(T.TableError, match="fields"):
        T.to_csv(cols[:3], rows)


def test_the_script_writes_every_table_once(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "figures"))
    import t3_tables as script
    cp, path, store, remote = FX.loaded_p1(tmp_path)
    (tmp_path / "gt.json").write_text(json.dumps(FX.ground_truth()))
    (tmp_path / "cells.json").write_text(json.dumps(
        {"results": [{"file": "gt.json", "arms": {"xattn": list(FX.XATTN),
                                                  "dense": list(FX.DENSE)}}]}))
    out = tmp_path / "tables"
    argv = ["--remote", remote, "--local", str(path), "--cells", str(tmp_path / "cells.json"),
            "--dense-arm", "dense", "--relative-floor", str(FLOOR), "--clock-skew-s", "0",
            "--out-dir", str(out)]
    assert script.main(argv, store=store) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([f"{n}.{x}" for n in ("t31_cells", "t33_by_predictor",
                                                  "t33_by_predictor_and_arm_class")
                            for x in ("csv", "md")] + ["unmatched.json"])
    assert "noise floor 0.25" in (out / "t31_cells.md").read_text()
    assert script.main(argv, store=store) == 2                      # never overwrites
