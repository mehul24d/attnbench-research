"""P1's training rows for the fitted baselines (`attnbench/predict/p1_training.py`;
researcher's decision, 8 Nov 2026): one row per cell, the target the median of
both repeats' samples pooled, each repeat's median kept as metadata, a
repeat-a-only sensitivity set, cross-validation groups by cell, no session
but the two repeats, and repeat a alone if T3.2b fails."""

from __future__ import annotations

import pytest

from attnbench.predict import p1_training as PT
from attnbench.predict.training import HardwareFeatures

import _figure_fixtures as FX
from test_predict_commit import MODEL

HW = HardwareFeatures(pinned_bandwidth=2e10, pageable_bandwidth=1e10)
REMOTE = "gs://bucket/predictions/P1-{}/sess/prediction_P1_x.json"


def gt(xattn, dense, *, xattn_density=0.55):
    """A ground-truth result with samples (ms) for the fixture's two cells."""
    r = FX.ground_truth()
    for row, xs in zip(r["rows"], (xattn, dense)):
        row.update(model=MODEL, budget=4, prefill_ms_samples=list(xs))
    r["rows"][0]["arm"] = "xattn@0.5"
    r["rows"][0]["nominal_density"] = 0.5
    r["rows"][1]["nominal_density"] = None
    r["rows"][0]["executed_density"] = xattn_density
    return r


def session(tmp_path, label, result, remote=None, xattn_ops=None):
    d = tmp_path / f"{label}_{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    cp = FX.loaded_p1(d, xattn_ops=xattn_ops)[0]
    return PT.RepeatSession(label=label, remote=remote or REMOTE.format(label), cp=cp,
                            results=[("gt_L4_b1.json", result)], hardware=HW)


def test_one_row_per_cell_whose_target_is_the_median_of_both_repeats_pooled(tmp_path):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]))
    b = session(tmp_path, "T3.2b", gt([40, 50, 60, 70], [8, 9]), xattn_ops=FX.OPS_HOST)
    out = PT.training_rows(a, b)
    assert out["primary"] == PT.PRIMARY and out["repeats"] == ["T3.2a", "T3.2b"]
    rows = {r.cell.arm: r for r in out["rows"]}
    assert len(out["rows"]) == 2
    # pooled xattn: 10 20 30 40 50 60 70 -> 40 ms; dense: 5 6 7 8 9 -> 7 ms
    assert rows["xattn"].measured == pytest.approx(0.040)
    assert rows["dense"].measured == pytest.approx(0.007)
    meta = {tuple(m["cell"]): m for m in out["metadata"]}
    x = meta[("xattn@0.5", 4, 1)]
    assert (x["median_a_s"], x["median_b_s"], x["n_a"], x["n_b"]) == \
        (pytest.approx(0.020), pytest.approx(0.055), 3, 4)
    assert x["between_session_spread"] == pytest.approx(0.035 / 0.0375)
    assert x["target_s"] == pytest.approx(0.040)
    # features are repeat a's committed cell
    ca = next(c for c in a.cp.cells if c.arm == "xattn")
    cb = next(c for c in b.cp.cells if c.arm == "xattn")
    assert ca.priced != cb.priced
    assert rows["xattn"].cell.priced == ca.priced and rows["xattn"].cell.hardware == HW
    assert rows["xattn"].cell.pairing == "P1"


def test_the_repeat_a_sensitivity_set_is_separate_and_never_the_primary(tmp_path):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]))
    b = session(tmp_path, "T3.2b", gt([40, 50, 60, 70], [8, 9]))
    out = PT.training_rows(a, b)
    sens = {r.cell.arm: r.measured for r in out["sensitivity"][PT.REPEAT_A_ONLY]}
    assert sens == {"xattn": pytest.approx(0.020), "dense": pytest.approx(0.006)}
    assert [r.measured for r in out["rows"]] != [r.measured for r in out["sensitivity"][PT.REPEAT_A_ONLY]]


def test_the_separate_rows_set_has_each_repeats_median_as_its_own_row(tmp_path):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]))
    b = session(tmp_path, "T3.2b", gt([40, 50, 60, 70], [8, 9]), xattn_ops=FX.OPS_HOST)
    out = PT.training_rows(a, b)
    sep = out["sensitivity"][PT.SEPARATE_ROWS]
    assert sorted((r.cell.arm, round(r.measured, 6)) for r in sep) == \
        [("dense", 0.006), ("dense", 0.0085), ("xattn", 0.02), ("xattn", 0.055)]
    ca = next(c for c in a.cp.cells if c.arm == "xattn")
    assert all(r.cell.priced == ca.priced for r in sep if r.cell.arm == "xattn")   # a's features
    g = PT.cell_groups(sep)
    assert len(set(g)) == 2 and g.count(g[0]) == 2
    rec = PT.to_record(out, HW)
    back = PT.rows_from_record(rec, sensitivity=PT.SEPARATE_ROWS)
    assert [(r.cell.key, r.measured) for r in back] == [(r.cell.key, r.measured) for r in sep]
    failed = PT.training_rows(a, None, b_failed="x")
    assert [r.measured for r in failed["sensitivity"][PT.SEPARATE_ROWS]] == \
        [r.measured for r in failed["rows"]]


def test_if_t32b_failed_the_row_is_repeat_a_alone_and_says_so(tmp_path):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]))
    out = PT.training_rows(a, None, b_failed="T3.2b stopped at step 6")
    assert {r.cell.arm: r.measured for r in out["rows"]} == \
        {"xattn": pytest.approx(0.020), "dense": pytest.approx(0.006)}
    assert out["repeats"] == ["T3.2a"] and out["b_failed"] == "T3.2b stopped at step 6"
    assert all(m["median_b_s"] is None and m["n_b"] == 0 for m in out["metadata"])
    with pytest.raises(PT.TrainingRowsError, match="not both"):
        PT.training_rows(a, None)
    b = session(tmp_path, "T3.2b", gt([1, 2], [3, 4]))
    with pytest.raises(PT.TrainingRowsError, match="not both"):
        PT.training_rows(a, b, b_failed="x")


@pytest.mark.parametrize("prefix", ["P1-debug", "P1-T3.1-exp", "P1", "P1-T3.2b"])
def test_only_the_two_repeats_train_never_an_exploratory_or_debug_session(tmp_path, prefix):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]),
                remote=f"gs://bucket/predictions/{prefix}/sess/p.json")
    with pytest.raises(PT.TrainingRowsError, match="only the T3.2 repeats"):
        PT.training_rows(a, None, b_failed="x")


def test_the_repeats_must_be_labelled_in_order_and_time_the_same_cells(tmp_path):
    a = session(tmp_path, "T3.2b", gt([10], [5]))
    with pytest.raises(PT.TrainingRowsError, match="expected repeat T3.2a"):
        PT.training_rows(a, None, b_failed="x")
    a = session(tmp_path, "T3.2a", gt([10, 20], [5, 6]))
    short = gt([1, 2], [3, 4])
    short["rows"] = short["rows"][:1]
    b = session(tmp_path, "T3.2b", short)
    with pytest.raises(PT.TrainingRowsError, match="different cells"):
        PT.training_rows(a, b)


def test_a_traced_result_is_not_ground_truth(tmp_path):
    r = gt([10], [5])
    r["traced"] = True
    with pytest.raises(PT.TrainingRowsError, match="traced"):
        PT.training_rows(session(tmp_path, "T3.2a", r), None, b_failed="x")


def test_cross_validation_groups_rows_by_cell_in_any_row_set(tmp_path):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]))
    b = session(tmp_path, "T3.2b", gt([40, 50], [8, 9]))
    out = PT.training_rows(a, b)
    rows = list(out["rows"]) + list(out["sensitivity"][PT.REPEAT_A_ONLY])
    g = PT.cell_groups(rows)
    assert g[0] == g[2] and g[1] == g[3] and g[0] != g[1]


def test_the_banked_file_round_trips_and_refuses_an_edit(tmp_path):
    a = session(tmp_path, "T3.2a", gt([10, 20, 30], [5, 6, 7]))
    b = session(tmp_path, "T3.2b", gt([40, 50, 60, 70], [8, 9]), xattn_ops=FX.OPS_HOST)
    out = PT.training_rows(a, b)
    rec = PT.to_record(out, HW)
    back = PT.rows_from_record(rec)
    assert [(r.cell.key, r.cell.priced, r.measured) for r in back] == \
        [(r.cell.key, r.cell.priced, r.measured) for r in out["rows"]]
    sens = PT.rows_from_record(rec, sensitivity=PT.REPEAT_A_ONLY)
    assert [r.measured for r in sens] == [r.measured for r in out["sensitivity"][PT.REPEAT_A_ONLY]]
    assert rec["predictions"] == {"T3.2a": a.cp.sha256, "T3.2b": b.cp.sha256}
    edited = dict(rec, rows=[dict(rec["rows"][0], target_s=1.0)] + rec["rows"][1:])
    with pytest.raises(PT.TrainingRowsError, match="sha256"):
        PT.rows_from_record(edited)
    with pytest.raises(PT.TrainingRowsError, match="no sensitivity set"):
        PT.rows_from_record(rec, sensitivity="repeat_b_only")
