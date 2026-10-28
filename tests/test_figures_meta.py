"""A figure's meta.json (C5): every input hashed, the prediction read as the
analysis reads it, ground truth untraced and complete."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from attnbench.figures import meta as M
from attnbench.predict import commit as C
from attnbench.predict.commit import CodeStamp, verify_bytes

import _figure_fixtures as FX

CLEAN = CodeStamp(git_commit="b" * 40, git_dirty=False)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    cp, pred, store, remote = FX.loaded_p1(tmp_path)
    res = tmp_path / "ground_truth.json"
    res.write_text(json.dumps(FX.ground_truth()))
    trace, seq = FX.write_pair(tmp_path, FX.trace())
    return dict(tmp=tmp_path, cp=cp, pred=pred, res=res, trace=trace, seq=seq,
                store=store, remote=remote)


def _meta(s, **over):
    kw = dict(figure="fig1", prediction=s["cp"], prediction_path=s["pred"], cell=FX.XATTN,
              result=json.loads(s["res"].read_text()), result_path=s["res"],
              inputs=[("trace", s["trace"]), ("sequence", s["seq"])], code=CLEAN)
    kw.update(over)
    return M.figure_meta(**kw)


def test_the_metadata_names_and_hashes_every_input(setup):
    s = setup
    m = _meta(s)
    assert m["schema"] == M.META_SCHEMA and m["figure"] == "fig1"
    by_role = {e["role"]: e for e in m["inputs"]}
    assert set(by_role) == {"prediction", "prediction_sidecar", "result", "trace", "sequence"}
    for role, path in (("prediction", s["pred"]), ("result", s["res"]),
                       ("trace", s["trace"]), ("sequence", s["seq"])):
        assert by_role[role]["sha256"] == _sha(path)
        assert by_role[role]["name"] == path.name           # a name, never a path
        assert str(s["tmp"]) not in json.dumps(m)
    assert m["prediction"]["sha256"] == s["cp"].sha256 == _sha(s["pred"])
    assert m["prediction"]["synced_at"] == "2026-11-02T03:00:05Z"
    assert m["generated_by"] == {"git_commit": "b" * 40, "git_dirty": False}


def test_the_cell_carries_its_committed_predictions_and_the_measured_seconds(setup):
    m = _meta(setup)
    assert m["cell"]["arm"] == "xattn" and m["cell"]["density"] == 0.5
    assert m["cell"]["predictions_s"] == {"recursion": 16, "step_form": 14, "sum": 13,
                                          "sum_plus_s0": 16}
    # 20000 ms in the result is 20 s beside predictions in seconds.
    assert m["measured"]["seconds"] == 20.0 and m["measured"]["statistic"] == "median"
    assert m["measured"]["first_timed_block_started_at"] == FX.STARTED_AT.timestamp()
    assert "stated by the caller" in m["measured"]["matched_by"]


def test_a_prediction_checked_by_hash_alone_is_refused(setup):
    s = setup
    side = s["pred"].with_name(s["pred"].name + ".sha256").read_text()
    unsynced = verify_bytes(s["pred"].read_bytes(), side, s["pred"].name)
    assert unsynced.synced_at is None
    with pytest.raises(M.MetaError, match="load_committed"):
        _meta(s, prediction=unsynced)


def test_the_file_on_disk_must_be_the_committed_one(setup):
    s = setup
    other = s["tmp"] / "copy.json"
    other.write_bytes(s["pred"].read_bytes() + b" ")
    with pytest.raises(M.MetaError, match="not the committed file"):
        _meta(s, prediction_path=other)
    s["pred"].with_name(s["pred"].name + ".sha256").unlink()
    with pytest.raises(M.MetaError, match="sidecar|sha256"):
        _meta(s)


@pytest.mark.parametrize("over, match", [
    (dict(traced=True), "traced"),
    (dict(traced=None), "traced"),
    (dict(complete=False), "not complete"),
])
def test_ground_truth_must_be_untraced_and_complete(setup, over, match):
    with pytest.raises(M.MetaError, match=match):
        _meta(setup, result=FX.ground_truth(**over))


def test_the_cells_arm_must_have_one_measured_row(setup):
    gt = FX.ground_truth()
    gt["rows"] = [r for r in gt["rows"] if r["arm"] != "xattn"]
    with pytest.raises(M.MetaError, match="0 rows for arm 'xattn'"):
        _meta(setup, result=gt)
    with pytest.raises(M.MetaError, match="no cell"):
        _meta(setup, cell=(FX.MODEL, FX.CP, "xattn", 0.25, 4, 1))


def test_a_dirty_or_unknown_commit_is_refused(setup):
    for code in (CodeStamp("b" * 40, True), CodeStamp("b" * 40, None), CodeStamp("abc", False)):
        with pytest.raises(M.MetaError, match="clean"):
            _meta(setup, code=code)


def test_inputs_need_a_known_role_and_distinct_names(setup):
    s = setup
    with pytest.raises(M.MetaError, match="role"):
        _meta(s, inputs=[("notes", s["trace"])])
    with pytest.raises(M.MetaError, match="two inputs named"):
        _meta(s, inputs=[("trace", s["trace"]), ("other", s["trace"])])
    with pytest.raises(M.MetaError, match="not a file"):
        _meta(s, inputs=[("trace", s["tmp"] / "missing.json")])


def test_the_metadata_is_written_once(setup):
    out = setup["tmp"] / "meta.json"
    M.write_meta(_meta(setup), out)
    assert json.loads(out.read_text())["figure"] == "fig1"
    with pytest.raises(FileExistsError):
        M.write_meta(_meta(setup), out)


def _script():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "figures"))
    import figure_meta
    return figure_meta


def _argv(s, out):
    return ["--figure", "fig1", "--remote", s["remote"], "--local", str(s["pred"]),
            "--result", str(s["res"]), "--cell", FX.MODEL, FX.CP, "xattn", "0.5", "4", "1",
            "--clock-skew-s", "0", "--input", f"trace={s['trace']}", "--out", str(out)]


def test_the_script_reads_the_prediction_from_the_store(setup, monkeypatch):
    s = setup
    script = _script()
    monkeypatch.setattr(C, "current_code_stamp", lambda: CLEAN)
    out = s["tmp"] / "meta.json"
    assert script.main(_argv(s, out), store=s["store"]) == 0
    assert json.loads(out.read_text())["measured"]["seconds"] == 20.0


def test_the_script_refuses_a_prediction_synced_after_the_workload_started(setup, monkeypatch,
                                                                          capsys):
    s = setup
    script = _script()
    monkeypatch.setattr(C, "current_code_stamp", lambda: CLEAN)
    early = FX.ground_truth(first_timed_block_started_at=(FX.SYNC_AT - timedelta(seconds=1))
                            .timestamp())
    s["res"].write_text(json.dumps(early))
    assert script.main(_argv(s, s["tmp"] / "meta.json"), store=s["store"]) == 2
    assert "K9" in capsys.readouterr().err
