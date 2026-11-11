"""P2's training rows for P4's committed fits (`attnbench/predict/p2_training.py`;
adviser's ruling 1h of 11 Nov 2026): one row per cell from P2's passed
blocks, the anchor pooled across them, a short group's cells left out and
named, only P2's own prediction, and P4 trained on P1's rows and these."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest

from attnbench.predict import baselines as BL
from attnbench.predict import p1_training as PT
from attnbench.predict import p2_training as P2T
from attnbench.predict.commit import load_committed
from attnbench.predict.training import committed_training_set

import _figure_fixtures as FX
import test_p1_training as TP1
from test_predict_commit import (CLEAN, CODE_PATHS, T0, MemoryStore, Clock, OPS_HOST, cell,
                                 build_record)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t61_session as S  # noqa: E402

REMOTE = "gs://bucket/predictions/P2/t61_p2_b1/prediction_P2_x.json"


def loaded_p2(tmp_path):
    store = MemoryStore(Clock(T0))
    record = build_record(pairing="P2", cells=[cell(pairing="P2"), cell(
        pairing="P2", ops=OPS_HOST, arm="dense", density=1.0)], queue_depth=8, banked=[],
        fitted_sum=None, learned=None, code=CLEAN, written_at=T0, code_paths=CODE_PATHS)
    d = tmp_path / f"p2_{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    f, receipt = FX.committed(d, record, store, FX.SYNC_AT)
    return load_committed(store, receipt.remote, ground_truth_started_at=FX.STARTED_AT,
                          clock_skew=FX.NO_SKEW, sync_gap=FX.NO_SKEW, banked_at={},
                          local=f.path)


def block(xattn=None, dense=None, **over):
    """One block's ground-truth result: the xattn cell and/or the dense anchor."""
    r = TP1.gt(xattn or [1], dense or [1], **over)
    keep = [row for row, xs in zip(r["rows"], (xattn, dense)) if xs is not None]
    return dict(r, rows=keep)


def session(tmp_path, blocks, remote=REMOTE, cp=None):
    return P2T.PairingSession(remote=remote, cp=cp or loaded_p2(tmp_path),
                              blocks=[(b, [("gt_L4_b1.json", r)]) for b, r in blocks],
                              hardware=TP1.HW)


def test_one_row_per_cell_the_anchor_pooled_over_the_passed_blocks(tmp_path):
    s = session(tmp_path, [("b1", block([10, 20, 30], [5, 6, 7])),
                           ("b2", block(None, [8, 9]))])
    out = P2T.training_rows(s)
    rows = {r.cell.arm: r for r in out["rows"]}
    assert len(out["rows"]) == 2 and {r.cell.pairing for r in out["rows"]} == {"P2"}
    assert rows["xattn"].measured == pytest.approx(0.020)
    assert rows["dense"].measured == pytest.approx(0.007)    # 5 6 7 8 9 pooled
    meta = {m["cell"][0]: m for m in out["metadata"]}
    assert [(b["block"], b["n"]) for b in meta["dense"]["blocks"]] == [("b1", 3), ("b2", 2)]
    assert meta["dense"]["blocks"][1]["median_s"] == pytest.approx(0.0085)
    assert out["blocks"] == ["b1", "b2"] and out["left_out"] == []
    ca = next(c for c in s.cp.cells if c.arm == "xattn")
    assert rows["xattn"].cell.priced == ca.priced and rows["xattn"].cell.hardware == TP1.HW


def test_a_cell_timed_twice_in_one_block_is_refused(tmp_path):
    s = session(tmp_path, [("b1", block([10], [5]))])
    s = P2T.PairingSession(remote=s.remote, cp=s.cp, hardware=s.hardware,
                           blocks=[("b1", [("gt_a.json", block([10], [5])),
                                           ("gt_b.json", block(None, [6]))])])
    with pytest.raises(P2T.TrainingRowsError, match="timed twice"):
        P2T.training_rows(s)


def test_a_short_groups_cells_are_left_out_and_named_never_filled(tmp_path):
    s = session(tmp_path, [("b1", block([10, 20], [5], clean_rounds=3)),
                           ("b2", block(None, [8, 9]))])
    out = P2T.training_rows(s)
    assert [r.cell.arm for r in out["rows"]] == ["dense"]
    assert out["rows"][0].measured == pytest.approx(0.0085)   # b2's alone
    assert [m["cell"][0] for m in out["left_out"]] == ["xattn@0.5"]
    assert "b1 gt_L4_b1.json: 3 clean" in out["left_out"][0]["why"][0]
    dense = out["metadata"][0]
    assert dense["left_out"] and [b["block"] for b in dense["blocks"]] == ["b2"]
    with pytest.raises(P2T.TrainingRowsError, match="every cell is short"):
        P2T.training_rows(session(tmp_path, [("b1", block([10], [5], clean_rounds=3))]))


@pytest.mark.parametrize("remote", ["gs://bucket/predictions/P2-R/t61_rehearsal/p.json",
                                    "gs://bucket/predictions/P1-T3.2d/s/p.json"])
def test_only_p2s_own_prediction_trains_p4(tmp_path, remote):
    with pytest.raises(P2T.TrainingRowsError, match="not under predictions/P2/"):
        P2T.training_rows(session(tmp_path, [("b1", block([10], [5]))], remote=remote))


def test_a_p1_prediction_and_no_passed_block_are_refused(tmp_path):
    p1 = FX.loaded_p1(tmp_path)[0]
    with pytest.raises(P2T.TrainingRowsError, match="P1's, not P2's"):
        P2T.training_rows(session(tmp_path, [("b1", block([10], [5]))], cp=p1))
    with pytest.raises(P2T.TrainingRowsError, match="no passed block"):
        P2T.training_rows(session(tmp_path, []))


def test_the_file_round_trips_and_refuses_damage_and_p1s_schema(tmp_path):
    s = session(tmp_path, [("b1", block([10, 20, 30], [5, 6, 7]))])
    out = P2T.training_rows(s)
    path = tmp_path / "p2_training.json.gz"
    P2T.dump(P2T.to_record(out, TP1.HW), path)
    back = P2T.rows_from_record(P2T.load(path))
    assert [(r.cell, r.measured) for r in back] == [(r.cell, r.measured) for r in out["rows"]]
    raw = gzip.decompress(path.read_bytes()).replace(b'"target_s":0.02', b'"target_s":0.03', 1)
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(P2T.TrainingRowsError, match="own sha256"):
        P2T.rows_from_record(P2T.load(path))
    with pytest.raises(P2T.TrainingRowsError, match="not a P2 training file"):
        P2T.rows_from_record(PT.to_record(PT.training_rows(
            TP1.session(tmp_path, "T3.2a", TP1.gt([10], [5])),
            TP1.session(tmp_path, "T3.2b", TP1.gt([20], [6]))), TP1.HW))


# --- P4 trains on P1's rows and P2's (the runner, `t61_session.training_rows`) ----------

def _files(tmp_path, monkeypatch):
    import t64_session as P4
    from test_t61_session import _lock_training, _p1_training_file
    p1 = _lock_training(tmp_path, monkeypatch, _p1_training_file(tmp_path / "p1"))
    out = P2T.training_rows(session(tmp_path, [("b1", block([12, 22], [4, 6]))]))
    p2 = tmp_path / "p2_training.json.gz"
    P2T.dump(P2T.to_record(out, TP1.HW), p2)
    monkeypatch.setattr(S, "P1_TRAINING", p1)
    monkeypatch.setattr(S, "P2_TRAINING", p2)
    return P4.T64, p1, p2


def test_p4s_rows_are_p1s_and_p2s_and_its_fits_use_both(tmp_path, monkeypatch):
    (tmp_path / "p1").mkdir()
    T64, p1, p2 = _files(tmp_path, monkeypatch)
    rows = S.training_rows(T64)
    assert sorted(r.cell.pairing for r in rows) == ["P1", "P1", "P2", "P2"]
    assert [r.cell.pairing for r in S.training_rows(S.T61)] == ["P1", "P1"]
    ts = committed_training_set(rows, "P4", T64.banked)
    assert ts.trained_on == ("P1", "P2") and len(ts.rows) == 4
    assert BL.fit_fitted_sum(ts)
    assert set(S.training_files(T64)) == {"P1", "P2"} and set(S.training_files(S.T61)) == {"P1"}
    p2.unlink()
    with pytest.raises(S.StepFailed, match="no P2 training file"):
        S.training_rows(T64)
    with pytest.raises(S.StepFailed, match="no P2 training file"):
        S.training_files(T64)


def test_p4s_staged_check_reads_p2s_file_against_the_pairing_ledger(tmp_path, monkeypatch):
    import t64_session as P4
    from argparse import Namespace
    from test_t61_session import bank, _sha
    (tmp_path / "p1").mkdir()
    T64, p1, p2 = _files(tmp_path, monkeypatch)
    real = Namespace(session=str(tmp_path), dry_run=False)
    bank(monkeypatch, tmp_path / "wrong", P1=("2026-11-19T00:00:00+00:00", _sha(p1)),
         P2=("2026-11-25T00:00:00+00:00", "d" * 64))
    found, bad = P4.p4_staged_inputs(T64, real, ("predict",))
    assert found["p2_training_rows"] == 2
    assert any("pairing ledger's P2 entry names " + "d" * 64 in b for b in bad), bad
    bank(monkeypatch, tmp_path / "right", P1=("2026-11-19T00:00:00+00:00", _sha(p1)),
         P2=("2026-11-25T00:00:00+00:00", _sha(p2)))
    found, bad = P4.p4_staged_inputs(T64, real, ("predict",))
    assert not [b for b in bad if "P2" in b], bad
    assert found["p2_training_sha256"] == _sha(p2)
    raw = gzip.decompress(p2.read_bytes()).replace(b'"target_s":0.0', b'"target_s":0.1', 1)
    p2.write_bytes(gzip.compress(raw))
    _, bad = P4.p4_staged_inputs(T64, real, ("predict",))
    assert any("P2 training file does not match its own sha256" in b for b in bad), bad


def test_p4s_predict_trains_on_synthetic_p2_rows_on_a_cpu(tmp_path, monkeypatch):
    """Ruling 1h's CPU pass: P4's real `real_record` on P1's rows and synthetic
    P2 rows, through the pairing ledger's banking times and files, both
    fitted sums and the learned fit (LightGBM, the locked configuration);
    only the components assembly (`build_prediction`, GPU recordings) is
    a stand-in, which records what it is handed."""
    from argparse import Namespace
    from datetime import datetime, timezone
    import t31_steps as st
    from attnbench.predict import learned as LM
    from test_t61_session import bank, _sha
    (tmp_path / "p1").mkdir()
    T64, p1, p2 = _files(tmp_path, monkeypatch)
    bank(monkeypatch, tmp_path, P1=("2026-11-19T00:00:00+00:00", _sha(p1)),
         P2=("2026-11-25T00:00:00+00:00", _sha(p2)))
    monkeypatch.setattr(LM, "LOCKED_CONFIG", LM.LearnedConfig(target="log_seconds"))
    got = {}
    monkeypatch.setattr(st, "load_config", lambda path, dry_run: {})
    monkeypatch.setattr(st, "build_prediction", lambda comp, session, **kw: got.update(kw) or {})
    args = Namespace(session=str(tmp_path), out=str(tmp_path), dry_run=False, banked_at=[])
    S.real_record(T64, args, datetime(2026, 12, 1, tzinfo=timezone.utc))
    b = got["baselines"]
    assert got["pairing"] == "P4" and b["banked"] == ["P1", "P2"]
    assert b["training_files"] == {"P1": _sha(p1), "P2": _sha(p2)}
    assert b["fitted_sum"] and b["fitted_sum_plus_s0"] and b["learned"] is not None
    # the ledger's P2 file is the one read, or predict refuses
    bank(monkeypatch, tmp_path / "other", P1=("2026-11-19T00:00:00+00:00", _sha(p1)),
         P2=("2026-11-25T00:00:00+00:00", "d" * 64))
    with pytest.raises(S.StepFailed, match="pairing ledger names training files"):
        S.real_record(T64, args, datetime(2026, 12, 1, tzinfo=timezone.utc))


def _block_dir(tmp_path, bid, result, *, passed=True):
    import json
    gt_dir = S.step_dir(Path(), "ground_truth", S.T61).name
    d = tmp_path / f"t61_p2_{bid}"
    (d / gt_dir / "g1").mkdir(parents=True)
    steps = ["predict", "ground_truth", "late_check"]
    (d / "manifest.json").write_text(json.dumps({
        "block": {"id": bid, "steps": steps, "spec_sha256": "s" * 64}, "code_commit": "c" * 40,
        "attempts": [{"step": s, "status": "passed" if passed or s != "late_check" else "failed"}
                     for s in steps]}))
    (d / gt_dir / "g1" / "stamps.json").write_text(json.dumps(
        {"clock_not_before": FX.STARTED_AT.timestamp(), "first_clocked_block_started_at": []}))
    (d / gt_dir / "g1" / "gt_L4_b1.json").write_text(json.dumps(result))
    return d


def test_the_builder_reads_only_passed_blocks_against_the_ledgers_p1(tmp_path, monkeypatch,
                                                                     capsys):
    import p2_training as B
    from attnbench.predict import commit as C
    from test_t61_session import _lock_training, _p1_training_file, _sha
    from attnbench.predict.ledger import Ledger
    (tmp_path / "p1").mkdir()
    p1 = _lock_training(tmp_path, monkeypatch, _p1_training_file(tmp_path / "p1"))
    monkeypatch.setattr(S, "P1_TRAINING", p1)
    cp = loaded_p2(tmp_path)
    seen = {}
    monkeypatch.setattr(C, "load_committed", lambda store, remote, **kw: seen.update(kw) or cp)
    local = tmp_path / "local.json"
    local.write_text('{"cells": [{"arm": "xattn", "context_length": 4, "batch": 1}]}')
    link = tmp_path / "link.json"
    link.write_text('{"hardware": {"pinned_bandwidth": 2e10, "pageable_bandwidth": 1e10}}')
    b1 = _block_dir(tmp_path, "b1", block([10, 20, 30], [5, 6, 7]))
    b2 = _block_dir(tmp_path, "b2", block(None, [100, 200]), passed=False)
    store = S.DirStore(tmp_path / "store", lambda: 1.8e9)
    out = tmp_path / "p2.json.gz"
    argv = ["--remote", "predictions/P2/t61_p2_b1/p.json", "--local", str(local), "--link",
            str(link), "--block", str(b1), "--block", str(b2), "--out", str(out)]
    assert B.main(argv, store=store, prefix=S.DirStore.PREFIX) == 2
    assert "P1 has no banked entry" in capsys.readouterr().err
    Ledger(store, S.DirStore.PREFIX, "P1").record_banked({"training_file_sha256": "e" * 64})
    assert B.main(argv, store=store, prefix=S.DirStore.PREFIX) == 2
    assert "the ledger banked P1's file " + "e" * 64 in capsys.readouterr().err
    store2 = S.DirStore(tmp_path / "store2", lambda: 1.8e9)
    Ledger(store2, S.DirStore.PREFIX, "P1").record_banked({"training_file_sha256": _sha(p1)})
    assert B.main(argv, store=store2, prefix=S.DirStore.PREFIX) == 0
    rec = P2T.load(out)
    assert rec["blocks"] == ["b1"] and [u["block"] for u in rec["unpassed"]] == ["b2"]
    dense = next(r for r in rec["rows"] if r["cell"]["arm"] == "dense")
    assert dense["target_s"] == pytest.approx(0.006)                     # b2's 100, 200 unread
    assert seen["banked_at"] == {"P1": datetime_of(1.8e9)}
    assert seen["training_files"] == {"P1": _sha(p1)} and len(seen["training_rows"]) == 2


def datetime_of(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, tz=timezone.utc)
