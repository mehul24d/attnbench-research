"""P4's training rows for the later pairings (`attnbench/predict/p4_training.py`;
the adviser's rulings dated 16 Nov 2026, scoring wave, item 4): P2's rules
applied to P4, held to P2's builder on the same blocks, only P4's own
prediction, and the builder script read against the ledger's P1 and P2."""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from attnbench.predict import p2_training as P2T
from attnbench.predict import p4_training as P4T
from attnbench.predict.commit import load_committed

import _figure_fixtures as FX
import test_p1_training as TP1
from test_p2_training import block, loaded_p2, session as p2_session
from test_predict_commit import (CLEAN, CODE_PATHS, T0, MemoryStore, Clock, OPS_HOST, cell,
                                 build_record)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t61_session as S  # noqa: E402
import t64_session as T  # noqa: E402

REMOTE = "gs://bucket/predictions/P4/t64_p4_b3/prediction_P4_x.json"
BANKED = {"P1": "2026-11-09T11:12:31+00:00", "P2": "2026-11-14T09:40:00+00:00"}


def loaded_p4(tmp_path):
    store = MemoryStore(Clock(T0))
    record = build_record(pairing="P4", cells=[cell(pairing="P4"), cell(
        pairing="P4", ops=OPS_HOST, arm="dense", density=1.0)], queue_depth=8, banked=[],
        fitted_sum=None, learned=None, code=CLEAN, written_at=T0, code_paths=CODE_PATHS)
    d = tmp_path / f"p4_{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    f, receipt = FX.committed(d, record, store, FX.SYNC_AT)
    return load_committed(store, receipt.remote, ground_truth_started_at=FX.STARTED_AT,
                          clock_skew=FX.NO_SKEW, sync_gap=FX.NO_SKEW, banked_at={},
                          local=f.path)


def session(tmp_path, blocks, remote=REMOTE, cp=None):
    return P4T.PairingSession(remote=remote, cp=cp or loaded_p4(tmp_path),
                              blocks=[(b, [("gt_L4_b1.json", r)]) for b, r in blocks],
                              hardware=TP1.HW)


def _cell(r):
    c = r.cell
    return (c.model_name, c.code_path, c.arm, c.context_length, c.density, c.batch, c.priced,
            c.hardware)


BLOCKS = {
    "anchor pooled": [("b3", block([10, 20, 30], [5, 6, 7])), ("b4", block(None, [8, 9]))],
    "short group": [("b3", block([10, 20], [5], clean_rounds=3)), ("b4", block(None, [8, 9]))],
    "both cells twice": [("b3", block([10, 40], [5, 7])), ("b4", block([12, 14, 90], [6]))],
}


@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_p4s_rows_are_p2s_builder_on_the_same_blocks(tmp_path, name):
    got = P4T.training_rows(session(tmp_path, BLOCKS[name]))
    ref = P2T.training_rows(p2_session(tmp_path, BLOCKS[name]))
    assert {r.cell.pairing for r in got["rows"]} == {"P4"}
    assert [(_cell(r), r.measured) for r in got["rows"]] == \
        [(_cell(r), r.measured) for r in ref["rows"]]
    for k in ("metadata", "left_out", "blocks", "unpassed", "min_clean_rounds"):
        assert got[k] == ref[k], k


def test_a_cell_timed_twice_in_one_block_is_refused(tmp_path):
    s = session(tmp_path, [])
    s = P4T.PairingSession(remote=s.remote, cp=s.cp, hardware=s.hardware,
                           blocks=[("b3", [("gt_a.json", block([10], [5])),
                                           ("gt_b.json", block(None, [6]))])])
    with pytest.raises(P4T.TrainingRowsError, match="timed twice"):
        P4T.training_rows(s)


@pytest.mark.parametrize("remote", ["gs://bucket/predictions/P2/t61_p2/p.json",
                                    "gs://bucket/predictions/P4-R/t64_rehearsal/p.json"])
def test_only_p4s_own_prediction_trains_the_later_pairings(tmp_path, remote):
    with pytest.raises(P4T.TrainingRowsError, match="not under predictions/P4/"):
        P4T.training_rows(session(tmp_path, [("b3", block([10], [5]))], remote=remote))


def test_p2s_prediction_and_no_passed_block_are_refused(tmp_path):
    with pytest.raises(P4T.TrainingRowsError, match="P2's, not P4's"):
        P4T.training_rows(session(tmp_path, [("b3", block([10], [5]))], cp=loaded_p2(tmp_path)))
    with pytest.raises(P4T.TrainingRowsError, match="no passed block"):
        P4T.training_rows(session(tmp_path, []))


def test_the_file_round_trips_and_refuses_damage_and_p2s_schema(tmp_path):
    out = P4T.training_rows(session(tmp_path, [("b3", block([10, 20, 30], [5, 6, 7]))]))
    path = tmp_path / "p4_training.json.gz"
    P4T.dump(P4T.to_record(out, TP1.HW), path)
    back = P4T.rows_from_record(P4T.load(path))
    assert [(r.cell, r.measured) for r in back] == [(r.cell, r.measured) for r in out["rows"]]
    raw = gzip.decompress(path.read_bytes()).replace(b'"target_s":0.02', b'"target_s":0.03', 1)
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(P4T.TrainingRowsError, match="own sha256"):
        P4T.rows_from_record(P4T.load(path))
    p2 = P2T.to_record(P2T.training_rows(p2_session(tmp_path, [("b1", block([10], [5]))])),
                       TP1.HW)
    with pytest.raises(P4T.TrainingRowsError, match="not a P4 training file"):
        P4T.rows_from_record(p2)


# --- the builder script --------------------------------------------------------------

def _block_dir(tmp_path, bid, result, *, passed=True):
    gt_dir = S.step_dir(Path(), "ground_truth", T.T64).name
    d = tmp_path / f"t64_p4_{bid}"
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


def test_the_builder_reads_only_passed_blocks_against_the_ledgers_p1_and_p2(
        tmp_path, monkeypatch, capsys):
    import p4_training as B
    from attnbench.predict import commit as C
    from attnbench.predict.ledger import Ledger
    from test_t61_session import _lock_training, _p1_training_file, _sha
    assert S.step_dir(Path(), "ground_truth", T.T64).name == "13_ground_truth"
    (tmp_path / "p1").mkdir()
    p1 = _lock_training(tmp_path, monkeypatch, _p1_training_file(tmp_path / "p1"))
    p2 = tmp_path / "p2_training.json.gz"
    P2T.dump(P2T.to_record(P2T.training_rows(p2_session(tmp_path, [("b1", block([12], [4]))])),
                           TP1.HW), p2)
    cp = loaded_p4(tmp_path)
    seen = {}
    monkeypatch.setattr(C, "load_committed", lambda store, remote, **kw: seen.update(kw) or cp)
    local = tmp_path / "local.json"
    local.write_text('{"cells": [{"arm": "xattn", "context_length": 4, "batch": 1}]}')
    link = tmp_path / "link.json"
    link.write_text('{"hardware": {"pinned_bandwidth": 2e10, "pageable_bandwidth": 1e10}}')
    b3 = _block_dir(tmp_path, "b3", block([10, 20, 30], [5, 6, 7]))
    b4 = _block_dir(tmp_path, "b4", block(None, [100, 200]), passed=False)
    out = tmp_path / "p4.json.gz"
    argv = ["--remote", "predictions/P4/t64_p4_b3/p.json", "--local", str(local), "--link",
            str(link), "--p1-training", str(p1), "--p2-training", str(p2),
            "--block", str(b3), "--block", str(b4), "--out", str(out)]

    def store(name, **entries):
        st = S.DirStore(tmp_path / name, lambda: 1.8e9)
        for p, sha in entries.items():
            Ledger(st, S.DirStore.PREFIX, p).record_banked({"training_file_sha256": sha,
                                                           "banked_at": BANKED[p]})
        return st

    assert B.main(argv, store=store("s0", P1=_sha(p1)), prefix=S.DirStore.PREFIX) == 2
    assert "P2 has no banked entry" in capsys.readouterr().err
    assert B.main(argv, store=store("s1", P1=_sha(p1), P2="e" * 64),
                  prefix=S.DirStore.PREFIX) == 2
    assert "the ledger banked P2's file " + "e" * 64 in capsys.readouterr().err
    assert B.main(argv, store=store("s2", P1=_sha(p1), P2=_sha(p2)),
                  prefix=S.DirStore.PREFIX) == 0
    rec = P4T.load(out)
    assert rec["pairing"] == "P4" and rec["schema"] == P4T.SCHEMA
    assert rec["blocks"] == ["b3"] and [u["block"] for u in rec["unpassed"]] == ["b4"]
    dense = next(r for r in rec["rows"] if r["cell"]["arm"] == "dense")
    assert dense["target_s"] == pytest.approx(0.006)                     # b4's 100, 200 unread
    assert seen["banked_at"] == {p: datetime.fromisoformat(t) for p, t in BANKED.items()}
    assert seen["training_files"] == {"P1": _sha(p1), "P2": _sha(p2)}
    assert len(seen["training_rows"]) == 2 + 2                           # P1's two, P2's two
