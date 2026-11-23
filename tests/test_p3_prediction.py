"""scripts/p3_prediction_from_p1.py and the committed P3 prediction
(adviser's task of 23 Nov 2026, item 4c; EXPLORATORY).

The generator on a synthetic P1 record (always runs), the committed file
against its sidecar and P1's grid (always runs), and the committed file
against P1's banked record where this machine has it (skips elsewhere)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_prediction_from_p1 as G  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402

COMMITTED_DIR = ROOT / "docs" / "p3_prediction"
P1_BANKED = (Path.home() / "Desktop/research/attnbench_backups/scoring_20261116/bucket/"
             "predictions/P1-T3.2d/t32d_p1_20261109T0726Z/"
             "prediction_P1_20261109T095833754749Z.json")
T = datetime(2026, 11, 23, 3, 0, tzinfo=timezone.utc)


def _p1(cells=None):
    cells = cells if cells is not None else [
        {"arm": "dense", "model": "primary", "batch": 1, "density": 1.0,
         "context_length": 1951, "code_path": "NVIDIA L4|eager|O0|x",
         "predictions": {"sum": 0.19, "step_form": 0.123}},
        {"arm": "dense", "model": "primary", "batch": 4, "density": 1.0,
         "context_length": 16370, "code_path": "NVIDIA L4|eager|O0|x",
         "predictions": {"sum": 2.5, "step_form": 2.4}}]
    return {"pairing": "P1", "statistic": "median", "written_at": "2026-11-09T09:58:33+00:00",
            "code": {"git_commit": "a" * 40, "git_dirty": False}, "cells": cells}


def _banked(tmp_path, record):
    p = tmp_path / "prediction_P1_x.json"
    data = json.dumps(record).encode()
    p.write_bytes(data)
    p.with_name(p.name + ".sha256").write_text(C.sidecar_text(C.sha256_hex(data), p.name))
    return p


def _committed():
    files = sorted(COMMITTED_DIR.glob("prediction_P3_exploratory_from_P1_*.json"))
    assert len(files) == 1, files
    return files[0]


def test_each_cell_carries_p1s_values_unchanged(tmp_path):
    p1 = _p1()
    rec, sha, _ = G.read_checked(_banked(tmp_path, p1))
    out = G.transfer(rec, source_name="x", source_sha=sha, written_at=T, code={})
    assert [c["predictions_s"] for c in out["cells"]] == [e["predictions"] for e in p1["cells"]]
    assert [c["budget"] for c in out["cells"]] == [2048, 16384]
    assert out["pairing"] == "P3" and out["label"] == "EXPLORATORY"
    assert out["source"]["sha256"] == sha


def test_a_p1_record_that_does_not_match_its_sidecar_is_refused(tmp_path):
    p = _banked(tmp_path, _p1())
    p.write_bytes(p.read_bytes().replace(b"0.19", b"0.18"))
    with pytest.raises(G.RefusedError, match="not its sidecar"):
        G.read_checked(p)


def test_a_missing_sidecar_is_refused(tmp_path):
    p = _banked(tmp_path, _p1())
    p.with_name(p.name + ".sha256").unlink()
    with pytest.raises(G.RefusedError, match="missing"):
        G.read_checked(p)


def test_a_source_that_is_not_p1_is_refused():
    with pytest.raises(G.RefusedError, match="not P1"):
        G.transfer({**_p1(), "pairing": "P2"}, source_name="x", source_sha="0", written_at=T,
                   code={})


def test_a_committed_file_is_never_overwritten(tmp_path):
    rec = G.transfer(_p1(), source_name="x", source_sha="0", written_at=T, code={})
    G.write(rec, tmp_path)
    with pytest.raises(G.RefusedError, match="never overwritten"):
        G.write(rec, tmp_path)


def test_the_written_file_checks_against_its_sidecar(tmp_path):
    rec = G.transfer(_p1(), source_name="x", source_sha="0", written_at=T, code={})
    path, digest = G.write(rec, tmp_path)
    side = path.with_name(path.name + ".sha256").read_text()
    assert C.parse_sidecar(side, path.name) == digest == C.sha256_hex(path.read_bytes())


def test_the_committed_prediction_matches_its_sidecar_and_is_exploratory():
    f = _committed()
    side = f.with_name(f.name + ".sha256").read_text()
    assert C.parse_sidecar(side, f.name) == C.sha256_hex(f.read_bytes())
    rec = json.loads(f.read_bytes())
    assert rec["label"] == "EXPLORATORY" and rec["pairing"] == "P3"
    assert rec["schema"] == G.SCHEMA and rec["code"]["git_dirty"] is False


def test_the_committed_prediction_covers_p1s_grid():
    """P1's 35 cells (configs/t31/p1_session.json): five arms at five lengths,
    batch 4 for dense at 16384, the oracle at 0.5 not at 32768, two
    densities each for the oracle and mean-pool."""
    rec = json.loads(_committed().read_bytes())
    keys = {(c["arm"], c["budget"], c["batch"]) for c in rec["cells"]}
    assert len(rec["cells"]) == 35
    cfg = json.loads((ROOT / "configs/t31/p1_session.json").read_text())
    for arm in cfg["arms"]:
        for L in cfg["lengths"]:
            assert (arm, L, 1) in keys
    assert ("dense", 16384, 4) in keys
    assert not any(c["arm"] == "oracle" and c["density"] == 0.5 and c["budget"] == 32768
                   for c in rec["cells"])


@pytest.mark.skipif(not P1_BANKED.exists(), reason="P1's banked prediction is on the Mac only")
def test_the_committed_prediction_is_p1s_banked_record_transferred():
    rec = json.loads(_committed().read_bytes())
    p1, sha, _ = G.read_checked(P1_BANKED)
    again = G.transfer(p1, source_name=P1_BANKED.name, source_sha=sha,
                       written_at=datetime.fromisoformat(rec["written_at"]), code=rec["code"])
    assert again == rec
