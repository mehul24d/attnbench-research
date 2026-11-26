"""scripts/p6_prediction_from_p4.py and the committed P6 prediction (the
researcher's task of 26 Nov 2026 after P3's push, item 2a; EXPLORATORY).

The generator on a synthetic P4 record (always runs), the committed file
against its sidecar and docs/p6_cells.json (always runs once it is
committed), and the committed file against P4's banked record where this
machine has it (skips elsewhere)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p6_prediction_from_p4 as G  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402

COMMITTED_DIR = ROOT / "docs" / "p6_prediction"
P4_BANKED = (Path.home() / "Desktop/research/attnbench_backups/scoring_20261116/bucket/"
             "predictions/P4/t64_p4_20261116T113724Z_b3/prediction_P4_20261116T120732377724Z.json")
T = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)
CELLS = [("dense", 1.0, 2048, 1), ("xattention", None, 16384, 1)]


def _p4(cells=None):
    cells = cells if cells is not None else [
        {"arm": "dense", "model": "primary", "batch": 1, "density": 1.0,
         "context_length": 1951, "code_path": "NVIDIA H100 80GB HBM3|eager|O0|x",
         "predictions": {"recursion": 0.037, "learned": 0.065}},
        {"arm": "xattention", "model": "primary", "batch": 1, "density": 0.572401,
         "context_length": 16370, "code_path": "NVIDIA H100 80GB HBM3|xattn:triton|O0|y",
         "predictions": {"recursion": 0.5, "learned": 0.6}}]
    return {"pairing": "P4", "statistic": "median", "written_at": "2026-11-16T12:07:32+00:00",
            "code": {"git_commit": "b" * 40, "git_dirty": False}, "cells": cells}


def _banked(tmp_path, record):
    p = tmp_path / "prediction_P4_x.json"
    data = json.dumps(record).encode()
    p.write_bytes(data)
    p.with_name(p.name + ".sha256").write_text(C.sidecar_text(C.sha256_hex(data), p.name))
    return p


def _committed():
    files = sorted(COMMITTED_DIR.glob("prediction_P6_exploratory_from_P4_*.json"))
    assert len(files) == 1, files
    return files[0]


def test_each_cell_carries_p4s_values_unchanged(tmp_path):
    p4 = _p4()
    rec, sha, _ = G.G.read_checked(_banked(tmp_path, p4))
    out = G.transfer(rec, source_name="x", source_sha=sha, written_at=T, code={}, cells=CELLS)
    assert [c["predictions_s"] for c in out["cells"]] == [e["predictions"] for e in p4["cells"]]
    assert [c["budget"] for c in out["cells"]] == [2048, 16384]
    assert out["pairing"] == "P6" and out["label"] == "EXPLORATORY"
    assert out["source"]["pairing"] == "P4" and out["source"]["sha256"] == sha


def test_a_source_that_is_not_p4_is_refused():
    with pytest.raises(G.RefusedError, match="not P4"):
        G.transfer({**_p4(), "pairing": "P1"}, source_name="x", source_sha="0", written_at=T,
                   code={}, cells=CELLS)


def test_cells_other_than_the_p6_cells_file_are_refused():
    with pytest.raises(G.RefusedError, match="not docs/p6_cells.json's"):
        G.transfer(_p4(), source_name="x", source_sha="0", written_at=T, code={},
                   cells=[("dense", 1.0, 2048, 1), ("xattention", None, 8192, 1)])
    with pytest.raises(G.RefusedError, match="not docs/p6_cells.json's"):
        G.transfer(_p4(), source_name="x", source_sha="0", written_at=T, code={},
                   cells=CELLS + [("meanpool", 0.25, 2048, 1)])


def test_a_p4_record_that_does_not_match_its_sidecar_is_refused(tmp_path):
    p = _banked(tmp_path, _p4())
    p.write_bytes(p.read_bytes().replace(b"0.037", b"0.036"))
    with pytest.raises(G.RefusedError, match="not its sidecar"):
        G.G.read_checked(p)


def test_a_committed_file_is_never_overwritten(tmp_path):
    rec = G.transfer(_p4(), source_name="x", source_sha="0", written_at=T, code={}, cells=CELLS)
    G.write(rec, tmp_path)
    with pytest.raises(G.RefusedError, match="never overwritten"):
        G.write(rec, tmp_path)


def test_the_p6_cells_file_has_35_cells():
    assert len(G.cell_keys()) == 35


@pytest.mark.skipif(not COMMITTED_DIR.exists(), reason="written after this test's commit")
def test_the_committed_prediction_matches_its_sidecar_and_is_exploratory():
    f = _committed()
    side = f.with_name(f.name + ".sha256").read_text()
    assert C.parse_sidecar(side, f.name) == C.sha256_hex(f.read_bytes())
    rec = json.loads(f.read_bytes())
    assert rec["label"] == "EXPLORATORY" and rec["pairing"] == "P6" and len(rec["cells"]) == 35
    assert rec["schema"] == G.SCHEMA and rec["code"]["git_dirty"] is False


@pytest.mark.skipif(not (COMMITTED_DIR.exists() and P4_BANKED.exists()),
                    reason="P4's banked prediction is on the Mac only")
def test_the_committed_prediction_is_p4s_banked_record_transferred():
    rec = json.loads(_committed().read_bytes())
    p4, sha, _ = G.G.read_checked(P4_BANKED)
    again = G.transfer(p4, source_name=P4_BANKED.name, source_sha=sha,
                       written_at=datetime.fromisoformat(rec["written_at"]), code=rec["code"],
                       cells=G.cell_keys())
    assert again == rec
