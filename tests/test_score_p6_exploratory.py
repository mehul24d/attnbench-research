"""scripts/score_p6_exploratory.py: P6's committed prediction against P6's ground
truth, EXPLORATORY (the researcher's task of 26 Nov 2026 after P3's push, item
2b). The checks before anything is measured, P4's committed values loaded from
the 16 Nov 2026 mirror, and the split of the dense error by matched and
mismatched backend. No P6 session exists yet, so nothing here measures one."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_p6_exploratory as S  # noqa: E402

B = Path.home() / "Desktop/research/attnbench_backups"
INPUTS = B / "scoring_20261116/run/inputs.json"
MIRROR = B / "scoring_20261116/mirror"


def test_the_committed_file_is_the_one_checked(tmp_path):
    rec = S.check_prediction(S.PREDICTION)
    assert rec["pairing"] == "P6" and rec["source"]["pairing"] == "P4" and len(rec["cells"]) == 35
    bad = tmp_path / S.PREDICTION.name
    bad.write_bytes(S.PREDICTION.read_bytes().replace(b'"EXPLORATORY"', b'"EXPLORATORY "'))
    (tmp_path / (bad.name + ".sha256")).write_text(Path(str(S.PREDICTION) + ".sha256").read_text())
    with pytest.raises(S.P6ScoreError, match="sha256"):
        S.check_prediction(bad)


def _r(L, b, p, rel, cls="dense"):
    return SimpleNamespace(arm="dense", density=1.0, context_length=L, batch=b, predictor=p,
                           arm_class=cls, predicted=1.0, measured=1.0 / (1 + rel), relative=rel,
                           abs_relative=abs(rel), inside_floor=False)


def test_the_dense_error_is_split_by_matched_and_mismatched_backend():
    res = [_r(1951, 1, "recursion", 0.10), _r(1951, 1, "learned", 0.30),
           _r(4068, 1, "recursion", 0.20), _r(4068, 1, "learned", 0.40),
           _r(8075, 1, "recursion", 0.50), _r(8075, 1, "learned", 0.70),
           _r(1951, 1, "recursion", 9.0, cls="synchronising")]          # not dense: left out
    chosen = {(2048, 1): "sdpa_flash", (4096, 1): "fa3", (8192, 1): "fa3"}
    d = S.backend_split(res, chosen, ["learned", "recursion"])
    assert d["split"]["matched"] == {"cells": 1, "median_abs_relative": {"learned": 0.30, "recursion": 0.10}}
    assert d["split"]["mismatched"]["cells"] == 2
    assert d["split"]["mismatched"]["median_abs_relative"] == pytest.approx({"learned": 0.55, "recursion": 0.35})
    assert [(r["budget"], r["chosen_kernel"], r["group"]) for r in d["rows"]] == [
        (2048, "sdpa_flash", "matched"), (4096, "fa3", "mismatched"), (8192, "fa3", "mismatched")]


def test_no_matched_cell_is_an_empty_group_not_an_error():
    d = S.backend_split([_r(1951, 1, "recursion", 0.1)], {(2048, 1): "fa3"}, ["recursion"])
    assert d["split"]["matched"] == {"cells": 0, "median_abs_relative": {"recursion": None}}


def test_a_session_with_a_step_not_passed_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"attempts": [{"step": "ground_truth", "status": "passed"}, {"step": "verify", "status": "failed"}]}))
    with pytest.raises(S.P6ScoreError, match="verify"):
        S.session_passed(tmp_path)


@pytest.mark.skipif(not (INPUTS.exists() and MIRROR.exists()),
                    reason="needs the 16 Nov 2026 scoring mirror on this Mac")
def test_the_committed_file_is_p4s_committed_values_loaded_by_the_scorer():
    spec = json.loads(INPUTS.read_text())
    store = S.SH.MirrorStore(MIRROR)
    prefix = spec["bucket"].rstrip("/") + "/"
    banked = {p: b for p in ("P1", "P2", "P4")
              if (b := S.Ledger(store, prefix, p).banked()) is not None}
    cp4, read = S.load_p4(store, prefix, spec, S.SH.lock_at(S.SH.TAG), banked)
    assert read["trained_on"] == ["P1", "P2"]
    rec = S.check_prediction(S.PREDICTION)
    assert S.same_as_p4(rec, cp4) == 35
    rec["cells"][3]["predictions_s"]["learned"] += 1e-9
    with pytest.raises(S.P6ScoreError, match="not P4's committed values"):
        S.same_as_p4(rec, cp4)
