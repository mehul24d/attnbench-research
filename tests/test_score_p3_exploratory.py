"""scripts/score_p3_exploratory.py: P3's 22 Nov 2026 prediction against P3's
ground truth, EXPLORATORY (26 Nov 2026). The checks before anything is
measured, the dense choices' keys, and the run on the banked data."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_p3_exploratory as S  # noqa: E402

B = Path.home() / "Desktop/research/attnbench_backups"
INPUTS = B / "scoring_20261116/run/inputs.json"
MIRROR = B / "scoring_20261116/mirror"
SESSION = B / "aws_p3_g6_20261126T0906Z/session"


def _cp(cells):
    return SimpleNamespace(cells=[SimpleNamespace(key=SimpleNamespace(arm=a, density=d, context_length=n, batch=b),
                                                  predictions=dict(p)) for a, d, n, b, p in cells])


def test_the_p3_file_must_be_p1s_committed_values_cell_for_cell():
    cells = [("dense", 1.0, 1951, 1, {"recursion": 0.12, "sum": 0.19}),
             ("meanpool", 0.25, 4068, 1, {"recursion": 0.3, "sum": 0.5})]
    rec = {"cells": [{"arm": a, "density": d, "context_length": n, "batch": b, "predictions_s": dict(p)}
                     for a, d, n, b, p in cells]}
    assert S.same_as_p1(rec, _cp(cells)) == 2
    rec["cells"][1]["predictions_s"]["recursion"] = 0.31
    with pytest.raises(S.P3ScoreError, match="is not P1's"):
        S.same_as_p1(rec, _cp(cells))
    with pytest.raises(S.P3ScoreError, match="P3 has 1 cells, P1 2"):
        S.same_as_p1({"cells": rec["cells"][:1]}, _cp(cells))


def test_the_dense_choices_are_keyed_by_budget_and_batch(tmp_path):
    d = tmp_path / "03_dense_select"
    d.mkdir()
    for n, b, k in ((2048, 1, "sdpa_flash"), (16384, 4, "fa2")):
        (d / f"dense_choice_L4_{n}_{b}.json").write_text(json.dumps({"chosen": k}))
    assert S.dense_choices(tmp_path) == {(2048, 1): "sdpa_flash", (16384, 4): "fa2"}


def test_a_session_with_a_step_not_passed_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"attempts": [{"step": "ground_truth", "status": "passed"}, {"step": "verify", "status": "failed"}]}))
    with pytest.raises(S.P3ScoreError, match="verify"):
        S.session_passed(tmp_path)


@pytest.mark.skipif(not (INPUTS.exists() and MIRROR.exists() and (SESSION / "manifest.json").exists()),
                    reason="needs the 16 Nov 2026 scoring mirror and P3's 26 Nov 2026 session on this Mac")
def test_the_run_on_the_banked_data(tmp_path):
    # its own process: the analysis-commit check reads the modules loaded, and pytest's are not
    import subprocess
    p = subprocess.run([sys.executable, str(ROOT / "scripts/score_p3_exploratory.py"), "--inputs", str(INPUTS),
                        "--mirror", str(MIRROR), "--session", str(SESSION), "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads((tmp_path / "p3_exploratory_scored.json").read_text())
    assert r["label"] == "EXPLORATORY" and r["scores_no_hypothesis"] is True
    assert r["prediction"]["cells_equal_to_p1"] == 35 and r["measured"]["cells"] == 35
    assert not r["measured"]["unmeasured"] and len(r["cells"]) == 35
    rows = r["dense_backend"]["rows"]
    assert len(rows) == 6 and all(x["chosen_kernel"] in ("sdpa_flash", "fa2", "sdpa_cudnn") for x in rows)
    split = r["dense_backend"]["split"]
    assert split["priced_kernel"]["n"] + split["other_kernel"]["n"] == 6
    committed = json.loads((ROOT / "docs/p3_run/p3_exploratory_scored.json").read_text())
    assert r["median_abs_relative"] == committed["median_abs_relative"]
