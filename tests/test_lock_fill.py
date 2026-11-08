"""A later pairing's session values come from the lock file (T4.1, Gate B),
never from P1's working values, and never from a lock that is not complete
(`t31_steps.fill_lock_values`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t31_steps as st  # noqa: E402
from attnbench.lock import checklist as L  # noqa: E402

P2_CONFIG = ROOT / "configs" / "t61" / "p2_session.json"
P2_VALUES = {"densities": {"dense": [1.0], "oracle": [0.25], "meanpool": [0.25],
                           "xattention": ["calibrated"], "xattention_noassert": ["calibrated"]},
             "gt_groups": {"eager": ["dense", "oracle"], "compiled_default": [],
                           "compiled_capture": []},
             "reps": 12, "dense_rounds": 5, "sampler_interval_s": 0.1,
             "max_rerun_rounds": 1, "chain_check_tolerance": {"aggregate": 0.1},
             "residual_spread_tolerance": 0.1, "pricing_mismatch_share": 0.05}


def lock(complete=True, values=P2_VALUES) -> dict:
    out = L.skeleton(L.PLAN.read_text(encoding="utf-8"))
    for k, e in out["values"].items():
        e["value"], e["source"] = f"value of {k}", "Appendix A, test"
    out["scored_pairings"] = {"H0": ["P1", "P2"], "H2": ["P2", "P4"]}
    if not complete:
        next(iter(out["values"].values()))["value"] = None
    out["pairing_values"] = {"P2": values}
    return out


def write(tmp_path, rec) -> Path:
    p = tmp_path / "lock_values.json"
    p.write_text(json.dumps(rec))
    return p


def test_without_a_lock_file_p2_is_refused_p1s_working_values(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "LOCK_FILE", tmp_path / "absent.json")
    with pytest.raises(st.StepFailed, match="the working values are P1's only"):
        st.load_config(str(P2_CONFIG), dry_run=False)


def test_a_complete_lock_fills_p2s_values_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    cfg = st.load_config(str(P2_CONFIG), dry_run=False)
    assert cfg["densities"] == P2_VALUES["densities"] and cfg["reps"] == 12
    assert cfg["gt_groups"] == P2_VALUES["gt_groups"] and cfg["max_rerun_rounds"] == 1
    assert "working_values" not in cfg
    assert cfg["lock_values"]["source"] == "lock_values.json"
    assert set(cfg["lock_values"]["filled"]) >= {"densities", "reps", "dense_rounds",
                                                 "gt_groups.eager", "max_rerun_rounds"}


def test_an_incomplete_lock_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock(complete=False)))
    with pytest.raises(st.StepFailed, match="not a complete lock"):
        st.load_config(str(P2_CONFIG), dry_run=False)


def test_a_value_the_lock_leaves_out_is_still_refused(tmp_path, monkeypatch):
    partial = {k: v for k, v in P2_VALUES.items() if k != "reps"}
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock(values=partial)))
    with pytest.raises(st.StepFailed, match="the working values are P1's only"):
        st.load_config(str(P2_CONFIG), dry_run=False)


def test_p1_never_reads_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock(complete=False)))
    cfg = st.load_config(str(ROOT / "configs" / "t31" / "p1_session.json"), dry_run=False)
    assert cfg["working_values"]["status"] == st.WORKING_STATUS and "lock_values" not in cfg


def test_the_lock_fills_only_what_the_pairings_config_leaves_null(tmp_path, monkeypatch):
    c = json.loads(P2_CONFIG.read_text())
    c["reps"] = 20
    p = tmp_path / "p2.json"
    p.write_text(json.dumps(c))
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    cfg = st.load_config(str(p), dry_run=False)
    assert cfg["reps"] == 20 and "reps" not in cfg["lock_values"]["filled"]
