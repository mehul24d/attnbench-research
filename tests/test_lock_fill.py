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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock_fixture as LF  # noqa: E402

P2_CONFIG = ROOT / "configs" / "t61" / "p2_session.json"
# `pairing_values` holds only what T4.1 does not list (ruling 1b, 11 Nov 2026).
P2_VALUES = {"gt_groups": {"eager": ["dense", "oracle"]},
             "reps": 12, "dense_rounds": 5, "sampler_interval_s": 0.1}


def lock(complete=True, values=P2_VALUES) -> dict:
    out = LF.complete_lock(L.PLAN.read_text(encoding="utf-8"))
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
    v = LF.VALUES
    assert cfg["densities"] == v["ground_truth_density_set"] and cfg["reps"] == 12
    assert cfg["gt_groups"] == P2_VALUES["gt_groups"] and cfg["max_rerun_rounds"] == 1
    assert cfg["flag_reasons"] == v["throttle_set"]
    assert cfg["lengths"] == v[st.GRID_ITEM]["lengths"]
    assert cfg["resolution_floor_ms"] == v["resolution_and_noise_floors"]["resolution_floor_ms"]
    assert cfg["chain_check_tolerance"] == {"aggregate": v["chain_check_tolerance"]}
    assert "working_values" not in cfg
    assert cfg["lock_values"]["source"] == "lock_values.json"
    assert set(cfg["lock_values"]["filled"]) >= {"densities", "reps", "dense_rounds",
                                                 "gt_groups.eager", "max_rerun_rounds"}


def test_each_filled_value_has_the_working_value_s_shape(tmp_path, monkeypatch):
    """The steps read a lock-filled value as they read P1's working one: a
    lock fraction the steps take as {"aggregate"} broke P2's pre-launch pass
    at host_components (12 Nov 2026)."""
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    cfg = st.load_config(str(P2_CONFIG), dry_run=False)
    work, _ = st.working_values()
    shared = set(st.LOCK_SOURCED) & set(work)
    assert "chain_check_tolerance" in shared
    for k in shared:
        assert type(cfg[k]) is type(work[k]), k
        if isinstance(work[k], dict):
            assert set(cfg[k]) == set(work[k]), k


def test_t41_items_come_from_the_lock_values_never_from_pairing_values(tmp_path, monkeypatch):
    """Ruling 1b (11 Nov 2026); T4.3 read B, finding 4."""
    shadow = dict(P2_VALUES, densities={"oracle": [0.5]}, max_rerun_rounds=0)
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock(values=shadow)))
    cfg = st.load_config(str(P2_CONFIG), dry_run=False)
    assert cfg["densities"] == LF.VALUES["ground_truth_density_set"]
    assert cfg["max_rerun_rounds"] == 1
    assert cfg["lock_values"]["ignored_pairing_values"] == ["densities", "max_rerun_rounds"]


@pytest.mark.parametrize("key, value", [("flag_reasons", ["hw_slowdown"]),
                                        ("lengths", [2048]), ("resolution_floor_ms", 5.0)])
def test_a_t41_item_the_repository_config_also_sets_is_refused(tmp_path, monkeypatch,
                                                                key, value):
    c = json.loads(P2_CONFIG.read_text())
    c[key] = value
    p = tmp_path / "p2.json"
    p.write_text(json.dumps(c))
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    with pytest.raises(st.StepFailed, match="T4.1 items"):
        st.load_config(str(p), dry_run=False)


def test_a_missing_rerun_budget_is_refused_not_zero():
    with pytest.raises(st.StepFailed, match="rerun budget"):
        st.rerun_budget({"pairing": "P2", "max_rerun_rounds": None})
    assert st.rerun_budget({"pairing": "P1", "max_rerun_rounds": None}) == 0


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


@pytest.mark.parametrize("group", st.DROPPED_GROUPS)
def test_a_compiled_group_is_refused_for_p2_and_p4(tmp_path, monkeypatch, group):
    # Dropped (adviser's ruling 4, third set of 12 Nov 2026): eager only.
    for cfg_path in (P2_CONFIG, ROOT / "configs" / "t64" / "p4_session.json"):
        assert set(json.loads(cfg_path.read_text())["gt_groups"]) == {"eager"}
    pv = dict(P2_VALUES, gt_groups={**P2_VALUES["gt_groups"], group: []})
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock(values=pv)))
    with pytest.raises(st.StepFailed, match="dropped from P2 and P4"):
        st.load_config(str(P2_CONFIG), dry_run=False)
    cfg = dict(json.loads(P2_CONFIG.read_text()), gt_groups={"eager": None, group: None})
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    with pytest.raises(st.StepFailed, match="dropped from P2 and P4"):
        st.fill_lock_values(cfg)


def test_held_out_pairings_flag_drift_relatively_and_take_the_locks_clean_minimum(tmp_path,
                                                                                  monkeypatch):
    # The adviser's rulings 1 and 2, fourth set of 12 Nov 2026.
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    cfg = st.load_config(str(P2_CONFIG), dry_run=False)
    assert cfg["min_clean_rounds"] == LF.VALUES["rerun_budget"]["min_clean_rounds"]
    assert st.drift_controls(cfg) == {"drift_limit_relative": st.RELATIVE_DRIFT_LIMIT,
                                      "drift_reference_min_rounds": 5,
                                      "min_clean_rounds": cfg["min_clean_rounds"]}
    assert st.DRIFT_REFERENCE_MIN_ROUNDS == 5   # declared in docs/drift_reference_correction.md
    assert st.RELATIVE_DRIFT_LIMIT == pytest.approx(0.0631, abs=5e-5)
    with pytest.raises(st.StepFailed, match="min_clean_rounds is missing"):
        st.drift_controls(dict(cfg, min_clean_rounds=None))
    assert st.drift_controls({"pairing": "P1"}) == {"drift_limit_relative": None,
                                                    "drift_reference_min_rounds": None,
                                                    "min_clean_rounds": None}


def test_the_ground_truth_step_hands_a_held_out_pairing_the_relative_controls(tmp_path,
                                                                              monkeypatch):
    # The GPU branch of `ground-truth-one`, its GPU pieces faked at the module boundary.
    import contextlib
    from types import SimpleNamespace
    from attnbench import ground_truth as gt
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, lock()))
    cfg = st.load_config(str(P2_CONFIG), dry_run=False)
    seen = {}

    class Controls:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(gt, "Controls", Controls)
    monkeypatch.setattr(st, "load_config", lambda path, dry_run: cfg)
    monkeypatch.setattr(st, "_skew", lambda c, d: (0, 0))
    monkeypatch.setattr(st, "_gap", lambda c, d: 0)
    monkeypatch.setattr(st.so, "check_session_skew", lambda *a, **k: None)
    monkeypatch.setattr(st.so, "ready_to_clock", lambda *a, **k: None)
    monkeypatch.setattr(st, "_receipt", lambda s: (SimpleNamespace(synced_at=st.so.utc_now()), None))
    monkeypatch.setattr(st, "_need_gpu", lambda what: None)
    monkeypatch.setattr(st, "_harness", lambda args: None)
    monkeypatch.setattr(st, "session_floor_ms", lambda *a: 1.0)
    monkeypatch.setattr(st, "_sampler", lambda c, d: contextlib.nullcontext("sampler"))
    monkeypatch.setattr(st, "_real_cells", lambda *a: [])
    args = SimpleNamespace(config=None, dry_run=False, out=str(tmp_path / "g"),
                           session=str(tmp_path / "s"), group="eager")
    assert st.cmd_ground_truth_one(args) == 0
    assert seen["drift_limit_relative"] == st.RELATIVE_DRIFT_LIMIT
    assert seen["drift_reference_min_rounds"] == st.DRIFT_REFERENCE_MIN_ROUNDS
    assert seen["min_clean_rounds"] == cfg["min_clean_rounds"]
    assert seen["max_rerun_rounds"] == cfg["max_rerun_rounds"]


def test_both_pairings_run_on_a_lock_whose_unscored_values_do_not_apply(tmp_path, monkeypatch):
    # The lock-wave rulings of 12 Nov 2026, items 1 and 2: no staged lock; rows 2
    # to 6 are typed "not applicable", H4 and H5 "not scored".
    rec = lock()
    why = {"reason": "not scored in this study", "appendix_a": "Appendix A, 12 Nov 2026"}
    for h in ("H4", "H5"):
        rec["scored_pairings"][h] = {"not_scored": dict(why)}
    rec["values"]["power_target_and_n"].update(value=None, not_applicable={
        **why, "limitation": "H4 is not tested", "hypotheses": ["H4", "H5"]})
    rec["pairing_values"]["P4"] = P2_VALUES
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, rec))
    assert st.load_config(str(P2_CONFIG), dry_run=False)["reps"] == 12
    assert st.load_config(str(ROOT / "configs" / "t64" / "p4_session.json"),
                          dry_run=False)["reps"] == 12
    rec["values"]["rerun_budget"].update(value=None, not_applicable={
        **why, "limitation": "none"})
    monkeypatch.setattr(st, "LOCK_FILE", write(tmp_path, rec))
    with pytest.raises(st.StepFailed, match="not applicable is refused"):
        st.load_config(str(P2_CONFIG), dry_run=False)
