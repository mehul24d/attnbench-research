"""Reader B's reproductions (T4.3 read B, 23f540e; imported 11 Nov 2026 unchanged except paths). Each test asserts what the plan requires; a FAIL
on 23f540e is the reproduction of a defect."""

from __future__ import annotations

import copy
import json
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys

# Path edit on import (T4.3 wave 0): the reader's conftest put these on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t31_steps as st
import t61_session as S
from attnbench.analysis import block_drift as BD
from attnbench.lock import checklist as CL
from attnbench.predict import commit as C
from attnbench.predict import p1_training as PT
from test_lock_blocks import A, B2, B3, C1, C2
from test_t61_session import inproc, with_harness
import test_predict_commit as TPC

ROOT = Path(st.__file__).resolve().parent.parent
PLAN_TEXT = CL.PLAN.read_text(encoding="utf-8")


def complete_lock(value="x", **extra):
    # Edit on import (T4.3 fixes, ruling 1b): the lock's values are typed per
    # item and every hypothesis states its pairings, so "x" for every value
    # and H0/H2 alone no longer make a complete lock. The fixture takes typed
    # test values (tests/lock_fixture.py); no assertion changes.
    import lock_fixture as LF
    return LF.complete_lock(PLAN_TEXT, **extra)


# ---------------------------------------------------------------- F1: dense prefill before predict

def test_F1_lock_cannot_fix_the_floor_the_p2_ground_truth_uses():
    """P2's config leaves resolution_floor_ms null and the lock fill cannot set it."""
    p2 = json.loads((ROOT / "configs/t61/p2_session.json").read_text())
    assert p2["resolution_floor_ms"] is None
    assert "resolution_floor_ms" in st.LOCK_KEYS          # FAILS: not a lock key


def test_F1_dense_select_clocks_dense_end_to_end_prefill_before_predict(tmp_path, monkeypatch):
    """With the floor null (P2), step_dense_select runs derive_session_floor:
    the dense arm's whole prefill, timed with a synchronise, at every length and
    batch, before `predict` (validate() puts dense_select before predict)."""
    import torch
    from attnbench import dense_selection as ds
    runs = []

    class Arm:
        def __init__(self, L, b):
            self.L, self.b = L, b

        def run_prefill(self):
            runs.append((self.L, self.b))

    class H:
        def prepare(self, arm, L, batch=1):
            assert arm == "dense"
            return Arm(L, batch)

        def unwrap(self):
            pass

        def _cfg(self, *a, **k):
            return SimpleNamespace()

    monkeypatch.setattr(st, "_harness", lambda args: H())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(ds, "select_dense_baseline",
                        lambda *a, **k: SimpleNamespace(chosen="sdpa", complete=True))
    monkeypatch.setattr(ds, "write_choice", lambda c, out: Path(out) / "c.json")
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {})
    monkeypatch.setattr(st, "replace", lambda c, **k: c, raising=False)
    import dataclasses
    monkeypatch.setattr(dataclasses, "replace", lambda c, **k: c)
    cfg = json.loads((ROOT / "configs/t61/p2_session.json").read_text())
    cfg.update(reps=10, dense_rounds=3)
    from attnbench.lock.floors import FloorsError
    err = None
    try:
        st.step_dense_select(SimpleNamespace(dry_run=False, out=str(tmp_path)), cfg)
    except FloorsError as e:          # P2's floor is refused, but only after the clocking
        err = e
    assert S.T61.names.index("dense_select") < S.T61.names.index("predict")
    # The plan: nothing of the timed workload before the prediction is synced.
    assert runs == [], (f"{len(runs)} dense end-to-end prefills ({len(set(runs))} cells) "
                        f"clocked before predict; then: {err}")


# ---------------------------------------------------------------- F2: rows of a refused ground truth

def _p2_lock(tmp_path):
    lock = tmp_path / "p2.json"
    lock.write_text(json.dumps({"blocks": {"P2": {"max_hours": 2.2, "anchor": A, "blocks": [
        {"id": "b1", "steps": ["link", "launch_queue", "probe", "mask_gate", "host_components"]},
        {"id": "b2", "steps": B2, "cells": [C1]}, {"id": "b3", "steps": B3, "cells": [C2]}]}}}))
    return lock


def _run_block(cfg, tmp_path, lock, bid, imports=()):
    br, why = S.block_run(cfg, Namespace(lock=str(lock), block=bid,
                                         import_block=[str(d) for d in imports], dry_run=True))
    assert why is None, why
    d = tmp_path / bid
    rc = S.run_session(cfg, d, start="prereqs", extra=[], dry_run=True, sync=False,
                       runner=inproc(cfg), block=br)
    return rc, d


@pytest.mark.parametrize("fault", ["stamp_before_open", "incomplete"])
def test_F2_block_drift_pools_rows_the_runner_refused(tmp_path, fault):
    def bad(cfg, args, block=None):
        r = S.FakeHarness(Path(args.session))(cfg, args, block)
        if fault == "stamp_before_open":           # the clock stepped back, or a stale stamp
            r[S.STRICT_KEY] -= 3600
        else:
            r["complete"] = False
        return r
    T = S.T61
    _, d1 = _run_block(T, tmp_path, _p2_lock(tmp_path), "b1")
    Tbad = with_harness(T, bad)
    rc, d2 = _run_block(Tbad, tmp_path, _p2_lock(tmp_path), "b2", [d1])
    m = json.loads((d2 / "manifest.json").read_text())
    assert rc == 1 and S.last_status(m)["ground_truth"] == "failed"
    got = BD.collect([d1, d2], ground_truth_dir=S.step_dir(Path(), "ground_truth", T).name)
    assert got["rows"] == [], (f"{len(got['rows'])} rows of a ground truth the runner "
                               f"refused ({S.last_status(m)['ground_truth']}) are pooled")


# ---------------------------------------------------------------- F3: second session re-predicts

def test_F3_a_second_session_predicts_after_the_first_sessions_ground_truth(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    T = S.T61
    assert S.run_session(T, a, start="prereqs", extra=[], dry_run=True, sync=False,
                         runner=inproc(T)) == 0
    off = json.loads((a / S.DRY_CLOCK).read_text())["offset_s"]
    b.mkdir()
    (b / S.DRY_CLOCK).write_text(json.dumps({"offset_s": off + 600}))   # ten minutes later
    assert S.run_session(T, b, start="prereqs", extra=[], dry_run=True, sync=False,
                         runner=inproc(T)) == 0
    a_start = S.session_start(a)
    b_written = S.epoch(json.loads((b / S.RECEIPT).read_text())["written_at"])
    assert b_written > a_start + 500                 # P2's ground truth existed when b predicted
    mb = json.loads((b / "manifest.json").read_text())
    # What K9 asks: a prediction written after the pairing's ground truth began is refused.
    assert S.last_status(mb)["late_check"] != "passed", \
        "the second prediction passes its late check; nothing knows session a exists"


# ---------------------------------------------------------------- F4: lock checker and runner values

@pytest.mark.parametrize("bad", ["TBD", {"2048": None}, [None], False])
def test_F4_checker_accepts_wrongly_typed_or_nested_null_values(bad):
    lock = complete_lock()
    lock["values"]["mask_gate_bound"]["value"] = bad
    lock["values"]["resolution_and_noise_floors"]["value"] = bad
    assert not CL.check(lock, PLAN_TEXT).complete, f"{bad!r} reads as fixed"


def _p2_load(tmp_path, monkeypatch, pairing_values):
    lock = complete_lock(pairing_values={"P2": pairing_values})
    lock["values"]["ground_truth_density_set"]["value"] = {"oracle": [0.25]}
    lock["values"]["throttle_set"]["value"] = ["hw_slowdown"]
    # Edit on import: the rerun budget is typed (flagged rounds and minimum clean rounds).
    lock["values"]["rerun_budget"]["value"] = {"max_rerun_rounds": 2, "min_clean_rounds": 3}
    p = tmp_path / "lock_values.json"
    p.write_text(json.dumps(lock))
    monkeypatch.setattr(st, "LOCK_FILE", p)
    assert CL.check(lock, PLAN_TEXT).complete
    return lock, st.load_config(str(ROOT / "configs/t61/p2_session.json"), dry_run=False)


def test_F4_runner_reads_values_the_complete_lock_does_not_hold(tmp_path, monkeypatch):
    pv = {"densities": {"oracle": [0.25, 0.5]}, "gt_groups": {"eager": ["dense", "oracle"],
          "compiled_default": [], "compiled_capture": []}, "reps": 10, "dense_rounds": 3,
          "sampler_interval_s": 0.01, "chain_check_tolerance": 0.1,
          "residual_spread_tolerance": 0.1, "pricing_mismatch_share": 0.05}   # no max_rerun_rounds
    lock, cfg = _p2_load(tmp_path, monkeypatch, pv)
    problems = []
    if st.rerun_budget(cfg) != lock["values"]["rerun_budget"]["value"]["max_rerun_rounds"]:
        problems.append(f"rerun budget {st.rerun_budget(cfg)} (lock: 2)")
    if cfg["densities"] != lock["values"]["ground_truth_density_set"]["value"]:
        problems.append(f"densities {cfg['densities']} (lock item: {{'oracle': [0.25]}})")
    if cfg["flag_reasons"] != lock["values"]["throttle_set"]["value"]:
        problems.append(f"throttle set {cfg['flag_reasons']} from the repo config")
    assert not problems, "; ".join(problems)


# ---------------------------------------------------------------- F5: lock files the runner never checks

def test_F5_a_block_spec_is_taken_from_a_lock_file_that_is_not_a_lock(tmp_path):
    br, why = S.block_run(S.T61, Namespace(lock=str(_p2_lock(tmp_path)), block="b1",
                                           import_block=[], dry_run=False))
    assert why is not None, "a real block runs from a file holding nothing but a block spec"


def test_F5_staged_check_reads_a_key_the_draft_package_does_not_have():
    draft = json.loads((ROOT / "docs/lock_package_draft.json").read_text())
    assert "p1_training" in draft or "banked_files" not in draft, \
        "p2_staged_inputs reads lock['p1_training'], the draft holds banked_files.p1_training"


# ---------------------------------------------------------------- F6: trained baselines unverifiable

def test_F6_learned_predictions_and_config_are_never_checked(tmp_path, monkeypatch):
    from attnbench.predict import learned as LM
    monkeypatch.setattr(LM, "LOCKED_CONFIG", TPC.CONFIG)
    rec = copy.deepcopy(TPC.p2_record())
    rec["training"]["learned"]["config"] = {"target": "log_seconds", "params": {"num_leaves": 63}}
    for c in rec["cells"]:
        c["predictions"]["learned"] = 42.0            # any numbers, from any model
    store = TPC.MemoryStore(TPC.Clock(TPC.T0), no_clobber=True)
    f, _ = TPC.committed(tmp_path, rec, store, TPC.T0 + timedelta(seconds=1))
    with pytest.raises(C.PredictionCommitError):
        C.load_committed(store, "gs://bucket/predictions/P/" + f.path.name,
                         ground_truth_started_at=TPC.T0 + timedelta(minutes=5),
                         clock_skew=timedelta(seconds=60), banked_at={"P1": TPC.T0 - timedelta(days=3)})


def test_F6_the_record_does_not_name_the_training_rows_it_was_fitted_on(monkeypatch):
    from attnbench.predict import learned as LM
    monkeypatch.setattr(LM, "LOCKED_CONFIG", TPC.CONFIG)
    rec = TPC.p2_record()
    flat = json.dumps(rec["training"])
    assert "sha256" in flat, f"training records hold only {sorted(rec['training']['fitted_sum'])}"


def test_F6_training_file_at_predict_is_any_self_consistent_file(tmp_path):
    ops = C._components(TPC.cell().priced)
    rec = {"schema": PT.SCHEMA, "hardware": {"pinned_bandwidth": 1.0, "pageable_bandwidth": 1.0},
           "rows": [{"cell": {"model_name": "m", "code_path": "x", "arm": "dense",
                              "context_length": 2048, "density": 1.0, "batch": 1},
                     "components": ops, "target_s": 0.1, "metadata": {}}],
           "sensitivity": {}}
    rec["sha256"] = PT._digest(rec)
    p = tmp_path / "other.json"
    p.write_text(json.dumps(rec))
    with pytest.raises(Exception):
        S.training_rows(S.T61, path=p)                # predict reads it with no lock check


# ---------------------------------------------------------------- F7: block identity

def _pre(d, findings):
    out = S.step_dir(d, "preflight", S.T61)
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps({"findings": findings}))


def test_F7_block_identity_ignores_image_driver_and_missing_fields(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    base = {"instance_type": "a2-ultragpu-1g", "gpu_name": "A100"}   # cpu_model absent in both
    _pre(a, {**base, "boot_image": "v6", "driver": "550"})
    _pre(b, {**base, "boot_image": "v7", "driver": "560"})
    br = SimpleNamespace(imports=(("b1", a),))
    assert S.block_identity_problems(S.T61, b, br) != []


# ---------------------------------------------------------------- F8: real gap under the allowed skew

def test_F8_a_start_two_seconds_after_the_sync_in_real_time_is_accepted(tmp_path):
    """The instance clock runs 59 s ahead of the store's (inside the 60 s allowed)."""
    rec = TPC.p1_record(written_at=TPC.T0 + timedelta(seconds=58))  # real T0-1, instance +59
    store = TPC.MemoryStore(TPC.Clock(TPC.T0), no_clobber=True)
    f, _ = TPC.committed(tmp_path, rec, store, TPC.T0)              # store: created at T0
    start_instance = TPC.T0 + timedelta(seconds=61)                 # real T0 + 2 s
    with pytest.raises(C.PredictionCommitError):
        C.load_committed(store, "gs://bucket/predictions/P/" + f.path.name,
                         ground_truth_started_at=start_instance,
                         clock_skew=timedelta(seconds=C.REQUIRED_SYNC_GAP_S), banked_at={})


# ---------------------------------------------------------------- F9: resume at another commit

def test_F9_a_block_resumed_after_a_fix_commit_still_records_its_first_commit(tmp_path, monkeypatch):
    """`--from ground_truth` after a committed fix (the documented recovery) runs the
    block's ground truth at commit Y; its manifest, which block_drift reads, still says X."""
    T = S.T61
    head = {"c": "X" * 40}
    monkeypatch.setattr(S, "code_commit", lambda dry_run: head["c"])
    _, d1 = _run_block(T, tmp_path, _p2_lock(tmp_path), "b1")
    fail = {"on": True}

    def flaky(cfg, args, block=None):
        r = S.FakeHarness(Path(args.session))(cfg, args, block)
        if fail["on"]:
            r["complete"] = False
        return r
    Tf = with_harness(T, flaky)
    rc, d2 = _run_block(Tf, tmp_path, _p2_lock(tmp_path), "b2", [d1])
    assert rc == 1
    head["c"], fail["on"] = "Y" * 40, False          # the fix, committed; resume on the same boot
    br, why = S.block_run(Tf, Namespace(lock=str(_p2_lock(tmp_path)), block="b2",
                                        import_block=[str(d1)], dry_run=True))
    rc = S.run_session(Tf, d2, start="ground_truth", extra=[], dry_run=True, sync=False,
                       runner=inproc(Tf), block=br)
    m = json.loads((d2 / "manifest.json").read_text())
    got = BD.collect([d1, d2], ground_truth_dir=S.step_dir(Path(), "ground_truth", T).name)
    assert rc != 0 or got["code_commit"] == "Y" * 40, \
        f"resumed at Y, passed (rc {rc}); the analysis reads one commit, {got['code_commit'][:4]}…"


# ---------------------------------------------------------------- F10: P4's predict is not built

def test_F10_p4_predict_cannot_train_on_p2_and_step_one_does_not_say_so(tmp_path):
    import t64_session as P4
    T = P4.T64
    with pytest.raises(S.NotBuilt):
        S.training_rows(T, path=tmp_path / "unused")   # refuses before any file is read
    assert T.staged is not None, "P4 learns this at predict, after its component steps"
