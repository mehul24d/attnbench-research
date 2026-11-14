"""The pre-launch CPU pass (adviser's ruling 4 of the 11 Nov 2026 paste).

Every launch card runs this before the researcher is asked for a yes:

    scripts/prelaunch_cpu_pass.sh <rehearsal|p2|p4> [training file]

It runs the session's REAL config under its real session name, on this
commit, through the real start checks (`start_refusal`: the exit wrapper,
a clean tree, the timed-path pin, the banking times) and every step's real
body that a CPU can run: prereqs (with the staged inputs), the host and
device components (T3.1's code on synthetic recordings), predict (the real
training file and the learned fit), the real 65 s hold, ground_truth (T3.1's
orchestration; only `time_arms` and the profiler are fakes), late_check,
trace_checks (the real `check_trace.py`) and host_speed_end. With a block
spec it then runs block B: a reset during ground truth, the resume refused
and discarded, a new block passing, and the drift script.

Mocked, and only these: the steps that need the GPU (preflight, link,
launch_queue, probe, the mask recordings, dense_select, host_speed_mid,
fa3_import) run their dry bodies; the device under ground truth and the
components; and the bucket (a directory store), since a CPU pass makes no
cloud call. Skipped unless ATTNBENCH_PRELAUNCH names the config; the record
goes to ATTNBENCH_PRELAUNCH_OUT/prelaunch_cpu_pass.json.

ATTNBENCH_PRELAUNCH_MASKS names a directory of banked XAttention masks,
staged into every session as launch stages them: T3.2d's step 5 masks as P2's
`p1_masks`, and for P4, until P2 banks its own, the same masks as its
`p2_masks` stand-in (as ruling 1h's synthetic P2 rows).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

CONFIG = os.environ.get("ATTNBENCH_PRELAUNCH", "")
pytestmark = pytest.mark.skipif(not CONFIG, reason="the pre-launch pass runs from "
                                "scripts/prelaunch_cpu_pass.sh, before a launch card's yes")

import t31_steps as st  # noqa: E402
import t61_session as S  # noqa: E402

# The steps whose real body needs the GPU: their dry body runs instead.
GPU_STEPS = {"preflight", "link", "launch_queue", "probe", "mask_gate", "masks",
             "dense_select", "host_speed_mid", "fa3_import"}
COMPONENT_STEPS = {"host_components", "device_components"}
BANKED = {"P1": "2026-11-09T11:12:31+00:00"}   # P1's banking time (T3.2d's training file)


def _config():
    if CONFIG == "rehearsal":
        import t61_rehearsal as R
        return R.REHEARSAL, str(ROOT / R.LOCK)
    if CONFIG == "p2":
        return S.T61, os.environ.get("ATTNBENCH_PRELAUNCH_LOCK") or None
    if CONFIG == "p4":
        import t64_session as P4
        return P4.T64, os.environ.get("ATTNBENCH_PRELAUNCH_LOCK") or None
    raise SystemExit(f"ATTNBENCH_PRELAUNCH={CONFIG!r}: rehearsal, p2 or p4")


BANKED["P2"] = "2026-11-18T23:00:00+00:00"     # a stand-in: P2 is not banked yet


def bank(monkeypatch, cfg, home: Path) -> dict:
    """The pairing ledger in a directory store (ruling 1c: the bucket is
    mocked), each pairing `cfg` trains on banked at its time with its
    training file's sha256, as `scripts/bank_pairing.py` banks it (ruling
    1e: a real run reads both from here, never from --banked-at)."""
    import hashlib
    from attnbench.predict.ledger import Ledger
    given = dict(x.split("=", 1) for x in os.environ.get("ATTNBENCH_PRELAUNCH_BANKED", "").split())
    clock = {"t": 0.0}
    store = S.DirStore(home / "pairing_store", lambda: clock["t"] or time.time())
    files = {"P1": S.P1_TRAINING, "P2": S.P2_TRAINING}
    out = {}
    for p in cfg.banked:
        out[p] = given.get(p, BANKED[p])
        clock["t"] = datetime.fromisoformat(out[p]).timestamp()
        Ledger(store, S.DirStore.PREFIX, p).record_banked(
            {"training_file_sha256": hashlib.sha256(Path(files[p]).read_bytes()).hexdigest(),
             "banked_at": out[p]})
    clock["t"] = 0.0
    monkeypatch.setattr(S, "ledger_for", lambda c, args, pairing=None: Ledger(
        store, S.DirStore.PREFIX, pairing or c.pairing))
    return out


def synthetic_p2(p1: Path, path: Path) -> Path:
    """P4's P2 rows before P2 has run (ruling 1h's CPU pass): P1's cells,
    relabelled P2, each target 0.8 of P1's, in P2's training file."""
    from dataclasses import replace
    from attnbench.predict import p1_training as PT
    from attnbench.predict import p2_training as P2T
    rec = PT.load(p1)
    rows = [type(r)(cell=replace(r.cell, pairing="P2"), measured=0.8 * r.measured)
            for r in PT.rows_from_record(rec)]
    out = {"rows": rows, "metadata": [{"cell": [r.cell.arm, r.cell.context_length,
                                                r.cell.batch], "synthetic": True} for r in rows],
           "left_out": [], "blocks": ["synthetic"], "unpassed": [], "min_clean_rounds": 0,
           "prediction": "synthetic: P1's cells relabelled P2"}
    from attnbench.predict.training import HardwareFeatures
    P2T.dump(P2T.to_record(out, HardwareFeatures(**rec["hardware"])), path)
    return path


class Trace:
    """The profiler's stand-in: its export writes a device trace (one kernel,
    one copy on one stream, no overlap, no managed memory)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def export_chrome_trace(self, path):
        ev = [{"ph": "X", "cat": "kernel", "name": "k", "ts": 0, "dur": 10, "pid": 0, "tid": 7,
               "args": {"stream": 7}},
              {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 20, "dur": 5,
               "pid": 0, "tid": 7, "args": {"stream": 7}}]
        Path(path).write_text(json.dumps({"traceEvents": ev}))


def make_runner(cfg, comp, seen):
    """Each step in this process: a GPU step's dry body, the components on
    synthetic recordings, every other step's real body."""
    def runner(cmd, log):
        argv = cmd[cmd.index("step"):]
        step = argv[1]
        session = Path(argv[argv.index("--session") + 1])
        out = Path(argv[argv.index("--out") + 1])
        seen.append((session.name, step))
        log.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        if step in GPU_STEPS:
            # the step's own options, as its process would parse them (P4's FA3 hash)
            fa3 = argv[argv.index("--fa3-sha256") + 1] if "--fa3-sha256" in argv else None
            ns = Namespace(session=str(session), out=str(out), dry_run=True, fa3_sha256=fa3)
            found = cfg.fakes[step](cfg, ns)
            if step == "link":
                import pandas as pd
                (out / "link").mkdir(exist_ok=True)
                pd.DataFrame([{"bw_h2d_pinned_Bps": 1e10, "bw_h2d_pageable_Bps": 5e9,
                               "bw_d2h_pinned_Bps": 8e9, "bw_d2h_pageable_Bps": 4e9}]).to_parquet(
                    out / "link" / "link.parquet")
                (out / "recursion_inputs.json").write_text(json.dumps(
                    {"queue_depth": 8, "fake": False,
                     "hardware": {"pinned_bandwidth": 1e10, "pageable_bandwidth": 5e9},
                     "by_length": {str(L): st.FAKE_MECHANISMS for L in comp["lengths"]}}))
            if step == "dense_select":
                (out / "session_floor.json").write_text(json.dumps({"floor_ms": 0.05}))
            S.write_json(out / "result.json", {"step": step, "status": S.PASSED, "reason": "",
                                               "findings": {"prelaunch_mock": "GPU step: its dry "
                                                            "body", **found}})
            log.write_text("# pre-launch CPU pass: GPU step, dry body\n")
            return 0
        if step in COMPONENT_STEPS:
            ns = SimpleNamespace(dry_run=False, session=str(session), out=str(out))
            fn = st.step_host_components if step == "host_components" else st.step_device_components
            try:
                found, status, reason = fn(ns, comp), S.PASSED, ""
            except Exception as e:  # noqa: BLE001
                found, status, reason = {}, S.FAILED, f"{type(e).__name__}: {e}"
            S.write_json(out / "result.json", {"step": step, "status": status, "reason": reason,
                                               "findings": found})
            log.write_text("# pre-launch CPU pass: components on synthetic recordings\n")
            return 0 if status == S.PASSED else 1
        rc = S.main(argv, cfg=cfg)            # the real body, in this process
        log.write_text(f"# $ {' '.join(cmd)}\n# exit {rc}\n")
        return rc
    return runner


def mock_device(monkeypatch, training: Path):
    """The GPU, and the bucket, and nothing else."""
    import torch
    import test_t31_wiring as W
    from attnbench import dense_selection as ds
    from attnbench import ground_truth as gt
    from attnbench.backends import compiled as CB
    W._components(monkeypatch, W.FakeHarness())
    monkeypatch.setattr(S, "P1_TRAINING", training)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(ds, "load_choice", lambda p: SimpleNamespace(chosen="sdpa_flash"))
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {"sdpa_flash": "FLASH"})
    monkeypatch.setattr(ds, "check_control", lambda *a, **k: None)
    monkeypatch.setattr(st, "_tracer", lambda: Trace)
    sampler = st._sampler
    monkeypatch.setattr(st, "_sampler", lambda cfg, dry_run: sampler(cfg, True))
    monkeypatch.setattr(st, "_run", lambda cmd: st.main(cmd[2:]))
    monkeypatch.setattr(CB, "_claim", lambda setting: None)   # one process per group on the GPU

    def timing_arms(names, L, densities, batch, backends, compile_settings):
        return "w", torch.zeros((batch, L - 10)), f"p{L}", [SimpleNamespace(name=n) for n in names], {}
    h = W.FakeHarness()
    h.timing_arms = timing_arms
    h._backend = lambda spec: SimpleNamespace(capability=SimpleNamespace(name=spec.name),
                                              forward=lambda *a, **k: None)
    h.input_ids = lambda L, index=0: (torch.zeros((1, L - 5)), f"p{L}_{index}")
    h.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))
    monkeypatch.setattr(st, "_harness", lambda args: h)

    def time_arms(w, ids, arms, *, reps, controls, tracer=None, warmup, **k):
        if tracer is not None:
            for _ in range(len(arms) * (warmup + reps)):
                with tracer():
                    pass
        return {"first_clocked_block_started_at": time.time(), "complete": True,
                "clean_rounds": reps, "rounds_run": reps, "traced": tracer is not None,
                "rows": [{"arm": a.name, "prefill_ms_median": 100.0 + 5 * j, "clean_rounds": reps,
                          "rounds_run": reps, "traced": tracer is not None,
                          "context_length": int(ids.shape[-1]), "batch": int(ids.shape[0])}
                         for j, a in enumerate(arms)]}
    monkeypatch.setattr(gt, "time_arms", time_arms)
    monkeypatch.setattr(S, "_host_speed_row", lambda out, tag: {
        "null_launch_s_median": 3.9e-6, "git_dirty": False})

    def store_for(args):
        session = Path(args.session)
        b = S.load_manifest(session).get("block") or {}
        home = Path((b.get("imported") or {}).get(b.get("predict_block"), session))
        return S.DirStore(home / "store", time.time), S.DirStore.PREFIX
    monkeypatch.setattr(S, "store_for", store_for)


def _record(out: Path, rec: dict) -> None:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        S.write_json(out / "prelaunch_cpu_pass.json", rec)


def _attempts(session: Path) -> list:
    m = S.load_manifest(session)
    return [{"step": a["step"], "status": a["status"], "reason": a["reason"][:400],
             "imported_from": a.get("imported_from")} for a in m["attempts"]]


def test_the_real_session_passes_on_this_commit_with_only_the_gpu_mocked(tmp_path, monkeypatch):
    cfg, lock = _config()
    out = Path(os.environ["ATTNBENCH_PRELAUNCH_OUT"]) if os.environ.get(
        "ATTNBENCH_PRELAUNCH_OUT") else None
    training = Path(os.environ.get("ATTNBENCH_PRELAUNCH_TRAINING") or S.P1_TRAINING)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    rec = {"config": CONFIG, "pairing": cfg.pairing, "task": cfg.task, "commit": commit,
           "training_file": str(training), "block_spec": lock, "sessions": {},
           "mocked": sorted(GPU_STEPS) + ["the device under ground truth and the components",
                                          "the bucket (a directory store)"],
           "passed": False}
    mock_device(monkeypatch, training)
    if "P2" in cfg.banked:
        p2 = os.environ.get("ATTNBENCH_PRELAUNCH_P2_TRAINING")
        monkeypatch.setattr(S, "P2_TRAINING", Path(p2) if p2 else synthetic_p2(
            training, tmp_path / "p2_training.json.gz"))
        rec["p2_training_file"] = p2 or "synthetic: P1's cells relabelled P2 (ruling 1h)"
    rec["banked"] = bank(monkeypatch, cfg, tmp_path)
    opts = [f"--fa3-sha256={'0' * 64}"] if "fa3_import" in cfg.names else []

    # the start checks, as `main` makes them under the armed exit wrapper
    monkeypatch.setenv(S.EXIT_WRAPPER_ENV, "armed")
    args = S.parser(cfg).parse_args(["run", *opts])
    extra = S._passthrough(args, cfg)
    why = S.start_refusal(cfg, args)
    rec["start_refusal"] = why
    if why:
        _record(out, rec)
        pytest.fail(f"start refused: {why}")
    try:
        comp = st.load_config(str(ROOT / cfg.components_config), dry_run=False)
    except st.StepFailed as e:
        rec["start_refusal"] = f"{cfg.components_config}: {e}"
        _record(out, rec)
        pytest.fail(rec["start_refusal"])
    seen = []
    runner = make_runner(cfg, comp, seen)

    def block(bid, imports=()):
        br, why = S.block_run(cfg, Namespace(lock=lock, block=bid, dry_run=False,
                                             import_block=[str(d) for d in imports]))
        assert why is None, why
        return br

    masks = os.environ.get("ATTNBENCH_PRELAUNCH_MASKS")
    into = S.P1_MASKS if cfg.pairing == "P2" else S.P2_MASKS
    rec["masks"] = {"staged_from": masks, "into": into,
                    "stand_in": bool(masks) and into == S.P2_MASKS}

    def run(name, bid=None, imports=(), start="prereqs", runner=runner):
        d = tmp_path / name
        if masks:
            shutil.copytree(masks, d / into, dirs_exist_ok=True)
        rc = S.run_session(cfg, d, start=start, extra=extra, dry_run=False, sync=False,
                           runner=runner, block=block(bid, imports) if bid else None)
        rec["sessions"][name] = {"rc": rc, "attempts": _attempts(d)}
        _record(out, rec)
        return d, rc

    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    monkeypatch.setattr(S, "boot_id", lambda: "boot-A")
    b1, rc = run(f"{cfg.session_prefix}_{stamp}" + ("_b1" if lock else ""), "b1" if lock else None)
    assert rc == 0, json.dumps(rec["sessions"], indent=1)
    receipt = json.loads((b1 / S.RECEIPT).read_text())
    assert receipt["remote_path"].startswith(f"predictions/{cfg.pairing}/")
    rec["receipt"] = receipt

    if lock:
        # block B: a reset during ground truth on its first boot
        class Reset(Exception):
            pass

        def resetting(cmd, log):
            if cmd[cmd.index("step") + 1] == "ground_truth":
                raise Reset()
            return runner(cmd, log)
        monkeypatch.setattr(S, "boot_id", lambda: "boot-B1")
        with pytest.raises(Reset):
            run(f"{cfg.session_prefix}_{stamp}_b2_interrupted", "b2", [b1], runner=resetting)
        monkeypatch.setattr(S, "boot_id", lambda: "boot-B2")
        x, rc_x = run(f"{cfg.session_prefix}_{stamp}_b2_interrupted", "b2", [b1],
                      start="ground_truth")
        assert rc_x == 2 and (x / S.DISCARD).exists(), "the resumed block was not discarded"
        y, rc_y = run(f"{cfg.session_prefix}_{stamp}_b2", "b2", [b1])
        assert rc_y == 0, json.dumps(rec["sessions"], indent=1)
        sh = ROOT / "scripts" / "t61_rehearsal_b.sh"
        if CONFIG == "rehearsal" and sh.exists():
            code = sh.read_text().split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
            monkeypatch.setattr(sys, "argv", ["-", str(b1), str(x), str(y)])
            exec(compile(code, "drift", "exec"), {"__name__": "__main__"})
            drift = json.loads((y / "block_drift.json").read_text())
            assert drift["blocks"] == ["b1", "b2"]
            assert [r["block"] for r in drift["discarded"]] == ["b2"]
            rec["drift"] = {"blocks": drift["blocks"], "discarded": ["b2"]}
    rec["steps_run"] = seen
    rec["passed"] = True
    _record(out, rec)
    shutil.rmtree(tmp_path, ignore_errors=True)
