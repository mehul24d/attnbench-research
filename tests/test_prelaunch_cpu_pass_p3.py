"""The pre-launch CPU pass for P3 (the researcher's task of 26 Nov 2026, item 2):

    scripts/prelaunch_cpu_pass.sh p3

P3 runs `scripts/t31_session.py`, not the P2/P4 runner, so its pass is its own
file. It runs `t31_session.main` exactly as `scripts/aws_p3_instance.sh` runs it
on the instance: the real config (`configs/t31/p3_session.json`, loaded by the
pinned `load_config`, so its T4.1 values come from the lock as P2's and P4's
do), the armed exit wrapper, the clean-tree refusal, and the three blocks
(`--until probe`; `--from host_components --until host_components`; `--from
device_components`) on one session directory. Every step's real body runs in
this process, and the store is the real `scripts/aws_p3_gsutil.py` on PATH as
`gsutil` (ATTNBENCH_BUCKET=gs://p3-mac), with a thread here standing in for the
Mac's witness (`witness_store` in `scripts/aws_p3_driver.sh`). After block 2
the P6 gate script reads the block's real output against P1's host components,
as the driver does.

Mocked, and only these (the record lists them):
  - probe and postflight run their dry bodies (the structural probe records on
    the GPU; the end probe requires one);
  - inside real bodies: T2.27's GPU probes run dry (`_t227`), the block-sparse
    kernel contract is not run, the dense selection's timing is a fixed choice
    (`fa2`, as the rehearsal's dense cell chose on the g6.2xlarge) and the GPU
    boundary of the components and ground truth (recorder, null pass, replay,
    the timed harness, the profiler, the sampler) is test_t31_wiring's;
  - step 12's leg-3 timing runs the dry leg 3 (`_dry_leg3`) inside the real step
    body, so its plan is still put in, and read back from, the P3 store.
The hold is the real one (PLANNED_SYNC_WAIT_S from the witness's time).

Skipped unless ATTNBENCH_PRELAUNCH=p3; the record goes to
ATTNBENCH_PRELAUNCH_OUT/prelaunch_cpu_pass.json. The checks of the lock values
(`lock_values_check`) and of the pass's wiring run in every suite run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import t31_session as S  # noqa: E402
import t31_steps as st  # noqa: E402

CONFIG = os.environ.get("ATTNBENCH_PRELAUNCH", "")
P3_CONFIG = "configs/t31/p3_session.json"
LOCK = ROOT / "docs" / "lock_values.json"
# The blocks, as scripts/aws_p3_instance.sh runs them.
BLOCKS = {"b1": ["--until", "probe"],
          "b2": ["--from", "host_components", "--until", "host_components"],
          "b3": ["--from", "device_components"]}
DRY_BODY = {"probe", "postflight"}
CHOSEN = "fa2"
P1_HOST = Path(os.environ.get(
    "ATTNBENCH_PRELAUNCH_P1_HOST",
    str(Path.home() / "Desktop/research/attnbench_backups/bucket_20261117/"
        "attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components")))
MOCKED = ["probe and postflight: their dry bodies",
          "T2.27's GPU probes inside preflight and link: dry",
          "the block-sparse kernel contract: not run",
          f"the dense selection's timing: a fixed choice ({CHOSEN})",
          "the GPU boundary of the components and ground truth (test_t31_wiring's fakes; "
          "the bare profile's per-operator prices are 1 us per call)",
          "step 12's leg-3 timing: the dry leg 3 inside the real step body"]


class LockValuesError(AssertionError):
    pass


def lock_values_check(cfg: dict, lock: dict) -> dict:
    """A pairing other than P1 takes its session values from the lock, as P2
    and P4 do: every T4.1 item from the lock's typed values, its pairing keys
    from the lock's P2 and P4 pairing values (the lock has no P3 block), and
    nothing from P1's working table. Returns what was checked."""
    if cfg.get("pairing") == st.WORKING_PAIRING:
        raise LockValuesError("the config is P1's: its values are the working table's")
    if "working_values" in cfg:
        raise LockValuesError(f"{cfg['pairing']} carries P1's working values")
    filled = sorted((cfg.get("lock_values") or {}).get("filled") or [])
    if filled != sorted(st.LOCK_SOURCED):
        raise LockValuesError(f"filled from the lock: {filled}, not every T4.1 item")
    if cfg["lock_values"].get("sha256") != hashlib.sha256(LOCK.read_bytes()).hexdigest():
        raise LockValuesError("the values were filled from another lock file")
    for k, (item, read) in st.LOCK_SOURCED.items():
        want = read(lock["values"][item]["value"])
        if cfg[k] != want:
            raise LockValuesError(f"{k} is {cfg[k]!r}, the lock's {item} gives {want!r}")
    for k in st.PAIRING_KEYS:
        if cfg[k] != lock["pairing_values"]["P2"][k] or cfg[k] != lock["pairing_values"]["P4"][k]:
            raise LockValuesError(f"{k} is {cfg[k]!r}, not the lock's P2 and P4 value")
    return {"pairing": cfg["pairing"], "filled_from_lock": filled,
            "pairing_keys_from": "the lock's P2 and P4 pairing_values",
            "gt_groups": sorted(cfg["gt_groups"]), "max_rerun_rounds": cfg["max_rerun_rounds"],
            "min_clean_rounds": cfg["min_clean_rounds"],
            "resolution_floor_ms": cfg["resolution_floor_ms"]}


# --- run in every suite ---------------------------------------------------------------

def test_p3s_loaded_config_takes_the_locks_values_and_none_of_p1s():
    cfg = st.load_config(str(ROOT / P3_CONFIG), dry_run=False)
    r = lock_values_check(cfg, json.loads(LOCK.read_text()))
    assert r["gt_groups"] == ["eager"] and r["pairing"] == "P3"
    assert len(st.cells(cfg)) == 35


@pytest.mark.parametrize("bad", ["p1_values", "one_not_filled", "pairing_key", "is_p1",
                                 "floor", "other_lock"])
def test_the_lock_values_check_refuses_a_config_that_did_not_take_them(bad):
    cfg = st.load_config(str(ROOT / P3_CONFIG), dry_run=False)
    lock = json.loads(LOCK.read_text())
    if bad == "p1_values":
        cfg["working_values"] = {"sha256": "x"}
    elif bad == "one_not_filled":
        cfg["lock_values"] = dict(cfg["lock_values"], filled=cfg["lock_values"]["filled"][1:])
    elif bad == "pairing_key":
        cfg["reps"] = cfg["reps"] + 1
    elif bad == "floor":
        cfg["max_rerun_rounds"] = cfg["max_rerun_rounds"] + 1
    elif bad == "other_lock":
        cfg["lock_values"] = dict(cfg["lock_values"], sha256="0" * 64)
    else:
        cfg["pairing"] = "P1"
    with pytest.raises(LockValuesError):
        lock_values_check(cfg, lock)


def test_the_pass_script_takes_p3_and_runs_this_file():
    sh = (ROOT / "scripts" / "prelaunch_cpu_pass.sh").read_text()
    assert "rehearsal|p2|p4|p3)" in sh
    assert "tests/test_prelaunch_cpu_pass_p3.py" in sh
    r = subprocess.run(["bash", str(ROOT / "scripts" / "prelaunch_cpu_pass.sh"), "p5"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "<rehearsal|p2|p4|p3>" in r.stderr


def test_the_blocks_are_the_instance_scripts():
    sh = (ROOT / "scripts" / "aws_p3_instance.sh").read_text()
    for args in BLOCKS.values():
        assert " ".join(args) in sh
    assert P3_CONFIG in sh


# --- the pass ---------------------------------------------------------------------------

def witness(spool: Path, stop: threading.Event, seen: list) -> threading.Thread:
    """The Mac's side of the store (`witness_store`): for each request, check
    the object against it, keep a copy, return it, then write the witness with
    this clock's time."""
    def loop():
        while not stop.is_set():
            for r in sorted(spool.rglob("*.request")):
                o = r.with_name(r.name[: -len(".request")])
                w = o.with_name(o.name + ".witness")
                if w.exists():
                    continue
                req = json.loads(r.read_text())
                data = o.read_bytes()
                got = hashlib.sha256(data).hexdigest()
                if got != req["sha256"]:
                    seen.append({"object": str(o.relative_to(spool)), "refused": got})
                    continue
                at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                o.with_name(o.name + ".mac").write_bytes(data)
                w.with_name(w.name + ".part").write_text(json.dumps(
                    {"sha256": got, "received_utc": at}))
                w.with_name(w.name + ".part").rename(w)
                seen.append({"object": str(o.relative_to(spool)), "sha256": got, "at": at})
            time.sleep(0.2)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def mock_gpu(monkeypatch, controls_seen: list):
    """The GPU boundary and nothing else (the module docstring's list)."""
    import torch
    import test_t31_wiring as W
    from attnbench import dense_selection as ds
    from attnbench import ground_truth as gt
    from attnbench.backends import compiled as CB
    from attnbench.components import bare_timer as bt
    W._components(monkeypatch, W.FakeHarness())
    fake_bare = bt.bare_profile

    def bare_profile(seq, bp, null, *, aggregate_tol):
        """The wiring fake's bare profile, with the per-operator prices the real
        one writes (`bare_timer.bare_profile`), which the P6 gate reads."""
        prof, rec = fake_bare(seq, bp, null, aggregate_tol=aggregate_tol)
        ops = {}
        for op in seq.ops:
            if op.aten_op:
                o = ops.setdefault(op.aten_op, {"calls": 0, "bare_s": 0.0})
                o["calls"] += 1
                o["bare_s"] += 1e-6
        return prof, {**rec, "host_total_s": prof.total_s, "operators": ops}
    monkeypatch.setattr(bt, "bare_profile", bare_profile)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    real_t227 = st._t227

    def t227(args, step, out):
        """T2.27's probe run dry; its link and queue rows then read as measured
        (as the P2/P4 pass writes them), so `step_link`'s real body reads them."""
        import pandas as pd
        real_t227(SimpleNamespace(dry_run=True), step, out)
        for p in Path(out).glob("*.parquet"):
            df = pd.read_parquet(p)
            if "dry_run" in df.columns:
                df["dry_run"] = False
                df.to_parquet(p)
    monkeypatch.setattr(st, "_t227", t227)
    monkeypatch.setattr(st, "kernel_contract", lambda args, out: {"status": "mocked: no GPU"})
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {CHOSEN: "FA2", "sdpa_flash": "FLASH"})

    def select(cell, attn_cfg, candidates, *, rounds, resolution_floor_ms, device):
        assert resolution_floor_ms is not None     # the lock's floor at this cell
        return SimpleNamespace(cell=cell, chosen=CHOSEN, complete=True,
                               floor=resolution_floor_ms)
    monkeypatch.setattr(ds, "select_dense_baseline", select)

    def write_choice(choice, directory):
        p = Path(directory) / f"dense_choice_{choice.cell.key.replace(' ', '_')}.json"
        p.write_text(json.dumps({"chosen": choice.chosen, "floor_ms": choice.floor,
                                 "mocked": "the dense selection's timing"}))
        return p
    monkeypatch.setattr(ds, "write_choice", write_choice)
    monkeypatch.setattr(ds, "load_choice", lambda p: SimpleNamespace(chosen=CHOSEN))
    monkeypatch.setattr(ds, "check_control", lambda c, name, stamp, key: controls_seen.append(
        (name, key.context_length, key.batch)))
    monkeypatch.setattr(st, "_tracer", lambda: __import__("test_prelaunch_cpu_pass").Trace)
    sampler = st._sampler
    monkeypatch.setattr(st, "_sampler", lambda cfg, dry_run: sampler(cfg, True))
    real_run = st._run
    monkeypatch.setattr(st, "_run", lambda cmd: st.main(cmd[2:]) if
                        Path(cmd[1]).name == "t31_steps.py" else real_run(cmd))
    monkeypatch.setattr(CB, "_claim", lambda setting: None)
    monkeypatch.setattr(st, "_real_leg3", lambda args, decl, mech, q, session, write_plan:
                        st._dry_leg3(decl, mech, q, write_plan))

    def timing_arms(names, L, densities, batch, backends, compile_settings):
        controls_seen.append(("control", backends[0], L, batch))
        return ("w", torch.zeros((batch, L - 10)), f"p{L}",
                [SimpleNamespace(name=n) for n in names], {})
    h = W.FakeHarness()
    h.timing_arms = timing_arms
    h._backend = lambda spec: SimpleNamespace(capability=SimpleNamespace(name=spec.name),
                                              forward=lambda *a, **k: None)
    h.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))
    from attnbench.config import AttnConfig
    h._cfg = lambda spec, L, density=None, batch=1: AttnConfig(   # the dense selection's input
        seq_len=L, batch=batch, n_heads_q=1, n_heads_kv=1, mask=spec.mask)
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


def make_runner(seen: list):
    """Each step in this process: probe's and postflight's dry bodies, every
    other step's real body (`t31_steps.main`, which loads the config itself)."""
    def runner(cmd, log):
        argv = cmd[2:]
        step = argv[0]
        seen.append(step)
        log.parent.mkdir(parents=True, exist_ok=True)
        rc = st.main(argv + (["--dry-run"] if step in DRY_BODY else []))
        log.write_text(f"# pre-launch CPU pass: {'dry body' if step in DRY_BODY else 'real body'}"
                       f"\n# $ {' '.join(cmd)}\n# exit {rc}\n")
        return rc
    return runner


def _record(out, rec):
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "prelaunch_cpu_pass.json").write_text(json.dumps(rec, indent=1, default=str))


@pytest.mark.skipif(CONFIG != "p3", reason="the P3 pre-launch pass runs from "
                    "scripts/prelaunch_cpu_pass.sh p3, before the launch card's yes")
def test_p3s_session_passes_in_its_three_blocks_on_this_commit(tmp_path, monkeypatch):
    out = Path(os.environ["ATTNBENCH_PRELAUNCH_OUT"]) if os.environ.get(
        "ATTNBENCH_PRELAUNCH_OUT") else None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    rec = {"config": "p3", "pairing": "P3", "session_config": P3_CONFIG, "commit": commit,
           "runner": "scripts/t31_session.py (main), as scripts/aws_p3_instance.sh runs it",
           "store": "scripts/aws_p3_gsutil.py as gsutil, gs://p3-mac, a witness thread",
           "mocked": MOCKED, "blocks": {}, "passed": False}
    cfg = st.load_config(str(ROOT / P3_CONFIG), dry_run=False)
    rec["lock_values"] = lock_values_check(cfg, json.loads(LOCK.read_text()))
    _record(out, rec)

    # the instance's environment (aws_p3_instance.sh), with the store's spool here
    spool, bindir = tmp_path / "p3_store", tmp_path / "bin"
    bindir.mkdir()
    g = bindir / "gsutil"
    g.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" '
                 f'"{ROOT / "scripts" / "aws_p3_gsutil.py"}" "$@"\n')
    g.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ATTNBENCH_BUCKET", "gs://p3-mac")
    monkeypatch.setenv("ATTNBENCH_P3_STORE", str(spool))
    monkeypatch.setenv(S.EXIT_WRAPPER_ENV, "armed")
    controls_seen, seen, witnessed = [], [], []
    mock_gpu(monkeypatch, controls_seen)
    runner = make_runner(seen)
    real_run_session = S.run_session
    monkeypatch.setattr(S, "run_session", lambda *a, **k: real_run_session(*a, **k, runner=runner))
    stop = threading.Event()
    witness(spool, stop, witnessed)
    session = tmp_path / "p3_session"
    try:
        for b, args in BLOCKS.items():
            t0 = time.time()
            rc = S.main(["--config", P3_CONFIG, "--session", str(session), "--no-sync", *args])
            m = S.load_manifest(session)
            rec["blocks"][b] = {"rc": rc, "seconds": round(time.time() - t0, 1),
                                "steps": [a["step"] for a in m["attempts"]
                                          if a["status"] == "passed" and a["step"] not in
                                          sum((v["steps"] for v in rec["blocks"].values()), [])],
                                "failed": [{"step": a["step"], "reason": a["reason"][:400]}
                                           for a in m["attempts"] if a["status"] == "failed"]}
            _record(out, rec)
            assert rc == 0, json.dumps(rec["blocks"][b], indent=1)
            if b == "b2" and P1_HOST.exists():
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "p6_host_effect.py"),
                                    str(st.step_dir(session, "host_components")), str(P1_HOST)],
                                   capture_output=True, text=True, cwd=ROOT)
                n3 = len(list(st.step_dir(session, "host_components").glob("*/bare_prices.json")))
                gate = json.loads(r.stdout) if r.returncode == 0 else {"stderr": r.stderr[-400:]}
                rec["gate"] = {"rc": r.returncode, "cells_with_bare_prices": n3,
                               "cells_read": gate.get("cells"), "verdict": gate.get("verdict"),
                               "note": "synthetic recordings: d means nothing; this checks "
                                       "that the gate reads the real step body's layout"}
                _record(out, rec)
                assert r.returncode == 0 and n3 == 35 and gate["cells"] == 35, rec["gate"]
    finally:
        stop.set()
    assert seen == list(st.STEPS), seen
    last = S.last_status(S.load_manifest(session))
    assert all(last[s] == "passed" for s in st.STEPS), last
    # ground truth's control is the dense selection's choice, at every cell
    ctrl = [c for c in controls_seen if c[0] == "control"]
    assert ctrl and all(c[1] == "FA2" for c in ctrl)
    # the store: the prediction, its sidecar and the leg-3 plan, each witnessed once
    receipt = json.loads((st.step_dir(session, "predict") / "receipt.json").read_text())
    assert receipt["remote"].startswith("predictions/P3/")
    objs = sorted(w["object"] for w in witnessed if "sha256" in w)
    assert len(objs) == 3 and any(o.endswith("leg3_plan.json") for o in objs), witnessed
    synced = datetime.fromisoformat(receipt["synced_at"])
    hold = json.loads((st.step_dir(session, "hold") / "result.json").read_text())["findings"]
    assert hold["held_s"] > 0
    rec.update(steps_run=seen, store=witnessed, receipt={k: receipt[k] for k in
                                                         ("remote", "synced_at", "sha256")},
               hold={k: hold[k] for k in ("synced_at", "released_at", "held_s")},
               ground_truth_controls=sorted({c[1] for c in ctrl}),
               synced_at=synced.isoformat(), passed=True)
    _record(out, rec)
