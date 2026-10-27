#!/usr/bin/env python3
"""The steps of the P1 session (T3.1), one subcommand each.

`scripts/t31_session.py` runs these in order, each in its own process with
its own log, and stops at the first failure. Each step writes `result.json`
in its directory, as the T2.27 steps do:

    {"step", "status": "passed" | "failed" | "skipped", "reason",
     "dry_run", "findings": {...}}

and exits 0 (passed), 1 (failed) or 3 (skipped).

Steps, in the runner's order (plan Section 5.3):

  preflight          T2.1 host-speed probe `start`, T2.20 fields (T2.27's step)
  link               T2.5 link row and T2.6 Q (T2.27's steps), turned into the
                     recursion's mechanism table per context length
  dense_select       T2.16 per (length, batch) cell, recorded before any timed block
  probe              T2.18 structural probe, every arm (durations discarded)
  host_components    T2.2 null-kernel profile, every arm and cell
  device_components  T2.4 replay, every arm, density and cell; the priced cells
  predict            T2.11 record, write, sync and read back: the receipt
  hold               until the instance clock passes sync + 60 s + guard
  ground_truth       T2.13, one process per compile-setting group
  verify             the analysis's own late-prediction check, in session
  postflight         T2.1 host-speed probe `end`

**Nothing clocked runs before `hold` has passed.** The component steps time
microbenchmarks, kernels and null-kernel passes, never the composed
workload (plan Section 5.1). The dense selection times dense attention
kernels in its own warm-up block, which is what T2.16 records before the
timed runs. `ground-truth-one` refuses to start unless the instance clock is
past the release (`session_order.ready_to_clock`), and stamps
`clock_not_before` before its first forward of any kind.

**What is not built.** On a GPU, `probe`, `host_components`,
`device_components` and `ground_truth` refuse, naming the missing piece of
their Phase 2 task (`NOT_BUILT`); `t31_session.py --list` prints them. A
CPU dry run replaces those four with fakes: synthetic priced cells, and a
dense-only ground truth on the two-layer random model through the real
`ground_truth.time_arms`. Everything from `predict` on is the real code in
both modes; the dry run syncs to a local store with the same no-clobber,
read-back and creation-time behaviour as `GsutilStore`. A real run syncs to
the results bucket, found as `run_phase.sh` finds it, under
`prediction_prefix/<session>`; the receipt holds bucket-relative paths, so
nothing banked names the project.

Usage, from the repo root:
    python3 scripts/t31_steps.py STEP --out DIR --session DIR [--dry-run] [--config F]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"
EXIT = {PASSED: 0, FAILED: 1, SKIPPED: 3}

STEPS = ("preflight", "link", "dense_select", "probe", "host_components",
         "device_components", "predict", "hold", "ground_truth", "verify", "postflight")

DEFAULT_CONFIG = ROOT / "configs" / "t31" / "p1_session.json"

# Values a real session must have; null in the file means not decided yet.
REQUIRED = ("densities", "reps", "max_rerun_rounds", "resolution_floor_ms",
            "dense_rounds")

# A dry run's own values: small, and labelled so nothing it writes can pass
# for P1's.
DRY_CONFIG = {
    "pairing": "P1-dryrun", "card": "cpu", "model": "dryrun",
    "lengths": [64, 128], "batches": {},
    "arms": ["dense", "fake_sparse"],
    "densities": {"dense": [1.0], "fake_sparse": [0.25, 0.5]},
    "gt_groups": {"eager": ["dense"]},
    "reps": 3, "warmup": 2, "max_rerun_rounds": 1,
    "resolution_floor_ms": 1e9, "dense_rounds": 2,
    "flag_reasons": ["hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown",
                     "hw_power_brake_slowdown"],
    "sampler_interval_s": None,
    "clock_skew_s": 2, "hold_guard_s": 1,
    "prediction_prefix": "predictions/P1-dryrun",
}

# What each refusing step still needs (plan Section 8 rows), so a GPU run
# stops at the first of them with the reason and `--list` can print them.
NOT_BUILT = {
    "probe": "T2.18: three probe lengths with at least two chunks each for a chunked "
             "arm, and the full-length structure recording where the result is "
             "'different' (XAttention at 32768)",
    "host_components": "T2.2: the per-cell arm runner (run_prefill, tape, recording) at "
                       "every length, batch and density, and the L4 smoke run; a "
                       "compiled arm's graph launch is refused (T2.21)",
    "device_components": "T2.4: pricing the positions the tape substitutes "
                         "(device_costs refuses an incomplete match), and the "
                         "conversion of every arm, density and cell to priced "
                         "operations",
    "ground_truth": "T2.13: the cell join (one wrapped model serving every arm's score "
                    "source, batch-4 inputs, the density set, compiled and precomputed "
                    "arms) and the real NVML reader on a card",
}


class Skip(Exception):
    pass


class StepFailed(Exception):
    def __init__(self, reason: str, findings: dict | None = None):
        super().__init__(reason)
        self.findings = findings or {}


# --- shared ----------------------------------------------------------------------

def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _py(script: str, *argv: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *argv]


def _run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def step_dir(session: Path, step: str) -> Path:
    return Path(session) / f"{STEPS.index(step) + 1:02d}_{step}"


def load_config(path, *, dry_run: bool) -> dict:
    """The session's values. A real run refuses any undecided one."""
    if path is None and dry_run:
        return dict(DRY_CONFIG)
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if not dry_run:
        missing = [k for k in REQUIRED if cfg.get(k) is None]
        missing += [f"gt_groups.{g}" for g, arms in (cfg.get("gt_groups") or {}).items()
                    if arms is None]
        if missing:
            raise StepFailed(f"the session config leaves {missing} undecided")
    return cfg


def cells(cfg: dict):
    """(arm, density, length, batch) for every ground-truth cell."""
    out = []
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            for arm in cfg["arms"]:
                for d in cfg["densities"][arm]:
                    out.append((arm, float(d), int(L), int(b)))
    return out


def _skew(cfg) -> tuple[timedelta, timedelta]:
    return timedelta(seconds=cfg["clock_skew_s"]), timedelta(seconds=cfg["hold_guard_s"])


def _not_built(args, step: str):
    if not args.dry_run:
        raise StepFailed(f"not built: {NOT_BUILT[step]}")


# --- stores --------------------------------------------------------------------

class LocalStore:
    """A directory standing in for the bucket in a dry run: `put` never
    clobbers, and `created` is the file's modification time plus one second,
    an upper bound as `GsutilStore.created` is."""

    BASE = "local:"

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, remote: str) -> Path:
        if not remote.startswith(self.BASE + "/"):
            raise ValueError(f"not a {self.BASE}/ URI: {remote!r}")
        return self.root / remote[len(self.BASE) + 1:]

    def put(self, local, remote):
        p = self._path(remote)
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, p)

    def get(self, remote):
        return self._path(remote).read_bytes()

    def created(self, remote):
        st = self._path(remote).stat()
        return datetime.fromtimestamp(st.st_mtime, timezone.utc) + timedelta(seconds=1)


METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/project/project-id"


def results_bucket() -> str:
    """The results bucket, found as `run_phase.sh` finds it: `ATTNBENCH_BUCKET`,
    else named after the instance's project from the metadata server. The
    project ID is never written in the repository, so nothing banked carries
    the bucket's name: receipts hold paths relative to it."""
    b = os.environ.get("ATTNBENCH_BUCKET")
    if b:
        return b.rstrip("/")
    import urllib.request
    req = urllib.request.Request(os.environ.get("ATTNBENCH_METADATA_URL", METADATA_URL),
                                 headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            project = r.read().decode().strip()
    except OSError as e:
        raise StepFailed(f"no results bucket: ATTNBENCH_BUCKET is unset and the "
                         f"metadata server did not answer ({type(e).__name__})")
    if not project:
        raise StepFailed("no results bucket: the metadata server gave no project")
    return f"gs://attnbench-results-{project}"


def make_store(session: Path, *, dry_run: bool):
    """(store, base): a local directory in a dry run, the results bucket
    through gsutil in a real one."""
    from attnbench.predict.commit import GsutilStore
    if dry_run:
        return LocalStore(Path(session) / "_store"), LocalStore.BASE
    return GsutilStore(), results_bucket()


def prediction_dir(cfg, session: Path) -> str:
    """Where this session's prediction goes, relative to the store's base."""
    prefix = (cfg.get("prediction_prefix") or "").strip("/")
    if not prefix:
        raise StepFailed("the session config names no prediction_prefix")
    return f"{prefix}/{Path(session).name}"


# --- operations on disk ------------------------------------------------------------

def op_to_dict(op) -> dict:
    return {"kind": op.kind.value, "host": op.host, "device": op.device, "link": op.link,
            "mechanism": op.mechanism, "payload_bytes": op.payload_bytes,
            "loaded": op.loaded, "layer": op.layer, "stream": op.stream, "label": op.label}


def op_from_dict(d: dict):
    from attnbench.predict.recursion import OpKind, Operation
    return Operation(OpKind(d["kind"]), host=d["host"], device=d["device"], link=d["link"],
                     mechanism=d["mechanism"], payload_bytes=d["payload_bytes"],
                     loaded=d["loaded"], layer=d["layer"], stream=d["stream"],
                     label=d["label"])


# --- fakes for the CPU dry run ---------------------------------------------------------

FAKE_MECHANISMS = {
    "synchronize": {"s_fixed": 1e-5},
    "item": {"s_fixed": 2e-5, "bandwidth": 1e9, "loaded_bandwidth": 5e8},
    "bool": {"s_fixed": 2e-5, "bandwidth": 1e9, "loaded_bandwidth": 5e8},
    "copy_to_host_pinned": {"s_fixed": 1e-5, "bandwidth": 2e10, "loaded_bandwidth": 1e10},
    "copy_to_host_pageable": {"s_fixed": 3e-5, "bandwidth": 5e9, "loaded_bandwidth": 2e9},
}


def fake_ops(arm: str, density: float, length: int, batch: int) -> list:
    """Two layers of kernels; a sparse arm adds a host mask build and an
    `item` wait per layer. Seconds scale with length, batch and density."""
    from attnbench.predict.recursion import copy_op, kernel_op, sync_op
    scale = length * batch / 1024
    ops = []
    for layer in range(2):
        if arm != "dense":
            ops.append(kernel_op(2e-5, 1e-4 * scale, layer=layer, label="score"))
            ops.append(sync_op("item", payload_bytes=4, layer=layer))
            ops.append(copy_op(1e-5, 5e-6 * scale, layer=layer, payload_bytes=4096))
        ops.append(kernel_op(1e-5, 1e-3 * scale * density, layer=layer, label="attention"))
        ops.append(kernel_op(1e-5, 5e-4 * scale, layer=layer, label="mlp"))
    return ops


FAKE_CODE_PATH = {"gpu": "cpu", "implementation": "dryrun-fake",
                  "libraries": [["attnbench", "dryrun"]], "python_optimize": 0}


class QuietCard:
    """A dry run's clock reader: counters present, nothing ever throttled."""
    observes_transients = True

    def read(self):
        from attnbench.clock_sampling import ClockSample
        return ClockSample(0, 0, 0, {"power": 0, "thermal": 0})


# --- steps -------------------------------------------------------------------------

def _t227(args, step: str, out: Path) -> None:
    cmd = _py("t227_steps.py", step, "--out", str(out), *(["--dry-run"] if args.dry_run else []))
    rc = _run(cmd)
    if rc != 0:
        raise StepFailed(f"T2.27's `{step}` step exited {rc}; see {out / 'result.json'}")


def step_preflight(args, cfg) -> dict:
    """The session config of record (copied, with its hash), then T2.27's
    `t21_t220`: the host-speed probe `start` and every T2.20 field, which
    refuses a dirty tree outside a dry run."""
    import hashlib
    out = Path(args.out)
    data = json.dumps(cfg, indent=1, sort_keys=True).encode()
    (out / "session_config.json").write_bytes(data)
    _t227(args, "t21_t220", out / "t21_t220")
    return {"config_sha256": hashlib.sha256(data).hexdigest(), "cells": len(cells(cfg))}


def step_link(args, cfg) -> dict:
    """T2.27's `link` and `launch_queue`, then the recursion's inputs: the
    mechanism table at each length (`link.mechanism_costs`, loaded figures
    included), the learned baseline's bandwidths and Q (the null kernel's)."""
    out = Path(args.out)
    _t227(args, "link", out / "link")
    _t227(args, "launch_queue", out / "launch_queue")
    if args.dry_run:
        inputs = {"queue_depth": 8, "fake": True,
                  "hardware": {"pinned_bandwidth": 2e10, "pageable_bandwidth": 5e9},
                  "by_length": {str(L): FAKE_MECHANISMS for L in cfg["lengths"]}}
    else:
        import pandas as pd
        from attnbench.components import launch_queue, link
        lrow = pd.read_parquet(out / "link" / "link.parquet").iloc[0].to_dict()
        qrow = pd.read_parquet(out / "launch_queue" / "launch_queue.parquet").iloc[0].to_dict()
        hw = link.hardware_features(lrow)
        inputs = {"queue_depth": int(launch_queue.queue_depth(qrow, "sleep0")), "fake": False,
                  "hardware": {"pinned_bandwidth": hw.pinned_bandwidth,
                               "pageable_bandwidth": hw.pageable_bandwidth},
                  "by_length": {str(L): {m: {"s_fixed": c.s_fixed, "bandwidth": c.bandwidth,
                                             "loaded_bandwidth": c.loaded_bandwidth}
                                         for m, c in link.mechanism_costs(lrow, L).items()}
                                for L in cfg["lengths"]}}
    _write_json(out / "recursion_inputs.json", inputs)
    return {"queue_depth": inputs["queue_depth"], "fake": inputs["fake"]}


def step_dense_select(args, cfg) -> dict:
    """T2.16 for every (length, batch) cell: the record, with its sidecar,
    before any timed block. A dry run records nothing and says so; the
    ground-truth step's `check_control` call is tested with fakes."""
    if args.dry_run:
        return {"dry_run": "no dense selection on a CPU; check_control is not called"}
    from dataclasses import replace
    from attnbench import dense_selection as ds
    from attnbench.components.shakedown_arms import ARMS, Harness
    h = Harness(device="cuda", dry_run=False)
    found = {}
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            cell = ds.Cell(card=cfg["card"], context_length=L, batch=b)
            attn_cfg = replace(h._cfg(ARMS["dense"], L), batch=b)
            choice = ds.select_dense_baseline(
                cell, attn_cfg, ds.registry_candidates(cfg["card"]),
                rounds=cfg["dense_rounds"], resolution_floor_ms=cfg["resolution_floor_ms"])
            path = ds.write_choice(choice, Path(args.out))
            found[cell.key] = {"chosen": choice.chosen, "complete": choice.complete,
                               "file": path.name}
            if not choice.complete:
                raise StepFailed(f"{cell.key}: the dense selection is incomplete", found)
    return found


def step_probe(args, cfg) -> dict:
    """T2.18 for every arm whose code path has not been traced (all, on P1)."""
    _not_built(args, "probe")
    return {"fake": True, "arms": cfg["arms"]}


def step_host_components(args, cfg) -> dict:
    """T2.2's null pass per arm and cell."""
    _not_built(args, "host_components")
    return {"fake": True}


def step_device_components(args, cfg) -> dict:
    """T2.4's replay per arm, density and cell, and the priced cells the
    prediction reads: `cells.json`, one entry per ground-truth cell."""
    _not_built(args, "device_components")
    entries = [{"arm": a, "density": d, "context_length": L, "batch": b,
                "code_path": FAKE_CODE_PATH,
                "ops": [op_to_dict(op) for op in fake_ops(a, d, L, b)]}
               for a, d, L, b in cells(cfg)]
    _write_json(Path(args.out) / "cells.json", {"fake": True, "cells": entries})
    return {"fake": True, "cells": len(entries)}


def build_prediction(cfg, session: Path, *, dry_run: bool, now=None):
    """The T2.11 record from the link and device-component steps' outputs."""
    from attnbench.predict import commit as C
    from attnbench.predict.recursion import MechanismCost, close_prefill, price
    from attnbench.predict.training import Cell, HardwareFeatures

    rin = json.loads((step_dir(session, "link") / "recursion_inputs.json").read_text())
    dev = json.loads((step_dir(session, "device_components") / "cells.json").read_text())
    if dev["fake"] != dry_run or rin["fake"] != dry_run:
        raise StepFailed("fake components in a real session, or real ones in a dry run")
    hw = HardwareFeatures(**rin["hardware"])
    code_paths, out = {}, []
    for e in dev["cells"]:
        label = C.code_path_label(e["code_path"])
        code_paths[label] = e["code_path"]
        mech = {m: MechanismCost(**v) for m, v in rin["by_length"][str(e["context_length"])].items()}
        priced = price(close_prefill(op_from_dict(o) for o in e["ops"]), mech)
        out.append(Cell(pairing=cfg["pairing"], model_name=cfg["model"], code_path=label,
                        arm=e["arm"], context_length=e["context_length"],
                        density=e["density"], batch=e["batch"], priced=priced, hardware=hw))
    want = {(a, d, L, b) for a, d, L, b in cells(cfg)}
    got = {(c.arm, c.density, c.context_length, c.batch) for c in out}
    if want != got:
        raise StepFailed(f"priced cells differ from the grid: missing {sorted(want - got)}, "
                         f"extra {sorted(got - want)}")
    code = C.CodeStamp("0" * 40, False) if dry_run else C.current_code_stamp()
    return C.build_record(cfg["pairing"], out, rin["queue_depth"], code_paths=code_paths,
                          banked=[], fitted_sum=None, learned=None, code=code,
                          written_at=now or datetime.now(timezone.utc))


def step_predict(args, cfg) -> dict:
    """Plan Section 5.3 step 6: write the file and its sha256, sync both, read
    them back. The receipt is what `hold` and the ground truth read. P1 has
    no banked pairing, so no fitted baseline (commit.NOT_FITTED_REASON)."""
    from attnbench.predict import commit as C
    session, out = Path(args.session), Path(args.out)
    record = build_prediction(cfg, session, dry_run=args.dry_run)
    committed = C.write_prediction(record, out)
    store, base = make_store(session, dry_run=args.dry_run)
    receipt = C.sync_prediction(committed, store, f"{base}/{prediction_dir(cfg, session)}")
    d = receipt.to_dict()
    for k in ("remote", "remote_sidecar"):          # relative to the bucket (results_bucket)
        if not d[k].startswith(base + "/"):
            raise StepFailed(f"{d[k]} is not under the store's base")
        d[k] = d[k][len(base) + 1:]
    _write_json(out / "receipt.json", {**d, "local": committed.path.name})
    return {"file": committed.path.name, "sha256": committed.sha256,
            "synced_at": d["synced_at"], "remote_in_bucket": d["remote"]}


def _receipt(session: Path):
    from attnbench.predict.session_order import receipt_from_dict
    p = step_dir(session, "predict") / "receipt.json"
    if not p.exists():
        raise StepFailed(f"no receipt at {p}: the prediction has not been synced")
    d = json.loads(p.read_text())
    return receipt_from_dict(d), step_dir(session, "predict") / d["local"]


def step_hold(args, cfg) -> dict:
    """Wait until the instance clock is past synced_at + skew + guard."""
    from attnbench.predict import session_order as so
    skew, guard = _skew(cfg)
    so.check_session_skew(skew, guard, dry_run=args.dry_run)
    receipt, _ = _receipt(Path(args.session))
    t0 = so.utc_now()
    released = so.hold(receipt.synced_at, clock_skew=skew, guard=guard)
    return {"synced_at": receipt.synced_at.isoformat(), "released_at": released.isoformat(),
            "held_s": round((released - t0).total_seconds(), 3),
            "skew_s": skew.total_seconds(), "guard_s": guard.total_seconds()}


def step_ground_truth(args, cfg) -> dict:
    """T2.13, one process per compile-setting group (Appendix A, 27 Oct 2026),
    each with its own dense control."""
    _not_built(args, "ground_truth")
    found = {}
    for group in cfg["gt_groups"]:
        d = Path(args.out) / group
        cmd = _py("t31_steps.py", "ground-truth-one", "--group", group, "--out", str(d),
                  "--session", str(args.session), *(["--dry-run"] if args.dry_run else []),
                  *(["--config", args.config] if args.config else []))
        if _run(cmd):
            raise StepFailed(f"ground truth for group {group} failed; see {d}", found)
        found[group] = json.loads((d / "stamps.json").read_text())
    return found


def _sampler(cfg):
    from attnbench.clock_sampling import REASON_NAMES, BlockSampler
    bits = {v: k for k, v in REASON_NAMES.items()}
    flag = 0
    for name in cfg["flag_reasons"]:
        flag |= bits[name]
    return BlockSampler(QuietCard(), flag_bits=flag, interval_s=cfg["sampler_interval_s"])


def cmd_ground_truth_one(args) -> int:
    """One group's arms at every cell, in this process only. Refuses unless
    the prediction's hold has passed by this process's own clock."""
    from attnbench import ground_truth as gt
    from attnbench.predict import session_order as so
    cfg = load_config(args.config, dry_run=args.dry_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skew, guard = _skew(cfg)
    so.check_session_skew(skew, guard, dry_run=args.dry_run)
    receipt, _ = _receipt(Path(args.session))
    so.ready_to_clock(receipt.synced_at, so.utc_now(), clock_skew=skew, guard=guard)
    not_before = time.time()          # before this process's first forward of any kind
    if not args.dry_run:
        raise StepFailed(f"not built: {NOT_BUILT['ground_truth']}")
    from attnbench.components.shakedown_arms import Harness
    h = Harness(device="cpu", dry_run=True, score_cache_dir=str(out / "score_cache"))
    arms_in_group = cfg["gt_groups"][args.group]
    controls = gt.Controls(resolution_floor_ms=cfg["resolution_floor_ms"],
                           max_rerun_rounds=cfg["max_rerun_rounds"], sampler=_sampler(cfg))
    first = []
    for L in cfg["lengths"]:
        pa = h.prepare("dense", L)
        arm = gt.Arm("dense", gt.DENSE, pa.backend, pa.extra["cfg"])
        res = gt.time_arms(pa.extra["wrapped"], pa.extra["input_ids"], [arm],
                           reps=cfg["reps"], controls=controls, warmup=cfg["warmup"])
        first.append(res["first_timed_block_started_at"])
        for row in res["rows"]:
            row.update(model=cfg["model"], code_path=None, density=1.0,
                       context_length=L, batch=1, group=args.group,
                       session_receipt_sha256=receipt.sha256)
        _write_json(out / f"gt_L{L}.json", res)
    stamps = {"group": args.group, "arms": arms_in_group, "clock_not_before": not_before,
              "first_timed_block_started_at": first,
              "synced_at": receipt.synced_at.isoformat()}
    _write_json(out / "stamps.json", stamps)
    print(f"[ground-truth-one {args.group}] {len(first)} cells", flush=True)
    return 0


def step_verify(args, cfg) -> dict:
    """The analysis's late-prediction check (`commit.load_committed`), run in
    the session against the store's copy, with the earliest stamp of any
    ground-truth process as the start."""
    from attnbench.predict import commit as C
    from attnbench.predict import session_order as so
    session = Path(args.session)
    receipt, local = _receipt(session)
    gt_dir = step_dir(session, "ground_truth")
    stamps = [json.loads(p.read_text()) for p in sorted(gt_dir.glob("*/stamps.json"))]
    if not stamps:
        raise StepFailed(f"no ground-truth stamps under {gt_dir}")
    if any(s["synced_at"] != receipt.synced_at.isoformat() for s in stamps):
        raise StepFailed("a ground-truth group read another prediction's receipt")
    start = so.session_start([s["clock_not_before"] for s in stamps],
                             [t for s in stamps for t in s["first_timed_block_started_at"]])
    store, base = make_store(session, dry_run=args.dry_run)
    skew, _guard = _skew(cfg)
    try:
        cp = C.load_committed(store, f"{base}/{receipt.remote}", ground_truth_started_at=start,
                              clock_skew=skew, local=local)
    except C.PredictionCommitError as e:
        raise StepFailed(f"{type(e).__name__}: {e}")
    return {"start": start.isoformat(), "synced_at": cp.synced_at.isoformat(),
            "margin_s": round((start - cp.synced_at).total_seconds(), 3),
            "sha256": cp.sha256}


def step_postflight(args, cfg) -> dict:
    """T2.1's host-speed probe, tag `end`."""
    out = Path(args.out)
    cmd = _py("host_cpu_probe.py", "speed", "--out", str(out), "--tag", "end")
    if not args.dry_run:
        cmd.append("--require-gpu")
    if _run(cmd):
        raise StepFailed("host_cpu_probe.py speed failed (T2.1)")
    return {"row": "host_speed_end.parquet"}


# --- entry point ---------------------------------------------------------------------

STEP_FUNCS = {
    "preflight": step_preflight, "link": step_link, "dense_select": step_dense_select,
    "probe": step_probe, "host_components": step_host_components,
    "device_components": step_device_components, "predict": step_predict,
    "hold": step_hold, "ground_truth": step_ground_truth, "verify": step_verify,
    "postflight": step_postflight,
}
if tuple(STEP_FUNCS) != STEPS:     # not `assert`: T2.17 keeps the harness free of them
    raise RuntimeError("STEP_FUNCS and STEPS disagree")


def run_step(name: str, args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        cfg = load_config(args.config, dry_run=args.dry_run)
        status, reason, findings = PASSED, "", STEP_FUNCS[name](args, cfg)
    except Skip as e:
        status, reason, findings = SKIPPED, str(e), {}
    except StepFailed as e:
        status, reason, findings = FAILED, str(e), e.findings
    except Exception as e:
        traceback.print_exc()
        status, reason, findings = FAILED, f"{type(e).__name__}: {e}", {}
    _write_json(out / "result.json", {
        "step": name, "status": status, "reason": reason, "dry_run": args.dry_run,
        "seconds": round(time.time() - started, 1), "findings": findings})
    print(f"[{name}] {status.upper()}{': ' + reason if reason else ''}", flush=True)
    return EXIT[status]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS + ("ground-truth-one",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None,
                    help=f"session values (default {DEFAULT_CONFIG.relative_to(ROOT)}; "
                         f"a dry run uses its own unless given)")
    ap.add_argument("--group", default=None)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.step == "ground-truth-one":
        try:
            return cmd_ground_truth_one(args)
        except Exception as e:
            traceback.print_exc()
            print(f"[ground-truth-one] FAILED: {type(e).__name__}: {e}", flush=True)
            return 1
    return run_step(args.step, args)


if __name__ == "__main__":
    raise SystemExit(main())
