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
  hold               until the instance clock passes sync + PLANNED_SYNC_WAIT_S
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

**Wired 31 Oct 2026, not yet run on a GPU.** `probe` (T2.18 against P1's
own full recordings), `host_components` (a recording and the T2.2 null
pass per cell), `device_components` (the T2.4 replay per cell, priced with
the host, device and link costs) and `ground_truth` (each group's arms at
every cell, the K2 choice as the control, compiled under the group's one
setting) take the working values of `docs/lock_value_table.md`, marked
"working, pre-lock". The floors and the rerun budget stay null: the session
floor comes from `dense_select`'s pre-pass and nothing is rerun. A
CPU dry run replaces those four with fakes: synthetic priced cells, and a
dense-only ground truth on the two-layer random model through the real
`ground_truth.time_arms`. Everything from `predict` on is the real code in
both modes; the dry run syncs to a local store with the same no-clobber,
read-back and creation-time behaviour as `GsutilStore`. A real run syncs to
the results bucket, found as `run_phase.sh` finds it, under
`prediction_prefix/<session>`; the receipt holds bucket-relative paths, so
nothing banked names the project.

**The CPU boundary** (`--cpu-boundary`, 31 Oct 2026). Shakedown 3 stopped
on wiring defects the fakes could not show, so this mode runs steps 1 to 8
with the real arms, backends, recorder, null pass, replay and pricing on
the two-layer random model, behind `attnbench.cpu_boundary` (the
block-sparse kernel, Triton, the CUDA device and the profiler's device
activity stubbed, nothing else). The GPU measurements take their dry-run
rows through the real parsing (`link`, `launch_queue`), the replay is
eager (`replay.EagerBackend`), the null kernel is a no-op, and the store,
skew and code stamp are a dry run's: nothing leaves the machine. Its
config is `CPU_BOUNDARY_CONFIG` with the working densities and groups.

Usage, from the repo root:
    python3 scripts/t31_steps.py STEP --out DIR --session DIR [--dry-run | --cpu-boundary]
        [--config F]
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

from attnbench.predict import session_order as so  # noqa: E402
from attnbench.predict.commit import (PLANNED_SYNC_WAIT_S,  # noqa: E402,F401
                                      REQUIRED_SYNC_GAP_S, SYNC_WAIT_MARGIN_S)

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"
EXIT = {PASSED: 0, FAILED: 1, SKIPPED: 3}

STEPS = ("preflight", "link", "dense_select", "probe", "host_components",
         "device_components", "predict", "hold", "ground_truth", "verify", "postflight")

DEFAULT_CONFIG = ROOT / "configs" / "t31" / "p1_session.json"

# Values a real session must have once the working values are in; null in the
# file means not decided yet. The floors and the rerun budget are not here:
# they stay null as inputs (researcher's decision, 31 Oct 2026). T3.1 derives
# a session floor from its own dense pre-pass (`dense_select`) and reruns
# nothing until the budget is decided after T3.2.
REQUIRED = ("densities", "reps", "dense_rounds")

# The working values T3.1 takes for what T4.1 has not fixed (one source: the
# fenced block in this file). Marked "working, pre-lock"; P1 only.
WORKING_TABLE = ROOT / "docs" / "lock_value_table.md"
WORKING_FENCE = "```json t31-working-values"
WORKING_KEYS = ("densities", "gt_groups", "reps", "dense_rounds", "sampler_interval_s",
                "chain_check_tolerance")
WORKING_STATUS = "working, pre-lock"
CALIBRATED = "calibrated"          # a density the estimator sets (XAttention), not the cell
# Arms that run at batch 1 only: XAttention's kernel call (T2.12), and the
# oracle's scoring pass and the mean-pool estimator, which refuse a batch
# above one as built (found by the CPU-boundary rehearsal, 31 Oct 2026). The
# batch-4 cells hold the dense control only: accepted as a default (ruling,
# 1 Nov 2026) on the condition that the plan promises no sparse arm above
# batch 1, and it does (Section 5.4 "Batch", H6 and Section 5.9 choose a
# sparse arm per batch), so the conflict is flagged and open; no scope line
# is written until it is decided.
BATCH1_ONLY = ("xattention", "xattention_noassert", "oracle", "meanpool")
HOST_PROFILE_REPS = 5

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

# The CPU boundary's own values: every arm at the working densities on a
# short grid, a batch above one, and the dry run's skew (31 Oct 2026). It is
# a rehearsal of P1 (the floors and working values are P1's only), with its
# own prefix, a local store and a zero code stamp, as a card rehearsal is.
CPU_BOUNDARY_CONFIG = {
    "pairing": "P1", "rehearsal": True, "cpu_boundary": True,
    "card": "cpu", "model": "tiny-qwen2-random",
    "lengths": [2048, 4096], "batches": {"4096": [1, 2]},
    "arms": ["dense", "oracle", "meanpool", "xattention", "xattention_noassert"],
    "densities": None, "gt_groups": None,
    "reps": 3, "warmup": 2, "max_rerun_rounds": 0,
    "resolution_floor_ms": None, "dense_rounds": 2,
    "flag_reasons": DRY_CONFIG["flag_reasons"],
    "sampler_interval_s": None,
    "clock_skew_s": 2, "hold_guard_s": 1,
    "prediction_prefix": "predictions/P1-cpu-boundary-rehearsal",
}

# What each refusing step still needs (plan Section 8 rows), so a GPU run
# stops at the first of them with the reason and `--list` can print them.
NOT_BUILT: dict = {}   # every step is wired (31 Oct 2026); none has run on a GPU


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


def working_values(path=None) -> tuple[dict, str]:
    """(values, sha256 of the block) from the table file's fenced block."""
    import hashlib
    import re
    text = Path(path or WORKING_TABLE).read_text(encoding="utf-8")   # read at call time
    m = re.search(re.escape(WORKING_FENCE) + r"\n(.*?)\n```", text, re.S)
    if not m:
        raise StepFailed(f"{WORKING_TABLE.name} has no {WORKING_FENCE!r} block")
    raw = m.group(1)
    vals = json.loads(raw)
    if vals.get("status") != WORKING_STATUS:
        raise StepFailed(f"the working values say {vals.get('status')!r}, not {WORKING_STATUS!r}")
    return vals, hashlib.sha256(raw.encode()).hexdigest()


def fill_working_values(cfg: dict, path=None) -> dict:
    """Fill each null working key from the table, and record that it was.
    Refused for any pairing but the table's (P1): these values must never
    come from, or go to, P2 or P4."""
    vals, sha = working_values(path)
    cfg = dict(cfg)
    filled = []
    for k in WORKING_KEYS:
        if k == "gt_groups":
            groups = dict(cfg.get(k) or {})
            for g, arms in vals[k].items():
                if groups.get(g) is None:
                    groups[g] = list(arms)
                    filled.append(f"gt_groups.{g}")
            cfg[k] = groups
        elif cfg.get(k) is None:
            cfg[k] = vals[k]
            filled.append(k)
    if filled:
        if cfg.get("pairing") != vals["pairing"]:
            raise StepFailed(f"the working values are {vals['pairing']}'s only; this session "
                             f"is {cfg.get('pairing')!r}")
        src = Path(path or WORKING_TABLE).resolve()
        src = src.relative_to(ROOT) if src.is_relative_to(ROOT) else src
        cfg["working_values"] = {"source": str(src), "sha256": sha,
                                 "status": WORKING_STATUS, "filled": filled}
    return cfg


def cpu_boundary_config(path=None) -> dict:
    """`CPU_BOUNDARY_CONFIG` with the working table's densities and groups,
    labelled as taken from it."""
    vals, sha = working_values(path)
    return dict(CPU_BOUNDARY_CONFIG, densities=vals["densities"], gt_groups=vals["gt_groups"],
                chain_check_tolerance=vals["chain_check_tolerance"],
                working_values={"sha256": sha, "status": WORKING_STATUS,
                                "filled": ["densities", "gt_groups", "chain_check_tolerance"]})


def load_config(path, *, dry_run: bool, cpu_boundary: bool = False) -> dict:
    """The session's values: the file, with its null working values filled
    from the table. A real run refuses any value still undecided."""
    if path is None and dry_run:
        return dict(DRY_CONFIG)
    if path is None and cpu_boundary:
        return cpu_boundary_config()
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if not dry_run:
        cfg = fill_working_values(cfg)
        missing = [k for k in REQUIRED if cfg.get(k) is None]
        missing += [f"gt_groups.{g}" for g, arms in (cfg.get("gt_groups") or {}).items()
                    if arms is None]
        if missing:
            raise StepFailed(f"the session config leaves {missing} undecided")
    return cfg


def nominal_density(d):
    """A cell's density setting: a float, or None where the estimator sets it."""
    return None if d == CALIBRATED else float(d)


def cells(cfg: dict):
    """(arm, density, length, batch) for every ground-truth cell; density is
    None for an arm whose estimator sets it. The XAttention arms run at batch
    1 only (the kernel call's limit, T2.12)."""
    out = []
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            for arm in cfg["arms"]:
                if b > 1 and arm in BATCH1_ONLY:
                    continue
                for d in cfg["densities"][arm]:
                    out.append((arm, nominal_density(d), int(L), int(b)))
    return out


def density_setting(arm: str, density):
    """What `Harness.ready` takes for a cell's density: none for dense, whose
    1.0 names the cell and is not a setting (Shakedown 3, step 5)."""
    return None if arm == "dense" else density


def ready_cell(h, arm: str, density, length: int, batch: int):
    """The arm prepared and warmed at one ground-truth cell."""
    return h.ready(arm, length, density=density_setting(arm, density), batch=batch)


def cell_dir(base: Path, arm: str, density, length: int, batch: int) -> Path:
    d = "cal" if density is None else f"{density:g}"
    return Path(base) / f"{arm}_d{d}_L{length}_b{batch}"


def _boundary() -> bool:
    """Running behind the stubbed GPU boundary (`--cpu-boundary`)."""
    from attnbench import cpu_boundary
    return cpu_boundary.active()


def _device() -> str:
    return "cpu" if _boundary() else "cuda"


def _local(args) -> bool:
    """A dry run's store, skew and code stamp: a dry run, or the CPU boundary."""
    return bool(args.dry_run or getattr(args, "cpu_boundary", False))


def _need_gpu(step: str) -> None:
    if not (_cuda() or _boundary()):
        raise StepFailed(f"{step} needs a CUDA device")


def _harness(args):
    from attnbench.components.shakedown_arms import Harness
    session = getattr(args, "session", None)
    return Harness(device=_device(), dry_run=_boundary(),
                   score_cache_dir=str(Path(session) / "score_cache") if session else None)


def _graph_backend():
    """T2.4's CUDA graph; behind the CPU boundary, its stand-in, whose capture
    writes nothing (`cpu_boundary.GraphLikeBackend`)."""
    from attnbench.components import replay
    if _boundary():
        from attnbench.cpu_boundary import GraphLikeBackend
        return GraphLikeBackend()
    return replay.CudaGraphBackend()


def _null_device():
    """T2.2's null kernel; behind the CPU boundary, a launch that does nothing."""
    from attnbench.components import host_profile as hp
    if _boundary():
        return hp.NullDevice(launch=lambda: None, device_type="cpu")
    return hp.cuda_device()


def host_profile_cell(a, rec, out: Path, *, reps: int, tolerance: dict, entries=()):
    """T2.2's null-kernel profile of one recording, and the chain check.

    The isolated replay of the original operators prices the positions the
    ruling of 31 Oct 2026 gives it (`host_profile.ISOLATED_OUTSIDE_LAYERS`;
    behind the CPU boundary its embedding kernel,
    `cpu_boundary.EMBEDDING_KERNEL_OP`) and any composite whose chain is not
    recorded, and is what `chain_check` holds the calls met at a composite or
    by their device argument against (ruling of 1 Nov 2026), within
    `tolerance` ({"rel", "abs_us"}, T4.1 row 32). Returns (profile, check)
    and writes the check to `chain_check.json`."""
    from attnbench import cpu_boundary
    from attnbench.components import host_profile as hp
    from attnbench.components import replay
    ops = hp.ISOLATED_OUTSIDE_LAYERS | (
        {cpu_boundary.EMBEDDING_KERNEL_OP} if _boundary() else set())
    pos = hp.isolated_positions(rec.sequence, ops)
    trace = replay.isolated_replay(a.run_prefill, rec.tape, Path(out) / "isolated_host.json",
                                   device=_device(), masks=rec.masks, substitute=False)
    iso = hp.isolated_host_s(trace, rec.sequence, pos)
    prof = hp.profile(a.run_prefill, rec.tape, rec.sequence, device=_null_device(), reps=reps,
                      entries=entries, layers=a.layers, isolated=iso, isolated_ops=ops)
    check = hp.chain_check(trace, rec.sequence, prof, rel_tol=float(tolerance["rel"]),
                           abs_tol_s=float(tolerance["abs_us"]) * 1e-6)
    check["gated"] = not _boundary()
    if not check["gated"]:
        check["why_not_gated"] = ("behind the CPU boundary a kernel runs inside its call, so "
                                  "the direct time holds the compute")
    _write_json(Path(out) / "chain_check.json", check)
    return prof, check


def chain_check_failure(cfg, check) -> "str | None":
    """The chain check's verdict for a step: a failure message where it gates,
    else None. It gates a real session; a rehearsal records it (its tolerance
    is a working value the rehearsal measures), as does the CPU boundary."""
    if check["passed"] or not check["gated"] or cfg.get("rehearsal"):
        return None
    return (f"the chain check failed for {check['failed']}: a composite's chain-priced cost "
            f"is outside the tolerance of its directly measured host time")


def code_path_definition(cp) -> dict:
    """T2.11's code-path definition (no arm) from a shakedown `CodePath`."""
    return {"gpu": cp.gpu, "implementation": cp.implementation,
            "libraries": [list(x) for x in cp.libraries], "python_optimize": cp.python_optimize}


def session_floor_ms(cfg, session: Path) -> float:
    """The resolution floor the ground truth flags against: the config's if
    the lock has set one, else the session floor `dense_select` derived."""
    if cfg.get("resolution_floor_ms") is not None:
        return float(cfg["resolution_floor_ms"])
    p = step_dir(session, "dense_select") / "session_floor.json"
    if not p.exists():
        raise StepFailed(f"no session floor at {p}: dense_select has not derived one")
    return float(json.loads(p.read_text())["floor_ms"])


def rerun_budget(cfg) -> int:
    """Extra rounds allowed for flagged ones: none until it is decided (T3.2)."""
    v = cfg.get("max_rerun_rounds")
    return 0 if v is None else int(v)


def label_outside_layers(operations) -> list:
    """Give work outside the decoder layers a layer of its own, which the step
    form needs (`baselines.steps`: "label work outside the layers with its
    own layer value"): kernels and copies before the first layer take -1
    (the embedding), after the last take one past it (the norm and output
    head). A synchronisation keeps None. An unlabelled kernel between two
    layers is refused: it would put work in the wrong step."""
    from dataclasses import replace
    from attnbench.predict.recursion import OpKind
    layered = [i for i, op in enumerate(operations) if op.layer is not None]
    if not layered:
        raise StepFailed("no operation carries a layer label: the recording saw no layers")
    first, last = layered[0], layered[-1]
    after = max(operations[i].layer for i in layered) + 1
    out = []
    for i, op in enumerate(operations):
        if op.layer is None and op.kind is not OpKind.SYNC:
            if i < first:
                op = replace(op, layer=-1)
            elif i > last:
                op = replace(op, layer=after)
            else:
                raise StepFailed(f"operation {i} ({op.kind.value}) sits between layers "
                                 f"with no layer label")
        out.append(op)
    return out


def link_costs(seq, row) -> dict[int, float]:
    """Seconds per host-device copy position: payload over the T2.5
    bandwidth of its direction and host memory (pinned or pageable)."""
    from attnbench.components.op_sequence import SeqKind
    out = {}
    for i, op in enumerate(seq.ops):
        if op.kind is not SeqKind.COPY:
            continue
        direction = {"HtoD": "h2d", "DtoH": "d2h"}.get(op.direction)
        mem = op.src_mem if op.direction == "HtoD" else op.dst_mem
        if direction is None or mem not in ("Pinned", "Pageable"):
            raise StepFailed(f"copy at position {i}: no link cost for {op.direction} "
                             f"from {op.src_mem} to {op.dst_mem}")
        bw = float(row[f"bw_{direction}_{mem.lower()}_Bps"])
        out[i] = op.payload_bytes / bw
    return out


def _skew(cfg, dry_run: bool = False) -> tuple[timedelta, timedelta]:
    """(clock skew, guard): the gap and the margin of `commit`, whose sum is
    PLANNED_SYNC_WAIT_S. Only a dry run's config may name its own."""
    if not dry_run:
        if "clock_skew_s" in cfg or "hold_guard_s" in cfg:
            raise StepFailed("a real session's hold is PLANNED_SYNC_WAIT_S; its config "
                             "may not set clock_skew_s or hold_guard_s")
        return so.REQUIRED_SKEW, so.PLANNED_GUARD
    return (timedelta(seconds=cfg.get("clock_skew_s", REQUIRED_SYNC_GAP_S)),
            timedelta(seconds=cfg.get("hold_guard_s", SYNC_WAIT_MARGIN_S)))


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
    if bool(cfg.get("rehearsal")) != prefix.endswith("-rehearsal"):
        raise StepFailed(f"a rehearsal's prediction goes under a '-rehearsal' prefix, and only "
                         f"a rehearsal's: rehearsal={cfg.get('rehearsal')!r}, prefix {prefix!r}")
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
    cmd = _py("t227_steps.py", step, "--out", str(out), *(["--dry-run"] if _local(args) else []))
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
                  "stub_rows": _boundary(),          # the CPU boundary's dry-run rows
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
    from attnbench.components.shakedown_arms import ARMS
    h = _harness(args)
    found = {}
    if cfg.get("resolution_floor_ms") is None:      # null as an input: derive it here
        floor = derive_session_floor(h, cfg, Path(args.out))
        cfg = dict(cfg, resolution_floor_ms=floor["floor_ms"])
        found["session_floor"] = floor
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            cell = ds.Cell(card=cfg["card"], context_length=L, batch=b)
            attn_cfg = replace(h._cfg(ARMS["dense"], L), batch=b)
            choice = ds.select_dense_baseline(
                cell, attn_cfg, ds.registry_candidates(cfg["card"]),
                rounds=cfg["dense_rounds"], resolution_floor_ms=cfg["resolution_floor_ms"],
                device=_device())
            path = ds.write_choice(choice, Path(args.out))
            found[cell.key] = {"chosen": choice.chosen, "complete": choice.complete,
                               "file": path.name}
            if not choice.complete:
                raise StepFailed(f"{cell.key}: the dense selection is incomplete", found)
    return found


def derive_session_floor(h, cfg, out: Path, *, clock=time.perf_counter) -> dict:
    """The floors are null as inputs (31 Oct 2026): T3.1 derives a session
    floor from its own dense pre-pass. The dense arm is run `reps` times at
    every (length, batch) after two warm-ups, and the floor is
    `lock.floors.resolution_floor` over those series at the provisional
    shakedown quantile, labelled working, pre-lock. The series are saved so
    the lock can recompute the floor at its own quantile. Run only when the
    config's floor is null; a floor the lock has set is used as it is."""
    import torch
    from attnbench.lock import floors
    series = {}
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            a = h.prepare("dense", L, batch=b)
            for _ in range(2):
                a.run_prefill()
            torch.cuda.synchronize()
            xs = []
            for _ in range(cfg["reps"]):
                t0 = clock()
                a.run_prefill()
                torch.cuda.synchronize()
                xs.append((clock() - t0) * 1e3)
            series[f"dense L{L} b{b}"] = xs
    h.unwrap()
    r = floors.resolution_floor(series, quantile=floors.SHAKEDOWN_QUANTILE, pairing=cfg["pairing"])
    rec = {"floor_ms": r.floor_ms, "quantile": r.quantile, "n_series": r.n_series,
           "n_repeats": r.n_repeats, "inputs_sha256": r.inputs_sha256, "pairing": r.pairing,
           "status": f"{WORKING_STATUS}: derived from this session's dense pre-pass at the "
                     f"provisional quantile; the lock recomputes it from `series`",
           "series": series}
    _write_json(out / "session_floor.json", rec)
    return {k: v for k, v in rec.items() if k != "series"}


def step_probe(args, cfg) -> dict:
    """T2.18 for every arm. P1's own recordings are the test of record: each
    arm is probed at its probe points (`structural_probe.probe_plan`; a
    chunked arm checked at the third, every length at or below the model's
    32768 positions), extended to every grid length, and checked against
    the full recording there. "Different" is a verdict (researcher's ruling,
    31 Oct 2026): there the full-length recording is the arm's structure at
    that length, with every duration discarded and nothing extrapolated, so
    dense and oracle may take one recording per grid length."""
    if args.dry_run:
        return {"fake": True, "arms": cfg["arms"]}
    _need_gpu("the structural probe")
    from attnbench.components import recorder, structural_probe as sp
    from attnbench.components.shakedown_arms import ARMS
    out = Path(args.out)
    h = _harness(args)
    found: dict = {}
    for name in cfg["arms"]:

        def record(L, d, name=name):
            a = h.ready(name, L)
            return recorder.record_arm_prefill(
                a.run_prefill, d, code_path=a.code_path, context_length=a.length, batch=1,
                prompt_id=a.prompt_id, layers=a.layers).sequence

        rule = sp.xattention_rule() if ARMS[name].xattention else sp.NO_CHUNKING
        plan = sp.probe_plan(rule)
        lengths = plan.lengths
        seqs = sp.probe(record, out / name, lengths=lengths)
        per = {}
        for L in cfg["lengths"]:
            full = record(L, out / name / f"full_{L}")
            ext = (sp.extend_checked(seqs, rule, full.context_length,
                                     full_third=plan.full_third) if rule.chunked
                   else sp.extend(seqs, rule, full.context_length))
            r = {"extension": ext.verdict, "reason": ext.reason, "length": full.context_length}
            if ext.extended:
                c = sp.check_against_full(ext, full)
                r.update(check=c.verdict, first_difference=c.first_difference)
            same = ext.extended and r.get("check") == "same sequence"
            r["structure"] = "extended" if same else "recorded at full length"
            per[str(L)] = r
        found[name] = {"rule": rule.name, "probe_lengths": list(lengths),
                       "probe_plan": plan.reason, "full_third": plan.full_third,
                       "by_length": per}
        if rule.chunked:    # the extension test (ruling, 1 Nov 2026): reported either way
            fit = sp.count_fit(seqs, rule)
            found[name]["extension_test"] = {
                "count_fit": fit,
                "periodic_order": all(r["extension"] == "same sequence" for r in per.values()),
                "passed": fit["fits"] and all(r["structure"] == "extended" for r in per.values()),
                "fallback": "full-length recording per grid cell"}
    h.unwrap()
    _write_json(out / "probe.json", found)
    return found


def step_host_components(args, cfg) -> dict:
    """T2.2 per cell: the arm's prefill recorded at the cell (trace,
    sequence, masks, tape), then the null-kernel host profile from its tape
    (`HOST_PROFILE_REPS` passes). Writes `cells_index.json`, which the
    device step reads; a calibrated arm's density is the mask density it
    produced on the cell's prompt."""
    if args.dry_run:
        return {"fake": True}
    _need_gpu("the null-kernel host profile")
    from attnbench.components import host_profile as hp
    from attnbench.components import recorder
    out = Path(args.out)
    h = _harness(args)
    index = []
    for arm, d, L, b in cells(cfg):
        cd = cell_dir(out, arm, d, L, b)
        a = ready_cell(h, arm, d, L, b)
        rec = recorder.record_arm_prefill(a.run_prefill, cd / "recording", code_path=a.code_path,
                                          context_length=a.length, batch=b,
                                          prompt_id=a.prompt_id, layers=a.layers)
        entries = () if arm == "dense" else (hp.block_sparse_entry(),)
        prof, check = host_profile_cell(a, rec, cd, reps=HOST_PROFILE_REPS, entries=entries,
                                        tolerance=cfg["chain_check_tolerance"])
        (cd / "host_profile.json").write_text(hp.dumps(prof))
        failure = chain_check_failure(cfg, check)
        if failure:
            raise StepFailed(f"{arm} at {L} (batch {b}): {failure}")
        realised = d
        if d is None:
            per_layer = list(getattr(a.backend, "last_layer_density", None) or ())
            if not per_layer:
                raise StepFailed(f"{arm}: its backend keeps no mask density, so its cell "
                                 f"has no density")
            realised = round(sum(float(x) for x in per_layer) / len(per_layer), 6)
        index.append({"arm": arm, "nominal_density": d, "density": realised, "budget": L,
                      "context_length": a.length, "batch": b, "prompt_id": a.prompt_id,
                      "dir": str(cd.relative_to(out)),
                      "code_path": code_path_definition(a.code_path),
                      "host_total_ms": prof.total_s * 1e3,
                      "host_cost_sources": prof.sources(),
                      "chain_check": {k: check[k] for k in ("passed", "gated", "failed",
                                                            "checked_calls", "unchecked_calls")}})
    h.unwrap()
    _write_json(out / "cells_index.json", {"cells": index})
    failed = [f"{c['arm']}@{c['context_length']}x{c['batch']}" for c in index
              if not c["chain_check"]["passed"]]
    return {"cells": len(index), "chain_check_failed_cells": failed}


def step_device_components(args, cfg) -> dict:
    """T2.4 per cell, on the host step's recording: the CUDA-graph replay
    from the tape and the bank (live masks checked bitwise against the
    bank), its nodes matched to the sequence, substituted positions priced
    from the original operators, then each position's host (T2.2), device
    (T2.4) and link (T2.5) cost as recursion operations: `cells.json`, one
    entry per ground-truth cell, which the prediction reads. A cell whose
    sequence holds memsets with no device record (31 Oct 2026 ruling) also
    carries their count and the same operations with those memsets priced
    at the memset zoo's duration (`memset_flags`), for the upper-bound
    sensitivity `predict` reports; the prediction itself prices them at
    zero device cost."""
    if not args.dry_run:
        return device_components_real(args, cfg)
    entries = [{"arm": a, "density": d, "context_length": L, "batch": b,
                "code_path": FAKE_CODE_PATH,
                "ops": [op_to_dict(op) for op in fake_ops(a, d, L, b)]}
               for a, d, L, b in cells(cfg)]
    _write_json(Path(args.out) / "cells.json", {"fake": True, "cells": entries})
    return {"fake": True, "cells": len(entries)}


def device_components_real(args, cfg) -> dict:
    _need_gpu("the CUDA-graph replay")
    import pandas as pd
    import torch
    from attnbench.components import host_profile as hp
    from attnbench.components import memset_flags as msf
    from attnbench.components import op_sequence as ops
    from attnbench.components import recorder, replay
    out, session = Path(args.out), Path(args.session)
    zoo: dict = {}
    host_dir = step_dir(session, "host_components")
    index = json.loads((host_dir / "cells_index.json").read_text())["cells"]
    row = pd.read_parquet(step_dir(session, "link") / "link" / "link.parquet").iloc[0].to_dict()
    h = _harness(args)
    entries, sources = [], {}
    for c in index:
        cd = out / c["dir"]
        cd.mkdir(parents=True, exist_ok=True)
        rec = recorder.load_recording(host_dir / c["dir"] / "recording")
        a = ready_cell(h, c["arm"], c["nominal_density"], c["budget"], c["batch"])
        cap = replay.capture(a.run_prefill, rec.tape, device=_device(), masks=rec.masks,
                             backend=_graph_backend())
        try:    # after one replay and a synchronise (ruling, 1 Nov 2026)
            agreement, failure = replay.mask_agreement(cap), None
        except replay.ReplayError as e:
            agreement, failure = None, str(e)
        if agreement is not None and not agreement.get("identical"):
            failure = "the replay's live masks differ from the bank"
        if failure:
            from attnbench.components import mask_diagnosis as md   # findings only (31 Oct 2026)
            diag = md.guarded(md.replay_diagnosis, a.run_prefill, rec, device=_device(),
                              graph_backend=_graph_backend())
            _write_json(cd / "mask_diagnosis.json", diag)
            raise StepFailed(f"{c['dir']}: {failure}", {"mask_diagnosis": md.summaries(diag)})
        m = replay.match(rec.sequence, replay.replay_nodes(
            replay.profile_replay(cap, cd / "replay_trace.json")))
        original = None
        if not m.complete:
            torch.cuda.synchronize()
            orig = replay.isolated_replay(a.run_prefill, rec.tape, cd / "isolated_original.json",
                                          device=_device(), masks=rec.masks, substitute=False)
            original = replay.match(rec.sequence, replay.replay_nodes(orig, exclude=replay.FLUSH))
        try:
            pricing = replay.price_positions(rec.sequence, m, original)
        except replay.ReplayError as e:
            raise StepFailed(f"{c['dir']}: {e}")
        prof = hp.from_record(json.loads((host_dir / c["dir"] / "host_profile.json").read_text()))
        costs = dict(host=hp.host_costs(prof, rec.sequence), device=pricing.device_s,
                     link=link_costs(rec.sequence, row))
        priced = label_outside_layers(ops.to_operations(rec.sequence, **costs))
        memsets = None
        if msf.count(rec.sequence):
            if not zoo:
                zoo.update(msf.measure_zoo(out / "memset_zoo_trace.json"))
            trace = rec.directory / "trace.json"
            sizes = (msf.flagged_sizes(trace, rec.sequence) if trace.exists()
                     else {i: None for i in msf.flagged(rec.sequence)})
            upper, src = msf.upper_bound_s(sizes, zoo)
            memsets = {"count": len(upper), "sources": sorted(set(src.values())),
                       "ops_upper": [op_to_dict(op) for op in label_outside_layers(
                           ops.to_operations(rec.sequence, **costs, flagged_memset_s=upper))]}
        sources[c["dir"]] = pricing.counts()
        entries.append({"arm": c["arm"], "density": c["density"],
                        "nominal_density": c["nominal_density"],
                        "context_length": c["context_length"], "budget": c["budget"],
                        "batch": c["batch"], "prompt_id": c["prompt_id"],
                        "code_path": c["code_path"], "cost_sources": pricing.counts(),
                        "ops": [op_to_dict(op) for op in priced],
                        **({"memset_flags": memsets} if memsets else {})})
    h.unwrap()
    _write_json(out / "cells.json", {"fake": False, "cells": entries})
    return {"cells": len(entries), "cost_sources": sources}


def build_prediction(cfg, session: Path, *, dry_run: bool, now=None, local=None):
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
        # The link table is per grid length; a prompt lands at or below its budget.
        grid_L = e.get("budget", e["context_length"])
        mech = {m: MechanismCost(**v) for m, v in rin["by_length"][str(grid_L)].items()}
        priced = price(close_prefill(op_from_dict(o) for o in e["ops"]), mech)
        out.append(Cell(pairing=cfg["pairing"], model_name=cfg["model"], code_path=label,
                        arm=e["arm"], context_length=e["context_length"],
                        density=e["density"], batch=e["batch"], priced=priced, hardware=hw))
    want = {(a, d, L, b) for a, d, L, b in cells(cfg)}
    # A cell is named by its setting: the nominal density (None where the
    # estimator sets it) and the length budget (the prompt lands at or below it).
    got = {(e["arm"], e.get("nominal_density", e["density"]),
            e.get("budget", e["context_length"]), e["batch"]) for e in dev["cells"]}
    if want != got:
        raise StepFailed(f"priced cells differ from the grid: missing {sorted(want - got)}, "
                         f"extra {sorted(got - want)}")
    code = C.CodeStamp("0" * 40, False) if (dry_run if local is None else local) \
        else C.current_code_stamp()
    return C.build_record(cfg["pairing"], out, rin["queue_depth"], code_paths=code_paths,
                          banked=[], fitted_sum=None, learned=None, code=code,
                          written_at=now or datetime.now(timezone.utc))


def step_predict(args, cfg) -> dict:
    """Plan Section 5.3 step 6: write the file and its sha256, sync both, read
    them back. The receipt is what `hold` and the ground truth read. P1 has
    no banked pairing, so no fitted baseline (commit.NOT_FITTED_REASON)."""
    from attnbench.predict import commit as C
    session, out = Path(args.session), Path(args.out)
    record = build_prediction(cfg, session, dry_run=args.dry_run, local=_local(args))
    committed = C.write_prediction(record, out)
    sensitivity = memset_sensitivity(session)
    _write_json(out / "memset_sensitivity.json", sensitivity)
    store, base = make_store(session, dry_run=_local(args))
    receipt = C.sync_prediction(committed, store, f"{base}/{prediction_dir(cfg, session)}")
    d = receipt.to_dict()
    for k in ("remote", "remote_sidecar"):          # relative to the bucket (results_bucket)
        if not d[k].startswith(base + "/"):
            raise StepFailed(f"{d[k]} is not under the store's base")
        d[k] = d[k][len(base) + 1:]
    _write_json(out / "receipt.json", {**d, "local": committed.path.name})
    return {"file": committed.path.name, "sha256": committed.sha256,
            "synced_at": d["synced_at"], "remote_in_bucket": d["remote"],
            "flagged_memsets": sensitivity["flagged_memsets"],
            "memset_max_shift_s": sensitivity["max_shift_s"]}


def memset_sensitivity(session: Path) -> dict:
    """The upper-bound sensitivity of the 31 Oct 2026 ruling, per cell: D_N
    with the flagged memsets at zero device cost (the prediction's) and at
    the memset zoo's duration, and the shift. Written beside the committed
    prediction, never into it. The paper reports the flagged-memset count."""
    from attnbench.predict.recursion import MechanismCost, close_prefill, price, recurse
    rin = json.loads((step_dir(session, "link") / "recursion_inputs.json").read_text())
    dev = json.loads((step_dir(session, "device_components") / "cells.json").read_text())

    def dn(op_dicts, mech):
        return recurse(price(close_prefill(op_from_dict(o) for o in op_dicts), mech),
                       rin["queue_depth"]).end_to_end

    rows = []
    for e in dev["cells"]:
        m = e.get("memset_flags")
        if not m:
            continue
        mech = {k: MechanismCost(**v)
                for k, v in rin["by_length"][str(e.get("budget", e["context_length"]))].items()}
        base, bound = dn(e["ops"], mech), dn(m["ops_upper"], mech)
        rows.append({k: e[k] for k in ("arm", "density", "context_length", "batch")} |
                    {"flagged_memsets": m["count"], "sources": m["sources"],
                     "prediction_s": base, "upper_bound_s": bound, "shift_s": bound - base})
    return {"flagged_memsets": sum(r["flagged_memsets"] for r in rows),
            "max_shift_s": max((r["shift_s"] for r in rows), default=0.0), "cells": rows}


def _receipt(session: Path):
    from attnbench.predict.session_order import receipt_from_dict
    p = step_dir(session, "predict") / "receipt.json"
    if not p.exists():
        raise StepFailed(f"no receipt at {p}: the prediction has not been synced")
    d = json.loads(p.read_text())
    return receipt_from_dict(d), step_dir(session, "predict") / d["local"]


def step_hold(args, cfg) -> dict:
    """Wait until the instance clock is past synced_at + skew + guard."""
    skew, guard = _skew(cfg, _local(args))
    so.check_session_skew(skew, guard, dry_run=_local(args))
    receipt, _ = _receipt(Path(args.session))
    t0 = so.utc_now()
    released = so.hold(receipt.synced_at, clock_skew=skew, guard=guard)
    return {"synced_at": receipt.synced_at.isoformat(), "released_at": released.isoformat(),
            "held_s": round((released - t0).total_seconds(), 3),
            "skew_s": skew.total_seconds(), "guard_s": guard.total_seconds()}


def step_ground_truth(args, cfg) -> dict:
    """T2.13, one process per compile-setting group (Appendix A, 27 Oct 2026),
    each with its own dense control."""
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


def _sampler(cfg, dry_run: bool):
    """The session's detector: the sampler process reading the throttle bits
    in every block (`ProcessSampler`); a dry run reads a scripted reader. A
    counters-only `BlockSampler` is refused by `gt.Controls` (audit C3)."""
    from attnbench import clock_sampling as cs
    bits = {v: k for k, v in cs.REASON_NAMES.items()}
    flag = 0
    for name in cfg["flag_reasons"]:
        flag |= bits[name]
    factory, fa = ((cs.scripted_reader_factory, ()) if dry_run
                   else (cs.nvml_reader_factory, (0,)))
    hz = cs.SAMPLER_HZ
    interval = cfg["sampler_interval_s"] or 1.0 / hz
    return cs.ProcessSampler(factory, fa, flag_bits=flag, interval_s=interval)


def cmd_ground_truth_one(args) -> int:
    """One group's arms at every cell, in this process only. Refuses unless
    the prediction's hold has passed by this process's own clock."""
    from attnbench import ground_truth as gt
    cfg = load_config(args.config, dry_run=args.dry_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skew, guard = _skew(cfg, args.dry_run)
    so.check_session_skew(skew, guard, dry_run=args.dry_run)
    receipt, _ = _receipt(Path(args.session))
    so.ready_to_clock(receipt.synced_at, so.utc_now(), clock_skew=skew, guard=guard)
    not_before = time.time()          # before this process's first forward of any kind
    arms_in_group = cfg["gt_groups"][args.group]
    if args.dry_run:
        from attnbench.components.shakedown_arms import Harness
        h = Harness(device="cpu", dry_run=True, score_cache_dir=str(out / "score_cache"))
        floor, rerun = cfg["resolution_floor_ms"], cfg["max_rerun_rounds"]
    else:
        _need_gpu("the ground truth")
        h = _harness(args)
        floor, rerun = session_floor_ms(cfg, Path(args.session)), rerun_budget(cfg)
    with _sampler(cfg, args.dry_run) as sampler:
        controls = gt.Controls(resolution_floor_ms=floor, max_rerun_rounds=rerun,
                               sampler=sampler)
        cells_fn = _dry_cells if args.dry_run else _real_cells
        first = cells_fn(h, gt, cfg, controls, args, receipt, out)
    stamps = {"group": args.group, "arms": arms_in_group, "clock_not_before": not_before,
              "first_clocked_block_started_at": first,
              "synced_at": receipt.synced_at.isoformat()}
    _write_json(out / "stamps.json", stamps)
    print(f"[ground-truth-one {args.group}] {len(first)} cells", flush=True)
    return 0


def group_setting(group: str):
    """None for the eager group, else the compile setting the group's name
    carries (`compiled_default` -> "default"); one setting per process."""
    if group == "eager":
        return None
    if not group.startswith("compiled_"):
        raise StepFailed(f"unknown ground-truth group {group!r}")
    return group[len("compiled_"):]


def group_arms(cfg, group: str, length: int, batch: int):
    """(names, densities) for one group's cell: the eager dense control first,
    then every other arm of the group at each of its densities."""
    names, dens = ["dense"], [None]
    for arm in cfg["gt_groups"][group]:
        if arm == "dense" or (batch > 1 and arm in BATCH1_ONLY):
            continue
        for d in cfg["densities"][arm]:
            names.append(arm)
            dens.append(nominal_density(d))
    return names, dens


def _real_cells(h, gt, cfg, controls, args, receipt, out) -> list:
    """Every (length, batch) of this group: the K2 choice as the dense
    control (checked with `check_control` against the session's record),
    the group's arms at their densities, compiled under the group's one
    setting, timed with the session's sampler, and each row given its cell."""
    import torch
    from attnbench import dense_selection as ds
    from attnbench.components.shakedown_arms import ARMS
    setting = group_setting(args.group)
    if setting is not None:
        from attnbench.backends.compiled import CompiledBackend
    sel_dir = step_dir(Path(args.session), "dense_select")
    first = []
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            cell = ds.Cell(card=cfg["card"], context_length=L, batch=b)
            choice = ds.load_choice(sel_dir / ds._filename(SimpleChoice(cell)))
            control = ds.registry_candidates(cfg["card"])[choice.chosen]
            names, dens = group_arms(cfg, args.group, L, b)
            backends, settings = [control], [None]
            for n in names[1:]:
                be = h._backend(ARMS[n])
                if setting is not None:
                    be = CompiledBackend(be, setting)
                backends.append(be)
                settings.append(setting)
            wrapped, ids, pid, arms, score_kwargs = h.timing_arms(
                names, L, densities=dens, batch=b, backends=backends, compile_settings=settings)
            res = gt.time_arms(wrapped, ids, arms, reps=cfg["reps"], controls=controls,
                               warmup=cfg["warmup"], density_of=gt.kernel_densities,
                               synchronize=torch.cuda.synchronize, score_kwargs=score_kwargs)
            stamp = res["first_clocked_block_started_at"]
            ds.check_control(choice, choice.chosen, stamp, key=cell)
            first.append(stamp)
            for row, n, d in zip(res["rows"], names, dens):
                row.update(model=cfg["model"], cell_arm=n, nominal_density=d, budget=L,
                           group=args.group, compile_setting=setting, prompt_id=pid,
                           session_receipt_sha256=receipt.sha256,
                           working_values=cfg.get("working_values"))
            _write_json(out / f"gt_L{L}_b{b}.json", res)
            h.unwrap()
    return first


class SimpleChoice:
    """Just enough of a `Choice` for `dense_selection._filename`."""
    def __init__(self, cell):
        self.cell = cell


def _dry_cells(h, gt, cfg, controls, args, receipt, out) -> list:
    first = []
    for L in cfg["lengths"]:
        pa = h.prepare("dense", L)
        arm = gt.Arm("dense", gt.DENSE, pa.backend, pa.extra["cfg"])
        res = gt.time_arms(pa.extra["wrapped"], pa.extra["input_ids"], [arm],
                           reps=cfg["reps"], controls=controls, warmup=cfg["warmup"],
                           density_of=gt.kernel_densities)
        first.append(res["first_clocked_block_started_at"])
        for row in res["rows"]:
            row.update(model=cfg["model"], code_path=None, density=1.0,
                       context_length=L, batch=1, group=args.group,
                       session_receipt_sha256=receipt.sha256)
        _write_json(out / f"gt_L{L}.json", res)
    return first


def step_verify(args, cfg) -> dict:
    """The analysis's late-prediction check (`commit.load_committed`), run in
    the session against the store's copy, with the earliest stamp of any
    ground-truth process as the start."""
    from attnbench.predict import commit as C
    session = Path(args.session)
    receipt, local = _receipt(session)
    gt_dir = step_dir(session, "ground_truth")
    stamps = [json.loads(p.read_text()) for p in sorted(gt_dir.glob("*/stamps.json"))]
    if not stamps:
        raise StepFailed(f"no ground-truth stamps under {gt_dir}")
    if any(s["synced_at"] != receipt.synced_at.isoformat() for s in stamps):
        raise StepFailed("a ground-truth group read another prediction's receipt")
    start = so.session_start([s["clock_not_before"] for s in stamps],
                             [t for s in stamps for t in s["first_clocked_block_started_at"]])
    store, base = make_store(session, dry_run=_local(args))
    skew, _guard = _skew(cfg, _local(args))
    try:
        cp = C.load_committed(store, f"{base}/{receipt.remote}", ground_truth_started_at=start,
                              clock_skew=skew, banked_at={}, local=local)   # P1 is the first pairing: none banked
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
        cfg = load_config(args.config, dry_run=args.dry_run, cpu_boundary=args.cpu_boundary)
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
        "cpu_boundary": args.cpu_boundary, "seconds": round(time.time() - started, 1), "findings": findings})
    print(f"[{name}] {status.upper()}{': ' + reason if reason else ''}", flush=True)
    return EXIT[status]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS + ("ground-truth-one",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-boundary", action="store_true",
                    help="the real arms on a CPU behind the stubbed GPU boundary (steps 1-8)")
    ap.add_argument("--config", default=None,
                    help=f"session values (default {DEFAULT_CONFIG.relative_to(ROOT)}; "
                         f"a dry run uses its own unless given)")
    ap.add_argument("--group", default=None)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.cpu_boundary:
        if args.dry_run:
            print("REFUSING: --cpu-boundary runs the real steps; it is not --dry-run",
                  file=sys.stderr)
            return 2
        if args.step not in STEPS[:STEPS.index("hold") + 1]:
            print(f"REFUSING: the CPU boundary runs steps 1 to 8, not {args.step}",
                  file=sys.stderr)
            return 2
        from attnbench import cpu_boundary
        with cpu_boundary.installed():
            return run_step(args.step, args)
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
