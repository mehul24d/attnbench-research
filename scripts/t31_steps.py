#!/usr/bin/env python3
"""The steps of the P1 session (T3.1), one subcommand each.

`scripts/t31_session.py` runs these in order, each in its own process with
its own log, and stops at the first failure. Each step writes `result.json`
in its directory, as the T2.27 steps do:

    {"step", "status": "passed" | "failed" | "skipped" | "incomplete", "reason",
     "dry_run", "findings": {...}}

and exits 0 (passed), 1 (failed), 3 (skipped) or 4 (incomplete: the step ran
to its end but a cell has fewer clean rounds than its reps; ground truth only,
adviser's ruling 2 of 9 Nov 2026).

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
  delay_injection    T2.25, Gate A's leg 3, after ground truth (8 Nov 2026)

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
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attnbench.predict import session_order as so  # noqa: E402
from attnbench.predict.commit import (CLOCK_SKEW_ALLOWED_S, PLANNED_SYNC_WAIT_S,  # noqa: E402,F401
                                      REQUIRED_SYNC_GAP_S, SYNC_WAIT_MARGIN_S)

PASSED, FAILED, SKIPPED, INCOMPLETE = "passed", "failed", "skipped", "incomplete"
EXIT = {PASSED: 0, FAILED: 1, SKIPPED: 3, INCOMPLETE: 4}

STEPS = ("preflight", "link", "dense_select", "probe", "host_components",
         "device_components", "predict", "hold", "ground_truth", "verify", "postflight",
         "delay_injection")

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
                "chain_check_tolerance", "residual_spread_tolerance",
                "pricing_mismatch_share")
WORKING_STATUS = "working, pre-lock"
WORKING_PAIRING = "P1"
# A later pairing's session values come from the lock file instead (T4.1).
# Every T4.1 item the session reads comes from the lock's typed `values` and
# nowhere else (adviser's ruling 1b of 11 Nov 2026; T4.3 read B, finding 4):
# session key -> (T4.1 item, how the session key is read from its value). The
# repository config must leave each of these null; `pairing_values` is never
# read for them. `pairing_values` holds only the session values T4.1 does not
# list (`PAIRING_KEYS`).
LOCK_FILE = ROOT / "docs" / "lock_values.json"
GRID_ITEM = "context_grid_checked_against_each_model_s_position_limit"
LOCK_SOURCED = {
    "densities": ("ground_truth_density_set", lambda v: v),
    "lengths": (GRID_ITEM, lambda v: v["lengths"]),
    "batches": (GRID_ITEM, lambda v: v["batches"]),
    "flag_reasons": ("throttle_set", lambda v: v),
    "max_rerun_rounds": ("rerun_budget", lambda v: v["max_rerun_rounds"]),
    "min_clean_rounds": ("rerun_budget", lambda v: v["min_clean_rounds"]),
    "resolution_floor_ms": ("resolution_and_noise_floors", lambda v: v["resolution_floor_ms"]),
    # the lock types it one fraction (T4.1 row 32); the steps read {"aggregate"},
    # as the working table holds it (found by P2's pre-launch pass, 12 Nov 2026)
    "chain_check_tolerance": ("chain_check_tolerance", lambda v: {"aggregate": v}),
    "residual_spread_tolerance": ("residual_spread_tolerance", lambda v: v),
    "pricing_mismatch_share": ("pricing_mismatch_share", lambda v: v),
}
PAIRING_KEYS = ("gt_groups", "reps", "dense_rounds", "sampler_interval_s")
# Dropped from P2 and P4 (adviser's ruling 4, third set of 12 Nov 2026): four
# rehearsal launches never reached the compiled stage on a GPU.
DROPPED_GROUPS = ("compiled_default", "compiled_capture")
LOCK_KEYS = PAIRING_KEYS + tuple(LOCK_SOURCED)
CALIBRATED = "calibrated"          # a density the estimator sets (XAttention), not the cell
# Arms that run at batch 1 only: XAttention's kernel call (T2.12), and the
# oracle's scoring pass and the mean-pool estimator, which refuse a batch
# above one as built (found by the CPU-boundary rehearsal, 31 Oct 2026). The
# batch-4 cells hold the dense control only: the scope is narrowed (ruling,
# 1 Nov 2026: sparse arms evaluated at batch 1, batch above 1 future work;
# plan Sections 5.4, 5.9, H6 and `docs/limitations.md` say so).
BATCH1_ONLY = ("xattention", "xattention_noassert", "oracle", "meanpool")
HOST_PROFILE_REPS = 5
BOUNDARY_BARE_REPS = 3
# The isolated replay's profiler is rolled every this many decoder layers
# (ruling, 2 Nov 2026, after T3.1's fourth session: `oracle_d0.5_L32768` hit
# the ceiling in one profiler session). At two layers the largest cell's
# blocks are a fourteenth of its events, about 2 GiB of profiler memory
# against 29 GiB for the whole; a session's config may set
# `isolated_segment_layers` (0: one session, as before).
ISOLATED_SEGMENT_LAYERS = 2

# Steps 5 and 6 run each cell in its own process under a host-memory ceiling
# (researcher's ruling, 2 Nov 2026, after T3.1's third session, whose step 5
# was killed by the out-of-memory killer at 31.6 GB on g2-standard-8's 31
# GiB; step 6 on the researcher's answer of the same day). The ceiling leaves
# 5 GiB for the parent, the system and the guard's sampling interval; a
# session's config may set `host_cell_rss_ceiling_gb`.
HOST_CELL_RSS_CEILING_GB = 26.0
DEBUG_LABEL = "debug, not T3.1 data"    # a debug session's every output (2 Nov 2026 ruling)
STAGED_INPUTS = "staged_inputs.json"     # what a debug session staged, and from where
CELL_CHILDREN = ("host-cell-one", "device-cell-one")     # steps 5 and 6
CELL_MANIFEST = "cell_manifest.json"      # the parent's record of the cell's process
CELL_INDEX = "cell_index.json"            # the child's entry for the step's index
CELL_EXIT = "cell_exit.json"              # the child's own high-water mark, error, findings
PRICE_CACHE = "bare_price_cache.pkl"      # the bare prices, shared by step 5's cells
MEMSET_ZOO = "memset_zoo.json"            # the memset zoo, measured once for step 6
XATTN_TRITON = "xattn:triton"             # XAttention's Triton path (`Harness.code_path`)
SMOKE = "preflight-smoke"                 # the preflight's smoke check (`preflight_smoke`)
SMOKE_LENGTH, SMOKE_BATCH = 2048, 1

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
    "clock_skew_s": 2, "hold_guard_s": 1, "sync_gap_s": 2,
    "prediction_prefix": "predictions/P1-dryrun",
    "delay_injection": {"arm": "dense", "length": 64, "batch": 1,
                        "mechanisms": ["synchronize", "item", "copy_to_host_pinned",
                                       "event_synchronize"],
                        "counts": {"busy": [0, 1, 2], "idle": [0, 2, 4]},
                        "reps": 2, "warmup": 1, "payload_bytes": 0},
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
    "clock_skew_s": 2, "hold_guard_s": 1, "sync_gap_s": 2,
    "prediction_prefix": "predictions/P1-cpu-boundary-rehearsal",
    # Leg 3's real path behind the boundary (8 Nov 2026): every mechanism, small
    # counts, the injections a stand-in that does nothing on the CPU.
    "delay_injection": {"arm": "dense", "length": 2048, "batch": 1,
                        "mechanisms": list(DRY_CONFIG["delay_injection"]["mechanisms"]),
                        "counts": {"busy": [0, 2, 4], "idle": [0, 10, 20]},
                        "reps": 2, "warmup": 1, "payload_bytes": 0},
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


class StepIncomplete(StepFailed):
    """The step ran to its end, but what it measured is short (a ground-truth
    cell with fewer clean rounds than its reps): never reported as passed."""


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


# A session another runner owns names its steps' directories here (the P2 and
# P4 runner, `t61_session.py`, numbers its steps in its own order).
STEP_LAYOUT = "step_layout.json"


def step_dir(session: Path, step: str) -> Path:
    """A step's directory: T3.1's numbering, or the one the session's
    `STEP_LAYOUT` names; a layout that does not name the step is refused."""
    layout = Path(session) / STEP_LAYOUT
    if layout.exists():
        names = json.loads(layout.read_text())
        if step not in names:
            raise StepFailed(f"{layout} names no directory for the step {step!r}")
        return Path(session) / names[step]
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
            # A config that names its groups gets only those (P2 and P4 name
            # eager alone: ruling 4, third set of 12 Nov 2026); one naming
            # none gets the table's.
            named = cfg.get(k)
            groups = dict(named or {})
            for g, arms in vals[k].items():
                if (named is None or g in named) and groups.get(g) is None:
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


def fill_lock_values(cfg: dict, path=None, plan_text=None) -> dict:
    """A later pairing's session values from the lock file (T4.1, Gate B),
    recorded as filled: each T4.1 item it reads (`LOCK_SOURCED`) from the
    lock's typed `values`, the others (`PAIRING_KEYS`) from its
    `pairing_values` block for that pairing. A T4.1 key the repository
    config also sets is refused (two sources); one `pairing_values` also
    names is not read, and is listed as ignored. With no lock file nothing is
    filled, and `fill_working_values` refuses P1's values for it as before. A
    lock file the checker does not find complete against this plan is
    refused: no pairing's values are taken from a lock that is not one."""
    from attnbench.lock import checklist
    p = Path(path or LOCK_FILE)
    if not p.exists():
        return cfg
    text = plan_text if plan_text is not None else checklist.PLAN.read_text(encoding="utf-8")
    lock = json.loads(p.read_text(encoding="utf-8"))
    res = checklist.check(lock, text)
    if not res.complete:
        raise StepFailed(f"{p.name} is not a complete lock, so no pairing's values come from "
                         f"it:\n{res.report()}")
    vals = (lock.get("pairing_values") or {}).get(cfg.get("pairing")) or {}
    dropped = sorted(set(DROPPED_GROUPS) & (set(vals.get("gt_groups") or {})
                                            | set(cfg.get("gt_groups") or {})))
    if dropped:
        raise StepFailed(f"{dropped} are dropped from P2 and P4 (adviser's ruling 4, third set "
                         f"of 12 Nov 2026): no later pairing runs a compiled group")
    cfg = dict(cfg)
    filled = []
    twice = [k for k in LOCK_SOURCED if cfg.get(k) is not None]
    if twice:
        raise StepFailed(f"the session config sets {twice}, which are T4.1 items: they come "
                         f"from the lock only (leave them null in the repository config)")
    for k, (item, read) in LOCK_SOURCED.items():
        cfg[k] = read(lock["values"][item]["value"])
        filled.append(k)
    ignored = sorted(k for k in vals if k in LOCK_SOURCED)
    for k in PAIRING_KEYS:
        if k == "gt_groups":
            groups = dict(cfg.get(k) or {})
            for g, arms in (vals.get(k) or {}).items():
                if groups.get(g) is None:
                    groups[g] = list(arms)
                    filled.append(f"gt_groups.{g}")
            cfg[k] = groups
        elif cfg.get(k) is None and vals.get(k) is not None:
            cfg[k] = vals[k]
            filled.append(k)
    if filled:
        import hashlib
        cfg["lock_values"] = {"source": p.name, "filled": filled,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                              "ignored_pairing_values": ignored}
    return cfg


def cpu_boundary_config(path=None) -> dict:
    """`CPU_BOUNDARY_CONFIG` with the working table's densities and groups,
    labelled as taken from it."""
    vals, sha = working_values(path)
    return dict(CPU_BOUNDARY_CONFIG, densities=vals["densities"], gt_groups=vals["gt_groups"],
                chain_check_tolerance=vals["chain_check_tolerance"],
                residual_spread_tolerance=vals["residual_spread_tolerance"],
                pricing_mismatch_share=vals["pricing_mismatch_share"],
                working_values={"sha256": sha, "status": WORKING_STATUS,
                                "filled": ["densities", "gt_groups", "chain_check_tolerance",
                                           "residual_spread_tolerance", "pricing_mismatch_share"]})


def load_config(path, *, dry_run: bool, cpu_boundary: bool = False) -> dict:
    """The session's values: the file, with its null working values filled
    from the table. A real run refuses any value still undecided."""
    if path is None and dry_run:
        return dict(DRY_CONFIG)
    if path is None and cpu_boundary:
        return cpu_boundary_config()
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if not dry_run:
        if cfg.get("pairing") != WORKING_PAIRING:
            cfg = fill_lock_values(cfg)
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


def not_measured(cfg: dict) -> dict:
    """The cells a session names as not measured (researcher's ruling, 2 Nov
    2026), (arm, density, length, batch) -> reason: a debug session's missing
    cell, or a cell dropped from the grid for a pairing's every session. No
    step prices, times or predicts one."""
    out = {}
    for e in cfg.get("not_measured") or ():
        key = (e["arm"], nominal_density(e["density"]), int(e["length"]), int(e["batch"]))
        out[key] = e["reason"]
    return out


def cells(cfg: dict):
    """(arm, density, length, batch) for every ground-truth cell; density is
    None for an arm whose estimator sets it. The XAttention arms run at batch
    1 only (the kernel call's limit, T2.12). A cell the config names as not
    measured (`not_measured`) is not in the grid."""
    out = []
    skip = not_measured(cfg)
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            for arm in cfg["arms"]:
                if b > 1 and arm in BATCH1_ONLY:
                    continue
                for d in cfg["densities"][arm]:
                    cell = (arm, nominal_density(d), int(L), int(b))
                    if cell not in skip:
                        out.append(cell)
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


def _isolated_ops() -> frozenset:
    """The operators the isolated replay prices outside the layers (ruling of
    31 Oct 2026); behind the CPU boundary, its embedding kernel too."""
    from attnbench import cpu_boundary
    from attnbench.components import host_profile as hp
    return hp.ISOLATED_OUTSIDE_LAYERS | (
        {cpu_boundary.EMBEDDING_KERNEL_OP} if _boundary() else frozenset())


def host_profile_cell(a, rec, out: Path, *, reps: int, tolerance: dict, entries=(),
                      segment_layers: int = 0):
    """T2.2's null-kernel profile of one recording, and the instrumented chain
    check.

    The isolated replay of the original operators prices the positions the
    ruling of 31 Oct 2026 gives it (`host_profile.ISOLATED_OUTSIDE_LAYERS`;
    behind the CPU boundary its embedding kernel,
    `cpu_boundary.EMBEDDING_KERNEL_OP`) and any composite whose chain is not
    recorded. `chain_check` holds the null pass's calls met at a composite or
    by their device argument against it (`tolerance` {"aggregate"}, T4.1 row
    32). Since the ruling of 1 Nov 2026 after Shakedown 5 that check compares
    two instrumented measurements and is recorded, never gated: the gate is
    the bare timer's (`bare_profile_cell`). Returns (profile, check) and
    writes the check to `chain_check.json`.

    The replay's trace is kept as its compact event stream
    (`isolated_host.events.jsonl`, `trace_stream`), the full trace deleted,
    and read once for both uses (ruling, 2 Nov 2026, after T3.1's third
    session); `check["isolated_trace"]` holds the counts.

    With `segment_layers` the replay runs in blocks of that many layers, a
    profiler session each, their streams stitched into the one stream
    (`replay.isolated_replay_compact`; ruling, 2 Nov 2026, after T3.1's
    fourth session); a prefill of no more layers than that is one block, run
    as before."""
    from attnbench.components import host_profile as hp
    from attnbench.components import replay
    ops = _isolated_ops()
    pos = hp.isolated_positions(rec.sequence, ops)
    if segment_layers and len(a.layers) > segment_layers:
        stream, counts = replay.isolated_replay_compact(
            a.run_prefill, rec.tape, Path(out) / "isolated_host.json", device=_device(),
            layers=a.layers, segment_layers=segment_layers, masks=rec.masks)
    else:
        trace = replay.isolated_replay(a.run_prefill, rec.tape, Path(out) / "isolated_host.json",
                                       device=_device(), masks=rec.masks, substitute=False)
        stream, counts = compact_isolated(trace)
    read = hp.read_isolated(stream, rec.sequence)
    iso = hp.isolated_host_s(read, rec.sequence, pos)
    prof = hp.profile(a.run_prefill, rec.tape, rec.sequence, device=_null_device(), reps=reps,
                      entries=entries, layers=a.layers, isolated=iso, isolated_ops=ops)
    check = hp.chain_check(read, rec.sequence, prof,
                           aggregate_tol=float(tolerance["aggregate"]))
    del read
    check["gated"] = False
    check["why_not_gated"] = (
        "both sides are instrumented (null pass, profiled isolated replay); the gate is the "
        "bare timer's (ruling, 1 Nov 2026, after Shakedown 5)")
    check["isolated_trace"] = counts
    _write_json(Path(out) / "chain_check.json", check)
    return prof, check


def compact_isolated(trace) -> tuple[Path, dict]:
    """The isolated replay's trace as its compact event stream, the full
    Chrome trace deleted; (stream, counts). The counts name the stream by
    its file name only."""
    from attnbench.components import trace_stream
    info = trace_stream.write_compact(trace)
    Path(trace).unlink()
    stream = Path(info.pop("path"))
    return stream, {"stream": stream.name, **info}


def _sync():
    import torch
    return (lambda: None) if _boundary() else torch.cuda.synchronize


def bare_pricer():
    """One bare-price cache for the process (`bare_timer.Pricer`): at least
    `bare_timer.MIN_REPS` repetitions on the card; behind the CPU boundary,
    `BOUNDARY_BARE_REPS`, whose CPU timings mean nothing."""
    from attnbench.components import bare_timer as bt
    if _boundary():
        return bt.Pricer(reps=BOUNDARY_BARE_REPS, sync=_sync(), min_reps=1)
    return bt.Pricer(reps=bt.MIN_REPS, sync=_sync())


def load_prices(pricer, path: Path):
    """The step's bare prices so far into `pricer`: one cache for the step,
    as when its cells shared a process, now that each has its own (ruling,
    2 Nov 2026). A signature is timed once per step, by the first cell that
    meets it."""
    import pickle
    path = Path(path)
    if path.exists():
        pricer.prices.update(pickle.loads(path.read_bytes()))
    return pricer


def save_prices(pricer, path: Path) -> None:
    import pickle
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(pickle.dumps(dict(pricer.prices)))
    os.replace(tmp, path)


def bare_profile_cell(a, rec, out: Path, null, *, pricer, tolerance: dict, entries=()):
    """The bare-timer prices of one recording (`bare_timer`; ruling of 1 Nov
    2026 after Shakedown 5): the prediction's host costs. `null` is the
    recording's null-pass profile, kept as the instrumented comparison. The
    chain gate is evaluated in the zoo; a cell that fails it falls back to
    each composite's own bare time and is labelled, never refused. Writes
    `host_profile.json` (bare), `host_profile_null_pass.json` and
    `bare_prices.json`; returns (bare profile, record)."""
    from attnbench.components import bare_timer as bt
    from attnbench.components import host_profile as hp
    bp = bt.bare_pass(a.run_prefill, rec.tape, rec.sequence, device=_null_device(),
                      pricer=pricer, entries=entries, layers=a.layers,
                      isolated_ops=_isolated_ops())
    bare, record = bt.bare_profile(rec.sequence, bp, null,
                                   aggregate_tol=float(tolerance["aggregate"]))
    out = Path(out)
    (out / "host_profile.json").write_text(hp.dumps(bare))
    (out / "host_profile_null_pass.json").write_text(hp.dumps(null))
    _write_json(out / "bare_prices.json", record)
    return bare, record


def residual_cell(a, rec, out: Path, bare, null, *, tolerance: float, entries=()):
    """The per-layer residual (`layer_residual`; ruling of 1 Nov 2026 after
    Shakedown 6, measured with the device idle since ruling B1 of 8 Nov
    2026): the host gaps between the prefill's calls per layer, each opened
    after a synchronise, under the tape alone, added back over the layer's
    positions. The declared host-side functions run bare inside their gaps
    (ruling 2 of 9 Nov 2026); on the CPU boundary, where the host is the
    device, none is. Writes `host_profile.json` (the bare prices with the
    residual: what the prediction reads), `host_profile_bare.json` and
    `layer_residual.json`; returns (profile, record)."""
    from attnbench.components import host_profile as hp
    from attnbench.components import layer_residual as lr
    dev = "cpu" if _boundary() else "cuda"
    host_side = () if _boundary() else lr.declared_host_side()
    gaps = lr.idle_gaps(a.run_prefill, rec.tape, a.layers, seq=rec.sequence, entries=entries,
                        host_side=host_side, device_type=dev, sync=_sync())
    prof, record = lr.residual(rec.sequence, bare, null, gaps, spread_tol=float(tolerance))
    record["host_side_functions"] = [f"{m.__name__}.{f}" for m, f in host_side]
    out = Path(out)
    (out / "host_profile_bare.json").write_text(hp.dumps(bare))
    (out / "host_profile.json").write_text(hp.dumps(prof))
    _write_json(out / "layer_residual.json", record)
    return prof, record


def code_path_definition(cp) -> dict:
    """T2.11's code-path definition (no arm) from a shakedown `CodePath`."""
    return {"gpu": cp.gpu, "implementation": cp.implementation,
            "libraries": [list(x) for x in cp.libraries], "python_optimize": cp.python_optimize}


def floor_ms_at(cfg, length: int, batch: int) -> float:
    """The resolution floor of the cell at (`length`, `batch`), ms: from the
    lock's per-(length, batch) table (`floors.FloorTable`, P1 only) when the
    config holds the table, else the config's one P1 working floor."""
    v = cfg["resolution_floor_ms"]
    if isinstance(v, dict):
        from attnbench.lock.floors import FloorTable
        return FloorTable.from_ms(v, pairing="P1").at(length, batch) * 1e3
    return float(v)


def session_floor_ms(cfg, session: Path, length: Optional[int] = None,
                     batch: Optional[int] = None) -> float:
    """The resolution floor the ground truth flags against at a cell: the
    lock's for that (length, batch) if the lock has set the table
    (`floor_ms_at`), else the session floor `dense_select` derived (P1)."""
    if cfg.get("resolution_floor_ms") is not None:
        if isinstance(cfg["resolution_floor_ms"], dict) and length is None:
            raise StepFailed("the lock's resolution floor is per (length, batch): name the cell")
        return floor_ms_at(cfg, length, batch) if length is not None \
            else float(cfg["resolution_floor_ms"])
    p = step_dir(session, "dense_select") / "session_floor.json"
    if not p.exists():
        raise StepFailed(f"no session floor at {p}: dense_select has not derived one")
    return float(json.loads(p.read_text())["floor_ms"])


def rerun_budget(cfg) -> int:
    """Extra rounds allowed for flagged ones: P1's none until it is decided
    (T3.2); a later pairing's is a lock value (T4.1 "rerun budget") and is
    refused when missing, never read as 0 (T4.3 read B, finding 4)."""
    v = cfg.get("max_rerun_rounds")
    if v is None:
        if cfg.get("pairing") != WORKING_PAIRING:
            raise StepFailed(f"{cfg.get('pairing')}'s rerun budget is a lock value (T4.1): "
                             f"max_rerun_rounds is missing")
        return 0
    return int(v)


# The in-session dense-drift limit for every pairing but P1: the 0.95 quantile
# of |drift| / warm-up time over T3.2a's and T3.2b's 120 rounds (Appendix A,
# 9 Nov 2026), chosen before any P2 data (adviser's ruling 1, fourth set of
# 12 Nov 2026). P1's sessions keep the absolute floor they ran under. The
# reference is the cell's running median after this many usable rounds, not
# the warm-up time (ruling 2 of the lock wave, 12 Nov 2026; declared in
# docs/drift_reference_correction.md before any record was re-read).
RELATIVE_DRIFT_LIMIT = 0.06311947327826187
DRIFT_REFERENCE_MIN_ROUNDS = 5


def drift_controls(cfg) -> dict:
    """`gt.Controls`' drift limit and clean-round minimum: P1's protocol (none
    of either) for P1, else the relative limit and the lock's minimum, which
    is refused when missing, as the rerun budget is."""
    if cfg.get("pairing") == WORKING_PAIRING:
        return {"drift_limit_relative": None, "drift_reference_min_rounds": None,
                "min_clean_rounds": None}
    v = cfg.get("min_clean_rounds")
    if v is None:
        raise StepFailed(f"{cfg.get('pairing')}'s minimum clean rounds is a lock value (T4.1 "
                         f"rerun budget): min_clean_rounds is missing")
    return {"drift_limit_relative": RELATIVE_DRIFT_LIMIT,
            "drift_reference_min_rounds": DRIFT_REFERENCE_MIN_ROUNDS, "min_clean_rounds": int(v)}


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
    """(clock skew, guard): the skew and the margin of `commit`, which with
    the gap (`_gap`) sum to PLANNED_SYNC_WAIT_S. Only a dry run's config may
    name its own."""
    if not dry_run:
        if "clock_skew_s" in cfg or "hold_guard_s" in cfg or "sync_gap_s" in cfg:
            raise StepFailed("a real session's hold is PLANNED_SYNC_WAIT_S; its config "
                             "may not set clock_skew_s, hold_guard_s or sync_gap_s")
        return so.REQUIRED_SKEW, so.PLANNED_GUARD
    return (timedelta(seconds=cfg.get("clock_skew_s", CLOCK_SKEW_ALLOWED_S)),
            timedelta(seconds=cfg.get("hold_guard_s", SYNC_WAIT_MARGIN_S)))


def _gap(cfg, dry_run: bool = False) -> timedelta:
    """The required gap after the sync (`commit`); a dry run's config may
    shorten it, so its hold does not wait out the real one."""
    _skew(cfg, dry_run)
    return so.REQUIRED_GAP if not dry_run else \
        timedelta(seconds=cfg.get("sync_gap_s", REQUIRED_SYNC_GAP_S))


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
    if bool(cfg.get("debug")) != prefix.endswith(("-debug", "-exploratory")):
        raise StepFailed(f"a debug or exploratory session's prediction goes under a '-debug' or "
                         f"'-exploratory' prefix, and only theirs: debug={cfg.get('debug')!r}, "
                         f"prefix {prefix!r}")
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
    "event_synchronize": {"s_fixed": 1e-5},
    "event_record_synchronize": {"s_fixed": 2e-5},
    "copy_from_host_pinned": {"s_fixed": 1e-5, "bandwidth": 2e10, "loaded_bandwidth": 1e10},
    "copy_from_host_pageable": {"s_fixed": 3e-5, "bandwidth": 5e9, "loaded_bandwidth": 2e9},
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
    refuses a dirty tree outside a dry run. Then the block-sparse kernel's
    contract (`kernel_contract`)."""
    import hashlib
    out = Path(args.out)
    data = json.dumps(cfg, indent=1, sort_keys=True).encode()
    (out / "session_config.json").write_bytes(data)
    _t227(args, "t21_t220", out / "t21_t220")
    return {"config_sha256": hashlib.sha256(data).hexdigest(), "cells": len(cells(cfg)),
            "kernel_contract": kernel_contract(args, out / "kernel_contract")}


def kernel_contract(args, out: Path) -> dict:
    """The block-sparse kernel's contract (`attnbench.bsa_contract`;
    researcher's ruling, 2 Nov 2026, after the XAttention crash): the pinned
    library, the adapter's mask buffer, the kernel against a reference masked
    attention on masks with every head's last row fully selected, and
    memcheck with and without the fix. Any refusal fails the step, so the
    session stops before any arm runs. A dry run has no kernel to check."""
    from attnbench import bsa_contract
    if args.dry_run:
        return {"status": "not run: dry run"}
    sanitizer = None
    if _cuda():
        from t31_xa_diag import find_sanitizer
        sanitizer = find_sanitizer()
    try:
        rep = bsa_contract.contract_check(out, sanitizer=sanitizer)
    except bsa_contract.ContractError as e:
        raise StepFailed(f"kernel contract: {e}",
                         json.loads((out / "kernel_contract.json").read_text())) from e
    return {"status": rep["status"], "clean": rep["clean"], "outputs": rep["outputs"],
            "real_prompt": rep.get("real_prompt"),
            "sanitizer": {k: rep["sanitizer"].get(k) for k in ("status", "shipped", "fixed")}}


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
                  "max_payload_bytes": link.payload_range(lrow)[1],
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
    from attnbench.lock import floors as F
    if cfg.get("resolution_floor_ms") is None and cfg.get("pairing") != F.FLOOR_PAIRING:
        # Nothing is clocked before `predict` (T4.3 read B, finding 1): a later
        # pairing's floor is the lock's P1 table, never derived here.
        raise F.FloorsError(f"{cfg.get('pairing')}'s resolution floor is the lock's P1 table "
                            f"(T4.1); it is null, so the lock was not filled. Nothing is "
                            f"clocked to derive it.")
    h = _harness(args)
    found = {}
    if cfg.get("resolution_floor_ms") is None:      # null as an input: derive it here (P1)
        floor = derive_session_floor(h, cfg, Path(args.out))
        cfg = dict(cfg, resolution_floor_ms=floor["floor_ms"])
        found["session_floor"] = floor
    for L in cfg["lengths"]:
        for b in cfg["batches"].get(str(L), [1]):
            cell = ds.Cell(card=cfg["card"], context_length=L, batch=b)
            attn_cfg = replace(h._cfg(ARMS["dense"], L), batch=b)
            choice = ds.select_dense_baseline(
                cell, attn_cfg, ds.registry_candidates(cfg["card"]),
                rounds=cfg["dense_rounds"], resolution_floor_ms=floor_ms_at(cfg, L, b),
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
    config's floor is null; a floor the lock has set is used as it is. Any
    pairing but P1 is refused before anything is run (ruling 1a, 11 Nov 2026)."""
    from attnbench.lock import floors
    floors._pairing(cfg["pairing"])
    import torch
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
    sequence, masks, tape), the null-kernel host profile from its tape
    (`HOST_PROFILE_REPS` passes, the instrumented comparison), then the
    bare-timer prices (`bare_profile_cell`; ruling of 1 Nov 2026 after
    Shakedown 5) and, added to them, each layer's residual, which the
    prediction reads (`residual_cell`; ruling of 1 Nov 2026 after Shakedown
    6; a layer over its spread tolerance is labelled, not refused). Writes `cells_index.json`, which the device
    step reads; a calibrated arm's density is the mask density it produced
    on the cell's prompt. A cell that fails the chain gate is labelled with
    its fallback, not refused (oracle re-evaluation and fallback labelling,
    the same ruling).

    **Each cell in its own process** (ruling, 2 Nov 2026, after T3.1's third
    session): `host-cell-one` under `resource_guard.run_guarded`, killed at
    the RSS ceiling (`HOST_CELL_RSS_CEILING_GB`). The parent writes each
    cell's `cell_manifest.json`: the cell, the code identity and instance
    spec, the status, and the peak RSS (sampled, and the child's own). A
    cell that fails or crosses the ceiling stops the step, as any failure
    does. A cell whose manifest says it passed under the same code identity
    (a clean tree) and instance spec is not run again; any other cell
    directory is cleared and the cell run afresh, so no cells of two code
    identities are mixed."""
    if args.dry_run:
        return {"fake": True}
    _need_gpu("the null-kernel host profile")
    out = Path(args.out)
    run = guarded_cells(args, cfg, cells(cfg), spawn=spawn_cell)
    index = run["index"]
    _write_json(out / "cells_index.json", {"cells": index})
    prices = load_prices(SimpleNamespace(prices={}), out / PRICE_CACHE).prices
    # every cell its own key: arm, nominal density, budget, batch (the
    # density was missing, so a 0.25 cell's entry was overwritten by its 0.5
    # cell's in both T3.2 repeats)
    name = lambda c: cell_dir(Path(), c["arm"], c["nominal_density"], c["budget"],  # noqa: E731
                              c["batch"]).name
    return {"cells": len(index), "signatures_timed": len(prices),
            "resumed": run["resumed"], "peak_rss_mb": run["peak_rss_mb"],
            "rss_ceiling_gb": run["rss_ceiling_gb"],
            "chain_gate_fallback_cells": [name(c) for c in index
                                          if c["chain_gate"]["fallback"]],
            "residual_labelled_cells": {name(c): c["layer_residual"]["labels"] for c in index
                                        if c["layer_residual"]["labels"]},
            "residual_added_ms": {name(c): c["layer_residual"]["residual_added_s"] * 1e3
                                  for c in index},
            "instrumented_over_bare": {name(c): c["chain_gate"]["instrumented_over_bare"]
                                       for c in index}}


def guarded_cells(args, cfg, cell_list, *, spawn, inputs=None) -> dict:
    """Steps 5 and 6: each cell of `cell_list` in its own process (`spawn`)
    under the RSS ceiling, in order, stopping at the first that fails or
    crosses it. Writes each cell's `cell_manifest.json` (the cell, the code
    identity, the instance spec, `inputs(cell)` if given, the status, the
    peak RSS); a cell that passed under the same identity, instance and
    inputs, from a clean tree, is not run again (`resumable`), and any other
    cell directory is cleared first. A failing child's own findings (a
    `StepFailed`'s, through `cell_exit.json`) are raised with the manifest.
    Returns the cells' entries (`cell_index.json`), in order, the cells
    resumed, the peaks and the ceiling. In a debug session a failing cell
    does not stop the step: every cell runs and the step fails at the end,
    naming each."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ident, spec = cell_identity(), instance_spec()
    accept = staged_accept(args.session, cfg)
    ceiling_gb = float(cfg.get("host_cell_rss_ceiling_gb") or HOST_CELL_RSS_CEILING_GB)
    index, resumed, peaks, failed = [], [], {}, {}
    for cell in cell_list:
        cd = cell_dir(out, *cell)
        name = cd.name
        given = inputs(cell) if inputs else None
        done = resumable(cd, ident, spec, given, accept, debug=bool(cfg.get("debug")))
        if done is not None:
            index.append(done)
            resumed.append(name)
            peaks[name] = json.loads((cd / CELL_MANIFEST).read_text()).get("peak_rss_mb")
            continue
        if cd.exists():
            shutil.rmtree(cd)
        cd.mkdir(parents=True)
        manifest = {"cell": dict(zip(("arm", "density", "length", "batch"), cell)),
                    "identity": ident, "instance": spec,
                    **({"inputs": given} if inputs else {}), "status": "running",
                    "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _write_json(cd / CELL_MANIFEST, manifest)
        guarded = spawn(args, cfg, cell, cd, ceiling_mb=ceiling_gb * 1024)
        exit_ = _read_json(cd / CELL_EXIT) or {}
        ok = guarded["returncode"] == 0 and (cd / CELL_INDEX).exists() and not exit_.get("error")
        status = "passed" if ok else ("over_rss_ceiling" if guarded["over_ceiling"] else "failed")
        self_peak = exit_.get("self_peak_rss_mb")
        manifest.update(guarded, status=status, self_peak_rss_mb=self_peak,
                        peak_rss_mb=max(x for x in (self_peak, guarded["sampled_peak_rss_mb"])
                                        if x is not None),
                        error=exit_.get("error") or (None if ok else (
                            f"killed by the guard at the {ceiling_gb:g} GiB RSS ceiling"
                            if guarded["over_ceiling"] else
                            f"exit {guarded['returncode']}; see {cd.name}/cell.log")))
        _write_json(cd / CELL_MANIFEST, manifest)
        peaks[name] = manifest["peak_rss_mb"]
        if not ok and cfg.get("debug"):
            # a debug session's job is every failure at once (2 Nov 2026 ruling): the
            # cell's failure is recorded, the others run, the step fails at the end
            failed[name] = {"error": manifest["error"], "status": status,
                            "findings": exit_.get("findings")}
            continue
        if not ok:
            raise StepFailed(f"{name}: {manifest['error']}",
                             {**(exit_.get("findings") or {}),
                              "cell_manifest": manifest, "cells_passed": len(index),
                              "resumed": resumed, "peak_rss_mb": peaks})
        index.append(json.loads((cd / CELL_INDEX).read_text()))
    if failed:
        raise StepFailed(f"{len(failed)} of {len(cell_list)} cells failed: " + "; ".join(
            f"{n}: {f['error']}" for n, f in failed.items()),
            {"cells_failed": failed, "cells_passed": len(index), "resumed": resumed,
             "peak_rss_mb": peaks})
    return {"index": index, "resumed": resumed, "peak_rss_mb": peaks,
            "rss_ceiling_gb": ceiling_gb}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cell_identity() -> dict:
    """What a cell's result depends on in code: the commit, whether the tree
    is clean, and every library a sequence can depend on (the BSA fix
    included, `shakedown_arms.libraries`)."""
    from attnbench import provenance
    from attnbench.components.shakedown_arms import libraries
    commit, dirty = provenance._git_state()
    return {"git_commit": commit, "git_dirty": dirty,
            "libraries": [list(x) for x in libraries()]}


def instance_spec() -> dict:
    """What a cell's result depends on in hardware: the machine, its image,
    the GPU and the host's memory."""
    from attnbench import provenance
    cloud = provenance._cloud_identity()
    gpu = None
    if _boundary():
        gpu = "cpu-boundary"
    elif _cuda():
        import torch
        gpu = torch.cuda.get_device_name(0)
    mem = None
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem = round(int(line.split()[1]) / 2**20, 1)
    return {k: cloud.get(k) for k in ("cloud_provider", "instance_type", "boot_image",
                                      "provisioning_model")} | {"gpu": gpu, "mem_total_gib": mem}


def staged_accept(session, cfg) -> dict | None:
    """In a debug session, the identity and instance of the cells it staged
    from an earlier session (`staged_inputs.json`), with each cell's entry
    sha256; otherwise None. A staged cell resumes under its own donor
    identity, as no cell of another identity does anywhere else (2 Nov 2026
    ruling: a debug session's data is discarded)."""
    if not cfg.get("debug"):
        return None
    staged = _read_json(Path(session) / STAGED_INPUTS)
    if not staged:
        raise StepFailed(f"a debug session needs {STAGED_INPUTS} in {session}")
    if staged.get("donor") is None:          # an exploratory session stages nothing
        return None
    return {"identity": staged["donor"]["identity"], "instance": staged["donor"]["instance"],
            "cells": {c["dir"]: c["cell_index_sha256"] for c in staged["cells"]}}


def resumable(cd: Path, ident: dict, spec: dict, inputs=None, accept=None, *,
              debug: bool = False):
    """The cell's entry (`cell_index.json`) if it passed under this code
    identity, instance spec and inputs, from a clean tree; else None. In a
    debug session (`accept`, from `staged_accept`) a cell it staged also
    resumes under its donor's identity and instance, if its entry is the one
    staged. A debug session's own passed cells also resume under an earlier
    commit of its fixes (`debug`: same libraries and instance, a clean tree;
    each fix would otherwise repeat every cell, 2 Nov 2026)."""
    cd = Path(cd)
    m = _read_json(cd / CELL_MANIFEST)
    if not m or m.get("status") != "passed" or m.get("inputs") != inputs:
        return None
    mi = m.get("identity") or {}
    if debug and not (accept and mi == accept["identity"]) \
            and mi.get("git_dirty") is False \
            and mi.get("git_commit") \
            and mi.get("libraries") == ident.get("libraries") and m.get("instance") == spec:
        return _read_json(cd / CELL_INDEX)
    if m.get("identity") == ident and m.get("instance") == spec:
        if not ident.get("git_commit") or ident.get("git_dirty") is not False:
            return None
        return _read_json(cd / CELL_INDEX)
    if accept and cd.name in accept["cells"] and m.get("identity") == accept["identity"] \
            and m.get("instance") == accept["instance"] \
            and _sha256(cd / CELL_INDEX) == accept["cells"][cd.name]:
        return _read_json(cd / CELL_INDEX)
    return None


def _density_arg(d) -> str:
    return CALIBRATED if d is None else repr(float(d))


def cell_command(args, cell, child: str = "host-cell-one") -> list[str]:
    arm, d, L, b = cell
    return _py("t31_steps.py", child, "--out", str(args.out),
               "--session", str(args.session), "--arm", arm, "--density", _density_arg(d),
               "--length", str(L), "--batch", str(b),
               *(["--cpu-boundary"] if getattr(args, "cpu_boundary", False) else []),
               *(["--config", args.config] if getattr(args, "config", None) else []))


def spawn_cell(args, cfg, cell, cd: Path, *, ceiling_mb: float,
               child: str = "host-cell-one") -> dict:
    """One cell in its own process under the RSS ceiling; its output in
    `cell.log`. Returns `resource_guard.Guarded.record()`. The child reads
    the step's config from the same file (`--config`, else the default), so
    `cfg` is not passed on."""
    from attnbench import resource_guard
    cmd = cell_command(args, cell, child)
    print("$ " + " ".join(cmd), flush=True)
    with open(Path(cd) / "cell.log", "w") as log:
        g = resource_guard.run_guarded(cmd, ceiling_mb=ceiling_mb, log=log, cwd=ROOT)
    return g.record()


def pricing_sampled() -> bool:
    """Whether steps 5 and 6 sample NVML (`pricing_clocks`; ruling 4b of 9 Nov
    2026): on a real GPU, never behind the CPU boundary."""
    import torch
    return torch.cuda.is_available() and not _boundary()


SOAK_S, SOAK_MIN_REPLAYS = 5.0, 2


def soak(cap, *, seconds: float = SOAK_S, min_replays: int = SOAK_MIN_REPLAYS,
         clock=time.monotonic) -> dict:
    """The sustained-clock protocol (ruling 3 of 9 Nov 2026): the captured
    replay run back to back, each one synchronised, until `seconds` have passed
    and at least `min_replays` ran, so the replay step 6 prices from finds
    the device in the sustained state the timed rounds run in (several
    seconds of back-to-back prefills under the software power cap)."""
    t0, n = clock(), 0
    while n < min_replays or clock() - t0 < seconds:
        cap.replay()
        cap.backend.synchronize()
        n += 1
    return {"replays": n, "seconds": round(clock() - t0, 3)}


def null_entries(arm: str, code_path, backend=None) -> tuple:
    """The entry points step 5's passes null for a cell (`host_profile.
    NullEntry`): none for dense; the block-sparse kernel's extension call for
    a sparse arm; on XAttention's Triton path its Triton kernels as well
    (P2's step 7, 12 Nov 2026: unnulled, they ran for real and the structure
    check refused the cell), in the module the cell's own backend launches
    them from (`kernels_module`; 13 Nov 2026: the asserts-removed arm's are
    its stripped copy's, which nulling `xattn.src.kernels` missed)."""
    from attnbench.components import host_profile as hp
    if arm == "dense":
        return ()
    entries = (hp.block_sparse_entry(),)
    if code_path.implementation == XATTN_TRITON:
        kernels = getattr(backend, "kernels_module", None) or \
            getattr(getattr(backend, "arm", None), "kernels_module", None)   # a compiled arm's
        if kernels is None:
            raise StepFailed(f"{arm} took {XATTN_TRITON}, but its backend names no kernels "
                             f"module to null")
        entries += hp.xattention_triton_entries(kernels())
    return entries


def host_cell(h, cfg, out: Path, cell, pricer) -> dict:
    """One cell of step 5, in this process: its `cells_index.json` entry."""
    from attnbench.components import bare_timer as bt
    from attnbench.components import host_profile as hp
    from attnbench.components import layer_residual as lr
    from attnbench.components import recorder
    from attnbench import pricing_clocks
    arm, d, L, b = cell
    cd = cell_dir(out, arm, d, L, b)
    a = ready_cell(h, arm, d, L, b)
    seconds, t0 = {}, time.time()
    entries = null_entries(arm, a.code_path, a.backend)
    with pricing_clocks.PricingClocks(cd, enabled=pricing_sampled()) as pc:
        with pc.phase("record"):
            rec = recorder.record_arm_prefill(a.run_prefill, cd / "recording",
                                              code_path=a.code_path, context_length=a.length,
                                              batch=b, prompt_id=a.prompt_id, layers=a.layers)
        seconds["record"], t0 = time.time() - t0, time.time()
        with pc.phase("isolated_and_null_pass"):
            prof, check = host_profile_cell(
                a, rec, cd, reps=HOST_PROFILE_REPS, entries=entries,
                tolerance=cfg["chain_check_tolerance"],
                segment_layers=int(cfg.get("isolated_segment_layers", ISOLATED_SEGMENT_LAYERS)))
        seconds["isolated_and_null_pass"], t0 = time.time() - t0, time.time()
        with pc.phase("bare_pass"):
            bare, record = bare_profile_cell(a, rec, cd, prof, pricer=pricer, entries=entries,
                                             tolerance=cfg["chain_check_tolerance"])
        seconds["bare_pass"], t0 = time.time() - t0, time.time()
        with pc.phase("residual"):
            priced, resid = residual_cell(a, rec, cd, bare, prof, entries=entries,
                                          tolerance=cfg["residual_spread_tolerance"])
        seconds["residual"] = time.time() - t0
    realised = d
    if d is None:
        per_layer = list(getattr(a.backend, "last_layer_density", None) or ())
        if not per_layer:
            raise StepFailed(f"{arm}: its backend keeps no mask density, so its cell "
                             f"has no density")
        realised = round(sum(float(x) for x in per_layer) / len(per_layer), 6)
    return {"arm": arm, "nominal_density": d, "density": realised, "budget": L,
            "context_length": a.length, "batch": b, "prompt_id": a.prompt_id,
            "dir": str(cd.relative_to(out)),
            "code_path": code_path_definition(a.code_path),
            "host_total_ms": priced.total_s * 1e3,
            "host_total_bare_ms": bare.total_s * 1e3,
            "host_total_null_pass_ms": prof.total_s * 1e3,
            "layer_residual": lr.summary(resid),
            "seconds": {k: round(v, 1) for k, v in seconds.items()},
            "host_cost_sources": prof.sources(),
            "price_routes": record["routes"],
            "chain_gate": bt.gate_summary(record),
            "chain_check_instrumented": {
                k: check[k] for k in ("passed", "gated", "share", "aggregate_tol",
                                      "checked_calls", "unchecked_calls")}}


def run_cell(args, cfg, cell):
    """One cell of step 5 in this process: the bare prices the step has so
    far, the cell, the prices saved back, then its entry (`cell_index.json`).
    Always writes `cell_exit.json` (`_cell_body`); returns the error."""
    return _cell_body(_host_cell_entry, args, cfg, cell)


def _host_cell_entry(args, cfg, cell) -> dict:
    out = Path(args.out)
    _need_gpu("the null-kernel host profile")
    h = _harness(args)
    pricer = load_prices(bare_pricer(), out / PRICE_CACHE)
    entry = host_cell(h, cfg, out, cell, pricer)
    h.unwrap()
    save_prices(pricer, out / PRICE_CACHE)
    return entry


def _cell_body(entry, args, cfg, cell):
    """Run a child's cell (`entry(args, cfg, cell)`): its entry to
    `cell_index.json`, and always `cell_exit.json` with this process's own
    peak RSS, the error, if any, and a `StepFailed`'s findings, which the
    parent raises. Returns the error."""
    from attnbench import resource_guard
    cd = cell_dir(Path(args.out), *cell)
    error, findings = None, None
    try:
        _write_json(cd / CELL_INDEX, entry(args, cfg, cell))
    except StepFailed as e:
        error, findings = str(e), e.findings or None
    except Exception as e:
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"
    _write_json(cd / CELL_EXIT, {"self_peak_rss_mb": round(resource_guard.self_peak_mb(), 1),
                                 "error": error, "findings": findings})
    return error


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
    zero device cost. Each cell runs in its own process under the RSS
    ceiling, with manifests and resume as step 5's (`device_components_real`).

    The step then gates the pricing per arm (`pricing_gate`, researcher's
    ruling 1 of 6 Nov 2026): an arm whose share of priced device time priced
    from the isolated original is above the T4.1 working value, or with a
    position neither replay matches, is labelled "pricing not validated" in
    `cells.json` and `pricing_gate.json`, and step 7 leaves it out of the
    prediction. The step itself does not fail on it."""
    if not args.dry_run:
        return device_components_real(args, cfg)
    entries = []
    for a, d, L, b in cells(cfg):
        ops = fake_ops(a, d, L, b)
        entries.append({"arm": a, "density": d, "context_length": L, "batch": b,
                        "code_path": FAKE_CODE_PATH, "ops": [op_to_dict(op) for op in ops],
                        "pricing": {"priced_s": sum(o.device for o in ops), "mismatch_s": 0.0,
                                    "unpriced": []}})
    gate = pricing_gate(Path(args.out), cfg, entries, fake=True)
    return {"fake": True, "cells": len(entries), "pricing_gate": gate_brief(gate)}


def pricing_gate(out: Path, cfg, entries, *, fake: bool) -> dict:
    """The aggregate pricing gate over step 6's entries (`replay.pricing_gate`
    at the working value `pricing_mismatch_share`, T4.1 row 35); writes it
    to `pricing_gate.json` and into `cells.json` beside the entries. A dry
    run's own config does not carry it; it takes the table's."""
    from attnbench.components import replay
    share = cfg.get("pricing_mismatch_share")
    if share is None:
        share = working_values()[0]["pricing_mismatch_share"]
    gate = replay.pricing_gate(entries, float(share))
    _write_json(out / "pricing_gate.json", gate)
    _write_json(out / "cells.json", {"fake": fake, "cells": entries, "pricing_gate": gate})
    return gate


def gate_brief(gate: dict) -> dict:
    """Each arm's share, whether it is priced, and the stand-in nodes step 6
    dropped (counted, deciding nothing: the adviser's ruling 3, second set of
    12 Nov 2026), for the step's result."""
    return {arm: {"share": g["share"], "validated": g["validated"],
                  "dropped_nodes": g["dropped_nodes"], "dropped_s": g["dropped_s"],
                  **({"label": g["label"], "why": g["why"]} if not g["validated"] else {})}
            for arm, g in gate.items()}


def device_components_real(args, cfg) -> dict:
    """Step 6's parent: each of step 5's cells in its own process
    (`device-cell-one`, `guarded_cells`; the researcher's answer of 2 Nov
    2026, after the step-5 fix, to build the same isolation for step 6). A
    cell's inputs are its host entry and the link row, by sha256, so a
    resumed cell is one priced from the same host step."""
    _need_gpu("the CUDA-graph replay")
    out, session = Path(args.out), Path(args.session)
    host_dir = step_dir(session, "host_components")
    index = json.loads((host_dir / "cells_index.json").read_text())["cells"]
    link = step_dir(session, "link") / "link" / "link.parquet"

    def inputs(cell):
        return {"host_cell_index_sha256": _sha256(cell_dir(host_dir, *cell) / CELL_INDEX),
                "link_sha256": _sha256(link)}
    dropped = not_measured(cfg)      # a debug session names more than step 5 ran
    run = guarded_cells(args, cfg, [k for k in map(device_cell_key, index) if k not in dropped],
                        spawn=spawn_device_cell, inputs=inputs)
    entries = run["index"]
    gate = pricing_gate(out, cfg, entries, fake=False)
    return {"cells": len(entries), "pricing_gate": gate_brief(gate),
            "dropped_nodes": sum(g["dropped_nodes"] for g in gate.values()),
            "cost_sources": {cell_dir(Path(), *device_cell_key(e)).name: e["cost_sources"]
                             for e in entries},
            "isolated_segments": segment_findings(entries),
            "resumed": run["resumed"], "peak_rss_mb": run["peak_rss_mb"],
            "rss_ceiling_gb": run["rss_ceiling_gb"]}


def _sha256(path: Path):
    import hashlib
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def device_cell_key(c: dict) -> tuple:
    """A host or device entry's cell: (arm, density setting, length, batch),
    as `cells` gives it and `cell_dir` names it."""
    return (c["arm"], c["nominal_density"], int(c["budget"]), int(c["batch"]))


def spawn_device_cell(args, cfg, cell, cd: Path, *, ceiling_mb: float) -> dict:
    return spawn_cell(args, cfg, cell, cd, ceiling_mb=ceiling_mb, child="device-cell-one")


def memset_zoo(out: Path) -> dict[int, float]:
    """The memset zoo's durations, measured once for the step and kept in
    `memset_zoo.json`, which the step's later cells read."""
    from attnbench.components import memset_flags as msf
    path = Path(out) / MEMSET_ZOO
    kept = _read_json(path)
    if kept is not None:
        return {int(k): float(v) for k, v in kept.items()}
    zoo = msf.measure_zoo(Path(out) / "memset_zoo_trace.json")
    tmp = path.with_name(path.name + ".part")
    _write_json(tmp, {str(k): v for k, v in zoo.items()})
    os.replace(tmp, path)
    return zoo


def isolated_original(a, rec, cd: Path) -> tuple[Path, dict]:
    """Step 6's isolated replay of the original operators as a compact stream
    with the device fields `replay.replay_nodes` reads (researcher's ruling 3
    of 6 Nov 2026: its 3 GB trace at `oracle_d0.25_L32768` was killed at
    26 GiB in a full read), in blocks of `ISOLATED_SEGMENT_LAYERS` layers as
    step 5's; a prefill of no more layers than that is one block, compacted
    after its export. (stream, counts)."""
    from attnbench.components import replay, trace_stream
    path = Path(cd) / "isolated_original.json"
    if len(a.layers) > ISOLATED_SEGMENT_LAYERS:
        return replay.isolated_replay_compact(
            a.run_prefill, rec.tape, path, device=_device(), layers=a.layers,
            segment_layers=ISOLATED_SEGMENT_LAYERS, masks=rec.masks, device_nodes=True)
    trace = replay.isolated_replay(a.run_prefill, rec.tape, path, device=_device(),
                                   masks=rec.masks, substitute=False)
    info = trace_stream.write_compact(trace, cats=trace_stream.REPLAY_CATS,
                                      fields=trace_stream.DEVICE_FIELDS)
    Path(trace).unlink()
    stream = Path(info.pop("path"))
    return stream, {"stream": stream.name, **info}


def match_original(seq, orig, counts) -> tuple:
    """The isolated original's match and, for a replay in blocks of layers,
    each segment's report (researcher's ruling 2 of 7 Nov 2026): matched
    segment by segment (`replay.match_segmented`), so no stretch runs across
    a cut and a loss is placed in the segment that lost it, with each
    segment's launches no device activity answered. (match, report or None)."""
    from attnbench.components import replay
    nodes = replay.replay_nodes(orig, exclude=replay.FLUSH)
    segments = replay.segments_from_counts(counts or {})
    if segments is None:
        return replay.match(seq, nodes), None
    m = replay.match_segmented(seq, nodes, segments)
    return m, replay.segment_report(
        m, segments, replay.launches_without_activity(orig, exclude=replay.FLUSH))


def segment_findings(entries) -> dict:
    """For the step's result (pulled with the logs): each cell replayed in
    segments, its count of segments and those with an unmatched position or
    a launch with no device activity."""
    return {cell_dir(Path(), *device_cell_key(e)).name: {
                "segments": len(e["isolated_segments"]),
                "with_loss": [s for s in e["isolated_segments"]
                              if s["unmatched"] or s["launches_without_activity"]]}
            for e in entries if e.get("isolated_segments")}


def device_cell(h, out: Path, session: Path, cell) -> dict:
    """One cell of step 6, in this process: its `cells.json` entry. Every
    device position is priced by structure and labelled (`replay.
    price_positions`, researcher's ruling 1 of 6 Nov 2026); the isolated
    original is replayed in blocks of layers into a compact stream and read
    lazily (ruling 3). Nothing here refuses the pricing: the entry carries
    its summary, and the step's aggregate gate decides per arm. A cell with a
    position neither replay matches has no operations."""
    import pandas as pd
    from attnbench import pricing_clocks
    from attnbench.components import recorder
    host_dir = step_dir(session, "host_components")
    name = cell_dir(Path(), *cell).name
    c = json.loads((host_dir / name / CELL_INDEX).read_text())
    row = pd.read_parquet(step_dir(session, "link") / "link" / "link.parquet").iloc[0].to_dict()
    cd = out / c["dir"]
    cd.mkdir(parents=True, exist_ok=True)
    rec = recorder.load_recording(host_dir / c["dir"] / "recording")
    a = ready_cell(h, c["arm"], c["nominal_density"], c["budget"], c["batch"])
    with pricing_clocks.PricingClocks(cd, enabled=pricing_sampled()) as pc:
        return _device_cell_priced(pc, a, c, rec, row, cd, out, host_dir)


def _device_cell_priced(pc, a, c, rec, row, cd: Path, out: Path, host_dir: Path) -> dict:
    """`device_cell` from the capture on, each phase marked for the pricing
    clocks (`pricing_clocks`); the profiled replay follows the soak (`soak`)."""
    import torch
    from attnbench.components import host_profile as hp
    from attnbench.components import memset_flags as msf
    from attnbench.components import op_sequence as ops
    from attnbench.components import replay
    with pc.phase("capture"):
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
    with pc.phase("soak"):     # a dry run's fake capture has no device to soak
        soaked = soak(cap) if torch.cuda.is_available() or _boundary() else \
            {"replays": 0, "seconds": 0.0, "skipped": "no device"}
    with pc.phase("profiled_replay"):
        trace = replay.profile_replay(cap, cd / "replay_trace.json")
    m = replay.match(rec.sequence, replay.replay_nodes(trace))
    original, stream, segments = None, None, None
    if replay.needs_original(rec.sequence, m):
        torch.cuda.synchronize()
        with pc.phase("isolated_original"):
            orig, stream = isolated_original(a, rec, cd)
        original, segments = match_original(rec.sequence, orig, stream)
    pricing = replay.price_positions(rec.sequence, m, original)
    summary = {**pricing.summary(rec.sequence),
               **({"isolated_original_stream": stream} if stream else {})}
    head = {"arm": c["arm"], "density": c["density"],
            "nominal_density": c["nominal_density"],
            "context_length": c["context_length"], "budget": c["budget"],
            "batch": c["batch"], "prompt_id": c["prompt_id"],
            "code_path": c["code_path"], "cost_sources": pricing.counts(),
            "pricing": summary, "soak": soaked,
            **({"isolated_segments": segments} if segments else {})}
    if pricing.unpriced:
        return {**head, "ops": None}
    prof = hp.from_record(json.loads((host_dir / c["dir"] / "host_profile.json").read_text()))
    costs = dict(host=hp.host_costs(prof, rec.sequence), device=pricing.device_s,
                 link=link_costs(rec.sequence, row))
    priced = label_outside_layers(ops.to_operations(rec.sequence, **costs))
    memsets = None
    if msf.count(rec.sequence):
        zoo = memset_zoo(out)
        trace = rec.directory / "trace.json"
        sizes = (msf.realised_sizes(trace, rec.sequence) if trace.exists()
                 else {i: None for i in msf.flagged(rec.sequence)})
        upper, src = msf.upper_bound_s(sizes, zoo)
        memsets = {"count": len(upper), "sources": sorted(set(src.values())),
                   "ops_upper": [op_to_dict(op) for op in label_outside_layers(
                       ops.to_operations(rec.sequence, **costs, flagged_memset_s=upper))]}
    return {**head, "ops": [op_to_dict(op) for op in priced],
            "event_syncs": event_sync_rule(rec),
            **({"memset_flags": memsets} if memsets else {})}


def event_sync_rule(rec) -> list:
    """`event_rule.kinds` of the cell's recording, one per event synchronise
    (ruling 4a of 9 Nov 2026); the trace is read only when there is one."""
    from attnbench.components import event_rule
    if not any(op.mechanism == event_rule.EVENT_SYNCHRONIZE for op in rec.sequence.ops):
        return []
    return event_rule.kinds(rec.directory / "trace.json")


def run_device_cell(args, cfg, cell):
    """One cell of step 6 in this process (`device_cell`); always writes
    `cell_exit.json` (`_cell_body`); returns the error."""
    return _cell_body(_device_cell_entry, args, cfg, cell)


def _device_cell_entry(args, cfg, cell) -> dict:
    _need_gpu("the CUDA-graph replay")
    h = _harness(args)
    entry = device_cell(h, Path(args.out), Path(args.session), cell)
    h.unwrap()
    return entry


# The smoke's stand-ins for the link step, which runs after the preflight: its
# prices are discarded, so only their shape matters (the dry run's table).
SMOKE_LINK_ROW = {f"bw_{d}_{m}_Bps": 1e10 for d in ("h2d", "d2h") for m in ("pinned", "pageable")}
SMOKE_RECURSION_INPUTS = {"queue_depth": 8,
                          "hardware": {"pinned_bandwidth": 2e10, "pageable_bandwidth": 5e9},
                          "by_length": {str(SMOKE_LENGTH): FAKE_MECHANISMS}}


def smoke_cells(cfg) -> list:
    """The smoke's cells: every arm of the session at 2048, batch 1, at its
    first configured density (a plan default; the code path does not depend
    on the density), the dense control first."""
    out = [(arm, nominal_density(cfg["densities"][arm][0]), SMOKE_LENGTH, SMOKE_BATCH)
           for arm in cfg["arms"]]
    if not out or out[0][0] != "dense":
        raise StepFailed(f"the smoke needs the dense arm first; the session's arms are "
                         f"{cfg['arms']}")
    return out


def preflight_smoke(args, cfg) -> dict:
    """The preflight's smoke check (the adviser's rulings dated 13 Nov 2026,
    items 3 and 5): every arm of the session at 2048, batch 1, before anything
    long, untimed, each through the code the session's later steps run:

      1. step 7's whole cell (`host_cell`: the recording, the isolated replay,
         the null pass and its structure check, the bare pass, the residual);
      2. step 8's whole cell on that recording (`_device_cell_priced`: the
         graph replay, its live masks against the bank, the match, the
         isolated original if needed, the pricing by structure), then its
         pricing check (`replay.pricing_gate`) with the arm's mismatch share,
         reported, as step 8 reports it;
      3. the prediction's input from that cell (`predict_cell`), on the
         stand-in link inputs above;
      4. the ground truth's harness (`ground_truth_smoke`): one untimed
         forward of every dense candidate and every arm, untraced and
         traced, under the probe, then the density twin.

    No timer runs and every output is discarded: these calls are not clocked
    blocks under the stamp rule (`docs/smoke_untimed_note_2026-11-13.md`).
    A failure stops the session in preflight. An XAttention cell that did not
    take the Triton path is refused, except behind the CPU boundary."""
    import torch
    from attnbench import pricing_clocks
    from attnbench.components import recorder
    from attnbench.components import replay
    from attnbench.predict.training import HardwareFeatures
    _need_gpu("the preflight smoke check")
    out = Path(args.out)
    share = cfg.get("pricing_mismatch_share")
    if share is None:
        share = working_values()[0]["pricing_mismatch_share"]
    hw = HardwareFeatures(**SMOKE_RECURSION_INPUTS["hardware"])
    h = _harness(args)
    cells, devs, found = smoke_cells(cfg), [], {"cells": {}}
    for cell in cells:
        arm = cell[0]
        t0 = time.time()
        entry = host_cell(h, cfg, out, cell, bare_pricer())                    # step 7
        impl = entry["code_path"]["implementation"]
        if _is_xattention(arm) and impl != XATTN_TRITON and not _boundary():
            raise StepFailed(f"{entry['dir']} took {impl}, not {XATTN_TRITON}")
        t1 = time.time()
        rec = recorder.load_recording(out / entry["dir"] / "recording")
        a = ready_cell(h, *cell)                                             # as step 8 readies it
        dd = out / "device" / entry["dir"]
        dd.mkdir(parents=True, exist_ok=True)
        with pricing_clocks.PricingClocks(dd, enabled=False) as pc:
            dev = _device_cell_priced(pc, a, entry, rec, SMOKE_LINK_ROW, dd, out / "device", out)
        devs.append(dev)
        t2 = time.time()
        predicted = None
        if dev["ops"] is not None:
            c = predict_cell(cfg["pairing"], cfg["model"], dev, SMOKE_RECURSION_INPUTS, hw)
            predicted = {"operations": len(dev["ops"]), "priced": len(c.priced)}
        found["cells"][entry["dir"]] = {
            "implementation": impl, "host_cost_sources": entry["host_cost_sources"],
            "entry_points": sum(1 for op in rec.sequence.ops if _entry_point(op)),
            "device_cost_sources": dev["cost_sources"],
            "mismatch_s": dev["pricing"]["mismatch_s"], "priced_s": dev["pricing"]["priced_s"],
            "unpriced": len(dev["pricing"]["unpriced"]), "predict_input": predicted,
            "seconds": {"step_7": round(t1 - t0, 1), "step_8": round(t2 - t1, 1),
                        "predict_input": round(time.time() - t2, 1)}}
        del rec, a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    found["pricing_gate"] = gate_brief(replay.pricing_gate(devs, float(share)))
    t3 = time.time()
    found["ground_truth"] = ground_truth_smoke(h, cfg, cells)
    found["ground_truth"]["seconds"] = round(time.time() - t3, 1)
    h.unwrap()
    _write_json(out / "smoke.json", found)
    return found


def _is_xattention(arm: str) -> bool:
    from attnbench.components.shakedown_arms import ARMS
    return ARMS[arm].xattention


def smoke_controls(card: str) -> dict:
    """The dense candidates the ground truth's control can be (K2 chooses
    among them in `dense_select`), by name; behind the CPU boundary, the CPU
    card's. A planned FA3 that is not available is refused: P4 depends on it
    (T2.16), and its import runs before the preflight."""
    from attnbench import dense_selection as ds
    planned = ds.registry_candidates(ds.CPU_CARD if _boundary() else card)
    if ds.FA3 in planned and planned[ds.FA3] is None:
        raise StepFailed(f"{ds.FA3} is planned on {card} and not available")
    return {n: be for n, be in planned.items() if be is not None}


def ground_truth_smoke(h, cfg, cells) -> dict:
    """Step 12's harness on the smoke's cells, untimed: the arms as
    `time_arms` takes them (`Harness.timing_arms`), each dense candidate as
    the control, then every arm under the probe with its setup
    (`untimed_setup`), one forward untraced and one inside the session's
    tracer, and the density twin (`kernel_densities`). No timer, no stamp,
    every output discarded. A deployable arm whose forward the probe saw
    compute no scores, or that scored in its setup, is refused, as
    `check_timed_region` refuses it."""
    import torch
    from attnbench import ground_truth as gt
    sync = torch.cuda.synchronize if (torch.cuda.is_available() or _boundary()) else (lambda: None)
    names = [c[0] for c in cells]
    dens = [None] + [c[1] for c in cells[1:]]
    wrapped, ids, pid, arms, score_kwargs = h.timing_arms(names, SMOKE_LENGTH, densities=dens,
                                                          batch=SMOKE_BATCH)
    trace = _tracer()
    found = {"prompt_id": pid, "controls": [], "arms": {}}

    def forward(backend, a, kw):
        wrapped.run_measured(ids, backend, cfg=a.cfg, logits_to_keep=1, **kw)
        sync()
    for name, be in smoke_controls(cfg["card"]).items():
        forward(be, arms[0], {})
        with trace():
            forward(be, arms[0], {})
        found["controls"].append(name)
    for a in arms:
        log = gt.EventLog()
        scored = []
        with gt.probe_for(log, a):
            kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)
            log.phase = gt.WARMUP
            for traced in (False, True):
                n = len(log.events)
                with (trace() if traced else contextlib.nullcontext()):
                    forward(a.backend, a, kw)
                scored.append(sum(e.kind == "score" for e in log.events[n:]))
        if a.deployable:
            setup = [e.what for e in log.events
                     if e.kind in ("score", "mask_build") and e.phase == gt.SETUP]
            if setup or not all(scored):
                raise StepFailed(f"{a.name}: the probe saw {scored} score call(s) in its "
                                 f"untraced and traced forwards and {setup} in its setup")

        def run(backend=None, input_ids=None, a=a, kw=kw):
            wrapped.run_measured(input_ids if input_ids is not None else ids,
                                 backend if backend is not None else a.backend,
                                 cfg=a.cfg, logits_to_keep=1, **kw)
        dens_found = gt.kernel_densities(a, run)
        sync()
        found["arms"][a.name] = {"score_calls": scored,
                                 "mask_density": dens_found.get("mask_density"),
                                 "executed_density": dens_found.get("executed_density")}
    return found


def _entry_point(op) -> bool:
    """A recorded launch that no aten operator issued (a Triton kernel, an
    extension call)."""
    from attnbench.components import host_profile as hp
    return hp._is_issue(op) and op.aten_op is None


def cmd_preflight_smoke(args) -> int:
    """The smoke check as its own process (`t61_session.run_preflight_smoke`):
    `smoke.json` holds its findings, or the error that stopped it."""
    cfg = load_config(args.config, dry_run=False, cpu_boundary=args.cpu_boundary)
    try:
        found = preflight_smoke(args, cfg)
    except Exception as e:
        traceback.print_exc()
        _write_json(Path(args.out) / "smoke.json", {"error": f"{type(e).__name__}: {e}"})
        print(f"[{SMOKE}] FAILED: {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"[{SMOKE}] passed: {json.dumps(found)[:2000]}", flush=True)
    return 0


def cmd_cell_one(args) -> int:
    """`host-cell-one` or `device-cell-one`: the cell `spawn_cell` names, in
    this process."""
    cell = (args.arm, nominal_density(args.density), int(args.length), int(args.batch))
    cfg = load_config(args.config, dry_run=False, cpu_boundary=args.cpu_boundary)
    run = run_cell if args.step == "host-cell-one" else run_device_cell
    error = run(args, cfg, cell)
    print(f"[{args.step} {cell_dir(Path(args.out), *cell).name}] "
          f"{'FAILED: ' + error if error else 'passed'}", flush=True)
    return 1 if error else 0


COPY_MECHANISM_NAMES = ("copy_to_host_pinned", "copy_to_host_pageable",
                        "copy_from_host_pinned", "copy_from_host_pageable")


def check_payload_bound(ops, max_payload_bytes, label: str) -> None:
    """Refuse a blocking copy that moves more than the link microbenchmark
    measured. The payload term is read at the largest measured size and is
    never extrapolated (researcher, 7 Nov 2026). `None` is a dry run's fakes."""
    if max_payload_bytes is None:
        return
    beyond = [(i + 1, o.mechanism, o.payload_bytes) for i, o in enumerate(ops)
              if o.mechanism in COPY_MECHANISM_NAMES and o.payload_bytes > max_payload_bytes]
    if beyond:
        raise StepFailed(
            f"{label}: {len(beyond)} blocking copies move more than the link microbenchmark "
            f"measured ({max_payload_bytes} bytes); first: operation {beyond[0][0]}, "
            f"{beyond[0][1]}, {beyond[0][2]} bytes. The payload term is never extrapolated.")


def build_prediction(cfg, session: Path, *, dry_run: bool, now=None, local=None,
                     baselines=None, pairing=None):
    """The T2.11 record from the link and device-component steps' outputs.
    `baselines`, for a pairing trained on banked ones (P2 on, a later
    pairing's runner): {"banked": [...], "fitted_sum": ..., "learned": ...,
    "fitted_sum_plus_s0": ...}; P1 has none. `pairing`, the session's own
    when it is not the config's: the L4 rehearsal runs P1's config as P2-R,
    and a record named P1 and trained on P1 is refused (launch 3, 10 Nov 2026)."""
    from attnbench.predict import commit as C
    from attnbench.predict.training import HardwareFeatures

    name = pairing or cfg["pairing"]
    rin = json.loads((step_dir(session, "link") / "recursion_inputs.json").read_text())
    dev = json.loads((step_dir(session, "device_components") / "cells.json").read_text())
    if dev["fake"] != dry_run or rin["fake"] != dry_run:
        raise StepFailed("fake components in a real session, or real ones in a dry run")
    refused = not_priced(dev)
    hw = HardwareFeatures(**rin["hardware"])
    code_paths, out = {}, []
    for e in dev["cells"]:
        if e["arm"] in refused:
            continue
        cell = predict_cell(name, cfg["model"], e, rin, hw)
        code_paths[cell.code_path] = e["code_path"]
        out.append(cell)
    want = {(a, d, L, b) for a, d, L, b in cells(cfg) if a not in refused}
    # A cell is named by its setting: the nominal density (None where the
    # estimator sets it) and the length budget (the prompt lands at or below it).
    got = {(e["arm"], e.get("nominal_density", e["density"]),
            e.get("budget", e["context_length"]), e["batch"])
           for e in dev["cells"] if e["arm"] not in refused}
    if want != got:
        raise StepFailed(f"priced cells differ from the grid: missing {sorted(want - got)}, "
                         f"extra {sorted(got - want)}")
    if not out:
        raise StepFailed(f"no arm is priced: the pricing gate refused {sorted(refused)}")
    code = C.CodeStamp("0" * 40, False) if (dry_run if local is None else local) \
        else C.current_code_stamp()
    fitted = baselines or {"banked": [], "fitted_sum": None, "learned": None}
    return C.build_record(name, out, rin["queue_depth"], code_paths=code_paths,
                          code=code, written_at=now or datetime.now(timezone.utc),
                          not_priced=refused, **fitted)


def predict_cell(pairing: str, model: str, e: dict, rin: dict, hw):
    """One of step 6's priced entries as the prediction's input
    (`training.Cell`): its operations closed and priced at the link table of
    its grid length, the payload bound checked and the event rule applied.
    `build_prediction`'s, and the preflight smoke's on stand-in link inputs."""
    from attnbench.components import event_rule
    from attnbench.predict import commit as C
    from attnbench.predict.intensity import shape_for
    from attnbench.predict.recursion import MechanismCost, close_prefill, price
    from attnbench.predict.training import Cell
    label = C.code_path_label(e["code_path"])
    # The link table is per grid length; a prompt lands at or below its budget.
    grid_L = e.get("budget", e["context_length"])
    mech = {m: MechanismCost(**v) for m, v in rin["by_length"][str(grid_L)].items()}
    ops = list(close_prefill(op_from_dict(o) for o in e["ops"]))
    check_payload_bound(ops, rin.get("max_payload_bytes"), label)
    priced = event_rule.apply(price(ops, mech), e.get("event_syncs", []), mech)
    return Cell(pairing=pairing, model_name=model, code_path=label, arm=e["arm"],
                context_length=e["context_length"], density=e["density"], batch=e["batch"],
                priced=priced, hardware=hw, model=shape_for(model))


def not_priced(dev: dict) -> dict:
    """The arms step 6's pricing gate refused, by arm: the label and why
    (researcher's ruling 1 of 6 Nov 2026). Step 6 always writes the gate."""
    gate = dev.get("pricing_gate")
    if gate is None:
        raise StepFailed("step 6's cells.json has no pricing gate; rerun step 6")
    return {arm: {k: g[k] for k in ("label", "why", "share", "max_share")}
            for arm, g in gate.items() if not g["validated"]}


def isolated_share(session: Path, record: dict) -> dict:
    """Each priced arm's share of prediction time priced from the isolated
    replay (researcher's ruling 1 of 6 Nov 2026, "itself a data point"): per
    cell, the device seconds priced from the isolated original over the
    committed recursion prediction, and per arm their sums' ratio. Written
    beside the committed prediction, never into it."""
    dev = json.loads((step_dir(session, "device_components") / "cells.json").read_text())
    mis = {(e["arm"], e["density"], e["context_length"], e["batch"]): e["pricing"]["mismatch_s"]
           for e in dev["cells"]}
    rows, arms = [], {}
    for c in record["cells"]:
        k = (c["arm"], c["density"], c["context_length"], c["batch"])
        pred = c["predictions"]["recursion"]
        rows.append(dict(zip(("arm", "density", "context_length", "batch"), k),
                         isolated_s=mis[k], prediction_s=pred, share=mis[k] / pred))
        a = arms.setdefault(c["arm"], {"isolated_s": 0.0, "prediction_s": 0.0})
        a["isolated_s"] += mis[k]
        a["prediction_s"] += pred
    for a in arms.values():
        a["share"] = a["isolated_s"] / a["prediction_s"]
    return {"arms": arms, "cells": rows, "not_priced": record.get("not_priced", {})}


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
    shares = isolated_share(session, record)
    _write_json(out / "isolated_share.json", shares)
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
            "memset_max_shift_s": sensitivity["max_shift_s"],
            "isolated_share": {a: v["share"] for a, v in shares["arms"].items()},
            "not_priced": record.get("not_priced", {})}


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
    refused = not_priced(dev)
    for e in dev["cells"]:
        m = e.get("memset_flags")
        if not m or e["arm"] in refused:
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
    gap = _gap(cfg, _local(args))
    so.check_session_skew(skew, guard, dry_run=_local(args), gap=gap)
    receipt, _ = _receipt(Path(args.session))
    t0 = so.utc_now()
    released = so.hold(receipt.synced_at, clock_skew=skew, guard=guard, gap=gap)
    return {"synced_at": receipt.synced_at.isoformat(), "released_at": released.isoformat(),
            "held_s": round((released - t0).total_seconds(), 3),
            "skew_s": skew.total_seconds(), "guard_s": guard.total_seconds()}


GROUP_NOT_RUN = "no arm: the session runs no arm of this group"


def step_ground_truth(args, cfg) -> dict:
    """T2.13, one process per compile-setting group (Appendix A, 27 Oct 2026),
    each with its own dense control. A group with no arm (`[]`, a T3.2a
    repeat's compiled groups; researcher's ruling 4 of 7 Nov 2026) is not
    run. Every group runs; then a cell whose result says `complete: false`
    makes the step incomplete, not passed (T3.2c's 16384 cells passed with 8
    and 2 of 10 clean rounds; adviser's ruling 2 of 9 Nov 2026)."""
    found, short = {}, []
    for group, arms in cfg["gt_groups"].items():
        if not arms:
            found[group] = {"not_run": GROUP_NOT_RUN}
            continue
        d = Path(args.out) / group
        cmd = _py("t31_steps.py", "ground-truth-one", "--group", group, "--out", str(d),
                  "--session", str(args.session), *(["--dry-run"] if args.dry_run else []),
                  *(["--config", args.config] if args.config else []))
        if _run(cmd):
            raise StepFailed(f"ground truth for group {group} failed; see {d}", found)
        found[group] = json.loads((d / "stamps.json").read_text())
        for p in sorted(d.glob("gt_L*_b*.json")):
            r = json.loads(p.read_text())
            if r.get("complete") is not True:
                short.append({"group": group, "file": p.name, "clean_rounds": r.get("clean_rounds"),
                              "rounds_run": r.get("rounds_run")})
    if short:
        found["incomplete"] = short
        raise StepIncomplete(
            f"{len(short)} ground-truth cell(s) with too few clean rounds: "
            + ", ".join(f"{x['group']}/{x['file']} {x['clean_rounds']} clean of "
                        f"{x['rounds_run']}" for x in short), found)
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
    gap = _gap(cfg, args.dry_run)
    so.check_session_skew(skew, guard, dry_run=args.dry_run, gap=gap)
    receipt, _ = _receipt(Path(args.session))
    so.ready_to_clock(receipt.synced_at, so.utc_now(), clock_skew=skew, guard=guard, gap=gap)
    not_before = time.time()          # before this process's first forward of any kind
    arms_in_group = cfg["gt_groups"][args.group]
    if args.dry_run:
        from attnbench.components.shakedown_arms import Harness
        h = Harness(device="cpu", dry_run=True, score_cache_dir=str(out / "score_cache"))
        floor, rerun = cfg["resolution_floor_ms"], cfg["max_rerun_rounds"]
        drift = {"drift_limit_relative": None, "drift_reference_min_rounds": None,
                 "min_clean_rounds": None}
    else:
        _need_gpu("the ground truth")
        h = _harness(args)
        L0 = cfg["lengths"][0]          # each cell takes its own floor in `_real_cells`
        floor = session_floor_ms(cfg, Path(args.session), L0, cfg["batches"].get(str(L0), [1])[0])
        rerun = rerun_budget(cfg)
        drift = drift_controls(cfg)
    with _sampler(cfg, args.dry_run) as sampler:
        controls = gt.Controls(resolution_floor_ms=floor, max_rerun_rounds=rerun,
                               sampler=sampler, **drift)
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
    skip = not_measured(cfg)
    for arm in cfg["gt_groups"][group]:
        if arm == "dense" or (batch > 1 and arm in BATCH1_ONLY):
            continue
        for d in cfg["densities"][arm]:
            if (arm, nominal_density(d), int(length), int(batch)) in skip:
                continue
            names.append(arm)
            dens.append(nominal_density(d))
    return names, dens


def _tracer():
    import torch
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    return lambda: torch.profiler.profile(activities=acts)


class TraceExport:
    """The traced pass's tracer with its Kineto export (T2.7): a fresh profiler
    per block, and each arm's first timed block written to `out_dir` as
    `<stem>_<label>.json` (`gt.tracer_layout`). The export runs after the
    profiler stops, outside the timed block. `check` refuses an entry count
    other than `time_arms`' layout, so a change to that order cannot leave a
    trace under the wrong arm's name."""

    def __init__(self, gt, make, out_dir, stem: str, labels: list, warmup: int):
        self.gt, self.make, self.dir, self.stem = gt, make, Path(out_dir), stem
        self.labels, self.warmup = labels, warmup
        self.layout = gt.tracer_layout(len(labels), warmup)
        self.entered, self.written = 0, []

    @contextlib.contextmanager
    def __call__(self):
        i = self.entered
        self.entered += 1
        with self.make() as prof:
            yield prof
        if i in self.layout:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{self.stem}_{self.labels[self.layout[i]]}.json"
            prof.export_chrome_trace(str(path))
            self.written.append(path)

    def check(self, traced: dict) -> None:
        want = self.gt.tracer_entries(len(self.labels), self.warmup, traced["rounds_run"])
        if self.entered != want or len(self.written) != len(self.labels):
            raise StepFailed(f"{self.stem}: the tracer was entered {self.entered} times and "
                             f"wrote {len(self.written)} traces; time_arms' layout gives "
                             f"{want} and {len(self.labels)}")


def trace_labels(names, densities) -> list:
    """One file label per arm, in arm order: index, arm, and density if any."""
    return [f"{j}_{n}" + ("" if d is None else f"_d{d}")
            for j, (n, d) in enumerate(zip(names, densities))]


def _real_cells(h, gt, cfg, controls, args, receipt, out) -> list:
    """Every (length, batch) of this group: the K2 choice as the dense
    control (checked with `check_control` against the session's record),
    the group's arms at their densities, compiled under the group's one
    setting, timed with the session's sampler, and each row given its cell.
    Then the traced pass (Section 5.3 step 8; Appendix A, 8 Nov 2026), after
    the cell's whole untraced measurement (`time_with_profiler_overhead`):
    its result and the per-arm overhead go to `traced_L*_b*.json`, never to
    `gt_*.json`, which stays untraced ground truth."""
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
            check_of, pid_b = None, None
            if setting is not None:        # ruling 1 of 7 Nov 2026: no probe in a compiled region
                from attnbench import compiled_checks as cc
                ids_b, pid_b = second_prompt(h, L, ids)
                n_layers = int(h.model.config.num_hidden_layers)
                check_of = lambda a, run, ids_b=ids_b: cc.run_checks(   # noqa: E731
                    a, run, second_ids=ids_b, n_layers=n_layers)
            tracer = TraceExport(gt, _tracer(), out / "traces", f"trace_L{L}_b{b}",
                                 trace_labels(names, dens), cfg["warmup"])
            # The cell's own floor when the lock's per-(length, batch) table is set
            # (ruling 1a); P1's one working floor is the controls' as they are.
            cell_controls = controls
            if isinstance(cfg.get("resolution_floor_ms"), dict):
                from dataclasses import replace as _replace
                cell_controls = _replace(controls, resolution_floor_ms=floor_ms_at(cfg, L, b))
            both = gt.time_with_profiler_overhead(
                wrapped, ids, arms, reps=cfg["reps"], controls=cell_controls, tracer=tracer,
                warmup=cfg["warmup"], density_of=gt.kernel_densities,
                synchronize=torch.cuda.synchronize, score_kwargs=score_kwargs,
                compiled_check_of=check_of)
            tracer.check(both["traced"])
            res = both["untraced"]
            stamp = res["first_clocked_block_started_at"]
            ds.check_control(choice, choice.chosen, stamp, key=cell)
            first.append(stamp)
            for row, n, d in zip(res["rows"], names, dens):
                row.update(model=cfg["model"], cell_arm=n, nominal_density=d, budget=L,
                           group=args.group, compile_setting=setting, prompt_id=pid,
                           session_receipt_sha256=receipt.sha256,
                           working_values=cfg.get("working_values"),
                           **({"second_prompt_id": pid_b} if "compiled_checks" in row else {}))
            _write_json(out / f"gt_L{L}_b{b}.json", res)
            _write_json(out / f"traced_L{L}_b{b}.json", {
                "cell_arms": names, "nominal_densities": dens, "budget": L, "batch": b,
                "group": args.group, "overhead": both["overhead"], "traced": both["traced"],
                "traces": [str(p.relative_to(out)) for p in tracer.written]})
            h.unwrap()
    return first


def second_prompt(h, length: int, ids):
    """The two-prompt mask dependence check's second prompt (researcher's
    ruling 1 of 7 Nov 2026): the next prompt at the cell's budget, cut to the
    cell prompt's tokens so every mask has the cell's shape, at the cell's
    batch. Refused if it is shorter than the cell's prompt. (ids, prompt id)."""
    other, pid = h.input_ids(length, index=1)
    n = int(ids.shape[-1])
    if int(other.shape[-1]) < n:
        raise StepFailed(f"the second prompt {pid} has {int(other.shape[-1])} tokens, fewer "
                         f"than the cell prompt's {n}: no mask of the cell's shape")
    return other[..., :n].repeat(int(ids.shape[0]), 1), pid


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
                              clock_skew=skew, banked_at={}, local=local,
                              sync_gap=_gap(cfg, _local(args)))   # P1 is the first pairing: none banked
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


LEG3_PLAN_REMOTE = "leg3_plan.json"


def step_delay_injection(args, cfg) -> dict:
    """Gate A's leg 3 (T2.25), the session's last step, after ground truth
    (adviser's ruling 1 of 8 Nov 2026, fifth paste): the declared set
    (`cfg["delay_injection"]`, `delay_session.declaration`) on the session's
    own priced dense cell and mechanism table. The plan, every variant's
    prediction, is written, put in the store beside the prediction and read
    back before the first timed pass (`delay_session.run`). A dry run plans
    on the fake dense cell and times fake passes against a clock only they
    advance (`_dry_leg3`), so its slopes mean nothing and are labelled fake.
    Behind the CPU boundary the real path runs on the random model with
    injections that do nothing, so its slopes mean nothing either."""
    from attnbench.components import delay_session as dls
    from attnbench.predict.recursion import MechanismCost
    if not cfg.get("delay_injection"):
        raise StepFailed("the session config declares no leg-3 set (delay_injection)")
    decl = dls.declaration(cfg["delay_injection"])
    session, out = Path(args.session), Path(args.out)
    rin = json.loads((step_dir(session, "link") / "recursion_inputs.json").read_text())
    # The injector records before each wait: its event synchronise is priced
    # record-then-wait (`event_rule`, ruling 4a of 9 Nov 2026).
    from attnbench.components import event_rule
    mech = event_rule.for_injection(
        {m: MechanismCost(**v) for m, v in rin["by_length"][str(decl.length)].items()})
    store, base_uri = make_store(session, dry_run=_local(args))
    rel = f"{prediction_dir(cfg, session)}/{LEG3_PLAN_REMOTE}"   # nothing banked names the bucket
    remote = f"{base_uri}/{rel}"
    stored = {}

    def write_plan(rec):
        p = _write_json(out / dls.PLAN, rec)
        store.put(p, remote)
        if store.get(remote) != p.read_bytes():
            raise StepFailed(f"the leg-3 plan read back from {remote} differs from {p}")
        # Read back and stored before `delay_session.run` times its first pass.
        stored.update(sha256=rec["sha256"], remote=rel,
                      store_created=store.created(remote).isoformat(),
                      read_back_at=datetime.now(timezone.utc).isoformat())

    if args.dry_run:
        res = _dry_leg3(decl, mech, rin["queue_depth"], write_plan)
    else:
        res = _real_leg3(args, decl, mech, rin["queue_depth"], session, write_plan)
    res["plan_stored"] = stored
    _write_json(out / dls.RESULT, res)
    m = res["mechanisms"]
    return {"plan_sha256": stored["sha256"], "variants": len(res["samples_s"]) - 1,
            "refused": sorted(res["plan"]["refused"]),
            "measured": sorted(k for k, v in m.items() if v["status"] == "measured"),
            "unresolved": sorted(k for k, v in m.items() if v["status"] != "measured"),
            "fake": bool(args.dry_run), "cpu_boundary": _boundary()}


def _real_leg3(args, decl, mech, queue_depth, session, write_plan) -> dict:
    """The declared dense cell: its recording (step 5) and priced operations
    (step 6), the arm's own prefill under the dispatch observer, the CUDA
    mechanisms."""
    from attnbench.components import delay_injector as di
    from attnbench.components import delay_session as dls
    from attnbench.components import recorder
    from attnbench.predict.recursion import close_prefill
    _need_gpu("the delay injection")
    host_dir = step_dir(session, "host_components")
    name = cell_dir(Path(), decl.arm, 1.0, decl.length, decl.batch).name
    c = json.loads((host_dir / name / CELL_INDEX).read_text())
    dev = json.loads((step_dir(session, "device_components") / "cells.json").read_text())
    entry = [e for e in dev["cells"] if e["arm"] == decl.arm and e["batch"] == decl.batch
             and e.get("budget", e["context_length"]) == decl.length]
    if len(entry) != 1 or entry[0]["ops"] is None:
        raise StepFailed(f"step 6 has no single priced {name}: leg 3 has no base")
    base = list(close_prefill(op_from_dict(o) for o in entry[0]["ops"]))
    rec = recorder.load_recording(host_dir / c["dir"] / "recording")
    h = _harness(args)
    a = ready_cell(h, decl.arm, c["nominal_density"], c["budget"], c["batch"])
    if a.prompt_id != c["prompt_id"]:
        raise StepFailed(f"{name}: the prompt is {a.prompt_id}, the recording's {c['prompt_id']}")
    from attnbench.components import host_profile as hp
    iso = hp.isolated_positions(rec.sequence, _isolated_ops())   # step 5's, seen by no call
    observer = dls.recorded_observer(rec.sequence, skip=iso, device_type=_device(),
                                     layers=a.layers)
    # Behind the CPU boundary the injection is a stand-in that does nothing.
    fire_of = (lambda m: (lambda: None)) if _boundary() else (lambda m: di.cuda_mechanism(m, 0))
    try:
        return dls.run(a.run_prefill, rec.sequence, base, mech, queue_depth, decl,
                       observer=observer, fire_of=fire_of,
                       synchronize=_sync(), clock=time.perf_counter,
                       write_plan=write_plan, align=dls.recorded_alignment(iso),
                       context={"cell": name, "prompt_id": c["prompt_id"],
                                "isolated_positions": list(iso),
                                "queue_depth": queue_depth,
                                "code_path": c["code_path"]})
    finally:
        h.unwrap()


DRY_LEG3_WAIT_S = 1e-4     # what a dry run's fake injection adds, besides s_fixed


def _dry_leg3(decl, mech, queue_depth, write_plan) -> dict:
    """The dry run's leg 3 (`step_delay_injection`): the real planning,
    schedules, timing loop and scoring on the fake dense cell, with a fake
    arm that makes one call per fake operation and a clock that only the
    passes advance (the prefill by the base prediction, each injection by its
    s_fixed plus `DRY_LEG3_WAIT_S`)."""
    from attnbench.components import delay_session as dls
    from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp
    from attnbench.predict.recursion import close_prefill, price, recurse
    fake = fake_ops(decl.arm, 1.0, decl.length, decl.batch)
    seq = OpSequence(code_path=CodePath(arm="dense", gpu="cpu", implementation="dryrun-fake"),
                     context_length=decl.length, batch=decl.batch, streams=(0,),
                     ops=tuple(SeqOp(SeqKind.KERNEL, layer=op.layer, aten_op=f"aten::op{i}",
                                     kernels=(f"k{i}",), stream=0)
                               for i, op in enumerate(fake)))
    base = list(close_prefill(fake))
    base_s = recurse(price(base, mech), queue_depth).end_to_end
    now, hook = [0.0], [None]

    def run_prefill():
        now[0] += base_s
        for op in seq.ops:
            if hook[0] is not None:
                hook[0](op.aten_op, op.layer)

    def observer(run, on_call):
        hook[0] = on_call
        try:
            run()
        finally:
            hook[0] = None

    def fire_of(m):
        return lambda: now.__setitem__(0, now[0] + mech[m].s_fixed + DRY_LEG3_WAIT_S)

    res = dls.run(run_prefill, seq, base, mech, queue_depth, decl, observer=observer,
                  fire_of=fire_of, synchronize=lambda: None, clock=lambda: now[0],
                  write_plan=write_plan,
                  context={"cell": "dryrun", "fake": True, "calls": len(seq.ops)})
    return {**res, "fake": True}


# --- entry point ---------------------------------------------------------------------

STEP_FUNCS = {
    "preflight": step_preflight, "link": step_link, "dense_select": step_dense_select,
    "probe": step_probe, "host_components": step_host_components,
    "device_components": step_device_components, "predict": step_predict,
    "hold": step_hold, "ground_truth": step_ground_truth, "verify": step_verify,
    "postflight": step_postflight, "delay_injection": step_delay_injection,
}
if tuple(STEP_FUNCS) != STEPS:     # not `assert`: T2.17 keeps the harness free of them
    raise RuntimeError("STEP_FUNCS and STEPS disagree")


def run_step(name: str, args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    debug = None
    try:
        cfg = load_config(args.config, dry_run=args.dry_run, cpu_boundary=args.cpu_boundary)
        debug = (cfg.get("debug_label") or DEBUG_LABEL) if cfg.get("debug") else None
        status, reason, findings = PASSED, "", STEP_FUNCS[name](args, cfg)
    except Skip as e:
        status, reason, findings = SKIPPED, str(e), {}
    except StepIncomplete as e:
        status, reason, findings = INCOMPLETE, str(e), e.findings
    except StepFailed as e:
        status, reason, findings = FAILED, str(e), e.findings
    except Exception as e:
        traceback.print_exc()
        status, reason, findings = FAILED, f"{type(e).__name__}: {e}", {}
    _write_json(out / "result.json", {
        "step": name, "status": status, "reason": reason, "dry_run": args.dry_run,
        "cpu_boundary": args.cpu_boundary, "seconds": round(time.time() - started, 1),
        **({"debug": debug} if debug else {}), "findings": findings})
    print(f"[{name}] {status.upper()}{': ' + reason if reason else ''}", flush=True)
    return EXIT[status]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS + ("ground-truth-one", SMOKE) + CELL_CHILDREN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-boundary", action="store_true",
                    help="the real arms on a CPU behind the stubbed GPU boundary (steps 1-8)")
    ap.add_argument("--config", default=None,
                    help=f"session values (default {DEFAULT_CONFIG.relative_to(ROOT)}; "
                         f"a dry run uses its own unless given)")
    ap.add_argument("--group", default=None)
    # host-cell-one, device-cell-one: a step-5 or step-6 cell (`guarded_cells`)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--density", default=None)
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.cpu_boundary:
        if args.dry_run:
            print("REFUSING: --cpu-boundary runs the real steps; it is not --dry-run",
                  file=sys.stderr)
            return 2
        if args.step not in STEPS[:STEPS.index("hold") + 1] + ("delay_injection", SMOKE) \
                + CELL_CHILDREN:
            print(f"REFUSING: the CPU boundary runs steps 1 to 8 and 12, not {args.step}",
                  file=sys.stderr)
            return 2
        from attnbench import cpu_boundary
        with cpu_boundary.installed():
            if args.step in CELL_CHILDREN:
                return cmd_cell_one(args)
            if args.step == SMOKE:
                return cmd_preflight_smoke(args)
            return run_step(args.step, args)
    if args.step in CELL_CHILDREN:
        return cmd_cell_one(args)
    if args.step == SMOKE:
        return cmd_preflight_smoke(args)
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
