"""Stage 2 driver: randomized-order kernel sweep, incremental checkpointing,
resumable across process death, gated on Stage 1 correctness and GPU
exclusivity.

Two preconditions gate every cell, both fail closed:

1. Stage 1 pass table: a cell whose (backend, config) has no recorded pass
   is skipped, not measured. A *missing* row counts as a failure, not a
   pass -- absence of evidence is not evidence of correctness, and after any
   AttnConfig field change (which forks every hash -- this already happened
   once in this project), missing rows are exactly what stale Stage 0/1 data
   looks like.
2. GPU exclusivity (provenance.assert_exclusive): anything other than
   "exclusive" blocks the run. Stage 2 is a timing stage; contaminated data
   here is worse than no data.

`plan()` is the single source of truth for both --dry-run's report and the
real run's per-cell gating -- dry-run must predict exactly what a real run
would do, not a separate approximation of it. Everything in this module
except the one `measure_fn` call is a pure function of (cells, Stage 1 pass
set, exclusivity status, already-done keys): fully testable on CPU. Only the
per-cell measurement needs a GPU, and it's injected so tests can stub it.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .backends.base import AttentionBackend
from .checkpoint import append_checkpoint
from .config import AttnConfig, SweepGrid
from . import provenance


@dataclass(frozen=True)
class SweepCell:
    cfg: AttnConfig
    backend_name: str


def build_cells(grid: SweepGrid, backends: list[AttentionBackend],
                 mask_source: Optional[str] = None) -> list[SweepCell]:
    """Cross grid x backends, dropping cells claims_support already rejects
    -- no point queuing what a backend has already declared it can't run.

    mask_source, if given, adds the sparse grid too. Stage 2 must only ever
    use "random" (constraint: random masks are timing-only, importance masks
    are accuracy-only) -- passing "importance" here is a caller error, not
    something this function silently accepts.
    """
    if mask_source == "importance":
        raise ValueError(
            "Stage 2 (kernel timing) must use mask_source='random', never "
            "'importance' -- accuracy-derived masks belong to Stage 3/4"
        )
    cells: list[SweepCell] = []
    for cfg in grid.dense_configs():
        for b in backends:
            ok, _ = b.claims_support(cfg)
            if ok:
                cells.append(SweepCell(cfg=cfg, backend_name=b.name))
    if mask_source:
        for cfg in grid.sparse_configs(mask_source=mask_source):
            for b in backends:
                ok, _ = b.claims_support(cfg)
                if ok:
                    cells.append(SweepCell(cfg=cfg, backend_name=b.name))
    return cells


def _order_seed(cells: list[SweepCell]) -> int:
    """Deterministic from the cell list's own content -- re-invoking with
    the identical cell list always reproduces the identical shuffle, no
    caller-tracked seed to forget or lose across a resume. A different cell
    list (grid edited) naturally gets a different shuffle: that's a
    different sweep, not the same one continuing.
    """
    ids = sorted(f"{c.cfg.key()}:{c.backend_name}" for c in cells)
    blob = "|".join(ids).encode()
    return int(hashlib.sha1(blob).hexdigest()[:8], 16)


def shuffled(cells: list[SweepCell]) -> list[SweepCell]:
    """Randomized visiting order so thermal drift can't correlate with
    backend identity."""
    order = list(range(len(cells)))
    random.Random(_order_seed(cells)).shuffle(order)
    return [cells[i] for i in order]


class Stage1CommitError(RuntimeError):
    """Stage 1 passes exist, but not for the code about to be measured."""


def load_stage1_pass_set(path: Path, *,
                         at_commit: Optional[str] = None) -> set[tuple[str, str]]:
    """(backend, config_key) pairs with a recorded Stage 1 pass. Everything
    else -- including a pair that was never run at all -- counts as not
    passing. A missing correctness.parquet returns the empty set, not an
    error: that's "nothing has passed," which is the correct conservative
    read, not a crash.

    `at_commit` restricts the pass set to passes recorded at that git commit.
    **A correctness pass is evidence about the code that was running when it
    was recorded, not about the code running now.** On 2026-09-03 a bug was
    found in `masks.to_dense_bool` where causal block-sparse masks permitted
    attention to up to `block_size - 1` future tokens inside diagonal blocks;
    every block_sparse correctness pass recorded before that fix describes a
    different function than the one the sweep would now time. Passing the
    current commit turns that from a silent inheritance into a re-run.

    Left as None, the commit is not checked and stale passes are accepted --
    which is the behaviour that existed before this parameter and is kept only
    so a run without git can proceed. Callers that care should pass it; see
    `docs/stage2_plan.md`.
    """
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    if df.empty:
        return set()
    passed = df[df["passed"] == True]  # noqa: E712 (pandas boolean column)

    if at_commit is not None:
        if "git_commit" not in passed.columns:
            raise Stage1CommitError(
                f"{p} has no git_commit column, so it cannot be shown these "
                f"passes describe the current code. Re-run Stage 1.")
        stale = passed[passed["git_commit"] != at_commit]
        passed = passed[passed["git_commit"] == at_commit]
        if passed.empty and not stale.empty:
            others = sorted({str(c)[:12] for c in stale["git_commit"].unique()})
            raise Stage1CommitError(
                f"{p} contains {len(stale)} pass(es), none recorded at the "
                f"current commit {at_commit[:12]} (found: {', '.join(others)}). "
                f"A correctness pass describes the code that produced it. "
                f"Re-run Stage 1 at this commit rather than inheriting them.")

    return set(zip(passed["backend"], passed["config_key"]))


def load_done_keys(checkpoint_path: Path) -> set[tuple[str, str, str]]:
    """(config_key, backend, host) triples already written to the
    checkpoint. Row key includes host so resuming on a different rented
    machine appends rather than colliding with or silently overwriting a
    prior machine's rows.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    if df.empty:
        return set()
    return set(zip(df["config_key"], df["backend"], df["host"]))


class HostMismatchError(RuntimeError):
    """A resume would append a different host's rows into this checkpoint."""


def check_host_continuity(checkpoint_path: Path, host: str) -> None:
    """Refuse to resume into a checkpoint recorded on a different host.

    load_done_keys keys on host, so a host change is invisible to it: cells
    already measured on the old host simply look "not done yet" for the new
    one and get silently re-measured and appended next to the old host's
    rows in the same file. On Spot that's exactly what a mid-sweep
    preemption-and-resume-elsewhere produces. A speedup ratio computed
    against a baseline row from a different physical machine is invalid
    (constraint 5), and nothing downstream of this file is host-aware
    enough to catch that -- so it has to be caught here, at the one place
    a run starts, before a single row is written.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return
    df = pd.read_parquet(p)
    if df.empty or "host" not in df.columns:
        return
    prior_hosts = set(df["host"].unique())
    if prior_hosts and prior_hosts != {host}:
        raise HostMismatchError(
            f"{p} was written on host(s) {sorted(prior_hosts)}; this run is "
            f"on host {host!r}. Resuming into the same results file across "
            f"a host change would mix measurements from different physical "
            f"machines, which invalidates any speedup ratio computed from "
            f"it. Point --out at a new directory for this host instead."
        )


@dataclass(frozen=True)
class CellDecision:
    cell: SweepCell
    action: str    # "run" | "skip_done" | "reject_stage1" | "reject_exclusivity"
    reason: str = ""


def plan(cells: list[SweepCell], *, stage1_passed: set[tuple[str, str]],
         done_keys: set[tuple[str, str, str]], host: str,
         exclusivity_status: str) -> list[CellDecision]:
    """One decision per cell. Order of checks matters: a cell already done
    is reported as done even if Stage 1/exclusivity would otherwise reject
    it now -- what happened when it actually ran is the fact of record, not
    a re-evaluation against today's (possibly stale) preconditions.
    """
    decisions = []
    for cell in cells:
        if (cell.cfg.key(), cell.backend_name, host) in done_keys:
            decisions.append(CellDecision(cell, "skip_done"))
            continue
        if (cell.backend_name, cell.cfg.key()) not in stage1_passed:
            decisions.append(CellDecision(
                cell, "reject_stage1",
                "no Stage 1 pass on record (missing counts as fail)"))
            continue
        if exclusivity_status != "exclusive":
            decisions.append(CellDecision(
                cell, "reject_exclusivity",
                f"GPU exclusivity: {exclusivity_status}"))
            continue
        decisions.append(CellDecision(cell, "run"))
    return decisions


@dataclass(frozen=True)
class SweepReport:
    total: int
    run: int
    skip_done: int
    reject_stage1: int
    reject_exclusivity: int

    @classmethod
    def from_decisions(cls, decisions: list[CellDecision]) -> "SweepReport":
        counts = {"run": 0, "skip_done": 0, "reject_stage1": 0, "reject_exclusivity": 0}
        for d in decisions:
            counts[d.action] += 1
        return cls(total=len(decisions), **counts)


def run_sweep(cells: list[SweepCell], *, out_dir: Path, stage1_path: Path,
              backend_lookup: dict[str, AttentionBackend],
              measure_fn: Callable,
              exclusivity_check: Callable[[], tuple[str, str]] = provenance.assert_exclusive,
              provenance_fn: Callable[[], "provenance.Provenance"] = provenance.capture,
              dry_run: bool = False, checkpoint_every: int = 1) -> SweepReport:
    """Stage 2 entry point.

    `measure_fn(backend, cfg, mask) -> Measurement`, `exclusivity_check()`
    and `provenance_fn()` are injected so this whole driver -- ordering,
    preconditions, checkpoint/resume -- is testable on CPU without a GPU or
    a real measure() call. Production callers pass timing.measure and the
    real provenance functions (the defaults); tests pass stubs.

    `dry_run=True` runs the full planning pass (order, both preconditions,
    resume dedup) and returns the report without calling measure_fn or
    writing anything -- run this first in any rented session. If the Stage 1
    table is stale or exclusivity misbehaves, this finds it in seconds
    instead of minutes into a real sweep.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "sweep.parquet"

    prov = provenance_fn()
    check_host_continuity(checkpoint_path, prov.host)
    stage1_passed = load_stage1_pass_set(stage1_path)
    done_keys = load_done_keys(checkpoint_path)
    status, _detail = exclusivity_check()

    ordered = shuffled(cells)
    decisions = plan(ordered, stage1_passed=stage1_passed, done_keys=done_keys,
                      host=prov.host, exclusivity_status=status)
    report = SweepReport.from_decisions(decisions)

    if dry_run:
        return report

    from .masks import mask_for   # local: keeps the dry-run/CPU path free of any torch CUDA surface

    buffer: list[dict] = []
    for d in decisions:
        if d.action != "run":
            continue
        cell = d.cell
        backend = backend_lookup[cell.backend_name]
        mask = mask_for(cell.cfg) if cell.cfg.mask == "block_sparse" else None
        m = measure_fn(backend, cell.cfg, mask)
        row = {**cell.cfg.to_dict(), **m.to_dict(),
               "backend": cell.backend_name, **prov.to_dict()}
        buffer.append(row)
        if len(buffer) >= checkpoint_every:
            append_checkpoint(checkpoint_path, buffer)
            buffer = []
    if buffer:
        append_checkpoint(checkpoint_path, buffer)

    return report
