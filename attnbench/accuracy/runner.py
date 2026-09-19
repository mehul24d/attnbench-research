"""Stage 3 driver: cross (config, backend, RULER example), gated on the
mask_source discipline, resumable across process death via the same
checkpoint mechanism Stage 2 uses.

Everything except the one `generate_fn` call is a pure function of (cells,
already-done keys): fully testable on CPU, mirroring sweep.py's
plan-then-run split. Only actually running a model needs a GPU, and it's
injected so tests can stub it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import provenance
from ..checkpoint import append_checkpoint
from ..config import AttnConfig
from . import ruler
from .schema import AccuracyResult, BackendRole, Generated


# Derived from the config, never passed in: a role that could disagree with
# the cfg it sits next to is a second source of truth for the same fact.
_ROLE_BY_BACKEND: dict[str, "BackendRole"] = {
    "gla": "linear",
    "sage": "quantized",
    "block_sparse": "block_sparse",
}


def backend_role(backend_name: str, cfg: AttnConfig) -> "BackendRole":
    """What this cell contributes to the study.

    `block_sparse` is keyed on the CONFIG's mask rather than the backend
    name, because the two can come apart: a sparse config can legitimately be
    measured on a different kernel, and it is the mask that decides whether a
    row is a sparsity point or a reference point.
    """
    if cfg.mask == "block_sparse":
        return "block_sparse"
    return _ROLE_BY_BACKEND.get(backend_name, "dense_reference")


@dataclass(frozen=True)
class AccuracyCell:
    cfg: AttnConfig
    backend_name: str
    task: str
    example_id: str


def build_cells(*, configs_by_backend: dict[str, list[AttnConfig]],
                 examples_by_task_length: dict[tuple[str, int], list[ruler.RulerExample]],
                 ) -> list[AccuracyCell]:
    """Cross each backend's own curated config list against exactly the
    examples generated for that config's seq_len -- not every example
    regardless of length.

    Two things this deliberately does NOT do, both found to matter in
    practice rather than assumed up front:

    1. It does not cross every backend against a shared config list via
       AttentionBackend.claims_support(), the way sweep.build_cells does
       for Stage 2. Stage 3's backends have curated, asymmetric roles (one
       dense baseline, block_sparse contributing only its sparse configs,
       GLA/SageAttention each contributing their own single operating
       point) rather than Stage 2's "every capable backend measured at
       every compatible cell" sweep -- claims_support would also match
       block_sparse against a plain causal config (its capability declares
       supports_causal=True by default), producing a redundant cell that
       isn't part of this study's design.
    2. It keys examples on (task, cfg.seq_len) explicitly rather than
       giving every config every example for a task regardless of length.
       An earlier version crossed all examples against all configs
       unconditionally, which silently paired e.g. a seq_len=2048 config
       with an example generated for seq_len=32768 -- inflating cell counts
       (and therefore compute estimates) by roughly the number of distinct
       lengths in the grid.

    Rejects mask_source="random" up front, symmetric to
    sweep.build_cells rejecting "importance" for Stage 2 -- accuracy-
    derived and timing-derived masks must never mix, on either side of the
    split (README: "conflating these is the failure mode this whole study
    exists to correct").
    """
    for configs in configs_by_backend.values():
        for cfg in configs:
            if cfg.mask_source == "random":
                raise ValueError(
                    "Stage 3 (accuracy) must use mask_source='importance', "
                    "never 'random' -- random masks are Stage 2 timing-only, "
                    "and accuracy under a random mask is meaningless"
                )

    tasks = sorted({task for task, _ in examples_by_task_length})
    cells = []
    for backend_name, configs in configs_by_backend.items():
        for cfg in configs:
            for task in tasks:
                examples = examples_by_task_length.get((task, cfg.seq_len), [])
                for ex in examples:
                    cells.append(AccuracyCell(cfg=cfg, backend_name=backend_name,
                                               task=task, example_id=ex.example_id))
    return cells


class DecodePinMismatch(RuntimeError):
    """A run would resume into rows measured under the other decode regime."""


def check_decode_pin_continuity(checkpoint_path: Path, *, pinned: bool) -> None:
    """Refuse to append pinned-decode rows to an unpinned checkpoint, or the
    reverse.

    `load_done_keys` identifies a row by (config_key, backend, task,
    example_id) -- no decode field -- so a resume across regimes would skip
    every example the other regime already measured and the output would
    silently mix two decode kernels under one set of cell names. Rows written
    before `decode_pinned` existed are unpinned by definition.
    """
    if not Path(checkpoint_path).exists():
        return
    df = pd.read_parquet(checkpoint_path)
    if df.empty:
        return
    have = (set(df["decode_pinned"].fillna(False).astype(bool))
            if "decode_pinned" in df.columns else {False})
    if have != {pinned}:
        raise DecodePinMismatch(
            f"{checkpoint_path} holds decode_pinned={sorted(have)} rows; this "
            f"run is decode_pinned={pinned}. Resume keys carry no decode "
            f"field, so mixing them would skip rows measured under the other "
            f"regime. Use a separate --out.")


def load_done_keys(checkpoint_path: Path) -> set[tuple[str, str, str, str]]:
    """(config_key, backend, task, example_id) quadruples already written.

    Deliberately does NOT include host, unlike sweep.load_done_keys: an
    accuracy score doesn't depend on which physical machine produced it,
    only on model + backend + mask, so resuming on a different host should
    recognize a cell as already done rather than re-running it. This is a
    correctness improvement over Stage 2's timing case, not an oversight --
    sweep.check_host_continuity exists specifically because a *timing*
    ratio is invalid across hosts, and that reasoning doesn't apply here.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    if df.empty:
        return set()
    return set(zip(df["config_key"], df["backend"], df["task"], df["example_id"]))


class CodeContinuityError(RuntimeError):
    """Resuming an accuracy run into rows written by different code."""


def check_code_continuity(checkpoint_path: Path, *,
                          current: "provenance.Provenance",
                          allow_mixed_commits: bool = False,
                          allow_dirty: bool = False) -> None:
    """Refuse to resume into a checkpoint written at a different commit.

    The mirror of `load_done_keys`'s deliberate omission, and the reason both
    belong here. An accuracy score does not depend on which MACHINE produced
    it -- so host is correctly ignored -- but it depends entirely on the CODE
    that produced it: the attention swap in `accuracy/model.py`, the example
    generation and scoring in `accuracy/ruler.py`, the mask, the kernel. A
    resume that silently mixes two commits produces one parquet answering two
    questions, and nothing downstream can separate them.

    This matters now rather than in principle: Stage 3 is ~13.4 h of compute
    and will be run in three sessions across several days, which is exactly
    the window in which a repository changes. Stage 2's equivalent hazard
    (`analysis.code_identity`) was found only after four segments had already
    been measured.

    Whole-commit equality, not the per-backend fingerprinting Stage 2 needed.
    There the question was "may these independently-motivated segments be
    joined?", and a blanket answer was wrong because it was yes for seven
    backends and no for two. Here the segments are one experiment split for
    scheduling, so any change to the code that produced a row makes the
    remainder a different run. The coarser check is the correct one, not a
    lazier one.

    `git_dirty` refuses in both directions: a checkpoint written from a dirty
    tree cannot be shown to match anything, and resuming FROM a dirty tree
    cannot be shown to match the checkpoint. A commit is a claim about
    history; only `git_dirty` says whether it describes what ran.
    """
    # A first segment has nothing to be continuous WITH, and blocking it here
    # would turn this into a general commit-hygiene check that fires on every
    # exploratory run -- which is how a guard acquires an unconditional
    # override in front of it. The dirty flag still lands on every row, and
    # the downstream join refuses it there.
    p = Path(checkpoint_path)
    if not p.exists():
        return
    df = pd.read_parquet(p)
    if df.empty:
        return

    if not allow_dirty:
        dirty_now = bool(getattr(current, "git_dirty", False))
        dirty_before = ("git_dirty" in df.columns
                        and bool(df["git_dirty"].fillna(False).astype(bool).any()))
        if dirty_now or dirty_before:
            where = " and ".join(
                w for w, flag in (("this working tree", dirty_now),
                                   ("the checkpoint's rows", dirty_before)) if flag)
            raise CodeContinuityError(
                f"{where} carry uncommitted changes, so the commit recorded on "
                f"these rows does not establish what code ran. Commit first, or "
                f"pass allow_dirty=True to state that you have checked.")

    if "git_commit" not in df.columns:
        raise CodeContinuityError(
            f"{p} has no git_commit column, so it cannot be shown that its "
            f"rows were produced by the code running now. Start a new "
            f"checkpoint rather than appending to one of unknown provenance.")

    prior = sorted({str(c) for c in df["git_commit"].dropna().unique()
                    if str(c) not in ("None", "nan", "HEAD", "")})
    now = str(getattr(current, "git_commit", "") or "")
    if not prior:
        raise CodeContinuityError(
            f"{p} records no usable commit (found "
            f"{sorted({str(c) for c in df['git_commit'].unique()})}).")
    if allow_mixed_commits:
        return
    if prior != [now]:
        raise CodeContinuityError(
            f"{p} was written at {', '.join(c[:12] for c in prior)} and this "
            f"process is at {now[:12] or '(none)'}. Stage 3 runs in several "
            f"sessions across several days; a repository change between them "
            f"makes the remainder a different experiment, and the rows would "
            f"be indistinguishable afterwards.\n\nCheck out the segment's "
            f"pinned commit and re-run, write the new segment to its own "
            f"out_dir, or pass allow_mixed_commits=True having established "
            f"the difference cannot affect these rows.")


@dataclass(frozen=True)
class CellDecision:
    cell: AccuracyCell
    action: str   # "run" | "skip_done"


def plan(cells: list[AccuracyCell], *,
         done_keys: set[tuple[str, str, str, str]]) -> list[CellDecision]:
    """One decision per cell. No Stage-1/exclusivity-style preconditions
    here -- those are Stage 2 GPU-timing concerns; accuracy has none."""
    decisions = []
    for cell in cells:
        key = (cell.cfg.key(), cell.backend_name, cell.task, cell.example_id)
        action = "skip_done" if key in done_keys else "run"
        decisions.append(CellDecision(cell, action))
    return decisions


@dataclass(frozen=True)
class AccuracyReport:
    total: int
    run: int
    skip_done: int

    @classmethod
    def from_decisions(cls, decisions: list[CellDecision]) -> "AccuracyReport":
        counts = {"run": 0, "skip_done": 0}
        for d in decisions:
            counts[d.action] += 1
        return cls(total=len(decisions), **counts)


def run_accuracy(cells: list[AccuracyCell], *, out_dir: Path,
                  examples_by_id: dict[tuple[str, str], "ruler.RulerExample"],
                  generate_fn: Callable[[AttnConfig, str, "ruler.RulerExample"],
                                         Generated],
                  provenance_fn: Callable[[], "provenance.Provenance"] = provenance.capture,
                  dry_run: bool = False, checkpoint_every: int = 1,
                  allow_mixed_commits: bool = False, allow_dirty: bool = False,
                  score_source: str = "dense_softmax_fp32",
                  ) -> AccuracyReport:
    """Stage 3 entry point.

    `generate_fn(cfg, backend_name, example) -> Generated` is injected so
    this driver -- planning, resume, checkpoint -- is testable on CPU
    without a real model. Production callers pass a function that runs
    SwappableAttentionModel.generate and decodes the output; tests pass a
    stub. A `Generated` and not a tuple: see schema.Generated.

    `examples_by_id` keys on (task, example_id) -- the same identity
    load_done_keys/plan use -- so a caller only has to build it once from
    whatever ruler.generate_examples() calls it made.
    """
    # Approximate sizing is a dry-run/testing affordance. Catching it at the
    # call site that chooses the counter is not enough -- the guard has to
    # hold where the harm would occur, which is here, immediately before
    # rows get written. A results file whose context lengths are estimates
    # would look exactly like one whose lengths are real, and every hour
    # estimate and accuracy-vs-length claim built on it would be wrong by an
    # unknown factor. Checked before any work runs, not per row.
    if not dry_run:
        approximate = sorted({
            ex.task for ex in examples_by_id.values()
            if getattr(ex, "sizing", "exact") != "exact"
        })
        if approximate:
            raise ValueError(
                f"refusing to run: examples for {approximate} were sized with "
                f"an approximate token counter (sizing.approximate_token_count), "
                f"not a real tokenizer. Grid lengths would not be real token "
                f"counts and every derived number would be wrong by an unknown "
                f"factor. Pass the target model's tokenizer to "
                f"build_examples_by_task_length, or use dry_run=True."
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "accuracy.parquet"

    check_code_continuity(checkpoint_path, current=provenance_fn(),
                          allow_mixed_commits=allow_mixed_commits,
                          allow_dirty=allow_dirty)
    done_keys = load_done_keys(checkpoint_path)
    decisions = plan(cells, done_keys=done_keys)
    report = AccuracyReport.from_decisions(decisions)

    if dry_run:
        return report

    prov = provenance_fn()
    buffer: list[dict] = []
    for d in decisions:
        if d.action != "run":
            continue
        cell = d.cell
        example = examples_by_id[(cell.task, cell.example_id)]
        gen = generate_fn(cell.cfg, cell.backend_name, example)
        if not isinstance(gen, Generated):
            # Named explicitly rather than unpacked-and-hoped. The old
            # contract was a 2-tuple, which unpacks silently into
            # (predicted, latency) and would drop stop_reason on the floor
            # for a whole 13-hour run.
            raise TypeError(
                f"generate_fn returned {type(gen).__name__}, expected "
                f"schema.Generated -- the (text, latency_ms) tuple contract "
                f"was replaced when generation started carrying a stopping "
                f"reason, and unpacking one here would discard it silently.")
        example_score = ruler.score(cell.task, gen.text, example.answer)

        result = AccuracyResult(
            backend=cell.backend_name,
            backend_role=backend_role(cell.backend_name, cell.cfg),
            config_key=cell.cfg.key(),
            task=cell.task,
            example_id=cell.example_id,
            context_length=example.context_length,
            mask_source=cell.cfg.mask_source,
            sparsity=cell.cfg.sparsity,
            # The scorer that actually ran, NOT a literal. This was hardcoded
            # to "dense_softmax_fp32" while only one scorer existed, which is
            # harmless exactly until a second one exists -- at which point
            # every cheap-estimator row would have stamped itself as the
            # oracle, and the comparison between them would have compared a
            # table against itself. The field exists to travel with the data;
            # a constant does not travel, it just looks like it does.
            score_source=(score_source if cell.cfg.mask == "block_sparse" else None),
            haystack_mode=ruler.haystack_mode_for(cell.task),
            predicted=gen.text,
            expected="; ".join(example.answer),
            score=example_score,
            correct=example_score >= 100.0,
            stop_reason=gen.stop_reason,
            n_generated=gen.n_generated,
            decode_backend=gen.decode_backend,
            decode_pinned=gen.decode_pinned,
            # Carried, not re-derived. The runner holds a backend NAME; only
            # generate_fn held the instance, and the instance is the only
            # thing that knows which gate ran. AccuracyResult.__post_init__
            # refuses the row if a gated backend arrives without one.
            gate_source=gen.gate_source,
            latency_ms=gen.latency_ms,
        )
        row = {**result.to_dict(), **prov.to_dict()}
        buffer.append(row)
        if len(buffer) >= checkpoint_every:
            append_checkpoint(checkpoint_path, buffer)
            buffer = []
    if buffer:
        append_checkpoint(checkpoint_path, buffer)

    return report
