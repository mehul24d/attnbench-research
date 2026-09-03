"""Stage 4: matched-accuracy operating points.

For each (task, context_length, sparsity), tests whether a block_sparse
backend's accuracy is non-inferior to the dense baseline via a paired
bootstrap one-sided confidence bound, at an epsilon sweep of {1, 2, 5}
points. This is the study's primary methodological contribution -- Stage 6's
Pareto frontier and Stage 7's decision tree are built on the sparsity budget
this stage certifies as accuracy-preserving, not on raw score deltas.

Two design points that came out of review and matter enough to restate here
rather than leave implicit in the code:

- The resampling is paired on example_id: a bootstrap draw resamples which
  *examples* are included, and both the dense and sparse arms are evaluated
  on the identical resampled set every time. Resampling dense and sparse
  independently would test "are these two populations different" instead of
  "does sparsity change the answer on the same inputs" -- the wrong
  question, and a strictly wider (less powered) one.
- `matched_sparsity=None` (see `best_matched_sparsity_budget`) is a real,
  reportable result -- it means no tested sparsity level cleared the bar at
  that (task, context_length, epsilon), not a gap in the data. It is always
  emitted as a row, never silently dropped from the output the way an
  empty-result skip would.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..accuracy.config import AccuracyGrid


@dataclass
class MatchedAccuracyResult:
    """One (task, context_length, sparsity, epsilon) non-inferiority test.

    `ci_lower` is the one-sided 95% lower bound on the paired bootstrap
    distribution of (sparse_score - dense_score), in the same points scale
    as AccuracyResult.score (0-100). `matched` is `ci_lower > -epsilon` --
    computed once per (task, context_length, sparsity) via
    `paired_bootstrap_diff_ci` and reused across the epsilon sweep, since
    the bound itself doesn't depend on epsilon.
    """

    task: str
    context_length: int
    sparsity: float
    epsilon: float
    dense_backend: str
    sparse_backend: str
    n: int
    mean_diff: float
    ci_lower: float
    matched: bool
    directional: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchedSparsityBudget:
    """Rollup of MatchedAccuracyResult into the single number Stage 6/7
    actually consume: the highest (most compute-saving) sparsity level that
    is still non-inferior to dense, per (task, context_length, epsilon).

    `matched_sparsity=None` means no tested sparsity level matched at this
    epsilon -- a real result, recorded as a row rather than omitted.
    """

    task: str
    context_length: int
    epsilon: float
    dense_backend: str
    sparse_backend: str
    matched_sparsity: Optional[float]
    directional: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap_diff_ci(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                              *, n_resamples: int = 10000, seed: int = 0,
                              confidence: float = 0.95,
                              ) -> tuple[float, float]:
    """Paired bootstrap over (sparse - dense), same resampled example
    indices applied to both arms.

    Returns (mean_diff, one_sided_lower_bound). `dense_scores[i]` and
    `sparse_scores[i]` must already be aligned to the same example (index i
    means the same example_id in both arrays) -- alignment is the caller's
    job (see `_paired_arrays`), not re-derived here, so this function stays
    a pure numeric primitive testable with synthetic arrays.

    Resampling `diffs = sparse - dense` directly (rather than resampling
    indices into the two original arrays separately) is equivalent to
    resampling one shared index array into both arms and taking the
    per-resample mean difference -- the pairing lives in `diffs` being
    computed from already-aligned arrays before any resampling happens.
    """
    dense_scores = np.asarray(dense_scores, dtype=float)
    sparse_scores = np.asarray(sparse_scores, dtype=float)
    if dense_scores.shape != sparse_scores.shape:
        raise ValueError(
            f"dense_scores shape {dense_scores.shape} != sparse_scores "
            f"shape {sparse_scores.shape} -- paired bootstrap requires "
            f"one score per example, in matching order, on both arms"
        )
    n = dense_scores.shape[0]
    if n == 0:
        raise ValueError("paired_bootstrap_diff_ci requires at least one paired example")

    diffs = sparse_scores - dense_scores
    mean_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)

    lower_pct = (1.0 - confidence) * 100.0
    ci_lower = float(np.percentile(boot_means, lower_pct))
    return mean_diff, ci_lower


def _paired_arrays(dense_rows: pd.DataFrame, sparse_rows: pd.DataFrame,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Align dense and sparse rows on example_id, in matching order.

    Raises if either arm has a duplicate example_id (ambiguous pairing) or
    if the two arms don't cover the identical example set -- a paired test
    on a mismatched set would silently compare different sample
    populations, not the same examples under two backends.
    """
    for name, rows in (("dense", dense_rows), ("sparse", sparse_rows)):
        if rows["example_id"].duplicated().any():
            raise ValueError(f"{name} arm has duplicate example_id rows -- "
                              f"pairing would be ambiguous")

    dense_by_id = dense_rows.set_index("example_id")["score"]
    sparse_by_id = sparse_rows.set_index("example_id")["score"]

    if set(dense_by_id.index) != set(sparse_by_id.index):
        only_dense = set(dense_by_id.index) - set(sparse_by_id.index)
        only_sparse = set(sparse_by_id.index) - set(dense_by_id.index)
        raise ValueError(
            f"dense and sparse example_id sets differ (dense-only: "
            f"{len(only_dense)}, sparse-only: {len(only_sparse)}) -- paired "
            f"bootstrap requires the identical example set on both arms"
        )

    common = sorted(dense_by_id.index)
    return dense_by_id.loc[common].to_numpy(), sparse_by_id.loc[common].to_numpy()


def _sparsity_levels_present(rows: pd.DataFrame) -> list[Optional[float]]:
    """Distinct sparsity values a backend's rows actually use in this cell.

    block_sparse contributes several (one per swept sparsity level); GLA
    and SageAttention have no sparsity knob and contribute exactly one
    value, `None` (schema.AccuracyResult.sparsity is None for every non-
    block-sparse row) -- treated here as that backend's single operating
    point, not a gap to skip. Reading levels off the data rather than off
    `grid.sparsities` is what lets one function serve every candidate
    backend: `grid.sparsities` only describes block_sparse's sweep.
    """
    non_null = sorted(rows["sparsity"].dropna().unique().tolist())
    if rows["sparsity"].isna().any():
        return non_null + [None]
    return non_null


def run_matched_analysis(df: pd.DataFrame, grid: AccuracyGrid, *,
                          epsilons: tuple[float, ...] = (1.0, 2.0, 5.0),
                          dense_backend: str = "sdpa_math",
                          sparse_backends: tuple[str, ...] = ("block_sparse", "gla", "sage"),
                          n_resamples: int = 10000, seed: int = 0,
                          ) -> list[MatchedAccuracyResult]:
    """Run the paired bootstrap non-inferiority test for every (task,
    context_length, sparse_backend, sparsity) cell present in `df`, at
    every epsilon in the sweep.

    Every candidate backend gets the same non-inferiority treatment, not
    just block_sparse: GLA and SageAttention are approximations of dense
    attention too (a different recurrence, and quantized arithmetic,
    respectively), and Stage 6/7 need to know whether each one's single
    operating point actually matches dense before it can be treated as a
    deployment candidate -- exact kernels (SDPA/FA2/FA3/cuDNN/xFormers/
    Flex) are the only backends exempt, since Stage 1 already verifies
    those compute identical attention to dense.

    A (task, context_length, sparse_backend, sparsity) combination missing
    either arm's data is skipped, not fabricated -- this is a "not run
    yet" gap, not a "tested and failed" result, and the two must never be
    conflated in the output. A `sparse_backends` entry absent from `df`
    entirely (e.g. "sage" when --include-sage wasn't used) contributes no
    rows, the same as any other missing cell. `grid.is_directional
    (context_length)` stamps the directional flag through from the grid
    rather than being re-inferred from `n` here, so an underpowered point
    can never silently look fully powered downstream.
    """
    results: list[MatchedAccuracyResult] = []
    tasks = sorted(df["task"].unique())
    context_lengths = sorted(int(c) for c in df["context_length"].unique())

    for task in tasks:
        for context_length in context_lengths:
            cell = df[(df["task"] == task) & (df["context_length"] == context_length)]
            dense_rows = cell[cell["backend"] == dense_backend]
            if dense_rows.empty:
                continue
            directional = grid.is_directional(context_length)

            for sparse_backend in sparse_backends:
                backend_rows = cell[cell["backend"] == sparse_backend]
                if backend_rows.empty:
                    continue

                for sparsity in _sparsity_levels_present(backend_rows):
                    sparse_rows = (backend_rows[backend_rows["sparsity"].isna()]
                                   if sparsity is None else
                                   backend_rows[backend_rows["sparsity"] == sparsity])

                    dense_scores, sparse_scores = _paired_arrays(dense_rows, sparse_rows)
                    mean_diff, ci_lower = paired_bootstrap_diff_ci(
                        dense_scores, sparse_scores, n_resamples=n_resamples, seed=seed)

                    for epsilon in epsilons:
                        results.append(MatchedAccuracyResult(
                            task=task, context_length=context_length, sparsity=sparsity,
                            epsilon=epsilon, dense_backend=dense_backend,
                            sparse_backend=sparse_backend, n=len(dense_scores),
                            mean_diff=mean_diff, ci_lower=ci_lower,
                            matched=ci_lower > -epsilon, directional=directional,
                            detail=("directional point: n reduced below the main "
                                     "grid's power-adequate budget, see stage3_grid.yaml"
                                     if directional else ""),
                        ))
    return results


def best_matched_sparsity_budget(matched_results: list[MatchedAccuracyResult],
                                  ) -> list[MatchedSparsityBudget]:
    """Roll up per-sparsity matched flags into the highest sparsity level
    that still matches dense, per (task, context_length, epsilon,
    sparse_backend).

    Meaningful only for a backend that actually sweeps sparsity
    (block_sparse) -- rows with `sparsity is None` (GLA, SageAttention:
    each contributes one fixed operating point, no sparsity knob) are
    excluded here, since "highest matched sparsity" has no referent for
    them. Whether a single-operating-point backend matched is a complete
    answer on its own `MatchedAccuracyResult` row; this rollup would only
    obscure it behind a `None` that already means something else here (see
    below).

    A group where nothing matched produces `matched_sparsity=None`, emitted
    as its own row -- Stage 6/7 need to see "tested, nothing qualified" as
    distinct from "not tested", and dropping the row would erase that
    distinction.
    """
    by_group: dict[tuple[str, int, float, str], list[MatchedAccuracyResult]] = {}
    for r in matched_results:
        if r.sparsity is None:
            continue
        by_group.setdefault(
            (r.task, r.context_length, r.epsilon, r.sparse_backend), []).append(r)

    out: list[MatchedSparsityBudget] = []
    for (task, context_length, epsilon, sparse_backend), rows in sorted(by_group.items()):
        matched_sparsities = [r.sparsity for r in rows if r.matched]
        best = max(matched_sparsities) if matched_sparsities else None
        out.append(MatchedSparsityBudget(
            task=task, context_length=context_length, epsilon=epsilon,
            dense_backend=rows[0].dense_backend, sparse_backend=sparse_backend,
            matched_sparsity=best, directional=rows[0].directional,
            detail=("" if best is not None else
                    "no tested sparsity level was non-inferior to dense at this epsilon"),
        ))
    return out


def to_dataframe(results: list) -> pd.DataFrame:
    """Both result dataclasses share the to_dict() convention every other
    stage's rows use -- this just batches that into one frame for writing
    to parquet, mirroring gates.probe_grid's row-list shape."""
    return pd.DataFrame([r.to_dict() for r in results])
