"""Stage 4 (matched-accuracy operating points): paired bootstrap
non-inferiority test, validated entirely with synthetic score arrays -- no
model, no GPU, mirroring test_masks_determinism.py's CPU-only discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attnbench.accuracy.config import AccuracyGrid
from attnbench.analysis.matched import (
    MatchedAccuracyResult,
    best_matched_sparsity_budget,
    paired_bootstrap_diff_ci,
    run_matched_analysis,
    to_dataframe,
)


def _grid(**overrides) -> AccuracyGrid:
    fields = dict(
        model_primary="x", model_alternate="y",
        seq_lens={2048: 300, 32768: 100},
        directional_seq_lens=frozenset({32768}),
        block_sizes=(128,), sparsities=(0.5, 0.75, 0.9),
        tasks=("niah_single",), score_dtype="fp16",
        score_cache_dir="results/accuracy/score_cache",
    )
    fields.update(overrides)
    return AccuracyGrid(**fields)


# ---------------------------------------------------------------------------
# paired_bootstrap_diff_ci
# ---------------------------------------------------------------------------

def test_bootstrap_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(1)
    dense = rng.uniform(60, 100, size=50)
    sparse = dense - 2.0
    r1 = paired_bootstrap_diff_ci(dense, sparse, seed=0, n_resamples=2000)
    r2 = paired_bootstrap_diff_ci(dense, sparse, seed=0, n_resamples=2000)
    assert r1 == r2


def test_identical_arms_give_zero_mean_diff_and_ci_lower_near_zero():
    scores = np.array([100.0, 80.0, 60.0, 90.0, 70.0, 100.0, 55.0, 85.0] * 10)
    mean_diff, ci_lower = paired_bootstrap_diff_ci(scores, scores, seed=0, n_resamples=5000)
    assert mean_diff == 0.0
    assert ci_lower == pytest.approx(0.0, abs=1e-9)


def test_large_negative_diff_fails_small_epsilon_passes_large_epsilon():
    """A sparse arm consistently ~3 points worse than dense: epsilon=1
    should not be non-inferior, epsilon=5 should be."""
    rng = np.random.default_rng(2)
    dense = rng.uniform(70, 100, size=300)
    sparse = dense - 3.0 + rng.normal(0, 0.3, size=300)
    mean_diff, ci_lower = paired_bootstrap_diff_ci(dense, sparse, seed=0, n_resamples=5000)

    assert mean_diff == pytest.approx(-3.0, abs=0.5)
    assert not (ci_lower > -1.0)   # epsilon=1: not matched
    assert ci_lower > -5.0         # epsilon=5: matched


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci(np.array([1.0, 2.0]), np.array([1.0]))


def test_empty_arrays_raise():
    with pytest.raises(ValueError):
        paired_bootstrap_diff_ci(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# run_matched_analysis
# ---------------------------------------------------------------------------

def _rows(task, context_length, backend, sparsity, scores, example_ids=None):
    if example_ids is None:
        example_ids = [f"ex{i}" for i in range(len(scores))]
    return [
        dict(task=task, context_length=context_length, backend=backend,
             sparsity=sparsity, example_id=eid, score=s)
        for eid, s in zip(example_ids, scores)
    ]


def test_run_matched_analysis_computes_one_row_per_sparsity_per_epsilon():
    grid = _grid(sparsities=(0.5, 0.75, 0.9))
    dense = _rows("niah_single", 2048, "sdpa_math", None, [90.0] * 20)
    sparse_rows = []
    for sp in (0.5, 0.75, 0.9):
        sparse_rows += _rows("niah_single", 2048, "block_sparse", sp, [90.0] * 20)
    df = pd.DataFrame(dense + sparse_rows)

    results = run_matched_analysis(df, grid, epsilons=(1.0, 2.0, 5.0), n_resamples=1000)

    assert len(results) == 3 * 3   # 3 sparsities x 3 epsilons
    assert all(isinstance(r, MatchedAccuracyResult) for r in results)
    assert all(r.matched for r in results)   # identical scores -> always non-inferior


def test_run_matched_analysis_skips_cells_missing_either_arm():
    """A (task, context_length, sparsity) with no data on one side is a
    'not run yet' gap, not a tested-and-failed result -- must be skipped,
    not fabricated as a row."""
    grid = _grid(sparsities=(0.5, 0.75, 0.9))
    dense = _rows("niah_single", 2048, "sdpa_math", None, [90.0] * 10)
    sparse_rows = _rows("niah_single", 2048, "block_sparse", 0.5, [90.0] * 10)
    df = pd.DataFrame(dense + sparse_rows)

    results = run_matched_analysis(df, grid, epsilons=(1.0,))

    assert len(results) == 1   # only sparsity=0.5 has both arms
    assert results[0].sparsity == 0.5


def test_run_matched_analysis_covers_single_operating_point_backends():
    """GLA/Sage have no sparsity knob -- their rows carry sparsity=None,
    which must surface as its own (sparse_backend, sparsity=None)
    operating point, not be silently skipped the way an unswept axis would
    be if the loop only ever iterated grid.sparsities."""
    grid = _grid(sparsities=(0.5,))
    dense = _rows("niah_single", 2048, "sdpa_math", None, [90.0] * 10)
    gla_rows = _rows("niah_single", 2048, "gla", None, [88.0] * 10)
    df = pd.DataFrame(dense + gla_rows)

    results = run_matched_analysis(df, grid, epsilons=(5.0,),
                                    sparse_backends=("block_sparse", "gla"))

    assert len(results) == 1
    assert results[0].sparse_backend == "gla"
    assert results[0].sparsity is None
    assert results[0].matched   # -2 mean diff, well within epsilon=5


def test_run_matched_analysis_handles_multiple_backends_in_one_call():
    grid = _grid(sparsities=(0.5, 0.75))
    dense = _rows("niah_single", 2048, "sdpa_math", None, [90.0] * 10)
    block_sparse_rows = (_rows("niah_single", 2048, "block_sparse", 0.5, [90.0] * 10)
                         + _rows("niah_single", 2048, "block_sparse", 0.75, [90.0] * 10))
    gla_rows = _rows("niah_single", 2048, "gla", None, [90.0] * 10)
    df = pd.DataFrame(dense + block_sparse_rows + gla_rows)

    results = run_matched_analysis(df, grid, epsilons=(5.0,),
                                    sparse_backends=("block_sparse", "gla"))

    by_backend = {}
    for r in results:
        by_backend.setdefault(r.sparse_backend, []).append(r.sparsity)
    assert sorted(by_backend["block_sparse"]) == [0.5, 0.75]
    assert by_backend["gla"] == [None]


def test_run_matched_analysis_propagates_directional_flag_from_grid():
    grid = _grid(seq_lens={2048: 300, 32768: 100},
                 directional_seq_lens=frozenset({32768}),
                 sparsities=(0.5,))
    rows = (_rows("niah_single", 2048, "sdpa_math", None, [90.0] * 10)
            + _rows("niah_single", 2048, "block_sparse", 0.5, [90.0] * 10)
            + _rows("niah_single", 32768, "sdpa_math", None, [90.0] * 10)
            + _rows("niah_single", 32768, "block_sparse", 0.5, [90.0] * 10))
    df = pd.DataFrame(rows)

    results = run_matched_analysis(df, grid, epsilons=(1.0,))
    by_len = {r.context_length: r.directional for r in results}
    assert by_len[2048] is False
    assert by_len[32768] is True


# ---------------------------------------------------------------------------
# best_matched_sparsity_budget
# ---------------------------------------------------------------------------

def _matched_result(sparsity, matched, task="niah_single", context_length=2048,
                     epsilon=5.0, sparse_backend="block_sparse"):
    return MatchedAccuracyResult(
        task=task, context_length=context_length, sparsity=sparsity, epsilon=epsilon,
        dense_backend="sdpa_math", sparse_backend=sparse_backend, n=100,
        mean_diff=-1.0, ci_lower=-1.5 if matched else -10.0, matched=matched,
        directional=False,
    )


def test_best_matched_sparsity_excludes_single_operating_point_backends():
    """A GLA/Sage row (sparsity=None) has no 'budget' to roll up -- must
    never surface as a MatchedSparsityBudget row, and must never be
    max()'d against a numeric sparsity from a different backend sharing
    the same (task, context_length, epsilon)."""
    results = [_matched_result(0.5, True, sparse_backend="block_sparse"),
               _matched_result(None, True, sparse_backend="gla")]
    budgets = best_matched_sparsity_budget(results)
    assert len(budgets) == 1
    assert budgets[0].sparse_backend == "block_sparse"
    assert budgets[0].matched_sparsity == 0.5


def test_best_matched_sparsity_groups_independently_per_backend():
    results = [_matched_result(0.5, True, sparse_backend="block_sparse"),
               _matched_result(0.9, False, sparse_backend="block_sparse")]
    budgets = best_matched_sparsity_budget(results)
    assert len(budgets) == 1
    assert budgets[0].matched_sparsity == 0.5


def test_best_matched_sparsity_picks_the_highest_matching_level():
    results = [_matched_result(0.5, True), _matched_result(0.75, True),
               _matched_result(0.9, False)]
    budgets = best_matched_sparsity_budget(results)
    assert len(budgets) == 1
    assert budgets[0].matched_sparsity == 0.75


def test_best_matched_sparsity_is_none_and_still_a_row_when_nothing_matches():
    results = [_matched_result(0.5, False), _matched_result(0.75, False)]
    budgets = best_matched_sparsity_budget(results)
    assert len(budgets) == 1   # recorded, not skipped
    assert budgets[0].matched_sparsity is None
    assert budgets[0].detail != ""


def test_best_matched_sparsity_groups_independently_per_epsilon():
    results = [_matched_result(0.5, True, epsilon=5.0),
               _matched_result(0.5, False, epsilon=1.0)]
    budgets = best_matched_sparsity_budget(results)
    by_epsilon = {b.epsilon: b.matched_sparsity for b in budgets}
    assert by_epsilon[5.0] == 0.5
    assert by_epsilon[1.0] is None


def test_to_dataframe_round_trips_both_result_types():
    matched = [_matched_result(0.5, True)]
    budgets = best_matched_sparsity_budget(matched)
    df_matched = to_dataframe(matched)
    df_budgets = to_dataframe(budgets)
    assert list(df_matched["sparsity"]) == [0.5]
    assert list(df_budgets["matched_sparsity"]) == [0.5]
