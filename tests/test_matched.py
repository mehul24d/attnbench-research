"""Stage 4 (matched-accuracy operating points): paired bootstrap
non-inferiority test, validated entirely with synthetic score arrays -- no
model, no GPU, mirroring test_masks_determinism.py's CPU-only discipline.

`min_n=1` appears in the STRUCTURAL tests -- the ones asserting how many rows
come back, which cells are skipped, whether a flag propagates. Those use
10-element frames because the shape of the output is what is under test, not
the statistics. The floor (matched.MIN_PAIRED_N = 30) exists because a small
paired sample does not merely widen the bound, it COLLAPSES it, so a
degenerate cell reports as the most confident one in the table. Passing
min_n=1 states that a test is opting out of a check it is not the subject of;
the tests that do assert statistical behaviour use n >= 120 and the default.
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
        dense_backend="sdpa_flash",
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

    results = run_matched_analysis(df, grid, epsilons=(1.0, 2.0, 5.0),
                                   n_resamples=1000, min_n=1)

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

    results = run_matched_analysis(df, grid, epsilons=(1.0,), min_n=1)

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

    results = run_matched_analysis(df, grid, epsilons=(5.0,), min_n=1,
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

    results = run_matched_analysis(df, grid, epsilons=(5.0,), min_n=1,
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

    results = run_matched_analysis(df, grid, epsilons=(1.0,), min_n=1)
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


# --------------------------------------------------------------------------
# oracle_sensitive: the flag that keeps `vt` reportable rather than dropped.
# --------------------------------------------------------------------------

def _paired_arm(task, n=200, dense_mean=70.0, sparse_mean=70.0, seed=0):
    """Paired rows where the sparse arm sits a fixed offset from dense."""
    rng = np.random.default_rng(seed)
    base = rng.normal(dense_mean, 5.0, n)
    rows = []
    for i, b in enumerate(base):
        rows.append(dict(task=task, context_length=2048, backend="sdpa_flash",
                         sparsity=np.nan, example_id=f"e{i}", score=b))
        rows.append(dict(task=task, context_length=2048, backend="block_sparse",
                         sparsity=0.75, example_id=f"e{i}",
                         score=b + (sparse_mean - dense_mean)))
    return pd.DataFrame(rows)


def test_superiority_reuses_the_non_inferiority_bound():
    """`exceeds_dense` is `ci_lower > 0`; `matched` is `ci_lower > -epsilon`.
    One bootstrap, one bound, two thresholds. A second estimator here would be
    a knob available for tuning after the answer is known."""
    rng = np.random.default_rng(1)
    dense = rng.normal(70, 5, 300)
    mean_diff, ci_lower = paired_bootstrap_diff_ci(dense, dense + 14.6)
    assert mean_diff == pytest.approx(14.6, abs=0.01)
    assert ci_lower > 0.0


def test_a_task_where_sparse_beats_dense_is_flagged_and_still_reported():
    """The `vt` case, measured 2026-09-07: block_sparse at 0.75 beat dense by
    14.6 points at 8192, ~9 SEs.

    It must still produce a matched budget. Dropping the one task where the
    oracle shows through would hand a reader a cleaner picture than the data
    supports, and would make the oracle caveat read as boilerplate.
    """
    df = _paired_arm("vt", dense_mean=70.5, sparse_mean=85.1)
    res = run_matched_analysis(df, _grid(sparsities=(0.75,), tasks=("vt",)),
                               dense_backend="sdpa_flash",
                               sparse_backends=("block_sparse",), n_resamples=2000)
    budgets = best_matched_sparsity_budget(res)
    assert any(r.exceeds_dense for r in res)
    assert budgets
    for b in budgets:
        assert b.oracle_sensitive is True
        assert b.matched_sparsity is not None       # reported, not excluded
        assert "oracle-sensitive" in b.detail


def test_a_task_where_sparse_merely_matches_is_not_flagged():
    """The flag has to discriminate or it carries no information -- the same
    reason the gate_source column has a test that two gates differ."""
    df = _paired_arm("niah_single", dense_mean=70.0, sparse_mean=70.0, seed=7)
    res = run_matched_analysis(df, _grid(sparsities=(0.75,)),
                               dense_backend="sdpa_flash",
                               sparse_backends=("block_sparse",), n_resamples=2000)
    budgets = best_matched_sparsity_budget(res)
    assert not any(r.exceeds_dense for r in res)
    assert all(b.oracle_sensitive is False for b in budgets)


# --------------------------------------------------------------------------
# Cells are bands, not token counts. The test whose absence let n=1 through.
# --------------------------------------------------------------------------

def test_varying_token_counts_collapse_into_one_band_cell():
    """The bug this file did not catch until Stage 4's first real run.

    Every fixture above uses `context_length=2048` exactly, so grouping on the
    raw column looked correct. Real Stage 3 rows carry the EXACT tokenized
    length -- 224 distinct values across three bands -- and the same grouping
    produced ~7 paired examples per cell instead of 300. The bootstrap ran on
    n=1 and printed a full table of budgets.

    The premise the old fixtures asserted was "context_length identifies a
    cell". This asserts the thing that actually has to hold: rows spread
    across a band's real token counts form ONE cell.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        cl = int(rng.integers(1972, 2063))        # the real observed spread
        base = float(rng.normal(70, 5))
        rows.append(dict(task="niah_single", context_length=cl,
                         backend="sdpa_flash", sparsity=np.nan,
                         example_id=f"e{i}", score=base))
        rows.append(dict(task="niah_single", context_length=cl,
                         backend="block_sparse", sparsity=0.75,
                         example_id=f"e{i}", score=base))
    df = pd.DataFrame(rows)

    res = run_matched_analysis(df, _grid(sparsities=(0.75,)),
                               dense_backend="sdpa_flash",
                               sparse_backends=("block_sparse",), n_resamples=500)
    assert len(res) > 0
    assert {r.context_length for r in res} == {2048}, "rows must land in one band"
    assert all(r.n == 120 for r in res), (
        f"cell sizes {[r.n for r in res]} -- every paired example belongs to "
        f"the one band cell, not to its own token count")


def test_a_length_far_from_every_band_is_refused():
    """Nearest-band assignment must not quietly absorb a row from a different
    grid. Refusing is the only safe answer -- filing it under the closer half
    mixes populations invisibly."""
    from attnbench.analysis.matched import band_for
    assert band_for(2062, (2048, 4096, 8192)) == 2048
    assert band_for(8010, (2048, 4096, 8192)) == 8192
    with pytest.raises(ValueError, match="not within 25%"):
        band_for(3000, (2048, 4096, 8192))


def test_the_estimator_refuses_a_sample_it_cannot_resolve():
    """MIN_PAIRED_N, and the reason it is not merely a warning.

    At n=1 every bootstrap resample is the same element, so the resample
    distribution has zero variance and `ci_lower` returns EQUAL to the point
    estimate. A degenerate cell does not look noisy, it looks certain — which
    is why 288 columns of budgets computed on ~7 paired examples raised
    nothing on 2026-09-07.
    """
    from attnbench.analysis.matched import MIN_PAIRED_N

    rng = np.random.default_rng(0)
    for n in (1, 7, MIN_PAIRED_N - 1):
        d = rng.normal(70, 5, n)
        with pytest.raises(ValueError, match="refuses n="):
            paired_bootstrap_diff_ci(d, d + 3.0)

    # And the thing the floor is defending against, shown rather than asserted
    # in prose: at n=1 the bound is the point estimate exactly.
    d1 = np.array([70.0])
    mean_diff, ci_lower = paired_bootstrap_diff_ci(d1, d1 + 3.0, min_n=1)
    assert ci_lower == pytest.approx(mean_diff), (
        "n=1 collapses the interval onto the point estimate -- the degenerate "
        "cell reports as the most confident one")

    # At the floor it works and the interval is real.
    dn = rng.normal(70, 5, MIN_PAIRED_N)
    mean_diff, ci_lower = paired_bootstrap_diff_ci(dn, dn + 3.0)
    assert ci_lower < mean_diff
