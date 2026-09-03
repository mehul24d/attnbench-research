"""Stage 6 (Pareto frontiers): pure multi-objective non-domination logic
over synthetic MatchedAccuracyResult + latency data -- no model, no GPU,
no real Stage 5 output needed, mirroring test_matched.py's discipline.
"""

from __future__ import annotations

import pytest

from attnbench.analysis.matched import MatchedAccuracyResult
from attnbench.analysis.pareto import (
    OperatingPoint, build_operating_points, compute_pareto_frontiers,
    pareto_frontier, to_dataframe)


def _matched(backend, sparsity, matched, ci_lower, *, task="niah_single",
             context_length=2048, epsilon=5.0, directional=False):
    return MatchedAccuracyResult(
        task=task, context_length=context_length, sparsity=sparsity, epsilon=epsilon,
        dense_backend="sdpa_math", sparse_backend=backend, n=100,
        mean_diff=ci_lower, ci_lower=ci_lower, matched=matched, directional=directional,
    )


def _point(backend="block_sparse", sparsity=0.5, latency_ms=10.0, margin=1.0,
           task="niah_single", context_length=2048, epsilon=5.0, directional=False):
    return OperatingPoint(task=task, context_length=context_length, epsilon=epsilon,
                          backend=backend, sparsity=sparsity, latency_ms=latency_ms,
                          margin=margin, directional=directional)


# ---------------------------------------------------------------------------
# build_operating_points
# ---------------------------------------------------------------------------

def test_build_operating_points_skips_unmatched_rows():
    results = [_matched("block_sparse", 0.5, matched=True, ci_lower=-1.0),
               _matched("block_sparse", 0.9, matched=False, ci_lower=-8.0)]
    latency = {("block_sparse", "niah_single", 2048, 0.5): 12.0,
               ("block_sparse", "niah_single", 2048, 0.9): 8.0}

    points = build_operating_points(results, latency)

    assert len(points) == 1
    assert points[0].sparsity == 0.5
    assert points[0].latency_ms == 12.0
    assert points[0].margin == pytest.approx(-1.0 + 5.0)


def test_build_operating_points_handles_single_operating_point_backend():
    results = [_matched("gla", None, matched=True, ci_lower=-0.5)]
    latency = {("gla", "niah_single", 2048, None): 5.0}
    points = build_operating_points(results, latency)
    assert len(points) == 1
    assert points[0].backend == "gla"
    assert points[0].sparsity is None


def test_build_operating_points_raises_on_missing_latency():
    results = [_matched("block_sparse", 0.5, matched=True, ci_lower=-1.0)]
    with pytest.raises(KeyError):
        build_operating_points(results, latency_ms_by_key={})


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------

def test_dominated_point_excluded():
    fast_and_safe = _point(sparsity=0.5, latency_ms=5.0, margin=3.0)
    slow_and_less_safe = _point(sparsity=0.9, latency_ms=10.0, margin=1.0)
    frontier = pareto_frontier([fast_and_safe, slow_and_less_safe])
    assert frontier == [fast_and_safe]


def test_tradeoff_points_all_survive():
    """Neither point dominates: A is faster but riskier, B is slower but
    safer -- both are genuine Pareto-optimal choices."""
    fast_risky = _point(sparsity=0.9, latency_ms=5.0, margin=0.5)
    slow_safe = _point(sparsity=0.5, latency_ms=10.0, margin=3.0)
    frontier = pareto_frontier([fast_risky, slow_safe])
    assert set(frontier) == {fast_risky, slow_safe}
    assert frontier[0] is fast_risky   # sorted by latency ascending


def test_identical_points_both_survive():
    a = _point(latency_ms=5.0, margin=1.0)
    b = _point(latency_ms=5.0, margin=1.0)
    frontier = pareto_frontier([a, b])
    assert len(frontier) == 2


def test_three_points_middle_dominated():
    best = _point(sparsity=0.9, latency_ms=5.0, margin=3.0)   # dominates everything
    middle = _point(sparsity=0.75, latency_ms=8.0, margin=2.0)
    worst = _point(sparsity=0.5, latency_ms=10.0, margin=1.0)
    frontier = pareto_frontier([best, middle, worst])
    assert frontier == [best]


# ---------------------------------------------------------------------------
# compute_pareto_frontiers / to_dataframe
# ---------------------------------------------------------------------------

def test_compute_pareto_frontiers_groups_by_grid_cell():
    results = [
        _matched("block_sparse", 0.5, True, -1.0, context_length=2048),
        _matched("block_sparse", 0.5, True, -1.0, context_length=4096),
    ]
    latency = {("block_sparse", "niah_single", 2048, 0.5): 10.0,
               ("block_sparse", "niah_single", 4096, 0.5): 20.0}
    frontiers = compute_pareto_frontiers(results, latency)
    assert len(frontiers) == 2
    by_len = {f.context_length: f for f in frontiers}
    assert by_len[2048].all_points[0].latency_ms == 10.0
    assert by_len[4096].all_points[0].latency_ms == 20.0


def test_to_dataframe_marks_frontier_membership():
    results = [
        _matched("block_sparse", 0.5, True, -0.5),   # slow, safe -> frontier
        _matched("block_sparse", 0.9, True, -4.5),   # fast, risky -> frontier
        _matched("block_sparse", 0.75, True, -4.9),  # dominated by 0.9 (slower, less safe)
    ]
    latency = {("block_sparse", "niah_single", 2048, 0.5): 10.0,
               ("block_sparse", "niah_single", 2048, 0.9): 5.0,
               ("block_sparse", "niah_single", 2048, 0.75): 8.0}
    frontiers = compute_pareto_frontiers(results, latency)
    df = to_dataframe(frontiers)

    assert len(df) == 3
    optimal = set(df[df["is_pareto_optimal"]]["sparsity"])
    assert optimal == {0.5, 0.9}
    assert 0.75 not in optimal
