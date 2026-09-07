"""Stage 7: the decision map. CPU-only, synthetic frontiers."""

from __future__ import annotations

import pytest

from attnbench.analysis.decision import (
    build_decision_map, fit_decision_tree, recommend)
from attnbench.analysis.pareto import OperatingPoint, ParetoResult, pareto_frontier


def _cell(task, cl, eps, sparse: list[tuple], dense_ms: float):
    """sparse: list of (sparsity, latency_ms, margin)."""
    pts = [OperatingPoint(task=task, context_length=cl, epsilon=eps,
                          backend="block_sparse", sparsity=s, latency_ms=l,
                          margin=m, directional=False, is_dense_reference=False,
                          dominated_by_dense=(dense_ms <= l and eps >= m))
           for s, l, m in sparse]
    pts.append(OperatingPoint(task=task, context_length=cl, epsilon=eps,
                              backend="sdpa_flash", sparsity=None,
                              latency_ms=dense_ms, margin=eps, directional=False,
                              is_dense_reference=True, dominated_by_dense=False))
    return ParetoResult(task=task, context_length=cl, epsilon=eps,
                        all_points=tuple(pts), frontier=tuple(pareto_frontier(pts)))


def test_dense_is_recommended_when_every_sparse_point_is_slower():
    r = recommend(_cell("niah_single", 8192, 1.0, [(0.5, 1415.0, 1.0)], 1140.0),
                  oracle_sensitive_tasks=frozenset())
    assert r.is_dense and r.backend == "sdpa_flash"
    assert r.speedup_vs_dense == pytest.approx(1.0)
    assert "dominated by dense" in r.detail


def test_a_faster_matched_sparse_point_is_recommended():
    r = recommend(_cell("vt", 4096, 1.0, [(0.5, 1423.0, 2.6)], 1503.0),
                  oracle_sensitive_tasks=frozenset())
    assert not r.is_dense and r.sparsity == 0.5
    assert r.speedup_vs_dense == pytest.approx(1503.0 / 1423.0, abs=1e-4)


def test_ties_go_to_dense():
    """Dense needs no scoring pass and carries no oracle dependency, so at
    equal latency it is the simpler system and wins."""
    r = recommend(_cell("vt", 2048, 1.0, [(0.5, 1000.0, 1.0)], 1000.0),
                  oracle_sensitive_tasks=frozenset())
    assert r.is_dense


def test_a_sparse_recommendation_on_an_oracle_sensitive_task_is_flagged():
    r = recommend(_cell("vt", 4096, 1.0, [(0.5, 1423.0, 2.6)], 1503.0),
                  oracle_sensitive_tasks=frozenset({"vt"}))
    assert r.oracle_sensitive is True
    assert "no deployable estimator has" in r.detail


def test_a_dense_recommendation_is_never_oracle_sensitive():
    """The flag warns about relying on the oracle. Recommending dense relies
    on nothing, so flagging it would train a reader to ignore the flag."""
    r = recommend(_cell("vt", 8192, 1.0, [(0.75, 2505.0, 13.2)], 1995.0),
                  oracle_sensitive_tasks=frozenset({"vt"}))
    assert r.is_dense
    assert r.oracle_sensitive is False


def test_a_cell_without_a_dense_reference_is_refused():
    pts = (OperatingPoint(task="vt", context_length=2048, epsilon=1.0,
                          backend="block_sparse", sparsity=0.5, latency_ms=1.0,
                          margin=1.0, directional=False),)
    cell = ParetoResult(task="vt", context_length=2048, epsilon=1.0,
                        all_points=pts, frontier=pts)
    with pytest.raises(ValueError, match="no dense reference"):
        recommend(cell, oracle_sensitive_tasks=frozenset())


def test_tree_fidelity_is_measured_against_the_map_not_held_out_data():
    """27 cells cannot support a generalisation claim. The tree is scored on
    whether it reproduces the map -- calling that 'accuracy' would
    manufacture a claim the grid does not license."""
    cells = [_cell(t, cl, 1.0, [(0.5, 900.0 if t == "vt" else 1400.0, 1.0)], 1000.0)
             for t in ("niah_single", "vt") for cl in (2048, 4096, 8192)]
    recs = build_decision_map(cells, oracle_sensitive_tasks=frozenset({"vt"}))
    assert len(recs) == 6
    fid = fit_decision_tree(recs, max_depth=3)
    assert fid.n_cells == 6
    assert fid.faithful and fid.mismatches == 0
    assert "task_idx" in fid.rules
