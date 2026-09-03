"""Stage 6: Pareto frontiers over Stage 4's matched-accuracy operating
points and Stage 5's end-to-end latency at those points.

Stage 5 (real per-operating-point latency, measured on a rented GPU) isn't
built yet -- and per the README's stage table it only exists *after* Stage 4
has already certified which points are worth timing at all ("End-to-end
model latency at those points" -- Stage 4's matched points, not every raw
backend/sparsity combination). So this module takes latency as an injected
mapping rather than reading a Stage 5 output file directly: fully testable
now with synthetic latency data, and wireable to a real Stage 5 parquet the
moment it exists without this module changing.

Two axes, both derived from `analysis.matched.MatchedAccuracyResult` rows:

- `latency_ms` (minimize) -- Stage 5's measurement at that operating point.
- `margin` (maximize) -- how far an operating point's one-sided 95% CI
  lower bound clears its epsilon threshold (`ci_lower + epsilon`). A
  matched point has margin > 0 by construction (`matched = ci_lower >
  -epsilon`); a bigger margin means more room before the point would flip
  to non-inferior-not-established at a stricter epsilon or with more data.

Only `matched=True` rows become operating points. An operating point that
failed non-inferiority is not a real deployment candidate regardless of how
fast it is, so it's excluded before frontier computation rather than
included and left to lose on the margin axis -- the latter would still let
a fast-but-unsafe point crowd out a slower-but-safe one whenever a caller
forgot to check `matched` first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from .matched import MatchedAccuracyResult

# Keys latency lookups on (sparse_backend, task, context_length, sparsity) --
# the same identity a matched operating point has, so a Stage 5 output row
# and a Stage 4 result row for "the same point" always key identically.
LatencyKey = tuple[str, str, int, Optional[float]]


@dataclass(frozen=True)
class OperatingPoint:
    """One matched-and-timed (backend, sparsity) point at one (task,
    context_length, epsilon) grid cell -- the unit Stage 6's frontier is
    computed over."""

    task: str
    context_length: int
    epsilon: float
    backend: str
    sparsity: Optional[float]
    latency_ms: float
    margin: float
    directional: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParetoResult:
    """Every matched operating point considered for one (task,
    context_length, epsilon) grid cell, plus which of them are
    non-dominated. Keeping `all_points` (not just `frontier`) is what lets
    `to_dataframe` emit a labeled table -- every candidate plus whether it
    made the frontier -- rather than only the winners, which is what
    Stage 7's decision tree needs to train on.
    """

    task: str
    context_length: int
    epsilon: float
    all_points: tuple[OperatingPoint, ...]
    frontier: tuple[OperatingPoint, ...]


def build_operating_points(matched_results: list[MatchedAccuracyResult],
                            latency_ms_by_key: dict[LatencyKey, float],
                            ) -> list[OperatingPoint]:
    """Join Stage 4's matched results to Stage 5's latency (or a synthetic
    stand-in for testing). Only `matched=True` rows are kept -- see module
    docstring.

    Raises if a matched point has no corresponding latency entry: Stage 5
    is supposed to measure exactly Stage 4's certified points, so a gap
    here means Stage 5 hasn't finished measuring that point yet, not that
    the point doesn't matter -- it must not be silently dropped from the
    frontier as if it had failed non-inferiority instead.
    """
    points = []
    for r in matched_results:
        if not r.matched:
            continue
        key: LatencyKey = (r.sparse_backend, r.task, r.context_length, r.sparsity)
        if key not in latency_ms_by_key:
            raise KeyError(
                f"no Stage 5 latency measurement for matched operating point "
                f"{key} -- Stage 5 must measure every point Stage 4 certifies "
                f"before a frontier can be computed"
            )
        points.append(OperatingPoint(
            task=r.task, context_length=r.context_length, epsilon=r.epsilon,
            backend=r.sparse_backend, sparsity=r.sparsity,
            latency_ms=latency_ms_by_key[key], margin=r.ci_lower + r.epsilon,
            directional=r.directional,
        ))
    return points


def _dominates(a: OperatingPoint, b: OperatingPoint) -> bool:
    """True if `a` is at least as good as `b` on both axes (lower latency,
    higher margin) and strictly better on at least one -- the standard
    Pareto-domination test. Two points identical on both axes dominate
    neither one another, so both remain on the frontier."""
    at_least_as_good = a.latency_ms <= b.latency_ms and a.margin >= b.margin
    strictly_better = a.latency_ms < b.latency_ms or a.margin > b.margin
    return at_least_as_good and strictly_better


def pareto_frontier(points: list[OperatingPoint]) -> list[OperatingPoint]:
    """Non-dominated points among `points`, sorted by latency_ms ascending.

    O(n^2) pairwise comparison -- the per-grid-cell point count here is a
    handful of backends/sparsities, not thousands, so the simple, readable
    definition is the right choice over a sweep-line algorithm.
    """
    frontier = [p for p in points
                if not any(_dominates(q, p) for q in points if q is not p)]
    return sorted(frontier, key=lambda p: p.latency_ms)


def compute_pareto_frontiers(matched_results: list[MatchedAccuracyResult],
                              latency_ms_by_key: dict[LatencyKey, float],
                              ) -> list[ParetoResult]:
    """One ParetoResult per (task, context_length, epsilon) grid cell --
    "Pareto frontiers per grid cell", per the README's stage table.
    """
    points = build_operating_points(matched_results, latency_ms_by_key)
    by_group: dict[tuple[str, int, float], list[OperatingPoint]] = {}
    for p in points:
        by_group.setdefault((p.task, p.context_length, p.epsilon), []).append(p)

    out = []
    for (task, context_length, epsilon), pts in sorted(by_group.items()):
        out.append(ParetoResult(
            task=task, context_length=context_length, epsilon=epsilon,
            all_points=tuple(pts), frontier=tuple(pareto_frontier(pts)),
        ))
    return out


def to_dataframe(pareto_results: list[ParetoResult]) -> pd.DataFrame:
    """One row per matched operating point considered in any grid cell,
    labeled with `is_pareto_optimal` -- the flat table Stage 7's decision
    tree trains on (predicting, from context_length etc., which backend/
    sparsity combination lands on the frontier).
    """
    rows = []
    for res in pareto_results:
        frontier_keys = {(p.backend, p.sparsity) for p in res.frontier}
        for p in res.all_points:
            rows.append({**p.to_dict(),
                        "is_pareto_optimal": (p.backend, p.sparsity) in frontier_keys})
    return pd.DataFrame(rows)
