"""Stage 7: the decision map -- what to actually use, per grid cell.

WHY THIS IS A MAP AND NOT A LEARNED MODEL

The README calls this stage a decision tree. Against the data that exists it
would be a fitted model over **27 cells** (3 tasks x 3 context lengths x 3
epsilons), every one of which is already an exhaustive enumeration of its own
candidates. A tree over 27 rows reproduces them and generalises nothing: its
splits would be a re-encoding of the table, and any apparent rule ("above
4096, prefer dense") would rest on three points.

So the primary artifact is the map: one recommendation per cell, derived
deterministically from Stage 6's frontier. `fit_decision_tree` is offered as
a **compression** of that map and is scored on FIDELITY to it -- does the
tree reproduce every recommendation? -- not on held-out accuracy, because
there is no held-out data and pretending otherwise would manufacture a
generalisation claim the grid cannot support.

If the tree is faithful it is a readable summary. If it is not, the map wins
and the tree is reported as lossy. Neither outcome licenses extrapolating to
a cell the grid never measured.

THE RECOMMENDATION RULE

Among the cell's Pareto-optimal points, take the lowest latency. Ties go to
the dense baseline: it needs no importance-scoring pass, carries no oracle
dependency, and is the simpler system. "Fastest among the non-dominated,
preferring the simpler thing when they tie" is the whole rule, stated before
the data was looked at.

Two flags travel with every recommendation, because a recommendation without
them is not usable:

- `oracle_sensitive` -- a sparse recommendation on a task where some sparsity
  level exceeded dense beyond noise. The advantage may be an artifact of
  building the mask from full attention scores, which no deployable estimator
  can do. See docs/claims.md.
- `directional` -- the cell's n was reduced below the power-adequate budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from .pareto import ParetoResult


# How precisely a normalized speedup can be compared, by context length.
#
# Measured from this study's own data, the same way `cross_arch`'s
# RATIO_RESOLUTION_BANDS were, and for the same reason: an unfloored map
# reports a 1.002x recommendation with the same face as a 1.32x one.
#
# Stage 5 measured prefill and decode_step for every (band, sparsity) in three
# independent rented sessions -- results/stage5/, stage5_flashdecode/ and
# stage5_ols/. Rebuilding the map's own ranking quantity
# (`normalized_total(dense) / normalized_total(sparse)`) from each session in
# turn gives three values for one cell. Their spread is how much of a
# difference the instrument cannot resolve:
#
#     band     sparsities   sessions   max spread across sessions
#     2048     0.5/0.75/0.9     3            0.20 %
#     4096     0.5/0.75/0.9     3            0.57 %
#     8192     0.5/0.75/0.9     3            1.65 %
#     16384    0.5/0.75/0.9     1            -- not measured twice --
#     32768    0.5/0.75/0.9     1            -- not measured twice --
#
# The spread grows with band, which is expected: prefill grows, and the
# between-session drift in the dense prefill alone is 9 ms at 4096
# (283.76 -> 288.10 -> 292.75) against a within-session sd of 1.2-2.4 ms.
#
# Collapsed to two bands, deliberately coarser than the table. Nine points do
# not support a per-band curve, and the point of a floor is to be defensible
# rather than tight. Each band takes the WORST spread observed in it, rounded
# up.
#
# 16384 and 32768 were each measured in exactly one session, so their spread
# is unknown, and they inherit the widest MEASURED value rather than the
# nearest one. That is a lower bound on their true uncertainty, not an
# estimate of it -- recorded in docs/limitations.md rather than hidden here.
# It does not endanger the headline: 1.321x at 32768 is 32 % above 1.0
# against a 1.7 % floor.
SPEEDUP_RESOLUTION_BANDS: tuple[tuple[float, float], ...] = (
    (4096.0, 0.006),        # context_length <= 4096
    (float("inf"), 0.017),
)


def speedup_resolution(context_length: Optional[int]) -> float:
    """Smallest speedup difference distinguishable from session-to-session
    variation at this context length.

    `None` returns the WIDEST band. An unknown length must not buy a claim
    more precision than a measured one -- the same rule as
    `cross_arch.ratio_resolution`.
    """
    if context_length is None:
        return max(r for _, r in SPEEDUP_RESOLUTION_BANDS)
    for upper, resolution in SPEEDUP_RESOLUTION_BANDS:
        if context_length <= upper:
            return resolution
    return SPEEDUP_RESOLUTION_BANDS[-1][1]


@dataclass(frozen=True)
class Recommendation:
    """What to use at one (task, context_length, epsilon) cell."""

    task: str
    context_length: int
    epsilon: float
    backend: str
    sparsity: Optional[float]
    latency_ms: float
    speedup_vs_dense: float
    is_dense: bool
    oracle_sensitive: bool
    directional: bool
    n_candidates: int

    # How far the recommendation is from being a coin flip. See
    # SPEEDUP_RESOLUTION_BANDS.
    resolution: float = 0.0

    # SIGNED, and `None` when the chosen point is the only one on the
    # frontier. Negative means the next-best point was nominally faster and
    # was passed over because the gap was inside the floor -- which is what a
    # tie-break to dense looks like from here, and is worth seeing rather
    # than hiding behind an absolute value.
    separation: Optional[float] = None

    # Two different questions. A cell can fail one and pass the other: where
    # a sparse point sits at identical latency to dense it is dominated off
    # the frontier, so "which point is fastest" is answered (dense, it cannot
    # be worse on either axis) while "is dense faster" is not answered at all.
    resolvable: bool = True                 # is THIS recommendation decided
    dense_choice_resolvable: bool = True    # is "sparse or dense" decided

    tie_broken_to_dense: bool = False       # sub-resolution advantage discarded

    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def recommend(result: ParetoResult, *, oracle_sensitive_tasks: frozenset[str],
              ) -> Recommendation:
    """The cell's recommendation: fastest non-dominated point, dense on ties.

    Raises on a cell with no dense reference -- without the baseline the
    question "is any of this worth taking" has no answer, and a
    recommendation that cannot answer it is not one.
    """
    dense = [p for p in result.all_points if p.is_dense_reference]
    if not dense:
        raise ValueError(
            f"cell ({result.task}, {result.context_length}, {result.epsilon}) "
            f"has no dense reference point, so no recommendation can be made: "
            f"there is nothing to say whether taking a sparse point is worth "
            f"it. Pass dense_backend to compute_pareto_frontiers.")
    dense_point = dense[0]
    dense_latency = dense_point.latency_ms

    # Sort so that ties resolve to dense: False < True, so `not is_dense_reference`
    # puts the baseline first at equal latency.
    ranked = sorted(result.frontier,
                    key=lambda p: (p.latency_ms, not p.is_dense_reference))
    best = ranked[0]

    # The tie rule, applied with a MEASURED tie tolerance instead of exact
    # equality. "Ties go to the dense baseline" was pre-registered above; a
    # difference smaller than the instrument can resolve IS a tie, and
    # treating it as a win is the whole failure this floor exists to stop.
    # Nothing new is decided here -- the existing rule is applied correctly
    # now that "equal" has a measured width.
    resolution = speedup_resolution(result.context_length)
    tie_broken = False
    if not best.is_dense_reference:
        advantage = dense_latency / best.latency_ms - 1.0
        if advantage < resolution:
            best = dense_point
            tie_broken = True

    others = [p for p in ranked if p is not best]
    separation = (others[0].latency_ms / best.latency_ms - 1.0) if others else None

    sparse_pts_all = [p for p in result.all_points if not p.is_dense_reference]
    if sparse_pts_all:
        fastest_sparse = min(sparse_pts_all, key=lambda p: p.latency_ms)
        dense_choice_resolvable = (
            abs(dense_latency / fastest_sparse.latency_ms - 1.0) >= resolution)
    else:
        # No accuracy-matched sparse point exists at all. Dense is the only
        # candidate, so the choice is not close -- it is uncontested.
        dense_choice_resolvable = True

    resolvable = separation is None or separation >= resolution

    notes = []
    if tie_broken:
        notes.append(
            f"the fastest non-dominated point was sparse but only "
            f"{advantage:.2%} ahead of dense, inside the {resolution:.1%} "
            f"resolution floor for this band -- a tie by measurement, and "
            f"the pre-registered tie rule gives it to dense")
    if not resolvable:
        notes.append(
            f"UNRESOLVABLE: the next-best point is {separation:.2%} away, "
            f"inside the {resolution:.1%} floor. Which point is fastest here "
            f"is decided by which Stage 5 session's phases are used, not by "
            f"the arms")
    if not dense_choice_resolvable:
        notes.append("sparse-versus-dense is itself undecided at this cell")

    if sparse_pts_all and all(p.dominated_by_dense for p in sparse_pts_all):
        notes.append("every accuracy-matched sparse point is dominated by "
                     "dense on both axes")
    oracle = (not best.is_dense_reference) and result.task in oracle_sensitive_tasks
    if oracle:
        notes.append("sparse recommendation on an oracle-sensitive task: the "
                     "advantage may come from building the mask with full "
                     "attention scores, which no deployable estimator has")

    return Recommendation(
        task=result.task, context_length=result.context_length,
        epsilon=result.epsilon, backend=best.backend, sparsity=best.sparsity,
        latency_ms=best.latency_ms,
        speedup_vs_dense=dense_latency / best.latency_ms,
        is_dense=best.is_dense_reference, oracle_sensitive=oracle,
        directional=best.directional, n_candidates=len(result.all_points),
        resolution=resolution, separation=separation, resolvable=resolvable,
        dense_choice_resolvable=dense_choice_resolvable,
        tie_broken_to_dense=tie_broken,
        detail="; ".join(notes))


def build_decision_map(pareto_results: list[ParetoResult], *,
                        oracle_sensitive_tasks: frozenset[str],
                        ) -> list[Recommendation]:
    return [recommend(r, oracle_sensitive_tasks=oracle_sensitive_tasks)
            for r in sorted(pareto_results,
                            key=lambda r: (r.task, r.context_length, r.epsilon))]


def to_dataframe(recs: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in recs])


@dataclass(frozen=True)
class TreeFidelity:
    """How well a fitted tree reproduces the map it summarises.

    `mismatches` is the count of cells where the tree's prediction differs
    from the map's recommendation. Not "test accuracy": there is no test set,
    and calling this accuracy would imply a generalisation claim 27 cells
    cannot support.
    """

    n_cells: int
    max_depth: int
    mismatches: int
    rules: str

    @property
    def faithful(self) -> bool:
        return self.mismatches == 0


def fit_decision_tree(recs: list[Recommendation], *, max_depth: int = 3,
                       ) -> TreeFidelity:
    """Fit a shallow tree over (task, context_length, epsilon) -> choice, and
    score it on fidelity to the map.

    Deliberately shallow. A deeper tree would reach zero mismatches on any
    27-row table by memorising it, which would tell us nothing except that
    27 < 2**depth.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    if not recs:
        raise ValueError("no recommendations to summarise")

    tasks = sorted({r.task for r in recs})
    X = [[tasks.index(r.task), float(r.context_length), r.epsilon] for r in recs]
    y = [f"{r.backend}" + ("" if r.sparsity is None else f"@{r.sparsity:g}")
         for r in recs]

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
    pred = clf.predict(X)
    mismatches = int(sum(1 for a, b in zip(pred, y) if a != b))
    rules = export_text(clf, feature_names=["task_idx", "context_length", "epsilon"])
    return TreeFidelity(n_cells=len(recs), max_depth=max_depth,
                        mismatches=mismatches, rules=rules)
