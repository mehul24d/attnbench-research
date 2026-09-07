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
    dense_latency = dense[0].latency_ms

    # Sort so that ties resolve to dense: False < True, so `not is_dense_reference`
    # puts the baseline first at equal latency.
    best = min(result.frontier,
               key=lambda p: (p.latency_ms, not p.is_dense_reference))

    notes = []
    sparse_pts = [p for p in result.all_points if not p.is_dense_reference]
    if sparse_pts and all(p.dominated_by_dense for p in sparse_pts):
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
