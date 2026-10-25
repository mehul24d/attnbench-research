"""The queue sensitivity of plan Section 5.4 (T2.15).

"Every committed prediction is recomputed with Q at 0.25x, 0.5x, 2x and 4x
the measured value, and with no queue term." It reads a committed
prediction (commit.load_committed or verify_bytes) and re-runs exactly the
committed components, so the sensitivity is of the prediction that was
committed, not of a re-measurement. The multiples are rounded once, in
`ablations.sensitivity_depths` (Appendix A, 22 Oct 2026: nearest, never
below 1); every multiple is reported even when two round to the same depth.

Only predictors with a queue term change: the recursion and, when it was
committed, the constant-host-cost baseline. The Sum, the step form, the
fitted sums and the learned predictor do not read Q, so they are the same
at every depth by construction and are not recomputed.

`decision_sensitivity` asks the question the sensitivity is for: at which
depths would the predicted sign of sparse minus dense change? It pairs
every non-dense cell with the dense cell at the same length and batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ablations import sensitivity_depths
from .commit import CommittedPrediction, PredictionCommitError
from .naive import with_constant_host_cost
from .recursion import recurse

QUEUE_PREDICTORS = ("recursion", "constant_host_cost")
NO_QUEUE = None   # the multiple reported for "no queue term"


@dataclass(frozen=True)
class QueuePoint:
    multiple: Optional[float]      # None: no queue term
    queue_depth: Optional[int]
    predicted: float
    relative_change: float         # (predicted - committed) / committed


@dataclass(frozen=True)
class CellQueueSensitivity:
    key: tuple
    predictor: str
    measured_q: int
    committed: float
    points: tuple[QueuePoint, ...]

    def at(self, multiple: Optional[float]) -> QueuePoint:
        for p in self.points:
            if p.multiple == multiple:
                return p
        raise KeyError(multiple)


def _priced_for(cp: CommittedPrediction, cell, predictor: str):
    if predictor == "recursion":
        return cell.priced
    return with_constant_host_cost(cell.priced, cp.constant_host_cost)


def queue_sensitivity(cp: CommittedPrediction,
                      predictor: str = "recursion") -> list[CellQueueSensitivity]:
    if predictor not in QUEUE_PREDICTORS:
        raise PredictionCommitError(
            f"{predictor!r} has no queue term; the queue sensitivity covers "
            f"{QUEUE_PREDICTORS}")
    if predictor == "constant_host_cost" and cp.constant_host_cost is None:
        raise PredictionCommitError(
            "this file committed no constant-host-cost baseline")
    Q = cp.queue_depth
    depths = sensitivity_depths(Q)
    out = []
    for cell in cp.cells:
        committed = cell.predictions[predictor]
        priced = _priced_for(cp, cell, predictor)
        if recurse(priced, Q).end_to_end != committed:
            raise PredictionCommitError(
                f"cell {cell.key}: the {predictor} at the measured Q does not "
                f"reproduce the committed prediction")
        points = []
        for m, q in [*depths.items(), (NO_QUEUE, None)]:
            t = recurse(priced, q).end_to_end
            points.append(QueuePoint(multiple=m, queue_depth=q, predicted=t,
                                     relative_change=(t - committed) / committed))
        out.append(CellQueueSensitivity(key=cell.key, predictor=predictor,
                                        measured_q=Q, committed=committed,
                                        points=tuple(points)))
    return out


@dataclass(frozen=True)
class DecisionSensitivity:
    sparse_key: tuple
    dense_key: tuple
    committed_sparse_faster: bool
    flips_at: tuple[Optional[float], ...]   # multiples where the sign changes


def decision_sensitivity(results: list[CellQueueSensitivity], *,
                         dense_arm: str) -> list[DecisionSensitivity]:
    """For each non-dense cell, the multiples at which "sparse is predicted
    faster than dense" (at the same length and batch) is not what the
    committed prediction said. A cell with no dense cell at its length and
    batch has nothing to be compared with and is left out."""
    by_key = {r.key: r for r in results}
    if len({r.predictor for r in results}) > 1:
        raise PredictionCommitError("mix of predictors; pass one at a time")
    dense = {(k[1], k[3]): r for k, r in by_key.items() if k[0] == dense_arm}
    if not dense:
        raise PredictionCommitError(f"no {dense_arm!r} cells to compare against")
    out = []
    for k, r in by_key.items():
        if k[0] == dense_arm:
            continue
        d = dense.get((k[1], k[3]))
        if d is None:
            continue
        base = r.committed < d.committed
        flips = tuple(p.multiple for p in r.points
                      if (p.predicted < d.at(p.multiple).predicted) != base)
        out.append(DecisionSensitivity(sparse_key=k, dense_key=d.key,
                                       committed_sparse_faster=base,
                                       flips_at=flips))
    return out
