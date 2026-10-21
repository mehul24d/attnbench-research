"""The learned (CTFusion-style LightGBM) baseline of plan Section 5.4 (T2.10).

Features are frozen here, in `FEATURES`, in order. Each one is marked with
where its definition comes from:

  - "plan": named in Section 5.4 with no room for another reading;
  - "proposed": named in Section 5.4, but the plan does not say how it is
    computed, so the definition here is a proposal awaiting the researcher;
  - "added": not in the plan; proposed so the learned predictor sees the
    synchronisation costs the recursion sees.

Two features the plan names cannot be computed from what the operation
sequence records, so they are in `PENDING_FEATURES` and not in `FEATURES`:
both need per-operation byte or FLOP counts that T2.3's sequence does not
carry.

Hyperparameters and the regression target are fixed at lock (Gate B), so
`LOCKED_CONFIG` is None until then and every fit takes its `LearnedConfig`
explicitly. LightGBM is imported only when a model is built.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

from .recursion import CostModelError
from .training import (RETROSPECTIVE, Cell, GroundTruthRow, TrainingSet,
                       retrospective_training_set)

FEATURES = (
    # Task features.
    "context_length",
    "density",
    "batch",
    "n_operations",
    "n_synchronisations",
    # Hardware microbenchmark features: host, link and device costs, and
    # pinned and pageable bandwidth.
    "sum_host",
    "sum_link",
    "sum_device",
    "pinned_bandwidth",
    "pageable_bandwidth",
    # Bottleneck ratios.
    "host_to_device_ratio",
    "synchronisations_per_operation",
    # Not in the plan.
    "sum_sync_cost",
)

FEATURE_SOURCE = {
    "context_length": "plan",
    "density": "plan",
    "batch": "plan",
    "n_operations": "plan",
    "n_synchronisations": "plan",
    "sum_host": "plan",
    "sum_link": "plan",
    "sum_device": "plan",
    "pinned_bandwidth": "plan",
    "pageable_bandwidth": "plan",
    "host_to_device_ratio": "proposed",   # sum h / (sum l + sum d)
    "synchronisations_per_operation": "plan",
    "sum_sync_cost": "added",
}

PENDING_FEATURES = {
    "arithmetic_intensity_per_operation_class":
        "needs FLOPs and bytes per operation and a list of operation "
        "classes; neither is defined or recorded",
    "transfer_demand_relative_to_link_bandwidth":
        "needs the bytes each copy moves; a copy's cost arrives as seconds",
}

Target = Literal["seconds", "log_seconds"]


@dataclass(frozen=True)
class LearnedConfig:
    target: Target
    params: Mapping[str, Any] = field(default_factory=dict)


LOCKED_CONFIG: Optional[LearnedConfig] = None   # set at lock (Gate B)

ModelFactory = Callable[[Mapping[str, Any]], Any]


def lightgbm_factory(params: Mapping[str, Any]):
    try:
        import lightgbm
    except ImportError as e:
        raise CostModelError(
            "lightgbm is not installed; the learned baseline is LightGBM "
            "(Section 5.4). Pass model_factory= to fit with something else "
            "in a test") from e
    return lightgbm.LGBMRegressor(**dict(params))


def features(cell: Cell) -> tuple[float, ...]:
    """The values of `FEATURES` for one cell, in order."""
    p = cell.priced
    n_ops = len(p)
    n_sync = sum(p.is_sync)
    sum_h = float(sum(p.host))
    sum_l = float(sum(p.link))
    sum_d = float(sum(p.device))
    if sum_l + sum_d <= 0:
        raise CostModelError(
            f"{cell.pairing}/{cell.arm} at {cell.context_length} has no link "
            f"or device time, so host_to_device_ratio is undefined")
    values = {
        "context_length": float(cell.context_length),
        "density": float(cell.density),
        "batch": float(cell.batch),
        "n_operations": float(n_ops),
        "n_synchronisations": float(n_sync),
        "sum_host": sum_h,
        "sum_link": sum_l,
        "sum_device": sum_d,
        "pinned_bandwidth": float(cell.hardware.pinned_bandwidth),
        "pageable_bandwidth": float(cell.hardware.pageable_bandwidth),
        "host_to_device_ratio": sum_h / (sum_l + sum_d),
        "synchronisations_per_operation": n_sync / n_ops,
        "sum_sync_cost": float(sum(p.sync_cost)),
    }
    return tuple(values[name] for name in FEATURES)


def _encode(y: float, target: str) -> float:
    return math.log(y) if target == "log_seconds" else y


def _decode(v: float, target: str) -> float:
    return math.exp(v) if target == "log_seconds" else v


@dataclass(frozen=True)
class LearnedModel:
    model: Any
    config: LearnedConfig
    target: str
    kind: str
    trained_on: tuple[str, ...]
    n_rows: int

    def predict(self, cells: Sequence[Cell]) -> list[float]:
        X = [list(features(c)) for c in cells]
        return [_decode(float(v), self.config.target)
                for v in self.model.predict(X)]


def fit_learned(training: TrainingSet, config: LearnedConfig,
                model_factory: Optional[ModelFactory] = None) -> LearnedModel:
    """Fit on a committed or retrospective `TrainingSet` (training.py)."""
    if config.target not in ("seconds", "log_seconds"):
        raise CostModelError(
            f"target must be 'seconds' or 'log_seconds', got {config.target!r}")
    if not training.rows:
        raise CostModelError("an empty training set fits nothing")
    factory = model_factory or lightgbm_factory
    model = factory(config.params)
    X = [list(features(r.cell)) for r in training.rows]
    y = [_encode(r.measured, config.target) for r in training.rows]
    model.fit(X, y)
    return LearnedModel(model=model, config=config, target=training.target,
                        kind=training.kind, trained_on=training.trained_on,
                        n_rows=len(training.rows))


@dataclass(frozen=True)
class CurvePoint:
    size: int
    trained_on: tuple[str, ...]
    median_abs_rel_error: float
    n_target_rows: int
    kind: str = RETROSPECTIVE


def learning_curve(rows: Iterable[GroundTruthRow], target: str,
                   sizes: Iterable[int], config: LearnedConfig,
                   model_factory: Optional[ModelFactory] = None
                   ) -> list[CurvePoint]:
    """Error on `target` when trained on every subset of k other pairings.

    One point per subset, so a size has C(n, k) points over a pool of n
    pairings. It draws on pairings run after `target`, so every point is
    retrospective. A size larger than the pool is refused rather than
    skipped: "1 to 6" needs seven pairings, P7 included.
    """
    rows = list(rows)
    target_rows = [r for r in rows if r.pairing == target]
    if not target_rows:
        raise CostModelError(f"{target} has no ground-truth rows to score")
    pool = retrospective_training_set(rows, target).trained_on
    sizes = list(sizes)
    bad = [k for k in sizes if not (1 <= k <= len(pool))]
    if bad:
        raise CostModelError(
            f"training sizes {bad} are impossible with {len(pool)} other "
            f"pairing(s) {list(pool)}")
    truth = [r.measured for r in target_rows]
    cells = [r.cell for r in target_rows]
    out = []
    for k in sizes:
        for subset in itertools.combinations(pool, k):
            chosen = tuple(r for r in rows if r.pairing in subset)
            ts = TrainingSet(target=target, kind=RETROSPECTIVE,
                             trained_on=subset, rows=chosen)
            pred = fit_learned(ts, config, model_factory).predict(cells)
            err = [abs(p - y) / y for p, y in zip(pred, truth)]
            out.append(CurvePoint(size=k, trained_on=subset,
                                  median_abs_rel_error=statistics.median(err),
                                  n_target_rows=len(err)))
    return out
