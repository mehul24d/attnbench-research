"""The learned (CTFusion-style LightGBM) baseline of plan Section 5.4 (T2.10).

Features are frozen here, in `FEATURES`, in order. Each one is marked with
where its definition comes from:

  - "plan": named in Section 5.4 with no room for another reading;
  - "decided": settled by the researcher (25 Oct 2026): host_to_device_ratio
    as sum h / (sum l + sum d), and sum_sync_cost, which the plan does not
    list, so the learned predictor sees the synchronisation costs the
    recursion sees;
  - "proposed": the researcher decided to compute it, and the definition
    here is a proposal awaiting confirmation: transfer_demand, and the
    arithmetic intensity of each kernel class in `intensity.KERNEL_CLASSES`.

transfer_demand is the bytes the arm moves between host and device (every
copy's and synchronisation's `payload_bytes`) divided by the pairing's
pinned bandwidth: the seconds the link needs for that traffic at its best
rate. Arithmetic intensities are analytic, from the model's shape and the
cell (`intensity.py`). Estimator scoring has no class yet
(`PENDING_FEATURES`).

Hyperparameters and the regression target are fixed at lock (Gate B), so
`LOCKED_CONFIG` is None until then and every fit takes its `LearnedConfig`
explicitly. LightGBM is the optional `learned` extra, pinned to
`LIGHTGBM_VERSION`; it is imported only when a model is built, and always
runs on one thread (see `lightgbm_factory`).
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

from .intensity import KERNEL_CLASSES, arithmetic_intensity
from .recursion import CostModelError
from .training import (RETROSPECTIVE, Cell, GroundTruthRow, TrainingSet,
                       retrospective_training_set, statistic_of)

LIGHTGBM_VERSION = "4.6.0"

AI_FEATURES = tuple(f"ai_{c}" for c in KERNEL_CLASSES)

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
    # Not in the plan; decided 25 Oct 2026.
    "sum_sync_cost",
    # Transfer demand relative to link bandwidth.
    "transfer_demand",
    # Arithmetic intensity per kernel class.
    *AI_FEATURES,
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
    "host_to_device_ratio": "decided",
    "synchronisations_per_operation": "plan",
    "sum_sync_cost": "decided",
    "transfer_demand": "proposed",
    **{name: "proposed" for name in AI_FEATURES},
}

PENDING_FEATURES = {
    "ai_estimator_scoring":
        "each estimator's scoring kernels have their own shape (stride, "
        "block size), so this class needs a formula per estimator",
}

Target = Literal["seconds", "log_seconds"]


@dataclass(frozen=True)
class LearnedConfig:
    target: Target
    params: Mapping[str, Any] = field(default_factory=dict)


LOCKED_CONFIG: Optional[LearnedConfig] = None   # set at lock (Gate B)

ModelFactory = Callable[[Mapping[str, Any]], Any]


def lightgbm_factory(params: Mapping[str, Any]):
    """An LGBMRegressor at the pinned version, on one thread.

    One thread because torch and LightGBM each load their own OpenMP
    runtime on macOS, and a multi-threaded LightGBM fit after `import torch`
    crashes the process (seen with 4.6.0 and the torch in this repo's
    environment, 25 Oct 2026). One thread also makes every fit
    reproducible, and the training sets here are small.
    """
    try:
        import lightgbm
    except ImportError as e:
        raise CostModelError(
            "lightgbm is not installed; install the optional extra with "
            "pip install -e '.[learned]'. Pass model_factory= to fit with "
            "something else in a test") from e
    if lightgbm.__version__ != LIGHTGBM_VERSION:
        raise CostModelError(
            f"lightgbm {lightgbm.__version__} is installed; the learned "
            f"baseline is pinned to {LIGHTGBM_VERSION}")
    params = dict(params)
    for key in ("num_threads", "n_jobs"):
        if key in params and params[key] != 1:
            raise CostModelError(
                f"{key}={params[key]!r}: the learned baseline runs on one "
                f"thread (see lightgbm_factory)")
        params.pop(key, None)
    return lightgbm.LGBMRegressor(num_threads=1, **params)


def features(cell: Cell) -> tuple[float, ...]:
    """The values of `FEATURES` for one cell, in order."""
    p = cell.priced
    if cell.model is None:
        raise CostModelError(
            f"{cell.pairing}/{cell.arm} has no model shape; the arithmetic "
            f"intensities need one")
    n_ops = len(p)
    n_sync = sum(p.is_sync)
    sum_h = float(sum(p.host))
    sum_l = float(sum(p.link))
    sum_d = float(sum(p.device))
    if sum_l + sum_d <= 0:
        raise CostModelError(
            f"{cell.pairing}/{cell.arm} at {cell.context_length} has no link "
            f"or device time, so host_to_device_ratio is undefined")
    pinned = float(cell.hardware.pinned_bandwidth)
    if not (math.isfinite(pinned) and pinned > 0):
        raise CostModelError(f"pinned bandwidth must be positive, got {pinned!r}")
    ai = arithmetic_intensity(cell.model, cell.context_length, cell.batch,
                              cell.density)
    values = {
        "context_length": float(cell.context_length),
        "density": float(cell.density),
        "batch": float(cell.batch),
        "n_operations": float(n_ops),
        "n_synchronisations": float(n_sync),
        "sum_host": sum_h,
        "sum_link": sum_l,
        "sum_device": sum_d,
        "pinned_bandwidth": pinned,
        "pageable_bandwidth": float(cell.hardware.pageable_bandwidth),
        "host_to_device_ratio": sum_h / (sum_l + sum_d),
        "synchronisations_per_operation": n_sync / n_ops,
        "sum_sync_cost": float(sum(p.sync_cost)),
        "transfer_demand": float(sum(p.payload_bytes)) / pinned,
        **{f"ai_{c}": ai[c] for c in KERNEL_CLASSES},
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
    statistic: str
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
                        statistic=training.statistic,
                        n_rows=len(training.rows))


@dataclass(frozen=True)
class CurvePoint:
    size: int
    trained_on: tuple[str, ...]
    median_abs_rel_error: float
    n_target_rows: int
    kind: str = RETROSPECTIVE


def learning_curve(rows: Iterable[GroundTruthRow], target: str,
                   config: LearnedConfig,
                   model_factory: Optional[ModelFactory] = None
                   ) -> list[CurvePoint]:
    """Error on `target` trained on every subset of k other pairings, for k
    from 1 to N - 1 with N the pairings available (researcher's decision,
    25 Oct 2026). One point per subset, C(N - 1, k) at size k. It draws on
    pairings run after `target`, so every point is retrospective.
    """
    rows = list(rows)
    target_rows = [r for r in rows if r.pairing == target]
    if not target_rows:
        raise CostModelError(f"{target} has no ground-truth rows to score")
    statistic_of(rows)
    pool = retrospective_training_set(rows, target).trained_on
    truth = [r.measured for r in target_rows]
    cells = [r.cell for r in target_rows]
    out = []
    for k in range(1, len(pool) + 1):
        for subset in itertools.combinations(pool, k):
            chosen = tuple(r for r in rows if r.pairing in subset)
            ts = TrainingSet(target=target, kind=RETROSPECTIVE,
                             trained_on=subset, rows=chosen,
                             statistic=statistic_of(chosen))
            pred = fit_learned(ts, config, model_factory).predict(cells)
            err = [abs(p - y) / y for p, y in zip(pred, truth)]
            out.append(CurvePoint(size=k, trained_on=subset,
                                  median_abs_rel_error=statistics.median(err),
                                  n_target_rows=len(err)))
    return out
