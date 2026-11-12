"""Bootstrap prediction intervals (plan Section 5.4, T2.19).

"A prediction interval from bootstrapping the component measurements":
every component a prediction reads (a host cost, a device cost, s_fixed of
a mechanism, a bandwidth) is a statistic over a microbenchmark's repeats.
A replicate resamples each measurement's repeats with replacement,
recomputes its statistic, and evaluates every prediction from those
values. All outputs of one replicate share the same resampled components,
so a difference between two predictions (sparse minus dense at one length,
for the crossover rule) is bootstrapped jointly, replicate by replicate.

The same function bootstraps measured ground truth: the samples are then
the timed repeats of each cell, and the outputs their statistic or their
difference.

**The resampling protocol**, fixed so an interval can be reproduced from
its recorded seed: a `numpy.random.Generator(numpy.random.PCG64(seed))`;
for each replicate, the measurement keys in sorted order, and for each key
`rng.integers(0, n, size=n)` indices into its n repeats. The interval is
the percentile interval: numpy's linear quantiles at (1 - level) / 2 and
(1 + level) / 2 of the replicates. The point is the outputs at the
full-sample statistics, not the replicates' mean.

**Refusals** (adviser's ruling 2e of 11 Nov 2026; T4.3 read A, finding 8):
a measurement with fewer than two repeats, whose every replicate is the
sample itself, and fewer than two replicates, either of which gives a
zero-width interval that would decide a sign; and, once the lock's
`bootstrap_settings` are set (`LOCKED`, T4.1), fewer replicates than its
`n_boot`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .recursion import CostModelError

Evaluate = Callable[[Mapping[str, float]], Mapping[str, float]]
METHOD = "percentile"


@dataclass(frozen=True)
class Settings:
    """T4.1's `bootstrap_settings`: the level, size and seed of every
    bootstrap the analysis draws."""
    level: float
    n_boot: int
    seed: int


LOCKED: Optional[Settings] = None      # set from the lock (`settings_from_lock`)


def settings_from_lock(lock: Mapping) -> Settings:
    """The lock's `bootstrap_settings`, refused unless all three are fixed."""
    v = ((lock.get("values") or {}).get("bootstrap_settings") or {}).get("value")
    if not isinstance(v, Mapping):
        raise CostModelError(f"the lock's bootstrap_settings are {v!r}, not fixed (T4.1)")
    level, n_boot, seed = v.get("level"), v.get("n_boot"), v.get("seed")
    for name, x in (("n_boot", n_boot), ("seed", seed)):
        if isinstance(x, bool) or not isinstance(x, int):
            raise CostModelError(f"the lock's bootstrap {name} is {x!r}, not an integer")
    if n_boot < 2 or seed < 0:
        raise CostModelError(f"the lock's bootstrap n_boot {n_boot} or seed {seed} is out of range")
    return Settings(_check_level(level), n_boot, seed)


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    level: float
    n_boot: int
    seed: int
    method: str = METHOD

    def to_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "level": self.level, "n_boot": self.n_boot, "seed": self.seed,
                "method": self.method}

    @staticmethod
    def from_dict(d: Mapping) -> "Interval":
        return Interval(**d)


def _check_level(level) -> float:
    if isinstance(level, bool) or not isinstance(level, (int, float)) \
            or not 0 < level < 1:
        raise CostModelError(f"level must be in (0, 1), got {level!r}")
    return float(level)


def percentile_interval(point: float, replicates: Sequence[float],
                        level: float, *, n_boot: int, seed: int) -> Interval:
    level = _check_level(level)
    r = np.asarray(replicates, dtype=float)
    if r.size == 0 or not np.all(np.isfinite(r)):
        raise CostModelError("replicates must be finite and non-empty")
    lo, hi = np.quantile(r, [(1 - level) / 2, (1 + level) / 2])
    return Interval(point=float(point), lo=float(lo), hi=float(hi),
                    level=level, n_boot=n_boot, seed=seed)


@dataclass(frozen=True)
class Bootstrap:
    point: Mapping[str, float]
    replicates: Mapping[str, tuple[float, ...]]
    n_boot: int
    seed: int

    def interval(self, name: str, level: float) -> Interval:
        return percentile_interval(self.point[name], self.replicates[name],
                                   level, n_boot=self.n_boot, seed=self.seed)

    def difference(self, a: str, b: str, level: float) -> Interval:
        """a - b, replicate by replicate."""
        ra, rb = np.asarray(self.replicates[a]), np.asarray(self.replicates[b])
        return percentile_interval(self.point[a] - self.point[b], ra - rb,
                                   level, n_boot=self.n_boot, seed=self.seed)


def _evaluate(evaluate: Evaluate, values: Mapping[str, float],
              keys: tuple[str, ...] | None) -> dict:
    out = dict(evaluate(values))
    if keys is not None and tuple(sorted(out)) != keys:
        raise CostModelError(
            f"evaluate returned {sorted(out)}, then {list(keys)}; every "
            f"replicate must give the same outputs")
    for k, v in out.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise CostModelError(f"output {k!r} is {v!r}, not a finite number")
    return out


def bootstrap(samples: Mapping[str, Sequence[float]], evaluate: Evaluate, *,
              n_boot: int, seed: int,
              statistic: Callable[[Sequence[float]], float] = statistics.median
              ) -> Bootstrap:
    """Resample every measurement and evaluate every output, `n_boot` times.

    `samples` maps a measurement's key to its repeats. `evaluate` maps
    {key: statistic} to {output name: seconds}. The statistic is the median
    unless stated, as for ground truth (training.GROUND_TRUTH_STATISTIC).
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise CostModelError(f"n_boot must be an int >= 2, got {n_boot!r}")
    if LOCKED is not None and n_boot < LOCKED.n_boot:
        raise CostModelError(f"n_boot {n_boot} is below the lock's {LOCKED.n_boot} (T4.1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CostModelError(f"seed must be an int >= 0, got {seed!r}")
    keys = tuple(sorted(samples))
    data = {}
    for k in keys:
        x = np.asarray(samples[k], dtype=float)
        if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
            raise CostModelError(
                f"measurement {k!r} needs at least two finite repeats; one gives every "
                f"replicate the sample itself")
        data[k] = x
    point = _evaluate(evaluate, {k: float(statistic(data[k])) for k in keys},
                      None)
    names = tuple(sorted(point))
    rng = np.random.Generator(np.random.PCG64(seed))
    reps: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(n_boot):
        values = {}
        for k in keys:
            x = data[k]
            values[k] = float(statistic(x[rng.integers(0, x.size, size=x.size)]))
        for n, v in _evaluate(evaluate, values, names).items():
            reps[n].append(float(v))
    return Bootstrap(point=point,
                     replicates={n: tuple(v) for n, v in reps.items()},
                     n_boot=n_boot, seed=seed)
