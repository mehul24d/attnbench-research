"""Residual analysis (T2.15), for T7.1's "residuals by arm and
synchronisation count".

A residual is a committed prediction against the measured statistic of the
same cell: predicted - measured in seconds, and divided by measured. Each
carries what the analysis groups by:

  - the model, the code path and the arm (the cell's key, so a residual is
    never pooled across a code path, rule 3);
  - its synchronisation count: the synchronisations in the committed
    sequence, not counting the closing synchronise every prefill ends with
    (Appendix A, 22 Oct 2026; commit.py checks it is there);
  - its arm class, the three H1 reports separately: dense (named by the
    caller), synchronisation-free (no synchronisation inside the step) and
    synchronising. The class is read from the committed sequence, so a
    compiled arm with no wait left is synchronisation-free whatever its name.

Errors inside the noise floor (Section 5.6) are not interpreted: each
residual is marked `inside_floor` when its absolute relative error is at or
below the floor the caller states, and every summary counts them. Nothing
is dropped. A committed cell with no measurement, and a measured cell with
no committed prediction, are both returned by name.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .commit import CommittedPrediction, PredictionCommitError, cell_key, require_loaded

DENSE = "dense"
SYNC_FREE = "synchronisation-free"
SYNCHRONISING = "synchronising"
GROUP_FIELDS = ("predictor", "model", "code_path", "arm", "n_synchronisations",
                "arm_class", "context_length", "batch", "density")


@dataclass(frozen=True)
class Residual:
    predictor: str
    model: str
    code_path: str
    arm: str
    context_length: int
    density: float
    batch: int
    n_synchronisations: int
    arm_class: str
    predicted: float
    measured: float
    residual: float           # predicted - measured, seconds
    relative: float           # residual / measured
    inside_floor: bool

    @property
    def abs_relative(self) -> float:
        return abs(self.relative)


@dataclass(frozen=True)
class ResidualSet:
    residuals: tuple[Residual, ...]
    unmeasured: tuple[tuple, ...]     # committed cells with no measurement
    unpredicted: tuple[tuple, ...]    # measured cells with no prediction


def synchronisations_in_step(cell) -> int:
    """Synchronisations, the closing synchronise not counted."""
    return sum(cell.priced.is_sync) - 1


def arm_class(arm: str, n_sync: int, dense_arms: Iterable[str]) -> str:
    if arm in set(dense_arms):
        return DENSE
    return SYNC_FREE if n_sync == 0 else SYNCHRONISING


def _floor(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) \
            or not math.isfinite(x) or x < 0:
        raise PredictionCommitError(
            f"the noise floor is a relative error >= 0, got {x!r}")
    return float(x)


def residuals(cp: CommittedPrediction, measured: Mapping[tuple, float], *,
              statistic: str, dense_arms: Iterable[str],
              relative_floor: float,
              predictors: Optional[Sequence[str]] = None) -> ResidualSet:
    """Residuals of every committed prediction (or of `predictors`) on the
    measured cells. `measured` maps a cell key (model, code path, arm,
    density, length, batch; `training.CellKey`) to the measured `statistic`,
    which must be the one the file predicts. `cp` must come from
    `load_committed` (`require_loaded`): an unchecked sync time is never scored."""
    cp = require_loaded(cp)
    if statistic != cp.statistic:
        raise PredictionCommitError(
            f"the file predicts the {cp.statistic}; these are {statistic}s")
    floor = _floor(relative_floor)
    dense_arms = set(dense_arms)
    if not dense_arms:
        raise PredictionCommitError("name the dense arm(s)")
    meas = {}
    for k, v in measured.items():
        k = cell_key(*k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v) or v <= 0:
            raise PredictionCommitError(f"measured {k} is {v!r}")
        meas[k] = float(v)
    committed = {c.key: c for c in cp.cells}
    out = []
    for k, c in committed.items():
        if k not in meas:
            continue
        n = synchronisations_in_step(c)
        cls = arm_class(c.arm, n, dense_arms)
        names = list(c.predictions) if predictors is None else list(predictors)
        for name in names:
            if name not in c.predictions:
                raise PredictionCommitError(
                    f"cell {k} has no committed {name!r} prediction")
            p, m = c.predictions[name], meas[k]
            rel = (p - m) / m
            out.append(Residual(
                predictor=name, model=c.model, code_path=c.code_path,
                arm=c.arm, context_length=c.context_length,
                density=c.density, batch=c.batch, n_synchronisations=n,
                arm_class=cls, predicted=p, measured=m, residual=p - m,
                relative=rel, inside_floor=abs(rel) <= floor))
    return ResidualSet(
        residuals=tuple(out),
        unmeasured=tuple(sorted(k for k in committed if k not in meas)),
        unpredicted=tuple(sorted(k for k in meas if k not in committed)))


@dataclass(frozen=True)
class GroupSummary:
    group: tuple
    n: int
    n_inside_floor: int
    median_abs_relative: float
    mean_relative: float          # the signed bias
    max_abs_relative: float


def summarise(rs: Iterable[Residual], by: Sequence[str]) -> list[GroupSummary]:
    """One summary per group, in sorted order of the group values."""
    by = tuple(by)
    bad = [f for f in by if f not in GROUP_FIELDS]
    if bad or not by:
        raise PredictionCommitError(
            f"group by some of {GROUP_FIELDS}, got {list(by)}")
    groups: dict[tuple, list[Residual]] = {}
    for r in rs:
        groups.setdefault(tuple(getattr(r, f) for f in by), []).append(r)
    out = []
    for g in sorted(groups):
        xs = groups[g]
        a = [r.abs_relative for r in xs]
        out.append(GroupSummary(
            group=g, n=len(xs), n_inside_floor=sum(r.inside_floor for r in xs),
            median_abs_relative=float(statistics.median(a)),
            mean_relative=float(statistics.mean(r.relative for r in xs)),
            max_abs_relative=float(max(a))))
    return out
