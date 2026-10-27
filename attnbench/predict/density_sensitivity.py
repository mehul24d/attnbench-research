"""The density sensitivity of plan Section 5.4 (T2.28).

"Every committed prediction is also recomputed at plus and minus the
measured cross-path density difference (the mask gate, Section 7)." The
cross-path difference is how far a sparse arm's mask density moves when the
code path changes: XAttention scores with Triton on the A100 and H100 and
with PyTorch elsewhere, and the masks, and so the densities, are not
guaranteed to agree (Section 9, "one code-path mismatch with three faces").
The committed prediction uses the masks recorded for its code path; the
sensitivity asks what it would be if the density were off by the amount the
gate measured.

**What is recomputed, and from what.** The committed file holds every cell's
priced components. An arm is committed at several densities (the ground-truth
density set includes every admissible density, Section 5.3 step 7), each with
its own device costs from the replay of its own recorded masks (T2.4), so the
file already holds how the arm's components respond to density. The
sensitivity reads that response and nothing else:

  - for a cell at density d, the *family* is the cells of the same model,
    code path, arm, context length and batch (the key without the density);
  - each position's host, link, device and synchronisation cost is
    interpolated linearly in density between the two committed densities
    nearest the target (extrapolated from the nearest two when the target
    lies outside the committed range, and then flagged `extrapolated`);
  - the recursion, and every other predictor that reads only the components,
    is re-run on the interpolated components at the committed Q.

Nothing is fitted and nothing is measured here. Linear in density is an
assumption about how a kernel's time moves between two replayed densities;
with three or more committed densities `nonlinearity` reports how far the
middle ones sit from the chord, so a family that is not linear says so
instead of being trusted.

**What cannot be recomputed**, and is listed rather than skipped: the learned
predictor (its fitted model is not in the file), the nearest-pairing baseline
(another pairing's measurement, which has no density response here), a family
with one density (nothing to interpolate), a family whose cells have
different operation sequences, a family with two cells at one density, a
density outside (0, 1], and a recomputed component that is negative.
The dense arm has no density; its cells are reported unchanged.

**Units.** `deltas` is the measured cross-path difference as an absolute
change in mask density (the fraction of blocks kept, Appendix A, 25 Oct 2026),
by arm or by (arm, context length). It is an input from the mask gate (T2.23),
never a default: an arm with no entry is listed as not measured. The
abscissa of the interpolation is the key's density unless the caller gives
`mask_density`, the measured mask density of each cell (T2.20), which is then
what the delta is added to; the unit of `deltas` and of the abscissa must be
the same thing.

`decision_flips` asks the question the sensitivity is for: does the predicted
sign of sparse minus dense change within plus or minus the difference? The
dense cell does not move with density.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .baselines import step_form, sum_as_written, sum_plus_s0
from .commit import (CommittedCell, CommittedPrediction, PredictionCommitError,
                     _constant_host_prediction, _fitted_from_record)
from .recursion import CostModelError, PricedSequence, recurse
from .sensitivity import dense_partners, partner_of
from .training import CellKey

COMPONENT_FIELDS = ("host", "link", "device", "sync_cost")
LAYOUT_FIELDS = ("kinds", "layers", "mechanisms", "payload_bytes", "is_sync")
# What the file lets us recompute, and what it does not.
RECOMPUTABLE = ("recursion", "sum", "sum_plus_s0", "step_form", "fitted_sum",
                "fitted_sum_plus_s0", "constant_host_cost")
NOT_RECOMPUTABLE = {
    "learned": "the learned predictor's fitted model is not in the file",
    "nearest_pairing": "it is another pairing's measurement, which has no "
                       "density response",
}
SIGNS = (-1, 0, 1)


@dataclass(frozen=True)
class DensityPoint:
    """One predictor on one cell at density + sign * delta."""

    sign: int                     # -1, 0 (the committed value) or +1
    density: float                # the abscissa the prediction is at
    predicted: float
    relative_change: float        # (predicted - committed) / committed
    extrapolated: bool


@dataclass(frozen=True)
class CellDensity:
    key: CellKey
    predictor: str
    delta: float
    committed: float
    points: tuple[DensityPoint, ...]      # sign -1, 0, +1, in that order
    nonlinearity: Optional[float]         # see `_nonlinearity`; None: < 3 densities

    def at(self, sign: int) -> DensityPoint:
        for p in self.points:
            if p.sign == sign:
                return p
        raise KeyError(sign)


@dataclass(frozen=True)
class NotRecomputed:
    key: CellKey
    predictor: Optional[str]      # None: the whole cell
    reason: str


@dataclass(frozen=True)
class DensitySensitivity:
    cells: tuple[CellDensity, ...]
    not_recomputed: tuple[NotRecomputed, ...]
    abscissa: str                 # "key density" or "mask density"


def _delta_for(deltas: Mapping, arm: str, context_length: int) -> Optional[float]:
    for k in ((arm, context_length), arm):
        if k in deltas:
            return deltas[k]
    return None


def _check_delta(d, where: str) -> float:
    if isinstance(d, bool) or not isinstance(d, (int, float)) \
            or not math.isfinite(d) or d < 0:
        raise PredictionCommitError(
            f"{where}: the cross-path density difference is an absolute "
            f"change in density >= 0, got {d!r}")
    return float(d)


def _layout(p: PricedSequence) -> tuple:
    return tuple(getattr(p, f) for f in LAYOUT_FIELDS)


def _interpolate(xs: Sequence[float], values: Sequence[Sequence[float]],
                 x: float) -> tuple[list[float], bool]:
    """Position-wise linear interpolation of `values` (one array per x,
    `xs` strictly increasing) at `x`, from the two nearest committed points,
    extrapolating from the nearest two outside the range."""
    n = len(xs)
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = n - 2
    else:
        i = max(j for j in range(n - 1) if xs[j] <= x)
    x0, x1 = xs[i], xs[i + 1]
    w = (x - x0) / (x1 - x0)
    out = [a + (b - a) * w for a, b in zip(values[i], values[i + 1])]
    return out, not (xs[0] <= x <= xs[-1])


def _nonlinearity(xs: Sequence[float], totals: Sequence[float]) -> Optional[float]:
    """The largest distance of a middle point from the chord through the end
    points, as a share of the larger end value. 0 for a straight line; None
    with fewer than three densities. `totals` is the family's committed
    recursion prediction at each density."""
    if len(xs) < 3:
        return None
    x0, x1, y0, y1 = xs[0], xs[-1], totals[0], totals[-1]
    worst = max(abs(y - (y0 + (y1 - y0) * (x - x0) / (x1 - x0)))
                for x, y in zip(xs[1:-1], totals[1:-1]))
    return worst / max(abs(y0), abs(y1))


def _predict(cp: CommittedPrediction, predictor: str,
             priced: PricedSequence) -> float:
    if predictor == "recursion":
        return recurse(priced, cp.queue_depth).end_to_end
    if predictor == "sum":
        return sum_as_written(priced)
    if predictor == "sum_plus_s0":
        return sum_plus_s0(priced)
    if predictor == "step_form":
        return step_form(priced)
    if predictor in ("fitted_sum", "fitted_sum_plus_s0"):
        return _fitted_from_record(cp.training[predictor], cp.pairing).predict(priced)
    if predictor == "constant_host_cost":
        return _constant_host_prediction(priced, cp.constant_host_cost,
                                         cp.queue_depth)
    raise PredictionCommitError(f"{predictor!r} cannot be recomputed")


def _with_components(base: PricedSequence, comps: Mapping[str, list]) -> PricedSequence:
    for f, xs in comps.items():
        if any(not math.isfinite(v) or v < -1e-15 for v in xs):
            raise CostModelError(f"the interpolated {f} has a negative or "
                                 f"non-finite component")
    return dataclasses.replace(base, **{f: tuple(max(0.0, v) for v in xs)
                                        for f, xs in comps.items()})


def density_sensitivity(cp: CommittedPrediction, *, deltas: Mapping,
                        dense_arms: Iterable[str],
                        mask_density: Optional[Mapping[tuple, float]] = None,
                        predictors: Optional[Sequence[str]] = None
                        ) -> DensitySensitivity:
    """Recompute `cp`'s predictions at density - delta and density + delta.

    `deltas`: arm -> difference, or (arm, context_length) -> difference
    (module docstring). `dense_arms`: arms that have no density.
    `mask_density`: measured mask density per cell key, the abscissa to use
    instead of the key's density. `predictors`: default is every committed
    prediction that can be recomputed; a requested one that cannot is
    refused.
    """
    dense_arms = set(dense_arms)
    if not dense_arms:
        raise PredictionCommitError("name the dense arm(s)")
    if predictors is not None:
        bad = [p for p in predictors if p not in RECOMPUTABLE]
        if bad:
            why = "; ".join(f"{p}: {NOT_RECOMPUTABLE.get(p, 'not a predictor')}"
                            for p in bad)
            raise PredictionCommitError(
                f"cannot recompute {bad} ({why}); the choices are {RECOMPUTABLE}")
    mask_density = {CellKey(*k): float(v) for k, v in (mask_density or {}).items()}
    committed = {c.key: c for c in cp.cells}
    extra = set(mask_density) - set(committed)
    if extra:
        raise PredictionCommitError(
            f"mask densities for cells not in the file: {sorted(extra)}")
    if mask_density:
        missing = sorted(set(committed) - set(mask_density))
        if missing:
            raise PredictionCommitError(
                f"mask densities are given for some cells and not for "
                f"{missing}; one abscissa for the whole file")
    xof = (lambda c: mask_density[c.key]) if mask_density else (lambda c: c.density)

    families: dict = {}
    for c in cp.cells:
        families.setdefault(
            (c.model, c.code_path, c.arm, c.context_length, c.batch), []).append(c)

    cells, skipped = [], []
    for fam_key, fam in families.items():
        for c in sorted(fam, key=lambda c: c.density):
            names = [p for p in (predictors or RECOMPUTABLE) if p in c.predictions]
            if predictors is None:
                for n in NOT_RECOMPUTABLE:
                    if n in c.predictions:
                        skipped.append(NotRecomputed(c.key, n, NOT_RECOMPUTABLE[n]))
            for n in predictors or ():
                if n not in c.predictions:
                    raise PredictionCommitError(
                        f"cell {tuple(c.key)} has no committed {n!r} prediction")
            if c.arm in dense_arms:
                for n in names:
                    cells.append(CellDensity(
                        key=c.key, predictor=n, delta=0.0,
                        committed=c.predictions[n], nonlinearity=None,
                        points=tuple(DensityPoint(s, xof(c), c.predictions[n], 0.0,
                                                  False) for s in SIGNS)))
                continue
            delta = _delta_for(deltas, c.arm, c.context_length)
            if delta is None:
                skipped.append(NotRecomputed(
                    c.key, None, f"no cross-path density difference was "
                    f"measured for arm {c.arm!r} at {c.context_length}"))
                continue
            delta = _check_delta(delta, f"{c.arm} at {c.context_length}")
            try:
                _recompute_cell(cp, c, fam, xof, delta, names, cells)
            except CostModelError as e:
                skipped.append(NotRecomputed(c.key, None, str(e)))
    return DensitySensitivity(
        cells=tuple(cells), not_recomputed=tuple(skipped),
        abscissa="mask density" if mask_density else "key density")


def _recompute_cell(cp, c: CommittedCell, fam, xof, delta: float, names,
                    out: list) -> None:
    ordered = sorted(fam, key=xof)
    xs = [xof(f) for f in ordered]
    if len(xs) < 2:
        raise CostModelError(
            f"only one density is committed for {c.arm!r} at "
            f"{c.context_length}, so there is no response to interpolate")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise CostModelError(
            f"two cells of {c.arm!r} at {c.context_length} share a density "
            f"{[x for a, x in zip(xs, xs[1:]) if x <= a]}")
    layouts = {_layout(f.priced) for f in ordered}
    if len(layouts) > 1:
        raise CostModelError(
            f"the cells of {c.arm!r} at {c.context_length} have different "
            f"operation sequences at different densities, so their "
            f"components are not comparable position by position")
    x0 = xof(c)
    targets = {s: x0 + s * delta for s in SIGNS}
    for s, x in targets.items():
        if not 0 < x <= 1:
            raise CostModelError(
                f"density {x:g} ({'+' if s > 0 else '-'}{delta:g} from "
                f"{x0:g}) is outside (0, 1]")
    series = {f: [getattr(g.priced, f) for g in ordered] for f in COMPONENT_FIELDS}
    priced_at, extrapolated = {}, {}
    for s, x in targets.items():
        comps, ext = {}, False
        for f in COMPONENT_FIELDS:
            comps[f], e = _interpolate(xs, series[f], x)
            ext = ext or e
        priced_at[s] = _with_components(c.priced, comps)
        extrapolated[s] = ext
    nonlin = _nonlinearity(xs, [g.predictions["recursion"] for g in ordered])
    for n in names:
        committed = c.predictions[n]
        pts = []
        for s in SIGNS:
            v = committed if s == 0 else _predict(cp, n, priced_at[s])
            pts.append(DensityPoint(sign=s, density=targets[s], predicted=v,
                                    relative_change=(v - committed) / committed,
                                    extrapolated=extrapolated[s]))
        out.append(CellDensity(key=c.key, predictor=n, delta=delta,
                               committed=committed, points=tuple(pts),
                               nonlinearity=nonlin))


@dataclass(frozen=True)
class DensityDecision:
    sparse_key: CellKey
    dense_key: CellKey
    delta: float
    committed_sparse_faster: bool
    flips_at: tuple[int, ...]     # signs (-1, +1) where the sign of sparse - dense changes


def decision_flips(result: DensitySensitivity, *, dense_arm: str,
                   predictor: str = "recursion") -> list[DensityDecision]:
    """For each sparse cell that was recomputed, the offsets at which "sparse
    is predicted faster than dense" differs from the committed prediction.
    A tie is dense (Appendix A, 26 Oct 2026), as everywhere. The dense cell
    does not move with density. A sparse cell with no dense cell at its
    model, length and batch is left out."""
    by_key = {c.key: c for c in result.cells if c.predictor == predictor}
    dense = {w: by_key[k] for w, k in dense_partners(by_key, dense_arm).items()}
    out = []
    for k, r in by_key.items():
        if k.arm == dense_arm:
            continue
        d = dense.get(partner_of(k))
        if d is None:
            continue
        base = r.committed < d.committed
        flips = tuple(p.sign for p in r.points
                      if p.sign and (p.predicted < d.committed) != base)
        out.append(DensityDecision(
            sparse_key=k, dense_key=d.key, delta=r.delta,
            committed_sparse_faster=base, flips_at=flips))
    return out
