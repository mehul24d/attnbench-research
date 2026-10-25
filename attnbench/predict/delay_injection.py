"""The delay-injection slope check of Gate A (plan Section 7, T2.25).

"Each synchronisation mechanism is injected separately into the dense arm
on the L4, where the GPU is busy and where it is idle, and the ratio of
predicted to measured time per injected synchronisation is within a bound
of 1.0 fixed at lock, one slope per mechanism (double counting gives about
2; an instrument check, Section 10)."

This module is the prediction and scoring side; the GPU session (T3.1)
times the variants it plans.

**The variants.** For one mechanism and one placement, the dense arm's
recorded sequence (ending with its closing synchronise) is copied with c
synchronisations of that mechanism inserted, for each count c in a grid
that includes 0. Where they go:

  - *busy*: one synchronisation after each of c kernels or copies at which
    the device is still working when the host has issued it (D_k > H_k in
    the base timeline), spread evenly over those sites. Every kernel with
    device time is such a site: right after its launch the device holds at
    least that kernel. What differs between a host-bound and a device-bound
    stretch is how long the injected synchronisation waits;
  - *idle*: all c synchronisations together at one site where the device
    has nothing pending, by default before the first operation (the device
    is idle at time 0), so each one after the first finds it drained too.
    After a launch the device is never idle, so the other idle sites are
    after a synchronisation or after an operation with no device time.

A variant's waits are the host's whole wait at each injected
synchronisation, the recursion's sync wait plus its queue stall there.

The recursion then predicts every variant, and its own timeline is checked:
in a busy variant every injected synchronisation must wait for the device,
and in an idle one none may. A plan that fails this is refused rather than
mislabelled.

**The slopes.** Per placement, the predicted and the measured end-to-end
times are each regressed on c by ordinary least squares; the placement's
ratio is predicted slope over measured slope. The one slope per mechanism
pools both placements: the predicted increments over c = 0 are regressed
on the measured increments, through the origin. Exactly 1 means each
injected synchronisation costs what the model charges; about 2 means the
model charges it twice.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .recursion import (
    CostModelError, Operation, SyncCosts, is_closing_sync, price, recurse,
    sync_op)

BUSY = "busy"
IDLE = "idle"
PLACEMENTS = (BUSY, IDLE)
START = -1    # the site before the first operation
INJECTED_LABEL = "injected synchronisation"


def _check_base(base: Sequence[Operation]) -> list[Operation]:
    base = list(base)
    if not base or not is_closing_sync(base[-1]):
        raise CostModelError(
            "the base sequence must be the dense arm's prefill, ending with "
            "its closing synchronise (recursion.close_prefill)")
    return base


def busy_sites(base: Sequence[Operation], sync_costs: SyncCosts,
               queue_depth: Optional[int]) -> tuple[int, ...]:
    """Indices k of kernels and copies after which the device is still
    busy: D_k > H_k in the base timeline. A synchronisation has D_k = H_k,
    so it is never one."""
    base = _check_base(base)
    t = recurse(price(base, sync_costs), queue_depth)
    return tuple(k for k in range(len(base) - 1)
                 if t.device_done[k] > t.host_issued[k])


def spread(sites: Sequence[int], count: int) -> tuple[int, ...]:
    """`count` sites spread evenly over `sites`: the j-th is
    sites[floor((2j + 1) n / 2c)], which are distinct for count <= n."""
    n = len(sites)
    if count > n:
        raise CostModelError(
            f"{count} injections need {count} distinct busy sites; the base "
            f"sequence has {n}")
    return tuple(sites[((2 * j + 1) * n) // (2 * count)] for j in range(count))


def inject(base: Sequence[Operation], sites: Sequence[int],
           mechanism: str, *, payload_bytes: int = 0,
           loaded: bool = False) -> tuple[list[Operation], tuple[int, ...]]:
    """The base sequence with one synchronisation inserted after each listed
    site (repeats allowed; START is before the first operation). Returns the
    new sequence and the positions of the injected operations in it."""
    base = _check_base(base)
    last = len(base) - 2          # the closing synchronise stays last
    for s in sites:
        if not START <= s <= last:
            raise CostModelError(
                f"site {s} is outside the arm's operations ({START} to {last})")
    counts: dict[int, int] = {}
    for s in sites:
        counts[s] = counts.get(s, 0) + 1
    out: list[Operation] = []
    positions = []

    def add_syncs(site):
        for _ in range(counts.get(site, 0)):
            positions.append(len(out))
            out.append(sync_op(mechanism, payload_bytes=payload_bytes,
                               loaded=loaded, layer=None,
                               label=INJECTED_LABEL))

    add_syncs(START)
    for k, op in enumerate(base):
        out.append(op)
        add_syncs(k)
    return out, tuple(positions)


@dataclass(frozen=True)
class Variant:
    mechanism: str
    placement: str
    count: int
    sites: tuple[int, ...]
    positions: tuple[int, ...]       # of the injected operations
    predicted: float                 # D_N, seconds
    predicted_waits: tuple[float, ...]


def _check_counts(counts: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(counts)
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0
           for c in counts):
        raise CostModelError(f"counts are ints >= 0, got {counts}")
    if 0 not in counts or len(set(counts)) < 2 or list(counts) != sorted(set(counts)):
        raise CostModelError(
            f"counts must be distinct, ascending, include 0 and one more, got "
            f"{counts}")
    return counts


def plan_variants(base: Sequence[Operation], mechanism: str, placement: str,
                  counts: Sequence[int], sync_costs: SyncCosts,
                  queue_depth: Optional[int], *, payload_bytes: int = 0,
                  loaded: bool = False,
                  idle_site: int = START) -> tuple[Variant, ...]:
    """Every variant for one mechanism and placement, predicted."""
    if placement not in PLACEMENTS:
        raise CostModelError(f"placement is one of {PLACEMENTS}, got {placement!r}")
    counts = _check_counts(counts)
    base = _check_base(base)
    if placement == BUSY:
        candidates = busy_sites(base, sync_costs, queue_depth)
    out = []
    for c in counts:
        sites = spread(candidates, c) if placement == BUSY else (idle_site,) * c
        ops, positions = inject(base, sites, mechanism,
                                payload_bytes=payload_bytes, loaded=loaded)
        t = recurse(price(ops, sync_costs), queue_depth)
        # The host's whole wait at the synchronisation: with a queue term the
        # stall may already have held it until the device drained, and the
        # recursion then books the wait as queue stall, not sync wait.
        waits = tuple(t.sync_wait[p] + t.queue_stall[p] for p in positions)
        if placement == BUSY and any(w <= 0 for w in waits):
            raise CostModelError(
                f"{c} busy injections of {mechanism}: the synchronisations at "
                f"{[p for p, w in zip(positions, waits) if w <= 0]} find the "
                f"device idle; choose fewer or other sites")
        if placement == IDLE and any(w > 0 for w in waits):
            raise CostModelError(
                f"{c} idle injections of {mechanism} at site {idle_site}: the "
                f"device is still busy there")
        out.append(Variant(mechanism=mechanism, placement=placement, count=c,
                           sites=tuple(sites), positions=positions,
                           predicted=t.end_to_end, predicted_waits=waits))
    return tuple(out)


# --------------------------------------------------------------------------
# Scoring against the GPU's measurements
# --------------------------------------------------------------------------

def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom == 0:
        raise CostModelError("a slope needs at least two distinct counts")
    return float(xc @ (y - y.mean())) / denom


@dataclass(frozen=True)
class PlacementSlope:
    placement: str
    counts: tuple[int, ...]
    predicted: tuple[float, ...]
    measured: tuple[float, ...]       # the statistic of each count's repeats
    predicted_slope: float            # seconds per injected synchronisation
    measured_slope: float
    ratio: float                      # predicted over measured


@dataclass(frozen=True)
class MechanismSlope:
    mechanism: str
    placements: Mapping[str, PlacementSlope]
    slope: float                      # pooled, through the origin


def _stat(samples: Sequence[float], statistic) -> float:
    xs = [float(s) for s in samples]
    if not xs or any(not math.isfinite(s) or s <= 0 for s in xs):
        raise CostModelError("timed repeats must be positive, finite, non-empty")
    return float(statistic(xs))


def slope_for(mechanism: str, variants: Sequence[Variant],
              measured: Mapping[tuple[str, int], Sequence[float]], *,
              statistic=statistics.median) -> MechanismSlope:
    """`measured` maps (placement, count) to the timed repeats of that
    variant, in seconds; every planned variant needs one."""
    placements = {}
    inc_p, inc_m = [], []
    by_place: dict[str, list[Variant]] = {}
    for v in variants:
        if v.mechanism != mechanism:
            raise CostModelError(f"a {v.mechanism} variant among {mechanism}'s")
        by_place.setdefault(v.placement, []).append(v)
    if not by_place:
        raise CostModelError(f"no variants for {mechanism}")
    extra = set(measured) - {(v.placement, v.count) for v in variants}
    if extra:
        raise CostModelError(f"measurements for unplanned variants {sorted(extra)}")
    for place in sorted(by_place):
        vs = sorted(by_place[place], key=lambda v: v.count)
        counts = tuple(v.count for v in vs)
        _check_counts(counts)
        missing = [c for c in counts if (place, c) not in measured]
        if missing:
            raise CostModelError(
                f"{mechanism}/{place}: no measurement for counts {missing}")
        pred = tuple(v.predicted for v in vs)
        meas = tuple(_stat(measured[(place, c)], statistic) for c in counts)
        ps, ms = ols_slope(counts, pred), ols_slope(counts, meas)
        if ms <= 0:
            raise CostModelError(
                f"{mechanism}/{place}: the measured time does not rise with "
                f"the number of injected synchronisations (slope {ms}); the "
                f"injection did not take effect, or the timing cannot "
                f"resolve it")
        placements[place] = PlacementSlope(
            placement=place, counts=counts, predicted=pred, measured=meas,
            predicted_slope=ps, measured_slope=ms, ratio=ps / ms)
        i0 = counts.index(0)
        for i, c in enumerate(counts):
            if c:
                inc_p.append(pred[i] - pred[i0])
                inc_m.append(meas[i] - meas[i0])
    m = np.asarray(inc_m)
    pooled = float(np.asarray(inc_p) @ m) / float(m @ m)
    return MechanismSlope(mechanism=mechanism, placements=placements,
                          slope=pooled)


def gate(slopes: Mapping[str, MechanismSlope], *, bound: float,
         mechanisms: Sequence[str]) -> dict[str, bool]:
    """Whether each mechanism's slope is within `bound` of 1.0 (the bound is
    fixed at lock). Every listed mechanism must have been measured: Gate A
    reads one slope per mechanism."""
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) \
            or not math.isfinite(bound) or bound <= 0:
        raise CostModelError(f"the slope bound is a number > 0, got {bound!r}")
    missing = [m for m in mechanisms if m not in slopes]
    if missing:
        raise CostModelError(f"no slope for {missing}")
    return {m: abs(slopes[m].slope - 1.0) <= bound for m in mechanisms}
