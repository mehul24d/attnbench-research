"""HDBI's boundedness threshold, fitted for the lock (Section 5.4; T4.1).

Section 5.4: "HDBI reported as a diagnostic only. Its boundedness is one
threshold, fitted on the training pairings by sign agreement, with ties
going to the lower value, and recorded at lock (Appendix A, 25 Oct 2026)."

**How the fit is read here** (the plan gives the rule's parts, not its
formula; the reading below is this module's, for the researcher to
confirm):

  - One threshold t splits HDBI (`predict.hdbi.hdbi`, sum d / (sum d +
    sum h)) into host-bound, HDBI < t, and device-bound, HDBI >= t.
  - A device-bound sparse cell is read as predicting that sparse is faster
    than dense (the device time sparse attention cuts is on the critical
    path); a host-bound one as predicting that it is not. HDBI is that of
    the sparse cell's own components, which hold its estimator's host work.
  - "Sign agreement" is H1's: the predicted sign of sparse minus dense
    against the measured one, outside the resolution floor. A cell whose
    measured difference is inside the floor is not scored (Appendix A,
    26 and 27 Oct 2026), and is counted apart.
  - The candidates are the HDBI values observed, and 1.0 (every cell
    host-bound). The fit is the candidate with the most agreements; among
    equals, the lowest.
  - Only rows from the training pairings are read, and the pairings are
    recorded with the fit.

`predict.hdbi.classify` takes two thresholds (host-bound below one,
device-bound above the other); the 25 Oct 2026 decision is one threshold, so
the classification here is binary: `classify(v, host_bound_below=t,
device_bound_above=t)` gives the same split except at exactly t, which
`classify` calls balanced and this module device-bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

HOST_BOUND = "host-bound"
DEVICE_BOUND = "device-bound"


class HdbiFitError(ValueError):
    """Rows the fit cannot be made from."""


@dataclass(frozen=True)
class HdbiRow:
    pairing: str
    cell: tuple                  # the cell key, for the record
    hdbi: float                  # of the sparse cell's components
    measured_difference: float   # sparse minus dense, same unit as the floor


@dataclass(frozen=True)
class HdbiFit:
    threshold: float
    agreements: int
    n_scored: int
    n_inside_floor: int
    pairings: tuple[str, ...]
    floor: float
    curve: tuple[tuple[float, int], ...]   # (candidate, agreements), ascending

    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.n_scored


def boundedness(value: float, threshold: float) -> str:
    return DEVICE_BOUND if value >= threshold else HOST_BOUND


def _finite(x, name: str, lo=-math.inf, hi=math.inf) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) \
            or not lo <= x <= hi:
        raise HdbiFitError(f"{name} must be a finite number in [{lo}, {hi}], got {x!r}")
    return float(x)


def fit_threshold(rows: Iterable[HdbiRow], *, training_pairings: Sequence[str],
                  floor: float) -> HdbiFit:
    """The threshold with the most sign agreements on the training pairings'
    rows; ties to the lower value (module docstring)."""
    training = tuple(sorted(set(training_pairings)))
    if not training:
        raise HdbiFitError("name the training pairings")
    floor = _finite(floor, "the resolution floor", 0.0)
    scored = []
    inside = 0
    seen = set()
    for r in rows:
        if r.pairing not in training:
            raise HdbiFitError(
                f"a row of {r.pairing}, which is not a training pairing {list(training)}; "
                f"the fit reads the training pairings only")
        if (r.pairing, r.cell) in seen:
            raise HdbiFitError(f"cell {r.cell} of {r.pairing} appears twice")
        seen.add((r.pairing, r.cell))
        h = _finite(r.hdbi, f"HDBI of {r.cell}", 0.0, 1.0)
        d = _finite(r.measured_difference, f"the measured difference of {r.cell}")
        if abs(d) <= floor:
            inside += 1
            continue
        scored.append((h, d < 0))
    if not scored:
        raise HdbiFitError("no row outside the resolution floor to fit on")
    candidates = sorted({h for h, _ in scored} | {1.0})
    curve = []
    for t in candidates:
        curve.append((t, sum((h >= t) == faster for h, faster in scored)))
    best = max(a for _, a in curve)
    threshold = min(t for t, a in curve if a == best)
    return HdbiFit(threshold=threshold, agreements=best, n_scored=len(scored),
                   n_inside_floor=inside, pairings=training, floor=floor,
                   curve=tuple(curve))
