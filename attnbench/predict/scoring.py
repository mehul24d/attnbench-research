"""Per-prediction errors of plan Section 5.4 (T2.19).

"Reported for every prediction: absolute relative error; decision error
(predicted profitable when it is not, and the reverse); crossover-length
error (crossover.py); and a prediction interval (intervals.py). Errors
inside the noise floor (Section 5.6) are not interpreted."

**Profitable** (Section 5.1): an end-to-end speedup above 1 over the
fastest correct dense kernel, at non-inferior accuracy. Admissibility is
an input, measured once on the L4 (or read from recall if H4 holds), the
same on the predicted and the measured side. So:

  - predicted profitable: admissible, and predicted sparse < predicted
    dense;
  - measured profitable: admissible, and measured dense - sparse above the
    resolution floor; measured not profitable: inadmissible, or measured
    sparse - dense above the floor; otherwise (admissible, within the
    floor) not interpreted.

The floor is applied to the measured side only: it is the noise in what
was measured. The dispatcher's own rule (dense unless sparse is predicted
faster by more than the floor, Section 5.9) is a choice, not this score;
it belongs to T2.22.

`sign_agreement` is H1's "sparse-vs-dense sign correct outside the
resolution floor": the decision error with admissibility set aside.
"""

from __future__ import annotations

import math
from typing import Optional

from .recursion import CostModelError

CORRECT = "correct"
FALSE_PROFITABLE = "predicted profitable when it is not"
MISSED_PROFITABLE = "predicted not profitable when it is"
NOT_INTERPRETED = "not interpreted: inside the resolution floor"


def _seconds(x, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) \
            or not math.isfinite(x) or x <= 0:
        raise CostModelError(f"{name} must be a positive finite time, got {x!r}")
    return float(x)


def _floor(floor) -> float:
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) \
            or not math.isfinite(floor) or floor < 0:
        raise CostModelError(
            f"the resolution floor is a time >= 0 (fixed at lock), got {floor!r}")
    return float(floor)


def absolute_relative_error(predicted: float, measured: float) -> float:
    """|predicted - measured| / measured, H1's statistic."""
    p = _seconds(predicted, "predicted")
    m = _seconds(measured, "measured")
    return abs(p - m) / m


def predicted_profitable(sparse: float, dense: float, admissible: bool) -> bool:
    _seconds(sparse, "predicted sparse")
    _seconds(dense, "predicted dense")
    return bool(admissible) and sparse < dense


def measured_profitable(sparse: float, dense: float, admissible: bool, *,
                        floor: float) -> Optional[bool]:
    _seconds(sparse, "measured sparse")
    _seconds(dense, "measured dense")
    floor = _floor(floor)
    if not admissible:
        return False
    if dense - sparse > floor:
        return True
    if sparse - dense > floor:
        return False
    return None


def decision_error(predicted_sparse: float, predicted_dense: float,
                   measured_sparse: float, measured_dense: float, *,
                   admissible: bool, floor: float) -> str:
    if not isinstance(admissible, bool):
        raise CostModelError(f"admissible must be a bool, got {admissible!r}")
    pred = predicted_profitable(predicted_sparse, predicted_dense, admissible)
    meas = measured_profitable(measured_sparse, measured_dense, admissible,
                               floor=floor)
    if meas is None:
        return NOT_INTERPRETED
    if pred and not meas:
        return FALSE_PROFITABLE
    if meas and not pred:
        return MISSED_PROFITABLE
    return CORRECT


def sign_agreement(predicted_sparse: float, predicted_dense: float,
                   measured_sparse: float, measured_dense: float, *,
                   floor: float) -> Optional[bool]:
    """Whether the predicted sign of sparse - dense is the measured one;
    None when the measured difference is inside the floor."""
    floor = _floor(floor)
    for x, n in ((predicted_sparse, "predicted sparse"),
                 (predicted_dense, "predicted dense"),
                 (measured_sparse, "measured sparse"),
                 (measured_dense, "measured dense")):
        _seconds(x, n)
    m = measured_sparse - measured_dense
    if abs(m) <= floor:
        return None
    return (predicted_sparse - predicted_dense > 0) == (m > 0)
