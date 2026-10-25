"""HDBI, the host-device balance index, as a diagnostic (plan Section 5.4,
T2.9). Reported only; it predicts nothing.

TaxBreak (Vellaisamy et al., ISPASS 2026; arXiv:2603.12465), Eq. 3:

    HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestration),  in (0, 1)

with T_DeviceActive the summed kernel execution time and T_Orchestration the
summed host orchestration overhead (framework translation, CUDA library
front end, launch floor) over all kernel invocations. HDBI -> 0 is a
host-bound regime and HDBI -> 1 a device-bound one.

Here it is computed from the recursion's components as sum d / (sum d +
sum h) (researcher's decision, 25 Oct 2026). Three things sit outside it:

  - link costs l: copies are not kernel executions, so sum l is in neither
    term;
  - s_0: a synchronisation's cost is neither kernel time nor orchestration;
  - nothing else is removed from sum h, so it is not exactly TaxBreak's
    T_Orchestration: c_host (Section 5.1) also holds host-side compute such
    as CPU mask building, and the host cost of issuing copies.

TaxBreak names the two limits but gives no thresholds between "host-bound",
"balanced" and "device-bound" (it reads 0.15 to 0.24 as host-bound, 0.37
to 0.41 as moderately balanced and 0.93 as strongly device-bound, and notes
host gains weaken above about 0.3). `classify` therefore takes both
thresholds from the caller and has no defaults.
"""

from __future__ import annotations

import math

from .recursion import CostModelError, PricedSequence

HOST_BOUND = "host-bound"
BALANCED = "balanced"
DEVICE_BOUND = "device-bound"


def hdbi(priced: PricedSequence) -> float:
    """sum d / (sum d + sum h)."""
    sum_d = float(sum(priced.device))
    sum_h = float(sum(priced.host))
    if sum_d + sum_h <= 0:
        raise CostModelError(
            "HDBI is undefined with no device time and no host time")
    return sum_d / (sum_d + sum_h)


def classify(value: float, *, host_bound_below: float,
             device_bound_above: float) -> str:
    """Host-bound below the first threshold, device-bound above the second,
    balanced between (both thresholds inclusive of balanced)."""
    for name, x in (("value", value), ("host_bound_below", host_bound_below),
                    ("device_bound_above", device_bound_above)):
        if (isinstance(x, bool) or not isinstance(x, (int, float))
                or not math.isfinite(x) or not 0 <= x <= 1):
            raise CostModelError(f"{name} must be a number in [0, 1], got {x!r}")
    if not host_bound_below <= device_bound_above:
        raise CostModelError(
            f"host_bound_below ({host_bound_below}) must not exceed "
            f"device_bound_above ({device_bound_above})")
    if value < host_bound_below:
        return HOST_BOUND
    if value > device_bound_above:
        return DEVICE_BOUND
    return BALANCED
