"""Model ablations of plan Section 5.4 (T2.9), from the same components.

"The recursion without the synchronisation term" can mean two different
models, so both are built and named:

  - `no_sync_cost`: s_0 = 0 at every synchronisation, the wait kept. The
    host still stops until the device drains; only the mechanism's own
    cost is gone.
  - `no_sync_wait`: the dependency removed. The host pays s_0 and goes on
    without waiting for the device (`recursion.recurse(sync_wait=False)`).

Both act on the arm's own synchronisations only. The closing synchronise
belongs to the ground-truth timer (Appendix A, 22 Oct 2026), so it keeps its
s_0 and its wait, and an arm with no synchronisation of its own predicts the
same under either ablation as under the full recursion. Both therefore read
a closed prefill (`recursion.is_closed`) and refuse any other.

The host-only and device-only lower bounds are sum h and sum l + sum d, as
written. The queue sensitivity's multiples of Q are rounded here, once, so
every caller gets the same depths.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

from .recursion import (CostModelError, PricedSequence, Timeline, is_closed,
                        recurse)

# Section 5.4: "recomputed with Q at 0.25x, 0.5x, 2x and 4x the measured
# value, and with no queue term" (the last is queue_depth=None).
QUEUE_MULTIPLES = (0.25, 0.5, 2, 4)


def no_sync_cost(priced: PricedSequence,
                 queue_depth: Optional[int]) -> Timeline:
    """The recursion with s_0 = 0 and the wait at each synchronisation kept.
    The closing synchronise keeps its s_0."""
    if not is_closed(priced):
        raise CostModelError(
            "the no-cost ablation keeps the closing synchronise's s_0, and "
            "this sequence does not end with one; build it with close_prefill")
    zero = dataclasses.replace(
        priced, sync_cost=(0.0,) * (len(priced) - 1) + priced.sync_cost[-1:])
    return recurse(zero, queue_depth)


def no_sync_wait(priced: PricedSequence,
                 queue_depth: Optional[int]) -> Timeline:
    """The recursion with s_0 kept and no wait for the device, except at the
    closing synchronise."""
    return recurse(priced, queue_depth, sync_wait=False)


def host_only_bound(priced: PricedSequence) -> float:
    """sum h, a lower bound on D_N."""
    return float(sum(priced.host))


def device_only_bound(priced: PricedSequence) -> float:
    """sum l + sum d, a lower bound on D_N."""
    return float(sum(priced.link) + sum(priced.device))


def _round_depth(x: float) -> int:
    # Nearest, with halves rounded up (Python's round() would send 0.5 and
    # 2.5 to the even neighbour), and never below one launch.
    return max(1, math.floor(x + 0.5))


def sensitivity_depths(measured_q: int) -> dict[float, int]:
    """Each multiple of Section 5.4 and the Q it rounds to.

    Several multiples can round to the same Q, and a multiple can round to
    the measured Q itself (Q = 1 gives 1 for both 0.25x and 0.5x). Every
    multiple is still reported; `distinct_sensitivity_depths` only saves
    recomputing the same depth twice.
    """
    if isinstance(measured_q, bool) or not isinstance(measured_q, int) \
            or measured_q < 1:
        raise CostModelError(
            f"the measured queue depth is a count of launches >= 1, got "
            f"{measured_q!r}")
    return {m: _round_depth(m * measured_q) for m in QUEUE_MULTIPLES}


def distinct_sensitivity_depths(measured_q: int) -> tuple[int, ...]:
    """The distinct depths of `sensitivity_depths`, ascending. "No queue
    term" is not among them: pass queue_depth=None for it."""
    return tuple(sorted(set(sensitivity_depths(measured_q).values())))
