"""The host-speed probe bounds (T4.1 item "probe bounds"; the lock-wave
rulings of 12 Nov 2026, item 4): flags the analysis reports beside a
pairing's results, never a stop. No session reads them.

A probe row is one `host_speed_<tag>.parquet` row (`t31_steps`, the
postflight probe). Three checks, each against the lock's bound:

  - `loop_cv`: the row's loop CV (`loop_s_per_iter_cv`) at most the bound;
  - `loop_median_relative`: |loop median - reference| / reference at most
    the bound, the reference the source pairing's;
  - `null_median_relative`: the same for the null-launch median
    (`null_launch_s_median`). The null launch is heavy-tailed (CV 1.2 to 1.5
    on P1), so only its median is bounded.

The source pairing's reference is the median over its repeats' rows
(`reference`): for P2 and P4, P1's four repeats T3.2a to T3.2d.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

BOUND_KEYS = ("loop_cv", "loop_median_relative", "null_median_relative")
LOOP_MEDIAN = "loop_s_per_iter_median"
LOOP_CV = "loop_s_per_iter_cv"
NULL_MEDIAN = "null_launch_s_median"


class ProbeFlagError(ValueError):
    """A probe row or bound the flags cannot be read from."""


def _value(row: Mapping, field: str) -> float:
    v = row.get(field)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ProbeFlagError(f"the probe row's {field} is not a time or ratio: {v!r}")
    return float(v)


def reference(rows: Sequence[Mapping]) -> dict:
    """The source pairing's loop and null-launch medians: the median over
    its repeats' probe rows."""
    if not rows:
        raise ProbeFlagError("the reference needs the source pairing's probe rows")
    return {LOOP_MEDIAN: statistics.median(_value(r, LOOP_MEDIAN) for r in rows),
            NULL_MEDIAN: statistics.median(_value(r, NULL_MEDIAN) for r in rows)}


def probe_flags(row: Mapping, ref: Mapping, bounds: Mapping) -> list[dict]:
    """One entry per check: {"check", "value", "bound", "flagged"}."""
    if set(bounds) != set(BOUND_KEYS):
        raise ProbeFlagError(f"the probe bounds are {list(BOUND_KEYS)}, got {sorted(bounds)}")
    out = []

    def add(check, value):
        b = float(bounds[check])
        out.append({"check": check, "value": value, "bound": b, "flagged": value > b})
    add("loop_cv", _value(row, LOOP_CV))
    for check, field in (("loop_median_relative", LOOP_MEDIAN),
                         ("null_median_relative", NULL_MEDIAN)):
        r = _value(ref, field)
        if r <= 0:
            raise ProbeFlagError(f"the reference {field} is not positive: {r!r}")
        add(check, abs(_value(row, field) - r) / r)
    return out
