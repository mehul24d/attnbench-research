"""EXPLORATORY / POST HOC: the paper-quality tables (the researcher's
paper-quality wave of 28 Nov 2026, items 2 to 5). Nothing here scores,
rescores or replaces a pre-registered result: H0-H3 stand as scored at
`83b2edf`, H1 stays 0.1183 on P2 and 0.2495 on P4, and the timed-path pin
`0e0fb012` is not read or touched. No parameter is fitted on P2 or P4.

Every function works on one committed `PricedSequence` and its queue depth,
exactly as `attnbench.predict.recursion.recurse` reads them, so each variant
below is the committed model with terms removed or one input repriced:

  - `knockouts`: the committed recursion with one term removed at a time
    (`VARIANTS`). Removing a term never adds a parameter.
  - `miss_split`: the committed error split exactly into a device-pricing
    part (priced device chain minus traced device-busy time) and an idle
    part (predicted idle minus measured idle). The traced busy time comes
    from a profiled run, so the device part assumes traced device durations
    equal untraced ones; no session records an untraced host timeline, so the
    measured idle time cannot be split further.
  - `host_scaled`: every host cost h multiplied by one stated factor k, for
    a sensitivity sweep over k. k is swept, never solved for.
  - `reprice_attention`: the device time at attention positions multiplied by
    a stated ratio (FA3 over `sdpa_flash`, both from the pairing's own
    dense-selection microbenchmark, untimed workload).
  - `median_interval`: a percentile bootstrap interval on one cell's median
    over its rounds, at the lock's bootstrap settings. Rounds within one
    cell are not independent replications of the study.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Mapping, Optional, Sequence

import numpy as np

from attnbench.predict.recursion import PricedSequence, recurse

LABEL = "EXPLORATORY / POST HOC"

# name -> what is removed from the committed recursion.
VARIANTS = {
    "recursion": "nothing (the committed model)",
    "no_queue": "the queue term (Q = None)",
    "no_fixed_sync": "every synchronisation's own cost s_0, the closing one's too (set to 0)",
    "no_wait": "every arm synchronisation's wait (the closing synchronise still waits)",
    "no_sync": "s_0 set to 0 and the arm's waits removed",
    "host_only": "every device and link time (set to 0): the host path alone",
    "device_only": "every host, s_0, queue and wait term: the device chain alone",
}


def _zero(xs) -> tuple:
    return tuple(0.0 for _ in xs)


def knockouts(priced: PricedSequence, queue_depth: Optional[int]) -> dict:
    """Seconds predicted by each variant in `VARIANTS`."""
    no_s0 = dataclasses.replace(priced, sync_cost=_zero(priced.sync_cost))
    no_dev = dataclasses.replace(priced, device=_zero(priced.device), link=_zero(priced.link))
    return {
        "recursion": recurse(priced, queue_depth).end_to_end,
        "no_queue": recurse(priced, None).end_to_end,
        "no_fixed_sync": recurse(no_s0, queue_depth).end_to_end,
        "no_wait": recurse(priced, queue_depth, sync_wait=False).end_to_end,
        "no_sync": recurse(no_s0, queue_depth, sync_wait=False).end_to_end,
        "host_only": recurse(no_dev, queue_depth).end_to_end,
        "device_only": float(sum(priced.device) + sum(priced.link)),
    }


def rel(predicted: float, measured: float) -> float:
    """Signed relative error, predicted / measured - 1."""
    if measured <= 0:
        raise ValueError("measured time must be positive")
    return predicted / measured - 1.0


def summarise(rels: Sequence[float]) -> dict:
    """n, median signed, median absolute and largest absolute relative error."""
    xs = list(rels)
    if not xs:
        return {"n": 0, "median_signed": None, "median_abs": None, "max_abs": None}
    return {"n": len(xs), "median_signed": float(statistics.median(xs)),
            "median_abs": float(statistics.median(abs(x) for x in xs)),
            "max_abs": float(max(abs(x) for x in xs))}


def arm_class(priced: PricedSequence, arm: str, dense_arm: str = "dense") -> str:
    """"dense", "synchronising" (any synchronisation besides the closing
    one) or "sync_free"."""
    if arm == dense_arm:
        return "dense"
    return "synchronising" if sum(priced.is_sync) > 1 else "sync_free"


def miss_split(priced: PricedSequence, queue_depth: Optional[int], measured: float,
               device_busy: float) -> dict:
    """The committed prediction's four terms and its error split exactly:
    error = device_part + idle_part, where device_part = priced chain - busy
    and idle_part = predicted idle - measured idle."""
    if measured <= 0 or device_busy < 0:
        raise ValueError("measured must be positive and device_busy non-negative")
    tl = recurse(priced, queue_depth)
    terms = {"host": float(sum(priced.host)), "fixed_sync": float(sum(priced.sync_cost)),
             "queue": float(sum(tl.queue_stall)), "sync_wait": float(sum(tl.sync_wait))}
    pred = tl.end_to_end
    if abs(sum(terms.values()) - pred) > 1e-9 * max(1.0, pred):
        raise ValueError("the four terms do not add to the prediction; the sequence "
                         "must end with its closing synchronise")
    chain = float(sum(priced.device) + sum(priced.link))
    out = {**terms, "predicted": pred, "priced_device_chain": chain,
           "predicted_idle": pred - chain, "measured": measured,
           "device_busy": device_busy, "measured_idle": measured - device_busy,
           "error_s": pred - measured,
           "device_part_s": chain - device_busy,
           "idle_part_s": (pred - chain) - (measured - device_busy)}
    out["rel"] = rel(pred, measured)
    return out


def host_scaled(priced: PricedSequence, queue_depth: Optional[int], k: float) -> float:
    """The recursion with every host cost h multiplied by k (s_0 unchanged)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    return recurse(dataclasses.replace(priced, host=tuple(h * k for h in priced.host)),
                   queue_depth).end_to_end


def reprice_attention(priced: PricedSequence, families: Sequence[str], ratio: float
                      ) -> tuple[PricedSequence, int]:
    """`priced` with the device time at each attention position multiplied by
    `ratio`, and the number of positions repriced. `families` are the
    positions' operation families (`breakdown.component_of` names them);
    only their names are used, never a traced duration."""
    from attnbench.exploratory.breakdown import component_of
    if len(families) != len(priced):
        raise ValueError(f"{len(families)} families for {len(priced)} positions")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    dev, n = list(priced.device), 0
    for i, f in enumerate(families):
        if component_of(f) == "attention":
            dev[i] *= ratio
            n += 1
    return dataclasses.replace(priced, device=tuple(dev)), n


def median_interval(samples: Sequence[float], *, n_boot: int, seed: int, level: float
                    ) -> tuple[float, float, float]:
    """(median, lo, hi): a percentile bootstrap interval on the median of one
    cell's rounds."""
    x = np.asarray(samples, float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("need at least two rounds")
    if not 0 < level < 1 or n_boot < 1:
        raise ValueError("level in (0, 1), n_boot >= 1")
    rng = np.random.default_rng(seed)
    reps = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    a = (1 - level) / 2
    lo, hi = (float(v) for v in np.quantile(reps, [a, 1 - a]))
    return float(np.median(x)), lo, hi


def rel_interval(predicted: float, lo: float, hi: float) -> tuple[float, float]:
    """The signed relative error's interval when the measured median lies in
    [lo, hi]: a larger measurement gives a smaller error."""
    if lo <= 0 or hi < lo:
        raise ValueError("need 0 < lo <= hi")
    return rel(predicted, hi), rel(predicted, lo)


def against_tolerance(lo: float, hi: float, tol: float) -> str:
    """Where a signed relative-error interval sits against |error| <= tol:
    "within", "outside" or "straddles"."""
    if -tol <= lo and hi <= tol:
        return "within"
    if hi < -tol or lo > tol:
        return "outside"
    return "straddles"


def count_by(rows: Sequence[Mapping], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items()))
