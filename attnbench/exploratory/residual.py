"""EXPLORATORY: what is left of a cell's recursion error once every device
time is the traced one (the adviser's task of 22 Nov 2026, item 2).

`breakdown` replaces each priced device and link time with the traced one.
What error remains is in the terms the traces cannot give untraced. This
module splits that remainder where the data allow, and no further.

**The prediction side is exact.** The recursion ends on the timer's closing
synchronise, so D_N = H_N, and H_N is the sum of four terms
(`recursion.recurse`): the host costs h, the synchronisations' own costs
s_0, the queue stalls and the synchronisation waits. `terms` returns them,
and they add up to the prediction.

**The measured side gives one split only.** The ground truth is untraced
and end to end. With the traced device durations taken as the untraced ones,
the device is busy for `busy` seconds (the union of its activities in the
prefill region), and idle for the rest. The recursion's idle time is its
prediction minus its device chain (Σ d + l). So the remaining error is the
predicted idle time minus the measured idle time: time the model says the
device waits for the host, the queue or a synchronisation, and it did not.
How the measured idle time divides among host work, queue and
synchronisation needs an untraced host timeline, which no session records.
The traced host timeline is inflated by the profiler (a 1951-token dense
cell's traced region is about 3.8 times its untraced time on P4), so it is
not used to split the measured side.

**Counterfactuals** say how far each predicted term could move the error:
the queue term removed (Q = None), every s_0 set to zero, every wait removed
(`sync_wait=False`), and the one factor on every h that would make the
prediction equal the measurement (`host_scale_to_match`). They are
sensitivities, not attributions: one term removed changes the others.

**Copies.** In P2's and P4's priced sequences every host-device copy is a
synchronising copy to the host, priced as s_0(m, payload) (Section 5.1), so
it is in the synchronisation term; no non-synchronising copy (link time l)
is priced or traced in any cell. `terms` reports the synchronisation term
with the payload bytes it carries.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence

from attnbench.predict.recursion import PricedSequence, recurse

TERMS = ("host", "sync_cost", "queue_stall", "sync_wait")


def terms(priced: PricedSequence, queue_depth: Optional[int]) -> dict:
    """The prediction as host + sync_cost + queue_stall + sync_wait, plus
    the device chain and the predicted idle time. Raises if the four terms
    do not add up to the prediction (the sequence must end with its closing
    synchronise)."""
    tl = recurse(priced, queue_depth)
    out = {"host": float(sum(priced.host)),
           "sync_cost": float(sum(priced.sync_cost)),
           "queue_stall": float(sum(tl.queue_stall)),
           "sync_wait": float(sum(tl.sync_wait))}
    total = sum(out.values())
    if abs(total - tl.end_to_end) > 1e-9 * max(1.0, tl.end_to_end):
        raise ValueError(f"the terms add to {total}, the prediction is {tl.end_to_end}; "
                         f"the sequence does not end with its closing synchronise")
    chain = float(sum(priced.device) + sum(priced.link))
    out.update(predicted=tl.end_to_end, device_chain=chain,
               predicted_idle=tl.end_to_end - chain,
               n_sync=int(sum(priced.is_sync)),
               sync_payload_bytes=int(sum(b for b, s in zip(priced.payload_bytes,
                                                            priced.is_sync) if s)),
               link=float(sum(priced.link)))
    return out


def busy(intervals: Sequence[tuple[float, float]]) -> float:
    """The length of the union of (start, end) intervals."""
    tot, end = 0.0, None
    start = None
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError(f"interval ends before it starts: {(a, b)}")
        if end is None or a > end:
            if end is not None:
                tot += end - start
            start, end = a, b
        elif b > end:
            end = b
    if end is not None:
        tot += end - start
    return tot


def host_scale_to_match(priced: PricedSequence, queue_depth: Optional[int],
                        measured: float, *, hi: float = 4.0, iters: int = 60
                        ) -> Optional[float]:
    """The factor k on every host cost h at which the recursion equals
    `measured`, by bisection on [0, hi] (the prediction does not decrease in
    k). None when no k in range reaches it (the device chain or the
    synchronisations alone already exceed the measurement, or the host
    would need more than `hi` times its price)."""
    def at(k: float) -> float:
        return recurse(dataclasses.replace(priced, host=tuple(h * k for h in priced.host)),
                       queue_depth).end_to_end
    lo_v, hi_v = at(0.0), at(hi)
    if not lo_v <= measured <= hi_v:
        return None
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if at(mid) < measured:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def counterfactuals(priced: PricedSequence, queue_depth: Optional[int],
                    measured: float) -> dict:
    """Relative error (prediction / measured - 1) as predicted and with one
    term removed, and the host factor that would match."""
    rel = lambda t: t / measured - 1.0
    zero_s = dataclasses.replace(priced, sync_cost=tuple(0.0 for _ in priced.sync_cost))
    return {"as_predicted": rel(recurse(priced, queue_depth).end_to_end),
            "no_queue_term": rel(recurse(priced, None).end_to_end),
            "sync_cost_zero": rel(recurse(zero_s, queue_depth).end_to_end),
            "no_sync_wait": rel(recurse(priced, queue_depth, sync_wait=False).end_to_end),
            "host_scale_to_match": host_scale_to_match(priced, queue_depth, measured)}


def split_cell(priced: PricedSequence, queue_depth: Optional[int], measured: float,
               device_busy: float) -> dict:
    """One cell, `priced` already carrying the traced device times. The
    error in seconds is the predicted idle time minus the measured idle
    time, up to the gap between the device chain and the traced busy time
    (`chain_minus_busy`, nonzero where activities overlap or a record is
    missing)."""
    if measured <= 0:
        raise ValueError("measured time must be positive")
    t = terms(priced, queue_depth)
    measured_idle = measured - device_busy
    return {**t, "measured": measured, "device_busy": device_busy,
            "measured_idle": measured_idle,
            "error_s": t["predicted"] - measured,
            "idle_error_s": t["predicted_idle"] - measured_idle,
            "chain_minus_busy": t["device_chain"] - device_busy,
            # The host issues every operation before the timer stops, so its
            # real host time is at most `measured`: above 1, this is a lower
            # bound on how far the priced host term overstates it.
            "host_over_measured": t["host"] / measured,
            "counterfactual_rel": counterfactuals(priced, queue_depth, measured)}


def per_kernel_excess(priced: PricedSequence, traced: Sequence[Optional[float]],
                      families: Sequence[str],
                      components: Sequence[str] = ("gemm", "other_device")) -> dict:
    """Traced minus priced device seconds per measured position, pooled over
    `components` (the trace inflation; attention is left out because P4's
    dense ground truth ran another kernel than the one priced). Returns
    {"positions": n, "excess_s_per_kernel": x or None}."""
    from attnbench.exploratory.breakdown import component_of
    n, ex = 0, 0.0
    for i, (t, f) in enumerate(zip(traced, families)):
        if t is None or component_of(f) not in components:
            continue
        n += 1
        ex += t - priced.device[i]
    return {"positions": n, "excess_s_per_kernel": ex / n if n else None}


def share_of_error(cells: Sequence[Mapping]) -> dict:
    """Over cells whose error is positive: the summed error in seconds, the
    summed predicted terms, and the error as a share of the summed predicted
    host term."""
    pos = [c for c in cells if c["error_s"] > 0]
    out = {"n_overpredicted": len(pos), "n": len(cells),
           "error_s": sum(c["error_s"] for c in pos)}
    for k in TERMS:
        out[k] = sum(c[k] for c in pos)
    out["error_over_host"] = out["error_s"] / out["host"] if out["host"] else None
    return out
