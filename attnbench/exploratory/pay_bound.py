"""EXPLORATORY / POST HOC: an exploratory screen, not a theorem, for when a
sparse arm's prefill cannot beat dense (the researcher's paper-quality wave of
28 Nov 2026, item 11). Nothing is scored; no parameter is fitted on P1-P4.

**The condition.** On one stream, a prefill cannot end before its device
work is done, and the device cannot start work the host has not issued:

  - the device-busy time B (the union of the traced device activities) is
    a floor on the end-to-end time;
  - every synchronisation inside the step drains the device, and the next
    operation cannot start on the device until the host has issued it after
    the synchronisation returned, which takes at least one launch floor f
    (the pairing's null kernel, bare-timed). So each in-step synchronisation
    adds at least f of device idle time: T >= B + n_sync * f;
  - the host issues N operations at no less than f each: T >= N * f.

So T_sparse >= LB = max(B + n_sync * f, N * f), and the sparse arm can pay
only if LB < T_dense. Every input is a measurement (B from the ground-truth
traces, f from the host-components step) or a count (N, n_sync from the
priced sequence); nothing is fitted.

**Two readings of B.** `traced`: the sparse arm's own ground-truth trace
(after the fact). `priced`: the committed device chain, the sum of the
priced device and copy times (kernels measured on the pairing before the
workload is timed, T2.18; one stream, so kernels do not overlap). The second
is usable before the workload is timed.

**Caveat on B.** B is read from a profiled run. If profiling stretches device
kernels (P2 at 1951 tokens: about 7 us a kernel, 22 Nov 2026), B overstates
the untraced busy time and the condition can wrongly say "cannot pay"; the
check reports any measured win the condition rules out.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from attnbench.predict.recursion import OpKind, PricedSequence

LABEL = "EXPLORATORY / POST HOC"


def counts(priced: PricedSequence) -> dict:
    """Issued operations (kernels and copies) and in-step synchronisations
    (every synchronisation but the timer's closing one, the last position)."""
    n_ops = sum(k in (OpKind.KERNEL, OpKind.COPY) for k in priced.kinds)
    syncs = list(priced.is_sync)
    closing = 1 if syncs and syncs[-1] else 0
    return {"ops": n_ops, "in_step_syncs": sum(syncs) - closing}


def lower_bound(busy_s: float, floor_s: float, ops: int, in_step_syncs: int) -> dict:
    """The end-to-end floor and which term sets it."""
    if busy_s < 0 or floor_s < 0 or ops < 0 or in_step_syncs < 0:
        raise ValueError("inputs must be non-negative")
    dev, host = busy_s + in_step_syncs * floor_s, ops * floor_s
    return {"lb_s": max(dev, host), "device_term_s": dev, "host_term_s": host,
            "binding": "device" if dev >= host else "host"}


def verdict(lb_s: float, dense_s: float) -> str:
    return "may pay" if lb_s < dense_s else "cannot pay"


def check(rows: Sequence[Mapping], key: str = "lb_s") -> dict:
    """Over sparse cells with `key` (the floor), `sparse_s` and `dense_s`: is
    the condition ever violated (a measured win it rules out), how many
    measured losses does it rule out, and is the floor ever above the
    measured sparse time (then it is not a floor there)?"""
    wins = [r for r in rows if r["sparse_s"] < r["dense_s"]]
    losses = [r for r in rows if r["sparse_s"] >= r["dense_s"]]
    violated = [r for r in wins if verdict(r[key], r["dense_s"]) == "cannot pay"]
    ruled_out = [r for r in losses if verdict(r[key], r["dense_s"]) == "cannot pay"]
    slack = sorted(r["sparse_s"] / r[key] for r in rows if r[key] > 0)
    return {"cells": len(rows), "measured_wins": len(wins), "measured_losses": len(losses),
            "violations": len(violated), "losses_ruled_out": len(ruled_out),
            "separates_any": len(ruled_out) > 0,
            "floor_above_measured_sparse": sum(r[key] > r["sparse_s"] for r in rows),
            "slack_median": slack[len(slack) // 2] if slack else None,
            "slack_min": slack[0] if slack else None}
