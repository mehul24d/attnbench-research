"""The critical-path recursion of plan Section 5.4 (T2.8).

One CUDA stream. Operation i has host cost h_i, device cost d_i and link cost
l_i. H_i is when the host has issued operation i; D_i is when the device
finishes it. With H_0 = D_0 = 0 and operations numbered from 1:

    H_i = H_(i-1) + h_i
    if Q launches are pending:  H_i = max(H_i, D_(i-Q))
    if i is a synchronisation:  H_i = max(H_i, D_(i-1)) + s_0(m_i, payload_i);  D_i = H_i
    otherwise:                  D_i = max(H_i, D_(i-1)) + l_i + d_i
    end-to-end prefill time = D_N

The queue line is applied as written, to every i > Q. The device runs in
order, so when fewer than Q launches are pending D_(i-Q) is already at or
before H_i and the max changes nothing; the "if" needs no separate test.

A synchronisation is costed once, by mechanism (Section 5.1):

    s_0(m, payload) = s_fixed(m) + payload / B

B is the copy bandwidth for that mechanism, the loaded figure where the
operation sequence puts the copy alongside host-side estimator work and the
unloaded one otherwise. Nothing here is fitted: every number comes in from a
microbenchmark or a replay, and the recursion only composes them. The wait
at a synchronisation is not an input. The recursion produces it, and
`Timeline.sync_wait` reports it.

What this module refuses, because each one would charge a physical event
twice or put the model outside its stated scope:

  - a synchronisation that carries device or link cost (l_i = d_i = 0);
  - a synchronisation that carries host cost: s_fixed(m) is already "the
    host cost of issuing and completing mechanism m", so h_i = 0 there and
    the recursion's H_(i-1) + h_i adds nothing a second time;
  - a composite mechanism (an `assert` on a GPU tensor, sizing from
    `nonzero()`), which must arrive already split into a kernel followed by
    a synchronisation;
  - a kernel with link cost, or a copy with device cost (only device costs
    transfer between hosts, H2, so a copy's time must sit in l alone);
  - a loaded-bandwidth request for a mechanism with no loaded figure (never
    a silent fall back to the unloaded one);
  - operations on more than one GPU stream.

Units are seconds and bytes per second. Host-side compute such as CPU mask
building has no operation of its own: it is part of c_host of the operation
it precedes (Section 5.1), so every operation is a kernel, a copy or a
synchronisation. With host cost refused at a synchronisation, host work
issued after the last launch and before a synchronisation has nowhere to
sit; it is charged to the next operation after the synchronisation.

**The closing synchronise.** The ground-truth timer (T2.13, adapting
`accuracy.phase_timing.time_repeated`) synchronises before reading the
clock, so the timed interval ends when the host returns from that
synchronise, not when the device finishes. A prefill sequence therefore
ends with one explicit synchronise, `closing_sync_op()`, appended by
`close_prefill`. D_N is then that operation's D_i = H_i. It applies only
while the timer reads a host clock; a timer on CUDA events would end at the
last device operation and the closing synchronise would not belong.

The interface for the baselines is `price`: it turns operations into the
per-operation h, l, d and s_0 arrays that the recursion reads, so the sum,
the fitted sum, the step form and the learned predictor are all built from
exactly the components the recursion used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence


class CostModelError(ValueError):
    """An operation sequence or cost table the recursion cannot price."""


class OpKind(str, Enum):
    KERNEL = "kernel"
    COPY = "copy"
    SYNC = "sync"


# The mechanisms Section 5.1 costs separately. These names are the keys of
# the s_fixed table that the link microbenchmark (T2.5) has to produce.
SYNC_MECHANISMS = (
    "synchronize",            # explicit torch.cuda.synchronize()
    "item",                   # .item() on a GPU tensor
    "bool",                   # bool() of a GPU tensor
    "copy_to_host_pinned",    # blocking copy into pinned host memory
    "copy_to_host_pageable",  # blocking copy into pageable host memory
)

# Composite mechanisms are split before they reach the recursion (Section
# 5.1). Named here so the refusal says what to do instead of "unknown".
COMPOSITE_MECHANISMS = {
    "assert": "an assert on a GPU tensor is a kernel (its device cost from "
              "replay) followed by a 'bool' synchronisation",
    "nonzero": "sizing from nonzero() is a kernel (its device cost from "
               "replay) followed by a synchronisation that reads the size",
}


@dataclass(frozen=True)
class Operation:
    """One operation of an arm's sequence (T2.3 records the order).

    `host` is h_i, `device` is d_i, `link` is l_i, all in seconds. A
    synchronisation names its `mechanism`, its `payload_bytes`, and whether
    the sequence puts it alongside host-side estimator work (`loaded`). Its
    s_0 is priced from the mechanism table, never stored here, so the
    queue, density and bandwidth sensitivities can re-price it.

    A copy may also state the bytes it moves in `payload_bytes`. That count
    does not price it (its l_i arrives already priced); it feeds the learned
    baseline's transfer demand (T2.10). A kernel moves no host-device bytes.

    `layer` is carried for the per-layer step form (T2.9) and `stream` for
    the one-stream check; neither changes the recursion.
    """

    kind: OpKind
    host: float
    device: float = 0.0
    link: float = 0.0
    mechanism: Optional[str] = None
    payload_bytes: int = 0
    loaded: bool = False
    layer: Optional[int] = None
    stream: Optional[int] = None
    label: str = ""


def kernel_op(host: float, device: float, **kw) -> Operation:
    return Operation(OpKind.KERNEL, host=host, device=device, **kw)


def copy_op(host: float, link: float, **kw) -> Operation:
    """A host-device copy the host does not wait for (c_link, Section 5.1)."""
    return Operation(OpKind.COPY, host=host, link=link, **kw)


def sync_op(mechanism: str, *, payload_bytes: int = 0, loaded: bool = False,
            **kw) -> Operation:
    """A synchronisation. Its host cost is s_fixed, so h_i is 0."""
    return Operation(OpKind.SYNC, host=0.0, mechanism=mechanism,
                     payload_bytes=payload_bytes, loaded=loaded, **kw)


CLOSING_SYNC_LABEL = "closing synchronize"


def closing_sync_op() -> Operation:
    """The explicit synchronise that ends the timed interval (module doc)."""
    return sync_op("synchronize", label=CLOSING_SYNC_LABEL)


def is_closing_sync(op: Operation) -> bool:
    return op.kind is OpKind.SYNC and op.label == CLOSING_SYNC_LABEL


def close_prefill(ops: Iterable[Operation]) -> list[Operation]:
    """The arm's operations followed by the closing synchronise."""
    ops = list(ops)
    if any(is_closing_sync(op) for op in ops):
        raise CostModelError(
            "this sequence already has a closing synchronise; a prefill has "
            "exactly one, as its last operation")
    return ops + [closing_sync_op()]


@dataclass(frozen=True)
class MechanismCost:
    """s_fixed(m) and the copy bandwidths for one mechanism, from T2.5.

    `s_fixed` is measured at the smallest payload with the device idle.
    `bandwidth` and `loaded_bandwidth` are bytes per second; either may be
    None for a mechanism that never moves a payload (an explicit
    synchronise), and a payload priced against a None bandwidth is refused.
    """

    s_fixed: float
    bandwidth: Optional[float] = None
    loaded_bandwidth: Optional[float] = None


SyncCosts = Mapping[str, MechanismCost]


def s0(cost: MechanismCost, payload_bytes: int, loaded: bool) -> float:
    """s_0(m, payload) = s_fixed(m) + payload / B (Section 5.1)."""
    _check_cost(cost.s_fixed, "s_fixed")
    if payload_bytes == 0:
        return float(cost.s_fixed)
    bw = cost.loaded_bandwidth if loaded else cost.bandwidth
    which = "loaded_bandwidth" if loaded else "bandwidth"
    if bw is None:
        raise CostModelError(
            f"a payload of {payload_bytes} bytes needs {which}, and this "
            f"mechanism has none; the unloaded figure is not substituted")
    if not (isinstance(bw, (int, float)) and math.isfinite(bw) and bw > 0):
        raise CostModelError(f"{which} must be a positive finite number, got {bw!r}")
    return float(cost.s_fixed) + payload_bytes / bw


@dataclass(frozen=True)
class PricedSequence:
    """Per-operation components exactly as the recursion reads them.

    s_0 is zero for every operation that is not a synchronisation, and h, l,
    d follow Section 5.1 (l = d = 0 at a synchronisation). The baselines of
    T2.9 and T2.10 read these arrays rather than re-pricing the operations.
    """

    host: tuple[float, ...]
    link: tuple[float, ...]
    device: tuple[float, ...]
    sync_cost: tuple[float, ...]
    is_sync: tuple[bool, ...]
    kinds: tuple[OpKind, ...]
    layers: tuple[Optional[int], ...]
    mechanisms: tuple[Optional[str], ...]
    payload_bytes: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.host)


@dataclass(frozen=True)
class Timeline:
    """The recursion's output. Index k is operation k + 1 of the plan.

    `queue_stall[k]` is how far the queue term moved H forward at operation
    k. `sync_wait[k]` is how long the host waited for the device at a
    synchronisation, max(H, D_(i-1)) - H before s_0 is added (zero
    elsewhere). Both are produced, never measured inputs.
    """

    host_issued: tuple[float, ...]
    device_done: tuple[float, ...]
    queue_stall: tuple[float, ...]
    sync_wait: tuple[float, ...]
    queue_depth: Optional[int]
    sync_dependency: bool = True

    @property
    def end_to_end(self) -> float:
        """D_N, the predicted end-to-end prefill time."""
        return self.device_done[-1]


def _check_cost(x, name: str, where: str = "") -> None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise CostModelError(f"{name}{where} must be a number, got {x!r}")
    if not math.isfinite(x) or x < 0:
        raise CostModelError(f"{name}{where} must be finite and >= 0, got {x!r}")


def _check_queue_depth(queue_depth) -> None:
    if queue_depth is None:
        return
    if isinstance(queue_depth, bool) or not isinstance(queue_depth, int):
        raise CostModelError(
            f"queue_depth is a count of launches, an int >= 1 or None for no "
            f"queue term; got {queue_depth!r}")
    if queue_depth < 1:
        raise CostModelError(
            f"queue_depth must be >= 1 (None means no queue term); got "
            f"{queue_depth}")


def _check_operation(op: Operation, i: int) -> None:
    where = f" (operation {i + 1}{', ' + op.label if op.label else ''})"
    if not isinstance(op.kind, OpKind):
        raise CostModelError(f"kind{where} must be an OpKind, got {op.kind!r}")
    for name in ("host", "device", "link"):
        _check_cost(getattr(op, name), name, where)
    if (isinstance(op.payload_bytes, bool)
            or not isinstance(op.payload_bytes, int)
            or op.payload_bytes < 0):
        raise CostModelError(
            f"payload_bytes{where} must be an int >= 0, got "
            f"{op.payload_bytes!r}")

    if op.kind is OpKind.SYNC:
        if op.device != 0 or op.link != 0:
            raise CostModelError(
                f"a synchronisation carries no device or link cost{where}: "
                f"got device={op.device}, link={op.link}. Split it into a "
                f"kernel or copy followed by the synchronisation")
        if op.host != 0:
            raise CostModelError(
                f"a synchronisation carries no host cost{where}: got "
                f"host={op.host}. s_fixed(m) already holds the host cost of "
                f"issuing and completing it, so charging h_i too would cost "
                f"it twice; host work before it belongs to the next "
                f"operation")
        if op.mechanism in COMPOSITE_MECHANISMS:
            raise CostModelError(
                f"{op.mechanism!r} is a composite mechanism{where}: "
                f"{COMPOSITE_MECHANISMS[op.mechanism]}")
        if op.mechanism not in SYNC_MECHANISMS:
            raise CostModelError(
                f"unknown synchronisation mechanism {op.mechanism!r}{where}; "
                f"expected one of {SYNC_MECHANISMS}")
        if op.mechanism == "synchronize" and op.payload_bytes:
            raise CostModelError(
                f"an explicit synchronise moves no payload{where}; a copy to "
                f"the host is its own mechanism")
        return

    if op.mechanism is not None or op.loaded:
        raise CostModelError(
            f"mechanism and loaded belong to synchronisations only{where}; a "
            f"{op.kind.value}'s link cost arrives already priced at the right "
            f"bandwidth")
    if op.kind is OpKind.KERNEL and op.payload_bytes:
        raise CostModelError(
            f"a kernel moves no host-device bytes{where}: got "
            f"payload_bytes={op.payload_bytes}")
    if op.kind is OpKind.KERNEL and op.link != 0:
        raise CostModelError(f"a kernel has no link cost{where}: got link={op.link}")
    if op.kind is OpKind.COPY and op.device != 0:
        raise CostModelError(
            f"a copy's time is link cost only{where}: got device={op.device}. "
            f"Device costs transfer between hosts (H2) and link costs do not, "
            f"so they cannot share an operation")


def price(ops: Iterable[Operation], sync_costs: SyncCosts) -> PricedSequence:
    """Check every operation and price each synchronisation's s_0."""
    ops = list(ops)
    if not ops:
        raise CostModelError("an empty operation sequence has no D_N")
    streams = {op.stream for op in ops if op.stream is not None}
    if len(streams) > 1:
        raise CostModelError(
            f"operations on {len(streams)} GPU streams {sorted(streams)}; the "
            f"recursion models one stream, and claims do not extend to arms "
            f"that spread a forward pass over several (Section 5.4)")

    closing = [i for i, op in enumerate(ops) if is_closing_sync(op)]
    if closing and closing != [len(ops) - 1]:
        raise CostModelError(
            f"a closing synchronise must be the last operation and appear "
            f"once; found at operations {[i + 1 for i in closing]} of "
            f"{len(ops)}")

    sync_cost = []
    for i, op in enumerate(ops):
        _check_operation(op, i)
        if op.kind is OpKind.SYNC:
            if op.mechanism not in sync_costs:
                raise CostModelError(
                    f"no s_fixed for mechanism {op.mechanism!r} (operation "
                    f"{i + 1}); the link microbenchmark must measure it")
            sync_cost.append(s0(sync_costs[op.mechanism], op.payload_bytes,
                                op.loaded))
        else:
            sync_cost.append(0.0)

    return PricedSequence(
        host=tuple(float(op.host) for op in ops),
        link=tuple(float(op.link) for op in ops),
        device=tuple(float(op.device) for op in ops),
        sync_cost=tuple(sync_cost),
        is_sync=tuple(op.kind is OpKind.SYNC for op in ops),
        kinds=tuple(op.kind for op in ops),
        layers=tuple(op.layer for op in ops),
        mechanisms=tuple(op.mechanism for op in ops),
        payload_bytes=tuple(op.payload_bytes for op in ops),
    )


def recurse(priced: PricedSequence, queue_depth: Optional[int], *,
            sync_wait: bool = True) -> Timeline:
    """Run the recursion over priced components.

    `queue_depth` is Q, measured per pairing (T2.6), or None for "no queue
    term" (the queue sensitivity, Section 5.4). T2.15 and T2.19 call this
    directly to re-run one priced sequence at other Q or resampled costs.

    `sync_wait=False` is the `ablations.no_sync_wait` model: a
    synchronisation still costs the host s_0 but no longer waits for the
    device. Its D_i is then max(D_(i-1), H_i), which never delays later
    device work, and the queue counts launches only. With the wait in place
    the two queue counts agree (a synchronisation drains the queue), so the
    full model keeps the plan's D_(i-Q) as written; without it they do not,
    and a synchronisation is not a launch.
    """
    _check_queue_depth(queue_depth)
    n = len(priced)
    if n == 0:
        raise CostModelError("an empty operation sequence has no D_N")
    h, l, d = priced.host, priced.link, priced.device
    s, is_sync = priced.sync_cost, priced.is_sync
    Q = queue_depth

    H = [0.0] * n
    D = [0.0] * n
    stall = [0.0] * n
    wait = [0.0] * n
    launches: list[float] = []   # D of kernels and copies, in order
    H_prev = 0.0
    D_prev = 0.0
    for k in range(n):
        t = H_prev + h[k]
        # Without the wait a synchronisation enqueues nothing, so the queue
        # never holds it back.
        if Q is not None and (sync_wait or not is_sync[k]):
            if sync_wait:
                # D_(i-Q) with i = k + 1 is D[k - Q]; it exists once k >= Q.
                bound = D[k - Q] if k >= Q else None
            else:
                bound = launches[-Q] if len(launches) >= Q else None
            if bound is not None and bound > t:
                stall[k] = bound - t
                t = bound
        if is_sync[k]:
            if sync_wait:
                start = D_prev if D_prev > t else t
                wait[k] = start - t
                t = start + s[k]
                D_k = t
            else:
                t = t + s[k]
                D_k = D_prev if D_prev > t else t
        else:
            D_k = (D_prev if D_prev > t else t) + l[k] + d[k]
            launches.append(D_k)
        H[k] = t
        D[k] = D_k
        H_prev = t
        D_prev = D_k

    return Timeline(host_issued=tuple(H), device_done=tuple(D),
                    queue_stall=tuple(stall), sync_wait=tuple(wait),
                    queue_depth=Q, sync_dependency=sync_wait)


def simulate(ops: Sequence[Operation], sync_costs: SyncCosts,
             queue_depth: Optional[int]) -> Timeline:
    """`price` then `recurse`: the whole recursion for one arm and cell."""
    return recurse(price(ops, sync_costs), queue_depth)


def predict_prefill(ops: Sequence[Operation], sync_costs: SyncCosts,
                    queue_depth: Optional[int]) -> float:
    """D_N in seconds."""
    return simulate(ops, sync_costs, queue_depth).end_to_end
