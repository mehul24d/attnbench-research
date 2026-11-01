"""The injector of Gate A's delay-injection slope check (T2.25).

`attnbench.predict.delay_injection` plans the variants: for one mechanism
and one placement it lists, per count c, the sites in the dense arm's
recursion sequence after which an extra synchronisation goes, and predicts
each variant. This module makes those variants happen on a real prefill: it
runs the arm's own prefill and, right after the launching call that issued
each planned site, performs the chosen mechanism with the chosen payload.
The ground-truth timer (T2.13) times the callable it returns; nothing here
reads a clock.

**From a plan site to a point in the prefill.** A site is an index into the
recursion's base sequence (`op_sequence.to_operations` then
`recursion.close_prefill`), or `START`, before the first operation.
`to_operations` drops the host-compute markers, so site k is the k-th
operation of the recorded `OpSequence` that is not a marker. At run time the
injector sees launching calls, not operations: one aten call can issue
several operations (a sort, a GEMM with a split-K reduce). The mapping from
calls to positions is the one `host_profile.assign` uses: maximal runs of
issued operations with one aten operator in one layer, adjacent equal runs
merged, each run met by c consecutive matching calls that divide its r
positions evenly. It is found once, by an untimed **calibration** pass of
the prefill with nothing injected, and every injected pass is then checked
call by call against it. A pass that takes another path is refused.

What can be injected where:

  - after a site that is the last operation its call issues: the injection
    runs as soon as that call returns, so the device holds that operation
    and everything before it;
  - at `START`: before the prefill's first call. The timer synchronises
    before it starts the clock, so the device is idle there;
  - not inside a call (a site that is not the last operation of its call),
    and not after a synchronisation or anything else that no launching
    call issues. These are refused, not moved:
    a moved site would be a different variant from the one predicted.

**Mechanisms.** Every mechanism the recursion prices
(`recursion.PRICED_MECHANISMS`), performed the way T2.5 measures s_fixed:
an explicit or event synchronise, `.item()` and `bool()` of a one-element
device tensor, and a blocking copy to or from pinned or pageable host
memory. Buffers are allocated when the mechanism is built, before any timed
pass, and reused, so no injection allocates. `.item()`, `bool()` and the
two synchronises carry no payload (Section 5.1); a copy carries
`payload_bytes`, and a copy planned at payload 0 moves
`link.SMALLEST_PAYLOAD_BYTES`, the size s_fixed is measured at, since s_0 at
payload 0 is s_fixed.

**Refused.** A `loaded` injection: the injector adds no CPU load, and the
dense arm has no host-side estimator work for a copy to sit alongside. A
sequence with an entry point that is not an aten call, or a CUDA-graph
launch: the injector sees launching calls at the aten dispatcher only. Gate
A's dense arm has neither.

**The observer is part of every variant.** Seeing calls means a Python
dispatch mode is active through the whole prefill, at c = 0 too, and it
adds host time at every aten call. The slopes are read from differences
over c, but the mode can also turn a device-bound stretch host-bound, which
moves the measured wait at a busy site. So (researcher's decision, relayed
28 Oct 2026) the prediction's host costs come from T2.2's null pass run
under the same observer: `observed` wraps the prefill for
`host_profile.profile` and checks the calls it sees against the
calibration. The observer's own effect is reported from an observed and an
unobserved c = 0 pass (`observer_pair`, `observer_effect`).

Sites the injector refuses are still refused. That `plan_variants` lists only
sites the injector accepts is added in `delay_injection.py` at integration.

The torch parts (`dispatch_observer`, `cuda_mechanism`) import torch when
called, so the planning and checking here run without it.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..predict import recursion
from ..predict.delay_injection import BUSY, IDLE, START, Variant, inject
from .link import SMALLEST_PAYLOAD_BYTES
from . import op_sequence
from .op_sequence import OpSequence, SeqKind

SCHEMA = "attnbench.delay_injection_schedule/1"

# Mechanisms that move no payload by definition (Section 5.1).
NO_PAYLOAD = ("synchronize", "event_synchronize", "item", "bool")
COPY_TO_HOST = {"copy_to_host_pinned": "pinned", "copy_to_host_pageable": "pageable"}
COPY_FROM_HOST = {"copy_from_host_pinned": "pinned",
                  "copy_from_host_pageable": "pageable"}


class InjectionError(RuntimeError):
    """A plan the injector cannot carry out as predicted, or a pass that did
    not meet the plan."""


@dataclass(frozen=True)
class Call:
    """One launching aten call as the observer saw it."""

    name: str
    layer: Optional[int]


# --- the sequence side ---------------------------------------------------------

def recursion_positions(seq: OpSequence) -> tuple[int, ...]:
    """Sequence position of each operation `to_operations` emits, in order:
    every operation except the host-compute markers."""
    return tuple(i for i, op in enumerate(seq.ops) if op.kind is not SeqKind.HOST_COMPUTE)


def _is_issue(op) -> bool:
    return op.is_device or op.kind is SeqKind.COPY


def _check_issuable(seq: OpSequence) -> None:
    for i, op in enumerate(seq.ops):
        if op.kind is SeqKind.GRAPH:
            raise InjectionError(
                f"operation {i} is a CUDA-graph launch; the injector sees aten "
                f"calls only (a compiled arm is out of its scope)")
        if _is_issue(op) and op.aten_op is None:
            raise InjectionError(
                f"operation {i} is issued from an entry point that is not an aten "
                f"call; the injector cannot see it at the dispatcher")


def launching_names(seq: OpSequence) -> frozenset:
    """The aten operators the recorded sequence issues work under."""
    _check_issuable(seq)
    return frozenset(op.aten_op for op in seq.ops if _is_issue(op))


@dataclass(frozen=True)
class _Run:
    name: str
    layer: Optional[int]
    positions: tuple[int, ...]


def _runs(seq: OpSequence) -> list[_Run]:
    """`host_profile.runs_of`, then `_merge_equal_neighbours` (that module
    imports torch, so the rule is restated here; a torch test checks the
    two agree)."""
    runs: list[_Run] = []
    cur, pos = None, []
    for i, op in enumerate(seq.ops):
        if not _is_issue(op):
            if cur is not None:
                runs.append(_Run(cur[0], cur[1], tuple(pos)))
                cur, pos = None, []
            continue
        key = (op.aten_op, op.layer)
        if key != cur:
            if cur is not None:
                runs.append(_Run(cur[0], cur[1], tuple(pos)))
            cur, pos = key, []
        pos.append(i)
    if cur is not None:
        runs.append(_Run(cur[0], cur[1], tuple(pos)))
    merged: list[_Run] = []
    for r in runs:
        if merged and (merged[-1].name, merged[-1].layer) == (r.name, r.layer):
            merged[-1] = _Run(r.name, r.layer, merged[-1].positions + r.positions)
        else:
            merged.append(r)
    return merged


def call_positions(seq: OpSequence, calls: Sequence[Call]) -> tuple[tuple[int, ...], ...]:
    """For each observed launching call, the sequence positions it issued.
    Raises `InjectionError` where the calls do not meet the sequence."""
    _check_issuable(seq)
    calls = list(calls)
    out: list[tuple[int, ...]] = []
    j = 0
    for run in _runs(seq):
        c = 0
        while j + c < len(calls) and (calls[j + c].name, calls[j + c].layer) == \
                (run.name, run.layer):
            c += 1
        r = len(run.positions)
        where = f"{run.name} in layer {run.layer}"
        if c == 0:
            nxt = calls[j] if j < len(calls) else None
            raise InjectionError(
                f"the recording issues {r} operation(s) under {where} and the "
                f"pass made no launching call there (next seen: "
                f"{nxt.name + ' in layer ' + str(nxt.layer) if nxt else 'nothing'})")
        if r % c:
            raise InjectionError(
                f"the recording issues {r} operation(s) under {where} and the "
                f"pass made {c} launching call(s); {r} is not a multiple of {c}")
        k = r // c
        out.extend(run.positions[n * k:(n + 1) * k] for n in range(c))
        j += c
    if j != len(calls):
        raise InjectionError(
            f"the pass made {len(calls) - j} launching call(s) after the recorded "
            f"sequence ended (first: {calls[j].name} in layer {calls[j].layer})")
    return tuple(out)


# --- the schedule ----------------------------------------------------------------

@dataclass(frozen=True)
class Schedule:
    """One variant, as the injector carries it out."""

    mechanism: str
    placement: str
    count: int
    sites: tuple[int, ...]               # the plan's, recursion indices
    payload_bytes: int                   # as planned (and predicted)
    at_start: int                        # injections before the first call
    after_call: tuple[tuple[int, int], ...]   # (call index, injections), ascending
    calls: tuple[Call, ...]              # the calibration pass, call by call
    sequence_sha256: str

    @property
    def total(self) -> int:
        return self.at_start + sum(n for _, n in self.after_call)


def _check_payload(mechanism: str, payload_bytes: int) -> None:
    if mechanism not in recursion.PRICED_MECHANISMS:
        raise InjectionError(
            f"{mechanism!r} is not a mechanism the recursion prices "
            f"({', '.join(recursion.PRICED_MECHANISMS)})")
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) \
            or payload_bytes < 0:
        raise InjectionError(f"payload_bytes is an int >= 0, got {payload_bytes!r}")
    if mechanism in NO_PAYLOAD and payload_bytes:
        raise InjectionError(
            f"{mechanism} carries no payload (Section 5.1); got {payload_bytes} bytes")


def _base_matches(base: Sequence[recursion.Operation], seq: OpSequence) -> tuple[int, ...]:
    """The base the plan was made on must be this sequence, operation for
    operation, followed by its closing synchronise."""
    positions = recursion_positions(seq)
    if len(base) != len(positions) + 1 or not recursion.is_closing_sync(base[-1]):
        raise InjectionError(
            f"the plan's base has {len(base)} operations; this sequence gives "
            f"{len(positions)} plus the closing synchronise")
    for k, p in enumerate(positions):
        sop, rop = seq.ops[p], base[k]
        want = (recursion.OpKind.SYNC if sop.kind is SeqKind.SYNC else
                recursion.OpKind.COPY if sop.kind is SeqKind.COPY else
                recursion.OpKind.KERNEL)
        if rop.kind is not want or (want is recursion.OpKind.SYNC
                                    and rop.mechanism != sop.mechanism):
            raise InjectionError(
                f"base operation {k} is a {rop.kind.value}"
                f"{' ' + str(rop.mechanism) if rop.mechanism else ''}; sequence "
                f"position {p} is a {sop.kind.value}"
                f"{' ' + str(sop.mechanism) if sop.mechanism else ''}")
    return positions


def injectable_sites(base: Sequence[recursion.Operation], seq: OpSequence,
                     calls: Sequence[Call]) -> tuple[int, ...]:
    """The sites (indices into `base`) the injector can take: those that end a
    launching call. A site inside a call's operations, or on an operation no
    launching call issues, is refused by `schedule_for`; passing this to
    `plan_variants(sites_allowed=...)` keeps the plan to sites it accepts."""
    positions = _base_matches(base, seq)
    last = {pos[-1] for pos in call_positions(seq, calls)}
    return tuple(k for k, p in enumerate(positions) if p in last)


def schedule_for(variant: Variant, base: Sequence[recursion.Operation],
                 seq: OpSequence, calls: Sequence[Call], *,
                 payload_bytes: int = 0, loaded: bool = False) -> Schedule:
    """The schedule for one planned variant. `base` is what the variant was
    planned on, `seq` the recorded sequence it came from, `calls` the
    calibration pass, and `payload_bytes` and `loaded` what `plan_variants`
    was given."""
    if loaded:
        raise InjectionError(
            "a loaded injection needs host-side work alongside it; the injector "
            "adds none (module docstring)")
    if variant.placement not in (BUSY, IDLE):
        raise InjectionError(f"unknown placement {variant.placement!r}")
    _check_payload(variant.mechanism, payload_bytes)
    if len(variant.sites) != variant.count:
        raise InjectionError(
            f"the variant lists {len(variant.sites)} sites for count {variant.count}")
    positions = _base_matches(base, seq)
    _, planned = inject(base, variant.sites, variant.mechanism,
                        payload_bytes=payload_bytes, loaded=loaded)
    if planned != variant.positions:
        raise InjectionError(
            f"injecting at {variant.sites} into this base puts the synchronisations "
            f"at {planned}; the variant has them at {variant.positions}, so it was "
            f"planned on another base")
    per_call = call_positions(seq, calls)
    last_of = {pos[-1]: j for j, pos in enumerate(per_call)}
    inside = {p: j for j, pos in enumerate(per_call) for p in pos[:-1]}
    at_start = 0
    after: dict[int, int] = {}
    for s in variant.sites:
        if s == START:
            at_start += 1
            continue
        p = positions[s]
        if p in last_of:
            after[last_of[p]] = after.get(last_of[p], 0) + 1
        elif p in inside:
            j = inside[p]
            raise InjectionError(
                f"site {s} (sequence position {p}) is inside call {j} "
                f"({calls[j].name}), which issues positions {list(per_call[j])}; "
                f"an injection cannot go between them")
        else:
            raise InjectionError(
                f"site {s} (sequence position {p}) is a {seq.ops[p].kind.value}, "
                f"which no launching call issues; the injector goes after a "
                f"launching call or at START only")
    return Schedule(
        mechanism=variant.mechanism, placement=variant.placement,
        count=variant.count, sites=tuple(variant.sites),
        payload_bytes=payload_bytes, at_start=at_start,
        after_call=tuple(sorted(after.items())), calls=tuple(calls),
        sequence_sha256=op_sequence.record_sha256(op_sequence.to_record(seq)))


def schedules(variants: Sequence[Variant], base, seq: OpSequence,
              calls: Sequence[Call], *, payload_bytes: int = 0,
              loaded: bool = False) -> tuple[Schedule, ...]:
    """`schedule_for` over every variant of one `plan_variants` call."""
    return tuple(schedule_for(v, base, seq, calls, payload_bytes=payload_bytes,
                              loaded=loaded) for v in variants)


def to_record(schedule: Schedule) -> dict:
    """What a measured variant's row carries about its injection."""
    return {"schema": SCHEMA, "mechanism": schedule.mechanism,
            "placement": schedule.placement, "count": schedule.count,
            "sites": list(schedule.sites), "payload_bytes": schedule.payload_bytes,
            "at_start": schedule.at_start,
            "after_call": [list(x) for x in schedule.after_call],
            "n_calls": len(schedule.calls),
            "sequence_sha256": schedule.sequence_sha256}


# --- running ---------------------------------------------------------------------

# An observer runs `run_prefill` and calls `on_call(name, layer)` after each
# launching call has returned. `dispatch_observer` is the real one.
Observer = Callable[[Callable[[], None], Callable[[str, Optional[int]], None]], None]


def calibrate(run_prefill: Callable[[], None], seq: OpSequence,
              observer: Observer) -> tuple[Call, ...]:
    """One untimed pass with nothing injected: the calls, checked against
    the sequence."""
    calls: list[Call] = []
    observer(run_prefill, lambda name, layer: calls.append(Call(name, layer)))
    call_positions(seq, calls)
    return tuple(calls)


class _Cursor:
    def __init__(self, schedule: Schedule, fire: Callable[[], object]):
        self.s, self.fire = schedule, fire
        self.after = dict(schedule.after_call)
        self.j = 0
        self.fired = 0

    def start(self) -> None:
        for _ in range(self.s.at_start):
            self.fire()
            self.fired += 1

    def on_call(self, name: str, layer: Optional[int]) -> None:
        calls = self.s.calls
        if self.j >= len(calls):
            raise InjectionError(
                f"call {self.j} ({name} in layer {layer}) is past the "
                f"{len(calls)} the calibration pass made")
        if (name, layer) != (calls[self.j].name, calls[self.j].layer):
            raise InjectionError(
                f"call {self.j} is {name} in layer {layer}; the calibration pass "
                f"made {calls[self.j].name} in layer {calls[self.j].layer} there")
        for _ in range(self.after.get(self.j, 0)):
            self.fire()
            self.fired += 1
        self.j += 1

    def finish(self) -> None:
        if self.j != len(self.s.calls):
            raise InjectionError(
                f"the pass made {self.j} launching calls; the calibration pass "
                f"made {len(self.s.calls)}")
        if self.fired != self.s.total:
            raise InjectionError(f"{self.fired} injections fired; {self.s.total} planned")


def injected_prefill(run_prefill: Callable[[], None], schedule: Schedule,
                     fire: Callable[[], object], observer: Observer) -> Callable[[], int]:
    """A callable that runs one prefill with the schedule's injections and
    returns how many fired. `fire` performs the mechanism once (from
    `cuda_mechanism`). The c = 0 variant runs under the same observer, so
    every variant carries its overhead alike."""
    if schedule.count != schedule.total:
        raise InjectionError(
            f"the schedule places {schedule.total} injections for count {schedule.count}")

    def run() -> int:
        cur = _Cursor(schedule, fire)
        cur.start()
        observer(run_prefill, cur.on_call)
        cur.finish()
        return cur.fired
    return run


def observed(run_prefill: Callable[[], None], observer: Observer,
             calls: Optional[Sequence[Call]] = None) -> Callable[[], None]:
    """`run_prefill` under the observer with nothing injected, for T2.2's
    null pass (`host_profile.profile(observed(...), ...)`). Since the ruling
    of 1 Nov 2026 after Shakedown 5 the prediction reads uninstrumented host
    costs (the bare timer's, `bare_timer`), not costs taken under the
    observer; this pass is the instrumented comparison, and the observer's
    own effect is reported (`observer_effect`), never priced. With `calls`
    (the calibration), each pass must make exactly those calls."""
    expected = None if calls is None else tuple(calls)

    def run() -> None:
        seen: list[Call] = []
        observer(run_prefill, lambda name, layer: seen.append(Call(name, layer)))
        if expected is not None and tuple(seen) != expected:
            n = next((i for i, (a, b) in enumerate(zip(seen, expected)) if a != b),
                     min(len(seen), len(expected)))
            raise InjectionError(
                f"the observed pass made {len(seen)} launching calls and first "
                f"differs from the calibration's {len(expected)} at call {n}")
    return run


def observer_pair(run_prefill: Callable[[], None], schedule: Schedule,
                  fire: Callable[[], object],
                  observer: Observer) -> tuple[Callable[[], int], Callable[[], None]]:
    """(observed, unobserved) c = 0 prefills, for the timer to interleave.
    The observed one is the c = 0 variant exactly as measured; the unobserved
    one is the arm's own prefill with no observer."""
    if schedule.count != 0:
        raise InjectionError(
            f"the observer's effect is read at c = 0; this schedule has count "
            f"{schedule.count}")
    return injected_prefill(run_prefill, schedule, fire, observer), run_prefill


@dataclass(frozen=True)
class ObserverEffect:
    observed_s: float          # the statistic of the observed c = 0 repeats
    unobserved_s: float
    effect_s: float            # observed minus unobserved
    ratio: float               # observed over unobserved
    repeats: tuple[int, int]


def observer_effect(observed_s: Sequence[float], unobserved_s: Sequence[float], *,
                    statistic=statistics.median) -> ObserverEffect:
    """The observer's own effect on the c = 0 prefill, from timed repeats in
    seconds. Reported, never subtracted from anything."""
    def stat(xs, which):
        xs = [float(x) for x in xs]
        if not xs or any(not math.isfinite(x) or x <= 0 for x in xs):
            raise InjectionError(
                f"{which} repeats must be positive, finite and non-empty")
        return float(statistic(xs)), len(xs)
    o, no = stat(observed_s, "observed")
    u, nu = stat(unobserved_s, "unobserved")
    return ObserverEffect(observed_s=o, unobserved_s=u, effect_s=o - u, ratio=o / u,
                          repeats=(no, nu))


# --- the torch side ----------------------------------------------------------------

def dispatch_observer(launching: frozenset, *, device_type: str = "cuda",
                      layers: Sequence = ()) -> Observer:
    """The real observer: a `TorchDispatchMode` that, after each aten call
    named in `launching` that touches a `device_type` tensor (the rule
    `host_profile.NullDeviceMode` uses), calls `on_call` with the call's name
    and the decoder layer it ran in (forward hooks on `layers`, as
    `host_profile` tracks them). Work done inside `on_call` runs with the
    mode off, so an injected mechanism is never itself observed."""
    import torch
    from torch.utils import _pytree as pytree
    from torch.utils._python_dispatch import TorchDispatchMode

    launching = frozenset(launching)

    class _Mode(TorchDispatchMode):
        def __init__(self, on_call, state):
            super().__init__()
            self.on_call, self.state = on_call, state

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            name = func._schema.name
            if name in launching and any(
                    isinstance(t, torch.Tensor) and t.device.type == device_type
                    for t in pytree.tree_leaves((args, kwargs))):
                self.on_call(name, self.state["layer"])
            return out

    def observe(run_prefill, on_call) -> None:
        state = {"layer": None}
        handles = []
        try:
            for i, m in enumerate(layers):
                handles.append(m.register_forward_pre_hook(
                    lambda mod, a, _i=i: state.__setitem__("layer", _i)))
                handles.append(m.register_forward_hook(
                    lambda mod, a, o: state.__setitem__("layer", None)))
            with _Mode(on_call, state):
                run_prefill()
        finally:
            for h in handles:
                h.remove()
    return observe


def payload_moved(mechanism: str, payload_bytes: int) -> int:
    """Bytes the performed mechanism moves (module docstring)."""
    _check_payload(mechanism, payload_bytes)
    if mechanism in NO_PAYLOAD:
        return 0
    return max(payload_bytes, SMALLEST_PAYLOAD_BYTES)


def cuda_mechanism(mechanism: str, payload_bytes: int = 0,
                   device: str = "cuda") -> Callable[[], object]:
    """A callable that performs `mechanism` once on `device`, every buffer
    allocated here and reused."""
    import torch
    n = payload_moved(mechanism, payload_bytes)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise InjectionError("the injector's mechanisms need a CUDA device")
    if mechanism == "synchronize":
        return lambda: torch.cuda.synchronize(dev)
    if mechanism == "event_synchronize":
        ev = torch.cuda.Event()

        def event_sync():
            ev.record()
            ev.synchronize()
        return event_sync
    if mechanism == "item":
        one = torch.zeros(1, dtype=torch.uint8, device=dev)
        return one.item
    if mechanism == "bool":
        flag = torch.zeros(1, dtype=torch.bool, device=dev)
        return lambda: bool(flag)
    gpu = torch.zeros(n, dtype=torch.uint8, device=dev)
    if mechanism in COPY_TO_HOST:
        host = torch.zeros(n, dtype=torch.uint8,
                           pin_memory=COPY_TO_HOST[mechanism] == "pinned")
        return lambda: host.copy_(gpu, non_blocking=False)
    host = torch.zeros(n, dtype=torch.uint8,
                       pin_memory=COPY_FROM_HOST[mechanism] == "pinned")
    return lambda: gpu.copy_(host, non_blocking=False)
