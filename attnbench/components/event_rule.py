"""Which price an event synchronise takes (adviser's ruling 4a of 9 Nov 2026,
seventh paste; leg 3 of Gate A v2 on T3.2c).

T2.5 prices `event_synchronize` as `Event.synchronize()` on an event recorded
once and already complete: a host-only check (1.16 us idle on T3.2c). Code
that records an event and then waits for it (`ev.record(); ev.synchronize()`,
as the delay injector does) enqueues a marker and waits for the device to
reach it: a round trip (8.59 us idle on T3.2c). Since 9 Nov 2026 T2.5 also
measures that, as `link.RECORD_THEN_WAIT`.

**The rule**, structural, read from the recorded trace's CUDA calls in the
prefill region in time order: an event synchronise takes the record-then-wait
price when a `cudaEventRecord` lies after the previous synchronising call
(a stream, device or event synchronise, an implicit one, or a blocking copy)
and before it; otherwise the event was recorded before the device last
drained, so it is complete, and takes the completed-event price. The
injector's synchronise is a record followed by a wait by construction
(`delay_injector.cuda_mechanism`), so its plan prices it record-then-wait
(`for_injection`).

`kinds` reads the rule from a trace; `apply` puts it into a priced sequence.
A sequence with no event synchronise (every P1 cell on T3.2c) is unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

from ..predict.recursion import PricedSequence, s0
from . import op_sequence as S
from .link import RECORD_THEN_WAIT

EVENT_SYNCHRONIZE = "event_synchronize"
RECORDED, COMPLETED = "record_then_wait", "completed_event"
_RECORDS = frozenset({"cudaEventRecord", "cuEventRecord", "cudaEventRecordWithFlags",
                      "cuEventRecordWithFlags"})
_DRAINS = frozenset({"sync", "event_sync", "implicit_sync", "memcpy_blocking"})


class EventRuleError(ValueError):
    pass


def kinds_from_calls(names: Sequence[str]) -> list[str]:
    """One kind per event synchronise in `names` (CUDA call names, in time order)."""
    out, recorded = [], False
    for n in names:
        if n in _RECORDS:
            recorded = True
            continue
        cls = S.api_class(n)
        if cls == "event_sync":
            out.append(RECORDED if recorded else COMPLETED)
        if cls in _DRAINS:
            recorded = False
    return out


def kinds(trace, *, region: str = S.REGION) -> list[str]:
    """`kinds_from_calls` over the CUDA calls inside the trace's one `region`."""
    events = S._events(S.load_trace(trace))
    reg = S._region(events, region, None)
    calls = sorted((e for e in events if e.cat in S._HOST_CATS
                    and reg.ts <= e.ts and e.end <= reg.end), key=lambda e: e.ts)
    return kinds_from_calls([e.name for e in calls])


def event_sync_positions(priced: PricedSequence) -> list[int]:
    return [i for i, (sync, m) in enumerate(zip(priced.is_sync, priced.mechanisms))
            if sync and m == EVENT_SYNCHRONIZE]


def apply(priced: PricedSequence, rule: Sequence[str],
          sync_costs: Mapping) -> PricedSequence:
    """`priced` with each event synchronise the rule names record-then-wait
    priced at `sync_costs[RECORD_THEN_WAIT]`; `rule` is `kinds` of the cell's
    recording, one per event synchronise in order."""
    pos = event_sync_positions(priced)
    if len(pos) != len(rule):
        raise EventRuleError(f"{len(pos)} event synchronises in the sequence and {len(rule)} "
                             f"in the rule")
    if not any(k == RECORDED for k in rule):
        return priced
    if RECORD_THEN_WAIT not in sync_costs:
        raise EventRuleError(f"the link table has no {RECORD_THEN_WAIT!r} price")
    cost = list(priced.sync_cost)
    for i, k in zip(pos, rule):
        if k == RECORDED:
            cost[i] = s0(sync_costs[RECORD_THEN_WAIT], priced.payload_bytes[i], False)
    return dataclasses.replace(priced, sync_cost=tuple(cost))


def for_injection(sync_costs: Mapping) -> dict:
    """The link table the leg-3 plan prices with: the injector records before
    each wait, so its `event_synchronize` takes the record-then-wait price."""
    out = dict(sync_costs)
    if RECORD_THEN_WAIT in out:
        out[EVENT_SYNCHRONIZE] = out[RECORD_THEN_WAIT]
    return out
