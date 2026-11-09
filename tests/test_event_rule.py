"""The event-synchronise rule (adviser's ruling 4a of 9 Nov 2026, seventh paste)."""

import pytest

from attnbench.components import event_rule as E
from attnbench.components import link
from attnbench.predict.recursion import MechanismCost, kernel_op, price, sync_op

COSTS = {"synchronize": MechanismCost(s_fixed=4e-6),
         "event_synchronize": MechanismCost(s_fixed=1.16e-6),
         link.RECORD_THEN_WAIT: MechanismCost(s_fixed=8.59e-6)}


def test_a_record_since_the_last_drain_makes_the_wait_a_round_trip():
    calls = ["cudaLaunchKernel", "cudaEventRecord", "cudaEventSynchronize",   # recorded, waited
             "cudaEventSynchronize",                                           # the same, complete
             "cudaEventRecord", "cudaStreamSynchronize", "cudaEventSynchronize",  # drained between
             "cudaEventRecord", "cudaMemcpy", "cudaEventSynchronize",          # a blocking copy drains
             "cudaEventRecord", "cudaMemcpyAsync", "cudaEventSynchronize"]     # an async one does not
    assert E.kinds_from_calls(calls) == [E.RECORDED, E.COMPLETED, E.COMPLETED, E.COMPLETED,
                                         E.RECORDED]


def test_kinds_reads_the_calls_inside_the_prefill_region_in_time_order():
    ev = lambda name, ts, cat="cuda_runtime", dur=1: {"ph": "X", "cat": cat, "name": name,
                                                       "ts": ts, "dur": dur, "pid": 1, "tid": 1}
    trace = {"traceEvents": [
        ev("cudaEventRecord", 5),                                   # before the region
        ev("attnbench::prefill", 10, cat="user_annotation", dur=100),
        ev("cudaEventSynchronize", 40), ev("cudaEventRecord", 20),  # out of order in the file
        ev("cudaEventSynchronize", 60),
        ev("cudaEventRecord", 200)]}                                # after it
    assert E.kinds(trace) == [E.RECORDED, E.COMPLETED]


def _seq(n_event_syncs):
    ops = [kernel_op(1e-5, 1e-4)]
    for _ in range(n_event_syncs):
        ops += [sync_op("event_synchronize"), kernel_op(1e-5, 1e-4)]
    return ops + [sync_op("synchronize")]


def test_apply_prices_each_recorded_wait_at_the_round_trip():
    p = price(_seq(2), COSTS)
    q = E.apply(p, [E.RECORDED, E.COMPLETED], COSTS)
    pos = E.event_sync_positions(p)
    assert [q.sync_cost[i] for i in pos] == pytest.approx([8.59e-6, 1.16e-6])
    assert [c for i, c in enumerate(q.sync_cost) if i not in pos] == \
        [c for i, c in enumerate(p.sync_cost) if i not in pos]


def test_a_sequence_with_no_event_synchronise_is_unchanged():
    p = price(_seq(0), COSTS)
    assert E.apply(p, [], COSTS) is p
    assert E.apply(p, [], {"synchronize": COSTS["synchronize"]}) is p   # no row needed


def test_the_rule_and_the_sequence_must_agree():
    p = price(_seq(2), COSTS)
    with pytest.raises(E.EventRuleError, match="2 event synchronises"):
        E.apply(p, [E.RECORDED], COSTS)
    with pytest.raises(E.EventRuleError, match="no 'event_record_synchronize'"):
        E.apply(p, [E.RECORDED, E.RECORDED], {k: v for k, v in COSTS.items()
                                              if k != link.RECORD_THEN_WAIT})


def test_the_injectors_event_synchronise_takes_the_round_trip():
    got = E.for_injection(COSTS)
    assert got["event_synchronize"] == COSTS[link.RECORD_THEN_WAIT]
    assert got["synchronize"] == COSTS["synchronize"]
    old = {k: v for k, v in COSTS.items() if k != link.RECORD_THEN_WAIT}
    assert E.for_injection(old) == old
