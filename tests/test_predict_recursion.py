"""T2.8: the recursion of plan Section 5.4, against hand-computed timelines.

Every number in the hand-computed tests is worked out in the comment beside
it from the plan's equations, not from the code. Costs are small integers so
the comparisons are exact. The randomised tests compare the recursion with an
oracle written a different way (it counts pending launches instead of
indexing D_(i-Q)), so a shared mistake would have to be made twice.
"""

from __future__ import annotations

import random

import pytest

from attnbench.predict.recursion import (
    COMPOSITE_MECHANISMS, SYNC_MECHANISMS, CostModelError, MechanismCost,
    OpKind, Operation, close_prefill, closing_sync_op, copy_op,
    is_closing_sync, kernel_op, predict_prefill, price, recurse, s0, simulate,
    sync_op)

# Bandwidths are powers of two so every s_0 is exact in binary floating
# point, and every comparison below can be exact.
COSTS = {
    "synchronize": MechanismCost(s_fixed=1),
    "item": MechanismCost(s_fixed=1, bandwidth=4, loaded_bandwidth=2),
    "bool": MechanismCost(s_fixed=2, bandwidth=4, loaded_bandwidth=2),
    "copy_to_host_pinned": MechanismCost(s_fixed=1, bandwidth=1024,
                                         loaded_bandwidth=512),
    "copy_to_host_pageable": MechanismCost(s_fixed=3, bandwidth=128,
                                           loaded_bandwidth=64),
}


def run(ops, Q=None, costs=COSTS):
    return simulate(ops, costs, Q)


# --------------------------------------------------------------------------
# Hand-computed timelines
# --------------------------------------------------------------------------

def test_host_bound_sequence_ends_one_kernel_after_the_last_issue():
    # h = 3,3,3; d = 1,1,1; no queue term.
    # H = 3, 6, 9.  D1 = max(3,0)+1 = 4; D2 = max(6,4)+1 = 7; D3 = max(9,7)+1 = 10.
    t = run([kernel_op(3, 1), kernel_op(3, 1), kernel_op(3, 1)])
    assert t.host_issued == (3, 6, 9)
    assert t.device_done == (4, 7, 10)
    assert t.end_to_end == 10


def test_device_bound_sequence_is_first_issue_plus_all_device_time():
    # h = 1,1,1; d = 5,5,5.  H = 1, 2, 3.  D = 6, 11, 16.
    t = run([kernel_op(1, 5), kernel_op(1, 5), kernel_op(1, 5)])
    assert t.host_issued == (1, 2, 3)
    assert t.device_done == (6, 11, 16)


def test_copy_and_kernel_are_serialised_on_the_one_stream():
    # K(h1,d2): H1 = 1, D1 = 3.  C(h1,l4): H2 = 2, D2 = max(2,3)+4 = 7.
    t = run([kernel_op(1, 2), copy_op(1, 4)])
    assert t.device_done == (3, 7)


def test_synchronisation_waits_for_the_device_then_pays_s0():
    # K(h1,d10): H1 = 1, D1 = 11.
    # S(item, 4 bytes, unloaded): s0 = 1 + 4/4 = 2.
    #   t = 1 + 0 = 1; max(1, 11) = 11, so the wait is 10; H2 = 11 + 2 = 13 = D2.
    # K(h1,d1): H3 = 14; D3 = max(14,13)+1 = 15.
    t = run([kernel_op(1, 10), sync_op("item", payload_bytes=4), kernel_op(1, 1)])
    assert t.host_issued == (1, 13, 14)
    assert t.device_done == (11, 13, 15)
    assert t.sync_wait == (0, 10, 0)


def test_synchronisation_on_an_idle_device_does_not_wait_but_pays_s0():
    # K(h1,d1):          H1 = 1, D1 = 2.
    # S(synchronize, 1): t = 1, max(1, 2) = 2 (wait 1), H2 = 3 = D2.
    # S(synchronize, 1): t = 3, max(3, 3) = 3 (no wait), H3 = 4 = D3.
    t = run([kernel_op(1, 1), sync_op("synchronize"), sync_op("synchronize")])
    assert t.host_issued == (1, 3, 4)
    assert t.device_done == (2, 3, 4)
    assert t.sync_wait == (0, 1, 0)


def test_a_synchronisation_with_host_cost_is_refused():
    # s_fixed already holds the host cost of issuing and completing it.
    with pytest.raises(CostModelError, match="no host cost"):
        run([kernel_op(1, 1),
             Operation(OpKind.SYNC, host=3, mechanism="synchronize")])


def test_closing_synchronise_ends_the_timed_interval():
    # K(h1,d5): H1 = 1, D1 = 6.
    # closing synchronise, s_fixed 1: t = 1, waits to 6, H2 = D2 = 7.
    ops = close_prefill([kernel_op(1, 5)])
    assert is_closing_sync(ops[-1]) and ops[-1].mechanism == "synchronize"
    t = run(ops)
    assert t.device_done == (6, 7)
    assert t.end_to_end == 7


def test_closing_synchronise_appears_once_and_last():
    ops = close_prefill([kernel_op(1, 5)])
    with pytest.raises(CostModelError, match="already has a closing"):
        close_prefill(ops)
    with pytest.raises(CostModelError, match="must be the last operation"):
        run([closing_sync_op(), kernel_op(1, 1)])
    with pytest.raises(CostModelError, match="must be the last operation"):
        run([kernel_op(1, 1), closing_sync_op(), closing_sync_op()])
    # An ordinary synchronise anywhere is still fine.
    run([sync_op("synchronize"), kernel_op(1, 1), closing_sync_op()])


def test_loaded_bandwidth_inflates_only_the_payload_term():
    # pinned copy to host, 1024 bytes: unloaded 1 + 1024/1024 = 2,
    # loaded 1 + 1024/512 = 3.  s_fixed is the same in both.
    c = COSTS["copy_to_host_pinned"]
    assert s0(c, 1024, loaded=False) == 2
    assert s0(c, 1024, loaded=True) == 3
    assert s0(c, 0, loaded=True) == 1
    # In the recursion: K(h1,d1) then the copy.  D1 = 2; t = 1, waits to 2,
    # H2 = 2 + s0.
    assert run([kernel_op(1, 1), sync_op("copy_to_host_pinned",
                                         payload_bytes=1024)]).end_to_end == 4
    assert run([kernel_op(1, 1), sync_op("copy_to_host_pinned",
                                         payload_bytes=1024,
                                         loaded=True)]).end_to_end == 5


def test_a_sparse_layer_with_an_estimator_read_and_a_host_mask_build():
    # K qkv (h2,d3):              H1 = 2,  D1 = 5.
    # S item, 4 bytes, s0 = 2:    t = 2, waits to 5 (wait 3), H2 = 7 = D2.
    # C mask H2D (h6 incl. the CPU mask build, l2):
    #                             H3 = 13, D3 = max(13,7)+2 = 15.
    # K sparse (h1,d4):           H4 = 14, D4 = max(14,15)+4 = 19.
    ops = [kernel_op(2, 3), sync_op("item", payload_bytes=4),
           copy_op(6, 2), kernel_op(1, 4)]
    t = run(ops)
    assert t.host_issued == (2, 7, 13, 14)
    assert t.device_done == (5, 7, 15, 19)
    assert t.sync_wait == (0, 3, 0, 0)
    # Same layer with the estimator's read alongside host-side work: the
    # payload term doubles (4/2 = 2), s0 = 3, and everything after moves by 1.
    ops[1] = sync_op("item", payload_bytes=4, loaded=True)
    assert run(ops).end_to_end == 20


def test_queue_holds_the_host_back_without_changing_a_device_bound_end():
    # h = 1 x4, d = 10 x4.
    # No queue: H = 1,2,3,4; D = 11,21,31,41.
    # Q = 2: op3 t = 3, D1 = 11 > 3, H3 = 11 (stall 8); D3 = max(11,21)+10 = 31.
    #        op4 t = 12, D2 = 21 > 12, H4 = 21 (stall 9); D4 = 41.
    ops = [kernel_op(1, 10) for _ in range(4)]
    free = run(ops)
    assert free.host_issued == (1, 2, 3, 4)
    assert free.device_done == (11, 21, 31, 41)
    q2 = run(ops, Q=2)
    assert q2.host_issued == (1, 2, 11, 21)
    assert q2.queue_stall == (0, 0, 8, 9)
    assert q2.device_done == (11, 21, 31, 41)


def test_queue_delays_host_work_that_follows_a_full_queue():
    # The previous sequence plus a long host op (a CPU mask build, h30, d1).
    # No queue: H5 = 4+30 = 34, D5 = max(34,41)+1 = 42.
    # Q = 2:    t = 21+30 = 51, D3 = 31 < 51 so no stall; D5 = max(51,41)+1 = 52.
    ops = [kernel_op(1, 10) for _ in range(4)] + [kernel_op(30, 1)]
    assert run(ops).end_to_end == 42
    assert run(ops, Q=2).end_to_end == 52


def test_queue_wait_comes_after_the_host_cost_of_the_operation():
    # The plan adds h_i first, then takes max with D_(i-Q).
    # Q = 1. K(h1,d10): H1 = 1, D1 = 11.
    # K(h4,d1): t = 1+4 = 5, D1 = 11 > 5, H2 = 11; D2 = 12.
    # (Waiting first, max(1,11)+4 = 15 and D2 = 16, would be the other order.)
    t = run([kernel_op(1, 10), kernel_op(4, 1)], Q=1)
    assert t.host_issued == (1, 11)
    assert t.device_done == (11, 12)


def test_queue_term_starts_only_once_Q_operations_exist():
    # Q = 3 with three ops: D_(i-3) never exists, so nothing stalls.
    # H = 1,2,3 and D = 11,21,31 either way.
    ops = [kernel_op(1, 10) for _ in range(3)]
    q3 = run(ops, Q=3)
    assert q3.host_issued == (1, 2, 3)
    assert q3.device_done == (11, 21, 31)
    assert q3.queue_stall == (0, 0, 0)


# --------------------------------------------------------------------------
# Refusals (Section 5.1 and the scope line of Section 5.4)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("op", [
    Operation(OpKind.SYNC, host=0, device=1, mechanism="item"),
    Operation(OpKind.SYNC, host=0, link=1, mechanism="item"),
])
def test_a_synchronisation_with_device_or_link_cost_is_refused(op):
    with pytest.raises(CostModelError, match="no device or link cost"):
        run([kernel_op(1, 1), op])


@pytest.mark.parametrize("mechanism", sorted(COMPOSITE_MECHANISMS))
def test_composite_mechanisms_must_arrive_split(mechanism):
    with pytest.raises(CostModelError, match="composite.*kernel"):
        run([kernel_op(1, 1), sync_op(mechanism)])


def test_unknown_mechanism_and_missing_s_fixed_are_refused():
    with pytest.raises(CostModelError, match="unknown synchronisation"):
        run([sync_op("event_query")])
    partial = {k: v for k, v in COSTS.items() if k != "bool"}
    with pytest.raises(CostModelError, match="no s_fixed for mechanism 'bool'"):
        run([sync_op("bool")], costs=partial)


def test_loaded_payload_without_a_loaded_bandwidth_is_refused_not_substituted():
    costs = {**COSTS, "item": MechanismCost(s_fixed=1, bandwidth=4)}
    with pytest.raises(CostModelError, match="loaded_bandwidth"):
        run([sync_op("item", payload_bytes=4, loaded=True)], costs=costs)
    # Zero payload needs no bandwidth at all.
    assert run([sync_op("item", loaded=True)], costs=costs).end_to_end == 1


def test_explicit_synchronise_moves_no_payload():
    with pytest.raises(CostModelError, match="moves no payload"):
        run([sync_op("synchronize", payload_bytes=8)])


def test_kernel_link_and_copy_device_costs_are_refused():
    with pytest.raises(CostModelError, match="kernel has no link"):
        run([Operation(OpKind.KERNEL, host=1, device=1, link=1)])
    with pytest.raises(CostModelError, match="link cost only"):
        run([Operation(OpKind.COPY, host=1, device=1, link=1)])


def test_sync_fields_on_a_kernel_or_copy_are_refused():
    with pytest.raises(CostModelError, match="synchronisations only"):
        run([kernel_op(1, 1, mechanism="item")])
    with pytest.raises(CostModelError, match="synchronisations only"):
        run([copy_op(1, 1, loaded=True)])
    with pytest.raises(CostModelError, match="moves no host-device bytes"):
        run([kernel_op(1, 1, payload_bytes=8)])


def test_a_copy_may_state_its_bytes_without_changing_its_price():
    # The bytes feed transfer demand (T2.10); l stays the priced cost.
    # K(h1,d2): D1 = 3.  C(h1,l4, 4096 bytes): D2 = max(2, 3) + 4 = 7.
    ops = [kernel_op(1, 2), copy_op(1, 4, payload_bytes=4096),
           sync_op("item", payload_bytes=4)]
    p = price(ops, COSTS)
    assert p.payload_bytes == (0, 4096, 4)
    assert recurse(p, None).device_done[:2] == (3, 7)
    assert recurse(p, None) == recurse(price([kernel_op(1, 2), copy_op(1, 4),
                                              sync_op("item", payload_bytes=4)],
                                             COSTS), None)


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_payload_bytes_must_be_a_non_negative_int_on_any_operation(bad):
    with pytest.raises(CostModelError, match="payload_bytes"):
        run([copy_op(1, 1, payload_bytes=bad)])
    with pytest.raises(CostModelError, match="payload_bytes"):
        run([sync_op("item", payload_bytes=bad)])


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "1"])
def test_costs_must_be_finite_non_negative_numbers(bad):
    with pytest.raises(CostModelError):
        run([kernel_op(bad, 1)])
    with pytest.raises(CostModelError):
        run([kernel_op(1, bad)])
    with pytest.raises(CostModelError):
        run([sync_op("item")], costs={"item": MechanismCost(s_fixed=bad)})


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_queue_depth_is_a_positive_int_or_none(bad):
    with pytest.raises(CostModelError, match="queue_depth"):
        run([kernel_op(1, 1)], Q=bad)


def test_more_than_one_stream_is_refused():
    with pytest.raises(CostModelError, match="2 GPU streams"):
        run([kernel_op(1, 1, stream=7), kernel_op(1, 1, stream=8)])
    # One named stream, or none recorded, is fine.
    run([kernel_op(1, 1, stream=7), sync_op("item", stream=7)])


def test_empty_sequence_is_refused():
    with pytest.raises(CostModelError, match="empty"):
        run([])


def test_the_mechanism_vocabulary_covers_section_5_1():
    # Section 5.1: explicit synchronise, .item(), bool() of a tensor, pinned
    # or pageable copy to the host. Composites are split, so never listed.
    assert set(SYNC_MECHANISMS) == {"synchronize", "item", "bool",
                                    "copy_to_host_pinned",
                                    "copy_to_host_pageable"}
    assert not set(SYNC_MECHANISMS) & set(COMPOSITE_MECHANISMS)


# --------------------------------------------------------------------------
# Pricing: the interface the baselines read
# --------------------------------------------------------------------------

def test_price_returns_the_components_the_recursion_reads():
    ops = [kernel_op(2, 3, layer=0), sync_op("bool", payload_bytes=8, layer=0),
           copy_op(1, 4, layer=1)]
    p = price(ops, COSTS)
    assert p.host == (2, 0, 1)
    assert p.device == (3, 0, 0)
    assert p.link == (0, 0, 4)
    assert p.sync_cost == (0, 2 + 8 / 4, 0)
    assert p.is_sync == (False, True, False)
    assert p.kinds == (OpKind.KERNEL, OpKind.SYNC, OpKind.COPY)
    assert p.layers == (0, 0, 1)
    assert p.mechanisms == (None, "bool", None)
    assert recurse(p, None) == simulate(ops, COSTS, None)
    assert predict_prefill(ops, COSTS, None) == recurse(p, None).end_to_end


# --------------------------------------------------------------------------
# Against an independent oracle, and properties the recursion must keep
# --------------------------------------------------------------------------

def oracle(ops, s0s, Q):
    """Event view: the host issues in order and blocks while Q launches are
    pending; the device runs launches in order; a synchronisation waits for
    the device to drain. Counts pending launches instead of using D_(i-Q)."""
    host = 0.0
    device_free = 0.0
    launched = []          # finish times of kernels and copies, in order
    out = []
    for op, s in zip(ops, s0s):
        host += op.host
        if Q is not None:
            while sum(1 for f in launched if f > host) >= Q:
                host = min(f for f in launched if f > host)
        if op.kind is OpKind.SYNC:
            host = max(host, device_free) + s
            out.append(host)
        else:
            device_free = max(host, device_free) + op.link + op.device
            launched.append(device_free)
            out.append(device_free)
    return out


def random_ops(rng, n):
    ops = []
    for _ in range(n):
        r = rng.random()
        if r < 0.55:
            ops.append(kernel_op(rng.randint(0, 6), rng.randint(0, 12)))
        elif r < 0.75:
            ops.append(copy_op(rng.randint(0, 4), rng.randint(0, 6)))
        else:
            m = rng.choice(SYNC_MECHANISMS)
            payload = 0 if m == "synchronize" else rng.choice([0, 4, 8, 512])
            ops.append(sync_op(m, payload_bytes=payload,
                               loaded=rng.random() < 0.5))
    if rng.random() < 0.5:
        ops = close_prefill(ops)
    return ops


CASES = [(seed, Q) for seed in range(60) for Q in (None, 1, 2, 3, 5, 16)]


@pytest.mark.parametrize("seed,Q", CASES)
def test_recursion_matches_the_pending_launch_oracle(seed, Q):
    rng = random.Random(seed)
    ops = random_ops(rng, rng.randint(1, 40))
    p = price(ops, COSTS)
    assert list(recurse(p, Q).device_done) == oracle(ops, p.sync_cost, Q)


@pytest.mark.parametrize("seed", range(60))
def test_bounds_monotonicity_and_accounting(seed):
    rng = random.Random(1000 + seed)
    ops = random_ops(rng, rng.randint(1, 40))
    p = price(ops, COSTS)
    t = recurse(p, None)
    D_N = t.end_to_end

    # Serial sum of every component is an upper bound; the host chain and
    # the device chain are lower bounds.
    assert D_N <= sum(p.host) + sum(p.link) + sum(p.device) + sum(p.sync_cost)
    assert D_N >= sum(p.host) + sum(p.sync_cost)
    assert D_N >= sum(p.link) + sum(p.device)

    # A shorter queue is never faster; Q at least N is no queue at all.
    ends = [recurse(p, Q).end_to_end for Q in (1, 2, 3, 5, 8)]
    assert ends == sorted(ends, reverse=True)
    assert ends[-1] >= D_N
    assert recurse(p, len(p)).device_done == t.device_done

    # H_i = H_(i-1) + h_i + stall_i + wait_i + s0_i, every operation.
    for Q in (None, 2):
        tq = recurse(p, Q)
        prev = 0.0
        for k in range(len(p)):
            assert tq.host_issued[k] == (prev + p.host[k] + tq.queue_stall[k]
                                         + tq.sync_wait[k] + p.sync_cost[k])
            prev = tq.host_issued[k]


@pytest.mark.parametrize("seed", range(30))
def test_raising_any_one_cost_never_shortens_the_prediction(seed):
    rng = random.Random(2000 + seed)
    ops = random_ops(rng, rng.randint(1, 25))
    Q = rng.choice([None, 1, 2, 4])
    base = simulate(ops, COSTS, Q).end_to_end
    launches = [k for k, op in enumerate(ops) if op.kind is not OpKind.SYNC]
    if launches:
        k = rng.choice(launches)
        op = ops[k]
        field = {"kernel": "device", "copy": "link"}[op.kind.value]
        bumped = list(ops)
        bumped[k] = Operation(**{**op.__dict__, field: getattr(op, field) + 3})
        bumped_host = list(ops)
        bumped_host[k] = Operation(**{**op.__dict__, "host": op.host + 3})
        assert simulate(bumped, COSTS, Q).end_to_end >= base
        assert simulate(bumped_host, COSTS, Q).end_to_end >= base
    slower = {m: MechanismCost(c.s_fixed + 1, c.bandwidth, c.loaded_bandwidth)
              for m, c in COSTS.items()}
    assert simulate(ops, slower, Q).end_to_end >= base
