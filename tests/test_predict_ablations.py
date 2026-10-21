"""T2.9 model ablations: the two synchronisation ablations, the host-only
and device-only bounds, and the rounding of Q for the queue sensitivity.

Hand-computed numbers are worked in the comments from Section 5.4. The
randomised tests compare `no_sync_wait` with the pending-launch oracle of
test_predict_recursion, extended with the same switch.
"""

from __future__ import annotations

import random

import pytest

from attnbench.predict.ablations import (
    QUEUE_MULTIPLES, device_only_bound, distinct_sensitivity_depths,
    host_only_bound, no_sync_cost, no_sync_wait, sensitivity_depths)
from attnbench.predict.recursion import (
    CostModelError, OpKind, close_prefill, kernel_op, price, recurse, sync_op)

from test_predict_recursion import COSTS, random_ops


def P(ops):
    return price(ops, COSTS)


# K(h1,d10), S(item, 4 bytes: s0 = 1 + 4/4 = 2), K(h1,d1).
# Full recursion (test_predict_recursion): D = 11, 13, 15.
BASE = [kernel_op(1, 10), sync_op("item", payload_bytes=4), kernel_op(1, 1)]


def test_full_recursion_reference_for_the_ablations():
    assert recurse(P(BASE), None).device_done == (11, 13, 15)


def test_no_sync_cost_keeps_the_wait_and_drops_s0():
    # S: t = 1, waits to 11 (wait 10), s0 = 0, H2 = D2 = 11.
    # K: H3 = 12, D3 = max(12, 11) + 1 = 13.
    t = no_sync_cost(P(BASE), None)
    assert t.host_issued == (1, 11, 12)
    assert t.device_done == (11, 11, 13)
    assert t.sync_wait == (0, 10, 0)


def test_no_sync_wait_keeps_s0_and_drops_the_wait():
    # S: t = 1 + 2 = 3, no wait; D2 = max(D1, H2) = max(11, 3) = 11.
    # K: H3 = 4, D3 = max(4, 11) + 1 = 12.
    t = no_sync_wait(P(BASE), None)
    assert t.host_issued == (1, 3, 4)
    assert t.device_done == (11, 11, 12)
    assert t.sync_wait == (0, 0, 0)
    assert t.sync_dependency is False


def test_no_sync_wait_ends_at_the_later_of_host_and_device():
    # K(h1,d2), closing synchronise (s0 1), no wait:
    # H1 = 1, D1 = 3; S: t = 1 + 1 = 2, D2 = max(3, 2) = 3.
    assert no_sync_wait(P(close_prefill([kernel_op(1, 2)])), None).end_to_end == 3
    # K(h1,d0.5): D1 = 1.5; S: t = 2, D2 = max(1.5, 2) = 2.
    assert no_sync_wait(P(close_prefill([kernel_op(1, 0.5)])), None).end_to_end == 2


def test_no_sync_wait_counts_only_launches_in_the_queue():
    # Q = 2. K(h1,d10): H1 = 1, D1 = 11.  K(h1,d10): H2 = 2, D2 = 21.
    # S(synchronize, s0 1), no wait, not a launch: H3 = 3, D3 = max(21, 3) = 21.
    # K(h1,d10): t = 4; the second-last launch is op 1 (D 11), so H4 = 11
    #   (stall 7); D4 = max(11, 21) + 10 = 31.
    # Indexing D_(i-Q) as in the full model would wait for op 2 (H4 = 21).
    ops = [kernel_op(1, 10), kernel_op(1, 10), sync_op("synchronize"),
           kernel_op(1, 10)]
    t = no_sync_wait(P(ops), 2)
    assert t.host_issued == (1, 2, 3, 11)
    assert t.queue_stall == (0, 0, 0, 7)
    assert t.device_done == (11, 21, 21, 31)


def test_bounds_are_the_host_and_device_sums():
    p = P([kernel_op(2, 3), sync_op("item", payload_bytes=4), kernel_op(5, 1)])
    assert host_only_bound(p) == 7
    assert device_only_bound(p) == 4


# --------------------------------------------------------------------------
# Against the oracle, and orderings that must hold
# --------------------------------------------------------------------------

def oracle(ops, s0s, Q, wait):
    host, device_free, launched, out = 0.0, 0.0, [], []
    for op, s in zip(ops, s0s):
        host += op.host
        is_launch = op.kind is not OpKind.SYNC
        if Q is not None and (wait or is_launch):
            while sum(1 for f in launched if f > host) >= Q:
                host = min(f for f in launched if f > host)
        if not is_launch:
            # Without the wait the device is untouched; the operation is
            # done once both sides are.
            host = (max(host, device_free) if wait else host) + s
            out.append(host if wait else max(host, device_free))
        else:
            device_free = max(host, device_free) + op.link + op.device
            launched.append(device_free)
            out.append(device_free)
    return out


@pytest.mark.parametrize("seed,Q", [(s, q) for s in range(60)
                                    for q in (None, 1, 2, 3, 5)])
def test_no_sync_wait_matches_the_oracle(seed, Q):
    rng = random.Random(seed)
    ops = random_ops(rng, rng.randint(1, 40))
    p = P(ops)
    assert list(no_sync_wait(p, Q).device_done) == oracle(ops, p.sync_cost, Q, False)
    assert list(no_sync_cost(p, Q).device_done) == oracle(
        ops, [0.0] * len(p), Q, True)


@pytest.mark.parametrize("seed", range(60))
def test_each_ablation_is_never_slower_than_the_full_recursion(seed):
    rng = random.Random(500 + seed)
    p = P(random_ops(rng, rng.randint(1, 40)))
    for Q in (None, 1, 3):
        full = recurse(p, Q).end_to_end
        assert no_sync_cost(p, Q).end_to_end <= full
        assert no_sync_wait(p, Q).end_to_end <= full
        # Still bounded below by the device chain and by the host chain.
        for t in (no_sync_cost(p, Q), no_sync_wait(p, Q)):
            assert t.end_to_end >= device_only_bound(p)
            assert t.end_to_end >= host_only_bound(p)


def test_without_synchronisations_both_ablations_are_the_recursion():
    p = P([kernel_op(1, 3), kernel_op(4, 1), kernel_op(1, 1)])
    for Q in (None, 1, 2):
        full = recurse(p, Q)
        assert no_sync_cost(p, Q).device_done == full.device_done
        assert no_sync_wait(p, Q).device_done == full.device_done


# --------------------------------------------------------------------------
# Q rounding for the queue sensitivity
# --------------------------------------------------------------------------

def test_the_multiples_are_section_5_4s():
    assert QUEUE_MULTIPLES == (0.25, 0.5, 2, 4)


@pytest.mark.parametrize("Q,expected,distinct", [
    # 0.25 -> 1 (floor 1), 0.5 -> 1 (half up), 2, 4.
    (1, {0.25: 1, 0.5: 1, 2: 2, 4: 4}, (1, 2, 4)),
    # 0.5 -> 1 (half up), 1, 4, 8.
    (2, {0.25: 1, 0.5: 1, 2: 4, 4: 8}, (1, 4, 8)),
    # 1.5 -> 2 (half up), 3, 12, 24.
    (6, {0.25: 2, 0.5: 3, 2: 12, 4: 24}, (2, 3, 12, 24)),
    # 2.5 -> 3 (round() would give 2), 5, 20, 40.
    (10, {0.25: 3, 0.5: 5, 2: 20, 4: 40}, (3, 5, 20, 40)),
    # 1.75 -> 2, 3.5 -> 4.
    (7, {0.25: 2, 0.5: 4, 2: 14, 4: 28}, (2, 4, 14, 28)),
    (1024, {0.25: 256, 0.5: 512, 2: 2048, 4: 4096}, (256, 512, 2048, 4096)),
])
def test_q_rounds_to_nearest_half_up_floor_one(Q, expected, distinct):
    assert sensitivity_depths(Q) == expected
    assert distinct_sensitivity_depths(Q) == distinct


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, None])
def test_measured_q_must_be_a_positive_int(bad):
    with pytest.raises(CostModelError):
        sensitivity_depths(bad)
