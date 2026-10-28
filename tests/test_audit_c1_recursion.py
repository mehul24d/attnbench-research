"""Audit C1: `predict/recursion.py` and `predict/ablations.py` against plan
Sections 5.1 and 5.4. Each test failed on main at 275faac and demonstrates
one finding. All three findings are fixed on this branch, and these tests
run as regression tests. None of this file needs torch.
"""

from __future__ import annotations

import pytest

from attnbench.components.link import SYNC_MECHANISMS as LINK_MECHANISMS
from attnbench.components.link import mechanism_costs
from attnbench.components.op_sequence import (CodePath, OpSequence, SeqKind,
                                              SeqOp, to_operations)
from attnbench.predict.ablations import no_sync_cost, no_sync_wait
from attnbench.predict.recursion import (
    MechanismCost, close_prefill, kernel_op, price, recurse, sync_op,
)


# ---------------------------------------------------------------------------
# Finding 1 (high). Section 5.1: "For `.item()` and `bool()` of a tensor the
# payload is zero by definition, so their s_0 is s_fixed (Appendix A, 25 Oct
# 2026)." On main the trace parser recorded item with the scalar's bytes and
# bool with one byte, `to_operations` passed them on, and the recursion
# priced them as payload / B (recursion.py:193-203); the T2.5 row gives these
# mechanisms no bandwidth (link.py:473-481), so a real sequence with an item
# or bool wait was refused, and with a bandwidth supplied it was charged
# payload / B. Fixed on this branch where the sequence meets the recursion:
# `to_operations` hands these waits over with payload zero, and the sequence
# keeps the recorded bytes for the "same sequence" comparison.

def _sequence(mechanism: str, payload: int) -> OpSequence:
    ops = (SeqOp(SeqKind.KERNEL, api="cudaLaunchKernel", kernels=("k1",)),
           SeqOp(SeqKind.SYNC, api="cudaStreamSynchronize", mechanism=mechanism,
                 payload_bytes=payload),
           SeqOp(SeqKind.KERNEL, api="cudaLaunchKernel", kernels=("k2",)))
    return OpSequence(code_path=CodePath(arm="audit", gpu="none",
                                         implementation="audit"),
                      context_length=2048, batch=1, ops=ops, streams=(7,))


def _operations(mechanism: str, payload: int, host: float, device: float):
    seq = _sequence(mechanism, payload)
    ops = to_operations(seq, host={0: host, 2: host},
                        device={0: device, 2: device}, link={})
    # The recorded payload stays in the sequence; the recursion gets none.
    assert seq.ops[1].payload_bytes == payload
    assert ops[1].mechanism == mechanism and ops[1].payload_bytes == 0
    return ops


@pytest.mark.parametrize("mechanism, payload", [("item", 4), ("bool", 1)])
def test_item_and_bool_s0_is_s_fixed_whatever_the_recorded_payload(mechanism, payload):
    ops = _operations(mechanism, payload, 1.0, 10.0)
    priced = price(ops, {mechanism: MechanismCost(7.0, bandwidth=4.0)})
    assert priced.sync_cost[1] == 7.0


@pytest.mark.parametrize("mechanism, payload", [("item", 4), ("bool", 1)])
def test_item_and_bool_price_from_a_table_shaped_like_the_t25_row(mechanism, payload):
    # link.mechanism_costs gives every non-copy mechanism bandwidth=None.
    ops = _operations(mechanism, payload, 1.0, 10.0)
    priced = price(ops, {mechanism: MechanismCost(7e-6)})
    assert priced.sync_cost[1] == 7e-6


def _t25_row() -> dict:
    row = {"dry_run": False}
    for m in LINK_MECHANISMS:
        row[f"s_fixed_{m}_median"] = 1e-5
    for mem in ("pinned", "pageable"):
        row[f"bw_blocking_d2h_{mem}_Bps"] = 1e10
    return row


@pytest.mark.parametrize("mechanism", ["copy_to_host_pinned", "copy_to_host_pageable"])
def test_a_copy_to_the_host_keeps_its_payload(mechanism):
    # Only item and bool lose their payload; a copy's bytes are its transfer.
    ops = to_operations(_sequence(mechanism, 1024), host={0: 1.0, 2: 1.0},
                        device={0: 10.0, 2: 10.0}, link={})
    assert ops[1].payload_bytes == 1024
    priced = price(ops, {mechanism: MechanismCost(7.0, bandwidth=256.0)})
    assert priced.sync_cost[1] == 7.0 + 1024 / 256.0


@pytest.mark.parametrize("mechanism, payload", [("item", 4), ("bool", 1)])
def test_a_prefill_with_an_item_or_bool_wait_prices_from_the_real_cost_table(mechanism, payload):
    ops = close_prefill(_operations(mechanism, payload, 1e-6, 1e-3))
    t = recurse(price(ops, mechanism_costs(_t25_row())), None)
    # D: 1.001e-3, then the wait and s_fixed, then the second kernel, then
    # the closing synchronise.
    assert t.end_to_end == pytest.approx(1.001e-3 + 1e-5 + 1e-3 + 1e-6 + 1e-5)


# ---------------------------------------------------------------------------
# Finding 2 (medium). Appendix A, 25 Oct 2026: "a blocking host-to-device copy
# and an event synchronise are synchronisation mechanisms with their own
# s_fixed" (Section 5.1 lists both). On main, recursion.SYNC_MECHANISMS
# (recursion.py:86-92) had neither, so the recursion refused them as unknown
# even with a cost supplied, while T2.8 is marked Done. The names are the
# ones the trace parser emits (op_sequence.py:135). Fixed on this branch:
# recursion.PRICED_MECHANISMS.

@pytest.mark.parametrize("mechanism, payload, expected", [
    ("event_synchronize", 0, 3.0),
    ("copy_from_host_pinned", 8, 3.0 + 8 / 2.0),
    ("copy_from_host_pageable", 8, 3.0 + 8 / 2.0),
])
def test_the_section_5_1_mechanisms_decided_on_25_oct_are_priced(mechanism, payload, expected):
    ops = [kernel_op(1.0, 10.0), sync_op(mechanism, payload_bytes=payload)]
    priced = price(ops, {mechanism: MechanismCost(3.0, bandwidth=2.0)})
    assert priced.sync_cost[1] == expected


# ---------------------------------------------------------------------------
# Finding 3 (low; plan ambiguity). Section 5.4: "Model ablations ...: the
# recursion without the synchronisation term". Appendix A, 22 Oct 2026 (2)
# adds the closing synchronise because the ground-truth timer synchronises
# before reading the clock; it is the timer's, not the arm's. On main both
# ablations also stripped it (ablations.py:33 zeroed its s_0;
# recursion.py:425-427 dropped its wait), so on a sequence with no
# synchronisation of its own the "without the synchronisation term" model
# still differed from the full model, by up to s_fixed(synchronize). Fixed on
# this branch by the researcher's decision: both act on the arm's own
# synchronisations only.

def _dense_closed():
    # Device-bound, no synchronisation of the arm's own.
    ops = close_prefill([kernel_op(1.0, 10.0), kernel_op(1.0, 10.0)])
    return price(ops, {"synchronize": MechanismCost(2.0)})


@pytest.mark.parametrize("ablation", [no_sync_cost, no_sync_wait],
                         ids=["no_sync_cost", "no_sync_wait"])
def test_an_ablation_leaves_a_sequence_without_arm_synchronisations_unchanged(ablation):
    priced = _dense_closed()
    assert recurse(priced, None).end_to_end == 23.0     # 21 + s_fixed 2
    assert ablation(priced, None).end_to_end == 23.0
