"""Audit C1: `predict/recursion.py` and `predict/ablations.py` against plan
Sections 5.1 and 5.4. Each test fails on main at 275faac and demonstrates
one finding. The file is not named test_*.py, so the default suite does not
collect it; run it by name: `python -m pytest tests/audit_C1_recursion.py`.

All of it ran on a CPU without torch, through a scratch loader that skips
the package __init__; none of it needs torch.
"""

from __future__ import annotations

import pytest

from attnbench.components.link import SYNC_MECHANISMS as LINK_MECHANISMS
from attnbench.components.link import mechanism_costs
from attnbench.predict.ablations import no_sync_cost, no_sync_wait
from attnbench.predict.recursion import (
    MechanismCost, close_prefill, kernel_op, price, recurse, sync_op,
)


# ---------------------------------------------------------------------------
# Finding 1 (high). Section 5.1: "For `.item()` and `bool()` of a tensor the
# payload is zero by definition, so their s_0 is s_fixed (Appendix A, 25 Oct
# 2026)." The recursion refuses a payload only for "synchronize"
# (recursion.py:306) and otherwise prices payload / B (recursion.py:193-203).
# The trace parser records item with the scalar's bytes and bool with one
# byte (tests/test_components_op_sequence.py:181 and :191), and the T2.5 row
# gives these mechanisms no bandwidth (link.py:473-481). So a real sequence
# with an item or bool wait is refused, and with a bandwidth supplied it is
# charged payload / B.

@pytest.mark.parametrize("mechanism, payload", [("item", 4), ("bool", 1)])
def test_item_and_bool_s0_is_s_fixed_whatever_the_recorded_payload(mechanism, payload):
    ops = [kernel_op(1.0, 10.0), sync_op(mechanism, payload_bytes=payload)]
    priced = price(ops, {mechanism: MechanismCost(7.0, bandwidth=4.0)})
    assert priced.sync_cost[1] == 7.0


@pytest.mark.parametrize("mechanism, payload", [("item", 4), ("bool", 1)])
def test_item_and_bool_price_from_a_table_shaped_like_the_t25_row(mechanism, payload):
    # link.mechanism_costs gives every non-copy mechanism bandwidth=None.
    ops = [kernel_op(1.0, 10.0), sync_op(mechanism, payload_bytes=payload)]
    priced = price(ops, {mechanism: MechanismCost(7e-6)})
    assert priced.sync_cost[1] == 7e-6


def _t25_row() -> dict:
    row = {"dry_run": False}
    for m in LINK_MECHANISMS:
        row[f"s_fixed_{m}_median"] = 1e-5
    for mem in ("pinned", "pageable"):
        row[f"bw_blocking_d2h_{mem}_Bps"] = 1e10
    return row


@pytest.mark.parametrize("mechanism, payload", [("item", 4), ("bool", 1)])
def test_a_prefill_with_an_item_or_bool_wait_prices_from_the_real_cost_table(mechanism, payload):
    ops = close_prefill([kernel_op(1e-6, 1e-3), sync_op(mechanism, payload_bytes=payload),
                         kernel_op(1e-6, 1e-3)])
    t = recurse(price(ops, mechanism_costs(_t25_row())), None)
    # D: 1.001e-3, then the wait and s_fixed, then the second kernel, then
    # the closing synchronise.
    assert t.end_to_end == pytest.approx(1.001e-3 + 1e-5 + 1e-3 + 1e-6 + 1e-5)


# ---------------------------------------------------------------------------
# Finding 2 (medium). Appendix A, 25 Oct 2026: "a blocking host-to-device copy
# and an event synchronise are synchronisation mechanisms with their own
# s_fixed" (Section 5.1 lists both). recursion.SYNC_MECHANISMS
# (recursion.py:86-92) has neither, so the recursion refuses them as unknown
# even with a cost supplied, while T2.8 is marked Done. The names are the
# ones the trace parser emits (op_sequence.py:135).

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
# before reading the clock; it is the timer's, not the arm's. Both ablations
# also strip it (ablations.py:33 zeroes its s_0; recursion.py:425-427 drops
# its wait), so on a sequence with no synchronisation of its own the
# "without the synchronisation term" model still differs from the full
# model, by up to s_fixed(synchronize), and is no longer on the ground
# truth's clock.

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
