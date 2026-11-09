"""Every mechanism the recursion prices has a measured entry (researcher, 7 Nov 2026).

The T3.1 debug session's `predict` failed on `copy_from_host_pageable`: the
recursion prices eight mechanisms (`recursion.PRICED_MECHANISMS`) and T2.5
measured five. The prediction keeps refusing a priced mechanism with no
measurement and never defaults; these tests keep the two lists from drifting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from attnbench.components import link
from attnbench.predict import recursion
from test_link_microbench import VirtualDevice


def _row(**kw):
    dev = VirtualDevice()
    return link.measure(dev.ops(), sync_reps=3, copy_reps=2,
                        sizes=link.DEFAULT_SIZES, context_lengths=(2048,),
                        load="python_loop", clock=dev.clock, **kw)


def test_the_measured_mechanisms_are_exactly_the_priced_ones_and_record_then_wait():
    row = _row()
    measured = {k[len("s_fixed_"):-len("_median")] for k in row
                if k.startswith("s_fixed_") and k.endswith("_median")}
    assert measured == set(recursion.PRICED_MECHANISMS) | {link.RECORD_THEN_WAIT}
    assert set(recursion.ADDED_MECHANISMS) <= measured


def test_every_priced_mechanism_has_a_cost_with_the_bandwidth_its_payload_needs():
    costs = link.mechanism_costs(_row(), context_length=2048)
    assert set(costs) == set(recursion.PRICED_MECHANISMS) | {link.RECORD_THEN_WAIT}
    for m, c in costs.items():
        assert c.s_fixed > 0
        if m in link.COPY_MECHANISMS:
            assert c.bandwidth and c.loaded_bandwidth, m
        else:
            assert c.bandwidth is None, m


def test_a_priced_mechanism_cannot_be_added_without_a_measurement():
    """A new name in PRICED_MECHANISMS that link.measure does not time fails here."""
    mechanisms = recursion.PRICED_MECHANISMS
    assert set(link.COPY_MECHANISMS) <= set(mechanisms)
    assert len(set(mechanisms)) == len(mechanisms)


def test_the_row_spans_the_payloads_the_recorded_sequences_use():
    row = _row()
    lo, hi = link.payload_range(row)
    assert lo == 1 and hi == 1 << 28
    ops = [SimpleNamespace(mechanism="copy_from_host_pageable", payload_bytes=2 << 20),
           SimpleNamespace(mechanism="copy_from_host_pinned", payload_bytes=1),
           SimpleNamespace(mechanism="event_synchronize", payload_bytes=0),
           SimpleNamespace(mechanism=None, payload_bytes=0)]            # a kernel
    assert link.uncovered_payloads(row, ops) == []


def test_a_copy_above_the_largest_measured_size_or_an_unmeasured_mechanism_is_named():
    row = _row()
    big = SimpleNamespace(mechanism="copy_from_host_pageable", payload_bytes=(1 << 28) + 1)
    assert link.uncovered_payloads(row, [big]) == [("copy_from_host_pageable", (1 << 28) + 1)]
    without = {k: v for k, v in row.items() if not k.startswith("s_fixed_copy_from_host_pageable")}
    ok = SimpleNamespace(mechanism="copy_from_host_pageable", payload_bytes=8)
    assert link.uncovered_payloads(without, [ok]) == [("copy_from_host_pageable", 8)]


def test_a_cost_table_without_a_priced_mechanism_still_refuses_and_never_defaults():
    row = _row()
    costs = link.mechanism_costs(row)
    del costs["copy_from_host_pageable"]
    op = recursion.sync_op("copy_from_host_pageable", payload_bytes=64)
    with pytest.raises(recursion.CostModelError, match="no s_fixed"):
        recursion.price([recursion.kernel_op(host=1e-6, device=1e-6), op], costs)


def test_a_device_that_cannot_time_a_host_to_device_copy_is_refused():
    dev = VirtualDevice()
    ops = dev.ops()
    ops.blocking_h2d = None
    with pytest.raises(ValueError, match="blocking h2d"):
        link.bandwidth_tables(ops, [1024], 1, link.s_fixed_table(ops, 2, clock=dev.clock),
                              clock=dev.clock)
