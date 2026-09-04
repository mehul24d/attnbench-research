"""First-call cost is recorded separately, because for flex it is the result.

The 2026-09-04 sweep estimate put `flex`'s per-shape compilation at 0.42 of a
0.71-hour sweep -- 59% of the total, against 0.23 h of actual kernel time
across all nine other backends combined. That is not sweep overhead to be
subtracted and forgotten. It is a practical finding about FlexAttention, and
it is invisible in a latency column by construction: `_do_bench` measures
steady state on purpose, so the compile is over before the first sample.

It also decides where flex sits on a Pareto frontier, and the honest answer is
two answers. Compilation amortises in deployment (compile once, serve many)
and does not amortise in a sweep (one compile per shape, 41 calls). A single
number cannot say both, so the sweep records both and the analysis chooses --
visibly.

CPU-only: these test the plumbing and the arithmetic, not a real compile.
"""

from __future__ import annotations

import pytest

from attnbench.timing import Measurement


def _m(**kw):
    base = dict(ok=True, status="ok", latency_ms_p50=10.0)
    base.update(kw)
    return Measurement(**base)


def test_the_overhead_is_the_excess_over_steady_state():
    assert _m(first_call_ms=1010.0).first_call_overhead_ms == pytest.approx(1000.0)


def test_a_backend_that_pays_nothing_extra_reports_zero():
    assert _m(first_call_ms=10.0).first_call_overhead_ms == pytest.approx(0.0)


def test_a_faster_first_call_does_not_produce_negative_overhead():
    """Timer noise can put the first call marginally below the median. A
    negative one-off cost is not a thing, and would poison any sum over it."""
    assert _m(first_call_ms=9.4).first_call_overhead_ms == 0.0


def test_it_is_none_when_either_side_is_missing():
    """An unsupported or OOM cell has no steady state to subtract from, and a
    zero there would read as 'measured, and free'."""
    assert _m(first_call_ms=None).first_call_overhead_ms is None
    assert Measurement(ok=False, status="oom",
                       first_call_ms=5.0).first_call_overhead_ms is None


def test_both_columns_reach_the_result_row():
    """The raw reading AND the derived one. Recording only the difference
    would discard the first call's absolute magnitude, which is what a reader
    comparing 'time to first token' style costs actually wants."""
    d = _m(first_call_ms=1010.0).to_dict()
    assert d["first_call_ms"] == 1010.0
    assert d["first_call_overhead_ms"] == pytest.approx(1000.0)


def test_the_column_is_not_called_compile_ms():
    """Naming discipline, asserted. Every backend pays something on its first
    call -- cuBLAS workspace, autotune, cache warming -- and only `flex` calls
    torch.compile at all. A column named compile_ms would attribute all of it
    to compilation for nine backends that never compile."""
    d = _m(first_call_ms=1010.0).to_dict()
    assert "compile_ms" not in d


def test_it_is_measured_on_the_host_clock():
    """CUDA events would miss the whole point: compilation happens on the CPU,
    so an event pair around the first call records only the kernel that runs
    after it -- which is the steady-state number already being measured."""
    import inspect

    from attnbench import timing

    src = inspect.getsource(timing.measure)
    assert "time.perf_counter()" in src
    head = src.split("first_call_ms =")[0]
    assert "torch.cuda.synchronize()" in head, (
        "the span must close after a synchronize, or device work is excluded")
