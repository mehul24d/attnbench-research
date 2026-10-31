"""Fakes for the ground-truth harness tests (T2.13): one fake clock that every
arm's backend and the clock reader share, so drift, throttle windows and
timestamps are scripted exactly instead of observed.

`CostedBackend` stands in for a kernel: each forward costs a scripted number
of milliseconds on the fake clock, split over the layers. `ScriptedReader`
stands in for NVML: its cumulative violation counters are the overlap of the
fake clock's history with scripted windows, which is what the real counters
are, so a block is flagged exactly when it overlapped a window.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from attnbench.backends.impls import NaiveAttention
from attnbench.clock_sampling import ClockSample

LAYERS = 3


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += ms / 1e3


class CostedBackend(NaiveAttention):
    """A `NaiveAttention` whose forward advances the fake clock.

    `cost_ms(forward_index)` is the whole forward's cost; it is charged
    `1/LAYERS` per layer. `forward_index` counts this backend's forwards from
    0, warm-up included. `traced_extra_ms` is added per forward while
    `tracing.on` is set, which is what a profiler's hooks do.
    """

    def __init__(self, clock: FakeClock, cost_ms, *, tracing=None,
                 traced_extra_ms: float = 0.0, setting=None, log=None, name="b"):
        super().__init__()
        self.clock, self.cost_ms = clock, cost_ms
        self.tracing, self.traced_extra_ms = tracing, traced_extra_ms
        self.setting = setting
        self.layer_calls = 0
        self.log = log if log is not None else []
        self.arm_name = name

    def forward(self, q, k, v, cfg, mask=None):
        idx = self.layer_calls // LAYERS
        self.layer_calls += 1
        if self.layer_calls % LAYERS == 1:
            self.log.append((self.arm_name, idx, self.clock.t,
                             bool(self.tracing and self.tracing.on)))
        extra = self.traced_extra_ms if (self.tracing and self.tracing.on) else 0.0
        self.clock.advance_ms((self.cost_ms(idx) + extra) / LAYERS)
        return super().forward(q, k, v, cfg, mask=mask)


class Tracing:
    """A tracer factory: `tracer()` is a context manager that sets `on`."""

    def __init__(self):
        self.on = False
        self.entered = 0

    @contextmanager
    def __call__(self):
        self.on = True
        self.entered += 1
        try:
            yield
        finally:
            self.on = False


class ScriptedReader:
    """A `ClockReader` over the fake clock. `windows` are (start_s, end_s,
    "power" | "thermal") intervals during which the card was below its
    requested clocks; the cumulative counters are the overlap so far."""

    def __init__(self, clock: FakeClock, windows=(), *, bits_always: int = 0,
                 read_cost_ms: float = 0.0, transients: bool = True,
                 thread_bits: int = 0):
        self.clock, self.windows = clock, list(windows)
        self.bits_always, self.read_cost_ms = bits_always, read_cost_ms
        self.observes_transients = transients
        self.thread_bits = thread_bits       # reported only to the sampling thread
        self.reads = 0
        self.thread_reads = 0
        self.thread_seen = threading.Event()
        self._main = threading.current_thread()

    def _violation(self):
        out = {"power": 0, "thermal": 0}
        for a, b, policy in self.windows:
            out[policy] += int(max(0.0, min(self.clock.t, b) - a) * 1e9)
        return out

    def read(self) -> ClockSample:
        self.reads += 1
        bits = self.bits_always
        if threading.current_thread() is not self._main:
            self.thread_reads += 1
            bits |= self.thread_bits
            if self.thread_reads >= 2:
                self.thread_seen.set()
        if threading.current_thread() is self._main:
            # the boundary reads, which must sit outside the clock; an in-block
            # thread's reads cost the host, not the fake clock
            self.clock.advance_ms(self.read_cost_ms)
        return ClockSample(sm_mhz=2040, mem_mhz=6250, throttle_bits=bits,
                           violation_ns=self._violation() if self.observes_transients else None)


def fake_densities(arm, run):
    """A `time_arms` density probe for the CPU rigs: they have no block-sparse
    kernel to record at. Dense rows are what `densities.DENSE_FIELDS` says;
    the others report a fixed pair. It does not call `run`, so the rigs'
    forward counts and fake clocks are untouched."""
    if arm.kind == "dense":
        return {"mask_density": None, "executed_density": 1.0}
    return {"mask_density": 0.25, "executed_density": 0.25}
