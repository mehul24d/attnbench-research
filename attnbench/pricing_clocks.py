"""The GPU's clocks and power through the pricing steps (adviser's rulings 3
and 4b of 9 Nov 2026, seventh paste).

T3.2c sampled NVML in the timed ground-truth rounds only (step 9, per block:
the SM clock's minimum and maximum, the memory clock, the throttle reasons).
Steps 5 and 6, whose traces and replays price every kernel, were unsampled,
so leg 2's clock-state reading could not be tested. This module samples the
SM and memory clocks, the board power and the throttle reasons at `HZ` from
a separate process through one cell of step 5 or step 6, and the cell marks
its phases (`phase`), so each phase's clock state is on record beside the
prices it produced.

A separate process, so the sampler never takes the timed host code's
interpreter lock. The timed ground-truth path does not import this module:
its own sampler (`clock_sampling`) is unchanged.

Without NVML (a CPU dry run, the CPU boundary) the phases are still marked
and the record says `sampled: false`.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

SCHEMA = "attnbench.pricing_clocks/1"
HZ = 10
_THROTTLE_FUNCS = ("nvmlDeviceGetCurrentClocksEventReasons",
                   "nvmlDeviceGetCurrentClocksThrottleReasons")


def nvml_read_factory(index: int = 0) -> Callable[[], dict]:
    """A reader of one sample: {sm_mhz, mem_mhz, power_w, reasons}. Called in
    the sampler's process."""
    import pynvml as n
    n.nvmlInit()
    h = n.nvmlDeviceGetHandleByIndex(index)
    throttle = next(getattr(n, f) for f in _THROTTLE_FUNCS if hasattr(n, f))

    def read() -> dict:
        return {"sm_mhz": int(n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                "mem_mhz": int(n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)),
                "power_w": n.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "reasons": int(throttle(h))}
    return read


def _child(path: str, interval_s: float, factory, args, ready, stop) -> None:
    read = factory(*args)
    with open(path, "w") as f:
        ready.set()
        while not stop.is_set():
            t = time.monotonic()
            f.write(json.dumps({"t": t, **read()}) + "\n")
            f.flush()
            stop.wait(max(0.0, interval_s - (time.monotonic() - t)))


def _stats(xs) -> Optional[dict]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def summarise(samples: list, phases: list) -> dict:
    """Per phase: its samples' count, SM and memory clock, power, and the
    throttle bits seen (OR of every sample in it)."""
    out = {}
    for p in phases:
        inside = [s for s in samples if p["t0"] <= s["t"] <= p["t1"]]
        bits = 0
        for s in inside:
            bits |= s["reasons"]
        out[p["name"]] = {"t0": p["t0"], "t1": p["t1"], "samples": len(inside),
                          "sm_mhz": _stats(s["sm_mhz"] for s in inside),
                          "mem_mhz": _stats(s["mem_mhz"] for s in inside),
                          "power_w": _stats(s["power_w"] for s in inside),
                          "reasons_bits": bits}
    return out


class PricingClocks:
    """`with PricingClocks(out_dir, enabled=...) as pc: with pc.phase("x"): ...`.
    Writes `pricing_clocks.jsonl` (the samples) and `pricing_clocks.json`
    (the phases and their summaries) into `out_dir` on exit."""

    def __init__(self, out_dir, *, enabled: bool, interval_s: float = 1.0 / HZ,
                 factory: Callable = nvml_read_factory, factory_args: tuple = (),
                 clock: Callable[[], float] = time.monotonic, start_timeout_s: float = 30.0):
        self.dir = Path(out_dir)
        self.enabled, self.interval_s = enabled, interval_s
        self.factory, self.factory_args, self.clock = factory, factory_args, clock
        self.start_timeout_s = start_timeout_s
        self.phases: list = []
        self.problem: Optional[str] = None
        self._proc = None

    @property
    def samples_path(self) -> Path:
        return self.dir / "pricing_clocks.jsonl"

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            ctx = mp.get_context("spawn")
            self._ready, self._stop = ctx.Event(), ctx.Event()
            self._proc = ctx.Process(target=_child, daemon=True, args=(
                str(self.samples_path), self.interval_s, self.factory, self.factory_args,
                self._ready, self._stop))
            self._proc.start()
            if not self._ready.wait(self.start_timeout_s):
                self.problem = "the sampler did not start"
        return self

    @contextlib.contextmanager
    def phase(self, name: str):
        t0 = self.clock()
        try:
            yield
        finally:
            self.phases.append({"name": name, "t0": t0, "t1": self.clock()})

    def __exit__(self, *exc):
        if self._proc is not None:
            self._stop.set()
            self._proc.join(10)
            if self._proc.exitcode not in (0, None) and self.problem is None:
                self.problem = f"the sampler exited with {self._proc.exitcode}"
            if self._proc.is_alive():
                self._proc.kill()
        samples = []
        if self.enabled and self.samples_path.exists():
            samples = [json.loads(line) for line in self.samples_path.read_text().splitlines()
                       if line.strip()]
        record = {"schema": SCHEMA, "sampled": bool(self.enabled and samples),
                  "interval_s": self.interval_s, "problem": self.problem,
                  "phases": summarise(samples, self.phases)}
        (self.dir / "pricing_clocks.json").write_text(json.dumps(record, indent=1))
        return False
