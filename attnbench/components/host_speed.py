"""Host-speed probe: a fixed single-thread loop and a null-kernel launch
burst (T2.1; plan Sections 5.3 step 2 and 5.6).

The probe runs at the start, middle and end of every session and between
ground-truth blocks. Its distributions decide two things at lock (T3.2): a
session is usable only if the probe's coefficient of variation is under a
bound, and a component transfers between hosts only if the two hosts' probe
medians differ by less than a bound. Neither bound is set here.

  - **Single-thread loop.** A fixed pure-Python integer loop, so it measures
    the interpreter on this core at this moment: the same work every time,
    nothing allocated, the garbage collector off while it runs.
  - **Null-kernel launch burst.** The host time to issue one null kernel,
    `torch.cuda._sleep(0)`, a single kernel that spins for zero cycles,
    timed launch by launch through PyTorch's Python entry point, which is the
    path every arm's launches take. The device is synchronised before each
    burst, outside the timer, so the launch queue starts empty; the burst
    must stay below the queue depth Q (T2.6) so that no launch waits for a
    free slot, and `null_burst` is recorded on every row so that can be
    checked.

Every function takes its clock, and the burst takes its launch and sync, as
arguments, so the GPU path is exercised on CPU with fakes (the dry run T2.1
asks for). Times are returned in seconds.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from typing import Callable, Optional, Sequence

LOOP_ITERATIONS = 200_000
NULL_KERNEL = "torch.cuda._sleep(0)"


def single_thread_loop(iterations: int = LOOP_ITERATIONS) -> int:
    """The fixed workload. Returns its result so it cannot be optimised out."""
    acc = 0
    for i in range(iterations):
        acc = (acc + i * i) & 0xFFFFFFFF
    return acc


def time_loop(reps: int, iterations: int = LOOP_ITERATIONS, *,
              clock: Callable[[], int] = time.perf_counter_ns,
              loop: Callable[[int], object] = single_thread_loop) -> list[float]:
    """Seconds per loop iteration, one value per repetition."""
    _check_count(reps, "reps")
    _check_count(iterations, "iterations")
    out = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            t0 = clock()
            loop(iterations)
            out.append((clock() - t0) * 1e-9 / iterations)
    finally:
        if was_enabled:
            gc.enable()
    return out


def null_kernel_burst(launch: Callable[[], None], sync: Callable[[], None],
                      burst: int, *,
                      clock: Callable[[], int] = time.perf_counter_ns) -> list[float]:
    """Host seconds per launch for `burst` back-to-back launches.

    `sync` runs before the first launch and after the last, both outside the
    timer: the first so the queue is empty, the second so the next burst
    starts from an idle device.
    """
    _check_count(burst, "burst")
    sync()
    out = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(burst):
            t0 = clock()
            launch()
            out.append((clock() - t0) * 1e-9)
    finally:
        if was_enabled:
            gc.enable()
        sync()
    return out


def cuda_null_kernel():
    """(launch, sync) for the real null kernel on the current CUDA device."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("the null-kernel burst needs a CUDA device")
    return (lambda: torch.cuda._sleep(0)), torch.cuda.synchronize


def summarise(xs: Sequence[float]) -> dict:
    """Median, p10, p90, mean, standard deviation and coefficient of
    variation (stdev / mean). Percentiles are nearest-rank on sorted values."""
    if not xs:
        raise ValueError("nothing to summarise")
    s = sorted(xs)
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if len(s) > 1 else 0.0

    def rank(p):
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]

    return {"n": len(s), "median": statistics.median(s), "p10": rank(0.10),
            "p90": rank(0.90), "mean": mean, "stdev": sd,
            "cv": sd / mean if mean > 0 else None}


def probe(tag: str, *, loop_reps: int = 30, loop_iterations: int = LOOP_ITERATIONS,
          burst: int = 64, bursts: int = 30,
          null_kernel: Optional[tuple] = None, null_path: str = "cuda",
          clock: Callable[[], int] = time.perf_counter_ns) -> dict:
    """One probe row: the loop, then `bursts` bursts of the null kernel.

    `null_kernel` is a (launch, sync) pair and `null_path` names it ("cuda"
    for `cuda_null_kernel()`, anything else for a fake); None means no GPU
    path, and the null-kernel fields are then None with `null_path` "none".
    Every launch
    of every burst enters the distribution, so `null_launch_n` is
    `burst * bursts`.
    """
    _check_count(bursts, "bursts")
    started = time.time()   # not `timestamp`, which is the provenance stamp's
    loop = summarise(time_loop(loop_reps, loop_iterations, clock=clock))
    row = {"tag": tag, "probe_started_at": started,
           "loop_iterations": loop_iterations, "loop_reps": loop_reps}
    row.update({f"loop_s_per_iter_{k}": v for k, v in loop.items() if k != "n"})

    row.update({"null_kernel": NULL_KERNEL, "null_burst": burst,
                "null_bursts": bursts})
    if null_kernel is None:
        row["null_path"] = "none"
        row.update({f"null_launch_s_{k}": None
                    for k in ("median", "p10", "p90", "mean", "stdev", "cv")})
        row["null_launch_n"] = 0
        return row
    launch, sync = null_kernel
    times: list[float] = []
    for _ in range(bursts):
        times.extend(null_kernel_burst(launch, sync, burst, clock=clock))
    null = summarise(times)
    row["null_path"] = null_path
    row.update({f"null_launch_s_{k}": v for k, v in null.items() if k != "n"})
    row["null_launch_n"] = null["n"]
    return row


def _check_count(n, name: str) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} must be an int >= 1, got {n!r}")
