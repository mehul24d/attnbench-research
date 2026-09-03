"""Timing and memory measurement.

Uses triton's do_bench where available since it is the convention the kernel
papers use, which keeps our absolute numbers comparable to published ones. Falls
back to a CUDA-event loop with the same warmup and quantile handling.

Medians and IQR throughout. Kernel timing distributions have long right tails
and a mean will quietly encode one unlucky launch.
"""

from __future__ import annotations

import gc
import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import torch

from . import compile_guard
from .config import AttnConfig

WARMUP = 10
REPS = 30


@dataclass
class Measurement:
    ok: bool
    # ok | unsupported | oom | error | compile_fallback
    #
    # `compile_fallback` is a REFUSAL, not a failure of the kernel. It means
    # torch.compile gave up and ran the function eagerly, so a latency here
    # would time a different implementation than the one named in `backend`.
    # See compile_guard.py -- this is the condition that made Stage 1 certify
    # flex block-sparse 72/72 while Stage 2 could not run a single cell.
    status: str
    detail: str = ""
    latency_ms_p50: Optional[float] = None
    latency_ms_p25: Optional[float] = None
    latency_ms_p75: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    # issued: throughput against the dense-equivalent (causal-respecting,
    # sparsity-ignoring) FLOP count -- comparable across sparsity levels and
    # directly against a dense backend's number at the same shape.
    issued_tflops: Optional[float] = None
    # useful: throughput against the FLOPs required under the *declared*
    # sparsity budget. A bookkeeping split derived from config, not a
    # measurement of whether the kernel actually skipped that work -- it
    # cannot show sparsity conversion by itself. Whether a kernel converts
    # sparsity into real speed is read off by comparing measured latency
    # across sparsity levels (or profiler counters), not from this column
    # alone. Equal to issued_tflops for every non-sparse config.
    useful_tflops: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _do_bench(fn: Callable[[], None], warmup: int, reps: int):
    try:
        from triton.testing import do_bench
        return do_bench(fn, warmup=warmup, rep=reps,
                        quantiles=[0.5, 0.25, 0.75])
    except Exception:
        pass

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    q = statistics.quantiles(times, n=4) if len(times) >= 4 else [times[0]] * 3
    return statistics.median(times), q[0], q[2]


def measure(backend, cfg: AttnConfig, mask=None,
            warmup: int = WARMUP, reps: int = REPS) -> Measurement:
    """Time one (backend, config) cell and capture peak memory.

    Every failure mode is caught and classified rather than raised. A backend
    that OOMs at 32K is data, not a crash, and the sweep must keep going.
    """
    from .backends.base import UnsupportedConfig

    ok, why = backend.claims_support(cfg)
    if not ok:
        return Measurement(False, "unsupported", why)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Inputs are allocated once, outside the timed closure, and reused
        # across warmup and every rep. `run_once`-per-rep would fold fresh
        # `torch.randn` allocation into each latency sample.
        q, k, v = backend.make_inputs(cfg)

        # Warm the allocator and any JIT before the peak-memory read, so we
        # measure steady state rather than compilation. This is also where a
        # torch.compile fallback would happen, so it is where we watch: a
        # fallback caught here costs one warmup call, whereas discovering it
        # after `_do_bench` means having run 30 reps of the eager O(S^2) path,
        # which at long seq_len is slow enough to matter and can OOM.
        with compile_guard.guard() as cg:
            backend.timed_call(q, k, v, cfg, mask=mask)
            torch.cuda.synchronize()
        if cg.fell_back:
            return Measurement(False, "compile_fallback", cg.detail)

        with compile_guard.guard() as cg:
            p50, p25, p75 = _do_bench(
                lambda: backend.timed_call(q, k, v, cfg, mask=mask), warmup, reps)
        if cg.fell_back:
            return Measurement(False, "compile_fallback", cg.detail)

        peak = torch.cuda.max_memory_allocated() / 1e6
        issued = backend.issued_flops(cfg) / (p50 * 1e-3) / 1e12
        useful = backend.useful_flops(cfg) / (p50 * 1e-3) / 1e12

        return Measurement(True, "ok",
                           latency_ms_p50=float(p50),
                           latency_ms_p25=float(p25),
                           latency_ms_p75=float(p75),
                           peak_memory_mb=round(peak, 2),
                           issued_tflops=round(issued, 2),
                           useful_tflops=round(useful, 2))

    except UnsupportedConfig as e:
        return Measurement(False, "unsupported", str(e)[:300])
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return Measurement(False, "oom", str(e)[:300])
    except RuntimeError as e:
        msg = str(e)
        kind = "oom" if "out of memory" in msg.lower() else "error"
        torch.cuda.empty_cache()
        return Measurement(False, kind, msg[:300])
    except Exception as e:
        return Measurement(False, "error", f"{type(e).__name__}: {e}"[:300])
    finally:
        gc.collect()
        torch.cuda.empty_cache()
