"""A profile of the reference mask builder on the target host (T2.2; plan
Sections 1, 2 and Appendix A, 19 Oct 2026).

The plan says about 94% of the reference builder's loop time is Python
overhead, from a laptop profile at 32768 tokens that was never banked
(`docs/limitations.md`: 0.212 s for the Python loop body and 0.013 s for the
torch operations inside it, per 10 calls, so (0.212 - 0.013) / 0.212 = 93.9%),
and that the figure is to be re-measured on the target host under T2.2. This
module is that measurement. The builder is `masks.importance_block_mask`; it
runs on the host's CPU, so it is profiled on the host the pairing has, with no
GPU involved.

**What is measured**, per sequence length, on `reps` calls of the builder:

  - *unprofiled* wall time per call (`perf_counter`, no profiler running): the
    builder's real cost on this host;
  - *profiled*: the same calls under `torch.profiler` (CPU activity), giving
    the wall time of the profiled block and the self CPU time of every aten
    operator in it. The torch-operator time is the sum of those self times, so
    an operator nested in another is not counted twice;
  - the Python share of the loop time is `1 - torch_operator_time /
    profiled_wall_time`, both from the one profiled run. That is the quantity
    the 94% was: the loop body less the torch operations inside it, over the
    loop body.

**What the profiler does to the number.** It adds a cost to every operator it
records, so the torch-operator time it reports is above the true one. One minus
that time over the *unprofiled* wall time (`python_share_vs_unprofiled`) is
therefore a lower bound on the Python share, and uses no profiled wall time at
all. The same-pass share (`python_share`) is an estimate, not a bound.
`profiler_dilation` (profiled over unprofiled wall time) says how large the
profiler's own cost was.

**Two instruments, because the answer depends on which.** The builder fills its
mask with about one tensor write per kept block (`active[qb, kvs[j]] = True`),
about 100,000 aten calls per build at 32768. `cProfile` sees only calls the
interpreter makes as calls. A tensor `__setitem__` or `__getitem__` goes through
a type slot, which `cProfile` does not count as a call, so its time lands in
the calling function's own time, and `cprofile_python_share` (one minus the
time in profiled builtins over all profiled time) reads those torch operations
as interpreter overhead. `torch.profiler` sees them as aten operators. On a
loop of tensor slot writes the first reads 1.00 and the second about 0.5
(tests/test_components_builder_profile.py). The two are reported side by side
and neither is preferred here: they answer different questions ("how much of
the time is the Python bytecode and the interpreter's own calls" against "how
much is outside the aten operators"), and the plan's "about 94%" does not say
which it was. On the Mac this was first run on, `cProfile` gave 94.2% at 32768
(0.374 s own time of 0.397 s, per 10 calls), the figure the plan quotes, and
`torch.profiler` gave 0.53 with a lower bound of 0.28.

**The inputs.** The scores are random numbers with a fixed seed: the builder's
loop does the same work on any values at a given length and budget (the
per-row top-k and the writes follow the candidates, not the scores), and no
accuracy is read from this. `sparsity` sets the budget, so it sets the
number of scalar writes; it is recorded. The plan's figure was profiled at a
setting that `limitations.md` does not state, so the script's default is
density 0.5, the load T2.5 proposes for the link microbenchmarks (awaiting the
researcher), and it takes the sparsity as an argument.
"""

from __future__ import annotations

import cProfile
import math
import pstats
import statistics
import time
from typing import Callable, Optional, Sequence

import torch

from .. import masks

SCHEMA = "attnbench.builder_profile/1"
BLOCK_SIZE = 128                  # the timed block size (plan Section 5.3)


def reference_builder(seq_len: int, block_size: int, sparsity: float, *,
                      seed: int = 0) -> Callable[[], object]:
    """A zero-argument call of `masks.importance_block_mask` on fixed random
    scores, as the sparse arms' CPU builder is called."""
    n = -(-seq_len // block_size)
    scores = torch.rand(n, n, generator=torch.Generator().manual_seed(seed))

    def build():
        return masks.importance_block_mask(
            seq_len, block_size, sparsity, scores, causal=True,
            identity_seed=f"builder-profile-{seed}", free_block="sink")
    return build


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def torch_operator_time(prof) -> tuple[float, int]:
    """(seconds, calls): the summed self CPU time of every aten operator the
    profiler recorded, and how many calls that is."""
    events = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    return (sum(e.self_cpu_time_total for e in events) * 1e-6,
            sum(e.count for e in events))


def cprofile_share(build: Callable[[], object], reps: int,
                   clock: Callable[[], float] = time.perf_counter) -> dict:
    """The `cProfile` view (module docstring): one minus the time inside
    profiled builtins over all profiled time, and the cost of the pass."""
    pr = cProfile.Profile()
    t0 = clock()
    pr.enable()
    try:
        for _ in range(reps):
            build()
    finally:
        pr.disable()
    wall = clock() - t0
    rows = pstats.Stats(pr).stats
    total = sum(v[2] for v in rows.values())
    builtin = sum(v[2] for k, v in rows.items() if k[0] == "~")
    calls = sum(v[1] for k, v in rows.items() if k[0] == "~")
    return {"cprofile_python_share": (1.0 - builtin / total) if total > 0 else None,
            "cprofile_s_per_call": wall / reps,
            "cprofile_builtin_s_per_call": builtin / reps,
            "cprofile_builtin_calls_per_call": calls / reps}


def profile_builder(build: Callable[[], object], *, reps: int, warmup: int = 1,
                    clock: Callable[[], float] = time.perf_counter,
                    profiler=None, cprofile: bool = True) -> dict:
    """Profile `build` (module docstring). `profiler` replaces
    `torch.profiler.profile` in tests; it is a context manager whose object has
    `key_averages()`. `cprofile=False` leaves out the `cProfile` pass."""
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be an int >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        build()
    plain = []
    for _ in range(reps):
        t0 = clock()
        build()
        plain.append(clock() - t0)
    if profiler is None:
        from torch.profiler import ProfilerActivity, profile
        profiler = profile(activities=[ProfilerActivity.CPU])
    with profiler as prof:
        t0 = clock()
        for _ in range(reps):
            build()
        profiled_wall = clock() - t0
    torch_s, n_calls = torch_operator_time(prof)
    unprofiled = statistics.median(plain)
    per = profiled_wall / reps
    torch_per = torch_s / reps
    cp = cprofile_share(build, reps, clock) if cprofile else {}
    if cp and unprofiled > 0:
        cp["cprofile_dilation"] = cp["cprofile_s_per_call"] / unprofiled
    return {**cp,
        "schema": SCHEMA, "reps": reps, "warmup": warmup,
        "unprofiled_s_per_call_median": unprofiled,
        "unprofiled_s_per_call_p10": _quantile(plain, 0.10),
        "unprofiled_s_per_call_p90": _quantile(plain, 0.90),
        "unprofiled_s_per_call_samples": plain,
        "profiled_s_per_call": per,
        "torch_operator_s_per_call": torch_per,
        "python_overhead_s_per_call": per - torch_per,
        "aten_calls_per_call": n_calls / reps,
        "python_share": (1.0 - torch_per / per) if per > 0 else None,
        "python_share_vs_unprofiled": (1.0 - torch_per / unprofiled)
        if unprofiled > 0 else None,
        "profiler_dilation": per / unprofiled if unprofiled > 0 else None,
        # The plan's units: seconds per 10 calls.
        "loop_body_s_per_10_calls": per * 10,
        "torch_operators_s_per_10_calls": torch_per * 10,
    }


def profile_reference_builder(seq_len: int, *, sparsity: float, reps: int,
                              block_size: int = BLOCK_SIZE, warmup: int = 1,
                              seed: int = 0, **kw) -> dict:
    """One row: the reference builder at `seq_len`."""
    for name, v in (("seq_len", seq_len), ("block_size", block_size)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be an int >= 1, got {v!r}")
    if not (isinstance(sparsity, (int, float)) and not isinstance(sparsity, bool)
            and 0 <= sparsity < 1):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity!r}")
    row = profile_builder(reference_builder(seq_len, block_size, sparsity, seed=seed),
                          reps=reps, warmup=warmup, **kw)
    row.update({"builder": "masks.importance_block_mask", "free_block": "sink",
                "seq_len": seq_len, "block_size": block_size,
                "n_blocks": -(-seq_len // block_size), "sparsity": float(sparsity),
                "density": 1.0 - float(sparsity), "scores": f"random, seed {seed}",
                "torch_version": torch.__version__,
                "torch_threads": torch.get_num_threads()})
    return row
