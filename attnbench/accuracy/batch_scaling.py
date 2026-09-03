"""Batch-size scaling: does batching independent examples buy throughput at
long context, and where does memory stop it?

This exists because the answer turned out to be counter-intuitive and the
reasoning behind the intuitive answer was wrong in two directions at once.

The planning assumption was that Stage 3's grid -- 300 independent examples
per cell, all the same length -- is ideally shaped for batching, and that a
bandwidth-bound decode-style workload would return 3-5x. The analytical
memory model then predicted that memory would be the binding constraint,
capping batch at 1 for the two longest lengths.

Both halves were wrong, and only measurement showed it:

- **Memory was never the constraint.** The model predicted ~14.8 GiB for GLA
  at batch=1/16384. The measured figure is 4.23 GiB. Batch=8 fits on a 24GB
  L4 with room to spare; batch=12 is where it stops.
- **Batching buys no throughput anyway.** Per-example wall time is flat once
  the kernels are warm (2026-09-03, 16383 real tokens: sdpa_flash 1.584 ->
  1.564 s/example across batch 1->2, and 1.543/1.543/1.563/1.613 across
  batch 2->12; GLA 1.384 -> 1.398). At 16K tokens a single example already
  saturates the GPU: the workload is compute-bound, and the weight-reuse
  argument that makes batching pay for short-sequence decode does not apply.
  The weights are already amortized across 16K tokens within one example.

**Warmup is not optional here, and getting it wrong inverts the conclusion.**
The first measured call to a backend pays one-time costs -- CUDA context
setup, cuBLAS handle creation, and for GLA a full Triton JIT compile -- that
have nothing to do with batch size. Measured on 2026-09-03, two *identical*
batch=1 GLA calls in one process took 5.394 s and 1.384 s: a 3.9x difference
between a measurement and its own repeat. Attributing that to the batch axis
produced "gla 4.28x -- batching helps", the exact opposite of the truth, from
data that was otherwise correct. So the first call per backend is measured
and recorded (`warmup=True`) but excluded from any speedup ratio: recording
it keeps the JIT cost visible in the data, and excluding it keeps the cost
out of a number where it would masquerade as a batching effect.

The second point is the load-bearing one, and it is stronger than the memory
argument it replaces. "Batching is blocked by memory" would have been
contingent -- a bigger card, or a batch-aware rewrite of
`model.compute_importance_scores` (which is hard-blocked at batch=1 and whose
`state.scores` carries no batch dimension), might have unlocked it. "Batching
wins nothing because the workload is already compute-bound" closes the
question: the rewrite would have bought nothing at all.

Recorded per (backend, seq_len, batch) with a provenance stamp, because
batch-size behavior is a study target in its own right and an OOM is a
result, not a failed run -- `status` distinguishes the two so a reader can
tell "we measured that this does not fit" from "we never tried it".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class BatchScalingResult:
    """One (backend, length, batch) measurement.

    `seq_len_nominal` is the grid's target length; `seq_len_real` is what the
    tokenizer actually produced. Both are recorded because they differ by
    and conflating them is what made every Stage 3 hour estimate low. Since
    accuracy/sizing.py started fitting contexts to a token budget against
    the real tokenizer, the two should agree to within ~1% -- a row where
    they diverge is a sizing regression worth investigating, which is only
    visible because both are recorded.

    `wall_seconds_per_example` is the number the compute-bound finding rests
    on -- flat across batch means batching returns nothing. It is stored
    rather than left to the reader to divide, so a plot of it cannot silently
    disagree with the conclusion recorded alongside it.
    """

    backend: str
    model_id: str
    seq_len_nominal: int
    seq_len_real: int
    batch: int
    status: str                       # ok | oom | error
    dtype: str = "bfloat16"
    peak_memory_mb: Optional[float] = None
    wall_seconds: Optional[float] = None
    wall_seconds_per_example: Optional[float] = None
    detail: str = ""
    warmup: bool = False              # first call per backend; excluded from ratios

    def __post_init__(self) -> None:
        if self.status not in ("ok", "oom", "error"):
            raise ValueError(f"unknown status {self.status!r} -- an OOM is a "
                              f"recorded result, not an absent one")
        if self.status == "ok" and self.wall_seconds is None:
            raise ValueError("status='ok' requires a wall_seconds measurement")
        if self.batch < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch}")

    def to_dict(self) -> dict:
        return asdict(self)


def per_example_seconds(wall_seconds: float, batch: int) -> float:
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    return wall_seconds / batch


def batching_speedup(results: list[BatchScalingResult]) -> dict[str, float]:
    """Per-example speedup from the smallest to the largest batch that fit,
    per backend. ~1.0 means batching bought nothing.

    Deliberately compares against the smallest *successful* batch rather than
    assuming batch=1 ran: at a length where batch=1 already OOMs, there is no
    baseline and the backend must be absent from the result rather than
    silently compared against something else.

    Rows flagged `warmup` are excluded. A cold first call carries CUDA context
    setup and, for Triton backends, a full JIT compile -- costs that scale
    with nothing on this axis. Including one as the numerator reports that
    one-time cost as a batching speedup; see the module docstring for the
    measurement where doing so inverted the conclusion.

    Backends whose only usable rows are warmup rows are omitted entirely
    rather than reported from a single point, for the same reason the
    smallest-successful-batch rule exists: no baseline, no ratio.
    """
    by_backend: dict[str, list[BatchScalingResult]] = {}
    for r in results:
        if r.status != "ok" or r.wall_seconds_per_example is None:
            continue
        if r.warmup:
            continue
        by_backend.setdefault(r.backend, []).append(r)

    speedups = {}
    for backend, rows in by_backend.items():
        rows.sort(key=lambda r: r.batch)
        if len(rows) < 2:
            continue
        speedups[backend] = (rows[0].wall_seconds_per_example
                             / rows[-1].wall_seconds_per_example)
    return speedups
