"""Stage 5: where the time actually goes, per phase, per operating point.

WHAT THIS EXISTS TO SETTLE

Stage 6 measured block_sparse ~30% slower end-to-end than dense at matched
accuracy. It could not say WHERE that lands, because Stage 3's `latency_ms`
is one wall-clock number around `generate()`. An attempt to decompose it did
not close: 62.1 ms/token measured at 8192, against a 55.5 ms slope-derived
decode step and ~580 ms of implied prefill (docs/limitations.md, "An open
discrepancy"). Three numbers, no consistent story, and this project's history
says such a mismatch is a regime error rather than an arithmetic one.

So the phases are measured SEPARATELY here rather than inferred by
subtraction, and `reconcile()` checks them against a known end-to-end total
at the terminal, immediately -- not in analysis a day later.

FOUR PHASES, AND WHY MASK BUILD IS ONE OF THEM

  scoring     the dense importance pass. EXCLUDED from every latency number
              in this study by deliberate decision, which is the same move
              the study criticises in LongCA-bench. Measuring it does not
              undo the exclusion; it converts "the estimator's cost is
              excluded" into "the estimator costs N x the attention it
              saves, and is excluded". A reader can then price the caveat
              instead of taking it on trust. Reported in its own column and
              NEVER folded into a latency figure.

  mask_build  converting importance scores into the backend's mask
              representation. Its own phase because tests/test_timed_region_
              setup.py shows block_sparse rebuilding this inside the timed
              region: charging it to every prefill would overstate a cost a
              real deployment pays once and reuses across calls and layers.
              Attributing it to prefill is how flex once read 4.22 TFLOPS
              against a true 40.72.

  prefill     the forward over the prompt, mask already built.

  decode_step one generation step against a populated cache. Bandwidth-bound
              (a batch-1 step reads all 3.09 GB of weights), so it must never
              be priced from a compute throughput -- see timing_probe.

The composition a reader wants is then explicit rather than assumed:

    end_to_end ~= prefill + n_generated * decode_step        (+ mask_build,
                  once per config, not per call)             (+ scoring, if
                                                              you are honest
                                                              about the
                                                              estimator)
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Callable, Optional

Phase = str   # "scoring" | "mask_build" | "prefill" | "decode_step"

PHASES = ("scoring", "mask_build", "prefill", "decode_step")

# Phases that are real per-call costs of producing a token. `scoring` and
# `mask_build` are deliberately absent: one is the excluded estimator, the
# other is paid once per config. Keeping the set explicit stops a future
# caller summing every phase into an "end to end" number that double-counts.
PER_CALL_PHASES = ("prefill", "decode_step")


@dataclass(frozen=True)
class PhaseMeasurement:
    """One phase, one operating point, aggregated over repeats."""

    backend: str
    sparsity: Optional[float]
    context_length: int
    phase: Phase
    ms_mean: float
    ms_median: float
    ms_stdev: float
    n_warmup: int
    n_reps: int
    clocks_locked: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def time_repeated(fn: Callable[[], None], *, warmup: int = 3, reps: int = 10,
                   synchronize: Callable[[], None] = lambda: None,
                   clock: Callable[[], float] = None) -> list[float]:
    """Run `fn` `warmup` times untimed, then `reps` times timed. Returns
    per-rep milliseconds.

    `synchronize` is called inside the timed region on both sides of `fn`,
    not around the whole loop: a device that queues work would otherwise let
    every rep but the last report a kernel launch. Forgetting it does not
    raise, it just reports the wrong thing in whichever direction flatters
    the backend that queues most.

    Warmup is untimed and its samples are DISCARDED rather than averaged in.
    On 2026-09-07 the dense arm's first 50 rows at 8192 averaged 1911 ms
    against 1663 for the rest -- a 15% first-touch effect that would have
    silently biased any small-rep measurement.
    """
    import time as _time
    if clock is None:
        clock = _time.perf_counter
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")

    for _ in range(warmup):
        fn()
    synchronize()

    out = []
    for _ in range(reps):
        synchronize()
        t0 = clock()
        fn()
        synchronize()
        out.append((clock() - t0) * 1000.0)
    return out


def summarize(samples: list[float], *, backend: str, sparsity: Optional[float],
               context_length: int, phase: Phase, n_warmup: int,
               clocks_locked: bool, detail: str = "") -> PhaseMeasurement:
    if not samples:
        raise ValueError(f"no samples for {phase} -- refusing to report a "
                         f"measurement with nothing behind it")
    return PhaseMeasurement(
        backend=backend, sparsity=sparsity, context_length=context_length,
        phase=phase, ms_mean=statistics.fmean(samples),
        ms_median=statistics.median(samples),
        ms_stdev=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        n_warmup=n_warmup, n_reps=len(samples), clocks_locked=clocks_locked,
        detail=detail)


# --------------------------------------------------------------------------
# The reconciliation. Pure, and the reason this session exists.
# --------------------------------------------------------------------------

# A warmed micro-benchmark and an un-warmed end-to-end mean over 900 rows
# differ for real reasons -- first-touch, allocator state, per-example token
# count variation. 10% is wide enough that exceeding it means the phases do
# not compose, rather than that they were measured on different days.
RECONCILE_TOLERANCE = 0.10


@dataclass(frozen=True)
class Reconciliation:
    """Do the measured phases compose into the observed end-to-end time?"""

    context_length: int
    backend: str
    sparsity: Optional[float]
    prefill_ms: float
    decode_step_ms: float
    n_generated: float
    implied_total_ms: float
    observed_total_ms: float
    residual_ms: float
    residual_frac: float
    closes: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        mark = "CLOSES" if self.closes else "DOES NOT CLOSE"
        sp = "dense" if self.sparsity is None else f"{self.sparsity:g}"
        return (f"{self.context_length:>6}  {self.backend:<13} {sp:>5}  "
                f"prefill {self.prefill_ms:8.1f} + {self.n_generated:5.1f} x "
                f"decode {self.decode_step_ms:6.2f} = {self.implied_total_ms:8.1f}"
                f"  vs observed {self.observed_total_ms:8.1f}"
                f"  ({self.residual_frac:+6.1%})  {mark}")


def reconcile(*, prefill_ms: float, decode_step_ms: float, n_generated: float,
               observed_total_ms: float, context_length: int, backend: str,
               sparsity: Optional[float] = None,
               tolerance: float = RECONCILE_TOLERANCE) -> Reconciliation:
    """Check `prefill + n * decode_step` against a known end-to-end total.

    `mask_build` and `scoring` are deliberately NOT in the sum. Neither is
    paid per generated token: mask build is per config, and the scoring pass
    is excluded from every latency number in the study. Including either
    would make the identity close for the wrong reason.

    A residual outside `tolerance` is not something to reconcile by adjusting
    an input until it fits. It means one of the three numbers measures
    something other than what it is being used for, which is the failure this
    project has made three times (kernel vs whole-model TFLOPS; GLA's 1.7
    against FA2's 61.8; prefill TFLOPS applied to decode).
    """
    if n_generated <= 0:
        raise ValueError(
            f"n_generated={n_generated}: a total implied by zero decode steps "
            f"is just the prefill, and comparing it to an end-to-end number "
            f"would report a residual that is entirely the decode phase.")
    implied = prefill_ms + n_generated * decode_step_ms
    residual = implied - observed_total_ms
    frac = residual / observed_total_ms if observed_total_ms else float("inf")
    return Reconciliation(
        context_length=context_length, backend=backend, sparsity=sparsity,
        prefill_ms=prefill_ms, decode_step_ms=decode_step_ms,
        n_generated=n_generated, implied_total_ms=implied,
        observed_total_ms=observed_total_ms, residual_ms=residual,
        residual_frac=frac, closes=abs(frac) <= tolerance)


def scoring_overhead_ratio(scoring_ms: float, dense_prefill_ms: float,
                            sparse_prefill_ms: float) -> float:
    """How many times the saved attention the estimator costs.

    The denominator is what sparsity SAVED on prefill (dense minus sparse),
    not the sparse prefill itself: the question a reader is asking is whether
    the estimator costs more than the thing it enables. A non-positive
    denominator means sparsity saved nothing on prefill, and the ratio is
    undefined rather than infinite -- reported as such.
    """
    saved = dense_prefill_ms - sparse_prefill_ms
    if saved <= 0:
        return float("nan")
    return scoring_ms / saved


# --------------------------------------------------------------------------
# The measured body, here rather than in the script.
#
# Both failures of the 2026-09-07 Stage 5 session were signature errors --
# `cache_dir=None`, and `scores[0]` where `scores[0][0]` was wanted. Both
# were in a script body that no test could reach without a GPU and a
# downloaded model, so "tested on CPU" covered the timing protocol and the
# reconciliation arithmetic and touched neither real interface.
#
# Extracted so a CPU test drives THIS function against a toy model at tiny
# shapes. A test that merely calls the same methods in its own code would
# re-encode the same assumption; the point is that there is one body and the
# test runs it.
# --------------------------------------------------------------------------

def measure_band(wrapped, input_ids, *, band: int, arms, cfg_for,
                  scratch_dir: str, observed=None, warmup: int = 3,
                  reps: int = 10, scoring_reps: int = 3,
                  gen_lo: int = 1, gen_hi: int = 8,
                  synchronize=lambda: None, clocks_locked: bool = False,
                  backend_factory=None):
    """Measure every phase for one band. Returns (rows, reconciliations).

    `arms` is a list of (backend_name, sparsity). `cfg_for(band, mask,
    sparsity)` builds the AttnConfig. `backend_factory(name)` returns a
    backend instance -- injected so a CPU test can hand back an SDPA/naive
    backend without importing the grid's registry choices.

    `mask_build` is deliberately NOT measured here. It was a phase added for
    interest, it is excluded from the reconciliation identity anyway (nothing
    paid per token belongs there), and it was the only thing that failed on
    the sparse path. Measuring it needs the correct per-head scores indexing,
    which is a question to answer with a test that instantiates the real
    objects -- not on a rented instance.
    """
    import itertools

    if backend_factory is None:
        from .grid_configs import backend_instance as backend_factory

    rows, recs = [], []
    observed = observed or {}

    # Scoring: unique example_id per call so every rep is a real cache MISS.
    # Reusing one id hits after the first call and times a disk read -- ~0 ms
    # reported for the number that prices the study's largest caveat.
    counter = itertools.count()
    sc = time_repeated(
        lambda: wrapped.compute_importance_scores(
            input_ids, task="phase_probe",
            example_id=f"b{band}_{next(counter)}", cache_dir=scratch_dir),
        warmup=1, reps=scoring_reps, synchronize=synchronize)
    rows.append(summarize(sc, backend="dense_softmax_fp32", sparsity=None,
                           context_length=band, phase="scoring", n_warmup=1,
                           clocks_locked=clocks_locked,
                           detail="excluded from every latency number in the "
                                  "study; measured so the exclusion can be "
                                  "priced rather than trusted"))
    scoring_ms = rows[-1].ms_mean
    scores = wrapped.compute_importance_scores(
        input_ids, task="phase_probe", example_id=f"b{band}_final",
        cache_dir=scratch_dir)

    prefill_ms = {}
    for name, sparsity in arms:
        be = backend_factory(name)
        cfg = cfg_for(band, "causal" if sparsity is None else "block_sparse", sparsity)
        layer_scores = None if sparsity is None else scores

        pf = time_repeated(
            lambda: wrapped.run_measured(input_ids, be, cfg=cfg,
                                          layer_scores=layer_scores,
                                          logits_to_keep=1),
            warmup=warmup, reps=reps, synchronize=synchronize)
        rows.append(summarize(pf, backend=name, sparsity=sparsity,
                               context_length=band, phase="prefill",
                               n_warmup=warmup, clocks_locked=clocks_locked,
                               detail="logits_to_keep=1"))
        prefill_ms[(name, sparsity)] = rows[-1].ms_mean

        def _gen(k):
            return time_repeated(
                lambda: wrapped.generate(
                    input_ids, be, cfg=cfg, max_new_tokens=k,
                    layer_scores=layer_scores,
                    eos_token_ids=frozenset(), newline_token_ids=frozenset(),
                    whitespace_token_ids=frozenset()),
                warmup=max(1, warmup - 1), reps=max(2, reps // 2),
                synchronize=synchronize)

        g_lo, g_hi = _gen(gen_lo), _gen(gen_hi)
        lo, hi = sum(g_lo) / len(g_lo), sum(g_hi) / len(g_hi)
        step = (hi - lo) / (gen_hi - gen_lo)
        rows.append(summarize([step], backend=name, sparsity=sparsity,
                               context_length=band, phase="decode_step",
                               n_warmup=warmup, clocks_locked=clocks_locked,
                               detail=f"slope over max_new_tokens {gen_lo}->{gen_hi} "
                                      f"({lo:.1f} -> {hi:.1f} ms), stops disabled"))

        recs.append(reconcile(prefill_ms=prefill_ms[(name, sparsity)],
                               decode_step_ms=step, n_generated=gen_hi,
                               observed_total_ms=hi, context_length=band,
                               backend=name, sparsity=sparsity))
        key = (name, sparsity, band)
        if key in observed:
            obs, ngen = observed[key]
            recs.append(reconcile(prefill_ms=prefill_ms[(name, sparsity)],
                                   decode_step_ms=step, n_generated=ngen,
                                   observed_total_ms=obs, context_length=band,
                                   backend=name, sparsity=sparsity))

    return rows, recs, scoring_ms, prefill_ms
