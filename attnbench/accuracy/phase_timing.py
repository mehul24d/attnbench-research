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

    end_to_end ~= prefill + (n_generated - 1) * decode_step  (+ mask_build,
                  once per config, not per call)             (+ scoring, if
                                                              you are honest
                                                              about the
                                                              estimator)

`(n - 1)`, not `n`: the prefill forward emits the first token's logits, so
generating n tokens costs one prefill plus n-1 decode steps. This line read
`n_generated *` until 2026-09-12, three days after `reconcile` below and
`analysis.decode_confound` were both corrected to `(n - 1)`, and it was the
most authoritative-looking of the four places the identity is written down.
The 24-of-24 same-sign residuals it produced fit inside a 10% tolerance and
every cell reported CLOSES -- see silent_failure_patterns #27, and the
standing rule that a tolerance bounds noise and says nothing about bias that
fits inside it.
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


def arms_for(dense_backend: str, sparsities) -> list:
    """The (backend_name, sparsity) list a Stage 5 run measures.

    Here, not in the script, because a test that builds its own arm list is a
    second source of truth for the same fact -- and on 2026-09-07 that is
    exactly what let a three-arm gap survive: the interface test ran
    `[("sdpa_math", None)]` while the script ran dense plus three sparsities,
    so `decode_backend` was never reached and the run died on the instance.

    The asymmetry worth naming: a single-source-of-truth violation in DATA
    produces visibly wrong rows. In TESTS it produces a green suite, which is
    why that one survived where `gate_source` and `backend_role` did not.
    """
    return [(dense_backend, None)] + [("block_sparse", float(sp))
                                       for sp in sparsities]


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
                f"prefill {self.prefill_ms:8.1f} + "
                f"({self.n_generated:5.1f}-1) x "
                f"decode {self.decode_step_ms:6.2f} = {self.implied_total_ms:8.1f}"
                f"  vs observed {self.observed_total_ms:8.1f}"
                f"  ({self.residual_frac:+6.1%})  {mark}")


# How many generated tokens the decode step is fitted over.
#
# Was (1, 8) -- two points -- until 2026-09-08. A two-point slope is
# (T(hi) - T(lo)) / (hi - lo), and T(lo) carries a full prefill, so prefill
# NOISE (not its value, which cancels) lands in the estimate divided by
# (hi - lo). At 8192 that made a dense decode step appear to move 8.5%
# between two clock-locked runs of an arm that had not changed, while dense
# prefill at the same band moved +3.5% the other way. See
# docs/silent_failure_patterns.md #26.
#
# For an OLS slope the noise scales as 1/sqrt(sum((k - kbar)^2)):
#
#     (1, 8)             -> sqrt(24.5)  = 4.95
#     (1, 2, 4, 8, 16)   -> sqrt(148.8) = 12.20     ~2.5x tighter
#
# Two points is the minimum that yields a slope, and the minimum is what
# makes the noise term maximal.
DECODE_FIT_STEPS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class DecodeFit:
    """A least-squares decode step, with the cross-check the two-point
    version could not do.

    T(k) = intercept + k * slope, so the fitted `intercept` is an INDEPENDENT
    estimate of prefill. Comparing it to the separately measured prefill is
    free and catches the case where the linear model does not hold at all --
    which a two-point fit cannot detect, because two points always fit a line
    exactly.
    """

    slope_ms: float
    intercept_ms: float
    r_squared: float
    ks: tuple[int, ...]
    totals_ms: tuple[float, ...]

    def intercept_vs_prefill(self, prefill_ms: float) -> float:
        """Fractional disagreement between the fitted intercept and the
        measured prefill. Near zero means the linear model holds."""
        return (self.intercept_ms - prefill_ms) / prefill_ms

    def render(self) -> str:
        pts = ", ".join(f"{k}:{t:.1f}" for k, t in zip(self.ks, self.totals_ms))
        return (f"OLS over max_new_tokens {list(self.ks)} ({pts} ms), "
                f"R^2={self.r_squared:.4f}, intercept {self.intercept_ms:.1f} ms, "
                f"stops disabled")


def fit_decode_step(totals_by_k: dict[int, float]) -> DecodeFit:
    """Least-squares fit of total time against tokens generated.

    Refuses fewer than three points: two always fit a line exactly, so R^2 is
    1.0 by construction and the intercept cross-check is vacuous. A guard
    that cannot fail is the thing this project keeps finding (#22), and a
    two-point "fit" is that in numerical form.
    """
    if len(totals_by_k) < 3:
        raise ValueError(
            f"need at least 3 points to fit a decode step, got "
            f"{sorted(totals_by_k)}. Two points fit a line exactly: R^2 is "
            f"1.0 whatever the data, and the intercept-vs-prefill check "
            f"cannot fail. See docs/silent_failure_patterns.md #26.")

    ks = tuple(sorted(totals_by_k))
    ys = tuple(float(totals_by_k[k]) for k in ks)
    n = len(ks)
    kbar = sum(ks) / n
    ybar = sum(ys) / n
    sxx = sum((k - kbar) ** 2 for k in ks)
    if sxx == 0:
        raise ValueError("all fit points have the same k; no slope exists")
    slope = sum((k - kbar) * (y - ybar) for k, y in zip(ks, ys)) / sxx
    intercept = ybar - slope * kbar
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * k)) ** 2 for k, y in zip(ks, ys))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return DecodeFit(slope_ms=slope, intercept_ms=intercept, r_squared=r2,
                     ks=ks, totals_ms=ys)


# A residual set that is inside tolerance in every cell and the SAME SIGN in
# every cell is a biased model, not a noisy one. Tolerance bounds noise; it
# says nothing about bias that fits inside it (silent_failure_patterns #27).
#
# Below this p-value the sign pattern is reported as bias. 0.01 is loose on
# purpose: this is a screen, not a hypothesis test anyone acts on directly,
# and the cost of looking is one line.
SIGN_BIAS_P = 0.01


def sign_test_p(residuals) -> float:
    """Two-sided binomial sign test that residuals are symmetric about zero.

    The check that would have caught the 2026-09-08 off-by-one for free. The
    2026-09-07 run produced 24 positive residuals out of 24 -- p = 2^-23
    ~ 1.2e-7 -- while every cell sat inside the 10% tolerance and reported
    CLOSES. The pattern was visible, described as "a small fixed per-call
    overhead", and not pursued.

    Zeros are dropped rather than split, which is the conservative choice:
    an exact zero is evidence for neither sign.
    """
    from math import comb
    nz = [r for r in residuals if r != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = sum(1 for r in nz if r > 0)
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def bias_warning(residuals) -> Optional[str]:
    """A line to print when the residual signs say the model is biased, or
    None. Separate from the per-cell tolerance check on purpose: they answer
    different questions and fail differently."""
    p = sign_test_p(residuals)
    if p >= SIGN_BIAS_P:
        return None
    nz = [r for r in residuals if r != 0]
    pos = sum(1 for r in nz if r > 0)
    direction = "OVER" if pos > len(nz) / 2 else "UNDER"
    mean = sum(nz) / len(nz)
    return (f"!! SIGN BIAS: {max(pos, len(nz) - pos)}/{len(nz)} residuals have "
            f"the same sign (p={p:.2g}), mean {mean:+.1f} ms. The identity "
            f"{direction}STATES systematically. Every cell can sit inside "
            f"tolerance and the model still be wrong -- a tolerance bounds "
            f"noise, not bias that fits inside it. See "
            f"docs/silent_failure_patterns.md #27.")


def reconcile(*, prefill_ms: float, decode_step_ms: float, n_generated: float,
               observed_total_ms: float, context_length: int, backend: str,
               sparsity: Optional[float] = None,
               tolerance: float = RECONCILE_TOLERANCE) -> Reconciliation:
    """Check `prefill + (n - 1) * decode_step` against an end-to-end total.

    THE (n - 1) IS NOT A FUDGE. The prefill forward produces the logits for
    the FIRST generated token, so generating n tokens costs one prefill plus
    n-1 decode steps. `run_measured(..., logits_to_keep=1)` measures exactly
    that prefill, and HF generation works the same way.

    Corrected 2026-09-08, after the OLS decode fit's intercept cross-check
    fired (silent_failure_patterns #27). The old form used `n *
    decode_step`, which overstates every implied total by exactly one decode
    step -- ~35 ms here. The 2026-09-07 run's 24 reconciliations were
    **24 positive residuals out of 24**, mean +50.3 ms, and every one still
    landed inside the 10% tolerance. A tolerance loose enough to absorb a
    systematic bias reports CLOSES on a wrong identity, which is why the
    intercept check -- a different question asked of the same data -- is
    what found it.

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
    implied = prefill_ms + (n_generated - 1) * decode_step_ms
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
                  fit_steps: tuple[int, ...] = DECODE_FIT_STEPS,
                  synchronize=lambda: None, clocks_locked: bool = False,
                  backend_factory=None, decode_backend_factory=None):
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
    if decode_backend_factory is None:
        # block_sparse REFUSES to generate without one -- sparsity is
        # prefill-only and decode runs dense over the cache, and the choice
        # changes what a row means so nothing falls back silently. Stage 3
        # resolves it with this same helper; not calling it is what killed
        # the 2026-09-07 re-run on the instance.
        from .grid_configs import decode_backend_for as decode_backend_factory

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
        dec = decode_backend_factory(be)
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
                    layer_scores=layer_scores, decode_backend=dec,
                    eos_token_ids=frozenset(), newline_token_ids=frozenset(),
                    whitespace_token_ids=frozenset()),
                warmup=max(1, warmup - 1), reps=max(2, reps // 2),
                synchronize=synchronize)

        totals_by_k = {}
        for k in fit_steps:
            g = _gen(k)
            totals_by_k[k] = sum(g) / len(g)
        fit = fit_decode_step(totals_by_k)
        step = fit.slope_ms
        rows.append(summarize([step], backend=name, sparsity=sparsity,
                               context_length=band, phase="decode_step",
                               n_warmup=warmup, clocks_locked=clocks_locked,
                               detail=fit.render()))

        # The fitted intercept is an independent estimate of prefill, so this
        # costs nothing and catches a linear model that does not hold. A
        # two-point fit could not do it: two points fit a line exactly.
        drift = fit.intercept_vs_prefill(prefill_ms[(name, sparsity)])
        if abs(drift) > 0.10:
            print(f"  !! {name} {sparsity}: fitted intercept "
                  f"{fit.intercept_ms:.1f} ms disagrees with measured prefill "
                  f"{prefill_ms[(name, sparsity)]:.1f} ms by {100 * drift:+.1f}%"
                  f" -- total time is not linear in tokens generated here, so "
                  f"the decode 'step' is not one number.", flush=True)

        hi_k = max(fit_steps)
        recs.append(reconcile(prefill_ms=prefill_ms[(name, sparsity)],
                               decode_step_ms=step, n_generated=hi_k,
                               observed_total_ms=totals_by_k[hi_k],
                               context_length=band,
                               backend=name, sparsity=sparsity))
        key = (name, sparsity, band)
        if key in observed:
            obs, ngen = observed[key]
            recs.append(reconcile(prefill_ms=prefill_ms[(name, sparsity)],
                                   decode_step_ms=step, n_generated=ngen,
                                   observed_total_ms=obs, context_length=band,
                                   backend=name, sparsity=sparsity))

    return rows, recs, scoring_ms, prefill_ms
