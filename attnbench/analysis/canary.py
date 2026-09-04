"""A small fixed cell set re-measured every session, to detect drift nobody
introduced on purpose.

Every other integrity check in this project defends against a mistake someone
made: a mask that means something different after a fix, a segment joined
twice, a metric that timed the compiler. Those are catchable by inspection
because there is a change to inspect. This one is different -- it catches a
driver update, a machine image rebuild, a differently-binned GPU of the same
model, a thermal regime. Nothing in the repository changed, and the numbers
moved anyway.

**Why ratios and not latencies.** The obvious canary compares this session's
raw milliseconds against last session's. That is invalid here for exactly the
reason `cross_arch` exists: each session rents a *different physical machine*,
so a raw comparison across sessions cannot separate real drift from ordinary
hardware variation, and it would fire constantly or never depending on the
tolerance chosen. Instead the canary compares each backend's ratio against a
reference backend measured on the same machine in the same session. A ratio
is normalised by its own host, so it is stable across machines of the same
architecture and moves only when the *relationship* between two kernels
changes -- which is what drift actually looks like.

**What a firing canary means.** Not necessarily that anything is wrong: an L4
and an H100 legitimately have different ratios, which is the study's whole
thesis. It means results from before and after are not interchangeable, and
something must explain why. Comparisons are therefore grouped by `gpu_name`,
and a comparison across architectures is refused rather than reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .cross_arch import CrossArchError, _both_locked, speedup_within_host

# The canary set: cheap, short, and deliberately unchanging.
#
# Short lengths only. The canary runs every session, so it must cost minutes,
# not hours; and drift in a kernel's behaviour shows up at 1024 as readily as
# at 32768 while costing ~1000x less. Long-context behaviour is what the
# actual sweep measures.
#
# These values must NOT be edited to make a failing canary pass. Changing the
# set silently resets the baseline, which is the one thing that would make
# this whole mechanism decorative.
CANARY_SEQ_LENS = (1024, 4096)
CANARY_REFERENCE_BACKEND = "sdpa_flash"

# Fractional change in a within-host ratio that counts as drift. 5% is well
# outside run-to-run noise for a warmed kernel timed at 30 reps (the observed
# spread between repeated identical measurements on 2026-09-03 was under 2%),
# and well inside the magnitude of a real regression.
DRIFT_TOLERANCE = 0.05

# A canary cell is only usable if its REFERENCE measurement is long enough that
# DRIFT_TOLERANCE is resolvable. This is not a widened tolerance -- the
# tolerance is unchanged. It excludes cells the instrument cannot measure to
# the required precision, which is a different thing and the opposite of
# permissive: it makes a firing canary mean something.
#
# Measured on 2026-09-04, comparing segment 1 against the segment 2 sweep on
# two rented L4s with identical driver, torch, triton and clock state:
#
#     reference latency   cells   median |change|   max |change|
#     < 3 ms                 50        1.2 %           89.7 %
#     3 - 5 ms               21        4.8 %           41.0 %
#     5 - 10 ms              11        0.4 %            5.6 %
#     > 20 ms                47        1.5 %            8.6 %
#
# `sdpa_flash` -- the canary's own reference, and therefore the denominator of
# every canary ratio -- has a median latency of 2.31 ms at these lengths and a
# median run-to-run change of 4.0%, with a maximum of 17.8%. A ratio built on
# that denominator cannot resolve 5% drift no matter how stable the numerator
# is, and on 2026-09-04 it produced 12 "drifts" of which none were
# environmental: the slow backends at the same configs moved 0.2-1.5%.
#
# 10 ms is chosen from the table, not for roundness: it is where the maximum
# excursion first falls to the same order as the tolerance itself.
CANARY_MIN_LATENCY_MS = 10.0


@dataclass(frozen=True)
class CanaryDrift:
    """One backend/config whose within-host ratio moved between sessions."""

    gpu_name: str
    backend: str
    config_key: str
    reference_ratio: float
    observed_ratio: float
    # False if either session measured this ratio on unlocked clocks.
    clocks_locked: Optional[bool] = None

    @property
    def fractional_change(self) -> float:
        return abs(self.observed_ratio - self.reference_ratio) / self.reference_ratio

    def __str__(self) -> str:
        # The lock state is in the drift line itself, not in a footnote. This
        # string is what a session sees when the canary fires, and it is the
        # moment someone decides whether to investigate an environment change
        # or shrug. DRIFT_TOLERANCE is 5%, which is the same order as
        # unlocked-clock variance, so a marginal drift on unlocked clocks may
        # be nothing at all -- and a reader who is not told that will either
        # chase a phantom or, worse, learn to widen the tolerance.
        note = "" if self.clocks_locked is not False else "  [CLOCKS UNLOCKED]"
        return (f"{self.gpu_name} {self.backend} {self.config_key}: "
                f"{self.reference_ratio:.4f} -> {self.observed_ratio:.4f} "
                f"({self.fractional_change * 100:.1f}%){note}")


def canary_rows(df: pd.DataFrame, *,
                min_latency_ms: float = CANARY_MIN_LATENCY_MS,
                reference_backend: str = CANARY_REFERENCE_BACKEND,
                ) -> pd.DataFrame:
    """The subset of a results frame that belongs to the canary set.

    Restricted to cells whose REFERENCE measurement clears
    `min_latency_ms` -- see that constant for why. The filter is on the
    reference, not on each backend's own latency, because the reference is the
    denominator of every ratio at that config: a numerator measured to 0.2%
    against a denominator measured to 18% still gives a ratio measured to 18%.
    """
    if "seq_len" not in df.columns:
        raise CrossArchError(
            "canary selection needs a seq_len column; without it the canary "
            "would silently select nothing and pass")
    rows = df[df["seq_len"].isin(CANARY_SEQ_LENS)]
    if min_latency_ms <= 0 or "latency_ms_p50" not in rows.columns:
        return rows

    ok = rows[(rows["backend"] == reference_backend)
              & (rows["latency_ms_p50"] >= min_latency_ms)]
    return rows[rows["config_key"].isin(set(ok["config_key"]))]


def canary_resolution_report(df: pd.DataFrame, *,
                             min_latency_ms: float = CANARY_MIN_LATENCY_MS,
                             reference_backend: str = CANARY_REFERENCE_BACKEND,
                             ) -> str:
    """How many canary configs the instrument can actually resolve, and why.

    Printed rather than inferred, because a canary that quietly shrinks to two
    cells and reports "no drift" is the failure this whole module is built to
    avoid.
    """
    allrows = df[df["seq_len"].isin(CANARY_SEQ_LENS)]
    refs = allrows[allrows["backend"] == reference_backend]
    if "latency_ms_p50" not in allrows.columns or refs.empty:
        return "no reference rows; resolution unknown"
    keep = refs[refs["latency_ms_p50"] >= min_latency_ms]
    return (f"{len(keep)} of {len(refs)} canary configs have a "
            f"{reference_backend} reference at or above {min_latency_ms:g} ms "
            f"(median {refs['latency_ms_p50'].median():.2f} ms). Configs below "
            f"it cannot resolve a {DRIFT_TOLERANCE * 100:.0f}% tolerance and "
            f"are excluded.")


def canary_ratios(df: pd.DataFrame, *,
                  reference_backend: str = CANARY_REFERENCE_BACKEND,
                  ) -> dict[tuple[str, str, str], float]:
    """(gpu_name, backend, config_key) -> mean within-host ratio.

    Built on `speedup_within_host`, so it inherits that function's refusal to
    form a ratio across machines. Averaging across hosts of the same
    architecture is valid precisely because each ratio was already normalised
    by its own machine's reference measurement.
    """
    rows = canary_rows(df)
    if rows.empty:
        raise CrossArchError(
            f"no usable canary rows (expected seq_len in {CANARY_SEQ_LENS} "
            f"with a {reference_backend} reference at or above "
            f"{CANARY_MIN_LATENCY_MS:g} ms). "
            f"{canary_resolution_report(df, reference_backend=reference_backend)} "
            f"An empty canary that reports success is worse than no canary, so "
            f"this refuses instead. If every config is below the floor, the "
            f"canary set needs a config where the reference is slow enough to "
            f"measure -- a larger batch, not a lower floor.")

    speedups = speedup_within_host(rows, baseline_backend=reference_backend)
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for s in speedups:
        grouped.setdefault((s.gpu_name, s.backend, s.config_key), []).append(s.speedup)
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def canary_clock_state(df: pd.DataFrame, *,
                       reference_backend: str = CANARY_REFERENCE_BACKEND,
                       ) -> dict[tuple[str, str, str], Optional[bool]]:
    """(gpu_name, backend, config_key) -> were BOTH sides' clocks locked.

    Separate from `canary_ratios` rather than folded into its return type, so
    the ratio dict keeps the shape every existing caller expects. False if any
    contributing measurement was taken unlocked; None where it is unknown.
    """
    speedups = speedup_within_host(canary_rows(df),
                                   baseline_backend=reference_backend)
    out: dict[tuple[str, str, str], Optional[bool]] = {}
    for s in speedups:
        key = (s.gpu_name, s.backend, s.config_key)
        out[key] = _both_locked(out.get(key, True), s.clocks_locked)
    return out


@dataclass(frozen=True)
class BaselineChange:
    """A deliberate change to what a backend does inside the timed region.

    The canary asks "did the environment move?" and answers it by comparing a
    backend's ratio across sessions. That question is only meaningful while the
    backend is computing the same thing both times. When we change the timed
    region ourselves, the ratio moves for a reason the canary was never meant
    to detect, and firing would be a false alarm -- the kind that teaches
    people to ignore the check.

    Silently excluding the backend would be worse. So an exclusion requires an
    entry here: a backend, the commit that changed it, and a written reason.
    `rebased_backends` in the check refuses any name that has no entry, which
    makes "exclude whatever is firing" impossible to do casually.
    """

    backend: str
    commit: str
    date: str
    reason: str


BASELINE_CHANGES: tuple[BaselineChange, ...] = (
    BaselineChange(
        backend="flex", commit="d62d392", date="2026-09-04",
        reason="Mask construction hoisted out of the timed region, and "
               "block_sparse pinned to kernel_options BLOCK_M=BLOCK_N=64. "
               "seq_len=1024/batch=1 moved 4.22 -> 40.72 useful TFLOPS "
               "(9.65x) on the same L4; the shift is largest at the smallest "
               "cells and ~1.0x at the largest, which is a removed constant "
               "addend, not an environment change."),
    BaselineChange(
        backend="naive", commit="0463528", date="2026-09-04",
        reason="Causal mask and block_sparse dense conversion hoisted out of "
               "the timed region, found by tests/test_timed_region_setup.py "
               "after the flex case. Expected to be small -- naive is O(S^2) "
               "itself, so this is a bounded factor rather than an addend -- "
               "but it is a change to the timed region and is recorded as "
               "one rather than assumed negligible."),
)


def _rebase_reason(backend: str) -> str:
    for c in BASELINE_CHANGES:
        if c.backend == backend:
            return f"{c.commit} ({c.date}): {c.reason}"
    return ""


def check_canary_drift(reference: pd.DataFrame, observed: pd.DataFrame, *,
                       tolerance: float = DRIFT_TOLERANCE,
                       reference_backend: str = CANARY_REFERENCE_BACKEND,
                       rebased_backends: frozenset[str] = frozenset(),
                       ) -> list[CanaryDrift]:
    """Compare two sessions' canary measurements, returning what moved.

    Only (gpu_name, backend, config_key) keys present in BOTH frames are
    compared. A key present in one and not the other is not drift -- it is a
    coverage difference, and reporting it as drift would train a reader to
    ignore the output.

    `rebased_backends` names backends whose timed region we changed on
    purpose between the two sessions. Every name must have a `BaselineChange`
    entry, or this raises: the escape hatch is only usable by someone who has
    already written down what changed and why.
    """
    unjustified = sorted(b for b in rebased_backends if not _rebase_reason(b))
    if unjustified:
        raise CrossArchError(
            f"cannot exclude {unjustified} from the canary: no BaselineChange "
            f"entry. If the timed region really did change, record it in "
            f"canary.BASELINE_CHANGES with the commit and the reason. If it "
            f"did not, this is environmental drift and excluding it would "
            f"hide exactly what the canary exists to find.")

    ref = canary_ratios(reference, reference_backend=reference_backend)
    obs = canary_ratios(observed, reference_backend=reference_backend)
    ref_locked = canary_clock_state(reference, reference_backend=reference_backend)
    obs_locked = canary_clock_state(observed, reference_backend=reference_backend)

    drifts = []
    for key in sorted(set(ref) & set(obs)):
        gpu_name, backend, config_key = key
        if backend in rebased_backends:
            continue
        d = CanaryDrift(gpu_name=gpu_name, backend=backend, config_key=config_key,
                        reference_ratio=ref[key], observed_ratio=obs[key],
                        # Unlocked in EITHER session makes the comparison
                        # between them unlocked: the drift is a difference of
                        # two ratios, and it inherits the noise of both.
                        clocks_locked=_both_locked(ref_locked.get(key),
                                                   obs_locked.get(key)))
        if d.fractional_change > tolerance:
            drifts.append(d)
    return drifts


def assert_no_canary_drift(reference: pd.DataFrame, observed: pd.DataFrame, *,
                           tolerance: float = DRIFT_TOLERANCE,
                           reference_backend: str = CANARY_REFERENCE_BACKEND,
                           rebased_backends: frozenset[str] = frozenset(),
                           ) -> None:
    """Raise if any canary ratio moved by more than `tolerance`.

    Intended to run at the START of a session, against the previous session's
    results, before any new measurement is recorded -- so a drifting
    environment is found before it has produced a sweep's worth of numbers
    that cannot be joined with what came before.
    """
    drifts = check_canary_drift(reference, observed, tolerance=tolerance,
                                reference_backend=reference_backend,
                                rebased_backends=rebased_backends)
    if not drifts:
        return
    raise CrossArchError(
        f"{len(drifts)} canary ratio(s) moved by more than "
        f"{tolerance * 100:.0f}%:\n  " + "\n  ".join(str(d) for d in drifts) +
        "\n\nNothing in the repository has to have changed for this to fire: a "
        "driver update, a rebuilt machine image, or a differently-binned GPU of "
        "the same model will do it. Results from before and after are not "
        "interchangeable until the cause is identified. Do NOT widen the "
        "tolerance or edit CANARY_SEQ_LENS to make this pass -- that resets the "
        "baseline and makes the check decorative."
    )
