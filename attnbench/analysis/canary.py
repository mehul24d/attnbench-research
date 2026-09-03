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

from .cross_arch import CrossArchError, speedup_within_host

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


@dataclass(frozen=True)
class CanaryDrift:
    """One backend/config whose within-host ratio moved between sessions."""

    gpu_name: str
    backend: str
    config_key: str
    reference_ratio: float
    observed_ratio: float

    @property
    def fractional_change(self) -> float:
        return abs(self.observed_ratio - self.reference_ratio) / self.reference_ratio

    def __str__(self) -> str:
        return (f"{self.gpu_name} {self.backend} {self.config_key}: "
                f"{self.reference_ratio:.4f} -> {self.observed_ratio:.4f} "
                f"({self.fractional_change * 100:.1f}%)")


def canary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The subset of a results frame that belongs to the canary set."""
    if "seq_len" not in df.columns:
        raise CrossArchError(
            "canary selection needs a seq_len column; without it the canary "
            "would silently select nothing and pass")
    return df[df["seq_len"].isin(CANARY_SEQ_LENS)]


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
            f"no canary rows found (expected seq_len in {CANARY_SEQ_LENS}). "
            f"An empty canary that reports success is worse than no canary.")

    speedups = speedup_within_host(rows, baseline_backend=reference_backend)
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for s in speedups:
        grouped.setdefault((s.gpu_name, s.backend, s.config_key), []).append(s.speedup)
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def check_canary_drift(reference: pd.DataFrame, observed: pd.DataFrame, *,
                       tolerance: float = DRIFT_TOLERANCE,
                       reference_backend: str = CANARY_REFERENCE_BACKEND,
                       ) -> list[CanaryDrift]:
    """Compare two sessions' canary measurements, returning what moved.

    Only (gpu_name, backend, config_key) keys present in BOTH frames are
    compared. A key present in one and not the other is not drift -- it is a
    coverage difference, and reporting it as drift would train a reader to
    ignore the output.
    """
    ref = canary_ratios(reference, reference_backend=reference_backend)
    obs = canary_ratios(observed, reference_backend=reference_backend)

    drifts = []
    for key in sorted(set(ref) & set(obs)):
        gpu_name, backend, config_key = key
        d = CanaryDrift(gpu_name=gpu_name, backend=backend, config_key=config_key,
                        reference_ratio=ref[key], observed_ratio=obs[key])
        if d.fractional_change > tolerance:
            drifts.append(d)
    return drifts


def assert_no_canary_drift(reference: pd.DataFrame, observed: pd.DataFrame, *,
                           tolerance: float = DRIFT_TOLERANCE,
                           reference_backend: str = CANARY_REFERENCE_BACKEND,
                           ) -> None:
    """Raise if any canary ratio moved by more than `tolerance`.

    Intended to run at the START of a session, against the previous session's
    results, before any new measurement is recorded -- so a drifting
    environment is found before it has produced a sweep's worth of numbers
    that cannot be joined with what came before.
    """
    drifts = check_canary_drift(reference, observed, tolerance=tolerance,
                                reference_backend=reference_backend)
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
