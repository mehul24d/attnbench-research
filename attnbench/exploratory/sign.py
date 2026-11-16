"""EXPLORATORY: how certain is each cell's dense-versus-sparse sign (item 5
of the 16 Nov 2026 rulings), from the timings already banked.

A paired bootstrap over rounds. Within one ground-truth file the arms are
timed interleaved, once each per round, so sample i of every row is round i;
pairing by round keeps the round-to-round drift that dense and sparse share
out of the difference. Each replicate resamples round indices with
replacement, the same indices for dense and sparse, and takes median(sparse)
minus median(dense). Rows pair only when the file kept every round it ran
(rounds_run == clean_rounds) and both rows have that many samples.

Reported per cell: the observed difference, the percentile interval, the
share of replicates in which sparse is faster, and two verdicts: the sign
(the interval excludes 0) and the sign beyond the cell's resolution floor
(the interval excludes the band +/- floor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


class PairingError(ValueError):
    pass


@dataclass(frozen=True)
class Sign:
    n_rounds: int
    diff_s: float               # median(sparse) - median(dense), observed
    lo_s: float
    hi_s: float
    p_sparse_faster: float      # share of replicates with diff < 0
    sign: str                   # "sparse faster", "dense faster", "undecided"
    beyond_floor: str           # the same, against +/- floor


def _verdict(lo: float, hi: float, band: float) -> str:
    if hi < -band:
        return "sparse faster"
    if lo > band:
        return "dense faster"
    return "undecided"


def paired(dense: Sequence[float], sparse: Sequence[float], *, floor_s: float,
           n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> Sign:
    d, s = np.asarray(dense, float), np.asarray(sparse, float)
    if d.ndim != 1 or d.shape != s.shape or len(d) < 2:
        raise PairingError(f"rounds differ or too few: {d.shape} vs {s.shape}")
    if not 0 < level < 1 or n_boot < 1 or floor_s < 0:
        raise ValueError("level in (0, 1), n_boot >= 1, floor >= 0")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    reps = np.median(s[idx], axis=1) - np.median(d[idx], axis=1)
    a = (1 - level) / 2
    lo, hi = (float(x) for x in np.quantile(reps, [a, 1 - a]))
    return Sign(n_rounds=len(d), diff_s=float(np.median(s) - np.median(d)), lo_s=lo, hi_s=hi,
                p_sparse_faster=float(np.mean(reps < 0)),
                sign=_verdict(lo, hi, 0.0), beyond_floor=_verdict(lo, hi, floor_s))


def pairable(result: dict) -> Optional[str]:
    """Why a ground-truth file's rows cannot be paired by round, or None."""
    if result.get("rounds_run") != result.get("clean_rounds"):
        return (f"rounds_run {result.get('rounds_run')} != clean_rounds "
                f"{result.get('clean_rounds')}: a dropped round breaks the pairing")
    n = {len(r["prefill_ms_samples"]) for r in result.get("rows", [])}
    if len(n) != 1 or n != {result["clean_rounds"]}:
        raise PairingError(f"rows hold {sorted(n)} samples, not {result['clean_rounds']}")
    return None
