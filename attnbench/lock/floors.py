"""The resolution floor and the noise floor from banked data (Section 5.6;
T4.1; Appendix A, 25 Oct 2026: "fixed at lock from the measured run-to-run
spread").

The plan fixes where the floors come from, not the formula. What each one is
used for decides its unit:

  - the **resolution floor** is a time. A measured sparse-minus-dense
    difference inside it is not interpreted (decision error, sign agreement,
    the crossover rule), and the dense control's drift beyond it flags a
    round (T2.13). It is in milliseconds, like `ground_truth.Controls`.
  - the **noise floor** is a relative error. A prediction error inside it is
    not interpreted (`predict.residuals`' `relative_floor`). It is a
    fraction of the measured time.

**The procedure here** (proposed; the statistic is a lock decision, so each
function takes its quantile with no default):

  - resolution floor: for every series of clean repeats of one cell in one
    session (an arm's `prefill_ms_samples` from `ground_truth.time_arms`),
    the absolute deviation of each repeat from the series median, pooled
    over every series; the floor is the q-quantile of the pool, in ms.
  - noise floor: for every cell timed in two or more of the P1 repeat
    sessions (T3.2), the spread of its per-session medians, (max - min) /
    median of the medians; the floor is the q-quantile over cells. The
    within-session relative spread (each series' (max - min) / median) is
    reported beside it, since Section 5.6 asks for both as numbers.

**Researcher's decisions (28 Oct 2026)**, to be dated in Appendix A by the
integration session:

  - *Source.* The floors come from P1 sessions only (the dense controls and
    the repeated runs), never from a held-out pairing. Each function takes
    `pairing` with no default and refuses any pairing but `FLOOR_PAIRING`;
    the result records it.
  - *Units.* Ground truth is in milliseconds; `predict.scoring` and
    `lock.hdbi_fit` take the resolution floor in seconds, the unit of the
    times they score. `ResolutionFloor.floor_s` is the one conversion, and
    callers use it, not `floor_ms / 1000` of their own.
  - *Quantile.* A lock value. `SHAKEDOWN_QUANTILE` (0.95) is provisional,
    for the shakedowns only; the final value is fixed at lock, before any
    held-out prediction is scored. No function defaults to it.

Quantiles are numpy's linear method. Every result records the procedure, the
quantile, the counts, and the sha256 of its inputs in canonical form, so a
lock value can be traced to the data it was computed from.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


FLOOR_PAIRING = "P1"            # the only pairing the floors are computed from
SHAKEDOWN_QUANTILE = 0.95       # provisional, shakedowns only; the final value is a lock value
MS_PER_S = 1000.0


class FloorsError(ValueError):
    """Banked data the floors cannot be computed from."""


def _pairing(p) -> str:
    if p != FLOOR_PAIRING:
        raise FloorsError(f"the floors come from {FLOOR_PAIRING} sessions only, "
                          f"never a held-out pairing; got {p!r}")
    return p


def _quantile(q) -> float:
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q <= 1:
        raise FloorsError(f"the quantile is a lock decision in (0, 1], got {q!r}")
    return float(q)


def _series(name, xs) -> list[float]:
    out = []
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) \
                or x <= 0:
            raise FloorsError(f"{name}: {x!r} is not a positive finite time")
        out.append(float(x))
    return out


def inputs_sha256(data: Mapping) -> str:
    """sha256 of `data` with keys as strings, sorted, values as given."""
    canon = {repr(k): list(map(float, v)) if not isinstance(v, Mapping)
             else {repr(kk): list(map(float, vv)) for kk, vv in sorted(v.items(), key=repr)}
             for k, v in data.items()}
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ResolutionFloor:
    floor_ms: float
    quantile: float
    n_series: int
    n_repeats: int
    inputs_sha256: str
    pairing: str = FLOOR_PAIRING
    procedure: str = ("q-quantile of |repeat - series median|, pooled over every "
                      "session's clean repeats of every cell")

    @property
    def floor_s(self) -> float:
        """The floor in seconds, the unit `predict.scoring` and
        `lock.hdbi_fit` take. The only ms-to-seconds conversion."""
        return self.floor_ms / MS_PER_S


def resolution_floor(series: Mapping, *, quantile: float, pairing: str) -> ResolutionFloor:
    """`series`: {(session, cell): [ms, ...]}, clean repeats of `pairing`'s
    sessions only, which must be P1."""
    pairing = _pairing(pairing)
    q = _quantile(quantile)
    pool = []
    for k, xs in series.items():
        xs = _series(k, xs)
        if len(xs) < 2:
            raise FloorsError(f"{k}: a spread needs at least two repeats, got {len(xs)}")
        med = statistics.median(xs)
        pool.extend(abs(x - med) for x in xs)
    if not pool:
        raise FloorsError("no series")
    return ResolutionFloor(floor_ms=float(np.quantile(pool, q)), quantile=q,
                           n_series=len(series), n_repeats=len(pool),
                           inputs_sha256=inputs_sha256(series), pairing=pairing)


@dataclass(frozen=True)
class NoiseFloor:
    floor_relative: float
    quantile: float
    n_cells: int
    n_sessions: int
    between_session_spread: dict      # cell -> (max - min) / median of medians
    within_session_spread: dict       # (session, cell) -> (max - min) / median
    inputs_sha256: str
    pairing: str = FLOOR_PAIRING
    procedure: str = ("q-quantile over cells of (max - min) / median of the cell's "
                      "per-session medians across the repeat sessions")


def noise_floor(sessions: Mapping, *, quantile: float, pairing: str) -> NoiseFloor:
    """`sessions`: {session: {cell: [ms, ...]}}, the P1 repeat sessions'
    clean repeats. Only cells timed in at least two sessions count; a floor
    from fewer than two sessions is refused. The result is a fraction, with
    no unit to convert."""
    pairing = _pairing(pairing)
    q = _quantile(quantile)
    if len(sessions) < 2:
        raise FloorsError(f"a between-session spread needs two sessions, got {len(sessions)}")
    medians: dict = {}
    within = {}
    for s, cells in sessions.items():
        if not isinstance(cells, Mapping) or not cells:
            raise FloorsError(f"session {s!r} holds no cells")
        for cell, xs in cells.items():
            xs = _series((s, cell), xs)
            if not xs:
                raise FloorsError(f"{(s, cell)}: no repeats")
            med = statistics.median(xs)
            within[(s, cell)] = (max(xs) - min(xs)) / med
            medians.setdefault(cell, []).append(med)
    between = {c: (max(m) - min(m)) / statistics.median(m)
               for c, m in medians.items() if len(m) >= 2}
    if not between:
        raise FloorsError("no cell was timed in two sessions")
    return NoiseFloor(floor_relative=float(np.quantile(list(between.values()), q)),
                      quantile=q, n_cells=len(between), n_sessions=len(sessions),
                      between_session_spread=between, within_session_spread=within,
                      inputs_sha256=inputs_sha256(sessions), pairing=pairing)


def clean_series(result: Mapping) -> dict[str, list[float]]:
    """{arm: clean repeats in ms} from one `ground_truth.time_arms` result.
    Its rows already exclude flagged rounds; a traced result is refused,
    since ground truth is untraced."""
    if result.get("traced"):
        raise FloorsError("a traced result is not ground truth")
    return {r["arm"]: list(r["prefill_ms_samples"]) for r in result["rows"]}
