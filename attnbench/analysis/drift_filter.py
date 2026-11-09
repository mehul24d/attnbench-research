"""The dense control's drift filter, applied after the session (Appendix A,
9 Nov 2026; adviser's ruling 1 of the sixth paste).

In session, `ground_truth.time_arms` flags a round whose dense time differs
from the dense control's warm-up time by more than the session's absolute
resolution floor, and keeps every round: `rounds[*]` holds each round's drift
and flag reasons, and each row its clean samples and its flagged samples, both
in round order. This module re-reads that record under another filter without
touching the timed path:

  - `RECORDED`: the flags as the session wrote them;
  - `RELATIVE`: a round is drift-flagged when |drift| / warm-up time exceeds
    `limit` (`relative_limit` derives it from other sessions' rounds);
  - `NONE`: no drift filter.

A throttle flag (any reason that is not the dense drift) is kept under every
filter. A group is complete when its clean rounds reach `min_clean`, the reps
it was run with unless the caller names a minimum (a lock value, T4.1 row 23).
"""

from __future__ import annotations

import copy
import statistics
from typing import Iterable, Mapping, Optional

from attnbench.ground_truth import quantile

RECORDED, RELATIVE, NONE = "recorded", "relative", "none"
FILTERS = (RECORDED, RELATIVE, NONE)
DRIFT_PREFIX = "dense_drift_"


class DriftFilterError(ValueError):
    pass


def relative_drifts(result: Mapping) -> list[float]:
    """|dense drift| / the dense control's warm-up time, one per round."""
    ref = float(result["dense_control"]["reference_ms"])
    return [abs(float(r["dense_drift_ms"])) / ref for r in result["rounds"]]


def relative_limit(results: Iterable[Mapping], *, q: float) -> float:
    """The `q` quantile of `relative_drifts` over every round of `results`."""
    xs = [x for r in results for x in relative_drifts(r)]
    if not xs:
        raise DriftFilterError("no rounds to derive a drift limit from")
    return quantile(xs, q)


def round_samples(row: Mapping, rounds: list) -> dict[int, float]:
    """{round: ms} for one row, from its clean and flagged samples in round order."""
    clean = list(row["prefill_ms_samples"])
    flagged = list(row["flagged_ms_samples"])
    n_flagged = sum(bool(r["flagged"]) for r in rounds)
    if len(clean) + len(flagged) != len(rounds) or len(flagged) != n_flagged:
        raise DriftFilterError(
            f"arm {row.get('arm')!r}: {len(clean)} clean and {len(flagged)} flagged samples "
            f"do not match {len(rounds)} rounds with {n_flagged} flagged")
    out, ic, if_ = {}, iter(clean), iter(flagged)
    for r in rounds:
        out[r["round"]] = next(if_) if r["flagged"] else next(ic)
    return out


def _reasons(rnd: Mapping, mode: str, ref_ms: float, limit: Optional[float]) -> list[str]:
    kept = [x for x in rnd["flag_reasons"] if not x.startswith(DRIFT_PREFIX)]
    drift = float(rnd["dense_drift_ms"])
    if mode == RECORDED:
        return list(rnd["flag_reasons"])
    if mode == RELATIVE and abs(drift) / ref_ms > limit:
        kept.append(f"{DRIFT_PREFIX}{drift / ref_ms:+.4f}_vs_relative_{limit}")
    return kept


def refilter(result: Mapping, *, mode: str, limit: Optional[float] = None,
             min_clean: Optional[int] = None) -> dict:
    """A copy of one ground-truth result with its rounds flagged by `mode`,
    each row's samples and statistics recomputed over the unflagged rounds,
    and `complete` set against `min_clean` (default: the reps)."""
    if mode not in FILTERS:
        raise DriftFilterError(f"unknown filter {mode!r}; one of {FILTERS}")
    if (mode == RELATIVE) != (limit is not None):
        raise DriftFilterError("a relative limit is given with the relative filter only")
    out = copy.deepcopy(dict(result))
    rounds = out["rounds"]
    per_row = [round_samples(row, result["rounds"]) for row in result["rows"]]
    ref = float(out["dense_control"]["reference_ms"])
    for rnd in rounds:
        rnd["flag_reasons"] = _reasons(rnd, mode, ref, limit)
        rnd["flagged"] = bool(rnd["flag_reasons"])
    flagged = {r["round"] for r in rounds if r["flagged"]}
    clean = len(rounds) - len(flagged)
    for row, by_round in zip(out["rows"], per_row):
        ms = [v for rd, v in sorted(by_round.items()) if rd not in flagged]
        row.update(clean_rounds=len(ms), prefill_ms_samples=ms,
                   flagged_ms_samples=[v for rd, v in sorted(by_round.items()) if rd in flagged],
                   prefill_ms_median=statistics.median(ms) if ms else None,
                   prefill_ms_p90=quantile(ms, 0.90) if ms else None,
                   prefill_ms_p99=quantile(ms, 0.99) if ms else None)
    need = int(min_clean if min_clean is not None else out["rows"][0]["reps"])
    out.update(clean_rounds=clean, complete=clean >= need,
               drift_filter={"mode": mode, "relative_limit": limit, "min_clean": need,
                             "recorded_clean_rounds": result["clean_rounds"]})
    return out
