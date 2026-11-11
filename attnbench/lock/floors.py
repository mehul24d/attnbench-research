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
from typing import Callable, Mapping, Optional, Sequence

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
        `lock.hdbi_fit` take, by `ms_to_s`."""
        return ms_to_s(self.floor_ms)


def ms_to_s(ms: float) -> float:
    """The only ms-to-seconds conversion of the lock tools."""
    return ms / MS_PER_S


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


def resolution_floors_by_length(series: Mapping, *, length_of: Callable, quantile: float,
                                pairing: str) -> dict:
    """The resolution floor per length (Appendix A, 8 Nov 2026): for each
    (context length, batch), `resolution_floor` over that cell's series
    only, `length_of(key)` giving a series' (length, batch). A difference
    at a cell is read against that cell's floor."""
    groups: dict = {}
    for k, xs in series.items():
        groups.setdefault(length_of(k), {})[k] = xs
    if not groups:
        raise FloorsError("no series")
    return {g: resolution_floor(s, quantile=quantile, pairing=pairing)
            for g, s in sorted(groups.items())}



def cell_label(length: int, batch: int) -> str:
    """"<length>x<batch>", the floors' key in files (`gate_a_repeats.cell_floors`)."""
    return f"{int(length)}x{int(batch)}"


@dataclass(frozen=True)
class FloorTable:
    """The resolution floor per (context length, batch), in seconds: the
    lock's table, from P1 only (Appendix A, 8 Nov 2026; T4.1; adviser's
    ruling 1a of 11 Nov 2026). Every consumer that scores or decides across
    cells takes one of these, never one scalar for every length (T4.3 read
    A, finding 4).

    The table is keyed by the token budget, as the floors are computed
    (`gate_a_repeats.cell_key`); a prediction is keyed by its realised token
    count, the budget a label (lock table row 27). **Plan default,
    flagged:** a realised length reads the floor of the smallest tabled
    budget at or above it at the same batch (1951 -> 2048, 16370 -> 16384);
    a length above every budget, at or below half the budget it would read
    (the grid doubles), or at a batch the table lacks, is refused."""

    floors_s: Mapping             # {(budget, batch): seconds}
    pairing: str = FLOOR_PAIRING

    def __post_init__(self) -> None:
        _pairing(self.pairing)
        if not isinstance(self.floors_s, Mapping) or not self.floors_s:
            raise FloorsError("a floor table holds at least one (length, batch)")
        clean = {}
        for k, v in self.floors_s.items():
            if not (isinstance(k, tuple) and len(k) == 2
                    and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in k)):
                raise FloorsError(f"a floor table key is (length, batch), got {k!r}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                    or v < 0:
                raise FloorsError(f"the floor at {k} is a time >= 0 in seconds, got {v!r}")
            clean[k] = float(v)
        object.__setattr__(self, "floors_s", dict(sorted(clean.items())))

    def budget_of(self, length, batch) -> int:
        """The tabled budget a realised `length` at `batch` reads (class docstring)."""
        budgets = sorted(L for (L, b) in self.floors_s if b == int(batch))
        if not budgets:
            raise FloorsError(f"the floor table has no batch {batch}")
        for L in budgets:
            if int(length) <= L:
                if int(length) <= L // 2:      # the grid doubles: no tabled budget holds it
                    raise FloorsError(f"length {length} has no tabled budget at batch {batch} "
                                      f"({budgets}); the next is {L}")
                return L
        raise FloorsError(f"length {length} is above every tabled budget at batch {batch} "
                          f"({budgets})")

    def at(self, length, batch) -> float:
        """The floor, seconds, of the cell at realised `length` and `batch`."""
        return self.floors_s[(self.budget_of(length, batch), int(batch))]

    def to_json(self) -> dict:
        return {"pairing": self.pairing,
                "floors_s": {cell_label(L, b): v for (L, b), v in self.floors_s.items()}}

    @classmethod
    def from_json(cls, d: Mapping) -> "FloorTable":
        try:
            return cls(floors_s={_parse_label(k): v for k, v in d["floors_s"].items()},
                       pairing=d["pairing"])
        except (KeyError, TypeError, AttributeError) as e:
            raise FloorsError(f"unreadable floor table: {e!r}") from e

    @classmethod
    def from_ms(cls, table: Mapping, *, pairing: str) -> "FloorTable":
        """From {"<length>x<batch>": ms}, the lock value's form."""
        if not isinstance(table, Mapping):
            raise FloorsError(f"the resolution floors are a table, got {table!r}")
        out = {}
        for k, v in table.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise FloorsError(f"the floor at {k!r} is not a time in ms: {v!r}")
            out[_parse_label(k)] = ms_to_s(v)
        return cls(floors_s=out, pairing=pairing)

    @classmethod
    def from_lock(cls, lock: Mapping) -> "FloorTable":
        """The lock file's `resolution_and_noise_floors` value:
        {"pairing": "P1", "resolution_floor_ms": {"<L>x<b>": ms}, ...}."""
        try:
            v = lock["values"]["resolution_and_noise_floors"]["value"]
            return cls.from_ms(v["resolution_floor_ms"], pairing=v["pairing"])
        except (KeyError, TypeError) as e:
            raise FloorsError(f"the lock holds no resolution floor table: {e!r}") from e


def _parse_label(k) -> tuple:
    import re
    m = re.fullmatch(r"(\d+)x(\d+)", str(k))
    if not m:
        raise FloorsError(f"a floor key is '<length>x<batch>', got {k!r}")
    return int(m.group(1)), int(m.group(2))


def as_table(floors, what: str = "the resolution floor") -> FloorTable:
    """`floors` if it is a table; a single number is refused (one scalar for
    every length is the defect the table replaces)."""
    if isinstance(floors, FloorTable):
        return floors
    raise FloorsError(f"{what} is a per-(length, batch) table (`FloorTable`, from the lock), "
                      f"not {floors!r}")

@dataclass(frozen=True)
class CellFloor:
    floor_ms: float                   # max(within_ms, between_ms)
    within_ms: float                  # `resolution_floor` over the cell's series
    between_ms: Optional[float]       # None: no series at the cell ran in two sessions
    n_between_series: int
    source: str                       # "within" or "between", whichever is larger


def resolution_floors_within_between(series: Mapping, *, length_of: Callable,
                                     quantile: float, pairing: str) -> dict:
    """The resolution floor per (context length, batch) since the amendment
    of 8 Nov 2026 (fifth paste; Appendix A): the larger of the within-session
    spread (`resolution_floors_by_length`'s floor at the cell) and the
    between-session spread, the `quantile` over the cell's series run in two
    or more sessions of the range (max - min) of the series' per-session
    medians. A key's first element is its session and the rest names the
    series across sessions; `length_of(key)` gives its (length, batch).
    {(L, b): CellFloor}"""
    q = _quantile(quantile)
    within = resolution_floors_by_length(series, length_of=length_of, quantile=q,
                                         pairing=pairing)
    medians: dict = {}
    for k, xs in series.items():
        session, cell = k[0], tuple(k[1:])
        medians.setdefault(length_of(k), {}).setdefault(cell, {})[session] = \
            statistics.median(_series(k, xs))
    out = {}
    for g, w in within.items():
        ranges = [max(m.values()) - min(m.values()) for m in medians[g].values() if len(m) > 1]
        between = float(np.quantile(ranges, q)) if ranges else None
        floor = w.floor_ms if between is None else max(w.floor_ms, between)
        out[g] = CellFloor(floor_ms=floor, within_ms=w.floor_ms, between_ms=between,
                           n_between_series=len(ranges),
                           source="between" if between is not None and between > w.floor_ms
                           else "within")
    return out


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
