"""The crossover of plan Section 5.4 (T2.19), and its error (H3).

**The bracket rule (primary).** "The crossover is reported as the bracket
between adjacent grid lengths where the sign of sparse minus dense time
changes, widened by the bootstrap interval of the difference at each
endpoint; the same rule is applied to predicted and to measured times."
Read here as: at each grid length the difference sparse - dense has a
point value and a bootstrap interval (intervals.py). Its sign is
*determined* when the interval excludes zero and the point lies outside
the resolution floor (Section 5.6: "a location inside the resolution floor
is not interpreted"); otherwise it is undetermined. A bracket runs from the
last determined length of one sign to the first determined length of the
other, so undetermined lengths between them widen it. Two crossovers are
possible, so every sign change is a bracket. With no bracket, the result
is "none in range" only when every length is determined; otherwise it is
"unresolved".

**The quadratic (secondary).** "A quadratic fit a·L² + b·L + c per arm is
a secondary point estimate, reported with its residual (a form that fits
dense well and sparse poorly is itself a finding)." Each arm's times are
fitted by ordinary least squares, unweighted, and the crossovers are the
real roots of the sparse fit minus the dense fit inside the grid's range.
The fit runs on L / L_max, which keeps the 32768² column from swamping the
constant; the coefficients are reported per token.

**Crossover error.** H3 asks whether the measured crossover "falls inside
the grid interval predicted". With the same number of brackets on both
sides, each pair is compared in order: whether the measured bracket lies
inside the predicted one, whether they overlap, whether the direction
agrees, and the shift between their centres in grid steps (the grid
doubles, so a step is a factor of two in length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .recursion import CostModelError

SPARSE_BECOMES_FASTER = "sparse becomes faster"   # sign + to -
SPARSE_BECOMES_SLOWER = "sparse becomes slower"   # sign - to +

CROSSOVER = "crossover"
NONE_IN_RANGE = "none in range"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class GridDifference:
    """Sparse minus dense end-to-end time at one grid length, seconds."""

    length: int
    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class Bracket:
    lower: int        # grid length
    upper: int
    lower_index: int  # position in the grid
    upper_index: int
    direction: str


@dataclass(frozen=True)
class CrossoverResult:
    status: str
    brackets: tuple[Bracket, ...]
    lengths: tuple[int, ...]
    signs: tuple[int, ...]    # +1 sparse slower, -1 sparse faster, 0 undetermined


def _check_floor(floor) -> float:
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) \
            or not math.isfinite(floor) or floor < 0:
        raise CostModelError(
            f"the resolution floor is a time >= 0 (fixed at lock), got {floor!r}")
    return float(floor)


def sign_at(d: GridDifference, floor: float) -> int:
    floor = _check_floor(floor)
    if not d.lo <= d.point <= d.hi:
        raise CostModelError(
            f"at {d.length} the point {d.point} is outside its interval "
            f"[{d.lo}, {d.hi}]")
    if d.lo > 0 and d.point > floor:
        return 1
    if d.hi < 0 and d.point < -floor:
        return -1
    return 0


def _check_grid(lengths: Sequence[int]) -> None:
    if len(lengths) < 2:
        raise CostModelError("a crossover needs at least two grid lengths")
    if any(b <= a for a, b in zip(lengths, lengths[1:])):
        raise CostModelError(f"grid lengths must increase: {list(lengths)}")


def crossover_brackets(diffs: Sequence[GridDifference], *,
                       floor: float) -> CrossoverResult:
    lengths = tuple(int(d.length) for d in diffs)
    _check_grid(lengths)
    signs = tuple(sign_at(d, floor) for d in diffs)
    determined = [(i, s) for i, s in enumerate(signs) if s != 0]
    brackets = []
    for (i, s), (j, t) in zip(determined, determined[1:]):
        if s != t:
            brackets.append(Bracket(
                lower=lengths[i], upper=lengths[j], lower_index=i,
                upper_index=j,
                direction=SPARSE_BECOMES_FASTER if s > 0 else SPARSE_BECOMES_SLOWER))
    if brackets:
        status = CROSSOVER
    elif len(determined) == len(signs):
        status = NONE_IN_RANGE
    else:
        status = UNRESOLVED
    return CrossoverResult(status=status, brackets=tuple(brackets),
                           lengths=lengths, signs=signs)


# --------------------------------------------------------------------------
# The quadratic
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadraticFit:
    a: float          # seconds per token^2
    b: float          # seconds per token
    c: float          # seconds
    scale: float      # L_max, the fit's internal unit
    lengths: tuple[int, ...]
    residuals: tuple[float, ...]       # measured minus fitted, seconds
    rms_residual: float
    max_abs_relative_residual: float

    def __call__(self, L: float) -> float:
        x = L / self.scale
        a, b, c = self._scaled()
        return (a * x + b) * x + c

    def _scaled(self) -> tuple[float, float, float]:
        s = self.scale
        return self.a * s * s, self.b * s, self.c


def fit_quadratic(lengths: Sequence[int], times: Sequence[float]) -> QuadraticFit:
    lengths = tuple(int(L) for L in lengths)
    _check_grid(lengths)
    if len(lengths) < 3:
        raise CostModelError("a quadratic needs at least three grid lengths")
    y = np.asarray(times, dtype=float)
    if y.shape != (len(lengths),) or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise CostModelError("times must be positive, finite and one per length")
    scale = float(max(lengths))
    x = np.asarray(lengths, dtype=float) / scale
    A = np.column_stack([x * x, x, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([a, b, c])
    return QuadraticFit(
        a=float(a) / (scale * scale), b=float(b) / scale, c=float(c),
        scale=scale, lengths=lengths, residuals=tuple(float(r) for r in resid),
        rms_residual=float(np.sqrt(np.mean(resid ** 2))),
        max_abs_relative_residual=float(np.max(np.abs(resid) / y)))


def _real_roots(a: float, b: float, c: float) -> list[float]:
    if a == 0:
        return [] if b == 0 else [-c / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
    roots = [q / a]
    if q != 0:
        roots.append(c / q)
    return sorted(set(roots))


def quadratic_crossovers(sparse: QuadraticFit,
                         dense: QuadraticFit) -> tuple[float, ...]:
    """Lengths in the shared grid range where the two fits cross."""
    if sparse.lengths != dense.lengths:
        raise CostModelError("the two arms were fitted on different grids")
    s, d = sparse._scaled(), dense._scaled()
    # Parallel or identical fits (a = b = 0) have no single crossing, and
    # _real_roots returns none for them.
    a, b, c = (s[0] - d[0], s[1] - d[1], s[2] - d[2])
    lo = sparse.lengths[0] / sparse.scale
    out = [r * sparse.scale for r in _real_roots(a, b, c) if lo <= r <= 1.0]
    return tuple(out)


# --------------------------------------------------------------------------
# Crossover error (H3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BracketComparison:
    predicted: Bracket
    measured: Bracket
    contains: bool          # measured bracket inside the predicted interval
    overlaps: bool
    direction_agrees: bool
    shift_grid_steps: float  # measured centre minus predicted centre


@dataclass(frozen=True)
class CrossoverError:
    status: str   # "compared", "both none in range", "missed", "spurious",
                  # "count differs", or "unresolved"
    comparisons: tuple[BracketComparison, ...] = ()


def crossover_error(predicted: CrossoverResult,
                    measured: CrossoverResult) -> CrossoverError:
    if predicted.lengths != measured.lengths:
        raise CostModelError("predicted and measured use different grids")
    if UNRESOLVED in (predicted.status, measured.status):
        return CrossoverError("unresolved")
    np_, nm = len(predicted.brackets), len(measured.brackets)
    if np_ == nm == 0:
        return CrossoverError("both none in range")
    if np_ == 0:
        return CrossoverError("missed")
    if nm == 0:
        return CrossoverError("spurious")
    if np_ != nm:
        return CrossoverError("count differs")
    comps = []
    for p, m in zip(predicted.brackets, measured.brackets):
        comps.append(BracketComparison(
            predicted=p, measured=m,
            contains=p.lower_index <= m.lower_index and m.upper_index <= p.upper_index,
            overlaps=m.lower_index < p.upper_index and p.lower_index < m.upper_index,
            direction_agrees=p.direction == m.direction,
            shift_grid_steps=((m.lower_index + m.upper_index)
                              - (p.lower_index + p.upper_index)) / 2))
    return CrossoverError("compared", tuple(comps))


def differences_from_intervals(lengths: Sequence[int], intervals: Sequence,
                               ) -> list[GridDifference]:
    """GridDifferences from `intervals.Interval`s of sparse - dense."""
    if len(lengths) != len(intervals):
        raise CostModelError("one interval per grid length")
    return [GridDifference(length=int(L), point=iv.point, lo=iv.lo, hi=iv.hi)
            for L, iv in zip(lengths, intervals)]


def bracket_of(result: CrossoverResult) -> Optional[Bracket]:
    """The single bracket, or None; refuses two (ask for `brackets`)."""
    if len(result.brackets) > 1:
        raise CostModelError("two crossovers; read result.brackets")
    return result.brackets[0] if result.brackets else None
