"""The profitability dispatcher and its comparators (T2.22; plan Sections
4 (H6), 5.1, 5.7 and 5.9).

**What it decides.** For each decision point, a model, a context length and
a batch on one pairing (`training.partner_of`), one configuration: dense, or
a sparse arm at a density. It picks the admissible configuration with the
shortest predicted end-to-end time, and dense unless that configuration is
predicted faster than dense by more than the resolution floor (Section 5.9).
It reads only what a prediction may use on an unmeasured pairing: the
committed predicted times, plus admissibility. Kernel times (device
components, Section 5.1) are inputs of the comparators only.

**The comparators** (Section 5.9; H6):

  - `kernel_speedup_rule` (primary): the admissible sparse configuration whose
    sparse kernel alone beats the selected dense kernel by the most, and dense
    if none beats it. It ignores estimator, host and synchronisation cost,
    which is the point of the comparison.
  - `always_dense`.
  - `best_kernel_config`: one fixed sparse configuration for every point, the
    admissible one with the best geometric-mean kernel speedup over the points
    where it exists. Applied wherever it exists and left unscored where not.
  - the best measured choice on the nearest measured pairing, supplied by
    the caller (`best_measured_choice` computes it from that pairing's
    ground truth).

**Admissibility** is an input and a set: a configuration is `Config(arm,
density)`, dense is always admissible, and the evaluation is run once per
labelled set, so the uncorrected and the Holm-corrected sets (Section 5.7)
are reported side by side. `admissible_at_margin` builds a set from the lower
confidence bounds of the accuracy test, with the rule `paired_exact_ni` uses
(`lower > -margin`), and `margin_frontier` sweeps the margin: for each margin,
whether each configuration is admissible and the speedup the dispatcher buys
on the measured grid (Section 5.7, "margin against speedup"). The per-arm
rule for a failed mask gate (Section 5.1) is applied by leaving the removed
arms out of the set: they are then out of the dispatcher's choices and out
of the reference that regret is measured against, because one set serves
both.

**Scoring** is on the measured ground-truth grid, so it needs no extra GPU
session (Section 5.9):

  - *regret* is the measured time of the chosen configuration over the best
    measured admissible configuration (dense included), minus 1; mean and
    worst case over the scored points;
  - *decision errors*: sparse chosen where dense was measured faster, and
    dense chosen where an admissible sparse configuration was measured
    faster. The same outcomes as `scoring.py`. A point whose measured
    difference between the chosen and the dense configuration (or, for a
    dense choice, the best admissible sparse one and dense) is inside the
    resolution floor is not scored (Appendix A, 26 and 27 Oct 2026);
  - *time gained or lost against always-dense*, and the speedup over it.

H6 is scored at 16384 and 32768 tokens (`H6_LENGTHS`, budget labels: a point
keyed by its realised token count is read at its budget,
`floors.FloorTable.budget_of`) and at batch 1 only (`H6_BATCHES`; Section 4:
"at batch 4 every rule chooses dense, so batch 4 is not scored"). Every
policy is scored on one common set of points, those where dense, the
dispatcher's choice and the kernel-speedup rule's choice were all measured;
a point outside it is listed with the reason, for every policy (adviser's
ruling 2b of 11 Nov 2026; T4.3 read A, findings 1 to 3). A policy with no
choice at a common point (a fixed configuration absent there) is listed
unscored there. With no common point, H6 reads "not scored". When no admissible
sparse configuration is measured faster anywhere, always-dense has zero regret
by construction and the dispatcher can only tie it; `PolicyScore.
dense_best_everywhere` says so and the test reduces to whether the dispatcher
avoids the kernel-speedup rule's losses (Section 4, H6).

**Ties** (nothing in the plan names a rule, so these are proposals): equal
predicted times, or equal kernel speedups, go to the configuration with the
smaller `(arm, density)`. A predicted tie with dense is dense, as everywhere.

The resolution floor is the lock's table per (context length, batch)
(`floors.FloorTable`, from P1 only; Appendix A, 8 Nov 2026): every function
that decides or scores takes the table and reads each point's own floor; a
single number is refused.

Units are seconds. This module imports nothing from `commit.py`, so the
committed file can carry its choices (`DispatchSpec`, `make_choices`) and
the analysis can recompute them from the file's own cells.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import (Callable, Iterable, Mapping, NamedTuple, Optional,
                    Sequence)

from ..lock.floors import FloorTable, FloorsError, as_table
from .scoring import CORRECT, FALSE_PROFITABLE, MISSED_PROFITABLE, NOT_INTERPRETED
from .training import CellKey, partner_of

H6_LENGTHS = (16384, 32768)     # budget labels
H6_BATCHES = (1,)
# The policies whose choices define the common scored set (module docstring).
COMMON_SET = ("dispatcher", "kernel_speedup_rule")
POLICIES = ("dispatcher", "kernel_speedup_rule", "always_dense",
            "best_kernel_config", "nearest_pairing")


class DispatchError(ValueError):
    """Inputs the dispatcher or its scoring cannot use."""


class Config(NamedTuple):
    arm: str
    density: float


class Point(NamedTuple):
    model: str
    context_length: int
    batch: int


@dataclass(frozen=True)
class Candidate:
    """One configuration at one point: its committed predicted time, and
    its kernel-alone time when the comparators are to be run."""

    key: CellKey
    predicted: float
    kernel: Optional[float] = None

    @property
    def config(self) -> Config:
        return Config(self.key.arm, float(self.key.density))


@dataclass(frozen=True)
class PointInputs:
    point: Point
    dense: Candidate
    sparse: tuple[Candidate, ...]

    def candidate(self, config: Config) -> Optional[Candidate]:
        if config == self.dense.config:
            return self.dense
        for c in self.sparse:
            if c.config == config:
                return c
        return None


def _positive(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) \
            or not math.isfinite(x) or x <= 0:
        raise DispatchError(f"{what} must be a positive finite time in "
                            f"seconds, got {x!r}")
    return float(x)


def _floors(floors) -> FloorTable:
    try:
        return as_table(floors)
    except FloorsError as e:
        raise DispatchError(str(e)) from e


def _floor_at(floors: FloorTable, point: Point) -> float:
    try:
        return floors.at(point.context_length, point.batch)
    except FloorsError as e:
        raise DispatchError(f"{tuple(point)}: {e}") from e


def build_points(predicted: Mapping[CellKey, float], *, dense_arms: Iterable[str],
                 kernel: Optional[Mapping[CellKey, float]] = None,
                 use_code_path: Optional[Mapping[str, str]] = None
                 ) -> list[PointInputs]:
    """The decision points in `predicted` (committed times by cell key).

    Each point needs exactly one dense cell. A sparse configuration with two
    cells at a point (one arm on two code paths, as the mask gate runs
    XAttention's two on one card) is refused unless `use_code_path` names
    the one that is deployed, by arm; the others are then left out.
    """
    dense_arms = set(dense_arms)
    if not dense_arms:
        raise DispatchError("name the dense arm(s)")
    kernel = dict(kernel or {})
    use = dict(use_code_path or {})
    by_point: dict = {}
    for k, v in predicted.items():
        k = CellKey(*k)
        by_point.setdefault(Point(*partner_of(k)), []).append(k)
    out = []
    for pt in sorted(by_point):
        keys = by_point[pt]
        dense = [k for k in keys if k.arm in dense_arms]
        if len(dense) != 1:
            raise DispatchError(
                f"{tuple(pt)} has {len(dense)} dense cells; a decision point "
                f"needs exactly one")
        cands: dict = {}
        for k in keys:
            if k.arm in dense_arms:
                continue
            if k.arm in use and k.code_path != use[k.arm]:
                continue
            cfg = Config(k.arm, float(k.density))
            if cfg in cands:
                raise DispatchError(
                    f"{tuple(pt)}: {cfg} has cells on code paths "
                    f"{cands[cfg].key.code_path!r} and {k.code_path!r}; name "
                    f"the deployed one in use_code_path")
            cands[cfg] = Candidate(k, _positive(predicted[k], f"prediction for {tuple(k)}"),
                                   kernel.get(k))
        d = dense[0]
        out.append(PointInputs(
            point=pt,
            dense=Candidate(d, _positive(predicted[d], f"prediction for {tuple(d)}"),
                            kernel.get(d)),
            sparse=tuple(cands[c] for c in sorted(cands))))
    return out


# --------------------------------------------------------------------------
# The dispatcher and its comparators. Each returns a Config, or None where it
# has no choice to make (best_kernel_config at a point it does not exist at).
# --------------------------------------------------------------------------

def _admissible(p: PointInputs, admissible) -> list[Candidate]:
    return [c for c in p.sparse if c.config in admissible]


def dispatcher(p: PointInputs, admissible, floors: FloorTable) -> Config:
    """Section 5.9: the admissible configuration with the shortest predicted
    time, and dense unless it is faster than dense by more than the point's
    floor."""
    floor_s = _floor_at(_floors(floors), p.point)
    adm = _admissible(p, admissible)
    if not adm:
        return p.dense.config
    best = min(adm, key=lambda c: (c.predicted, c.config))
    return best.config if p.dense.predicted - best.predicted > floor_s \
        else p.dense.config


def always_dense(p: PointInputs, admissible=None, floors=None) -> Config:
    return p.dense.config


def _kernel(c: Candidate, p: PointInputs) -> float:
    if c.kernel is None:
        raise DispatchError(
            f"{tuple(c.key)} has no kernel time, which the kernel-speedup "
            f"comparators need (a device component, Section 5.1)")
    return _positive(c.kernel, f"kernel time of {tuple(c.key)}")


def _kernel_speedups(p: PointInputs, admissible) -> list[tuple[float, Candidate]]:
    adm = _admissible(p, admissible)
    if not adm:
        return []
    dk = _kernel(p.dense, p)
    return [(dk / _kernel(c, p), c) for c in adm]


def kernel_speedup_rule(p: PointInputs, admissible, floors=None) -> Config:
    """Section 5.1: use an estimator at its admissible density wherever its
    sparse kernel alone beats the selected dense kernel; among those, the
    best kernel speedup."""
    wins = [(s, c) for s, c in _kernel_speedups(p, admissible) if s > 1]
    if not wins:
        return p.dense.config
    return min(wins, key=lambda sc: (-sc[0], sc[1].config))[1].config


def best_kernel_config(points: Sequence[PointInputs], admissible) -> Optional[Config]:
    """The one sparse configuration with the best geometric-mean kernel
    speedup over the points where it is admissible and present; None if no
    admissible sparse configuration exists anywhere."""
    logs: dict = {}
    for p in points:
        for s, c in _kernel_speedups(p, admissible):
            logs.setdefault(c.config, []).append(math.log(s))
    if not logs:
        return None
    return min(logs, key=lambda cfg: (-statistics.fmean(logs[cfg]), cfg))


def static_choice(p: PointInputs, config: Optional[Config]) -> Optional[Config]:
    """A fixed configuration applied at a point: itself where the point has
    it, None where it does not (or when there is no fixed configuration)."""
    if config is None or p.candidate(config) is None:
        return None
    return config


def best_measured_choice(p: PointInputs, measured: Mapping[Config, float],
                         admissible) -> Config:
    """The best measured admissible configuration at a point, dense included
    (ties to dense, then to the smaller configuration): the reference
    regret is measured against, and the nearest-pairing comparator's choice
    when `measured` is that pairing's ground truth."""
    cfgs = [p.dense.config] + [c.config for c in _admissible(p, admissible)]
    have = [c for c in cfgs if c in measured]
    if p.dense.config not in measured:
        raise DispatchError(f"{tuple(p.point)} has no measured dense time")
    return min(have, key=lambda c: (measured[c], c != p.dense.config, c))


# --------------------------------------------------------------------------
# Scoring on the measured grid
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PointScore:
    point: Point
    chosen: Optional[Config]
    best: Optional[Config]
    regret: Optional[float]
    outcome: Optional[str]          # a `scoring` outcome, None where unscored
    gain_s: Optional[float]         # measured always-dense minus chosen
    unscored: Optional[str]         # why, when something is missing


@dataclass(frozen=True)
class PolicyScore:
    points: tuple[PointScore, ...]
    n_scored: int                   # points with a regret
    mean_regret: Optional[float]
    worst_regret: Optional[float]
    decisions: Mapping[str, int]    # outcome -> count (CORRECT, ...)
    inside_floor: int               # points not scored for decision error
    gain_s: float
    speedup_vs_dense: Optional[float]
    # At every scored point dense is the best measured admissible
    # configuration. A property of the grid and the set, not of the policy.
    dense_best_everywhere: bool

    @property
    def decision_errors(self) -> int:
        return self.decisions[FALSE_PROFITABLE] + self.decisions[MISSED_PROFITABLE]

    @property
    def unscored(self) -> tuple[PointScore, ...]:
        return tuple(p for p in self.points if p.unscored)


def choice_outcome(p: PointInputs, chosen: Config, measured: Mapping[Config, float],
                   admissible, floors: FloorTable) -> str:
    """Sparse chosen where dense was measured faster, or dense chosen where
    an admissible sparse configuration was measured faster; inside the
    point's floor is not interpreted (module docstring)."""
    floor_s = _floor_at(_floors(floors), p.point)
    dense = p.dense.config
    td = measured[dense]
    if chosen != dense:
        diff = measured[chosen] - td
        if abs(diff) <= floor_s:
            return NOT_INTERPRETED
        return FALSE_PROFITABLE if diff > 0 else CORRECT
    sparse = [measured[c.config] for c in _admissible(p, admissible)
              if c.config in measured]
    if not sparse:
        return CORRECT
    gap = td - min(sparse)
    if gap > floor_s:
        return MISSED_PROFITABLE
    return NOT_INTERPRETED if gap > 0 else CORRECT


def in_scope(p: PointInputs, floors: FloorTable, lengths: Optional[Iterable[int]],
             batches: Optional[Iterable[int]]) -> bool:
    """Whether a point is at the scored budget labels and batches (None:
    every one)."""
    if batches is not None and p.point.batch not in set(batches):
        return False
    if lengths is None:
        return True
    try:
        budget = floors.budget_of(p.point.context_length, p.point.batch)
    except FloorsError as e:
        raise DispatchError(f"{tuple(p.point)}: {e}") from e
    return budget in set(lengths)


def score_policy(points: Sequence[PointInputs],
                 choices: Mapping[Point, Optional[Config]],
                 measured: Mapping[Point, Mapping[Config, float]], admissible,
                 floors: FloorTable, *, lengths: Optional[Iterable[int]] = H6_LENGTHS,
                 batches: Optional[Iterable[int]] = H6_BATCHES,
                 outside: Optional[Mapping[Point, str]] = None) -> PolicyScore:
    """Score `choices` on the measured grid, at `lengths` (budget labels)
    and `batches` (None: every point). A point with no measured dense time,
    or whose chosen configuration was not measured, or with no choice, or
    named in `outside` (point -> why, the points outside the common set), is
    listed and not scored."""
    floors = _floors(floors)
    outside = dict(outside or {})
    scores, regrets = [], []
    decisions = {CORRECT: 0, FALSE_PROFITABLE: 0, MISSED_PROFITABLE: 0}
    inside = 0
    gain = 0.0
    dense_total = chosen_total = 0.0
    dense_best = True
    for p in points:
        if not in_scope(p, floors, lengths, batches):
            continue
        ch = choices.get(p.point)
        m = measured.get(p.point, {})
        why = None
        if p.point in outside:
            why = outside[p.point]
        elif p.dense.config not in m:
            why = "no measured dense time"
        elif ch is None:
            why = "the policy has no choice here"
        elif ch not in m:
            why = f"the chosen configuration {tuple(ch)} was not measured"
        if why:
            scores.append(PointScore(p.point, ch, None, None, None, None, why))
            continue
        best = best_measured_choice(p, m, admissible)
        regret = m[ch] / m[best] - 1.0
        outcome = choice_outcome(p, ch, m, admissible, floors)
        g = m[p.dense.config] - m[ch]
        scores.append(PointScore(p.point, ch, best, regret,
                                 None if outcome == NOT_INTERPRETED else outcome,
                                 g, None))
        regrets.append(regret)
        gain += g
        dense_total += m[p.dense.config]
        chosen_total += m[ch]
        dense_best = dense_best and best == p.dense.config
        if outcome == NOT_INTERPRETED:
            inside += 1
        else:
            decisions[outcome] += 1
    return PolicyScore(
        points=tuple(scores), n_scored=len(regrets),
        mean_regret=statistics.fmean(regrets) if regrets else None,
        worst_regret=max(regrets) if regrets else None,
        decisions=decisions, inside_floor=inside, gain_s=gain,
        speedup_vs_dense=dense_total / chosen_total if regrets else None,
        dense_best_everywhere=bool(regrets) and dense_best)


def choices_of(policy: str, points: Sequence[PointInputs], admissible, floors: FloorTable,
               *, nearest: Optional[Mapping[Point, Config]] = None
               ) -> dict[Point, Optional[Config]]:
    """Every point's choice under a named policy."""
    if policy not in POLICIES:
        raise DispatchError(f"policy must be one of {POLICIES}, got {policy!r}")
    if policy == "dispatcher":
        return {p.point: dispatcher(p, admissible, floors) for p in points}
    if policy == "kernel_speedup_rule":
        return {p.point: kernel_speedup_rule(p, admissible) for p in points}
    if policy == "always_dense":
        return {p.point: always_dense(p) for p in points}
    if policy == "best_kernel_config":
        fixed = best_kernel_config(points, admissible)
        return {p.point: static_choice(p, fixed) for p in points}
    if nearest is None:
        raise DispatchError("the nearest-pairing comparator needs the nearest "
                            "pairing's best measured choices")
    return {p.point: static_choice(p, nearest.get(p.point)) for p in points}


def outside_common_set(points: Sequence[PointInputs],
                       choices: Mapping[str, Mapping[Point, Optional[Config]]],
                       measured: Mapping[Point, Mapping[Config, float]]) -> dict[Point, str]:
    """Point -> why, for every point where dense or a `COMMON_SET` policy's
    choice was not measured: no policy is scored there (module docstring)."""
    out = {}
    for p in points:
        m = measured.get(p.point, {})
        if p.dense.config not in m:
            out[p.point] = "no measured dense time"
            continue
        for name in COMMON_SET:
            ch = choices[name].get(p.point)
            if ch is None or ch not in m:
                out[p.point] = (f"not in the common set: {name}'s choice "
                                f"{None if ch is None else tuple(ch)} was not measured")
                break
    return out


def evaluate(points: Sequence[PointInputs], measured: Mapping[Point, Mapping[Config, float]],
             admissible_sets: Mapping[str, Iterable[Config]], floors: FloorTable, *,
             nearest: Optional[Mapping[Point, Config]] = None,
             lengths: Optional[Iterable[int]] = H6_LENGTHS,
             batches: Optional[Iterable[int]] = H6_BATCHES
             ) -> dict[str, dict[str, PolicyScore]]:
    """label -> policy -> `PolicyScore`, for every admissible set, every
    policy on the common set of points (module docstring). The
    nearest-pairing comparator is scored only when `nearest` is given."""
    if not admissible_sets:
        raise DispatchError("give at least one admissible set")
    floors = _floors(floors)
    out = {}
    for label, adm in admissible_sets.items():
        adm = frozenset(Config(a, float(d)) for a, d in adm)
        names = [n for n in POLICIES if not (n == "nearest_pairing" and nearest is None)]
        chs = {n: choices_of(n, points, adm, floors, nearest=nearest) for n in names}
        outside = outside_common_set(points, chs, measured)
        out[label] = {n: score_policy(points, chs[n], measured, adm, floors, lengths=lengths,
                                      batches=batches, outside=outside)
                      for n in names}
    return out


NOT_SCORED = "not scored"


@dataclass(frozen=True)
class H6Line:
    """The dispatcher against the kernel-speedup rule on one pairing (H6),
    over the same scored points. With none, every comparison is None and
    `verdict` reads "not scored". The regret bound and the other lock values
    are the caller's."""

    lower_mean_regret: Optional[bool]
    lower_worst_regret: Optional[bool]
    no_more_decision_errors: Optional[bool]
    dispatcher_mean_regret: Optional[float]
    rule_mean_regret: Optional[float]
    dispatcher_errors: int
    rule_errors: int
    dense_best_everywhere: bool       # the test reduces to avoiding the rule's losses
    n_scored: int = 0
    verdict: Optional[str] = None     # NOT_SCORED when no point is scored


def h6_line(disp: PolicyScore, rule: PolicyScore) -> H6Line:
    """Refuses two scores over different points (`evaluate` scores both on
    the common set)."""
    def scored(s):
        return {p.point for p in s.points if p.regret is not None}
    if scored(disp) != scored(rule):
        raise DispatchError(
            f"the dispatcher is scored at {len(scored(disp))} points and the rule at "
            f"{len(scored(rule))}: H6 compares them on the same points (`evaluate`)")
    if disp.n_scored == 0:
        return H6Line(None, None, None, None, None, 0, 0, False, 0, NOT_SCORED)

    def lower(a, b):
        return None if a is None or b is None else a < b
    return H6Line(
        lower_mean_regret=lower(disp.mean_regret, rule.mean_regret),
        lower_worst_regret=lower(disp.worst_regret, rule.worst_regret),
        no_more_decision_errors=disp.decision_errors <= rule.decision_errors,
        dispatcher_mean_regret=disp.mean_regret, rule_mean_regret=rule.mean_regret,
        dispatcher_errors=disp.decision_errors, rule_errors=rule.decision_errors,
        dense_best_everywhere=disp.dense_best_everywhere, n_scored=disp.n_scored)


PRIMARY_COMPARATOR = "kernel_speedup_rule"


@dataclass(frozen=True)
class H6RuleLine:
    """The dispatcher against one comparator over the points both are scored
    at, beyond the point's floor. `excess_errors` is the dispatcher's
    decision errors minus the comparator's, the scored quantity; the regret
    excesses (the dispatcher's mean or worst regret minus the comparator's)
    are the secondary report. None where no point is shared."""

    comparator: str
    n_points: int
    dispatcher_errors: Optional[int]
    comparator_errors: Optional[int]
    excess_errors: Optional[int]
    mean_excess: Optional[float]
    worst_excess: Optional[float]
    passed: Optional[bool]


@dataclass(frozen=True)
class H6Rule:
    bound: Mapping[str, float]
    lines: tuple[H6RuleLine, ...]
    inside_floor: tuple[Point, ...]   # shared points left out: no choice matters there
    verdict: str                      # "passed", "failed" or NOT_SCORED


def beyond_floor(p: PointInputs, measured: Mapping[Config, float], admissible,
                 floors: FloorTable) -> bool:
    """Whether some measured admissible sparse configuration differs from
    measured dense by more than the point's floor: where it does not, every
    choice is dense to within the floor and regret there is noise."""
    floor_s = _floor_at(_floors(floors), p.point)
    td = measured[p.dense.config]
    return any(abs(measured[c.config] - td) > floor_s
               for c in _admissible(p, admissible) if c.config in measured)


ERRORS = (FALSE_PROFITABLE, MISSED_PROFITABLE)


def h6_rule(scores: Mapping[str, PolicyScore], points: Sequence[PointInputs],
            measured: Mapping[Point, Mapping[Config, float]], admissible,
            floors: FloorTable, *, bound: Mapping[str, float]) -> H6Rule:
    """H6 as the plan words it (Section 4; the lock-wave rulings of 12 Nov
    2026, the second H6 ruling; T4.1 row 1): the dispatcher makes no more
    decision errors than each comparator, to within `bound`
    ({"decision_errors"}, 0 at the lock), each pair on the points both are
    scored at (`scores` from one admissible set of `evaluate`, so batch 4 is
    already out) and beyond the point's floor (`beyond_floor`). Regret, mean
    and worst case, is reported beside it and decides nothing. Passed when
    every comparator with a shared point is; not scored when the primary
    comparator, the kernel-speedup rule, shares none."""
    if set(bound) != {"decision_errors"} or any(not (v >= 0) for v in bound.values()):
        raise DispatchError(f"the H6 bound is {{'decision_errors'}}, >= 0: {bound!r}")
    if "dispatcher" not in scores or PRIMARY_COMPARATOR not in scores:
        raise DispatchError("H6 compares the dispatcher with the kernel-speedup rule at least")
    by_point = {p.point: p for p in points}
    adm = frozenset(admissible)

    def scored(s):
        return {x.point: x for x in s.points if x.regret is not None}
    disp = scored(scores["dispatcher"])
    lines, inside = [], set()
    for name in sorted(n for n in scores if n != "dispatcher"):
        comp = scored(scores[name])
        shared = sorted(set(disp) & set(comp))
        kept = [pt for pt in shared
                if beyond_floor(by_point[pt], measured[pt], adm, floors)]
        inside |= set(shared) - set(kept)
        if not kept:
            lines.append(H6RuleLine(name, 0, None, None, None, None, None, None))
            continue
        d_err = sum(disp[pt].outcome in ERRORS for pt in kept)
        c_err = sum(comp[pt].outcome in ERRORS for pt in kept)
        mean_x = (statistics.fmean(disp[pt].regret for pt in kept)
                  - statistics.fmean(comp[pt].regret for pt in kept))
        worst_x = max(disp[pt].regret for pt in kept) - max(comp[pt].regret for pt in kept)
        lines.append(H6RuleLine(name, len(kept), d_err, c_err, d_err - c_err, mean_x, worst_x,
                                d_err - c_err <= bound["decision_errors"]))
    primary = next(x for x in lines if x.comparator == PRIMARY_COMPARATOR)
    if primary.passed is None:
        verdict = NOT_SCORED
    else:
        verdict = "passed" if all(x.passed for x in lines if x.passed is not None) else "failed"
    return H6Rule(dict(bound), tuple(lines), tuple(sorted(inside)), verdict)


# --------------------------------------------------------------------------
# Margin against speedup (Section 5.7)
# --------------------------------------------------------------------------

def admissible_at_margin(lower_pts: Mapping[Config, float],
                         margin_pts: float) -> frozenset:
    """The sparse configurations whose lower confidence bound on the paired
    accuracy difference, in points, is above -margin: the rule
    `exact_noninferiority.paired_exact_ni` applies (`lower > -margin`)."""
    if isinstance(margin_pts, bool) or not isinstance(margin_pts, (int, float)) \
            or not math.isfinite(margin_pts) or margin_pts <= 0:
        raise DispatchError(f"a margin is a positive number of points, got {margin_pts!r}")
    return frozenset(Config(a, float(d)) for (a, d), v in lower_pts.items()
                     if v > -margin_pts)


@dataclass(frozen=True)
class FrontierRow:
    margin_pts: float
    n_admissible: int
    policy: str
    n_scored: int
    speedup_vs_dense: Optional[float]
    gain_s: float
    mean_regret: Optional[float]
    worst_regret: Optional[float]
    decision_errors: int


def margin_frontier(points: Sequence[PointInputs],
                    measured: Mapping[Point, Mapping[Config, float]],
                    lower_pts: Mapping[Config, float], margins_pts: Sequence[float],
                    floors: FloorTable, *, policies: Sequence[str] = (
                        "dispatcher", "kernel_speedup_rule"),
                    lengths: Optional[Iterable[int]] = H6_LENGTHS,
                    batches: Optional[Iterable[int]] = H6_BATCHES) -> list[FrontierRow]:
    """For each margin, the admissible set it gives and what the policies
    buy on the measured grid. Computed from the accuracy rows and the
    ground-truth grid alone."""
    if "nearest_pairing" in policies:
        raise DispatchError("the nearest-pairing comparator is not swept")
    rows = []
    for m in sorted(set(margins_pts)):
        adm = admissible_at_margin(lower_pts, m)
        for name in policies:
            ch = choices_of(name, points, adm, floors)
            s = score_policy(points, ch, measured, adm, floors, lengths=lengths,
                             batches=batches)
            rows.append(FrontierRow(
                margin_pts=float(m), n_admissible=len(adm), policy=name,
                n_scored=s.n_scored, speedup_vs_dense=s.speedup_vs_dense,
                gain_s=s.gain_s, mean_regret=s.mean_regret,
                worst_regret=s.worst_regret, decision_errors=s.decision_errors))
    return rows


# --------------------------------------------------------------------------
# What the committed file carries (K9: the choice is written before the
# workload is timed)
# --------------------------------------------------------------------------

PRIMARY_SET = "uncorrected"
HOLM_SET = "holm"
FILE_POLICIES = ("dispatcher", "kernel_speedup_rule", "always_dense",
                 "best_kernel_config")


@dataclass(frozen=True)
class DispatchSpec:
    """Everything the dispatcher read, as the file records it."""

    predictor: str
    floors: FloorTable
    dense_arms: tuple[str, ...]
    admissible: Mapping[str, tuple[Config, ...]]
    kernel_times: Mapping[CellKey, float]
    use_code_path: Mapping[str, str]
    nearest: Optional[Mapping[Point, Config]] = None

    def to_json(self) -> dict:
        return {
            "predictor": self.predictor, "floors": self.floors.to_json(),
            "dense_arms": sorted(self.dense_arms),
            "admissible": {k: [[c.arm, c.density] for c in sorted(v)]
                           for k, v in sorted(self.admissible.items())},
            "kernel_times": [{"key": list(k), "seconds": v}
                             for k, v in sorted(self.kernel_times.items())],
            "use_code_path": dict(sorted(self.use_code_path.items())),
            "nearest": None if self.nearest is None else
            [{"point": list(p), "arm": c.arm, "density": c.density}
             for p, c in sorted(self.nearest.items())],
        }


def spec_from_json(d: Mapping) -> DispatchSpec:
    try:
        return DispatchSpec(
            predictor=d["predictor"], floors=FloorTable.from_json(d["floors"]),
            dense_arms=tuple(d["dense_arms"]),
            admissible={k: tuple(Config(a, float(x)) for a, x in v)
                        for k, v in d["admissible"].items()},
            kernel_times={CellKey(*e["key"]): float(e["seconds"])
                          for e in d["kernel_times"]},
            use_code_path=dict(d["use_code_path"]),
            nearest=None if d["nearest"] is None else
            {Point(*e["point"]): Config(e["arm"], float(e["density"]))
             for e in d["nearest"]})
    except (KeyError, TypeError, ValueError, FloorsError) as e:
        raise DispatchError(f"unreadable dispatch record: {e!r}") from e


def make_choices(spec: DispatchSpec, predicted: Mapping[CellKey, float]) -> list[dict]:
    """The choices the file records: for each admissible set and each
    decision point, the dispatcher's and the comparators' (and the nearest
    pairing's, echoed from the spec)."""
    if not spec.admissible:
        raise DispatchError("a dispatch record names at least one admissible set")
    pts = build_points(predicted, dense_arms=spec.dense_arms,
                       kernel=spec.kernel_times, use_code_path=spec.use_code_path)
    out = []
    for label in sorted(spec.admissible):
        adm = frozenset(spec.admissible[label])
        names = list(FILE_POLICIES) + (["nearest_pairing"] if spec.nearest is not None
                                       else [])
        for name in names:
            ch = choices_of(name, pts, adm, spec.floors, nearest=spec.nearest)
            for p in pts:
                c = ch[p.point]
                cand = None if c is None else p.candidate(c)
                out.append({
                    "set": label, "policy": name,
                    "model": p.point.model, "context_length": p.point.context_length,
                    "batch": p.point.batch,
                    "choice": None if c is None else
                    {"arm": c.arm, "density": c.density,
                     "code_path": cand.key.code_path},
                })
    return out
