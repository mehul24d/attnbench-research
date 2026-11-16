"""EXPLORATORY: regret of a safe dispatcher (item 3 of the 16 Nov 2026
rulings).

Four policies on the deployable configurations (every arm except the oracle,
which needs the exact mask and cannot be deployed):

  - always dense;
  - BUNGA, the recursion's committed times through `dispatcher.dispatcher`;
  - BUNGA, step form: the same dispatcher fed the step-form times;
  - BUNGA with an abstain margin: the recursion dispatcher, but a sparse
    configuration is chosen only when it is also predicted faster than dense
    by more than `margin` of dense's predicted time.

The margin is fixed from P1 alone (`margin_from`) and recorded in Appendix A
before P2 or P4 is read: the smallest margin at which no P1 point chooses a
sparse configuration that measured slower than dense.

Reported per policy: mean and worst regret (measured chosen over the best
measured admissible configuration, minus one), the worst slowdown against
always dense, and the share of sparse wins kept, where a win is a point at
which an admissible sparse configuration measured faster than dense by more
than the point's floor, and it is kept when the policy chose a sparse
configuration that also beat dense by more than the floor.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from attnbench.predict import dispatcher as DP
from attnbench.predict.training import partner_of

NOT_DEPLOYABLE = ("oracle",)


def admissible_of(keys) -> frozenset:
    """Every deployable sparse configuration among `keys`."""
    return frozenset(DP.Config(k.arm, float(k.density)) for k in keys
                     if k.arm != "dense" and k.arm not in NOT_DEPLOYABLE)


def measured_by_point(seconds: Mapping) -> dict:
    """CellKey -> seconds, regrouped as Point -> Config -> seconds."""
    out: dict = {}
    for k, v in seconds.items():
        out.setdefault(DP.Point(*partner_of(k)), {})[DP.Config(k.arm, float(k.density))] = v
    return out


def advantage(p: DP.PointInputs, admissible) -> tuple[Optional[DP.Candidate], float, float]:
    """(the predicted-fastest admissible sparse candidate, its predicted
    advantage over dense in seconds, and as a fraction of dense)."""
    adm = [c for c in p.sparse if c.config in admissible]
    if not adm:
        return None, 0.0, 0.0
    best = min(adm, key=lambda c: (c.predicted, c.config))
    gap = p.dense.predicted - best.predicted
    return best, gap, gap / p.dense.predicted


def abstaining(p: DP.PointInputs, admissible, floors, margin: float) -> DP.Config:
    """`dispatcher.dispatcher`, abstaining to dense unless the chosen sparse
    configuration's predicted advantage also exceeds `margin` of dense."""
    if margin < 0:
        raise ValueError("the margin is a non-negative fraction")
    ch = DP.dispatcher(p, admissible, floors)
    if ch == p.dense.config:
        return ch
    _, _, rel = advantage(p, admissible)
    return ch if rel > margin else p.dense.config


def margin_from(points: Sequence[DP.PointInputs], measured: Mapping, admissible,
                floors, batches=(1,)) -> tuple[float, list]:
    """The smallest margin at which no point chooses a sparse configuration
    measured slower than dense (by any amount): the largest relative
    predicted advantage among the points where the dispatcher chose such a
    configuration, and 0.0 when there is none. Returns (margin, the
    offending points as (point, config, relative advantage))."""
    bad = []
    for p in points:
        if p.point.batch not in batches:
            continue
        ch = DP.dispatcher(p, admissible, floors)
        m = measured.get(p.point, {})
        if ch == p.dense.config or ch not in m or p.dense.config not in m:
            continue
        if m[ch] > m[p.dense.config]:
            bad.append((p.point, ch, advantage(p, admissible)[2]))
    return (max(b[2] for b in bad) if bad else 0.0), bad


@dataclass(frozen=True)
class SafeScore:
    n: int
    mean_regret: float
    worst_regret: float
    worst_vs_dense: float        # max of measured chosen / measured dense - 1
    mean_vs_dense: float
    wins: int
    wins_kept: int
    choices: Mapping


def score(points: Sequence[DP.PointInputs], choices: Mapping, measured: Mapping,
          admissible, floors, batches=(1,)) -> SafeScore:
    """Regret and wins kept over the points at `batches` with a measured
    dense time and a measured choice."""
    ft = floors
    reg, vs, wins, kept, used = [], [], 0, 0, {}
    for p in points:
        if p.point.batch not in batches:
            continue
        m = measured.get(p.point, {})
        ch = choices.get(p.point)
        if p.dense.config not in m or ch is None or ch not in m:
            continue
        floor_s = ft.at(p.point.context_length, p.point.batch)
        best = DP.best_measured_choice(p, m, admissible)
        td = m[p.dense.config]
        reg.append(m[ch] / m[best] - 1.0)
        vs.append(m[ch] / td - 1.0)
        used[p.point] = ch
        if td - m[best] > floor_s:
            wins += 1
            kept += ch != p.dense.config and td - m[ch] > floor_s
    if not reg:
        raise ValueError("no scored point")
    return SafeScore(n=len(reg), mean_regret=statistics.fmean(reg), worst_regret=max(reg),
                     worst_vs_dense=max(vs), mean_vs_dense=statistics.fmean(vs),
                     wins=wins, wins_kept=kept, choices=used)


def points_for(cells, prediction: str) -> list:
    """Decision points from committed cells, on one predictor's times."""
    return DP.build_points({c.key: c.predictions[prediction] for c in cells},
                           dense_arms={"dense"})


def policies(cells, admissible, floors, margin: float) -> dict:
    """policy -> (points, choices)."""
    rec, step = points_for(cells, "recursion"), points_for(cells, "step_form")
    return {
        "always_dense": (rec, {p.point: p.dense.config for p in rec}),
        "bunga_recursion": (rec, {p.point: DP.dispatcher(p, admissible, floors) for p in rec}),
        "bunga_step_form": (step, {p.point: DP.dispatcher(p, admissible, floors) for p in step}),
        "bunga_abstain": (rec, {p.point: abstaining(p, admissible, floors, margin) for p in rec}),
    }


def advantage_errors(points: Sequence[DP.PointInputs], measured: Mapping, admissible,
                     batches=(1,)) -> list:
    """For every admissible sparse configuration at every point: (point,
    config, predicted relative advantage over dense, measured relative
    advantage, predicted minus measured). A positive last value overstates
    the sparse configuration's advantage, the direction that makes a
    dispatcher choose a loss."""
    out = []
    for p in points:
        if p.point.batch not in batches:
            continue
        m = measured.get(p.point, {})
        if p.dense.config not in m:
            continue
        td, pd = m[p.dense.config], p.dense.predicted
        for c in p.sparse:
            if c.config in admissible and c.config in m:
                a_p, a_m = (pd - c.predicted) / pd, (td - m[c.config]) / td
                out.append((p.point, c.config, a_p, a_m, a_p - a_m))
    return out


def fallback_margin(points, measured, admissible, batches=(1,)) -> float:
    """The margin used when `margin_from` is degenerate (0.0): the largest
    overstatement of a sparse configuration's relative advantage over dense,
    across every admissible configuration and point (0.0 if none is
    overstated). Choosing sparse only beyond it means a loss would need a
    larger overstatement than any seen."""
    errs = advantage_errors(points, measured, admissible, batches)
    if not errs:
        raise ValueError("no admissible measured configuration")
    return max(0.0, max(e[4] for e in errs))
