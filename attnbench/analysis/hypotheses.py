"""The confirmatory analysis: H0 to H5 and the marginal label, as Section 4
and Appendix A state them (T4.3 wave 2a, adviser's ruling 2a of 11 Nov 2026;
T4.3 read A, finding 5). H6 is `predict.dispatcher`'s (`evaluate`, `h6_line`).

Every threshold is read from the lock (`Locked.from_lock`): the tolerances,
the floors (P1's table), the bootstrap's level, size and seed, H4's bound,
family and n, the scored pairings and the marginal label's sensitivity
floors. Nothing here has a default for a lock value. Each function returns a
labelled verdict, `HOLDS`, `FAILS`, `NOT_SCORED` (nothing could be scored),
`NOT_TESTED` (a condition upstream failed) or `INCONCLUSIVE` (H4 below its
power), with what was scored, what was left out and why.

  - **H0** (Appendix A, 9 and 10 Nov 2026): per compared pair of pairings and
    component side, operation by operation on cells of the same arm, density,
    length and batch whose operation sequences match (Section 5.1); an
    operation is scored where its component time is above the cell's floor on
    both pairings (**plan default, flagged:** both), else listed unscorable;
    the median relative difference (other - reference) / reference over the
    scored operations at most H0's tolerance, the largest and the share of
    the side's time covered reported beside it (host time on the host side,
    device time on the device side). A cell whose sequence differs is
    reported separately.
  - **H1** (Section 4): per confirmatory pairing, (a) the recursion's median
    absolute relative error at most H1's tolerance; (b) the sparse-vs-dense
    sign correct at every comparison outside the floor (**plan default,
    flagged:** every one, none disagreeing); (c) the recursion's median below
    each of the fitted sum's (primary), the sum's, the step form's and the
    learned predictor's (strict); (d) the XAttention pair's predicted and
    measured differences agreeing within H1's tolerance, relative to the
    measured difference, at each length where the measured difference is
    outside the floor (**plan default, flagged:** the relative reading; a
    pair with no interpretable length is reported, not failed). By arm class
    as reported, never scored. P1 is refused: it is the development pairing.
  - **H2**: H1's statistic (a) and (b) on the recursion re-run with one
    side's components from another pairing (`transferred`: the target's
    priced sequence with the source's host times, its own queue depth,
    computed from the two committed files, both written before the target's
    ground truth). Not tested if H0 fails.
  - **H3**: on P4, each estimator's measured crossover bracket (signs from
    the bootstrap interval of the measured difference and the floor) inside
    the predicted one (`crossover.crossover_error`'s `contains`); none in
    range is reported as such.
  - **H4**: the equivalence test at matched recall, two one-sided exact
    paired tests against the bound per test, Holm across the lock's family
    (m = 18, Appendix A, 10 and 11 Nov 2026): a test is equivalent when the
    larger of its two one-sided p-values clears its Holm step; below the
    lock's n it is inconclusive, never support or refutation.
  - **H5** (secondary): the predicted sign of each estimator's crossover
    shift between two pairings against the measured shift in grid steps;
    (b) the GPU-only estimator's shift smaller in magnitude than each
    host-side one's. (c) needs a fixed GPU on two hosts, which no pairing
    run gives: not scored.
  - **The marginal label** (Appendix A, 9 Nov 2026 and 11 Nov 2026): a
    sparse-minus-dense comparison above the floor is "marginal" at 16384
    when it is inside the sensitivity floor, and on P2 or P4 when it is
    inside the pairing's own within-session spread (`within_session_spread`,
    descriptive, never a floor). A label, not a verdict: H1's and H3's
    verdicts count a marginal comparison like any other and list it.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from ..lock.floors import FloorTable, FloorsError
from ..predict import crossover as X
from ..predict import intervals as IV
from ..predict import residuals as R
from ..predict import scoring
from ..predict.recursion import recurse

HOLDS, FAILS = "holds", "fails"
NOT_SCORED, NOT_TESTED, INCONCLUSIVE = "not scored", "not tested", "inconclusive"
MARGINAL = "marginal"
DEVELOPMENT_PAIRING = "P1"
H1_RIVALS = ("fitted_sum", "sum", "step_form", "learned")
XA_PAIR = ("xattention", "xattention_noassert")
MARGINAL_LENGTH = 16384


class HypothesisError(ValueError):
    pass


def _fraction(v, what) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1:
        raise HypothesisError(f"{what} is {v!r}, not a fraction fixed at lock")
    return float(v)


@dataclass(frozen=True)
class Locked:
    """The lock values the analysis reads."""
    tolerances: Mapping[str, float]          # H0, H1, H2
    floors: FloorTable
    bootstrap: IV.Settings
    h4_bound_pts: float
    h4_tests: tuple
    h4_alpha: float
    h4_n: int
    scored: Mapping[str, tuple]
    sensitivity_s: Mapping[tuple, float]     # {(16384, batch): seconds}
    analysis_commit: str

    @classmethod
    def from_lock(cls, lock: Mapping) -> "Locked":
        vals = lock.get("values") or {}

        def value(k):
            v = (vals.get(k) or {}).get("value")
            if v is None:
                raise HypothesisError(f"the lock's {k} is not fixed (T4.1)")
            return v
        tol = value("h0_to_h3_tolerances")
        fam = value("h4_confirmatory_family")
        power = value("power_target_and_n")
        sens = value("marginal_label_s_sensitivity_floors")
        try:
            floors = FloorTable.from_lock(lock)
        except FloorsError as e:
            raise HypothesisError(str(e)) from e
        try:
            boot = IV.settings_from_lock(lock)
        except ValueError as e:
            raise HypothesisError(str(e)) from e
        tests = tuple(fam["tests"])
        if len(tests) != fam["m"]:
            raise HypothesisError(f"H4's family names {len(tests)} tests, m = {fam['m']}")
        scored = {h: tuple(p) for h, p in (lock.get("scored_pairings") or {}).items()}
        for h in ("H1", "H2", "H3"):
            if DEVELOPMENT_PAIRING in scored.get(h, ()):
                raise HypothesisError(f"{h} names P1, the development pairing")
        commit = value("analysis_commit")
        return cls(tolerances={h: _fraction(tol[h], f"{h}'s tolerance") for h in ("H0", "H1", "H2")},
                   floors=floors, bootstrap=boot,
                   h4_bound_pts=float(value("h4_equivalence_bound_with_its_justification_from_the_tasks")),
                   h4_tests=tests, h4_alpha=_fraction(power["alpha"], "H4's alpha"),
                   h4_n=int(power["n"]), scored=scored,
                   sensitivity_s={_cell(k): float(v) / 1000.0 for k, v in sens.items()},
                   analysis_commit=str(commit))


def _cell(label) -> tuple:
    """'16384x1' -> (16384, 1)."""
    L, _, b = str(label).partition("x")
    return int(L), int(b)


# --- the marginal label --------------------------------------------------------

def within_session_spread(samples: Mapping, *, quantile: float, budget_of) -> dict:
    """A pairing's own spread per (budget, batch), in seconds: the quantile of
    |repeat - cell median| pooled over its cells, the rule P1's resolution
    floor uses within a session. Descriptive (Appendix A, 11 Nov 2026): it
    labels comparisons, it never replaces P1's floors. `samples`: {cell key:
    [seconds, ...]}; `budget_of(length, batch)` names the budget."""
    pool: dict = {}
    for k, xs in samples.items():
        xs = [float(x) for x in xs]
        if len(xs) < 2:
            raise HypothesisError(f"{k}: a spread needs at least two repeats")
        med = statistics.median(xs)
        pool.setdefault((budget_of(k.context_length, k.batch), k.batch), []).extend(
            abs(x - med) for x in xs)
    return {c: float(np.quantile(v, quantile)) for c, v in sorted(pool.items())}


def marginal(diff_s: float, length: int, batch: int, *, floors: FloorTable,
             sensitivity_s: Mapping, spread_s: Optional[Mapping] = None) -> bool:
    """Whether a measured sparse-minus-dense difference above the floor is
    marginal (module docstring). Inside the floor it is not interpreted, and
    not marginal either."""
    d = abs(float(diff_s))
    if d <= floors.at(length, batch):
        return False
    cell = (floors.budget_of(length, batch), int(batch))
    if cell[0] == MARGINAL_LENGTH:
        if cell not in sensitivity_s:
            raise HypothesisError(f"no sensitivity floor at {cell}: a lock value")
        if d <= sensitivity_s[cell]:
            return True
    return spread_s is not None and cell in spread_s and d <= spread_s[cell]


# --- H0 -------------------------------------------------------------------------

def _match_key(c) -> tuple:
    return (c.model, c.arm, round(float(c.density), 6), int(c.context_length), int(c.batch))


def _same_sequence(a, b) -> bool:
    return (a.kinds, a.mechanisms, a.layers, a.payload_bytes) == \
        (b.kinds, b.mechanisms, b.layers, b.payload_bytes)


def matched_cells(ref, other) -> tuple:
    """([(ref cell, other cell)] with matching sequences, [differing],
    [unmatched keys]) between two committed files, matched by arm, density,
    length and batch (the code path names the GPU, so it differs)."""
    rk = {_match_key(c): c for c in ref.cells}
    ok, differ, unmatched = [], [], []
    for c in other.cells:
        r = rk.pop(_match_key(c), None)
        if r is None:
            unmatched.append(list(_match_key(c)))
        elif _same_sequence(r.priced, c.priced):
            ok.append((r, c))
        else:
            differ.append(list(_match_key(c)))
    unmatched += [list(k) for k in rk]
    return ok, differ, unmatched


def h0(ref, other, *, side: str, floors: FloorTable, tolerance: float) -> dict:
    """H0 on one component side ("host" or "device") between two committed
    files (`ref` the earlier pairing)."""
    if side not in ("host", "device"):
        raise HypothesisError(f"a component side is host or device, got {side!r}")
    tolerance = _fraction(tolerance, "H0's tolerance")
    pairs, differ, unmatched = matched_cells(ref, other)
    rel, covered, total, unscorable = [], 0.0, 0.0, 0
    for r, c in pairs:
        floor = floors.at(r.context_length, r.batch)
        for a, b in zip(getattr(r.priced, side), getattr(c.priced, side)):
            total += a
            if a > floor and b > floor:
                rel.append((b - a) / a)
                covered += a
            else:
                unscorable += 1
    out = {"hypothesis": "H0", "side": side, "pairings": [ref.pairing, other.pairing],
           "tolerance": tolerance, "cells_compared": len(pairs),
           "cells_with_a_different_sequence": differ, "cells_unmatched": unmatched,
           "operations_scored": len(rel), "operations_unscorable": unscorable,
           "coverage_share": covered / total if total > 0 else None}
    if not rel:
        return {**out, "verdict": NOT_SCORED,
                "why": "no operation is above its cell's floor on both pairings"}
    med = float(statistics.median(abs(x) for x in rel))
    return {**out, "median_abs_relative_difference": med,
            "max_abs_relative_difference": float(max(abs(x) for x in rel)),
            "verdict": HOLDS if med <= tolerance else FAILS}


# --- H1 and H2 ------------------------------------------------------------------

def _median_abs(res, predictor) -> Optional[float]:
    xs = [r.abs_relative for r in res if r.predictor == predictor]
    return float(statistics.median(xs)) if xs else None


def _signs(pred: Mapping, seconds: Mapping, *, dense_arm: str, floors: FloorTable,
           sensitivity_s: Mapping, spread_s: Optional[Mapping]) -> dict:
    """Sparse-vs-dense sign agreement per sparse cell with a measured dense
    partner; inside the floor not scored; marginal ones listed."""
    dense = {(k.model, k.context_length, k.batch): k for k in seconds if k.arm == dense_arm}
    agree, disagree, inside, marg = 0, [], 0, []
    for k in sorted(seconds):
        d = dense.get((k.model, k.context_length, k.batch))
        if k.arm == dense_arm or d is None or k not in pred or d not in pred:
            continue
        s = scoring.sign_agreement(pred[k], pred[d], seconds[k], seconds[d],
                                   floor=floors.at(k.context_length, k.batch))
        if s is None:
            inside += 1
            continue
        if marginal(seconds[k] - seconds[d], k.context_length, k.batch, floors=floors,
                    sensitivity_s=sensitivity_s, spread_s=spread_s):
            marg.append(list(k))
        if s:
            agree += 1
        else:
            disagree.append(list(k))
    n = agree + len(disagree)
    return {"agree": agree, "disagree": disagree, "inside_floor": inside, "marginal": marg,
            "verdict": NOT_SCORED if n == 0 else (HOLDS if not disagree else FAILS)}


def _xa_pair(cp, seconds, *, floors, tolerance) -> dict:
    a, b = XA_PAIR
    cells = {c.key: c for c in cp.cells}
    at_cell: dict = {}
    for k in seconds:
        if k.arm in XA_PAIR:
            at_cell.setdefault((k.model, k.context_length, k.batch), {}).setdefault(k.arm, []).append(k)
    rows, agree, disagree = [], 0, 0
    for at, arms in sorted(at_cell.items()):
        if len(arms.get(a, ())) != 1 or len(arms.get(b, ())) != 1:
            rows.append({"cell": list(at), "why": "not one cell of each arm"})
            continue
        ka, kb = arms[a][0], arms[b][0]
        pred = cells[ka].predictions["recursion"] - cells[kb].predictions["recursion"]
        meas = seconds[ka] - seconds[kb]
        if abs(meas) <= floors.at(ka.context_length, ka.batch):
            rows.append({"cell": list(at), "predicted_s": pred, "measured_s": meas,
                         "why": "the measured difference is inside the floor"})
            continue
        ok = abs(pred - meas) <= tolerance * abs(meas)
        agree, disagree = agree + ok, disagree + (not ok)
        rows.append({"cell": list(at), "predicted_s": pred, "measured_s": meas, "agrees": ok})
    return {"rows": rows, "agree": agree, "disagree": disagree,
            "verdict": NOT_SCORED if agree + disagree == 0 else (HOLDS if not disagree else FAILS)}


def h1(cp, seconds: Mapping, *, locked: Locked, dense_arm: str,
       spread_s: Optional[Mapping] = None, development: bool = False) -> dict:
    """H1's verdict on one pairing (module docstring). `development=True`
    reads P1 as a dry run, labelled: never a test."""
    if (cp.pairing == DEVELOPMENT_PAIRING) != development:
        raise HypothesisError("P1 is the development pairing: read it only as a labelled "
                              "dry run, and no other pairing as one")
    tol = locked.tolerances["H1"]
    missing = [p for p in ("recursion",) + H1_RIVALS
               if not all(p in c.predictions for c in cp.cells)]
    if "recursion" in missing or (missing and not development):
        raise HypothesisError(f"H1 compares the recursion with {list(H1_RIVALS)}; "
                              f"{cp.pairing}'s file lacks {missing}")
    rivals_in = [p for p in H1_RIVALS if p not in missing]   # P1 is trained on nothing
    rs = R.residuals(cp, seconds, statistic=cp.statistic, dense_arms=[dense_arm],
                     relative_floor=0.0, predictors=["recursion", *rivals_in])
    res = rs.residuals
    rec = _median_abs(res, "recursion")
    if rec is None:
        return {"hypothesis": "H1", "pairing": cp.pairing, "verdict": NOT_SCORED,
                "why": "no measured cell"}
    rivals = {p: _median_abs(res, p) for p in rivals_in}
    below = {p: rec < m for p, m in rivals.items()}
    pred = {c.key: c.predictions["recursion"] for c in cp.cells}
    signs = _signs(pred, seconds, dense_arm=dense_arm, floors=locked.floors,
                   sensitivity_s=locked.sensitivity_s, spread_s=spread_s)
    xa = _xa_pair(cp, seconds, floors=locked.floors, tolerance=tol)
    by_class = {}
    for cls in (R.DENSE, R.SYNC_FREE, R.SYNCHRONISING):
        xs = [r for r in res if r.arm_class == cls]
        if xs:
            by_class[cls] = {p: _median_abs(xs, p) for p in ("recursion", *rivals_in)}
    parts = {"tolerance": HOLDS if rec <= tol else FAILS,
             "beats_every_rival": HOLDS if all(below.values()) else FAILS,
             "sign": signs["verdict"], "xattention_pair": xa["verdict"]}
    verdict = (FAILS if FAILS in parts.values() else
               NOT_SCORED if parts["sign"] == NOT_SCORED else HOLDS)
    return {"hypothesis": "H1", "pairing": cp.pairing,
            **({"label": "development pairing, dry run: not a test"} if development else {}),
            "tolerance": tol, "n_cells": sum(r.predictor == "recursion" for r in res),
            "unmeasured": [list(k) for k in rs.unmeasured],
            "recursion_median_abs_relative": rec, "rivals_median_abs_relative": rivals,
            "rivals_not_in_file": missing,
            "recursion_below": below, "primary_comparison": "fitted_sum",
            "sign": signs, "xattention_pair": xa, "by_arm_class": by_class,
            "parts": parts, "verdict": verdict}


def transferred(target, source, *, side: str = "host") -> tuple:
    """({target cell key: the recursion with `side`'s times from `source`},
    [target cells with no source cell of the same sequence])."""
    pairs, differ, unmatched = matched_cells(source, target)
    out = {}
    for s, t in pairs:
        p = dataclasses.replace(t.priced, **{side: getattr(s.priced, side)})
        out[t.key] = recurse(p, target.queue_depth).end_to_end
    return out, differ + unmatched


def h2(target, source, seconds: Mapping, *, locked: Locked, dense_arm: str, h0_verdict: str,
       side: str = "host", spread_s: Optional[Mapping] = None) -> dict:
    """H2 on `target` with `side`'s components from `source`."""
    head = {"hypothesis": "H2", "pairing": target.pairing, "source": source.pairing,
            "side": side}
    if h0_verdict == FAILS:
        return {**head, "verdict": NOT_TESTED, "why": "H0 failed (Section 4: H2 is conditional on H0)"}
    if target.pairing == DEVELOPMENT_PAIRING:
        raise HypothesisError("P1 is the development pairing")
    tol = locked.tolerances["H2"]
    pred, left_out = transferred(target, source, side=side)
    rel = [abs(pred[k] - seconds[k]) / seconds[k] for k in pred if k in seconds]
    if not rel:
        return {**head, "verdict": NOT_SCORED, "left_out": left_out,
                "why": "no transferred cell is measured"}
    med = float(statistics.median(rel))
    signs = _signs(pred, seconds, dense_arm=dense_arm, floors=locked.floors,
                   sensitivity_s=locked.sensitivity_s, spread_s=spread_s)
    parts = {"tolerance": HOLDS if med <= tol else FAILS, "sign": signs["verdict"]}
    verdict = FAILS if FAILS in parts.values() else (
        NOT_SCORED if signs["verdict"] == NOT_SCORED else HOLDS)
    return {**head, "tolerance": tol, "n_cells": len(rel), "left_out": left_out,
            "median_abs_relative": med, "sign": signs, "parts": parts, "verdict": verdict}


# --- H3 -------------------------------------------------------------------------

def measured_differences(samples: Mapping, sparse: Sequence, dense: Sequence, *,
                         statistic, settings: IV.Settings) -> list:
    """GridDifferences of sparse - dense per grid length, each with its
    bootstrap interval at the lock's settings."""
    out = []
    for s, d in zip(sparse, dense):
        b = IV.bootstrap({"s": samples[s], "d": samples[d]},
                         lambda v: {"s": v["s"], "d": v["d"]},
                         n_boot=settings.n_boot, seed=settings.seed, statistic=statistic)
        iv = b.difference("s", "d", settings.level)
        out.append(X.GridDifference(length=int(s.context_length), point=iv.point,
                                    lo=iv.lo, hi=iv.hi))
    return out


def h3(predicted: Mapping, measured: Mapping, *, floors: FloorTable, pairing: str) -> dict:
    """`predicted` and `measured`: {estimator: [GridDifference per length]}."""
    if set(predicted) != set(measured):
        raise HypothesisError(f"predicted {sorted(predicted)} and measured {sorted(measured)} "
                              f"name different estimators")
    rows = {}
    for e in sorted(predicted):
        p = X.crossover_brackets(predicted[e], floors=floors)
        m = X.crossover_brackets(measured[e], floors=floors)
        err = X.crossover_error(p, m)
        if err.status == "compared":
            ok = all(c.contains for c in err.comparisons)
            rows[e] = {"status": err.status, "verdict": HOLDS if ok else FAILS,
                       "brackets": [[c.measured.lower, c.measured.upper, c.predicted.lower,
                                     c.predicted.upper] for c in err.comparisons]}
        elif err.status == "both none in range":
            rows[e] = {"status": err.status, "verdict": NOT_SCORED,
                       "why": "no crossover in range, predicted or measured (reported as such)"}
        elif err.status == "unresolved":
            rows[e] = {"status": err.status, "verdict": NOT_SCORED,
                       "why": "a sign is undetermined where it decides the bracket"}
        else:
            rows[e] = {"status": err.status, "verdict": FAILS}
    vs = [r["verdict"] for r in rows.values()]
    return {"hypothesis": "H3", "pairing": pairing, "estimators": rows,
            "verdict": FAILS if FAILS in vs else (HOLDS if HOLDS in vs else NOT_SCORED)}


# --- H4 -------------------------------------------------------------------------

def _tost_p(cnt, bound_pts: float) -> float:
    from ..lock.holm import Counts, ni_p_value
    lower = ni_p_value(cnt, bound_pts)                              # diff > -bound
    upper = ni_p_value(Counts(b=cnt.c, c=cnt.b, n=cnt.n), bound_pts)  # diff < bound
    return max(lower, upper)


def h4(counts: Mapping, *, locked: Locked) -> dict:
    """`counts`: {test name: holm.Counts (b = second-only, c = first-only,
    n)} for exactly the lock's family."""
    if set(counts) != set(locked.h4_tests):
        raise HypothesisError(f"H4's tests are the lock's {len(locked.h4_tests)}; got "
                              f"{len(counts)}, differing in "
                              f"{sorted(set(counts) ^ set(locked.h4_tests))}")
    m = len(locked.h4_tests)
    p = {t: _tost_p(counts[t], locked.h4_bound_pts) for t in counts}
    holm, stopped = {}, False
    for i, t in enumerate(sorted(p, key=lambda t: (p[t], t))):
        step = locked.h4_alpha / (m - i)
        ok = (not stopped) and p[t] <= step
        stopped = stopped or not ok
        holm[t] = ok
    short = sorted(t for t in counts if counts[t].n < locked.h4_n)
    rows = {t: {"p": p[t], "n": counts[t].n, "equivalent": holm[t],
                "verdict": INCONCLUSIVE if t in short else (HOLDS if holm[t] else FAILS)}
            for t in sorted(counts)}
    vs = [r["verdict"] for r in rows.values()]
    return {"hypothesis": "H4", "m": m, "bound_pts": locked.h4_bound_pts,
            "alpha": locked.h4_alpha, "n_required": locked.h4_n, "tests": rows,
            "below_power": short,
            "verdict": INCONCLUSIVE if short else (HOLDS if all(holm.values()) else FAILS)}


# --- H5 -------------------------------------------------------------------------

def h5(predicted_sign: Mapping, first: Mapping, second: Mapping, *, floors: FloorTable,
       gpu_only: str) -> dict:
    """`predicted_sign`: {estimator: +1 (crossover moves longer) or -1},
    recorded before data; `first`, `second`: {estimator: [GridDifference]}
    on the two pairings (the fixed host's slower and faster GPU)."""
    rows = {}
    for e in sorted(predicted_sign):
        err = X.crossover_error(X.crossover_brackets(first[e], floors=floors),
                                X.crossover_brackets(second[e], floors=floors))
        if err.status != "compared":
            rows[e] = {"status": err.status, "verdict": NOT_SCORED}
            continue
        shift = err.comparisons[0].shift_grid_steps    # second relative to first
        sign = 0 if shift == 0 else int(math.copysign(1, shift))
        rows[e] = {"shift_grid_steps": shift, "predicted_sign": predicted_sign[e],
                   "verdict": HOLDS if sign == predicted_sign[e] else FAILS}
    b = NOT_SCORED
    if gpu_only in rows and "shift_grid_steps" in rows[gpu_only]:
        others = [r["shift_grid_steps"] for e, r in rows.items()
                  if e != gpu_only and "shift_grid_steps" in r]
        if others:
            g = abs(rows[gpu_only]["shift_grid_steps"])
            b = HOLDS if all(g < abs(s) for s in others) else FAILS
    return {"hypothesis": "H5", "secondary": True, "a": rows, "b": b,
            "c": {"verdict": NOT_SCORED, "why": "no pairing run puts one GPU on two hosts"}}
