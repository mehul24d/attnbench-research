"""Gate A's inputs on P1 (plan Section 7, T3.3), from one session's banked
files: the committed prediction, read through `load_committed`, the
untraced ground truth, step 5's recordings and step 6's priced cells.

Gate A has three legs, and this module reports each; the decision is the
researcher's (T3.3), never made here.

  1. **The recursion against the sum and the step form** (`predictors`):
     per predictor, overall and per arm class (dense, synchronisation-free,
     synchronising, as H1 reports them), the count and the median absolute
     relative error, from `predict.residuals`. Fitted baselines do not exist
     on P1 (Section 5.4), so the comparison is with the sum and the step
     form only.
  2. **Replay against trace, per operation family** (`families`): for each
     cell, the device time the prediction priced at each position (step 6's
     priced cell: the in-order graph replay where it holds the position, the
     isolated original elsewhere, each position labelled by its source)
     against the same position's device time in step 5's traced recording,
     summed per family (`replay.replay_vs_trace`). The bound is a lock value
     set from this distribution (T4.1; Appendix A, 27 Oct 2026); a run before
     the lock passes a bound it names as provisional, or none, and the
     report gives the distribution either way.
  3. **The delay-injection slope check** (`slopes`): one predicted-over-
     measured slope per synchronisation mechanism (`predict.delay_injection`).
     T3.1's step 12 runs it since 8 Nov 2026 (`components.delay_session`);
     a session without it reports the leg as not measured, and Gate A cannot
     be read from it. Gate A v2 (Appendix A, 8 Nov 2026) reads each slope
     against [1/sqrt 2, sqrt 2] (`V2_SLOPE_RANGE`) and exempts, per cell, a
     leg-2 family below the cell's resolution floor (`exempt_families`).

Ground-truth rows name their cell by a label (`time_arms`' block names):
`<arm>@<nominal density>` for an arm run at a nominal density (oracle,
mean-pool), `<arm>` for one with a single density (dense, XAttention at its
calibrated threshold). `measured` maps each row to exactly one committed
cell and refuses a row it cannot place, a row two cells fit, and a cell
measured twice; a committed cell with no row is listed, never dropped.

`sanity` is a descriptive within-session summary per arm (median absolute
relative error, and sparse-versus-dense sign agreement with each arm's
dense cell of the same length and batch): it is labelled descriptive and
is not a hypothesis test (P1 is the development pairing, Section 4, H1).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..figures import results as RS
from ..predict import residuals as R
from ..predict import scoring
from ..predict.commit import CommittedPrediction, require_loaded

GATE_PREDICTORS = ("recursion", "sum", "step_form")
# A ground-truth group with fewer clean rounds than this, or not complete, is
# listed and left out of every aggregate, never dropped silently (Appendix A,
# 9 Nov 2026; adviser's ruling 5 of the seventh paste: the minimum stays 10).
MIN_CLEAN_ROUNDS = 10
SCOPES = ("all", R.DENSE, R.SYNC_FREE, R.SYNCHRONISING)
# A label's density and a calibrated cell's density agree to the precision
# the prediction writes it at (six decimals).
DENSITY_TOL = 1e-6
DESCRIPTIVE = ("descriptive, within one session: not a hypothesis test (P1 is the "
               "development pairing)")
NOT_MEASURED = "not measured"


class GateAError(ValueError):
    pass


def _label(label: str) -> tuple[str, Optional[float]]:
    arm, sep, d = label.partition("@")
    if not sep:
        return arm, None
    try:
        return arm, float(d)
    except ValueError:
        raise GateAError(f"row label {label!r}: {d!r} is not a density") from None


def place_row(cp: CommittedPrediction, row: Mapping, name: str):
    """The one committed cell a ground-truth row measures, or GateAError."""
    label = row.get("arm")
    if not isinstance(label, str):
        raise GateAError(f"{name}: a row with no arm label")
    arm, nominal = _label(label)
    fits = [c for c in require_loaded(cp).cells
            if c.arm == arm and c.model == row.get("model")
            and c.context_length == row.get("context_length")
            and c.batch == row.get("batch")
            and (c.density == nominal if nominal is not None
                 else abs(c.density - float(row.get("executed_density") or 0.0))
                 <= DENSITY_TOL)]
    if len(fits) != 1:
        raise GateAError(f"{name}: row {label!r} at {row.get('context_length')} "
                         f"x {row.get('batch')} fits {len(fits)} committed cells")
    return fits[0]


def short_of_rounds(result: Mapping, min_clean_rounds: int = MIN_CLEAN_ROUNDS) -> bool:
    """An untraced group that is not complete, or has fewer clean rounds than
    `min_clean_rounds`."""
    if result.get("traced") is not False:
        return False                       # refused by `check_result`, not excluded
    clean = result.get("clean_rounds")
    return result.get("complete") is not True or (clean is not None
                                                  and int(clean) < min_clean_rounds)


def measured(cp: CommittedPrediction, results: Sequence[tuple[str, Mapping]], *,
             min_clean_rounds: int = MIN_CLEAN_ROUNDS) -> dict:
    """{"seconds": {cell key: seconds}, "unmeasured": [cell keys], "excluded":
    [groups]} from (name, ground-truth result) pairs, each row placed on one
    committed cell. A group short of clean rounds (`short_of_rounds`) is
    listed under "excluded" with its rounds, and its cells are unmeasured."""
    cp = require_loaded(cp)
    out: dict = {}
    excluded = []
    for name, result in results:
        if short_of_rounds(result, min_clean_rounds):
            excluded.append({"result": name, "complete": result.get("complete"),
                             "clean_rounds": result.get("clean_rounds"),
                             "rounds_run": result.get("rounds_run"),
                             "min_clean_rounds": min_clean_rounds,
                             "arms": [row.get("arm") for row in result.get("rows", ())]})
            continue
        RS.check_result(result, name)
        for row in result.get("rows", ()):
            k = place_row(cp, row, name).key
            label = row["arm"]
            if k in out:
                raise GateAError(f"cell {k} is measured twice")
            out[k] = RS.measured_seconds(result, label, cp.statistic, name)
    if not out:
        raise GateAError(f"no measured cells ({len(excluded)} groups short of clean rounds)")
    return {"seconds": out, "unmeasured": sorted(c.key for c in cp.cells if c.key not in out),
            "excluded": excluded}


def _residuals(cp, seconds, *, dense_arm, relative_floor, predictors):
    rs = R.residuals(cp, seconds, statistic=cp.statistic, dense_arms=[dense_arm],
                     relative_floor=relative_floor, predictors=list(predictors))
    if rs.unpredicted:
        raise GateAError(f"measured cells with no committed prediction: {rs.unpredicted}")
    return rs.residuals


def _summary(xs) -> dict:
    a = [r.abs_relative for r in xs]
    return {"n": len(xs), "median_abs_relative": float(statistics.median(a)),
            "mean_relative": float(statistics.mean(r.relative for r in xs)),
            "max_abs_relative": float(max(a)),
            "n_inside_floor": sum(r.inside_floor for r in xs)}


def predictors(cp: CommittedPrediction, seconds: Mapping, *, dense_arm: str,
               relative_floor: float, predictors: Sequence[str] = GATE_PREDICTORS) -> dict:
    """Leg 1: per scope (every cell, then each arm class) and predictor, the
    summary, and whether the recursion's median absolute relative error is
    below each other predictor's (a strict inequality; a tie is not below)."""
    if "recursion" not in predictors:
        raise GateAError("leg 1 compares the recursion with the others: name it")
    res = _residuals(cp, seconds, dense_arm=dense_arm, relative_floor=relative_floor,
                     predictors=predictors)
    out = {}
    for scope in SCOPES:
        xs = [r for r in res if scope == "all" or r.arm_class == scope]
        if not xs:
            out[scope] = None
            continue
        by = {p: _summary([r for r in xs if r.predictor == p]) for p in predictors}
        rec = by["recursion"]["median_abs_relative"]
        out[scope] = {"by_predictor": by,
                      "recursion_below": {p: rec < by[p]["median_abs_relative"]
                                          for p in predictors if p != "recursion"}}
    return out


def sanity(cp: CommittedPrediction, seconds: Mapping, *, dense_arm: str,
           floors, predictors: Sequence[str] = GATE_PREDICTORS) -> dict:
    """Per arm and predictor: the median absolute relative error, and the
    sparse-versus-dense sign agreement (`scoring.sign_agreement`) of each of
    the arm's cells with the dense cell of the same model, length and batch,
    a measured difference inside the cell's floor in `floors` (the
    per-(length, batch) `FloorTable`) not scored."""
    from attnbench.lock.floors import FloorsError, as_table
    try:
        floors = as_table(floors)
    except FloorsError as e:
        raise GateAError(str(e)) from e
    res = _residuals(cp, seconds, dense_arm=dense_arm, relative_floor=0.0,
                     predictors=predictors)
    cells = {c.key: c for c in cp.cells}
    dense = {(k.model, k.context_length, k.batch): k for k in seconds if k.arm == dense_arm}
    out: dict = {"label": DESCRIPTIVE, "resolution_floors": floors.to_json(), "arms": {}}
    for arm in sorted({r.arm for r in res}):
        row: dict = {}
        for p in predictors:
            xs = [r for r in res if r.arm == arm and r.predictor == p]
            entry = {"n": len(xs),
                     "median_abs_relative": float(statistics.median(r.abs_relative for r in xs))}
            if arm != dense_arm:
                agree = disagree = inside = 0
                for k in (k for k in seconds if k.arm == arm):
                    d = dense.get((k.model, k.context_length, k.batch))
                    if d is None:
                        continue
                    s = scoring.sign_agreement(cells[k].predictions[p], cells[d].predictions[p],
                                               seconds[k], seconds[d],
                                               floor=floors.at(k.context_length, k.batch))
                    if s is None:
                        inside += 1
                    else:
                        agree += s
                        disagree += not s
                entry.update(sign_agree=agree, sign_disagree=disagree, inside_floor=inside)
            row[p] = entry
        out["arms"][arm] = row
    return out


def recorded_sequence(trace: Path, sequence: Path):
    """Step 5's sequence with the correlations `op_sequence.device_durations`
    reads: parsed again from the recording's trace, and refused unless the
    parse is the banked `sequence.json` (its own sha256, which a sequence
    record on disk keeps without correlations)."""
    from ..components import op_sequence as ops
    rec = json.loads(Path(sequence).read_text())
    sha = rec.pop("sha256", None)
    if ops.record_sha256(rec) != sha:
        raise GateAError(f"{sequence} does not match its own sha256")
    banked = ops.from_record(rec)
    seq = ops.parse_trace(trace, banked.code_path, context_length=banked.context_length,
                          batch=banked.batch)
    if ops.record_sha256(ops.to_record(seq)) != sha:
        raise GateAError(f"{trace} does not parse to the banked {sequence}")
    return seq


def replay_report(seq, trace_s: Mapping[int, float], priced_ops: Sequence[Mapping]) -> dict:
    """One cell's replay-versus-trace report: the priced device time at each
    position the trace timed (`priced_ops` is step 6's cell `ops`, one per
    sequence position) against the trace's (`op_sequence.device_durations`)."""
    from ..components import replay
    if len(priced_ops) != len(seq.ops):
        raise GateAError(f"{len(priced_ops)} priced operations for a sequence of "
                         f"{len(seq.ops)}")
    priced = {i: float(priced_ops[i]["device"]) for i in trace_s}
    return replay.replay_vs_trace(seq, trace_s, priced)


V2_SLOPE_RANGE = (2 ** -0.5, 2 ** 0.5)   # Gate A v2's leg 3 (Appendix A, 8 Nov 2026)


def exempt_families(rep: Mapping, floor_s: float) -> list[str]:
    """Gate A v2's leg 2 exemption in one cell: the families whose traced
    device time is below the cell's resolution floor."""
    return sorted(f for f, r in rep["families"].items() if r["trace_s"] < floor_s)


def families(reports: Mapping[str, Mapping], *, bound: Optional[float],
             bound_provisional: bool,
             cell_floors_s: Optional[Mapping[str, float]] = None) -> dict:
    """Leg 2 over every cell's report: per family, the cells and the spread
    of the relative difference, pooled sums, and, at a bound, each cell's
    `replay.check_passes`. With `cell_floors_s` (report name -> that cell's
    resolution floor, seconds) it is Gate A v2's leg 2: a family below its
    cell's floor is exempt there (`exempt_families`), and listed; a cell
    with no floor is refused."""
    from ..components import replay
    if not reports:
        raise GateAError("no cell's replay-versus-trace report")
    fam: dict = {}
    for name, rep in reports.items():
        for f, r in rep["families"].items():
            e = fam.setdefault(f, {"cells": 0, "n": 0, "trace_s": 0.0, "replay_s": 0.0,
                                   "relative_differences": []})
            e["cells"] += 1
            e["n"] += r["n"]
            e["trace_s"] += r["trace_s"]
            e["replay_s"] += r["replay_s"]
            if r["relative_difference"] is not None:
                e["relative_differences"].append(r["relative_difference"])
    for e in fam.values():
        d = sorted(abs(x) for x in e.pop("relative_differences"))
        e["pooled_relative_difference"] = ((e["replay_s"] - e["trace_s"]) / e["trace_s"]
                                           if e["trace_s"] > 0 else None)
        e["abs_relative_difference"] = (
            {"median": statistics.median(d), "max": d[-1],
             "q90": d[min(len(d) - 1, int(0.9 * len(d)))]} if d else None)
    out = {"families": dict(sorted(fam.items())), "cells": len(reports),
           "bound": bound, "bound_provisional": bound_provisional}
    if cell_floors_s is not None:
        missing = sorted(set(reports) - set(cell_floors_s))
        if missing:
            raise GateAError(f"no resolution floor for cells {missing}")
        out["exempt"] = {n: exempt_families(rep, cell_floors_s[n])
                         for n, rep in sorted(reports.items())}
        out["cell_floors_s"] = {n: cell_floors_s[n] for n in sorted(reports)}
    if bound is not None:
        failing = {}
        for name, rep in sorted(reports.items()):
            if cell_floors_s is not None:
                drop = set(out["exempt"][name])
                rep = {**rep, "families": {f: r for f, r in rep["families"].items()
                                           if f not in drop}}
            ok, why = replay.check_passes(rep, bound)
            if not ok:
                failing[name] = why
        out["failing_cells"] = failing
    return out


def slopes(measured_slopes: Optional[Mapping], *, bound: Optional[float],
           mechanisms: Sequence[str], slope_range: Optional[tuple] = None) -> dict:
    """Leg 3: `delay_injection.gate` on the measured slopes, or not measured.
    With `slope_range` (Gate A v2: `V2_SLOPE_RANGE`), each listed mechanism's
    slope must lie in [low, high], and a listed mechanism with no slope is
    outside it."""
    if slope_range is not None and bound is not None:
        raise GateAError("leg 3 takes a bound about 1.0 or a range, not both")
    if measured_slopes and slope_range is not None:
        lo, hi = slope_range
        got = {m: s.slope for m, s in measured_slopes.items()}
        return {"status": "measured", "range": [lo, hi],
                "within_range": {m: m in got and lo <= got[m] <= hi for m in mechanisms},
                "missing": [m for m in mechanisms if m not in got], "slopes": got}
    if not measured_slopes:
        return {"status": NOT_MEASURED,
                "why": "the session ran no delay injection (T2.25); Gate A needs one slope "
                       "per synchronisation mechanism on the L4, busy and idle"}
    from ..predict import delay_injection as DI
    if bound is None:
        return {"status": "measured, no bound", "slopes":
                {m: s.slope for m, s in measured_slopes.items()}}
    return {"status": "measured", "within_bound": DI.gate(measured_slopes, bound=bound,
                                                          mechanisms=mechanisms),
            "slopes": {m: s.slope for m, s in measured_slopes.items()}, "bound": bound}


def report(*, label: str, cp: CommittedPrediction, results: Sequence[tuple[str, Mapping]],
           dense_arm: str, relative_floor: float, floors,
           family_reports: Mapping[str, Mapping], family_bound: Optional[float],
           family_bound_provisional: bool, measured_slopes: Optional[Mapping] = None,
           slope_bound: Optional[float] = None, mechanisms: Sequence[str] = (),
           cell_floors_s: Optional[Mapping[str, float]] = None,
           slope_range: Optional[tuple] = None) -> dict:
    """Every leg, labelled; `decision` is always the researcher's."""
    m = measured(cp, results)
    secs = m["seconds"]
    return {
        "label": label, "prediction_sha256": cp.sha256, "pairing": cp.pairing,
        "statistic": cp.statistic, "relative_floor": relative_floor,
        "measured_cells": len(secs), "unmeasured": [list(k) for k in m["unmeasured"]],
        "excluded_groups": m["excluded"],
        "predictors": predictors(cp, secs, dense_arm=dense_arm,
                                 relative_floor=relative_floor),
        "families": families(family_reports, bound=family_bound,
                             bound_provisional=family_bound_provisional,
                             cell_floors_s=cell_floors_s),
        "slopes": slopes(measured_slopes, bound=slope_bound, mechanisms=mechanisms,
                         slope_range=slope_range),
        "sanity": sanity(cp, secs, dense_arm=dense_arm, floors=floors),
        "decision": "the researcher's (T3.3); this report gives the inputs only",
    }


def to_markdown(rep: Mapping) -> str:
    """The report for reading: leg 1's table, leg 2's per-family spread, leg
    3's status, and the descriptive per-arm table."""
    L = [f"# Gate A inputs: {rep['label']}", "",
         f"Prediction `{rep['prediction_sha256'][:12]}`, {rep['pairing']}, the "
         f"{rep['statistic']}; {rep['measured_cells']} cells measured, "
         f"{len(rep['unmeasured'])} committed cells not measured. Decision: {rep['decision']}.",
         *([""] + [f"Excluded, short of {g['min_clean_rounds']} clean rounds: `{g['result']}` "
                  f"({g['clean_rounds']} of {g['rounds_run']} rounds clean; arms "
                  f"{', '.join(map(str, g['arms']))})."
                  for g in rep.get("excluded_groups", ())]),
         "", "## Leg 1: the recursion against the sum and the step form", "",
         "| Scope | Predictor | n | Median abs. rel. error | Signed mean | Max |",
         "|---|---|---|---|---|---|"]
    for scope, s in rep["predictors"].items():
        if s is None:
            L.append(f"| {scope} | (no cells) | | | | |")
            continue
        for p, v in s["by_predictor"].items():
            L.append(f"| {scope} | {p} | {v['n']} | {v['median_abs_relative']:.4f} | "
                     f"{v['mean_relative']:+.4f} | {v['max_abs_relative']:.4f} |")
        below = ", ".join(f"{p}: {'yes' if b else 'no'}" for p, b in s["recursion_below"].items())
        L.append(f"| {scope} | recursion below | | {below} | | |")
    fam = rep["families"]
    bound = ("none" if fam["bound"] is None else
             f"{fam['bound']}{' (provisional)' if fam['bound_provisional'] else ''}")
    L += ["", f"## Leg 2: replay against trace, per family ({fam['cells']} cells; bound {bound})",
          "", "| Family | Cells | Positions | Pooled rel. diff. | Median abs. | q90 abs. | Max abs. |",
          "|---|---|---|---|---|---|---|"]
    for f, e in fam["families"].items():
        a = e["abs_relative_difference"] or {}
        pooled = e["pooled_relative_difference"]
        L.append(f"| {f} | {e['cells']} | {e['n']} | "
                 f"{'n/a' if pooled is None else f'{pooled:+.4f}'} | "
                 f"{a.get('median', float('nan')):.4f} | {a.get('q90', float('nan')):.4f} | "
                 f"{a.get('max', float('nan')):.4f} |")
    if "exempt" in fam:
        n = sum(len(v) for v in fam["exempt"].values())
        L += ["", f"Gate A v2: {n} (cell, family) pairs below their cell's resolution floor "
                  f"are exempt (listed in the JSON)."]
    if "failing_cells" in fam:
        L += ["", f"Cells outside the bound: {len(fam['failing_cells'])}."]
    sl = rep["slopes"]
    L += ["", "## Leg 3: delay-injection slopes", "", f"{sl['status']}: {sl.get('why', '')}".rstrip(": ")]
    if "within_range" in sl:
        lo, hi = sl["range"]
        L += ["", f"Range [{lo:.3f}, {hi:.3f}].", "", "| Mechanism | Slope | Within |", "|---|---|---|"]
        for m, ok in sl["within_range"].items():
            v = sl["slopes"].get(m)
            L.append(f"| {m} | {'n/a' if v is None else f'{v:.3f}'} | {'yes' if ok else 'no'} |")
    L += [
          "", f"## Per arm ({rep['sanity']['label']})", "",
          f"Sign agreement against the dense cell of the same length and batch; a measured "
          "difference within its cell's resolution floor ("
          + ", ".join(f"{k} {v * 1e3:.1f} ms"
                      for k, v in rep['sanity']['resolution_floors']['floors_s'].items())
          + ") is not scored.",
          "", "| Arm | Predictor | n | Median abs. rel. error | Sign agrees | Disagrees | Inside floor |",
          "|---|---|---|---|---|---|---|"]
    for arm, row in rep["sanity"]["arms"].items():
        for p, v in row.items():
            L.append(f"| {arm} | {p} | {v['n']} | {v['median_abs_relative']:.4f} | "
                     f"{v.get('sign_agree', '')} | {v.get('sign_disagree', '')} | "
                     f"{v.get('inside_floor', '')} |")
    return "\n".join(L) + "\n"


def write(rep: Mapping, out: Path) -> tuple[Path, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    j, md = out / "gate_a.json", out / "gate_a.md"
    j.write_text(json.dumps(rep, indent=1, default=list))
    md.write_text(to_markdown(rep))
    return j, md

