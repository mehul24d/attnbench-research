"""Gate A over P1's repeat sessions (T3.2a and T3.2b; plan Section 7, T3.3),
with the floors their ground truth gives (Section 5.6; `lock.floors`).

Each session is read on its own by `gate_a.report`: its committed
prediction against its own ground truth, its own replay-versus-trace
reports. This module adds what needs both:

  - **the floors** (`floors`): the noise floor from the between-session
    spread of each cell's median, and the resolution floor from the
    within-session spread of every series, both at a quantile the caller
    names (a lock value; before the lock it is labelled provisional), with
    the per-cell table the noise floor is the quantile of;
  - **the plan's thresholds, as written** (`as_written`): leg 1 holds when
    the recursion's median absolute relative error is below the sum's and the
    step form's over every cell, in every session (a session where it is not
    is a failure, never averaged away); leg 2 holds when a lock bound is set
    and no cell of any session is outside it, and is not applied with no
    bound or a provisional one; leg 3 holds when the slopes are measured
    within a lock bound in every session, and is not applied when they are
    not measured. Gate A passes only when all three hold. Nothing here
    widens a threshold: a leg that cannot be applied is reported as such,
    and the gate does not pass on it.

A cell is named across sessions by its row label, its token budget and its
batch (`cell_key`): XAttention's executed density is its estimator's and
differs between sessions, so it is not part of the name.

The decision stays the researcher's (T3.3); `as_written` is the reading the
plan's text gives, for the researcher to check.
"""

from __future__ import annotations

import statistics
from typing import Mapping, Optional, Sequence

from ..lock import floors as F
from .gate_a import GateAError, NOT_MEASURED, short_of_rounds

HOLDS, FAILS, NOT_APPLIED = "holds", "fails", "not applied"
LEG1_RIVALS = ("sum", "step_form")


def cell_key(row: Mapping) -> tuple:
    """(row label, token budget, batch): the cell's name in every session."""
    return (row["arm"], int(row["budget"]), int(row["batch"]))


def cell_series(results: Sequence[tuple[str, Mapping]]) -> dict:
    """{cell key: clean repeats in ms} from one session's ground-truth
    results; a traced result, and a cell timed twice, are refused."""
    out: dict = {}
    for name, result in results:
        F.clean_series(result)                     # refuses a traced result
        for row in result.get("rows", ()):
            k = cell_key(row)
            if k in out:
                raise GateAError(f"{name}: cell {k} is timed twice in one session")
            out[k] = list(row["prefill_ms_samples"])
    if not out:
        raise GateAError("no ground-truth rows")
    return out


def floors(sessions: Mapping[str, Sequence[tuple[str, Mapping]]], *, quantile: float,
           provisional: bool, pairing: str) -> dict:
    """Both floors from the sessions' ground truth, and per cell each
    session's median and within-session spread and the between-session
    spread."""
    series = {s: cell_series(r) for s, r in sessions.items()}
    noise = F.noise_floor(series, quantile=quantile, pairing=pairing)
    res = F.resolution_floor({(s, k): xs for s, cells in series.items() for k, xs in cells.items()},
                             quantile=quantile, pairing=pairing)
    cells = sorted({k for c in series.values() for k in c})
    table = []
    for k in cells:
        table.append({
            "cell": list(k),
            "median_ms": {s: (statistics.median(series[s][k]) if k in series[s] else None)
                          for s in series},
            "within_session_spread": {s: noise.within_session_spread.get((s, k)) for s in series},
            "between_session_spread": noise.between_session_spread.get(k)})
    return {"quantile": quantile, "provisional": provisional,
            "noise_floor_relative": noise.floor_relative, "noise_procedure": noise.procedure,
            "noise_inputs_sha256": noise.inputs_sha256, "n_cells_both": noise.n_cells,
            "resolution_floor_ms": res.floor_ms, "resolution_procedure": res.procedure,
            "resolution_inputs_sha256": res.inputs_sha256, "n_series": res.n_series,
            "n_repeats": res.n_repeats, "cells": table,
            "only_in_one_session": [list(k) for k in cells
                                    if sum(k in c for c in series.values()) < 2]}


def drop_short_groups(sessions: Mapping[str, Sequence[tuple[str, Mapping]]], *,
                      min_clean_rounds: int) -> tuple[dict, list]:
    """(kept sessions, excluded groups): a ground-truth group that is not
    complete or has fewer than `min_clean_rounds` clean rounds
    (`gate_a.short_of_rounds`) leaves the floors whole, within- and
    between-session, as it leaves leg 1 (Appendix A, 9 Nov 2026, ninth paste)."""
    kept, excluded = {}, []
    for s, results in sessions.items():
        kept[s] = []
        for name, result in results:
            if short_of_rounds(result, min_clean_rounds):
                excluded.append({"session": s, "result": name, "complete": result.get("complete"),
                                 "clean_rounds": result.get("clean_rounds"),
                                 "min_clean_rounds": min_clean_rounds})
            else:
                kept[s].append((name, result))
    return kept, excluded


def cell_floors(sessions: Mapping[str, Sequence[tuple[str, Mapping]]], *, quantile: float,
                pairing: str) -> dict:
    """The resolution floor per (context length, batch) by the amended rule
    (Appendix A, 8 Nov 2026, fifth paste; `floors.resolution_floors_within_between`)
    over every session's ground truth: {"<length>x<batch>": {floor_ms,
    within_ms, between_ms, n_between_series, source}}, keyed as
    `scripts/gate_a.py --cell-floors` reads it."""
    series = {(s, *k): xs for s, r in sessions.items() for k, xs in cell_series(r).items()}
    got = F.resolution_floors_within_between(series, length_of=lambda k: (k[2], k[3]),
                                             quantile=quantile, pairing=pairing)
    return {f"{L}x{b}": {"floor_ms": c.floor_ms, "within_ms": c.within_ms,
                         "between_ms": c.between_ms, "n_between_series": c.n_between_series,
                         "source": c.source}
            for (L, b), c in got.items()}


def _leg1(reports: Mapping[str, Mapping]) -> dict:
    """Carries each session's cell count and exclusions; a session whose
    report states no cell read is not applied (adviser's ruling 2d of
    11 Nov 2026; T4.3 read A, finding 7e)."""
    per, n, excluded = {}, {}, {}
    for s, rep in reports.items():
        scope = rep["predictors"]["all"]
        below = scope["recursion_below"]
        per[s] = {p: bool(below[p]) for p in LEG1_RIVALS}
        n[s] = ((scope.get("by_predictor") or {}).get("recursion") or {}).get("n")
        excluded[s] = {"excluded_groups": rep.get("excluded_groups", []),
                       "unmeasured": rep.get("unmeasured", [])}
    rule = ("the recursion's median absolute relative error below the sum's and the "
            "step form's over every cell, in every session")
    unread = [s for s, k in n.items() if not (isinstance(k, int) and k > 0)]
    if unread:
        return {"verdict": NOT_APPLIED, "n_cells": n, "recursion_below": per, "rule": rule,
                "why": f"no cell count in {unread}'s leg 1"}
    ok = all(all(v.values()) for v in per.values())
    return {"verdict": HOLDS if ok else FAILS, "n_cells": n, "exclusions": excluded,
            "recursion_below": per, "rule": rule}


def _leg2(reports: Mapping[str, Mapping]) -> dict:
    fams = {s: rep["families"] for s, rep in reports.items()}
    bounds = {(f["bound"], f["bound_provisional"]) for f in fams.values()}
    if len(bounds) != 1:
        raise GateAError(f"the sessions were read at different family bounds: {sorted(bounds, key=str)}")
    bound, provisional = bounds.pop()
    if bound is None or provisional:
        return {"verdict": NOT_APPLIED, "bound": bound, "bound_provisional": provisional,
                "why": "the per-family bound is fixed at lock (Appendix A, 27 Oct 2026); "
                       + ("none is set" if bound is None else "this one is provisional")}
    failing = {s: f["failing_cells"] for s, f in fams.items() if f["failing_cells"]}
    return {"verdict": FAILS if failing else HOLDS, "bound": bound, "failing_cells": failing}


def _leg3(reports: Mapping[str, Mapping]) -> dict:
    sl = {s: rep["slopes"] for s, rep in reports.items()}
    if any(v["status"] == NOT_MEASURED for v in sl.values()):
        return {"verdict": NOT_APPLIED, "why": "the slopes are not measured in "
                + ", ".join(s for s, v in sl.items() if v["status"] == NOT_MEASURED)}
    ranged = [s for s, v in sl.items() if "within_range" in v]
    if ranged:                     # Gate A v2's leg 3 (Appendix A, 8 Nov 2026)
        if len(ranged) != len(sl):
            raise GateAError(f"only {ranged} were read at Gate A v2's slope range")
        if any(not v["within_range"] for v in sl.values()):
            raise GateAError("a v2 leg 3 with no mechanism holds vacuously")
        ok = all(all(v["within_range"].values()) and not v.get("missing") for v in sl.values())
        return {"verdict": HOLDS if ok else FAILS,
                "within_range": {s: v["within_range"] for s, v in sl.items()},
                "missing": {s: v.get("missing", []) for s, v in sl.items()}}
    if any("within_bound" not in v for v in sl.values()):
        return {"verdict": NOT_APPLIED, "why": "no slope bound is set (a lock value)"}
    ok = all(v["within_bound"] for v in sl.values())
    return {"verdict": HOLDS if ok else FAILS, "within_bound": {s: v["within_bound"] for s, v in sl.items()}}


def as_written(reports: Mapping[str, Mapping]) -> dict:
    """Each leg's verdict under the plan's text, and the gate's: it passes
    only when every leg holds."""
    if len(reports) < 2:
        raise GateAError(f"Gate A reads both repeat sessions, got {len(reports)}")
    legs = {"leg1": _leg1(reports), "leg2": _leg2(reports), "leg3": _leg3(reports)}
    passes = all(v["verdict"] == HOLDS for v in legs.values())
    return {"legs": legs, "gate": "passes" if passes else "does not pass",
            "not_holding": [k for k, v in legs.items() if v["verdict"] != HOLDS],
            "decision": "the researcher's (T3.3); this is the plan's text applied to the inputs"}


def report(*, label: str, reports: Mapping[str, Mapping], floors_: Mapping) -> dict:
    return {"label": label, "sessions": {s: r["prediction_sha256"] for s, r in reports.items()},
            "floors": floors_, "as_written": as_written(reports), "per_session": dict(reports)}


def to_markdown(rep: Mapping) -> str:
    fl, aw = rep["floors"], rep["as_written"]
    sessions = list(rep["sessions"])
    tag = " (provisional quantile, not a lock value)" if fl["provisional"] else ""
    L = [f"# Gate A over the repeats: {rep['label']}", "",
         "Sessions: " + ", ".join(f"{s} (prediction `{h[:12]}`)" for s, h in rep["sessions"].items())
         + f". Decision: {aw['decision']}.", "",
         f"## The plan's thresholds as written: Gate A {aw['gate']}", "",
         "| Leg | Verdict | Detail |", "|---|---|---|"]
    l1 = aw["legs"]["leg1"]
    L.append(f"| 1: recursion below the sum and the step form | {l1['verdict']} | "
             + "; ".join(f"{s}: " + ", ".join(f"{p} {'yes' if b else 'no'}" for p, b in v.items())
                         for s, v in l1["recursion_below"].items()) + " |")
    l2, l3 = aw["legs"]["leg2"], aw["legs"]["leg3"]
    L.append(f"| 2: replay against trace per family | {l2['verdict']} | "
             f"{l2.get('why') or ('failing cells: ' + str(sum(map(len, l2['failing_cells'].values()))))} |")
    L.append(f"| 3: delay-injection slopes | {l3['verdict']} | {l3.get('why', '')} |")
    L += ["", f"## Floors at quantile {fl['quantile']}{tag}", "",
          f"- Noise floor (between-session spread of a cell's median, (max - min) / median, "
          f"over {fl['n_cells_both']} cells timed in both sessions): **{fl['noise_floor_relative']:.4f}**.",
          f"- Resolution floor (|repeat - series median|, pooled over {fl['n_series']} series, "
          f"{fl['n_repeats']} repeats): **{fl['resolution_floor_ms']:.2f} ms**.",
          f"- Cells timed in one session only: {len(fl['only_in_one_session'])}.", "",
          "| Cell | " + " | ".join(f"{s} median ms" for s in sessions) + " | Between | "
          + " | ".join(f"{s} within" for s in sessions) + " |",
          "|---|" + "---|" * (2 * len(sessions) + 1)]
    for c in fl["cells"]:
        b = c["between_session_spread"]
        L.append(f"| {c['cell'][0]} L{c['cell'][1]} b{c['cell'][2]} | "
                 + " | ".join("" if c["median_ms"][s] is None else f"{c['median_ms'][s]:.1f}"
                              for s in sessions)
                 + f" | {'' if b is None else f'{b:.4f}'} | "
                 + " | ".join("" if c["within_session_spread"][s] is None
                              else f"{c['within_session_spread'][s]:.4f}" for s in sessions) + " |")
    return "\n".join(L) + "\n"
