"""EXPLORATORY: the residual after device substitution, split where the data
allow, and the dispatcher's margin sensitivity (the adviser's task of 22 Nov
2026, item 2). Nothing here scores a hypothesis; H0-H3, the analysis commit
83b2edf and the 17 Nov 2026 margin 0.02264 are untouched.

CPU only and offline, from the same inputs as `exploratory_wave.py` (the
scoring wave's mirror and local trace copies):

  - per cell, with every device time the traced one
    (`breakdown.substitute`): the prediction's four terms, the predicted and
    measured device idle time, and the counterfactuals of
    `attnbench.exploratory.residual`;
  - the trace inflation at short lengths: traced minus priced device time
    per kernel for the GEMM and other-device components;
  - the margin sweep: the abstaining dispatcher's regret on P1, P2 and P4
    at margins from 0 to 0.30 (post hoc: P2 and P4 were read before it).

    python3 scripts/exploratory_residual.py --inputs <inputs.json> --mirror <dir> \\
        --traces P2=<glob> --traces P4=<glob> --out <file.json>

The output names no store and no bucket.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exploratory_wave as EW  # noqa: E402
from attnbench.components import op_sequence as ops  # noqa: E402
from attnbench.exploratory import breakdown as BR  # noqa: E402
from attnbench.exploratory import residual as RS  # noqa: E402
from attnbench.exploratory import safe_dispatch as SD  # noqa: E402
from attnbench.exploratory import traces as TR  # noqa: E402

PAIRINGS = EW.PAIRINGS
MARGINS = (0.0, 0.01, 0.02, 0.02264, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


def device_busy(f) -> float:
    """Seconds of the union of device activities that start inside the
    trace's prefill region (the embedding call to the end of the timer's
    synchronise)."""
    tj = ops.load_trace(f)
    e0, s = TR.window(tj)
    lo, hi = e0["ts"], s["ts"] + s["dur"]
    iv = [(e["ts"], e["ts"] + e["dur"]) for e in tj["traceEvents"]
          if e.get("cat") in TR.DEVICE_CATS and e.get("ph") == "X" and lo <= e["ts"] <= hi]
    return RS.busy(iv) * 1e-6


def cells(cp, ms, recs, busy_of) -> list[dict]:
    by = {c.key: c for c in cp.cells}
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["key"]].append(r)
    out = []
    for k in sorted(groups):
        c = by[k]
        kinds = [x.value for x in c.priced.kinds]
        al = [TR.align(r["family"], r["device"], kinds) for r in groups[k]]
        fam = al[0][0]
        if any(a[0] != fam for a in al):
            raise TR.TraceError(f"{tuple(k)}: its traces differ in sequence")
        dev = [None if any(a[1][i] is None for a in al)
               else statistics.fmean(a[1][i] for a in al) for i in range(len(fam))]
        sub = BR.substitute(c.priced, dev, fam)
        b = statistics.fmean(busy_of[r["trace"]] for r in groups[k])
        sp = RS.split_cell(sub, cp.queue_depth, ms["seconds"][k], b)
        out.append({"arm": c.arm, "density": c.density, "context_length": c.context_length,
                    "batch": c.batch, "traces": len(groups[k]),
                    "trace_inflation": RS.per_kernel_excess(c.priced, dev, fam), **sp})
    return out


def summary(cs) -> dict:
    med = lambda xs: statistics.median(xs) if xs else None
    rel = lambda key: med([abs(c["counterfactual_rel"][key]) for c in cs])
    hs = [c["counterfactual_rel"]["host_scale_to_match"] for c in cs]
    hb = [c for c in cs if c["host"] > c["device_chain"]]
    return {"n": len(cs),
            "median_abs_rel": {k: rel(k) for k in ("as_predicted", "no_queue_term",
                                                    "sync_cost_zero", "no_sync_wait")},
            "host_scale_to_match": {"median": med([h for h in hs if h is not None]),
                                    "unreachable": sum(h is None for h in hs),
                                    "median_host_bound": med(
                                        [c["counterfactual_rel"]["host_scale_to_match"]
                                         for c in hb if c["counterfactual_rel"]
                                         ["host_scale_to_match"] is not None])},
            "overpredicted": RS.share_of_error(cs),
            "host_over_measured_above_1": sorted(c["host_over_measured"] for c in cs
                                                 if c["host_over_measured"] > 1),
            "idle_error_equals_error": max(abs(c["idle_error_s"] - c["error_s"]
                                               + c["chain_minus_busy"]) for c in cs),
            "max_abs_chain_minus_busy_over_measured": max(
                abs(c["chain_minus_busy"]) / c["measured"] for c in cs)}


def margin_sweep(D) -> dict:
    out = {}
    floors = D["locked"].floors
    for p in ("P1",) + PAIRINGS:
        cs = D["cps"][p].cells
        adm = SD.admissible_of([c.key for c in cs])
        meas = SD.measured_by_point(D["p1_measured"] if p == "P1" else D["ms"][p]["seconds"])
        pts = SD.points_for(cs, "recursion")
        rows = []
        for m in MARGINS:
            ch = {q.point: SD.abstaining(q, adm, floors, m) for q in pts}
            s = SD.score(pts, ch, meas, adm, floors)
            rows.append({"margin": m, "n": s.n, "mean_regret": s.mean_regret,
                         "worst_regret": s.worst_regret, "worst_vs_dense": s.worst_vs_dense,
                         "wins": s.wins, "wins_kept": s.wins_kept,
                         "sparse_chosen": sum(ch[q.point] != q.dense.config for q in pts)})
        errs = SD.advantage_errors(pts, meas, adm)
        out[p] = {"sweep": rows,
                  "advantage_overstatement": sorted(
                      ({"point": f"{q.context_length}|b{q.batch}", "config": f"{c.arm}|{c.density:g}",
                        "predicted_adv": a, "measured_adv": b, "overstatement": e}
                       for q, c, a, b, e in errs), key=lambda r: -r["overstatement"])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--mirror", required=True, type=Path)
    ap.add_argument("--traces", action="append", default=[], help="PAIRING=<glob>")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    globs = dict(t.split("=", 1) for t in a.traces)
    if set(globs) != set(PAIRINGS):
        ap.error(f"--traces for exactly {PAIRINGS}")
    D = EW.load(a.inputs, a.mirror)
    out = {"label": "EXPLORATORY", "task": "adviser's task of 22 Nov 2026, item 2"}
    for p in PAIRINGS:
        files = sorted(glob.glob(globs[p]))
        recs = EW.extract(D["cps"][p], files)
        busy_of = {Path(f).name: device_busy(f) for f in files}
        cs = cells(D["cps"][p], D["ms"][p], recs, busy_of)
        out[p] = {"queue_depth": D["cps"][p].queue_depth, "summary": summary(cs), "cells": cs}
    out["margin_sweep"] = margin_sweep(D)
    text = json.dumps(out, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    EW.refuse_store(text, json.loads(a.inputs.read_text())["bucket"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
