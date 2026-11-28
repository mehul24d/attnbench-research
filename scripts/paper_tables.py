"""EXPLORATORY / POST HOC: the paper-quality tables (the researcher's wave of
28 Nov 2026, items 2 to 5). Nothing here scores or rescores a hypothesis:
H0-H3 stand as scored at `83b2edf` (H1: P2 0.1183, P4 0.2495), the timed-path
pin `0e0fb012` is not read, and no parameter is fitted on P2 or P4.

CPU only and offline, from the scoring wave's mirror (through
`exploratory_wave.load`, i.e. `score_hypotheses`' own loaders), the local
copies of P2's and P4's ground-truth traces, and P4's dense-selection step:

  2. the committed error per cell, split exactly into the priced device
     chain minus the traced device-busy time and the predicted minus the
     measured idle time, with the four predicted terms;
  3. the committed recursion with one term removed at a time, beside the
     committed baselines (fitted sum, sum, step form, learned);
  4. a bootstrap interval on each cell's measured median over its rounds, at
     the lock's bootstrap settings, and the recursion's error interval
     against H1's tolerance (described, not a test);
  5. post hoc: P4's dense attention repriced at FA3 / sdpa_flash from P4's own
     dense-selection microbenchmark, and every host cost scaled by k.

    python3 scripts/paper_tables.py --inputs <inputs.json> --mirror <dir> \\
        --traces P2=<glob> --traces P4=<glob> --dense-select P4=<dir> \\
        --out-dir docs/paper/exploratory

The output names no store and no bucket.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exploratory_residual as ER  # noqa: E402
import exploratory_wave as EW  # noqa: E402
from attnbench.exploratory import paper as PP  # noqa: E402
from attnbench.exploratory import traces as TR  # noqa: E402

PAIRINGS = ("P2", "P4")
BASELINES = ("recursion", "fitted_sum", "fitted_sum_plus_s0", "sum", "sum_plus_s0",
             "step_form", "learned")
HOST_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# H1's tolerance at the lock (docs/lock_values.json, h0_to_h3_tolerances H1).
H1_SCORED = {"P2": 0.1183, "P4": 0.2495}
BUDGETS = (2048, 4096, 8192, 16384, 32768)


def budget_of(context_length: int) -> int:
    for b in BUDGETS:
        if context_length <= b:
            return b
    raise ValueError(f"{context_length} is above every grid budget")


def fa3_ratios(select_dir: Path) -> dict:
    """{(budget, batch): {"fa3_ms", "sdpa_flash_ms", "ratio", "chosen"}} from
    the dense-selection files."""
    out = {}
    for f in sorted(Path(select_dir).glob("dense_choice_*_*_*.json")):
        _, _, gpu, b, n = f.stem.split("_", 4)
        d = json.loads(f.read_text())
        m = {c["name"]: c["median_ms"] for c in d["candidates"]}
        if not m.get("fa3") or not m.get("sdpa_flash"):
            raise SystemExit(f"{f.name}: no FA3 or no sdpa_flash timing")
        out[(int(b), int(n))] = {"fa3_ms": m["fa3"], "sdpa_flash_ms": m["sdpa_flash"],
                                 "ratio": m["fa3"] / m["sdpa_flash"], "chosen": d.get("chosen")}
    if not out:
        raise SystemExit(f"{select_dir}: no dense_choice files")
    return out


def families_of(cp, recs) -> dict:
    """{cell key: per-position operation families} from the traces (names only)."""
    out = {}
    by = {c.key: c for c in cp.cells}
    for r in recs:
        if r["key"] in out:
            continue
        kinds = [x.value for x in by[r["key"]].priced.kinds]
        out[r["key"]] = TR.align(r["family"], r["device"], kinds)[0]
    return out


def name(c) -> str:
    return f"{c.arm}|{c.density:g}|{c.context_length}|b{c.batch}"


def pairing_tables(D, p: str, files: list, ratios: dict | None) -> dict:
    cp, ms = D["cps"][p], D["ms"][p]
    bs, tol = D["locked"].bootstrap, D["locked"].tolerances["H1"]
    recs = EW.extract(cp, files)
    busy_of = {Path(f).name: ER.device_busy(f) for f in files}
    busy = {}
    for r in recs:
        busy.setdefault(r["key"], []).append(busy_of[r["trace"]])
    fams = families_of(cp, recs)
    Q = cp.queue_depth
    miss, knock, unc, fa3 = [], [], [], []
    scales = {k: [] for k in HOST_SCALES}
    for c in sorted(cp.cells, key=lambda c: (c.context_length, c.batch, c.arm, c.density)):
        k, m = c.key, ms["seconds"][c.key]
        cls = PP.arm_class(c.priced, c.arm)
        row = {"cell": name(c), "class": cls, "arm": c.arm}
        # 2. the miss split
        if k in busy:
            miss.append({**row, "traces": len(busy[k]),
                         **PP.miss_split(c.priced, Q, m, statistics.fmean(busy[k]))})
        # 3. knockouts and the committed baselines
        ko = PP.knockouts(c.priced, Q)
        if abs(ko["recursion"] - c.predictions["recursion"]) > 1e-12:
            raise SystemExit(f"{name(c)}: the recursion does not reproduce the committed value")
        knock.append({**row, "measured": m,
                      "rel": {**{b: PP.rel(c.predictions[b], m) for b in BASELINES},
                              **{v: PP.rel(x, m) for v, x in ko.items() if v != "recursion"}}})
        # 4. the measured median's interval
        med, lo, hi = PP.median_interval(ms["samples"][k], n_boot=bs.n_boot, seed=bs.seed,
                                         level=bs.level)
        rlo, rhi = PP.rel_interval(c.predictions["recursion"], lo, hi)
        unc.append({**row, "rounds": len(ms["samples"][k]), "measured": m,
                    "median_of_rounds": med, "lo": lo, "hi": hi,
                    "width_over_median": (hi - lo) / med,
                    "rel": PP.rel(c.predictions["recursion"], m), "rel_lo": rlo, "rel_hi": rhi,
                    "against_tolerance": PP.against_tolerance(rlo, rhi, tol)})
        # 5. post hoc sensitivities
        for s in HOST_SCALES:
            scales[s].append(PP.rel(PP.host_scaled(c.priced, Q, s), m))
        if ratios is not None:
            if c.arm == "dense":
                r = ratios[(budget_of(c.context_length), c.batch)]
                rp, n = PP.reprice_attention(c.priced, fams[k], r["ratio"])
                att = sum(c.priced.device[i] for i, f in enumerate(fams[k])
                          if TR_component(f) == "attention")
                pred = PP.knockouts(rp, Q)["recursion"]
                fa3.append({**row, "ratio": r["ratio"], "positions": n,
                            "priced_attention_s": att,
                            "priced_over_select_flash": att / (n * r["sdpa_flash_ms"] / 1000),
                            "committed_rel": PP.rel(c.predictions["recursion"], m),
                            "fa3_rel": PP.rel(pred, m), "rp": rp})
            else:
                fa3.append({**row, "committed_rel": PP.rel(c.predictions["recursion"], m),
                            "fa3_rel": PP.rel(c.predictions["recursion"], m), "rp": c.priced})
    out = {"queue_depth": Q, "h1_tolerance": tol, "h1_scored_median_abs": H1_SCORED[p],
           "miss": miss, "knockout": knock, "uncertainty": unc}
    out["miss_summary"] = {
        "n": len(miss),
        "median_rel": statistics.median(r["rel"] for r in miss),
        "sum_error_s": sum(r["error_s"] for r in miss),
        "sum_device_part_s": sum(r["device_part_s"] for r in miss),
        "sum_idle_part_s": sum(r["idle_part_s"] for r in miss),
        "max_identity_gap_s": max(abs(r["error_s"] - r["device_part_s"] - r["idle_part_s"])
                                  for r in miss),
        "cells_host_above_measured": sum(r["host"] > r["measured"] for r in miss)}
    variants = BASELINES + tuple(v for v in PP.VARIANTS if v != "recursion")
    out["knockout_summary"] = {
        v: {"all": PP.summarise([r["rel"][v] for r in knock]),
            **{cl: PP.summarise([r["rel"][v] for r in knock if r["class"] == cl])
               for cl in ("dense", "sync_free", "synchronising")},
            "by_arm": {arm: PP.summarise([r["rel"][v] for r in knock if r["arm"] == arm])
                       for arm in sorted({r["arm"] for r in knock})}}
        for v in variants}
    if abs(out["knockout_summary"]["recursion"]["all"]["median_abs"] - H1_SCORED[p]) > 5e-5:
        raise SystemExit(f"{p}: the recursion's median does not reproduce H1's scored value")
    out["uncertainty_summary"] = {
        "n": len(unc), "rounds": PP.count_by(unc, "rounds"),
        "median_width_over_median": statistics.median(r["width_over_median"] for r in unc),
        "max_width_over_median": max(r["width_over_median"] for r in unc),
        "against_tolerance": PP.count_by(unc, "against_tolerance"),
        "max_abs_median_vs_measured": max(abs(r["median_of_rounds"] - r["measured"])
                                          for r in unc)}
    out["host_scale"] = {str(s): PP.summarise(v) for s, v in scales.items()}
    if ratios is not None:
        out["fa3_reprice"] = [{k2: v for k2, v in r.items() if k2 != "rp"} for r in fa3
                              if r["class"] == "dense"]
        out["fa3_summary"] = {
            "committed": PP.summarise([r["committed_rel"] for r in fa3]),
            "fa3_repriced": PP.summarise([r["fa3_rel"] for r in fa3]),
            "fa3_repriced_dense": PP.summarise([r["fa3_rel"] for r in fa3
                                                if r["class"] == "dense"]),
            "fa3_and_host_scale": {str(s): PP.summarise(
                [PP.rel(PP.host_scaled(r["rp"], Q, s), knock[i]["measured"])
                 for i, r in enumerate(fa3)]) for s in HOST_SCALES}}
    return out


def TR_component(f: str):
    from attnbench.exploratory.breakdown import component_of
    return component_of(f)


def _p(x, d=3):
    return "—" if x is None else f"{x:+.{d}f}"


def _a(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def to_markdown(out: dict) -> str:
    L = ["# EXPLORATORY / POST HOC: paper-quality tables (28 Nov 2026)", "",
         "**EXPLORATORY / POST HOC. Nothing here scores or rescores a hypothesis.** "
         "H1 stays as scored at `83b2edf`: P2 0.1183, P4 0.2495 (both fail). "
         "The timed-path pin `0e0fb012` is untouched, and no parameter is fitted on P2 or P4; "
         "the knockout and host-scale tables are read on P2's and P4's own ground truth and are "
         "labelled IN-SAMPLE. "
         "Generated by `scripts/paper_tables.py`; numbers in `paper_tables.json`.", ""]
    for p in PAIRINGS:
        o = out[p]
        ms = o["miss_summary"]
        L += [f"## {p}", "",
              f"### Miss decomposition ({p}, committed model, {ms['n']} cells)", "",
              "Error = (priced device chain − traced device-busy) + (predicted idle − measured idle), "
              "exact by construction (largest gap "
              f"{ms['max_identity_gap_s']:.1e} s). The traced busy time is from a profiled run; "
              "**no session records an untraced host timeline**, so measured idle time is not "
              "split into host, queue and synchronisation.", "",
              f"Summed over cells: error {ms['sum_error_s']:+.4f} s = device part "
              f"{ms['sum_device_part_s']:+.4f} s + idle part {ms['sum_idle_part_s']:+.4f} s. "
              f"Priced host alone exceeds the whole measured time in "
              f"{ms['cells_host_above_measured']} of {ms['n']} cells.", "",
              "| cell | class | host | s_0 | queue | wait | predicted | measured | busy | "
              "meas. idle | device part | idle part | rel |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in o["miss"]:
            L.append(f"| {r['cell']} | {r['class']} | {r['host']*1e3:.1f} | "
                     f"{r['fixed_sync']*1e3:.2f} | {r['queue']*1e3:.1f} | {r['sync_wait']*1e3:.1f} | "
                     f"{r['predicted']*1e3:.1f} | {r['measured']*1e3:.1f} | "
                     f"{r['device_busy']*1e3:.1f} | {r['measured_idle']*1e3:.1f} | "
                     f"{r['device_part_s']*1e3:+.1f} | {r['idle_part_s']*1e3:+.1f} | "
                     f"{r['rel']:+.3f} |")
        L += ["", "Times in ms.", "",
              f"### Knockout and baselines ({p})", "",
              "Committed model with one term removed (no parameter fitted). Median signed / "
              "median absolute relative error; n per class in brackets.", "",
              f"**IN-SAMPLE:** every row is read against {p}'s own ground truth, so a knockout "
              "that lowers the error here (for example `no_wait`) was found on the data it is "
              "scored on. It is not a held-out result and belongs in no summary line.", "",
              "| predictor | all | dense | sync-free | synchronising |", "|---|---|---|---|---|"]
        for v, s in o["knockout_summary"].items():
            cells = [f"{_p(s[c]['median_signed'])} / {_a(s[c]['median_abs'])} ({s[c]['n']})"
                     for c in ("all", "dense", "sync_free", "synchronising")]
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        arms = sorted(next(iter(o["knockout_summary"].values()))["by_arm"])
        L += ["", "By arm (median signed / median absolute):", "",
              "| predictor | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for v, s in o["knockout_summary"].items():
            L.append(f"| {v} | " + " | ".join(
                f"{_p(s['by_arm'][a]['median_signed'])} / {_a(s['by_arm'][a]['median_abs'])} "
                f"({s['by_arm'][a]['n']})" for a in arms) + " |")
        L += ["", "Class is structural: a cell is synchronising if its priced sequence has any "
              "synchronisation besides the timer's closing one. Every sparse cell on P2 and P4 "
              "has one (a synchronising copy to the host), so the sync-free class is empty here, "
              "unlike the plan's list of synchronisation-free arms; the by-arm table gives that view.", ""]
        L += ["Variants: " + "; ".join(f"`{k}` removes {v}" for k, v in PP.VARIANTS.items()
                                          if k != "recursion") + ".", ""]
        u = o["uncertainty_summary"]
        L += [f"### Per-cell uncertainty ({p})", "",
              f"Percentile bootstrap (lock settings) on each cell's median over its rounds "
              f"(rounds per cell: {u['rounds']}). These are rounds within one session, "
              "not replications of the study. Median interval width "
              f"{u['median_width_over_median']:.4f} of the median (largest "
              f"{u['max_width_over_median']:.4f}). Recursion error interval against H1's "
              f"tolerance {o['h1_tolerance']}: {u['against_tolerance']}. Described, not a test; "
              "no significance test is pooled across pairings.", "",
              "| cell | rounds | measured ms | 95% interval ms | rel | rel interval | vs tolerance |",
              "|---|---|---|---|---|---|---|"]
        for r in o["uncertainty"]:
            L.append(f"| {r['cell']} | {r['rounds']} | {r['measured']*1e3:.2f} | "
                     f"{r['lo']*1e3:.2f}–{r['hi']*1e3:.2f} | {r['rel']:+.3f} | "
                     f"{r['rel_lo']:+.3f} to {r['rel_hi']:+.3f} | {r['against_tolerance']} |")
        L += ["", f"### POST HOC sensitivity ({p})", "",
              "Every host cost h scaled by k (s_0 unchanged). H1 stays "
              f"{o['h1_scored_median_abs']}.", "",
              "**IN-SAMPLE:** the grid of k (0.5 to 1.0) is fixed in this script, but any single "
              f"k read off it, such as the k = 0.6 that minimises the error here, is chosen on "
              f"{p}'s own ground truth. It is not a correction, not held out, and belongs in no "
              "summary line.", "",
              "| k | median signed | median abs | max abs |", "|---|---|---|---|"]
        for k, s in o["host_scale"].items():
            L.append(f"| {k} | {_p(s['median_signed'])} | {_a(s['median_abs'])} | {_a(s['max_abs'])} |")
        if "fa3_summary" in o:
            f = o["fa3_summary"]
            L += ["", "Dense attention repriced at FA3 / sdpa_flash from P4's own dense-selection "
                  "step (untimed workload; ground truth ran FA3, the committed prediction priced "
                  "sdpa_flash). Only dense cells change.", "",
                  "| dense cell | ratio | positions | priced / select flash | committed rel | "
                  "FA3-repriced rel |", "|---|---|---|---|---|---|"]
            for r in o["fa3_reprice"]:
                L.append(f"| {r['cell']} | {r['ratio']:.3f} | {r['positions']} | "
                         f"{r['priced_over_select_flash']:.3f} | {r['committed_rel']:+.3f} | "
                         f"{r['fa3_rel']:+.3f} |")
            L += ["", f"All 35 cells: committed {_a(f['committed']['median_abs'])}, FA3-repriced "
                  f"{_a(f['fa3_repriced']['median_abs'])} median abs (dense only: "
                  f"{_a(f['fa3_repriced_dense']['median_abs'])}).", "",
                  "| k (with FA3 repricing) | median signed | median abs |", "|---|---|---|"]
            for k, s in f["fa3_and_host_scale"].items():
                L.append(f"| {k} | {_p(s['median_signed'])} | {_a(s['median_abs'])} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--mirror", required=True, type=Path)
    ap.add_argument("--traces", action="append", default=[], help="PAIRING=<glob>")
    ap.add_argument("--dense-select", action="append", default=[], help="P4=<dir>")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    globs = dict(t.split("=", 1) for t in a.traces)
    if set(globs) != set(PAIRINGS):
        ap.error(f"--traces for exactly {PAIRINGS}")
    sel = dict(t.split("=", 1) for t in a.dense_select)
    if set(sel) != {"P4"}:
        ap.error("--dense-select P4=<dir> only")
    D = EW.load(a.inputs, a.mirror)
    out = {"label": PP.LABEL, "task": "paper-quality wave, 28 Nov 2026, items 2-5",
           "bootstrap": vars(D["locked"].bootstrap), "host_scales": list(HOST_SCALES)}
    for p in PAIRINGS:
        files = sorted(glob.glob(globs[p]))
        if not files:
            ap.error(f"{p}: no trace matches {globs[p]}")
        ratios = fa3_ratios(Path(sel[p])) if p in sel else None
        out[p] = pairing_tables(D, p, files, ratios)
    text = json.dumps(out, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    EW.refuse_store(text, json.loads(a.inputs.read_text())["bucket"])
    md = to_markdown(out)
    EW.refuse_store(md, json.loads(a.inputs.read_text())["bucket"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "paper_tables.json").write_text(text + "\n")
    (a.out_dir / "paper_tables.md").write_text(md)
    print(f"wrote {a.out_dir}/paper_tables.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
