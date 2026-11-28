"""EXPLORATORY / POST HOC: how far the host prices overstate host time, per
pairing and per operation class (the researcher's paper-quality wave of
28 Nov 2026, item 10). Nothing is scored or rescored, no price is changed,
BUNGA is not refitted, and nothing is measured: banked files only.

    python3 scripts/null_kernel_bias.py --inputs <inputs.json> --mirror <dir> \\
        --banked <attnbench_backups dir> --s4 <Shakedown 4 results dir> \\
        --out-dir docs/paper/exploratory

Per pairing (P2, P3, P4), from each cell's host-components step:

  A. the null pass against the committed price at the same positions, per
     operation class (`host_bias.class_sums`), as a ratio of sums with a
     bootstrap interval over cells (oracle cells apart);
  B. the committed host price (bare + per-layer residual) against the
     measured prefill, per cell (`host_bias.lower_bound`), split by route.
     P2 and P4 are measured as scored (the mirror, through
     `exploratory_wave.load`); P3 through the same `measure` on its own
     session, against P1's committed cell keys (as the P3 exploratory score
     does). P3's host prices are its own session's, not the P1 transfer's.

S4 (L4, 1 Nov 2026; no bare timer then): the null pass per layer against the
recording run's layer spans (a profiled run, so this is a lower bound too),
for the one cell whose recording trace is banked.

P4 against an nsys host timeline: read from the H100 diagnostic's banked
verdict (Q4), not recomputed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exploratory_wave as EW  # noqa: E402
import gate_a as GA  # noqa: E402
import score_hypotheses as SH  # noqa: E402
from attnbench.exploratory import h100_diag as HD  # noqa: E402
from attnbench.exploratory import host_bias as HB  # noqa: E402
from attnbench.exploratory import paper as PP  # noqa: E402
from attnbench.exploratory import traces as TR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOST = {
    "P2": "bucket_20261117/attnbench-a100-p2e-20261114-0546/results/t61_p2_20261114T0022Z/"
          "07_host_components",
    "P3": "aws_p3_g6_20261126T0906Z/session/05_host_components",
    "P4": "bucket_20261117/attnbench-h100-p4s-20261116-170657-b3-a9/results/"
          "t64_p4_20261116T113724Z_b3/08_host_components",
}
P3_SESSION = "aws_p3_g6_20261126T0906Z/session"
S4_CELL = "t31_rehearsal_20261101T0723Z/05_host_components/dense_d1_L2048_b1"
DIAG = ROOT / "docs/exploratory/h100_diag_verdict_20261117.json"
BUDGET_OF = {v: k for k, v in TR.BUDGET.items()}


def cell_dir(root: Path, c) -> Path:
    d = "1" if c.arm == "dense" else ("cal" if c.arm.startswith("xattention") else f"{c.density:g}")
    return root / f"{c.arm}_d{d}_L{BUDGET_OF[c.context_length]}_b{c.batch}"


def cell_name(c) -> str:
    return f"{c.arm} {c.density:g} L{c.context_length} b{c.batch}"


def pairing(p: str, cells, ms, root: Path, bs, committed: dict | None) -> dict:
    rows, by_class, oracle = [], {}, {}
    for c in sorted(cells, key=lambda c: (c.context_length, c.batch, c.arm, c.density)):
        if c.key not in ms["seconds"]:
            continue
        d = cell_dir(root, c)
        hp = json.loads((d / "host_profile.json").read_text())
        if hp.get("pricing") != "bare_timer":
            raise SystemExit(f"{p} {d.name}: the committed profile is not the bare timer's")
        parts = HB.host_parts(hp)
        if committed is not None:
            gap = abs(parts["total"] - committed[c.key])
            if gap > 1e-9:
                raise SystemExit(f"{p} {d.name}: the profile is not the committed price ({gap:.2e} s)")
        npp = json.loads((d / "host_profile_null_pass.json").read_text())
        null = HB.null_total(npp)
        bp = json.loads((d / "bare_prices.json").read_text())
        group = by_class if c.arm != "oracle" else oracle
        for k, s in HB.class_sums(bp, npp, hp).items():
            group.setdefault(k, []).append(s)
        m = ms["seconds"][c.key]
        med, lo, hi = PP.median_interval(ms["samples"][c.key], n_boot=bs.n_boot, seed=bs.seed,
                                         level=bs.level)
        lb = HB.lower_bound(parts["total"], m, lo, hi)
        rows.append({"cell": cell_name(c), "arm": c.arm, "measured_s": m, "lo_s": lo, "hi_s": hi,
                     **{f"{k}_s": v for k, v in parts.items()}, "null_pass_s": null,
                     "committed_over_measured": lb,
                     "bare_over_measured": parts["bare"] / m,
                     "residual_over_measured": parts["residual"] / m,
                     "null_over_measured": null / m})
    kw = dict(n_boot=bs.n_boot, seed=bs.seed, level=bs.level)
    def table(groups):
        t = {k: {**HB.ratio_of_sums([(s["null_s"], s["committed_s"]) for s in v], **kw),
                 "positions": sum(s["positions"] for s in v),
                 "share": sum(s["committed_s"] for s in v)} for k, v in groups.items()}
        tot = sum(x["share"] for x in t.values())
        for x in t.values():
            x["share"] /= tot
        return dict(sorted(t.items(), key=lambda x: -x[1]["share"]))
    classes, oracle_classes = table(by_class), table(oracle)
    oc = [r for r in rows if r["arm"] == "oracle"]
    lbs = [r["committed_over_measured"]["ratio"] for r in rows]
    med, lo, hi = PP.median_interval(lbs, **kw)
    return {"cells": rows, "by_class": classes, "by_class_oracle": oracle_classes,
            "oracle_null_over_committed": HB.ratio_of_sums(
                [(r["null_pass_s"], r["total_s"]) for r in oc], **kw),
            "summary": {
                "n": len(rows),
                "committed_over_measured_median": {"ratio": med, "lo": lo, "hi": hi},
                "cells_committed_above_measured": sum(r > 1 for r in lbs),
                "above_one_median": statistics.median([r for r in lbs if r > 1]) if any(r > 1 for r in lbs) else None,
                "max": max(lbs),
                "cells_above_beyond_interval": sum(r["committed_over_measured"]["lo"] > 1 for r in rows),
                "cells_bare_alone_above": sum(r["bare_over_measured"] > 1 for r in rows),
                "cells_residual_alone_above": sum(r["residual_over_measured"] > 1 for r in rows),
                "residual_share_median": statistics.median(r["residual_s"] / r["total_s"] for r in rows),
                "null_over_committed_non_oracle": HB.ratio_of_sums(
                    [(r["null_pass_s"], r["total_s"]) for r in rows if r["arm"] != "oracle"], **kw),
                "null_over_measured_median": statistics.median(r["null_over_measured"] for r in rows)}}


def s4(cell: Path, bs) -> dict:
    hp = json.loads((cell / "host_profile.json").read_text())
    seq = json.loads((cell / "recording" / "sequence.json").read_text())
    tr = json.loads((cell / "recording" / "trace.json").read_text())
    nl, sp = HB.null_by_layer(hp, seq), HB.layer_spans(tr)
    if set(nl) != set(sp):
        raise SystemExit("S4: the null pass and the trace annotate different layers")
    rec = HD.kineto_host(tr, "attnbench::prefill")
    return {"cell": "dense 1 L1951 b1 (L4)", "layers": len(sp),
            "null_over_recorded_layer_span": HB.ratio_of_sums(
                [(nl[k], sp[k]) for k in sorted(sp)], n_boot=bs.n_boot, seed=bs.seed, level=bs.level),
            "null_total_s": HB.null_total(hp), "recorded_host_s": rec["host_s"],
            "recorded_wall_s": rec["wall_s"], "null_over_recorded_host": HB.null_total(hp) / rec["host_s"]}


def nsys_q4() -> dict:
    q4 = json.loads(DIAG.read_text())["Q4"]
    return {L: {"host_priced_s": v["host_priced_s"], "host_nsys_s": v["host_measured_s"],
                "priced_over_nsys": 1 / v["host_ratio"], "comparable": v["comparable"]}
            for L, v in q4.items() if isinstance(v, dict)}


def _r(x, d=2):
    return f"{x['ratio']:.{d}f} [{x['lo']:.{d}f}, {x['hi']:.{d}f}]"


def to_markdown(o: dict) -> str:
    L = ["# EXPLORATORY / POST HOC: how far the host prices overstate host time (28 Nov 2026)", "",
         "**EXPLORATORY / POST HOC. Nothing is scored or rescored; no host price is changed; "
         "BUNGA is not refitted; nothing new is measured.** H1 stays P2 0.1183, P4 0.2495. "
         "Generated by `scripts/null_kernel_bias.py`; numbers in `null_kernel_bias.json`. "
         "Intervals: percentile bootstrap at the lock's settings, resampling whole cells "
         "(or layers for S4).", "",
         "**The mechanism is the per-layer idle-gap residual, not null-kernel bias.** The "
         "committed prices do not read the null pass, and nothing here is a finding about "
         "TaxBreak's method, which uses its null kernel only for a constant launch floor.", "",
         "Two instruments, kept apart. The **null pass** is this study's own extension of a "
         "null kernel to every launching call, under the tape's and null device's Python "
         "modes; it is banked as a comparison. The **committed price**, which P2's and P4's "
         "predictions read, is the bare timer's per-call price plus the per-layer idle-gap "
         "residual (ruling of 1 Nov 2026). Priced / measured is a **lower bound** on priced / "
         "true host time: the host issues every operation before the timer stops.", ""]
    L += ["## Per pairing: committed host price over the measured prefill", "",
          "| pairing | cells | median committed / measured | cells > 1 | median of those (max) | > 1 beyond interval | "
          "bare alone > 1 | residual alone > 1 | residual share of host (median) | "
          "null pass / committed (non-oracle) | null pass / committed (oracle) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in ("P2", "P3", "P4"):
        s = o[p]["summary"]
        L.append(f"| {p} | {s['n']} | {_r(s['committed_over_measured_median'])} | "
                 f"{s['cells_committed_above_measured']} | "
                 f"{'—' if s['above_one_median'] is None else format(s['above_one_median'], '.2f')} "
                 f"({s['max']:.2f}) | {s['cells_above_beyond_interval']} | "
                 f"{s['cells_bare_alone_above']} | {s['cells_residual_alone_above']} | "
                 f"{s['residual_share_median']:.2f} | {_r(s['null_over_committed_non_oracle'])} | "
                 f"{_r(o[p]['oracle_null_over_committed'])} |")
    L += ["", "## Per operation class: null pass over committed price, same positions", "",
          "Ratio of summed seconds over cells, oracle cells left out (their host-side mask "
          "builder runs inside the null pass's gaps under both Python modes, and lands on "
          "the next launching call). Share is the class's share of the committed host "
          "seconds. The residual is shared out in proportion to the null pass's gaps, so "
          "class differences come from the bare part.", "",
          "| class | " + " | ".join(f"{p} ratio | {p} share" for p in ("P2", "P3", "P4")) + " |",
          "|---|" + "---|---|" * 3]
    ks = sorted({k for p in ("P2", "P3", "P4") for k in o[p]["by_class"]},
                key=lambda k: -o["P4"]["by_class"].get(k, {"share": 0})["share"])
    for k in ks:
        cells = []
        for p in ("P2", "P3", "P4"):
            x = o[p]["by_class"].get(k)
            cells += [_r(x), f"{x['share']:.2f}"] if x else ["—", "—"]
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    s = o["S4"]
    L += ["## S4 (L4, 1 Nov 2026): null pass against the recording run", "",
          f"One cell ({s['cell']}; the only S4 cell whose recording trace is banked). Null pass "
          f"over the recording's layer span, over {s['layers']} layers: "
          f"{_r(s['null_over_recorded_layer_span'])}. Whole prefill: null pass "
          f"{s['null_total_s']*1e3:.1f} ms against the recording's host time "
          f"{s['recorded_host_s']*1e3:.1f} ms ({s['null_over_recorded_host']:.2f}x). The "
          "recording is profiled, so this too is a lower bound.", "",
          "## P4 against an nsys host timeline (banked H100 diagnostic, Q4)", "",
          "| length | committed host ms | nsys host ms | committed / nsys | comparable |",
          "|---|---|---|---|---|"]
    for k, v in o["nsys_q4"].items():
        L.append(f"| {k} | {v['host_priced_s']*1e3:.1f} | {v['host_nsys_s']*1e3:.1f} | "
                 f"{v['priced_over_nsys']:.2f} | {v['comparable']} |")
    L += ["", "XAttention cells; nsys adds little host overhead but the run was a separate "
          "H100 session (Q3: walls within 1.02 to 1.05 of P4's).", ""]
    for p in ("P2", "P3", "P4"):
        L += [f"## {p} per cell", "", "| cell | measured ms | committed host ms | bare | residual | "
              "committed / measured | null pass / measured |", "|---|---|---|---|---|---|---|"]
        for r in o[p]["cells"]:
            L.append(f"| {r['cell']} | {r['measured_s']*1e3:.1f} | {r['total_s']*1e3:.1f} | "
                     f"{r['bare_s']*1e3:.1f} | {r['residual_s']*1e3:.1f} | "
                     f"{_r(r['committed_over_measured'])} | {r['null_over_measured']:.2f} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--mirror", required=True, type=Path)
    ap.add_argument("--banked", required=True, type=Path)
    ap.add_argument("--s4", required=True, type=Path, help="Shakedown 4's results/ directory")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    D = EW.load(a.inputs, a.mirror)
    bs = D["locked"].bootstrap
    out = {"label": HB.LABEL, "task": "paper-quality wave, 28 Nov 2026, item 10",
           "bootstrap": vars(bs), "classes": {k: list(v) for k, v in HB.CLASSES.items()}}
    for p in ("P2", "P4"):
        cp = D["cps"][p]
        committed = {c.key: float(sum(c.priced.host)) for c in cp.cells}
        out[p] = pairing(p, cp.cells, D["ms"][p], a.banked / HOST[p], bs, committed)
    tagged = SH.lock_at(SH.TAG)
    mcr = int(tagged["values"]["rerun_budget"]["value"]["min_clean_rounds"])
    session = a.banked / P3_SESSION
    results, _ = GA.ground_truth(session / "09_ground_truth")
    ms3 = SH.measure(D["cps"]["P1"], [(session.name, results)], min_clean_rounds=mcr)
    out["P3"] = pairing("P3", D["cps"]["P1"].cells, ms3, a.banked / HOST["P3"], bs, None)
    out["S4"] = s4(a.s4 / S4_CELL, bs)
    out["nsys_q4"] = nsys_q4()
    text = json.dumps(out, indent=1, default=str)
    md = to_markdown(out)
    bucket = json.loads(a.inputs.read_text())["bucket"]
    EW.refuse_store(text, bucket)
    EW.refuse_store(md, bucket)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "null_kernel_bias.json").write_text(text + "\n")
    (a.out_dir / "null_kernel_bias.md").write_text(md)
    print(f"wrote {a.out_dir}/null_kernel_bias.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
