#!/usr/bin/env python3
"""Stage 7: the decision map, from Stage 3 rows through Stages 4 and 6.

One recommendation per (task, context_length, epsilon), plus a shallow tree
scored on fidelity to that map rather than on held-out accuracy -- see
attnbench/analysis/decision.py for why a 27-cell grid does not support a
generalisation claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.config import load_grid                        # noqa: E402
from attnbench.analysis.decision import (                              # noqa: E402
    build_decision_map, fit_decision_tree, to_dataframe)
from attnbench.analysis.matched import (                               # noqa: E402
    best_matched_sparsity_budget, run_matched_analysis)
from attnbench.analysis.pareto import compute_pareto_frontiers          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attnbench import provenance                                       # noqa: E402
from run_pareto import (EXCLUDE_BACKENDS, cross_arm_cells,              # noqa: E402
                        latency_table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/stage7")
    ap.add_argument(
        "--allow-cross-arm-decode", action="store_true",
        help="With --corrected absent, permit a map built from arms that "
             "decoded through different kernels. Stamped onto the output.")
    ap.add_argument("--epsilons", default="1,2,5")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--corrected", default=None,
                    help="a decode_corrected.parquet. When given, the map is "
                         "built on `normalized_ms` -- matched decode kernel "
                         "and matched generation length -- instead of the "
                         "banked `latency_ms`, which for vt and "
                         "niah_multikey was measured with the sparse arms "
                         "decoding through sdpa_math. Rebuilding the map on "
                         "the banked column would reproduce the confound "
                         "(claims.md, 'two confounds'). DERIVED: see "
                         "analysis/decode_confound.py.")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    df = pd.concat([pd.read_parquet(p) for p in args.results], ignore_index=True)
    df = df[~df["backend"].isin(EXCLUDE_BACKENDS)]
    epsilons = tuple(float(e) for e in args.epsilons.split(","))

    matched = run_matched_analysis(
        df, grid, dense_backend=grid.dense_backend,
        sparse_backends=("block_sparse",), epsilons=epsilons)
    budgets = best_matched_sparsity_budget(matched)
    oracle_tasks = frozenset(b.task for b in budgets if b.oracle_sensitive)

    if args.corrected:
        c = pd.read_parquet(args.corrected)
        c = c.drop_duplicates(subset=["backend", "task", "context_length", "sparsity"])
        lat = {(r.backend, r.task, int(r.context_length),
                None if pd.isna(r.sparsity) else float(r.sparsity)):
               float(r.normalized_ms) for r in c.itertuples()}
        latency_source = f"normalized (DERIVED) from {args.corrected}"
    else:
        lat = latency_table(
            df, grid.seq_lens,
            allow_cross_arm_decode=args.allow_cross_arm_decode)
        latency_source = "measured latency_ms as banked"
    print(f"latency source: {latency_source}")
    pareto = compute_pareto_frontiers(matched, lat, dense_backend=grid.dense_backend)
    recs = build_decision_map(pareto, oracle_sensitive_tasks=oracle_tasks)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tab = to_dataframe(recs)
    tab["latency_source"] = latency_source
    tab["cross_arm_decode_confound"] = bool(
        cross_arm_cells(df, grid.seq_lens)) and not args.corrected
    provenance.stamp_analysis(tab, "scripts/run_decision_map.py")
    tab.to_parquet(out / "decision_map.parquet", index=False)

    print(f"oracle-sensitive tasks: {sorted(oracle_tasks) or 'none'}")
    print(f"\n=== DECISION MAP ({len(recs)} cells) ===")
    # An unresolvable cell must not render with the same face as a decided
    # one -- a "?" suffix, so the map cannot be read as 27 decisions when it
    # holds fewer. See decision.SPEEDUP_RESOLUTION_BANDS.
    # Marked on EITHER flag. They answer different questions and a cell can
    # fail one and pass the other: at niah_single/4096 the sparse point sits
    # at identical latency and is dominated off the frontier, so "which point
    # is fastest" is answered (dense, it cannot be worse) while "is dense
    # faster" is not. A reader wants to know about both.
    def _choice(r):
        c = r.backend if pd.isna(r.sparsity) else f"{r.backend}@{r.sparsity:g}"
        return c if (r.resolvable and r.dense_choice_resolvable) else c + " ?"
    tab["choice"] = tab.apply(_choice, axis=1)
    for eps in epsilons:
        e = tab[tab.epsilon == eps]
        print(f"\nepsilon = {eps:g}")
        print(e.pivot_table(index="task", columns="context_length",
                            values="choice", aggfunc="first").to_string())

    unres = tab[~(tab.resolvable & tab.dense_choice_resolvable)]
    print(f"\n  ? = the next-best point is inside this band's resolution "
          f"floor; which point is fastest there is decided by which Stage 5 "
          f"session supplied the phases, not by the arms.")

    print(f"\ndense recommended in {int(tab.is_dense.sum())} / {len(tab)} cells")
    fully = int((tab.resolvable & tab.dense_choice_resolvable).sum())
    print(f"RESOLVED by the measurement: {fully} / {len(tab)} cells")
    print(f"  of the {len(tab) - fully} that are not, "
          f"{int((~tab.dense_choice_resolvable).sum())} cannot say whether "
          f"sparse or dense is faster")
    if len(unres):
        print(f"\n=== UNRESOLVABLE ({len(unres)}) ===")
        u = unres.copy()
        u["sep_%"] = (u.separation * 100).round(2)
        u["floor_%"] = (u.resolution * 100).round(1)
        print(u[["task", "context_length", "epsilon", "choice", "sep_%",
                 "floor_%", "dense_choice_resolvable"]].to_string(index=False))
    tied = tab[tab.tie_broken_to_dense]
    if len(tied):
        print(f"\n{len(tied)} cell(s) had a sparse point ahead of dense by "
              f"less than the floor; the pre-registered tie rule gave them to "
              f"dense:")
        print(tied[["task", "context_length", "epsilon"]].to_string(index=False))
    ors = tab[tab.oracle_sensitive]
    print(f"oracle-sensitive recommendations: {len(ors)} / {len(tab)}")
    if len(ors):
        print(ors[["task", "context_length", "epsilon", "sparsity",
                   "speedup_vs_dense"]].round(3).to_string(index=False))
    sp = tab[~tab.is_dense]
    if len(sp):
        print(f"\nspeedup where a sparse point is recommended: "
              f"{sp.speedup_vs_dense.min():.3f}x - {sp.speedup_vs_dense.max():.3f}x")

    fid = fit_decision_tree(recs, max_depth=args.max_depth)
    print(f"\n=== TREE AS A SUMMARY OF THE MAP (depth {fid.max_depth}) ===")
    print(f"cells {fid.n_cells}, mismatches {fid.mismatches} -> "
          f"{'FAITHFUL' if fid.faithful else 'LOSSY: the map is authoritative'}")
    print(fid.rules)
    print("Scored on fidelity to the map, NOT on held-out accuracy: 27 cells")
    print("cannot support a generalisation claim, and calling this accuracy")
    print("would manufacture one. Do not extrapolate to unmeasured cells.")
    print(f"\nwritten to {out}/decision_map.parquet")


if __name__ == "__main__":
    main()
