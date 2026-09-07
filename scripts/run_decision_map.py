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
from run_pareto import EXCLUDE_BACKENDS, latency_table                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/stage7")
    ap.add_argument("--epsilons", default="1,2,5")
    ap.add_argument("--max-depth", type=int, default=3)
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

    lat = latency_table(df, grid.seq_lens)
    pareto = compute_pareto_frontiers(matched, lat, dense_backend=grid.dense_backend)
    recs = build_decision_map(pareto, oracle_sensitive_tasks=oracle_tasks)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tab = to_dataframe(recs)
    tab.to_parquet(out / "decision_map.parquet", index=False)

    print(f"oracle-sensitive tasks: {sorted(oracle_tasks) or 'none'}")
    print(f"\n=== DECISION MAP ({len(recs)} cells) ===")
    tab["choice"] = tab.apply(
        lambda r: r.backend if pd.isna(r.sparsity) else f"{r.backend}@{r.sparsity:g}",
        axis=1)
    for eps in epsilons:
        e = tab[tab.epsilon == eps]
        print(f"\nepsilon = {eps:g}")
        print(e.pivot_table(index="task", columns="context_length",
                            values="choice", aggfunc="first").to_string())

    print(f"\ndense recommended in {int(tab.is_dense.sum())} / {len(tab)} cells")
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
