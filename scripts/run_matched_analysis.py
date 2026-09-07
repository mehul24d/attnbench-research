#!/usr/bin/env python3
"""Stage 4: matched-accuracy operating points, from banked Stage 3 rows.

CPU-only. Reads one or more Stage 3 results files, refuses to join them
blindly, and writes the per-comparison rows and the per-(task, length,
epsilon) budgets.

WHY IT TAKES SEVERAL INPUTS RATHER THAN ONE

Stage 3 ran in band-aligned segments across two sessions at two commits
(d27c650 for band 2048, 1126bd2 for 4096/8192), deliberately written to
separate directories so each file stays single-commit -- see
docs/stage3_segmentation.md. Joining them is therefore an explicit, dated
step rather than an assumption, and this script is where it happens. The
commits are reported, not hidden, and any row whose backend set differs from
the others is refused rather than silently averaged in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.config import load_grid                # noqa: E402
from attnbench.analysis.matched import (                       # noqa: E402
    best_matched_sparsity_budget, run_matched_analysis, to_dataframe)


EXCLUDE_BACKENDS = ("gla",)   # see results/stage3_s1/INVALID_ROWS.md


def load_rows(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        before = len(d)
        d = d[~d["backend"].isin(EXCLUDE_BACKENDS)]
        commits = sorted({str(c)[:12] for c in d["git_commit"].dropna().unique()})
        print(f"  {p}: {len(d)} rows"
              f"{f' ({before - len(d)} excluded: {EXCLUDE_BACKENDS})' if before != len(d) else ''}"
              f"  commit(s) {', '.join(commits)}"
              f"  bands {sorted(set((d.context_length // 1000 * 1000).unique()))}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # A segment that measured a different set of arms cannot be compared
    # against one that did not -- the composition difference would show up as
    # an accuracy difference. Refuse rather than average.
    per_band = df.groupby(df.context_length // 1000)["backend"].apply(
        lambda s: tuple(sorted(s.unique())))
    if per_band.nunique() != 1:
        raise SystemExit(
            f"REFUSING: bands do not share a backend set, so a joined "
            f"analysis would compare differently-composed halves:\n{per_band}")
    print(f"\njoined: {len(df)} rows, arms {per_band.iloc[0]}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="Stage 3 accuracy.parquet files")
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/stage4")
    ap.add_argument("--epsilons", default="1,2,5")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    grid = load_grid(args.grid)
    print("loading:")
    df = load_rows(args.results)
    epsilons = tuple(float(e) for e in args.epsilons.split(","))

    results = run_matched_analysis(
        df, grid, dense_backend=grid.dense_backend,
        sparse_backends=("block_sparse",), epsilons=epsilons,
        n_resamples=args.n_resamples, seed=args.seed)
    budgets = best_matched_sparsity_budget(results)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    to_dataframe(results).to_parquet(out / "matched_comparisons.parquet", index=False)
    to_dataframe(budgets).to_parquet(out / "matched_budgets.parquet", index=False)

    b = to_dataframe(budgets)
    print(f"\n=== matched sparsity budget (max level non-inferior to dense) ===")
    for eps in epsilons:
        e = b[b.epsilon == eps]
        print(f"\nepsilon = {eps:g} points")
        print(e.pivot_table(index="task", columns="context_length",
                            values="matched_sparsity", dropna=False).to_string())
    flagged = sorted(b[b.oracle_sensitive]["task"].unique())
    print(f"\noracle_sensitive tasks: {flagged or 'none'}")
    if flagged:
        print("  At least one sparsity level EXCEEDED dense beyond noise on these.")
        print("  Their budgets certify non-inferiority WHEN THE MASK IS CHOSEN")
        print("  WITH FULL KNOWLEDGE OF THE ATTENTION SCORES -- not that the")
        print("  budget is free. See docs/claims.md.")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
