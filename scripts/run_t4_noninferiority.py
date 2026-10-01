#!/usr/bin/env python3
"""T4 sparse pilot: exact paired non-inferiority, as pre-registered.

CPU-only. Reads the pilot's accuracy parquets (the oracle run, which also
holds the dense reference, and the inline run), and writes one row per
(task, band, arm, sparsity) with the exact bound, the tier, and whether the
fixed-sequence rule reached that sparsity. Everything it decides with comes
from attnbench/accuracy/t4_pilot.py; see docs/t4_sparse_pilot.md.

Refuses rather than repairs: a dirty row, rows from more than one commit,
a task or band outside the plan, or an arm the plan does not name.

    python scripts/run_t4_noninferiority.py \\
        results/t4_sparse_pilot_oracle/accuracy.parquet \\
        results/t4_sparse_pilot_inline/accuracy.parquet \\
        --out results/t4_sparse_pilot_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy import t4_pilot                                  # noqa: E402
from attnbench.analysis.exact_noninferiority import run_exact_ni         # noqa: E402


def load(paths: list[str], *, allow_mixed_commits: bool) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if df["git_dirty"].fillna(True).astype(bool).any():
        raise SystemExit("REFUSING: rows with git_dirty=True (or unknown); their "
                         "commit does not establish the code that produced them")
    commits = sorted(df["git_commit"].astype(str).unique())
    if len(commits) != 1 and not allow_mixed_commits:
        raise SystemExit(f"REFUSING: rows from {len(commits)} commits {commits}; the "
                         f"pilot runs every arm at one commit")
    off_plan = sorted(set(df.task) - set(t4_pilot.SPARSE_PILOT_TASKS))
    if off_plan:
        raise SystemExit(f"REFUSING: tasks outside the plan: {off_plan}")
    sources = set(df.score_source.dropna())
    if sources - set(t4_pilot.SPARSE_PILOT_SCORE_SOURCES):
        raise SystemExit(f"REFUSING: arms outside the plan: "
                         f"{sorted(sources - set(t4_pilot.SPARSE_PILOT_SCORE_SOURCES))}")
    return df


def fixed_sequence(results: pd.DataFrame) -> pd.DataFrame:
    """Mark which sparsities the pre-registered sequence reached.

    Within each (task, band, score_source), sparsities are tested in
    t4_pilot.SPARSITY_SEQUENCE order and testing stops at the first that is
    not non-inferior. A sparsity after the stop is reported with its bound
    but `tested=False`, and no claim may be made from it.
    """
    results = results.copy()
    results["score_source"] = results.arm.str.split("@").str[0]
    results["sparsity"] = results.arm.str.split("@").str[1].astype(float)
    order = {s: i for i, s in enumerate(t4_pilot.SPARSITY_SEQUENCE)}
    results["seq_pos"] = results.sparsity.map(order)
    if results.seq_pos.isna().any():
        raise SystemExit("REFUSING: a sparsity outside the pre-registered sequence")
    tested = []
    for _, grp in results.sort_values("seq_pos").groupby(
            ["task", "band", "score_source"], sort=False):
        still = True
        expected = 0
        for _, r in grp.iterrows():
            # A missing step breaks the sequence: nothing after it is tested.
            still = still and r.seq_pos == expected
            tested.append((r.name, still))
            still = still and bool(r.non_inferior)
            expected += 1
    results["tested"] = pd.Series(dict(tested))
    results["claim"] = results.tested & results.non_inferior
    return results.drop(columns="seq_pos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-mixed-commits", action="store_true")
    args = ap.parse_args()

    df = load(args.parquets, allow_mixed_commits=args.allow_mixed_commits)
    dense = df[df.backend_role == "dense_reference"]
    sparse = df[df.backend_role == "block_sparse"]
    res = run_exact_ni(dense, sparse, bands=t4_pilot.PILOT_BANDS,
                       margin_pts=t4_pilot.MARGIN_PTS, alpha=t4_pilot.ALPHA)
    res = fixed_sequence(res)
    res["tier"] = [t4_pilot.TIERS[(t, int(b))] for t, b in zip(res.task, res.band)]
    res["git_commit"] = df.git_commit.iloc[0]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / "t4_noninferiority.parquet")
    res.to_csv(out / "t4_noninferiority.csv", index=False)
    cols = ["tier", "task", "band", "score_source", "sparsity", "n", "dense_correct",
            "sparse_correct", "b_sparse_only", "c_dense_only", "diff_pts",
            "lower_pts", "tested", "claim"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(res.sort_values(["tier", "task", "band", "score_source", "sparsity"])[cols]
              .round(1).to_string(index=False))
    print(f"\nmargin {t4_pilot.MARGIN_PTS} pts, one-sided alpha {t4_pilot.ALPHA}; "
          f"written to {out}/")


if __name__ == "__main__":
    main()
