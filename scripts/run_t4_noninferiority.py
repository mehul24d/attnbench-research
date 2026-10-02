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


# The sparse pilot's two arms and the XAttention phase's one
# (docs/t4_sparse_pilot.md, docs/t4_xattention_pilot.md).
PLANNED_SOURCES = (set(t4_pilot.SPARSE_PILOT_SCORE_SOURCES)
                   | {t4_pilot.XATTN_SCORE_SOURCE})


def _sequence(score_source: str) -> tuple:
    """The pre-registered order of an arm's settings: sparsities for the
    fixed-sparsity arms, thresholds (least aggressive first) for XAttention."""
    if score_source == t4_pilot.XATTN_SCORE_SOURCE:
        return t4_pilot.XATTN_THRESHOLD_SEQUENCE
    return t4_pilot.SPARSITY_SEQUENCE


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
    if sources - PLANNED_SOURCES:
        raise SystemExit(f"REFUSING: arms outside the plan: "
                         f"{sorted(sources - PLANNED_SOURCES)}")
    return df


def fixed_sequence(results: pd.DataFrame) -> pd.DataFrame:
    """Mark which sparsities the pre-registered sequence reached.

    Within each (task, band, score_source), settings are tested in their
    pre-registered order (`_sequence`: sparsities 0.5 -> 0.9 for the
    fixed-sparsity arms, thresholds 0.95 -> 0.8 for XAttention) and testing
    stops at the first that is not non-inferior. A sparsity after the stop is reported with its bound
    but `tested=False`, and no claim may be made from it.
    """
    results = results.copy()
    results["score_source"] = results.arm.str.split("@").str[0]
    raw = results.arm.str.split("@").str[1]
    is_cal = raw.str.startswith("cal=")
    # A calibrated table is one setting, so each is its own one-test family:
    # position 0, and the family key carries the calibration name so two
    # calibrations never share a sequence.
    results["xattn_calibration"] = raw.where(is_cal).str[len("cal="):]
    setting = pd.to_numeric(raw.where(~is_cal), errors="coerce")
    is_tau = results.score_source.eq(t4_pilot.XATTN_SCORE_SOURCE) & ~is_cal
    results["sparsity"] = setting.where(~is_tau & ~is_cal)
    results["xattn_threshold"] = setting.where(is_tau)
    results["seq_pos"] = [
        0 if c else {v: i for i, v in enumerate(_sequence(src))}.get(v)
        for src, v, c in zip(results.score_source, setting, is_cal)]
    results["seq_pos"] = results["seq_pos"].astype(float)
    unknown = sorted(set(results.xattn_calibration.dropna())
                     - set(t4_pilot.XATTN_CALIBRATIONS))
    if unknown:
        raise SystemExit(f"REFUSING: calibrations the plan does not register: {unknown}")
    if results.seq_pos.isna().any():
        raise SystemExit("REFUSING: a sparsity outside the pre-registered sequence")
    tested = []
    family = results.score_source + "|" + results.xattn_calibration.fillna("")
    for _, grp in results.assign(_family=family).sort_values("seq_pos").groupby(
            ["task", "band", "_family"], sort=False):
        still = True
        expected = 0
        for _, r in grp.iterrows():
            # A missing step breaks the sequence: nothing after it is tested.
            still = still and r.seq_pos == expected
            tested.append((r.name, still))
            still = still and bool(r.non_inferior)
            expected += 1
    results["tested"] = pd.Series(dict(tested))
    # A descriptive calibration is tested and reported, and never claims.
    descriptive = results.xattn_calibration.map(
        lambda c: t4_pilot.XATTN_CALIBRATIONS.get(c) == "descriptive")
    results["claim"] = results.tested & results.non_inferior & ~descriptive.astype(bool)
    return results


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
    cols = ["tier", "task", "band", "score_source", "sparsity", "xattn_threshold",
            "xattn_calibration",
            "n", "dense_correct", "sparse_correct", "b_sparse_only", "c_dense_only",
            "diff_pts", "lower_pts", "mean_realised_density", "tested", "claim"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        # Round the point columns only: a blanket round(1) printed sparsity
        # 0.75 as 0.8, a grid value that does not exist.
        print(res.sort_values(["tier", "task", "band", "score_source", "seq_pos"])[cols]
              .round({"diff_pts": 1, "lower_pts": 1, "mean_realised_density": 3})
              .to_string(index=False))
    print(f"\nmargin {t4_pilot.MARGIN_PTS} pts, one-sided alpha {t4_pilot.ALPHA}; "
          f"written to {out}/")


if __name__ == "__main__":
    main()
