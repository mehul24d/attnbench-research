#!/usr/bin/env python3
"""Compare two scorers (oracle vs cheap estimator) on the SAME model and band.

The mirror image of run_scale_comparison.py, and deliberately a separate
script rather than a flag on it. That one REFUSES when `score_source` differs,
because a scale comparison across two different scorers is the pattern-36
collision arriving through the analysis. Here a differing `score_source` is
the entire experiment, so the guard has to be inverted -- it refuses when the
scorers are the SAME. Weakening the other script's guard to serve both would
have deleted the check that caught a 30-point artifact.

Held fixed and asserted: model, band, tasks, n per cell, block_size,
mask_source, and `git_dirty=False` on both sides.

The dense arm is the calibration. `sdpa_flash` rows build no mask and consult
no scores, so the two runs compute the identical thing; any disagreement
between their dense rows is run-to-run noise, and it bounds how much of the
sparse difference can be believed. A scorer gap smaller than the dense
disagreement is not a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPARSE = "block_sparse"


def _load(path: str, label: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    if "git_dirty" in d and bool(d.git_dirty.any()):
        raise SystemExit(f"{label}: {int(d.git_dirty.sum())} rows carry "
                         f"git_dirty=True and cannot license a comparison.")
    return d


def _scorers(d: pd.DataFrame) -> set:
    return set(d[d.backend == SPARSE].score_source.dropna().unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--allow-n-mismatch", action="store_true")
    args = ap.parse_args()

    a = _load(args.oracle, "oracle")
    b = _load(args.cheap, "cheap")
    for d in (a, b):
        d.drop(d[~d.context_length.between(args.band * 0.75,
                                           args.band * 1.25)].index, inplace=True)
    if a.empty or b.empty:
        raise SystemExit(f"no rows at band {args.band} in one of the inputs")

    sa, sb = _scorers(a), _scorers(b)
    if sa == sb:
        raise SystemExit(
            f"both inputs use score_source={sorted(sa)}. This script exists to "
            f"compare two DIFFERENT scorers; identical ones mean one of the "
            f"paths is not the arm you think it is.")
    print(f"scorers  : oracle={sorted(sa)}  cheap={sorted(sb)}")

    for col in ("model_id", "block_size", "mask_source"):
        if col in a and col in b:
            va = set(a[col].dropna().unique()); vb = set(b[col].dropna().unique())
            if va != vb:
                raise SystemExit(f"{col} differs: {sorted(va)} vs {sorted(vb)}")
    if "model_id" in a:
        print(f"model    : {sorted(set(a.model_id.dropna().unique()))}")

    key = ["task", "backend", "sparsity"]
    da = a.groupby(key, dropna=False).agg(n=("score", "size"), mean=("score", "mean")).reset_index()
    db = b.groupby(key, dropna=False).agg(n=("score", "size"), mean=("score", "mean")).reset_index()
    m = da.merge(db, on=key, how="inner", suffixes=("_oracle", "_cheap"))
    bad = m[m.n_oracle != m.n_cheap]
    if not bad.empty and not args.allow_n_mismatch:
        raise SystemExit("n differs on these cells:\n"
                         + bad[key + ["n_oracle", "n_cheap"]].to_string(index=False))

    # Calibration first: it sets the bar everything below is read against.
    dense = m[m.backend != SPARSE]
    print("\n=== dense control (builds no mask, consults no scores) ===")
    noise = 0.0
    for _, r in dense.iterrows():
        d = abs(r.mean_oracle - r.mean_cheap)
        noise = max(noise, d)
        print(f"  {r.task:<16} oracle {r.mean_oracle:6.1f}  cheap {r.mean_cheap:6.1f}  |Δ| {d:.1f}")
    print(f"  run-to-run noise floor: {noise:.1f} points")

    print("\n=== oracle - cheap, on the sparse arm ===")
    print(f"{'task':<16}{'sparsity':>9}{'oracle':>9}{'cheap':>9}{'gap':>9}   verdict")
    for _, r in m[m.backend == SPARSE].sort_values(key).iterrows():
        gap = r.mean_oracle - r.mean_cheap
        if abs(gap) <= noise:
            v = f"within noise ({noise:.1f})"
        elif r.mean_cheap <= 1.0:
            v = "cheap at FLOOR -- gap understates"
        else:
            v = ""
        print(f"{r.task:<16}{r.sparsity:>9g}{r.mean_oracle:>9.1f}"
              f"{r.mean_cheap:>9.1f}{gap:>+9.1f}   {v}")

    print("\n=== each scorer vs its OWN dense control ===")
    for task in sorted(m.task.unique()):
        t = m[m.task == task]
        dz = t[t.backend != SPARSE]
        if dz.empty:
            continue
        do, dc = float(dz.mean_oracle.iloc[0]), float(dz.mean_cheap.iloc[0])
        print(f"  {task}: dense oracle-run {do:.1f} | cheap-run {dc:.1f}")
        for _, r in t[t.backend == SPARSE].sort_values("sparsity").iterrows():
            print(f"    sparsity {r.sparsity:<5g} oracle {r.mean_oracle:6.1f} "
                  f"({r.mean_oracle-do:+6.1f})  cheap {r.mean_cheap:6.1f} "
                  f"({r.mean_cheap-dc:+6.1f})")


if __name__ == "__main__":
    main()
