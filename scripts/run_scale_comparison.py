#!/usr/bin/env python3
"""Compare the accuracy arm across model scale at one band.

Exists because the 1.5B-vs-7B comparison is the accuracy analogue of the
L4-vs-A100 comparison, and the project has already been burned once by
reading a difference off two result sets without checking they were
comparable. So this refuses rather than reports when they are not.

Checked before any number is printed:
  - same band, same tasks, same n per (task, backend, sparsity)
  - same score_source on the sparse rows (an oracle arm and a cheap arm are
    not the same experiment -- that collision is pattern 36)
  - same block_size and mask_source
  - neither side carries git_dirty=True
  - same MASK ERA, from `git_commit` via `analysis/eras.py`. This one was
    missing until 2026-09-21, and its absence was being described in three
    documents as a guard that `analysis/composition.py` provided. It does
    not: composition refuses on facet composition and has no concept of a
    commit. Crossing an era boundary needs --cross-era LEFT:RIGHT with a
    stated reason, and the declaration is verified against the register.

The comparison is model-vs-model, so a host or card difference is NOT a
disqualifier and is deliberately not checked: an accuracy score does not
depend on which machine produced it, which is the same reasoning that keeps
the host-continuity guard out of the accuracy runner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.analysis import eras  # noqa: E402

KEY = ["task", "backend", "sparsity"]


def _load(path: str, label: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    if "git_dirty" in d and bool(d.git_dirty.any()):
        raise SystemExit(f"{label}: {int(d.git_dirty.sum())} rows carry "
                         f"git_dirty=True and cannot license a comparison.")
    return d


def _describe(d: pd.DataFrame) -> pd.DataFrame:
    return (d.groupby(KEY, dropna=False)
             .agg(n=("score", "size"), mean=("score", "mean"))
             .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True, help="accuracy.parquet, smaller model")
    ap.add_argument("--large", required=True, help="accuracy.parquet, larger model")
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--allow-n-mismatch", action="store_true",
                    help="report cells whose n differs instead of refusing. "
                         "Use only for a run still in progress; a partial "
                         "cell's mean is not the cell's mean.")
    eras.add_cross_era_flags(ap)
    args = ap.parse_args()

    a = _load(args.small, "small")
    b = _load(args.large, "large")
    a = a[a.context_length.apply(lambda c: abs(int(c) - args.band) <= args.band // 4)]
    b = b[b.context_length.apply(lambda c: abs(int(c) - args.band) <= args.band // 4)]
    if a.empty or b.empty:
        raise SystemExit(f"no rows at band {args.band} in one of the inputs")

    # After the band filter, so the commits reported are the ones that
    # actually contribute rows to the tables below rather than every commit
    # in the file.
    for line in eras.licence(*eras.sides(eras.commits_of(a), eras.commits_of(b),
                                         "small", "large"), args):
        print(line, flush=True)

    # A column present on ONE side only used to be skipped in silence: the
    # inner `for name, d in ...: if col not in d: continue` here was a loop
    # whose body was only that `continue`, so it iterated and did nothing, and
    # the real check then sat behind `if col in a and col in b`. So a file
    # carrying `mask_source` compared against one that does not passed the
    # comparability check without comment -- which is the same asymmetry the
    # column exists to expose. Refuse instead: a missing provenance column is
    # not evidence that the provenance matches.
    for col in ("score_source", "block_size", "mask_source"):
        present = [name for name, d in (("small", a), ("large", b)) if col in d]
        if not present:
            continue          # neither side records it; nothing to compare
        if len(present) == 1:
            raise SystemExit(
                f"{col} is recorded on the {present[0]} side only, so the two "
                f"inputs cannot be shown to share it. One file predates the "
                f"column and the other does not, which is itself a reason to "
                f"doubt they are the same experiment.")
        sa = set(a[a.backend == "block_sparse"][col].dropna().unique())
        sb = set(b[b.backend == "block_sparse"][col].dropna().unique())
        if sa != sb:
            raise SystemExit(
                f"{col} differs on the sparse rows: small={sorted(sa)} "
                f"large={sorted(sb)}. These are not the same experiment.")

    da, db = _describe(a), _describe(b)
    m = da.merge(db, on=KEY, how="outer", suffixes=("_small", "_large"))

    missing = m[m.n_small.isna() | m.n_large.isna()]
    if not missing.empty:
        print("cells present on only one side (excluded):", flush=True)
        print(missing[KEY].to_string(index=False), flush=True)
        m = m.dropna(subset=["n_small", "n_large"])

    bad = m[m.n_small != m.n_large]
    if not bad.empty and not args.allow_n_mismatch:
        raise SystemExit(
            "n differs on these cells; a partial cell's mean is not the "
            "cell's mean:\n"
            + bad[KEY + ["n_small", "n_large"]].to_string(index=False))

    m["delta"] = m.mean_large - m.mean_small
    m = m.sort_values(KEY)
    print(f"\n=== band {args.band}: small vs large ===", flush=True)
    print(m[KEY + ["n_small", "mean_small", "n_large", "mean_large", "delta"]]
          .to_string(index=False), flush=True)

    # The question the comparison exists to answer, stated per task: does
    # sparsity cost accuracy at this scale, relative to that scale's OWN dense
    # baseline? Comparing a sparse cell across models without re-basing on
    # each model's dense control would confound "sparsity is cheaper at scale"
    # with "the bigger model is better at the task".
    print("\n=== sparsity penalty vs each model's OWN dense control ===",
          flush=True)
    for task in sorted(m.task.unique()):
        t = m[m.task == task]
        dense = t[t.backend != "block_sparse"]
        if dense.empty:
            print(f"  {task}: no dense control on one side, skipped")
            continue
        ds, dl = float(dense.mean_small.iloc[0]), float(dense.mean_large.iloc[0])
        print(f"  {task}: dense small {ds:.1f} -> large {dl:.1f}", flush=True)
        for _, r in t[t.backend == "block_sparse"].sort_values("sparsity").iterrows():
            print(f"    sparsity {r.sparsity:<5g} "
                  f"small {r.mean_small:6.1f} ({r.mean_small-ds:+6.1f}) | "
                  f"large {r.mean_large:6.1f} ({r.mean_large-dl:+6.1f})",
                  flush=True)


if __name__ == "__main__":
    main()
