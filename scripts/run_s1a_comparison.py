"""Audit S1a: per-example comparison of forced-sink accuracy against the
banked pre-fix rows, same example ids, same (pinned) decode kernel.

Accuracy is deterministic per example given the mask, and the dense canary
has already shown the environment reproduces, so the comparison is exact on
the overlap: every difference is a flip, and the flips are attributable to
the sink rule (the only thing S1a changes). Reported per (task, sparsity):
banked and new accuracy, both sparse-minus-dense deltas, and flips in each
direction.

Refuses rather than compares when the new rows are not decode-pinned to the
banked kernel, when the dense arm does not match exactly, or when any cell
has other than the expected number of paired examples.

    python scripts/run_s1a_comparison.py --new results/s1a/accuracy.parquet \\
        --banked results/stage3_s1/accuracy.parquet results/stage3_s1b/accuracy.parquet \\
        --seq-len 2048 --n 100
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

KEY = ["backend", "sparsity", "task", "example_id"]


def band_of(df):
    return df["example_id"].str.rsplit("_", n=2).str[-2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--banked", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--n", type=int, required=True, help="examples per (task, arm)")
    ap.add_argument("--out", default=None, help="write the table as parquet")
    a = ap.parse_args()

    new = pd.read_parquet(a.new)
    old = pd.concat([pd.read_parquet(p) for p in a.banked], ignore_index=True)
    new = new[band_of(new) == str(a.seq_len)]
    old = old[(band_of(old) == str(a.seq_len)) & old.backend.isin(["sdpa_flash", "block_sparse"])]

    sp = new[new.backend == "block_sparse"]
    if "decode_pinned" not in new.columns or not sp["decode_pinned"].all():
        sys.exit("REFUSED: new sparse rows are not all decode_pinned")
    kernels_new = set(sp.decode_backend)
    kernels_old = set(old[old.backend == "block_sparse"].decode_backend)
    if kernels_new != kernels_old:
        sys.exit(f"REFUSED: sparse decode kernel {kernels_new} vs banked {kernels_old}")

    m = new.merge(old, on=KEY, suffixes=("_new", "_old"))
    # sparsity NaN (dense) does not merge on NaN in every pandas; do dense apart
    dn = new[new.backend == "sdpa_flash"].merge(
        old[old.backend == "sdpa_flash"], on=["task", "example_id"], suffixes=("_new", "_old"))
    if len(dn) == 0 or (dn.predicted_new != dn.predicted_old).any():
        sys.exit(f"REFUSED: dense arm not identical ({len(dn)} paired)")

    rows = []
    for task in sorted(m.task.unique()):
        d = dn[dn.task == task]
        if len(d) != a.n:
            sys.exit(f"REFUSED: {task} dense has {len(d)} paired examples, expected {a.n}")
        dense = d.score_new.mean()
        for s, g in m[(m.task == task) & (m.backend == "block_sparse")].groupby("sparsity"):
            if len(g) != a.n:
                sys.exit(f"REFUSED: {task}/{s} has {len(g)} paired examples, expected {a.n}")
            o, n_ = g.score_old.mean(), g.score_new.mean()
            rows.append(dict(
                task=task, sparsity=f"{s:g}", n=len(g), dense=dense,
                old=o, new=n_, old_delta=o - dense, new_delta=n_ - dense,
                shift=n_ - o,
                wrong_to_right=int((~g.correct_old & g.correct_new).sum()),
                right_to_wrong=int((g.correct_old & ~g.correct_new).sum()),
                text_changed=int((g.predicted_new != g.predicted_old).sum())))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"S1a band {a.seq_len}: forced sink vs banked pre-fix, same {a.n} examples, "
          f"sparse decode {sorted(kernels_new)} both sides; dense identical "
          f"{len(dn)}/{len(dn)}\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if a.out:
        t.to_parquet(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
