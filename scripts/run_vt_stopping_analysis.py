"""Is `vt` sparse-above-dense an attention result, or a stopping artifact?

Three explanations are on the table for the `vt` gap: the oracle acting as a
denoiser (claims.md), block structure suiting multi-hop tracking (Sparse
Frontier), and -- found 2026-09-19 from S1a's stop_reason columns -- the
sparse arms simply STOPPING DIFFERENTLY. `vt` is scored by recall over five
variable names under a 40-token cap, so an arm that emits a bare name list
and stops scores full, while an arm that restates the assignment chain runs
into the cap and loses the names it never reached. That is a property of the
output, not of the attention.

Only the third has a mechanism testable on banked data: condition on how the
example stopped and on how much it generated, and see whether the gap
survives. This does that per (file, band, sparsity):

  ALL       every paired example (what claims.md reports)
  SAME-STOP pairs whose dense and sparse rows share a stop_reason
  CAP-CAP   pairs where BOTH hit the token cap -- neither arm got the
            free ride, so the comparison is at equal output budget

Paired throughout: each sparse row against the dense row for the same
example, within one run, so model, band, scorer and host are held fixed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [
    ("1.5B 2048 pre-fix", "results/stage3_s1/accuracy.parquet"),
    ("1.5B 4096 pre-fix", "results/stage3_s1b/accuracy.parquet"),
    ("1.5B 8192 pre-fix", "results/stage3_s1b/accuracy.parquet"),
    ("1.5B 2048 forced-sink", "results/s1a/accuracy.parquet"),
    ("1.5B 16384 forced-sink", "results/accuracy_forced_sink/accuracy.parquet"),
    ("1.5B 16384 cheap", "results/accuracy_forced_sink_cheap/accuracy.parquet"),
    ("7B 16384 forced-sink", "results/s7_7b_16384/accuracy.parquet"),
    ("7B 16384 cheap", "results/s9_7b_cheap_16384/accuracy.parquet"),
]


def ci(x, rng, n=4000):
    if len(x) < 5:
        return (float("nan"), float("nan"))
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="vt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for label, path in RUNS:
        if not Path(path).exists():
            print(f"[skip] {label}: {path} absent", file=sys.stderr)
            continue
        d = pd.read_parquet(path)
        d = d[d.task == a.task]
        band = label.split()[1]
        d = d[d.example_id.str.rsplit("_", n=2).str[-2] == band]
        if d.empty:
            continue
        # dense arm only: gla rows are also sparsity-NaN and are not a baseline
        dense = d[d.sparsity.isna() & (d.backend == "sdpa_flash")].set_index("example_id")
        for s, g in d[~d.sparsity.isna()].groupby("sparsity"):
            g = g.set_index("example_id")
            common = dense.index.intersection(g.index)
            de, sp = dense.loc[common], g.loc[common]
            gap = sp.score.values - de.score.values
            same = (sp.stop_reason.values == de.stop_reason.values)
            cap = ((sp.stop_reason.values == "cap") & (de.stop_reason.values == "cap"))
            lo, hi = ci(gap, rng)
            slo, shi = ci(gap[same], rng)
            clo, chi = ci(gap[cap], rng)
            rows.append(dict(
                run=label, sparsity=f"{s:g}", n=len(common),
                gap_all=gap.mean(), all_ci=f"[{lo:+.1f},{hi:+.1f}]",
                n_same=int(same.sum()),
                gap_same=gap[same].mean() if same.any() else float("nan"),
                same_ci=f"[{slo:+.1f},{shi:+.1f}]",
                n_cap=int(cap.sum()),
                gap_cap=gap[cap].mean() if cap.any() else float("nan"),
                cap_ci=f"[{clo:+.1f},{chi:+.1f}]",
                dense_cap=int((de.stop_reason == "cap").sum()),
                sparse_cap=int((sp.stop_reason == "cap").sum()),
                dense_tok=de.n_generated.mean(), sparse_tok=sp.n_generated.mean()))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"task={a.task}: paired sparse-minus-dense, within run\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\ngap_all: every pair. gap_same: pairs that stopped the same way. "
          "gap_cap: pairs where both hit the cap.")
    if a.out:
        t.to_parquet(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
