#!/usr/bin/env python3
"""Recompute Stage 6's dominance under a matched decode kernel and a matched
generation length, and print which operating points move.

    python scripts/run_decode_confound.py \
        --pareto results/stage6/pareto.parquet \
        --phases results/stage5/phases.parquet \
        results/stage3_s1/accuracy.parquet results/stage3_s1b/accuracy.parquet

Writes results/stage6/decode_corrected.parquet, carrying the measured value
beside both derived ones so nothing downstream can pick up a corrected number
without seeing what it was corrected from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.config import load_grid                # noqa: E402
from attnbench.analysis import decode_confound                 # noqa: E402
from attnbench.analysis.matched import band_for                # noqa: E402

from attnbench.accuracy.grid_configs import (                          # noqa: E402
    ACCURACY_EXCLUDED_BACKENDS as EXCLUDE_BACKENDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="Stage 3 accuracy parquet(s)")
    ap.add_argument("--pareto", default="results/stage6/pareto.parquet")
    ap.add_argument("--phases", default="results/stage5/phases.parquet")
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/stage6/decode_corrected.parquet")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    df = pd.concat([pd.read_parquet(p) for p in args.results], ignore_index=True)
    df = df[~df.backend.isin(EXCLUDE_BACKENDS)]
    df["_band"] = [band_for(int(c), grid.seq_lens) for c in df.context_length]

    n_gen = {}
    for (b, t, band, sp), rows in df.groupby(
            ["backend", "task", "_band", "sparsity"], dropna=False):
        n_gen[(b, t, int(band), None if pd.isna(sp) else float(sp))] = \
            float(rows.n_generated.mean())

    pareto = pd.read_parquet(args.pareto)
    phases = pd.read_parquet(args.phases)

    pts = decode_confound.correct(pareto, phases, n_gen, grid.dense_backend)
    out = decode_confound.to_dataframe(pts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    sparse = [p for p in pts if not p.is_dense_reference]
    print("=== decode-kernel penalty measured by Stage 5 ===")
    model = decode_confound.phase_model_from(phases, grid.dense_backend)
    for band in sorted({p.context_length for p in pts}):
        d = model.dense_decode(band)
        for sp in sorted({p.sparsity for p in sparse
                          if p.context_length == band and p.sparsity is not None}):
            pen = model.decode_penalty(band, sp)
            print(f"  {band:>5} sp={sp:<5g} dense {d:6.2f}  sparse "
                  f"{d + pen:6.2f}  penalty {pen:+6.2f} ms/token "
                  f"({100 * pen / d:+5.1f}%)")

    print("\n=== dominated by dense, of "
          f"{len(sparse)} matched sparse operating points ===")
    print(f"  as measured (what Stage 6 banked) : "
          f"{sum(p.dominated_measured for p in sparse)}/{len(sparse)}")
    print(f"  normalized (matched kernel + n)   : "
          f"{sum(p.dominated_normalized for p in sparse)}/{len(sparse)}")

    mv = decode_confound.movement(pts)
    for label, key in (("LEAVE the dominated set", "freed"),
                       ("ENTER the dominated set", "newly_dominated")):
        print(f"\n  points that {label}: {len(mv[key])}")
        for p in mv[key]:
            print(f"    {p.task:<12} {p.context_length:>5} eps={p.epsilon:<4g} "
                  f"sp={p.sparsity:<5g} measured {p.measured_ms:8.1f} -> "
                  f"normalized {p.normalized_ms:8.1f}")

    print("\n  `normalized` is DERIVED from Stage 5 phases (random ids, band "
          "length, n=10),\n  not measured at these operating points. It is "
          "what the arms WOULD have cost\n  under a matched decode kernel.")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
