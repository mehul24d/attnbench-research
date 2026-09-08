#!/usr/bin/env python3
"""Stage 6: Pareto frontiers per grid cell, with the dense baseline carried
as a point rather than left implicit.

LATENCY SOURCE, AND WHAT IT IS NOT

Stage 5 (dedicated end-to-end timing at Stage 4's certified points) is not
built. This uses Stage 3's per-row `latency_ms` instead -- one wall-clock
measurement of `generate()` per example, aggregated over n=900 per arm.

That is a weaker instrument than Stage 5 in three specific ways, all stated
because the result is strong enough not to need help:

1. No warmup control and no repeats. Mitigated by ordering rather than by
   design: the dense arm runs FIRST in every band, so warmup penalises the
   baseline, not the sparse arms. The observed gap is therefore conservative
   (dense's first 50 rows at 8192 average 1911 ms against 1663 for the rest).
2. It excludes the importance-scoring pass, by the same decision that
   excludes it from every other latency number in the study. Including it
   would make the sparse arms slower still.
3. It cannot be decomposed into prefill and decode. An attempt to do so did
   not reconcile against the session's measured per-step decode figure, and
   this project's history says such a mismatch is a regime error rather than
   an arithmetic one. The end-to-end gap does not depend on the split, so the
   split is left to Stage 5 rather than guessed at here.

THE SCOPE THAT MAKES THE RESULT READABLE

Sparsity here is applied during PREFILL ONLY; generation runs dense over the
cache (`decode_backend` on every Stage 3 row). At batch 1, generating ~30
tokens, decode is the larger share of end-to-end cost and sparsity does not
touch it. So this measures a regime in which prefill sparsity attacks the
smaller part of the bill by construction.

That is a scope statement, not a defence: it is exactly the regime a
single-stream interactive deployment runs in. But it is why the finding must
be read as "in this regime" and not as "sparse attention is always slower".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.config import load_grid                       # noqa: E402
from attnbench.analysis import decode_backend_guard            # noqa: E402
from attnbench.analysis.matched import (                              # noqa: E402
    band_for, best_matched_sparsity_budget, run_matched_analysis)
from attnbench.analysis.pareto import (                               # noqa: E402
    compute_pareto_frontiers, to_dataframe)

from attnbench.accuracy.grid_configs import (                          # noqa: E402
    ACCURACY_EXCLUDED_BACKENDS as EXCLUDE_BACKENDS)


def latency_table(df: pd.DataFrame, seq_lens) -> dict:
    """Mean latency per (backend, task, band, sparsity) -- the key an
    operating point has, so Stage 4 rows and these join identically."""
    d = df.copy()
    d["_band"] = [band_for(int(c), seq_lens) for c in d["context_length"]]
    # Before any mean over latency_ms. DENSE_DECODE_BACKEND changed
    # sdpa_math -> sdpa_flash on 2026-09-08 and the gap is 23-64% per decode
    # token, so a cell pooling both eras describes neither.
    decode_backend_guard.assert_uniform(d)
    out = {}
    g = d.groupby(["backend", "task", "_band", "sparsity"], dropna=False)
    for (backend, task, band, sparsity), rows in g:
        key = (backend, task, int(band),
               None if pd.isna(sparsity) else float(sparsity))
        out[key] = float(rows["latency_ms"].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/stage6")
    ap.add_argument("--epsilons", default="1,2,5")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    df = pd.concat([pd.read_parquet(p) for p in args.results], ignore_index=True)
    df = df[~df["backend"].isin(EXCLUDE_BACKENDS)]
    epsilons = tuple(float(e) for e in args.epsilons.split(","))

    matched = run_matched_analysis(
        df, grid, dense_backend=grid.dense_backend,
        sparse_backends=("block_sparse",), epsilons=epsilons)
    lat = latency_table(df, grid.seq_lens)

    results = compute_pareto_frontiers(matched, lat, dense_backend=grid.dense_backend)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tab = to_dataframe(results)
    tab.to_parquet(out / "pareto.parquet", index=False)

    print("=== end-to-end latency by arm (Stage 3 generate(), n=900/arm) ===")
    lt = pd.DataFrame(
        [dict(backend=k[0], task=k[1], band=k[2], sparsity=k[3], latency_ms=v)
         for k, v in lat.items()])
    print(lt.pivot_table(index=["backend", "sparsity"], columns="band",
                         values="latency_ms", dropna=False).round(0).to_string())

    sparse = tab[~tab.is_dense_reference]
    print(f"\n=== matched sparse operating points: {len(sparse)} ===")
    print(f"dominated by the dense baseline: "
          f"{int(sparse.dominated_by_dense.sum())} / {len(sparse)}")
    surv = sparse[~sparse.dominated_by_dense]
    if len(surv):
        print("\nNOT dominated by dense (faster, or better on the accuracy axis):")
        print(surv[["task", "context_length", "epsilon", "sparsity",
                    "latency_ms", "margin"]].round(2).to_string(index=False))
    print(f"\nwritten to {out}/pareto.parquet")


if __name__ == "__main__":
    main()
