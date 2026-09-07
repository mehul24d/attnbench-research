#!/usr/bin/env python3
"""Stage 5: measure prefill, one decode step, mask build and the scoring pass
separately, then check they compose against a known end-to-end total.

The reconciliation is printed AT THE TERMINAL as each band completes. The
62.1 / 55.5 / 580 ms discrepancy is what this session exists to settle, and a
result that only surfaces in analysis a day later is a result that gets
settled a day later on a machine that has been deleted.

No accuracy rows are written, so the Stage 0/1 correctness probe is not a
precondition here. The clock lock IS: this is the project's first per-phase
kernel timing, the canary showed 6% host-to-host variance with clocks
unlocked, and a decomposition is exactly the measurement that variance
corrupts. Its outcome is stamped on every row and reported loudly if it fails.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                                       # noqa: E402
from attnbench.accuracy.config import load_grid                        # noqa: E402
from attnbench.accuracy.generation import ModelGeometry                # noqa: E402
from attnbench.accuracy.grid_configs import backend_instance           # noqa: E402
from attnbench.accuracy.phase_timing import (                          # noqa: E402
    measure_band, scoring_overhead_ratio)
from attnbench.config import AttnConfig                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/stage5")
    ap.add_argument("--bands", default="2048,4096,8192")
    ap.add_argument("--sparsities", default="0.5,0.75,0.9")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lock-clocks", action="store_true")
    ap.add_argument("--observed", default=None,
                    help="Stage 3 accuracy.parquet(s), comma-separated, for "
                         "the observed end-to-end totals the phases are "
                         "reconciled against. Without them the phases are "
                         "measured but nothing is checked, which is most of "
                         "the point.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from attnbench.accuracy.model import SwappableAttentionModel

    grid = load_grid(args.grid)
    bands = [int(b) for b in args.bands.split(",")]
    sparsities = [float(s) for s in args.sparsities.split(",")]

    clocks_locked = False
    if args.lock_clocks:
        clocks_locked = provenance.lock_clocks()
        if not clocks_locked:
            print("!! CLOCK LOCK FAILED. A per-phase decomposition is exactly "
                  "the measurement host clock variance corrupts (the canary "
                  "showed 6% unlocked). Every row below is stamped "
                  "clocks_locked=False; treat the split as indicative.",
                  flush=True)
        else:
            print("clocks locked -- stamped on every row", flush=True)

    tok = AutoTokenizer.from_pretrained(grid.model_primary)
    model = AutoModelForCausalLM.from_pretrained(
        grid.model_primary, torch_dtype=getattr(torch, args.dtype))
    model = model.to(args.device).eval()
    geom = ModelGeometry.from_config(model.config, args.dtype)

    rows, recs = [], []
    observed = {}
    if args.observed:
        from attnbench.analysis.matched import band_for
        df = pd.concat([pd.read_parquet(p) for p in args.observed.split(",")],
                        ignore_index=True)
        df = df[df.backend != "gla"]
        df["_band"] = [band_for(int(c), grid.seq_lens) for c in df.context_length]
        g = df.groupby(["backend", "sparsity", "_band"], dropna=False)
        for (b, sp, band), r in g:
            observed[(b, None if pd.isna(sp) else float(sp), int(band))] = (
                float(r.latency_ms.mean()), float(r.n_generated.mean()))

    def cfg_for(band, mask, sparsity):
        base = AttnConfig(seq_len=band, batch=1, n_heads_q=1, n_heads_kv=1,
                          head_dim=128, mask=mask, sparsity=sparsity,
                          block_size=grid.finest_block_size,
                          mask_source="importance" if mask == "block_sparse" else None)
        return geom.onto(base, seq_len=band)

    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Stop tokens are deliberately EMPTY for timing: `generate` must run
    # exactly `max_new_tokens` steps, or the divisor in the slope below is not
    # the number of steps that ran.
    NO_STOPS = dict(eos_token_ids=frozenset(), newline_token_ids=frozenset(),
                    whitespace_token_ids=frozenset())
    K_LO, K_HI = 1, 8

    scratch = tempfile.mkdtemp(prefix="stage5_scores_")
    arms = [("sdpa_flash", None)] + [("block_sparse", sp) for sp in sparsities]

    for band in bands:
        print(f"\n===== band {band} =====", flush=True)
        ids = torch.randint(0, model.config.vocab_size, (1, band), device=args.device)
        wrapped = SwappableAttentionModel(
            model, cfg_for(band, "causal", None), model_id=grid.model_primary,
            finest_block_size=grid.finest_block_size)
        try:
            # The SAME body tests/test_phase_timing_interfaces.py drives
            # against a toy model. Both 2026-09-07 failures were in a script
            # body no test could reach.
            band_rows, band_recs, scoring_ms, prefill_ms = measure_band(
                wrapped, ids, band=band, arms=arms, cfg_for=cfg_for,
                scratch_dir=scratch, observed=observed, warmup=args.warmup,
                reps=args.reps, scoring_reps=max(3, args.reps // 3),
                synchronize=sync, clocks_locked=clocks_locked,
                backend_factory=backend_instance)
        finally:
            wrapped.unwrap()

        rows.extend(band_rows)
        recs.extend(band_recs)
        print(f"  scoring          {scoring_ms:9.1f} ms", flush=True)
        for (name, sp), pf in prefill_ms.items():
            step = [r for r in band_rows if r.phase == "decode_step"
                    and r.backend == name and r.sparsity == sp][0].ms_mean
            print(f"  {name:<13}{'' if sp is None else sp:>5}  "
                  f"prefill {pf:8.1f}  decode/step {step:7.2f}", flush=True)
        for r in band_recs:
            print("    " + r.render(), flush=True)
        for sp in sparsities:
            ratio = scoring_overhead_ratio(
                scoring_ms, prefill_ms[("sdpa_flash", None)],
                prefill_ms[("block_sparse", sp)])
            if ratio == ratio:
                print(f"  scoring costs {ratio:.1f}x what sparsity {sp:g} saved "
                      f"on prefill", flush=True)
            else:
                print(f"  sparsity {sp:g} saved NOTHING on prefill -- scoring "
                      f"overhead ratio undefined, which is itself the finding",
                      flush=True)

        pd.DataFrame([r.to_dict() for r in rows]).to_parquet(
            out / "phases.parquet", index=False)
        if recs:
            pd.DataFrame([r.to_dict() for r in recs]).to_parquet(
                out / "reconciliation.parquet", index=False)

    prov = provenance.capture().to_dict()
    df = pd.DataFrame([r.to_dict() for r in rows])
    for k, v in prov.items():
        df[k] = v
    df.to_parquet(out / "phases.parquet", index=False)
    if recs:
        pd.DataFrame([r.to_dict() for r in recs]).to_parquet(
            out / "reconciliation.parquet", index=False)
        bad = [r for r in recs if not r.closes]
        print(f"\n=== RECONCILIATION: {len(recs) - len(bad)}/{len(recs)} close "
              f"within {int(100 * 0.10)}% ===")
        for r in bad:
            print("  " + r.render())
        if bad:
            print("\n  A residual outside tolerance is NOT to be fixed by "
                  "adjusting an input until it fits. It means one of the "
                  "three numbers measures something other than what it is "
                  "being used for.")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
