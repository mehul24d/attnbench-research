#!/usr/bin/env python3
"""Measure what the deployable estimator costs, per layer, against what
sparsity saves (audit C1, C2).

**What this settles.** claims.md says the deployable estimator's cost was
never measured: the only cost on record is the oracle's (11.8 s per example
at 16384, ~36x the saving). The inline arm (`score_source=
"minference_meanpool_inline"`) runs MInference's mean-pool estimator and the
exact device mask builder inside every layer's forward. This times those two
steps in isolation, at the real model's head geometry, next to the two
kernels whose difference is the saving:

  meanpool       minference_meanpool_scores_on_device(q, k)
  mask_build     masks.importance_block_mask_device, warm, per sparsity
  dense          sdpa_flash causal attention (the dense arm's kernel)
  block_sparse   Block-Sparse-Attention at that sparsity's mask
  xattn_estimate XAttention's official estimator per threshold, when the
                 package is installed (skipped and recorded otherwise)

All per layer, one example, batch 1; CUDA events around each call, warm-up
excluded, every sample banked. Inputs are random: the mean-pool and the
builder do the same arithmetic on any values, and the block-sparse kernel's
work is fixed by the mask's density, which the sparsity fixes.

It also runs the warm estimator and builder once under
`torch.cuda.set_sync_debug_mode("error")` and records whether they
synchronised (`sync_free`). The inline arm exists to be sync-free; until
8a758ab it was not.

**What would falsify what, stated before the run:**

  - meanpool + mask_build >= dense - block_sparse at a sparsity
        -> the deployable estimator does not pay for itself there, kernel
           against kernel, before any framework overhead
  - sync_free is False -> the inline arm's latency includes a per-layer
           device sync; its end-to-end numbers must not be reported
  - mask_build varies with sparsity by more than its own spread
        -> the builder is not the fixed-cost sort-and-scatter it is
           designed to be

**What it does not settle.** Per-layer costs exclude the model's other work,
so "pays for itself" here is necessary, not sufficient, for an end-to-end
speedup; the end-to-end comparison is the accuracy rows' latency and a
dedicated timing run (docs/t4_sparse_pilot.md). XAttention on a card whose
name lacks "100" runs its torch fallback, not its Triton kernel, so its cost
there is not its official cost; the row records the device.

Usage (on a CUDA instance, from the repo root):
    python3 scripts/measure_estimator_cost.py --out results/estimator_cost_<date>
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import masks, provenance                                  # noqa: E402
from attnbench.accuracy.grid_configs import backend_instance             # noqa: E402
from attnbench.accuracy.model import minference_meanpool_scores_on_device  # noqa: E402
from attnbench.config import AttnConfig                                  # noqa: E402

# Qwen2.5-1.5B-Instruct, the grid's primary model: 28 layers, 12 query heads,
# 2 KV heads, head_dim 128. Checked against the HF config when --model is
# given and the geometry flags are not.
QWEN_1_5B = dict(layers=28, heads_q=12, heads_kv=2, head_dim=128)


def _timer(device: str):
    """(start, stop) -> ms. CUDA events on a GPU; perf_counter on CPU, which
    exists for the smoke test and is never banked as a measurement."""
    if device.startswith("cuda"):
        def run(fn):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b)
    else:
        def run(fn):
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1000
    return run


def _samples(fn, *, device: str, warmup: int, reps: int) -> list[float]:
    run = _timer(device)
    for _ in range(warmup):
        run(fn)
    return [run(fn) for _ in range(reps)]


def _row(component, seq_len, samples, **extra):
    return dict(component=component, seq_len=seq_len,
                median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), reps=len(samples), samples_ms=list(samples),
                available=True, reason="", **extra)


def _unavailable(component, seq_len, reason, **extra):
    return dict(component=component, seq_len=seq_len, median_ms=float("nan"),
                min_ms=float("nan"), max_ms=float("nan"), reps=0, samples_ms=[],
                available=False, reason=reason[:300], **extra)


def sync_free(fn, device: str) -> bool | None:
    if not device.startswith("cuda"):
        return None
    fn()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
        return True
    except RuntimeError:
        return False
    finally:
        torch.cuda.set_sync_debug_mode("default")


def measure(*, seq_lens, sparsities, thresholds, block_size, geometry, device,
            dtype, warmup, reps) -> list[dict]:
    rows = []
    hq, hkv, d = geometry["heads_q"], geometry["heads_kv"], geometry["head_dim"]
    tdtype = getattr(torch, dtype)
    g = torch.Generator(device=device).manual_seed(0)
    for s in seq_lens:
        q = torch.randn(1, hq, s, d, device=device, dtype=tdtype, generator=g)
        k = torch.randn(1, hkv, s, d, device=device, dtype=tdtype, generator=g)
        v = torch.randn(1, hkv, s, d, device=device, dtype=tdtype, generator=g)
        cfg = AttnConfig(seq_len=s, batch=1, n_heads_q=hq, n_heads_kv=hkv,
                         head_dim=d, dtype=dtype, mask="causal")

        def meanpool():
            return minference_meanpool_scores_on_device(q, k, n_heads_kv=hkv,
                                                        block_size=block_size)
        rows.append(_row("meanpool", s, _samples(meanpool, device=device,
                                                  warmup=warmup, reps=reps)))
        ranking = meanpool().mean(dim=0)

        try:
            dense = backend_instance("sdpa_flash")
            rows.append(_row("dense", s, _samples(lambda: dense.forward(q, k, v, cfg),
                                                   device=device, warmup=warmup, reps=reps)))
        except Exception as e:   # recorded, not raised: one missing kernel must not hide the rest
            rows.append(_unavailable("dense", s, f"{type(e).__name__}: {e}"))

        try:
            sparse_backend = backend_instance("block_sparse")
            sparse_backend._import_check()
        except Exception as e:
            sparse_backend, sparse_reason = None, f"{type(e).__name__}: {e}"

        for sp in sparsities:
            sp_cfg = AttnConfig(seq_len=s, batch=1, n_heads_q=hq, n_heads_kv=hkv,
                                head_dim=d, dtype=dtype, mask="block_sparse",
                                sparsity=sp, block_size=block_size,
                                mask_source="importance")
            seed = masks._mask_identity_key(sp_cfg)

            def build():
                return masks.importance_block_mask_device(
                    s, block_size, sp, ranking, causal=True, identity_seed=seed)
            mask = build()
            kept = mask.active.float().mean().item()
            rows.append(_row("mask_build", s, _samples(build, device=device,
                                                        warmup=warmup, reps=reps),
                             sparsity=sp, block_density=kept))

            def estimator():
                sc = minference_meanpool_scores_on_device(q, k, n_heads_kv=hkv,
                                                          block_size=block_size)
                return masks.importance_block_mask_device(
                    s, block_size, sp, sc.mean(dim=0), causal=True, identity_seed=seed)
            rows[-1]["sync_free"] = sync_free(estimator, device)

            if sparse_backend is None:
                rows.append(_unavailable("block_sparse", s, sparse_reason, sparsity=sp))
            else:
                try:
                    rows.append(_row("block_sparse", s, _samples(
                        lambda: sparse_backend.forward(q, k, v, sp_cfg, mask=mask),
                        device=device, warmup=warmup, reps=reps), sparsity=sp,
                        block_density=kept))
                except Exception as e:
                    rows.append(_unavailable("block_sparse", s,
                                             f"{type(e).__name__}: {e}", sparsity=sp))

        for tau in thresholds:
            try:
                from attnbench.backends.xattention import XAttentionBackend
                from attnbench.backends.impls import _expand_kv
                xb = XAttentionBackend(threshold=tau)
                xb._import_check()
                k_rep, _ = _expand_kv(k, v, cfg)
                rows.append(_row("xattn_estimate", s, _samples(
                    lambda: xb.estimate(q, k_rep), device=device,
                    warmup=warmup, reps=reps), threshold=tau))
            except Exception as e:
                rows.append(_unavailable("xattn_estimate", s,
                                         f"{type(e).__name__}: {e}", threshold=tau))
    return rows


def summarise(df: pd.DataFrame, layers: int) -> pd.DataFrame:
    """Per (seq_len, sparsity): estimator vs saving, per layer and per forward.
    Only where every component was measured."""
    ok = df[df.available]

    def med(component, s, sp=None):
        m = ok[(ok.component == component) & (ok.seq_len == s)]
        if sp is not None:
            m = m[m.sparsity == sp]
        return m.median_ms.iloc[0] if len(m) == 1 else None

    out = []
    for _, r in ok[ok.component == "block_sparse"].iterrows():
        s, sp = r.seq_len, r.sparsity
        dense, meanpool, build = med("dense", s), med("meanpool", s), med("mask_build", s, sp)
        if None in (dense, meanpool, build):
            continue
        saving, est = dense - r.median_ms, meanpool + build
        out.append(dict(seq_len=s, sparsity=sp, dense_ms=dense,
                        block_sparse_ms=r.median_ms, saving_ms=saving,
                        meanpool_ms=meanpool, mask_build_ms=build, estimator_ms=est,
                        estimator_over_saving=(est / saving if saving > 0 else float("inf")),
                        net_saving_per_forward_ms=layers * (saving - est)))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-lens", default="16384,32768")
    ap.add_argument("--sparsities", default="0.5,0.75,0.9")
    ap.add_argument("--xattn-thresholds", default="0.9,0.95",
                    help="XAttention thresholds to time when installed; '' for none")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="geometry is checked against this model's HF config")
    ap.add_argument("--skip-config-check", action="store_true",
                    help="use the built-in Qwen2.5-1.5B geometry unchecked (no download)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    geometry = dict(QWEN_1_5B)
    if not args.skip_config_check:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(args.model)
        real = dict(layers=c.num_hidden_layers, heads_q=c.num_attention_heads,
                    heads_kv=c.num_key_value_heads,
                    head_dim=getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads)
        if real != geometry:
            raise SystemExit(f"geometry {geometry} does not match {args.model}: {real}")

    stamp = provenance.capture().to_dict()
    if stamp.get("git_dirty") and not args.allow_dirty:
        raise SystemExit("REFUSING: dirty tree; the commit would not establish the code measured")

    rows = measure(
        seq_lens=[int(x) for x in args.seq_lens.split(",") if x.strip()],
        sparsities=[float(x) for x in args.sparsities.split(",") if x.strip()],
        thresholds=[float(x) for x in args.xattn_thresholds.split(",") if x.strip()],
        block_size=args.block_size, geometry=geometry, device=args.device,
        dtype=args.dtype, warmup=args.warmup, reps=args.reps)
    df = pd.DataFrame(rows)
    df["device_name"] = (torch.cuda.get_device_name() if args.device.startswith("cuda")
                         else "cpu")
    for key, val in geometry.items():
        df[key] = val
    df["block_size"] = args.block_size
    provenance.stamp_onto(df, stamp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "estimator_cost.parquet", index=False)
    summary = summarise(df, geometry["layers"])
    summary.to_csv(out / "estimator_vs_saving.csv", index=False)
    cols = [c for c in ("component", "seq_len", "sparsity", "threshold", "median_ms",
                        "available", "sync_free", "reason") if c in df]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df[cols].to_string(index=False))
        print()
        print(summary.round(3).to_string(index=False))
    print(f"\nwritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
