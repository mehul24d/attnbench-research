"""S5: does the cold-cache fp32 / warm-cache fp16 split change masks, and
does it break nesting across the sparsity ladder?

`generation.generate_one` fetches scores per (arm, example). Arms run in the
order dense -> 0.5 -> 0.75 -> 0.9, so on a cold cache the 0.5 arm ranks from
the fp32 tensor `compute_importance_scores` returns on a miss, and 0.75 / 0.9
rank from the fp16 copy `score_cache.save` wrote. This reproduces that on CPU
with the real model and the real mask path (pool to block_size, mean over KV
heads, `masks.mask_for`), under both the current forced-sink rule and the
pre-2026-09-16 rule the banked 2048-8192 accuracy was measured with.

What it reports, per (band, rule):
  - blocks differing between the fp32 and fp16 mask at the same sparsity;
  - nesting breaks AS RUN: blocks in the fp16 0.75/0.9 mask absent from the
    fp32 0.5 mask (a single ranking gives zero by construction);
  - the same count within one ranking, as a control that must be zero.

CPU only. Usage:
    python scripts/check_score_dtype_asymmetry.py --bands 2048 4096 --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import masks  # noqa: E402
from attnbench.accuracy.config import load_grid  # noqa: E402
from attnbench.accuracy.generation import ModelGeometry  # noqa: E402
from attnbench.accuracy.grid_configs import build_examples_by_task_length  # noqa: E402
from attnbench.accuracy.model import (SwappableAttentionModel,  # noqa: E402
                                      pool_scores_to_block_size)
from attnbench.config import AttnConfig  # noqa: E402

SPARSITIES = (0.5, 0.75, 0.9)


def pre_fix_importance_mask(seq_len, block_size, sparsity, scores, *, identity_seed):
    """`masks.importance_block_mask` as it was before 37675a0: kv_block 0 is an
    ordinary candidate, nothing but the diagonal is granted free. Copied, not
    imported, because the current module no longer has this rule."""
    n = masks._n_blocks(seq_len, block_size)
    active = torch.zeros(n, n, dtype=torch.bool)
    active.fill_diagonal_(True)
    g = torch.Generator().manual_seed(masks._int_seed(identity_seed))
    for qb in range(n):
        kvs = list(range(qb))
        if not kvs:
            continue
        budget = round((1.0 - sparsity) * len(kvs))
        if budget <= 0:
            continue
        jitter = torch.rand(len(kvs), generator=g) * 1e-9
        top = (scores[qb, kvs] + jitter).topk(k=min(budget, len(kvs))).indices
        for j in top.tolist():
            active[qb, kvs[j]] = True
    return active


def build_mask(rule, cfg, importance):
    if rule == "forced_sink":
        return masks.mask_for(cfg, importance_scores=importance).active
    return pre_fix_importance_mask(cfg.seq_len, cfg.block_size, cfg.sparsity,
                                   importance,
                                   identity_seed=masks._mask_identity_key(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--bands", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--n", type=int, default=5, help="examples per (task, band)")
    ap.add_argument("--tasks", nargs="+", default=["niah_single", "niah_multikey", "vt"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    grid = load_grid(args.grid)
    model_id = grid.primary_model if hasattr(grid, "primary_model") else "Qwen/Qwen2.5-1.5B-Instruct"
    block_size = 128
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()
    geometry = ModelGeometry.from_config(model.config, "bfloat16")
    template = geometry.onto(AttnConfig(seq_len=max(args.bands), batch=1, n_heads_q=1,
                                        n_heads_kv=1, head_dim=128, mask="causal"),
                             seq_len=max(args.bands))
    wrapped = SwappableAttentionModel(model, template, model_id=model_id,
                                      finest_block_size=grid.finest_block_size)

    exs = build_examples_by_task_length(
        grid, seed=0, count_tokens=lambda t: len(tok(t).input_ids),
        tasks=tuple(args.tasks), seq_lens={b: args.n for b in args.bands})

    agg = {}
    for (task, band), examples in sorted(exs.items()):
        for ex in examples:
            ids = tok(ex.context, return_tensors="pt").input_ids
            L = int(ids.shape[-1])
            t0 = time.time()
            with tempfile.TemporaryDirectory() as cache:
                s32 = wrapped.compute_importance_scores(ids, task=task,
                        example_id=ex.example_id, cache_dir=cache)   # miss: fp32
                s16 = wrapped.compute_importance_scores(ids, task=task,
                        example_id=ex.example_id, cache_dir=cache)   # hit: fp16
            assert s32[0].dtype == torch.float32 and s16[0].dtype == torch.float16, \
                (s32[0].dtype, s16[0].dtype)
            for rule in ("pre_fix", "forced_sink"):
                a = agg.setdefault((band, rule), dict(
                    examples=0, layers=0, active_fp32={s: 0 for s in SPARSITIES},
                    differ={s: 0 for s in SPARSITIES},
                    layers_differing={s: 0 for s in SPARSITIES},
                    nest_break_as_run={s: 0 for s in SPARSITIES[1:]},
                    layers_nest_break={s: 0 for s in SPARSITIES[1:]},
                    nest_break_single_ranking=0))
                a["examples"] += 1
                for layer in range(wrapped.n_layers):
                    m = {}
                    for dt, sc in (("32", s32), ("16", s16)):
                        imp = pool_scores_to_block_size(
                            sc[layer], finest_block_size=grid.finest_block_size,
                            target_block_size=block_size).mean(dim=0)
                        for s in SPARSITIES:
                            cfg = replace(template, seq_len=L, mask="block_sparse",
                                          sparsity=s, block_size=block_size,
                                          mask_source="importance")
                            m[(dt, s)] = build_mask(rule, cfg, imp)
                    a["layers"] += 1
                    for s in SPARSITIES:
                        d = int((m[("32", s)] ^ m[("16", s)]).sum())
                        a["active_fp32"][s] += int(m[("32", s)].sum())
                        a["differ"][s] += d
                        a["layers_differing"][s] += d > 0
                    for s in SPARSITIES[1:]:
                        b = int((m[("16", s)] & ~m[("32", 0.5)]).sum())
                        a["nest_break_as_run"][s] += b
                        a["layers_nest_break"][s] += b > 0
                        a["nest_break_single_ranking"] += int(
                            (m[("16", s)] & ~m[("16", 0.5)]).sum())
            print(f"{task:14s} {band} {ex.example_id} L={L} {time.time()-t0:.0f}s", flush=True)

    rows = []
    for (band, rule), a in sorted(agg.items()):
        row = {"band": band, "rule": rule, **a}
        rows.append(row)
        print(f"\n== band {band}, rule {rule}: {a['examples']} examples, {a['layers']} layer-masks")
        for s in SPARSITIES:
            print(f"  sparsity {s}: fp32 vs fp16 differ in {a['differ'][s]} blocks "
                  f"of {a['active_fp32'][s]} active ({a['differ'][s]/max(1,a['active_fp32'][s]):.3%}); "
                  f"{a['layers_differing'][s]}/{a['layers']} layer-masks differ")
        for s in SPARSITIES[1:]:
            print(f"  nesting as run, fp16@{s} not in fp32@0.5: {a['nest_break_as_run'][s]} blocks, "
                  f"{a['layers_nest_break'][s]}/{a['layers']} layer-masks")
        print(f"  control, single ranking: {a['nest_break_single_ranking']} (must be 0)")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
