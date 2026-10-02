#!/usr/bin/env python3
"""Calibrate XAttention's per-(layer, head) thresholds for this study's model.

XAttention's own RULER evaluation does not use a scalar threshold. Its
`scripts/run_ruler.sh` passes none, so `FastPrefillConfig` loads
`llama_fuse_8`: a 32x32 table of per-(layer, head) thresholds, profiled for
Llama-3.1-8B by `xattn/threshold/profile_threshold/profile_threshold.py`.
No such table exists for Qwen2.5. This script makes one with that same
profiler, so the calibrated phase (docs/t4_xattention_calibrated.md) runs
the method as its authors ran RULER, not a scalar approximation of it.

The profiler is the authors' code, imported from the pinned checkout and
called per layer through `attnbench.backends.xattention.ThresholdProfiler`.
Per text, layer and head it finds the threshold on XAttention's own
estimate that selects at least the fewest blocks covering 90% of the exact
attention mass. The table is the maximum over texts, as the official
script's `final_threshold` is.

Two sources, one per run, each its own arm:

  authors        the method's own profiling set (`text.json` at the pinned
                 commit, 156 multi-document QA prompts), with the Llama-3
                 chat markers stripped because they mean nothing to Qwen.
                 Disjoint from RULER. This calibration carries the claim.
  ruler_heldout  RULER examples of the pilot's tasks at its bands, from
                 another seed, checked to share no context with any test
                 example. In-distribution and at the test lengths.
                 Descriptive only.

Writes `<out>/thresholds.json` (the table, its sha256 and its provenance,
committed to configs/xattn_thresholds/ before any test row uses it) and
`<out>/per_text.parquet` (every text's table, for the record).

    python scripts/calibrate_xattn_thresholds.py --source authors \\
        --out results/xattn_calibration_authors_<date>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import numerics, provenance                              # noqa: E402
from attnbench.accuracy import t4_pilot                                 # noqa: E402

CHAT_MARKER = re.compile(r"<\|[^|>]*\|>")


def strip_chat_markers(text: str) -> str:
    """Remove Llama-3 special-token markers (`<|begin_of_text|>`,
    `<|start_header_id|>` ...). To Qwen's tokenizer they are not special,
    so left in they would be profiled as literal text."""
    return CHAT_MARKER.sub("", text)


def assert_disjoint(calibration: dict, test: dict) -> None:
    """Refuse a calibration set that shares any context with a test example.

    `calibration` and `test` map (task, band) to lists of examples. A
    different seed should guarantee this; it is checked, not assumed."""
    test_contexts = {ex.context for exs in test.values() for ex in exs}
    shared = [(t, b, ex.example_id) for (t, b), exs in calibration.items()
              for ex in exs if ex.context in test_contexts]
    if shared:
        raise SystemExit(f"REFUSING: {len(shared)} calibration examples are test "
                         f"examples, e.g. {shared[:3]}")


def authors_texts() -> tuple[list[str], str]:
    import xattn
    path = (Path(xattn.__file__).resolve().parent / "threshold" / "profile_threshold"
            / "text.json")
    raw = path.read_bytes()
    texts = [strip_chat_markers(t) for t in json.loads(raw)]
    return texts, hashlib.sha256(raw).hexdigest()


def ruler_texts(grid, count_tokens) -> tuple[list[str], str]:
    from attnbench.accuracy.grid_configs import build_examples_by_task_length
    bands = {b: t4_pilot.XATTN_CALIBRATION_RULER_N for b in t4_pilot.PILOT_BANDS}
    cal = build_examples_by_task_length(
        grid, seed=t4_pilot.XATTN_CALIBRATION_SEED, count_tokens=count_tokens,
        tasks=t4_pilot.SPARSE_PILOT_TASKS, seq_lens=bands)
    test = build_examples_by_task_length(
        grid, seed=0, count_tokens=count_tokens, tasks=t4_pilot.SPARSE_PILOT_TASKS,
        seq_lens={b: max(t4_pilot.SPARSE_PILOT_N.values()) for b in t4_pilot.PILOT_BANDS})
    assert_disjoint(cal, test)
    texts = [ex.context for key in sorted(cal) for ex in cal[key]]
    digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    return texts, digest


def main() -> int:
    numerics.enforce_fp32_matmul()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(t4_pilot.XATTN_CALIBRATIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    prov = provenance.capture()
    if prov.git_dirty:
        raise SystemExit("REFUSING: uncommitted changes. The table is committed "
                         "and cited by commit; it must come from one.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from attnbench.accuracy.config import load_grid
    from attnbench.accuracy.generation import ModelGeometry
    from attnbench.accuracy.model import SwappableAttentionModel
    from attnbench.backends.xattention import (XATTN_COMMIT, ThresholdProfiler,
                                               ThresholdTable, installed_checkout)
    from attnbench.config import AttnConfig

    head, dirty = installed_checkout()
    if (head, dirty) != (XATTN_COMMIT, False):
        raise SystemExit(f"REFUSING: x-attention is {head} (dirty={dirty}), not "
                         f"{XATTN_COMMIT}. Run scripts/install_xattention.sh.")

    grid = load_grid(args.grid)
    model_id = grid.model_primary
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if args.source == "authors":
        texts, source_sha = authors_texts()
    else:
        texts, source_sha = ruler_texts(
            grid, lambda text: len(tokenizer(text).input_ids))

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    geometry = ModelGeometry.from_config(model.config, args.dtype)
    template = geometry.onto(AttnConfig(seq_len=1024, batch=1, n_heads_q=1, n_heads_kv=1,
                                        head_dim=128, mask="causal"), seq_len=1024)
    wrapped = SwappableAttentionModel(model, template, model_id=model_id,
                                      score_source=t4_pilot.XATTN_SCORE_SOURCE)
    profiler = ThresholdProfiler(n_layers=wrapped.n_layers, stride=t4_pilot.XATTN_STRIDE)
    lengths = []
    try:
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            lengths.append(int(ids.shape[-1]))
            cfg = geometry.onto(template, seq_len=lengths[-1])
            profiler.start_text()
            wrapped.run_measured(ids, profiler, cfg=cfg, logits_to_keep=1)
            profiler.end_text()
            print(f"  text {i + 1}/{len(texts)}: {lengths[-1]} tokens", flush=True)
    finally:
        wrapped.unwrap()

    table = profiler.table()
    values = [[round(float(x), 8) for x in row] for row in table.tolist()]
    doc = dict(
        name=args.source,
        sha256=ThresholdTable.digest(values),
        model=model_id, stride=t4_pilot.XATTN_STRIDE, block_size=128,
        exact_mass_coverage=0.9,                 # fixed inside the official profiler
        aggregation="max over texts",
        n_texts=len(texts),
        tokens=dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)),
        source_sha256=source_sha,
        source=("x-attention xattn/threshold/profile_threshold/text.json, chat markers stripped"
                if args.source == "authors" else
                f"RULER {list(t4_pilot.SPARSE_PILOT_TASKS)} at {list(t4_pilot.PILOT_BANDS)}, "
                f"seed {t4_pilot.XATTN_CALIBRATION_SEED}, "
                f"{t4_pilot.XATTN_CALIBRATION_RULER_N} per (task, band), "
                f"disjoint from the seed-0 test examples"),
        xattn_commit=XATTN_COMMIT,
        flash_attn_stubbed=profiler.flash_attn_stubbed,
        git_commit=prov.git_commit, git_dirty=prov.git_dirty,
        thresholds=values,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps(doc, indent=1) + "\n")
    per_text = torch.stack(profiler.per_text)
    pd.DataFrame([dict(text=t, layer=l, head=h, threshold=float(per_text[t, l, h]))
                  for t in range(per_text.shape[0]) for l in range(per_text.shape[1])
                  for h in range(per_text.shape[2])]).to_parquet(out / "per_text.parquet")

    flat = [x for row in values for x in row]
    print(f"\n{args.source}: {len(texts)} texts, {doc['tokens']} tokens; "
          f"table {len(values)}x{len(values[0])}, sha256 {doc['sha256'][:12]}")
    print(f"thresholds: min {min(flat):.3f}  median {statistics.median(flat):.3f}  "
          f"max {max(flat):.3f}  zero {sum(x == 0 for x in flat)}  "
          f">=0.95 {sum(x >= 0.95 for x in flat)} of {len(flat)}")
    for layer, row in enumerate(values):
        print(f"  layer {layer:2d}: " + " ".join(f"{x:.2f}" for x in row))
    print(f"\nwritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
