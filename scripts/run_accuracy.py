#!/usr/bin/env python
"""Stage 3 runner: end-to-end accuracy on RULER-style tasks.

    python scripts/run_accuracy.py --dry-run
    python scripts/run_accuracy.py --out results/accuracy

--dry-run builds the pinned grid (configs/accuracy/stage3_grid.yaml),
generates examples, and reports what would run/skip -- no model load, no
GPU, no --dry-run/real-run divergence in the planning logic itself (same
plan()-is-the-single-source-of-truth discipline as scripts/run_sweep.py).

Backends have curated, asymmetric roles, not a uniform sweep grid:
  - one dense baseline (default: sdpa_math) covers the sparsity=0 reference
    point. Every exact-math kernel (SDPA/FA2/FA3/cuDNN/xFormers/Flex)
    computes identical attention and Stage 1 already verifies that against
    a float64 reference -- so only one needs to actually run here; running
    all of them would test kernel-selection noise, not attention-mechanism
    differences, and Stage 3 is about the latter.
  - block_sparse contributes its sparsity sweep (3 sparsity levels; the
    real kernel only supports block_size=128, not 64 -- see
    stage3_grid.yaml).
  - gla (linear attention) and sage (quantized, --include-sage) each
    contribute their own single operating point -- neither has a sparsity
    knob in this codebase's config model.

The real (non-dry-run) execution path is NOT wired up yet: loading
grid.model_primary, wrapping it with SwappableAttentionModel, and decoding
generated text all need a GPU this project doesn't have quota for at the
time this was written. Running without --dry-run raises NotImplementedError
rather than silently producing wrong numbers.

Before committing the full grid on a rented session, run
scripts/time_one_accuracy_example.py first -- the per-length cost estimates
in the Stage 3 planning doc are a FLOPs-based projection (assumed 15
TFLOPS effective throughput), not a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench.accuracy.config import load_grid                        # noqa: E402
from attnbench.accuracy.grid_configs import (                          # noqa: E402
    build_configs_by_backend, build_examples_by_task_length)
from attnbench.accuracy.sizing import approximate_token_count  # noqa: E402
from attnbench.accuracy.runner import build_cells, run_accuracy        # noqa: E402


def _not_wired_up(cfg, backend_name, example):
    raise NotImplementedError(
        "run_accuracy.py's real execution path (loading grid.model_primary, "
        "wrapping with SwappableAttentionModel, generating and decoding "
        "text) isn't wired up -- this project doesn't have GPU quota yet. "
        "Use --dry-run, which builds and reports the grid without this call."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/accuracy")
    ap.add_argument("--include-sage", action="store_true",
                    help="Include the sage backend (conditional on H100 "
                         "quota per the Stage 3 plan; needs a real "
                         "quant_scheme confirmed and a matching gates.TOL "
                         "entry before this produces trustworthy rows).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    print(f"grid          : {args.grid}")
    print(f"model primary : {grid.model_primary}")
    print(f"model alt     : {grid.model_alternate}")
    print(f"tasks         : {', '.join(grid.tasks)}")
    print("seq_lens (n)  : " + ", ".join(
        f"{s}{'*' if grid.is_directional(s) else ''}={n}"
        for s, n in sorted(grid.seq_lens.items())))
    print("                (* = directional/underpowered point, see stage3_grid.yaml)")

    # Contexts are sized against a real tokenizer so a grid seq_len is an
    # exact token count (see attnbench/accuracy/sizing.py). A --dry-run only
    # exercises planning/cell-counting and must not require downloading a
    # tokenizer, so it uses the explicitly-named approximate counter -- and
    # says so, because a silent estimate is the failure this replaced.
    if args.dry_run:
        print("sizing        : APPROXIMATE (--dry-run; no tokenizer loaded) "
              "-- context lengths are structural only, not real token counts")
        count_tokens = approximate_token_count
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(grid.model_primary)

        def count_tokens(text: str) -> int:
            return len(tokenizer(text).input_ids)

        print(f"sizing        : exact, via {grid.model_primary} tokenizer")

    examples_by_task_length = build_examples_by_task_length(
        grid, seed=args.seed, count_tokens=count_tokens)
    examples_by_id = {
        (task, ex.example_id): ex
        for (task, _seq_len), exs in examples_by_task_length.items()
        for ex in exs
    }

    configs_by_backend = build_configs_by_backend(grid, include_sage=args.include_sage)
    print(f"backends      : {', '.join(configs_by_backend)}")

    cells = build_cells(configs_by_backend=configs_by_backend,
                        examples_by_task_length=examples_by_task_length)

    report = run_accuracy(cells, out_dir=Path(args.out),
                          examples_by_id=examples_by_id,
                          generate_fn=_not_wired_up, dry_run=args.dry_run)

    print(f"\ntotal cells   : {report.total}")
    print(f"would run     : {report.run}")
    print(f"skip (done)   : {report.skip_done}")
    if args.dry_run:
        print("\n[dry run -- nothing executed, nothing written]")
    else:
        print(f"\nwritten to {args.out}/accuracy.parquet")


if __name__ == "__main__":
    main()
