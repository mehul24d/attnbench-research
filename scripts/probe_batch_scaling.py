#!/usr/bin/env python
"""Measure batch-size scaling (peak memory and per-example wall time) for
each dense-equivalent backend at a given context length, and write the rows
to parquet with a provenance stamp.

    python scripts/probe_batch_scaling.py --seq-len 16384
    python scripts/probe_batch_scaling.py --seq-len 16384 --batches 1,2,4,8,12

Needs a GPU and the real model. Run it on an **idle** machine: per-example
wall time is the measurement the compute-bound conclusion rests on, and a
number taken while a kernel build saturates the CPUs reads low for reasons
that have nothing to do with the GPU.

An OOM is a recorded result (`status="oom"`), not a failed run -- the sweep
continues to the next backend rather than aborting, so one backend's memory
wall doesn't cost the measurements of the others.

block_sparse is included when installed but is measured at batch=1 only: its
importance-scoring path is architecturally batch=1 (model.py's
_scoring_forward_chunked raises, and state.scores has no batch dimension), so
a batch>1 row for it would be a measurement of something the study cannot
actually run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from attnbench import provenance  # noqa: E402
from attnbench.accuracy.batch_scaling import (  # noqa: E402
    BatchScalingResult, batching_speedup, per_example_seconds)
from attnbench.accuracy.config import load_grid  # noqa: E402
from attnbench.accuracy.grid_configs import build_examples_by_task_length  # noqa: E402
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.backends.impls import SDPABackend  # noqa: E402
from attnbench.backends.linear import GatedLinearAttention  # noqa: E402
from attnbench.checkpoint import append_checkpoint  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--batches", default="1,2,4,8,12,16")
    ap.add_argument("--task", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/batch_scaling")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a GPU")

    batches = [int(b) for b in args.batches.split(",")]
    grid = load_grid(args.grid)
    task = args.task or grid.tasks[0]
    seq_len = args.seq_len
    model_id = grid.model_primary

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Tokenizer first: contexts are sized against it to hit an exact token
    # budget, so a grid seq_len of 16384 is 16384 tokens rather than the
    # ~19821 the old word-count heuristic produced. The memory figures this
    # probe reports depend directly on that -- see the hard gate in
    # docs/retry_session_runbook.md for the recomputed expected values.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text).input_ids)

    example = build_examples_by_task_length(
        grid, seed=args.seed, tasks=(task,), seq_lens={seq_len: 1},
        count_tokens=count_tokens)[(task, seq_len)][0]

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to("cuda").eval()

    ids1 = tokenizer(example.context, return_tensors="pt").input_ids.to("cuda")
    real = ids1.shape[-1]
    n_heads_q = model.config.num_attention_heads
    n_heads_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // n_heads_q)

    print(f"model         : {model_id}")
    print(f"seq_len       : {seq_len} nominal -> {real} real tokens "
          f"({real / seq_len:.4f}x)")
    print(f"weights only  : {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    backends = [("sdpa_flash", SDPABackend(kernel="flash"), batches),
                ("gla", GatedLinearAttention(gate_source="synthetic"), batches)]
    try:
        from attnbench.backends.block_sparse import BlockSparseAttention
        # batch=1 only -- see module docstring.
        backends.append(("block_sparse", BlockSparseAttention(), [1]))
    except ImportError as e:
        print(f"[skip] block_sparse not installed: {e}")

    results: list[BatchScalingResult] = []
    print(f"\n{'backend':<14} {'batch':>5} {'peak_GiB':>9} {'wall_s':>8} "
          f"{'s/example':>10} {'status':>8}")
    for name, backend, backend_batches in backends:
        # One warmup call per backend at its smallest batch, BEFORE any
        # measured cell. The first call pays CUDA context setup and, for
        # Triton backends like GLA, a full JIT compile: measured 2026-09-03,
        # two identical batch=1 GLA calls took 5.394 s then 1.384 s. Without
        # this the cost lands on whichever cell happens to run first, which
        # is always the smallest batch, which is exactly the numerator of the
        # batching speedup -- and it reported "4.28x, batching helps" for a
        # workload where batching does nothing.
        #
        # It is recorded (warmup=True) rather than thrown away so the JIT
        # cost stays visible in the data; `batching_speedup` filters it out.
        schedule = [(backend_batches[0], True)] + [(b, False) for b in backend_batches]
        for b, is_warmup in schedule:
            ids = ids1.repeat(b, 1)
            cfg = AttnConfig(seq_len=real, batch=b, n_heads_q=n_heads_q,
                             n_heads_kv=n_heads_kv, head_dim=head_dim,
                             dtype="bfloat16", mask="causal")
            wrapped = SwappableAttentionModel(model, cfg, model_id=model_id,
                                              finest_block_size=grid.finest_block_size)
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                wrapped.run_measured(ids, backend, cfg=cfg, logits_to_keep=1)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 2**20
                row = BatchScalingResult(
                    backend=name, model_id=model_id, seq_len_nominal=seq_len,
                    seq_len_real=real, batch=b, status="ok", peak_memory_mb=peak_mb,
                    wall_seconds=wall,
                    wall_seconds_per_example=per_example_seconds(wall, b),
                    warmup=is_warmup)
                print(f"{name:<14} {b:5d} {peak_mb / 1024:9.2f} {wall:8.3f} "
                      f"{row.wall_seconds_per_example:10.3f} "
                      f"{'warmup' if is_warmup else 'ok':>8}")
            except torch.cuda.OutOfMemoryError as e:
                row = BatchScalingResult(
                    backend=name, model_id=model_id, seq_len_nominal=seq_len,
                    seq_len_real=real, batch=b, status="oom", warmup=is_warmup,
                    detail=str(e).split("\n")[0][:200])
                print(f"{name:<14} {b:5d} {'--':>9} {'--':>8} {'--':>10} {'OOM':>8}")
                torch.cuda.empty_cache()
            except Exception as e:  # a real crash is a result too, not a stop
                row = BatchScalingResult(
                    backend=name, model_id=model_id, seq_len_nominal=seq_len,
                    seq_len_real=real, batch=b, status="error", warmup=is_warmup,
                    detail=f"{type(e).__name__}: {e}"[:200])
                print(f"{name:<14} {b:5d} {'--':>9} {'--':>8} {'--':>10} "
                      f"{type(e).__name__:>8}")
                torch.cuda.empty_cache()
            finally:
                wrapped.unwrap()
            results.append(row)

    speedups = batching_speedup(results)
    print("\nper-example speedup, smallest fitting batch -> largest fitting batch:")
    for backend, s in sorted(speedups.items()):
        verdict = "batching wins nothing" if s < 1.25 else "batching helps"
        print(f"  {backend:<14} {s:5.2f}x   {verdict}")

    prov = provenance.capture()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "batch_scaling.parquet"
    append_checkpoint(path, [{**r.to_dict(), **prov.to_dict()} for r in results])
    print(f"\nwrote {len(results)} rows -> {path}")


if __name__ == "__main__":
    main()
