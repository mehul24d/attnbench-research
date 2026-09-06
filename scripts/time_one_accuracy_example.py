#!/usr/bin/env python
"""Stage 3 pre-flight: time one real example at the grid's longest length,
across every backend the full grid actually runs, before committing a
rented GPU session to the whole sweep.

    python scripts/time_one_accuracy_example.py
    python scripts/time_one_accuracy_example.py --include-sage --task vt

Needs a GPU and the real model downloaded -- this is the first thing to run
in the rented session, per the Stage 3 plan, specifically *because*
scripts/run_accuracy.py's per-length cost estimates were a FLOPs-based
projection assuming 15 TFLOPS effective throughput, not a measurement.
This script replaces that assumption with a real one: it runs the one
example through the scoring pass (shared/amortized across every
block_sparse sparsity level, exactly as accuracy/model.py's cache means it
would be in the real sweep) and then through each backend's measured pass,
timing every phase, and reports:

  1. per-phase wall time, effective TFLOPS, and peak CUDA memory
  2. the corrected total-grid-hours estimate (attnbench.accuracy.timing_probe),
     using these measured TFLOPS figures in place of the planning-time
     assumption -- alongside the original 15-TFLOPS-assumed number, so the
     two can be compared directly
  3. peak memory at this length, as a sanity check against the 24GB/80GB
     hardware ceiling in CLAUDE.md, before the full grid is committed

Only the longest configured length is probed (the worst case for both
memory and the causal-quadratic scoring pass) -- if that fits and the
corrected estimate still clears the rented-hour budget, the full grid is
safe to commit.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from attnbench.accuracy.config import load_grid  # noqa: E402
from attnbench.accuracy.grid_configs import (  # noqa: E402
    SAGE_QUANT_SCHEME, build_configs_by_backend, build_examples_by_task_length)
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.accuracy.timing_probe import (  # noqa: E402
    DECODE_STEPS_BY_TASK, DECODE_STEPS_BY_TASK_AT_CAP,
    PEAK_BANDWIDTH_BYTES_PER_S, ModelArchitecture, PhaseTiming,
    blended_tflops_by_category, corrected_grid_hours, decode_seconds,
    total_grid_flops_by_category, whole_model_flops)
from attnbench.backends.block_sparse import BlockSparseAttention  # noqa: E402
from attnbench.backends.impls import SDPABackend  # noqa: E402
from attnbench.backends.linear import GatedLinearAttention  # noqa: E402
from attnbench.backends.sage_attention import SageAttention  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

ASSUMED_TFLOPS = 15.0  # the planning-time guess this script exists to replace


def _timed(fn, *, warmup: int = 0):
    """Run fn(), returning (result, wall_seconds, peak_memory_bytes).
    Synchronizes CUDA on both sides of the timer so async kernel launches
    don't make the measurement look faster than the work actually was.

    `warmup` discards that many runs first. It defaults to 0 and must be
    passed explicitly, because it is NOT safe everywhere in this script: the
    scoring pass writes to `score_cache`, so a warmup run would leave the
    timed run measuring a cache load instead of the dense-softmax computation
    it is supposed to be measuring -- turning an expensive pass into a
    spuriously cheap one, in the direction that flatters the study. Warm up
    the measured backend passes, which are pure recomputation; leave scoring
    cold and note it as a lower bound.

    Measured 2026-09-03: a cold GLA call took 5.493 s against 1.384 s warm --
    4x, entirely Triton JIT. Without this the anchor reported GLA at 12.018
    effective TFLOPS when the real figure is ~47.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    return result, wall_seconds, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--task", default=None,
                    help="defaults to the first task in the grid")
    ap.add_argument("--model", choices=("primary", "alternate"), default="primary")
    ap.add_argument("--include-sage", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense-backend", default="sdpa_math")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="defaults to the grid's longest configured length "
                         "(the worst case); pass one of the grid's other "
                         "seq_lens to probe a different point")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a GPU -- see CLAUDE.md's hardware "
                          "ceiling for what this study targets")

    grid = load_grid(args.grid)
    task = args.task or grid.tasks[0]
    if args.seq_len is not None:
        if args.seq_len not in grid.seq_lens:
            raise SystemExit(f"--seq-len {args.seq_len} is not in the grid's "
                             f"seq_lens: {sorted(grid.seq_lens)}")
        seq_len = args.seq_len
    else:
        seq_len = max(grid.seq_lens)   # the longest configured length -- worst case
    model_id = grid.model_primary if args.model == "primary" else grid.model_alternate

    print(f"grid          : {args.grid}")
    print(f"model         : {model_id}")
    print(f"task          : {task}")
    print(f"probed seq_len: {seq_len}"
          f"{' (directional/underpowered point)' if grid.is_directional(seq_len) else ''}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the tokenizer BEFORE generating: contexts are now sized against
    # it to hit an exact token budget, rather than estimated by word count
    # and corrected afterward. See attnbench/accuracy/sizing.py.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text).input_ids)

    examples = build_examples_by_task_length(
        grid, seed=args.seed, tasks=(task,), seq_lens={seq_len: 1},
        count_tokens=count_tokens)[(task, seq_len)]
    example = examples[0]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    input_ids = tokenizer(example.context, return_tensors="pt").input_ids.to("cuda")
    actual_seq_len = input_ids.shape[-1]
    print(f"tokenized length: {actual_seq_len} tokens "
          f"(budget {example.token_budget}, sized to {example.context_length} "
          f"at generation, {example.haystack_units} filler units)")

    n_heads_q = model.config.num_attention_heads
    n_heads_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // n_heads_q)
    cfg_template = AttnConfig(seq_len=actual_seq_len, batch=1, n_heads_q=n_heads_q,
                              n_heads_kv=n_heads_kv, head_dim=head_dim,
                              dtype="bfloat16", mask="causal")
    # The real model's geometry -- needed so effective-TFLOPS is computed
    # against whole-forward-pass FLOPs (every layer's attention AND
    # projections AND MLP, plus one lm_head call), matching what the
    # wall-clock below actually times. See timing_probe.whole_model_flops.
    arch = ModelArchitecture(
        n_layers=model.config.num_hidden_layers, hidden_size=model.config.hidden_size,
        intermediate_size=model.config.intermediate_size, n_heads_q=n_heads_q,
        n_heads_kv=n_heads_kv, head_dim=head_dim, vocab_size=model.config.vocab_size)

    wrapped = SwappableAttentionModel(model, cfg_template, model_id=model_id,
                                      finest_block_size=grid.finest_block_size)

    phases: list[PhaseTiming] = []
    try:
        layer_scores, wall, peak = _timed(lambda: wrapped.compute_importance_scores(
            input_ids, task=task, example_id=example.example_id,
            cache_dir=grid.score_cache_dir))
        scoring_cfg = replace(cfg_template, mask="causal")
        phases.append(PhaseTiming(label="scoring_pass", category="scoring",
                                  flops=whole_model_flops(scoring_cfg, arch, logits_to_keep=1),
                                  wall_seconds=wall, peak_memory_bytes=peak))

        dense_cfg = replace(cfg_template, mask="causal")
        dense_kernel = args.dense_backend.removeprefix("sdpa_")
        # logits_to_keep=1: this probe times generation-style inference
        # (RULER only ever needs the final position's next-token logits),
        # not the full-sequence correctness check run_measured's default
        # serves in tests -- see SwappableAttentionModel.run_measured's
        # docstring. Confirmed necessary: full-sequence logits at 32K
        # context / ~150K vocab OOM'd (11+ GiB) independent of backend.
        _, wall, peak = _timed(warmup=1, fn=lambda: wrapped.run_measured(
            input_ids, SDPABackend(kernel=dense_kernel), cfg=dense_cfg,
            logits_to_keep=1))
        phases.append(PhaseTiming(label=f"measured_{args.dense_backend}",
                                  category="measured",
                                  flops=whole_model_flops(dense_cfg, arch, logits_to_keep=1),
                                  wall_seconds=wall, peak_memory_bytes=peak))

        # Each optional backend below is tried independently -- a missing
        # package (not installed this session) must not lose the phases
        # already measured above it. Only ImportError/ModuleNotFoundError is
        # swallowed; a real crash inside an installed backend still surfaces.
        try:
            block_sparse_backend = BlockSparseAttention()
            for sparsity in grid.sparsities:
                sparse_cfg = replace(cfg_template, mask="block_sparse", sparsity=sparsity,
                                     block_size=grid.finest_block_size,
                                     mask_source="importance")
                _, wall, peak = _timed(warmup=1, fn=lambda: wrapped.run_measured(
                    input_ids, block_sparse_backend, cfg=sparse_cfg,
                    layer_scores=layer_scores, logits_to_keep=1))
                phases.append(PhaseTiming(label=f"measured_block_sparse_{sparsity}",
                                          category="measured",
                                          flops=whole_model_flops(sparse_cfg, arch, logits_to_keep=1),
                                          wall_seconds=wall, peak_memory_bytes=peak))
        except ImportError as e:
            print(f"[skip] block_sparse not installed this session: {e}")

        try:
            gla_cfg = replace(cfg_template, mask="causal")
            _, wall, peak = _timed(warmup=1, fn=lambda: wrapped.run_measured(
                input_ids, GatedLinearAttention(), cfg=gla_cfg, logits_to_keep=1))
            phases.append(PhaseTiming(label="measured_gla", category="measured",
                                      flops=whole_model_flops(gla_cfg, arch, logits_to_keep=1),
                                      wall_seconds=wall, peak_memory_bytes=peak))
        except ImportError as e:
            print(f"[skip] gla not installed this session: {e}")

        if args.include_sage:
            try:
                sage_cfg = replace(cfg_template, mask="causal",
                                   quant_scheme=SAGE_QUANT_SCHEME)
                _, wall, peak = _timed(warmup=1, fn=lambda: wrapped.run_measured(
                    input_ids, SageAttention(), cfg=sage_cfg, logits_to_keep=1))
                phases.append(PhaseTiming(label="measured_sage", category="measured",
                                          flops=whole_model_flops(sage_cfg, arch, logits_to_keep=1),
                                          wall_seconds=wall, peak_memory_bytes=peak))
            except ImportError as e:
                print(f"[skip] sage not installed this session: {e}")
    finally:
        wrapped.unwrap()

    # --- the decode step, measured -----------------------------------------
    #
    # This is the one number that cannot be projected. A Stage 3 row is a
    # prefill plus greedy decode, and decode is BANDWIDTH-bound at batch 1
    # while prefill is compute-bound -- so the measured prefill TFLOPS above
    # says nothing about it. The planning estimate brackets 1.5-6.6 h for the
    # whole grid's decode purely because this implementation's per-step
    # dispatch overhead is unknown. Two minutes here closes that.
    decode_ms = None
    try:
        dense = SDPABackend(kernel=args.dense_backend.removeprefix("sdpa_"))
        wrapped_for_decode = SwappableAttentionModel(
            model, cfg_template, model_id=model_id,
            finest_block_size=grid.finest_block_size)
        gen_cfg = replace(cfg_template, mask="causal")
        n_steps = 8
        # Warm first: the first decode step pays one-time allocation and any
        # JIT, and charging that to a per-step average would inflate the whole
        # grid's decode estimate. Same reason _timed takes a warmup.
        wrapped_for_decode.generate(input_ids, dense, cfg=gen_cfg,
                                    max_new_tokens=2, decode_backend=dense)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = wrapped_for_decode.generate(input_ids, dense, cfg=gen_cfg,
                                             max_new_tokens=n_steps,
                                             decode_backend=dense)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        # The prefill is inside that timer; subtract the measured dense
        # prefill so what is left is decode.
        prefill_s = next(p.wall_seconds for p in phases
                         if p.label == f"measured_{args.dense_backend}")
        decode_ms = max(wall - prefill_s, 0.0) / max(result.n_generated, 1) * 1000
    except Exception as e:                       # noqa: BLE001
        print(f"[skip] decode step not measured: {type(e).__name__}: {e}")
    finally:
        try:
            wrapped_for_decode.unwrap()
        except Exception:                        # noqa: BLE001
            pass

    print("\nphase                          wall_s   eff_TFLOPS   peak_mem_GiB")
    for p in phases:
        mem_gib = (p.peak_memory_bytes or 0) / (1024 ** 3)
        print(f"{p.label:<30} {p.wall_seconds:8.3f} {p.effective_tflops:12.3f} "
              f"{mem_gib:14.2f}")

    measured_tflops = blended_tflops_by_category(phases)
    print("\nblended effective TFLOPS by category (vs. the planning-time "
          f"assumption of {ASSUMED_TFLOPS}):")
    for category, tflops in sorted(measured_tflops.items()):
        print(f"  {category:<10}: {tflops:.3f} TFLOPS")

    configs_by_backend = build_configs_by_backend(grid, include_sage=args.include_sage,
                                                  dense_backend=args.dense_backend)
    examples_by_task_length = build_examples_by_task_length(
        grid, seed=args.seed, count_tokens=count_tokens)
    # Grid seq_lens are real token counts now (sizing.py fits each context
    # to its budget), so no inflation correction is applied. Report how
    # closely the probed example actually landed, as a check on that claim.
    print(f"\nsizing accuracy at {seq_len}: {actual_seq_len} tokens "
          f"({actual_seq_len / seq_len:.4f}x budget)")
    # A row is a prefill plus its greedy decode steps, not one forward.
    # Both step tables are priced: the expected one assumes the model emits
    # a newline where the measured answer lengths say it will, the cap one
    # assumes it never does. The gap between them is the estimate's exposure
    # to that assumption, and it belongs on the probe's output rather than
    # in a doc nobody re-reads at 2am in a rented session.
    total_flops = total_grid_flops_by_category(
        configs_by_backend, examples_by_task_length, arch=arch,
        decode_steps_by_task=DECODE_STEPS_BY_TASK,
        dense_backend=args.dense_backend)
    at_cap_flops = total_grid_flops_by_category(
        configs_by_backend, examples_by_task_length, arch=arch,
        decode_steps_by_task=DECODE_STEPS_BY_TASK_AT_CAP,
        dense_backend=args.dense_backend)
    prefill_only_flops = total_grid_flops_by_category(
        configs_by_backend, examples_by_task_length, arch=arch,
        decode_steps_by_task={}, dense_backend=args.dense_backend)

    # Decode is dropped from every FLOPs-costed total on purpose: it is
    # bandwidth-bound and corrected_grid_hours now refuses to price it. It is
    # priced below, from the MEASURED per-step time.
    def _prefill_only(t):
        return {k: v for k, v in t.items() if k != "decode"}

    measured_hours = corrected_grid_hours(_prefill_only(total_flops), measured_tflops)
    assumed_hours = corrected_grid_hours(
        _prefill_only(total_flops), {cat: ASSUMED_TFLOPS for cat in total_flops})

    print(f"\n{'category':<10} {'assumed_15TFLOPS_h':>20} {'measured_h':>12}")
    for category in sorted(total_flops):
        print(f"{category:<10} {assumed_hours[category]:20.2f} "
              f"{measured_hours[category]:12.2f}")
    print(f"{'total':<10} {assumed_hours['total']:20.2f} {measured_hours['total']:12.2f}")

    floor_ms = decode_seconds(
        replace(cfg_template, mask="causal"), arch, n_steps=1,
        peak_bandwidth_bytes_per_s=PEAK_BANDWIDTH_BYTES_PER_S["L4"]) * 1000
    if decode_ms is not None:
        print(f"\ndecode step    : {decode_ms:.2f} ms measured vs "
              f"{floor_ms:.2f} ms bandwidth floor "
              f"({decode_ms / floor_ms:.2f}x) at seq_len {actual_seq_len}")
        overhead_ms = max(decode_ms - floor_ms, 0.0)
        total_steps = sum(
            len(examples_by_task_length.get((task, sl), [])) * k
            * (1 + len(grid.sparsities) + 1)
            for task, k in DECODE_STEPS_BY_TASK.items()
            for sl in grid.seq_lens)
        print(f"                 per-step dispatch overhead {overhead_ms:.2f} ms; "
              f"grid decode ~= {total_steps * decode_ms / 1000 / 3600:.2f} h "
              f"over {total_steps} steps")
    else:
        print(f"\ndecode step    : NOT MEASURED. Bandwidth floor is "
              f"{floor_ms:.2f} ms/step; the grid's decode term is a bracket "
              f"until this is measured -- see scripts/reestimate_stage3.py.")

    print(f"\nunit of work   : 1 prefill + greedy decode "
          f"({DECODE_STEPS_BY_TASK} steps by task)")
    print(f"  prefill      : {measured_hours['total']:.2f} h "
          f"(FLOPs / measured TFLOPS -- compute-bound)")
    if decode_ms is not None:
        n_cfg = 1 + len(grid.sparsities) + 1
        expected_steps = sum(
            len(examples_by_task_length.get((task, sl), [])) * k * n_cfg
            for task, k in DECODE_STEPS_BY_TASK.items() for sl in grid.seq_lens)
        cap_steps = sum(
            len(examples_by_task_length.get((task, sl), [])) * k * n_cfg
            for task, k in DECODE_STEPS_BY_TASK_AT_CAP.items() for sl in grid.seq_lens)
        dec_h = expected_steps * decode_ms / 1000 / 3600
        cap_h = cap_steps * decode_ms / 1000 / 3600
        print(f"  + decode     : {dec_h:.2f} h at the MEASURED "
              f"{decode_ms:.2f} ms/step ({expected_steps} steps) "
              f"= {dec_h / measured_hours['total'] * 100:.1f}% of prefill")
        print(f"  GRID TOTAL   : {measured_hours['total'] + dec_h:.2f} h")
        print(f"  if no example ever stops early: "
              f"{measured_hours['total'] + cap_h:.2f} h")
    else:
        print("  + decode     : NOT MEASURED -- the grid total is unknown, not "
              "the prefill figure above")


if __name__ == "__main__":
    main()
