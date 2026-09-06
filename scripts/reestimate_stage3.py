"""Stage 3 re-estimate from session 4's MEASURED throughput.

Replaces the last projection in the planning chain. Every prefill input below
is a measurement from 2026-09-03 session 4 (commit b6ed63b), not an
assumption.

Sizing is now token-exact (measured 0.9996-0.9999x of budget at 16384 and
32768), so a synthetic example whose token count equals the grid seq_len is
faithful -- which is what lets this run on CPU without the tokenizer.

REVISED 2026-09-06: a Stage 3 row is a prefill plus greedy decode, and the
decode term is NOT a small correction. See the DECODE section at the bottom.
The prefill tables above it are unchanged and still correct for what they
measure; they are simply not the whole row.
"""
import sys
from dataclasses import dataclass

sys.path.insert(0, "/Users/mehuldahiya/Desktop/research/attnbench_scaffold")

from attnbench.accuracy.config import load_grid
from attnbench.accuracy.timing_probe import (
    ACHIEVED_BANDWIDTH_FRACTION, DECODE_STEPS_BY_TASK,
    DECODE_STEPS_BY_TASK_AT_CAP, PEAK_BANDWIDTH_BYTES_PER_S, ModelArchitecture,
    decode_flops, decode_seconds, model_parameter_bytes, whole_model_flops)
from attnbench.config import AttnConfig

# Qwen2.5-1.5B-Instruct
ARCH = ModelArchitecture(n_layers=28, hidden_size=1536, intermediate_size=8960,
                         n_heads_q=12, n_heads_kv=2, head_dim=128,
                         vocab_size=151936)

# --- MEASURED throughput, session 4 -----------------------------------------
# Scoring is not monotonic in length: it peaks near 16K and degrades at 32K.
# Measured: 8192 -> 5.185 (session 3), 16384 -> 5.448, 32768 -> 3.927.
SCORING_TFLOPS = {2048: 5.185, 4096: 5.185, 8192: 5.185,
                  16384: 5.448, 32768: 3.927}

# Per-backend measured TFLOPS at 16384. block_sparse was not measured at
# 32768, so its 16384 figure is reused there -- conservative, since every
# other backend got FASTER at 32768 (sdpa_flash 42.2 -> 45.2, gla 48.0 ->
# 63.6), so reusing the shorter-length number understates throughput.
MEASURED_TFLOPS = {
    "sdpa_flash":         {16384: 42.151, 32768: 45.231},
    "block_sparse_0.5":   {16384: 38.122},
    "block_sparse_0.9":   {16384: 35.937},
    # INTERPOLATED, not measured -- 0.75 was cut from the grid before it was
    # ever run, so there is no anchor for it. Linear between the two measured
    # points (0.5 -> 38.122, 0.9 -> 35.937). The interpolation is over a
    # narrow range and the trend is monotone, but any variant below that
    # restores 0.75 rests partly on this number rather than on a measurement.
    "block_sparse_0.75":  {16384: 36.756},
    "gla":                {16384: 47.988, 32768: 63.595},
}


def tflops_for(backend: str, seq_len: int) -> float:
    table = MEASURED_TFLOPS[backend]
    if seq_len in table:
        return table[seq_len]
    return table[min(table, key=lambda k: abs(k - seq_len))]


def cfg_for(seq_len: int, mask: str, sparsity=None) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=ARCH.n_heads_q,
                      n_heads_kv=ARCH.n_heads_kv, head_dim=ARCH.head_dim,
                      dtype="bfloat16", mask=mask, block_size=128,
                      sparsity=sparsity)


def estimate(seq_lens: dict[int, int], sparsities: list[float],
             tasks: int) -> dict:
    """Hours by category for one grid shape."""
    scoring_h = 0.0
    measured_h = 0.0
    per_length = {}

    for seq_len, n in sorted(seq_lens.items()):
        n_examples = n * tasks
        # scoring: one dense pass per example, amortised across sparsities
        f_score = whole_model_flops(cfg_for(seq_len, "causal"), ARCH)
        s_h = n_examples * f_score / (SCORING_TFLOPS[seq_len] * 1e12) / 3600

        # measured: dense baseline + one pass per sparsity + linear arm
        m_h = 0.0
        f_dense = whole_model_flops(cfg_for(seq_len, "causal"), ARCH)
        m_h += n_examples * f_dense / (tflops_for("sdpa_flash", seq_len) * 1e12) / 3600
        m_h += n_examples * f_dense / (tflops_for("gla", seq_len) * 1e12) / 3600
        for sp in sparsities:
            key = f"block_sparse_{sp}"
            f_sp = whole_model_flops(cfg_for(seq_len, "block_sparse", sp), ARCH)
            m_h += n_examples * f_sp / (tflops_for(key, seq_len) * 1e12) / 3600

        scoring_h += s_h
        measured_h += m_h
        per_length[seq_len] = dict(n=n, scoring_h=s_h, measured_h=m_h,
                                   total_h=s_h + m_h)

    return dict(scoring_h=scoring_h, measured_h=measured_h,
                total_h=scoring_h + measured_h, per_length=per_length)


grid = load_grid("/Users/mehuldahiya/Desktop/research/attnbench_scaffold/"
                 "configs/accuracy/stage3_grid.yaml")
TASKS = len(grid.tasks)
COMMITTED = dict(grid.seq_lens)
SPARSITIES = list(grid.sparsities)

print(f"tasks={TASKS}  sparsities={SPARSITIES}")
print(f"committed grid: {COMMITTED}\n")

base = estimate(COMMITTED, SPARSITIES, TASKS)
print(f"{'seq_len':>8} {'n':>5} {'scoring_h':>10} {'measured_h':>11} {'total_h':>9}")
for sl, d in base["per_length"].items():
    print(f"{sl:8d} {d['n']:5d} {d['scoring_h']:10.2f} {d['measured_h']:11.2f} "
          f"{d['total_h']:9.2f}")
print(f"{'TOTAL':>8} {'':5} {base['scoring_h']:10.2f} {base['measured_h']:11.2f} "
      f"{base['total_h']:9.2f}\n")

# ---- variants --------------------------------------------------------------
#
# Labels are DERIVED from the diff against the loaded grid, not written by
# hand. The hand-written ones went stale the moment the two-sided rule
# restored cuts A and B in the YAML: this script printed
# "committed (16384:200, 32768:100)" directly above a table showing
# 16384: 300, and reported +0.00 deltas for restorations that were already
# in the baseline. A label that contradicts the data it sits next to is the
# same defect as a stale comment, except a reader trusts it more.
print("=" * 68)
print("VARIANTS (delta vs the grid as committed today)\n")


def describe(sls: dict, sps: list) -> str:
    bits = []
    for k in sorted(set(sls) | set(COMMITTED)):
        if sls.get(k) != COMMITTED.get(k):
            bits.append(f"{k}: {COMMITTED.get(k)} -> {sls.get(k)}")
    if sorted(sps) != sorted(SPARSITIES):
        bits.append(f"sparsities {SPARSITIES} -> {sorted(sps)}")
    return "; ".join(bits) if bits else "committed grid (no change)"


variants = [
    (COMMITTED, SPARSITIES),
    ({**COMMITTED, 16384: 200}, SPARSITIES),
    (COMMITTED, [0.5, 0.9]),
    ({**COMMITTED, 32768: 150}, SPARSITIES),
    ({**COMMITTED, 32768: 200}, SPARSITIES),
    ({**COMMITTED, 32768: 50}, SPARSITIES),
]
for sls, sps in variants:
    e = estimate(sls, sps, TASKS)
    print(f"  {describe(sls, sps):<44} {e['total_h']:6.2f} h  "
          f"({e['total_h'] - base['total_h']:+6.2f})")

# ---- what to actually book -------------------------------------------------
#
# The number above is pure compute. A session is not pure compute, and this
# project has a measured record of the difference: every rented session so far
# has spent time on instance creation, image boot, source deploy, the
# pre-flight test suite, Stage 0/1 gates, and teardown before and after any
# measured work. The A100 session on 2026-09-05 ran 165.9 minutes wall for
# roughly 100 minutes of measurement.
#
# 1.5 h of fixed overhead is the conservative read of that record for a run
# that boots from the prepared image and runs the gates once. It does NOT
# include a preemption, which is why Stage 3 goes on an on-demand L4 rather
# than Spot: at this duration a preemption costs more than the price
# difference saves several times over.
OVERHEAD_H = 1.5
print()
print("=" * 68)
print(f"BOOK: {base['total_h']:.2f} h compute + {OVERHEAD_H:.1f} h fixed "
      f"overhead = {base['total_h'] + OVERHEAD_H:.2f} h")
print("  On-demand L4 in asia-south1. Not Spot: at this duration one")
print("  preemption costs more than the discount saves.")
print("  Caveat carried forward: block_sparse throughput at sparsity 0.75 is")
print("  interpolated between the 0.5 and 0.9 measurements, never measured --")
print("  it was cut from the grid before it ever ran, and the two-sided rule")
print("  restored it. That is roughly 0.7 h of the total resting on an")
print("  interpolation, and it affects this estimate, not the run's validity.")


# ---------------------------------------------------------------------------
# DECODE
# ---------------------------------------------------------------------------
#
# Everything above prices one forward pass per (cell, example). A Stage 3 row
# is a prefill plus k greedy decode steps, because Stage 3 scores generated
# TEXT. That was missed once already, at ~20x, when the KV cache did not
# exist. With the cache built, docs/stage3_generation_decision.md priced the
# remaining tax at "under 1%" -- by taking the RATIO of decode FLOPs to
# prefill FLOPs, k/seq_len.
#
# That ratio is right and the conclusion drawn from it is wrong, because it
# silently assumes decode runs at the prefill's throughput. It does not. A
# batch-1 decode step reads all 3.09 GB of the model's bf16 weights to
# produce one token: it is bandwidth-bound, and its effective TFLOPS is
# ~177x below the prefill's. Same arithmetic, different regime.
#
# So the decode term is modelled from BANDWIDTH here, not from FLOPs.
print()
print("=" * 68)
print("DECODE -- a row is a prefill plus greedy decode\n")

PEAK_BW = PEAK_BANDWIDTH_BYTES_PER_S["L4"]
print(f"weights per step   : {model_parameter_bytes(ARCH) / 1e9:.2f} GB "
      f"(read in full, every step, to produce one token)")
print(f"L4 peak bandwidth  : {PEAK_BW / 1e9:.0f} GB/s "
      f"x {ACHIEVED_BANDWIDTH_FRACTION:.2f} achieved (ASSUMED, not measured)")
_step_cfg = cfg_for(8192, "causal")
_floor_ms = decode_seconds(_step_cfg, ARCH, n_steps=1,
                           peak_bandwidth_bytes_per_s=PEAK_BW) * 1000
_flops_ms = decode_flops(_step_cfg, ARCH, n_steps=1) / (42.151e12) * 1000
print(f"step at 8192       : {_floor_ms:.2f} ms bandwidth floor "
      f"vs {_flops_ms:.3f} ms if priced at the prefill's 42.2 TFLOPS "
      f"({_floor_ms / _flops_ms:.0f}x)")

# Two step tables (expected vs every example running to its cap) x two
# overhead assumptions (hardware floor vs a realistic Python/launch cost for
# an eager HF forward with a custom attention module per layer and no CUDA
# graphs). The answer is a bracket, and it is reported as one.
CONFIGS_PER_LENGTH = 1 + len(SPARSITIES) + 1     # dense + block_sparse + gla
PER_STEP_OVERHEAD_MS = 15.0

def decode_hours(steps_by_task: dict, overhead_ms: float) -> dict:
    out = {}
    for seq_len, n in sorted(COMMITTED.items()):
        seconds = 0.0
        for task, k in steps_by_task.items():
            seconds += n * CONFIGS_PER_LENGTH * decode_seconds(
                cfg_for(seq_len, "causal"), ARCH, n_steps=k,
                peak_bandwidth_bytes_per_s=PEAK_BW,
                per_step_overhead_s=overhead_ms / 1000)
        out[seq_len] = seconds / 3600
    return out

SCENARIOS = [
    ("expected stops, bandwidth floor", DECODE_STEPS_BY_TASK, 0.0),
    (f"expected stops, +{PER_STEP_OVERHEAD_MS:.0f}ms/step dispatch",
     DECODE_STEPS_BY_TASK, PER_STEP_OVERHEAD_MS),
    ("every example to its cap, floor", DECODE_STEPS_BY_TASK_AT_CAP, 0.0),
    (f"every example to its cap, +{PER_STEP_OVERHEAD_MS:.0f}ms/step",
     DECODE_STEPS_BY_TASK_AT_CAP, PER_STEP_OVERHEAD_MS),
]

S1_BANDS = (2048, 4096, 8192)
for label, steps, overhead in SCENARIOS:
    dec = decode_hours(steps, overhead)
    print(f"\n{label}")
    print(f"{'band':>7} {'prefill_h':>10} {'decode_h':>9} {'total_h':>8} {'tax':>8}")
    for seq_len in sorted(COMMITTED):
        pre = base["per_length"][seq_len]["total_h"]
        print(f"{seq_len:7d} {pre:10.2f} {dec[seq_len]:9.2f} "
              f"{pre + dec[seq_len]:8.2f} {dec[seq_len] / pre * 100:7.1f}%")
    total_dec = sum(dec.values())
    print(f"{'TOTAL':>7} {base['total_h']:10.2f} {total_dec:9.2f} "
          f"{base['total_h'] + total_dec:8.2f} "
          f"{total_dec / base['total_h'] * 100:7.1f}%")
    s1 = sum(base["per_length"][b]["total_h"] + dec[b] for b in S1_BANDS)
    print(f"        S1 (2048/4096/8192): {s1:.2f} h compute "
          f"-> {s1 + OVERHEAD_H:.2f} h wall")

print("""
The tax is largest where prefill is CHEAPEST, and that is structural, not a
quirk: the weight-read term does not depend on context, so decode costs
roughly the same per row at 2048 as at 32768 while prefill grows
quadratically. The 2048 band pays more for generating ~58 tokens per row than
for attending 2048 of them.

WHAT WOULD RESOLVE THIS: one measured decode step on the instance. The
bracket above spans 1.5-6.6 h entirely because the per-step dispatch
overhead of this implementation is unknown, and it takes about two minutes to
measure. scripts/time_one_accuracy_example.py does it before the grid
commits.""")
