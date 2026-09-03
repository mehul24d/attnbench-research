"""Stage 3 re-estimate from session 4's MEASURED throughput.

Replaces the last projection in the planning chain. Every input below is a
measurement from 2026-09-03 session 4 (commit b6ed63b), not an assumption.

Sizing is now token-exact (measured 0.9996-0.9999x of budget at 16384 and
32768), so a synthetic example whose token count equals the grid seq_len is
faithful -- which is what lets this run on CPU without the tokenizer.
"""
import sys
from dataclasses import dataclass

sys.path.insert(0, "/Users/mehuldahiya/Desktop/research/attnbench_scaffold")

from attnbench.accuracy.config import load_grid
from attnbench.accuracy.timing_probe import ModelArchitecture, whole_model_flops
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

# ---- the two-sided rule ----------------------------------------------------
print("=" * 68)
print("VARIANTS (delta vs committed)\n")
variants = {
    "committed (16384:200, 32768:100)": (COMMITTED, SPARSITIES),
    "B restored: 16384 -> 300": ({**COMMITTED, 16384: 300}, SPARSITIES),
    "B restored + 32768 -> 150": ({**COMMITTED, 16384: 300, 32768: 150}, SPARSITIES),
    "A restored: sparsity 0.75 back": (COMMITTED, [0.5, 0.75, 0.9]),
    "A+B restored (full original)": ({**COMMITTED, 16384: 300}, [0.5, 0.75, 0.9]),
    "A+B + 32768 -> 200": ({**COMMITTED, 16384: 300, 32768: 200}, [0.5, 0.75, 0.9]),
}
for label, (sls, sps) in variants.items():
    e = estimate(sls, sps, TASKS)
    print(f"  {label:<36} {e['total_h']:6.2f} h  "
          f"({e['total_h'] - base['total_h']:+6.2f})")
