"""Analytic arithmetic intensity per kernel class for the learned baseline
(T2.10; researcher's decision, 25 Oct 2026: computed from the model config,
not measured).

Arithmetic intensity is FLOPs divided by the bytes a kernel must move
between device memory and the chip. Both are counted here from the model's
shape and the cell (context length L, batch B, density), never from a
profile:

  - FLOPs count multiply-adds as two, matmuls only plus the stated
    elementwise terms;
  - bytes are the compulsory traffic: every weight read once, every input
    activation read once and every output written once, at `itemsize` bytes
    per element. Tiling re-reads are not counted, so these intensities are
    upper bounds on what a kernel achieves.

**Kernel classes (proposed; the list awaits the researcher).** One forward
pass of prefill, T = B * L tokens, per layer unless stated:

  qkv_projection     X (T x d) times W_qkv (d x (h_q + 2 h_kv) hd)
  attention          causal Q K^T and P V over the kept blocks: FLOPs from
                     `config.AttnConfig.useful_flops` (4 B h_q L^2 hd,
                     halved for causal, times density), bytes Q, K, V read
                     and O written once
  output_projection  attention output (T x h_q hd) times W_o (h_q hd x d)
  mlp                gated MLP: gate and up (d x I each), down (I x d); each
                     of the three matmuls reads its input and writes its
                     output, plus 2 T I FLOPs for the gate's elementwise
                     product
  norm               one RMSNorm over T x d (4 FLOPs per element: square,
                     sum, scale by the reciprocal root, weight multiply);
                     reads T x d and the d weights, writes T x d
  lm_head            the last position of each sequence only, B x d times
                     W_out (d x V): prefill needs the next-token logits, not
                     all L of them

Not a class here: estimator scoring (XAttention's antidiagonal scoring,
mean-pool scoring). Its FLOPs and bytes depend on each estimator's own
parameters (stride, block size), so it needs a formula per estimator.
Embedding lookup, RoPE and residual adds are memory-bound gathers and
elementwise passes with negligible FLOPs; they are left out rather than
given a near-zero intensity that a tree model would read as a signal.

Per-layer classes have the same intensity for every layer, so the per-layer
count does not enter the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AttnConfig
from .recursion import CostModelError

KERNEL_CLASSES = ("qkv_projection", "attention", "output_projection", "mlp",
                  "norm", "lm_head")


@dataclass(frozen=True)
class ModelShape:
    """The shape fields arithmetic intensity needs. No identifiers."""

    hidden: int            # d
    n_heads_q: int         # h_q
    n_heads_kv: int        # h_kv
    head_dim: int          # hd
    intermediate: int      # I, the MLP width
    vocab: int             # V
    itemsize: int = 2      # bytes per element (bfloat16)

    def __post_init__(self):
        for name in ("hidden", "n_heads_q", "n_heads_kv", "head_dim",
                     "intermediate", "vocab", "itemsize"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise CostModelError(f"{name} must be an int >= 1, got {v!r}")
        if self.n_heads_q % self.n_heads_kv:
            raise CostModelError(
                f"n_heads_q={self.n_heads_q} is not a multiple of "
                f"n_heads_kv={self.n_heads_kv}")


def _check_cell(context_length: int, batch: int, density: float) -> None:
    for name, v in (("context_length", context_length), ("batch", batch)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise CostModelError(f"{name} must be an int >= 1, got {v!r}")
    if isinstance(density, bool) or not isinstance(density, (int, float)) \
            or not 0 < density <= 1:
        raise CostModelError(f"density must be in (0, 1], got {density!r}")


def flops_and_bytes(model: ModelShape, context_length: int, batch: int,
                    density: float) -> dict[str, tuple[int, int]]:
    """(FLOPs, bytes) per kernel class, for one layer (lm_head: once)."""
    _check_cell(context_length, batch, density)
    m, L, B = model, context_length, batch
    T = B * L
    d, hq, hkv, hd = m.hidden, m.n_heads_q, m.n_heads_kv, m.head_dim
    I, V, s = m.intermediate, m.vocab, m.itemsize
    qkv_out = (hq + 2 * hkv) * hd

    attn = AttnConfig(seq_len=L, batch=B, n_heads_q=hq, n_heads_kv=hkv,
                      head_dim=hd, mask="causal" if density == 1 else "block_sparse",
                      sparsity=None if density == 1 else 1.0 - density)
    return {
        "qkv_projection": (2 * T * d * qkv_out,
                           s * (d * qkv_out + T * d + T * qkv_out)),
        "attention": (attn.useful_flops(),
                      s * T * hd * (2 * hq + 2 * hkv)),
        "output_projection": (2 * T * hq * hd * d,
                              s * (hq * hd * d + T * hq * hd + T * d)),
        "mlp": (3 * 2 * T * d * I + 2 * T * I,
                s * (3 * d * I + 3 * T * d + 3 * T * I)),
        "norm": (4 * T * d, s * (2 * T * d + d)),
        "lm_head": (2 * B * d * V, s * (d * V + B * d + B * V)),
    }


def arithmetic_intensity(model: ModelShape, context_length: int, batch: int,
                         density: float) -> dict[str, float]:
    """FLOPs per byte for each class in `KERNEL_CLASSES`."""
    fb = flops_and_bytes(model, context_length, batch, density)
    return {c: fb[c][0] / fb[c][1] for c in KERNEL_CLASSES}
