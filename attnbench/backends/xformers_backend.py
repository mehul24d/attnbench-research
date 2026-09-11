"""xFormers memory_efficient_attention.

Named xformers_backend.py, not xformers.py -- Python 3's absolute imports
mean `import xformers.ops` inside this file would resolve correctly to the
real package either way, but matching the package's own name invites
confusion for anyone reading the module list or running `python -m` against
this package by hand.

xFormers wants (B, S, H, D); ours is (B, H, S, D), so we transpose in and
out -- same pattern as FlashAttention2. GQA goes through the same
`_expand_kv` repeat_interleave the naive reference (and Stage 1's
GQA-convention test) already verify, rather than xFormers' own broadcast
reshape path, whose exact semantics vary across versions and aren't
confirmed against anything in this repo.

Unverified against an installed package: not yet run, per this project's
standing rule that nothing here is trusted until Stage 0/1 has actually
executed it (`pip install xformers` first).
"""

from __future__ import annotations

import torch

from ..config import AttnConfig
from .base import AttentionBackend, Capability, UnsupportedConfig, register
from .impls import _expand_kv


@register
class XFormersAttention(AttentionBackend):
    """xformers.ops.memory_efficient_attention, cutlass-backed kernels."""

    capability = Capability(
        name="xformers",
        family="dense_exact",
        # xFormers' cutlass-based kernels historically support sm70+ (Volta),
        # wider than FA2's sm80 floor -- claiming that broadly and letting
        # Stage 0 determine actual support empirically, per the claim-vs-
        # actual design this harness is built around.
        min_compute_capability=(7, 0),
        supports_gqa=True,
        supports_sliding=False,
        supports_block_sparse=False,
        head_dims=(64, 128),
        dtypes=("bfloat16", "float16"),
        notes="pip install xformers. GQA via explicit KV expansion, not xFormers' native broadcast.",
    )

    @staticmethod
    def _import_check():
        import xformers.ops  # noqa: F401

    def forward(self, q, k, v, cfg, mask=None):
        import xformers.ops as xops

        if cfg.mask not in ("causal", "full"):
            raise UnsupportedConfig(f"xformers: mask {cfg.mask} not wired")

        k_, v_ = _expand_kv(k, v, cfg)
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k_, v_))  # (B,H,S,D) -> (B,S,H,D)

        attn_bias = xops.LowerTriangularMask() if cfg.mask == "causal" else None
        try:
            out = xops.memory_efficient_attention(q_, k_, v_, attn_bias=attn_bias)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as e:
            raise UnsupportedConfig(f"xformers: {e}") from e
        return out.transpose(1, 2)
