"""Exact block-sparse attention at ANY block size, for accuracy only.

Built 2026-10-01 for the block-size axis. Its three pre-registered
predictions (sink underranking, fp16 tie density, pooling dilution) are about
WHICH blocks get selected, so measuring them needs outputs that are exactly
what a mask at that block size computes, not a fast kernel.
Block-Sparse-Attention hardcodes 128, and Flex below 64 needs a newer torch
than the image's plus 16-wide tiles that would make any latency comparison a
measurement of tile efficiency. This backend sidesteps both:
`scaled_dot_product_attention` with the block mask expanded to a token-level
boolean mask (`BlockSparseMask.to_dense_bool`, causal triangle included).

**No latency from this backend is a block-sparse latency.** It computes every
score and discards the masked ones. Rows it produces carry its name, and the
block-size accuracy rows are compared with each other and with the 128-block
BSA rows only through accuracy (the 128 run on both paths is the bridge).

Memory: the token mask is processed in query chunks (`chunk_rows`), so peak
mask memory is chunk_rows x seq_len, not seq_len^2 -- 2048 x 32768 bools is
64 MiB, where the whole mask would be 1 GiB.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import AttnConfig
from ..masks import BlockSparseMask
from .base import AttentionBackend, Capability, UnsupportedConfig, register


@register
class BlockMaskedSDPA(AttentionBackend):
    capability = Capability(
        name="block_masked_sdpa",
        family="reference",
        min_compute_capability=(0, 0),
        supports_backward=False,
        supports_gqa=True,
        supports_causal=True,
        supports_block_sparse=True,
        block_sizes=(16, 32, 64, 128, 256),
        head_dims=(8, 16, 32, 64, 128),
        dtypes=("bfloat16", "float16", "float32"),
        notes=("Accuracy-only: SDPA over the expanded block mask, chunked by "
               "query rows. Exact at any block size; its latency is NOT a "
               "block-sparse latency and must never be reported as one."),
    )

    def __init__(self, chunk_rows: int = 2048):
        self.chunk_rows = int(chunk_rows)
        # Token index vector per (seq_len, device), reused across calls and
        # chunks for both the column index and the causal row index. Without
        # it every call rebuilt two seq_len-sized aranges, which
        # tests/test_timed_region_setup.py flags; this backend is never timed
        # as block-sparse, but the guard's allowlist stays empty.
        self._idx: dict = {}

    def _index(self, s: int, device) -> torch.Tensor:
        key = (s, str(device))
        if key not in self._idx:
            self._idx[key] = torch.arange(s, device=device)
        return self._idx[key]

    def forward(self, q, k, v, cfg: AttnConfig, mask: BlockSparseMask = None):
        if cfg.mask == "causal" and mask is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  enable_gqa=cfg.is_gqa)
        if cfg.mask != "block_sparse":
            raise UnsupportedConfig(f"block_masked_sdpa: mask kind {cfg.mask!r}")
        if mask is None:
            raise UnsupportedConfig("block_masked_sdpa: block_sparse needs a BlockSparseMask")
        if mask.seq_len != cfg.seq_len or mask.block_size != cfg.block_size:
            raise UnsupportedConfig(
                f"mask is ({mask.seq_len}, block {mask.block_size}) but cfg is "
                f"({cfg.seq_len}, block {cfg.block_size})")
        s, bs = q.shape[2], mask.block_size
        active = mask.active.to(q.device)
        cols = self._index(s, q.device)
        outs = []
        for r0 in range(0, s, self.chunk_rows):
            r1 = min(r0 + self.chunk_rows, s)
            outs.append(F.scaled_dot_product_attention(
                q[:, :, r0:r1], k, v, attn_mask=self._chunk_mask(active, bs, r0, r1, s, cols, mask.causal),
                enable_gqa=cfg.is_gqa))
        return torch.cat(outs, dim=2)

    @staticmethod
    def _chunk_mask(active, bs, r0, r1, s, cols, causal):
        """Rows r0:r1 of `BlockSparseMask.to_dense_bool()`, built without the
        rest: only this chunk's block rows are expanded. Equality with the
        full expansion is a test."""
        qb0, qb1 = r0 // bs, (r1 - 1) // bs + 1
        rows = active[qb0:qb1].repeat_interleave(bs, dim=0)[r0 - qb0 * bs:r1 - qb0 * bs]
        dense = rows.repeat_interleave(bs, dim=1)[:, :s]
        if causal:
            dense = dense & (cols[r0:r1].unsqueeze(1) >= cols)
        return dense
