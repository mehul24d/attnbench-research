"""mit-han-lab/Block-Sparse-Attention: FlashAttention-2-derived block-sparse
kernels (fwd + bwd), compiled from source.

Research fork, not a versioned release -- pinned to a specific commit in
pyproject.toml's `kernels` extra (git+https URL @ SHA) rather than a version
string, since the repo has no tagged releases and `main` can move under us.

Everything below is confirmed from the repo's own source, not assumed:
block_sparse_attn/block_sparse_attn_interface.py (commit 49d6c39) and
block_sparse_tests/fwd_bwd/test_performance/blocksparse.py in the same repo.

- block_size is hardcoded to 128x128 *inside* `block_sparse_attn_func`
  itself (`m_block_dim, n_block_dim` are passed as literal 128, 128, not a
  caller argument) -- a structural constraint of the kernel, not a
  configuration choice. Capability.block_sizes=(128,) reflects that; Stage 0
  rejects block_size=64 configs before ever calling forward, rather than
  silently running them at the wrong granularity.
- q/k/v are the unpadded/varlen layout: (total_tokens, n_heads, head_dim),
  not our canonical (batch, n_heads, seq_len, head_dim). For the uniform-
  length batches this study uses (AttnConfig.varlen is always False here),
  that's a reshape plus `cu_seqlens = arange(0, (batch+1)*seq_len, seq_len)`
  -- no actual padding waste, since every sequence in a cell is the same
  length.
- BlockSparseAttnFunc is a torch.autograd.Function with both forward() and
  backward() implemented -- backward support is an evidenced claim, not an
  assumption; Stage 0/1 confirm it actually runs on real hardware, same as
  every other claim here.
- head_mask_type=[1]*n_heads (every head uses the same block-sparse
  pattern) and base_blockmask is broadcast across batch and heads:
  (batch, n_heads, n_q_blocks, n_kv_blocks) -- see
  BlockSparseMask.to_block_sparse_attn_mask. This study never varies the
  sparsity pattern per batch element or per head, so the broadcast is
  exactly right, not a simplification.
- streaming_info=None is the confirmed value for a pure block-sparse call
  (no streaming-type heads), per the library's own performance test.
- No native GQA path: q and k/v must share the same head count at this
  interface. _expand_kv pays that cost inside forward(), same policy as
  every other backend here.

Unverified end-to-end: not yet run against an installed build. Requires
CUDA 11.6+ and sm80+ -- cannot be validated on Colab free tier (T4, sm75),
only on the rented 4090/A100.
"""

from __future__ import annotations

import torch

from .. import bsa_contract
from ..config import AttnConfig
from ..masks import BlockSparseMask
from .base import AttentionBackend, Capability, UnsupportedConfig, register
from .impls import _expand_kv


def kernel_interface(*args, unsupported: bool = False, **kwargs):
    """The library's `block_sparse_attn_func`, called outside Dynamo when
    compiled. Every arm that runs the block-sparse kernel calls it through
    here: the oracle and mean-pool (`BlockSparseAttention`) and XAttention
    (researcher's ruling, 1 Nov 2026, after Shakedown 6: one exclusion of the
    kernel interface for every arm, held by the recompile gate). Traced, the
    interface's own break (`replace_ones_with_count` sizes a tensor from its
    data) left a resume frame whose guard held a tensor that died after every
    call, so the frame recompiled per layer: XAttention in Shakedown 5, and
    mean-pool under the default settings in Shakedown 6, reached the
    recompile limit (256). Under Dynamo the call goes through
    `torch.compiler.disable`, a break at this function in every arm and
    setting (`compiled.DECLARED_BREAKS`); eagerly it is the library's own
    call, after one `is_compiling` check (about 0.1 us).

    `unsupported`: a `RuntimeError` from the kernel is an `UnsupportedConfig`
    (the oracle and mean-pool backend's rule). The `try` is inside the
    excluded call: a break inside a `try` in a traced frame is a break of its
    own (`step_unsupported`, the L4 report of Shakedown 2)."""
    if torch.compiler.is_compiling():
        return torch.compiler.disable(_call)(*args, unsupported=unsupported, **kwargs)
    return _call(*args, unsupported=unsupported, **kwargs)


def _call(*args, unsupported: bool, **kwargs):
    from block_sparse_attn import block_sparse_attn_func
    # The library's mask buffer with a -1 row after it: its kernel reads one
    # entry past a fully selected row (`bsa_contract`; 2 Nov 2026).
    bsa_contract.install()
    if not unsupported:
        return block_sparse_attn_func(*args, **kwargs)
    try:
        return block_sparse_attn_func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as e:
        raise UnsupportedConfig(f"block_sparse: {e}") from e


@register
class BlockSparseAttention(AttentionBackend):
    capability = Capability(
        name="block_sparse",
        family="sparse",
        min_compute_capability=(8, 0),
        supports_backward=True,
        supports_gqa=True,          # via explicit KV expansion, not a native path -- see forward()
        supports_causal=True,
        supports_block_sparse=True,
        block_sizes=(128,),         # hardcoded inside block_sparse_attn_func itself, not a choice
        head_dims=(32, 64, 128),
        dtypes=("bfloat16", "float16"),
        notes=("pip install from a pinned commit of mit-han-lab/"
               "Block-Sparse-Attention (research fork, no releases; needs "
               "ninja + matching gcc, compiles from source). sm80+ only -- "
               "untestable on Colab free tier."),
    )

    @staticmethod
    def _import_check():
        import block_sparse_attn  # noqa: F401

    def forward(self, q, k, v, cfg: AttnConfig, mask: BlockSparseMask = None):
        if cfg.mask != "block_sparse":
            raise UnsupportedConfig(f"block_sparse: mask kind {cfg.mask} not wired")
        if mask is None:
            raise UnsupportedConfig("block_sparse: requires a BlockSparseMask")
        if mask.seq_len != cfg.seq_len or mask.block_size != cfg.block_size:
            raise UnsupportedConfig(
                f"mask is ({mask.seq_len}, block {mask.block_size}) but cfg "
                f"is ({cfg.seq_len}, block {cfg.block_size}) -- a mismatch "
                f"would measure a different sparsity pattern")

        # Absent library: refused here, before any setup, as before the
        # kernel call moved to `kernel_interface` (the timed-region test
        # reads a backend that cannot run as not exercised).
        import block_sparse_attn  # noqa: F401

        k_, v_ = _expand_kv(k, v, cfg)   # kernel wants n_heads_k == n_heads_q here

        b, h, s, d = q.shape
        # (B, H, S, D) -> (B*S, H, D): the unpadded layout this kernel
        # expects. Valid because every sequence in a cell is the same
        # length (AttnConfig.varlen is always False in this study) -- no
        # padding to lose or account for.
        q_ = q.transpose(1, 2).reshape(b * s, h, d)
        k_ = k_.transpose(1, 2).reshape(b * s, h, d)
        v_ = v_.transpose(1, 2).reshape(b * s, h, d)
        cu_seqlens = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=q.device)

        head_mask_type = torch.ones(h, dtype=torch.int32, device=q.device)
        base_blockmask = mask.to_block_sparse_attn_mask(b, h, device=q.device)

        out = kernel_interface(
            q_, k_, v_, cu_seqlens, cu_seqlens,
            head_mask_type, None, base_blockmask,
            s, s, 0.0,
            is_causal=mask.causal,
            exact_streaming=False,
            unsupported=True,
        )

        return out.reshape(b, s, h, d).transpose(1, 2)   # back to (B, H, S, D)
