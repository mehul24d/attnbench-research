"""The one kernel call site of every non-XAttention frontier arm
(estimator-frontier pre-registration sec. 4.4).

`bsa_prefill` takes q, k and v with the KV heads already repeated, and a
per-head block mask on q's device, and calls Block-Sparse-Attention's
`block_sparse_attn_func` (BSD-3). It is the per-head form of the call in
`backends/block_sparse.py`, which stays as it is for the banked era-3 path.

Gate G1b (CUDA, `tests/test_kernels.py`): on the same q, k, v and per-head
mask this and the XAttention backend's kernel invocation are bitwise equal.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

BLOCK_SIZE = 128        # fixed inside block_sparse_attn_func, not a choice


def _check(q, k, v, mask):
    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(f"bsa_prefill takes (1, H_q, S, D); got q {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("bsa_prefill takes k and v with the KV heads already repeated: "
                         f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    _, h, s, _ = q.shape
    nb = -(-s // BLOCK_SIZE)
    if mask.dtype != torch.bool or tuple(mask.shape) != (1, h, nb, nb):
        raise ValueError(f"mask must be bool (1, {h}, {nb}, {nb}); got {mask.dtype} "
                         f"{tuple(mask.shape)}")
    if mask.device != q.device:
        raise ValueError(f"the mask is on {mask.device} and q on {q.device}: masks never "
                         "leave the device (sec. 4.7)")
    if not mask.is_contiguous():
        raise ValueError("the mask must be contiguous")


def bsa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, *,
                kernel: Optional[Callable] = None) -> torch.Tensor:
    """Block-sparse causal prefill under a per-head mask. Returns
    (1, H_q, S, D).

    The arguments are those sec. 4.4 fixes: every head masked
    (`head_mask_type` all ones), no streaming, causal, deterministic, no
    exact streaming, no dropout. `kernel` replaces the import, for tests on
    a machine without the library."""
    _check(q, k, v, mask)
    if kernel is None:
        from block_sparse_attn import block_sparse_attn_func as kernel
    _, h, s, d = q.shape
    q_ = q.transpose(1, 2).reshape(s, h, d)
    k_ = k.transpose(1, 2).reshape(s, h, d)
    v_ = v.transpose(1, 2).reshape(s, h, d)
    cu_seqlens = torch.tensor([0, s], dtype=torch.int32, device=q.device)
    head_mask_type = torch.ones(h, dtype=torch.int32, device=q.device)
    out = kernel(q_, k_, v_, cu_seqlens, cu_seqlens, head_mask_type, None, mask,
                 s, s, p_dropout=0.0, deterministic=True, is_causal=True,
                 exact_streaming=False)
    return out.view(1, s, h, d).transpose(1, 2)


def reference_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """What `bsa_prefill` computes, in plain torch: causal attention in which
    query token t of head h sees key token u only if the mask keeps block
    (t // 128, u // 128) for that head. For checking the kernel and for CPU
    tests; never timed."""
    _check(q, k, v, mask)
    _, h, s, d = q.shape
    tok = torch.arange(s, device=q.device) // BLOCK_SIZE
    allowed = mask[0][:, tok][:, :, tok]                       # (H, S, S)
    allowed = allowed & torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
    logits = torch.matmul(q[0].float(), k[0].float().transpose(-2, -1)) / math.sqrt(d)
    weights = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
    return torch.matmul(weights, v[0].float()).unsqueeze(0).to(q.dtype)
