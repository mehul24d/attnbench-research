"""The XAttention backend (audit C1): official estimator, faithful plumbing.

On the workstation the official package and the kernel are absent, so the
CPU tests install stand-ins for `xattn.src.Xattention` and
`block_sparse_attn` that RECORD what the backend passes them. What is pinned:
the estimator is called with exactly the arguments `Xattention_prefill`
passes (pinned commit), KV heads are repeated as `repeat_kv` does, the kernel
receives the mask sliced as `Xattention_prefill` slices it, and realised
density counts only causally valid blocks.

The CUDA test is the one that matters for the paper: on the instance, with
the real package, this backend's output must be bitwise identical to
`Xattention_prefill`'s. It runs in the Phase A gate.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.xattention import (XATTN_COMMIT, XAttentionBackend,
                                           official_chunk_size, realised_density)
from attnbench.config import AttnConfig

S, HQ, HKV, D = 300, 4, 2, 16          # 300 tokens -> 3 blocks of 128


def _cfg(**kw):
    base = dict(seq_len=S, batch=1, n_heads_q=HQ, n_heads_kv=HKV, head_dim=D,
                dtype="bfloat16", mask="causal")
    base.update(kw)
    return AttnConfig(**base)


def _qkv(seed=0, batch=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, HQ, S, D, generator=g)
    k = torch.randn(batch, HKV, S, D, generator=g)
    v = torch.randn(batch, HKV, S, D, generator=g)
    return q, k, v


@pytest.fixture
def fakes(monkeypatch):
    calls = {}
    nb = -(-S // 128)

    def xattn_estimate(q, k, **kw):
        calls["estimate"] = (q, k, kw)
        padded = 4                               # pretend padding to a larger grid
        m = torch.zeros(1, k.shape[1], padded, padded, dtype=torch.bool)
        m[:, :, :nb, :nb] = torch.ones(nb, nb, dtype=torch.bool).tril()
        m[:, :, 0, :] = True                     # keep_sink's whole row 0, as the original does
        return torch.zeros(1), m

    def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming,
                               base_blockmask, sq, sk, **kw):
        calls["kernel"] = dict(q=q, k=k, v=v, cu_q=cu_q, cu_k=cu_k,
                               head_mask_type=head_mask_type, streaming=streaming,
                               mask=base_blockmask, sq=sq, sk=sk, kw=kw)
        return q.clone()

    pkg = types.ModuleType("xattn"); src = types.ModuleType("xattn.src")
    mod = types.ModuleType("xattn.src.Xattention"); mod.xattn_estimate = xattn_estimate
    bsa = types.ModuleType("block_sparse_attn"); bsa.block_sparse_attn_func = block_sparse_attn_func
    for name, m in (("xattn", pkg), ("xattn.src", src), ("xattn.src.Xattention", mod),
                    ("block_sparse_attn", bsa)):
        monkeypatch.setitem(sys.modules, name, m)
    return calls


def test_the_estimator_gets_xattention_prefills_arguments(fakes):
    b = XAttentionBackend(threshold=0.9)
    b.forward(*_qkv(), _cfg())
    q, k, kw = fakes["estimate"]
    assert kw == dict(block_size=128, stride=8, norm=1, threshold=0.9,
                      select_mode="inverse", use_triton=True, causal=True,
                      chunk_size=official_chunk_size(S), kdb=1,
                      keep_sink=True, keep_recent=True)


def test_kv_heads_are_repeated_as_repeat_kv_does(fakes):
    q, k, v = _qkv()
    XAttentionBackend().forward(q, k, v, _cfg())
    _, k_seen, _ = fakes["estimate"]
    assert k_seen.shape[1] == HQ
    assert torch.equal(k_seen, k.repeat_interleave(HQ // HKV, dim=1))


def test_the_kernel_call_matches_xattention_prefill(fakes):
    q, k, v = _qkv()
    out = XAttentionBackend().forward(q, k, v, _cfg())
    c = fakes["kernel"]
    nb = -(-S // 128)
    assert c["mask"].shape == (1, HQ, nb, nb) and c["mask"].is_contiguous()
    assert c["q"].shape == (S, HQ, D) and c["k"].shape == (S, HQ, D)
    assert c["cu_q"].tolist() == [0, S] and c["cu_k"].tolist() == [0, S]
    assert c["head_mask_type"].tolist() == [1] * HQ and c["streaming"] is None
    assert (c["sq"], c["sk"]) == (S, S)
    assert c["kw"] == dict(p_dropout=0.0, deterministic=True, is_causal=True)
    assert out.shape == q.shape


def test_official_chunk_size_at_known_lengths():
    """Hand-evaluated from Xattention_prefill's expression."""
    assert official_chunk_size(100) == 2048
    assert official_chunk_size(4096) == 4096
    assert official_chunk_size(8192) == 8192
    assert official_chunk_size(16383) == 16384
    assert official_chunk_size(32768) == 8192        # the 128*1024*2048 // p branch


def test_realised_density_counts_only_causal_blocks(fakes):
    b = XAttentionBackend()
    b.forward(*_qkv(), _cfg())
    # fake mask: full lower triangle (6 of 9 blocks valid) plus row 0's future
    # blocks from keep_sink, which must NOT count.
    assert b.last_layer_density == [1.0]
    m = torch.zeros(1, 2, 3, 3, dtype=torch.bool)
    m[:, :, [0, 1, 2], [0, 1, 2]] = True                  # diagonal only
    assert realised_density(m, 3, 3) == pytest.approx(3 / 6)


def test_refusals(fakes):
    b = XAttentionBackend()
    with pytest.raises(UnsupportedConfig, match="external mask"):
        b.forward(*_qkv(), _cfg(), mask=object())
    with pytest.raises(UnsupportedConfig, match="batch 1"):
        b.forward(*_qkv(batch=2), _cfg(batch=2))
    with pytest.raises(UnsupportedConfig, match="expects 'causal'"):
        b.forward(*_qkv(), _cfg(mask="block_sparse", block_size=128, sparsity=0.5,
                                mask_source="importance"))
    with pytest.raises(ValueError):
        XAttentionBackend(threshold=0.0)


def test_the_pinned_commit_is_a_full_sha():
    assert len(XATTN_COMMIT) == 40 and int(XATTN_COMMIT, 16) >= 0


def _real_xattn():
    if not torch.cuda.is_available():
        return False
    try:
        import block_sparse_attn  # noqa: F401
        import xattn.src.Xattention  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_xattn(), reason="needs CUDA + x-attention + block-sparse-attn (Phase A gate)")
@pytest.mark.parametrize("seq_len", [4096, 16384])
@pytest.mark.parametrize("threshold", [0.8, 0.9, 0.95])
def test_bitwise_equal_to_xattention_prefill_on_cuda(seq_len, threshold):
    from xattn.src.Xattention import Xattention_prefill
    g = torch.Generator(device="cuda").manual_seed(seq_len)
    q = torch.randn(1, 12, seq_len, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, seq_len, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, seq_len, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    cfg = AttnConfig(seq_len=seq_len, batch=1, n_heads_q=12, n_heads_kv=2,
                     head_dim=128, dtype="bfloat16", mask="causal")
    ours = XAttentionBackend(threshold=threshold).forward(q, k, v, cfg)
    kr, vr = k.repeat_interleave(6, dim=1), v.repeat_interleave(6, dim=1)
    theirs = Xattention_prefill(q, kr, vr, stride=8, norm=1, threshold=threshold,
                                use_triton=True, keep_sink=True, keep_recent=True)
    assert torch.equal(ours, theirs)
