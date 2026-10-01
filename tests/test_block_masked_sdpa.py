"""`block_masked_sdpa` computes exactly what a block mask at its block size
computes -- the property the block-size accuracy arms rest on.

Checked against `NaiveAttention`, the float64-gated correctness oracle, at
block sizes the BSA kernel cannot run (16, 32, 64), with GQA, and with the
chunked mask construction held equal to `BlockSparseMask.to_dense_bool()`
row for row.
"""

from __future__ import annotations

import pytest
import torch

from attnbench import masks
from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.block_masked import BlockMaskedSDPA
from attnbench.backends.impls import NaiveAttention
from attnbench.config import AttnConfig


def _setup(seq_len, block, sparsity=0.75, hq=4, hkv=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, hq, seq_len, 16, generator=g)
    k = torch.randn(1, hkv, seq_len, 16, generator=g)
    v = torch.randn(1, hkv, seq_len, 16, generator=g)
    cfg = AttnConfig(seq_len=seq_len, batch=1, n_heads_q=hq, n_heads_kv=hkv, head_dim=16,
                     dtype="float32", mask="block_sparse", block_size=block,
                     sparsity=sparsity, mask_source="importance")
    n = masks._n_blocks(seq_len, block)
    scores = torch.rand(n, n, generator=g)
    mask = masks.mask_for(cfg, importance_scores=scores)
    return q, k, v, cfg, mask


@pytest.mark.parametrize("block", [16, 32, 64])
@pytest.mark.parametrize("seq_len", [256, 300])          # 300: a partial final block
@pytest.mark.parametrize("hkv", [4, 2, 1])
def test_equals_the_naive_oracle(block, seq_len, hkv):
    q, k, v, cfg, mask = _setup(seq_len, block, hkv=hkv)
    ours = BlockMaskedSDPA(chunk_rows=64).forward(q, k, v, cfg, mask=mask)
    ref = NaiveAttention().forward(q, k, v, cfg, mask=mask)
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-5), (ours - ref).abs().max()


@pytest.mark.parametrize("block", [16, 64, 128])
def test_chunk_masks_reassemble_to_the_full_expansion(block):
    _, _, _, _, mask = _setup(300, block)
    s = 300
    cols = torch.arange(s)
    full = mask.to_dense_bool(device="cpu")
    for chunk in (1, 7, 64, 300):
        parts = [BlockMaskedSDPA._chunk_mask(mask.active, block, r0, min(r0 + chunk, s), s,
                                             cols, mask.causal)
                 for r0 in range(0, s, chunk)]
        assert torch.equal(torch.cat(parts, dim=0), full), chunk


def test_output_does_not_depend_on_chunk_size():
    q, k, v, cfg, mask = _setup(300, 16)
    a = BlockMaskedSDPA(chunk_rows=7).forward(q, k, v, cfg, mask=mask)
    b = BlockMaskedSDPA(chunk_rows=4096).forward(q, k, v, cfg, mask=mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_a_sparser_mask_changes_the_output():
    q, k, v, cfg, mask = _setup(256, 16, sparsity=0.5)
    _, _, _, cfg9, mask9 = _setup(256, 16, sparsity=0.9)
    a = BlockMaskedSDPA().forward(q, k, v, cfg, mask=mask)
    b = BlockMaskedSDPA().forward(q, k, v, cfg9, mask=mask9)
    assert not torch.allclose(a, b, atol=1e-4)


def test_refusals():
    q, k, v, cfg, mask = _setup(256, 16)
    with pytest.raises(UnsupportedConfig, match="needs a BlockSparseMask"):
        BlockMaskedSDPA().forward(q, k, v, cfg)
    _, _, _, _, other = _setup(256, 32)
    with pytest.raises(UnsupportedConfig, match="but cfg is"):
        BlockMaskedSDPA().forward(q, k, v, cfg, mask=other)
