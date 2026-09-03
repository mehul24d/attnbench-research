"""FlexAttention's block-sparse path: the study's second sparse arm.

**Why this backend exists.** Until it was wired up, the only sparse arm was
`BlockSparseAttention`, whose block size is hardcoded to 128 *inside*
`block_sparse_attn_func` -- not a config knob. Stage 2's grid runs
`block_sizes=(64, 128)`, and `sweep.build_cells` drops cells a backend does
not claim, so every 64-wide cell in Stage 2 had no sparse arm at all. Flex
covers both. It also needs no compilation step: it ships with torch, so a
failed BSA build session no longer means zero sparse timing results.

Almost everything here is CPU-testable, including the numerical check --
`flex_attention` runs eagerly on CPU (unfused, materialising the score
matrix, which is fine at test sizes). `torch.compile(flex_attention)` is
what the backend uses on GPU and is unavailable on CPU, so the numerical
test calls the eager form deliberately: it validates the *mask semantics*,
which is what this study wrote, rather than the fused kernel, which is
torch's.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.impls import (FlexAttentionBackend, NaiveAttention,
                                      _expand_kv)
from attnbench.config import AttnConfig
from attnbench.masks import random_block_mask

CAUSAL_AND_SIZES = [(c, bs) for c in (True, False) for bs in (64, 128)]


# head_dim=64: FlexAttentionBackend declares head_dims=(64, 128), so 32
# is rejected by claims_support before forward() is ever reached.
HEAD_DIM = 64


def _cfg(seq_len=128, block_size=64, sparsity=0.5, dtype="float32"):
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=HEAD_DIM, dtype=dtype, mask="block_sparse",
                      block_size=block_size, sparsity=sparsity)


@pytest.mark.parametrize("causal,block_size", CAUSAL_AND_SIZES)
def test_flex_block_mask_matches_the_dense_mask_elementwise(causal, block_size):
    """The two representations of one pattern must agree exactly.

    `BlockSparseMask` exists so that two backends asked for "the same" mask
    cannot diverge, but that guarantee is only real if the derived views
    actually agree. They differ in how causality is applied -- `to_dense_bool`
    triangularises after expanding, `to_flex_block_mask` evaluates `q >= kv`
    inside `mask_mod` -- so agreement is a claim to test, not to assume.
    """
    m = random_block_mask(seq_len=256, block_size=block_size, sparsity=0.5,
                          causal=causal, identity_seed="equivalence")
    bm = m.to_flex_block_mask(device="cpu")

    s = m.seq_len
    qi = torch.arange(s).view(-1, 1).expand(s, s)
    ki = torch.arange(s).view(1, -1).expand(s, s)
    zeros = torch.zeros_like(qi)
    flex_elementwise = bm.mask_mod(zeros, zeros, qi, ki).bool()

    assert torch.equal(flex_elementwise, m.to_dense_bool(device="cpu"))


@pytest.mark.parametrize("block_size", (64, 128))
def test_flex_block_size_is_pinned_to_the_masks_block_size(block_size):
    """If flex is left at its default BLOCK_SIZE=128, a 64-wide pattern is
    still computed *correctly* -- mask_mod is exact -- but the kernel skips
    work in 128-wide chunks, so the sparsity exploited is not the sparsity
    configured and the latency is not comparable to the other backends.
    A silent measurement error, which is why it is asserted.
    """
    m = random_block_mask(seq_len=512, block_size=block_size, sparsity=0.5,
                          causal=True, identity_seed="blocksize")
    bm = m.to_flex_block_mask(device="cpu")
    assert tuple(bm.BLOCK_SIZE) == (block_size, block_size)


@pytest.mark.parametrize("causal", (True, False))
def test_flex_output_matches_the_naive_oracle(causal):
    """End-to-end semantic check: flex consuming the BlockMask must produce
    the same attention as NaiveAttention consuming the dense mask.

    This is the test that would have caught the sub-block causality bug from
    the other direction -- the oracle permitted up to `block_size - 1` future
    keys inside diagonal blocks, which flex's elementwise `mask_mod` never
    would have.
    """
    from torch.nn.attention.flex_attention import flex_attention

    cfg = _cfg()
    m = random_block_mask(seq_len=cfg.seq_len, block_size=cfg.block_size,
                          sparsity=cfg.sparsity, causal=causal,
                          identity_seed="oracle")
    torch.manual_seed(0)
    q = torch.randn(1, 4, cfg.seq_len, HEAD_DIM)
    k = torch.randn(1, 2, cfg.seq_len, HEAD_DIM)
    v = torch.randn(1, 2, cfg.seq_len, HEAD_DIM)

    k_, v_ = _expand_kv(k, v, cfg)
    # eager flex warns that it is unfused; torch emits that via _warn_once,
    # so it cannot be asserted on across parametrisations. Suppressed rather
    # than asserted -- the numbers below are the actual claim.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        got = flex_attention(q, k_, v_, block_mask=m.to_flex_block_mask(device="cpu"))
    expected = NaiveAttention().forward(q, k, v, cfg, mask=m)

    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), (
        f"max abs err {(got - expected).abs().max().item():.3e}")


def test_block_sparse_without_a_mask_is_rejected():
    with pytest.raises(UnsupportedConfig, match="requires an explicit mask"):
        FlexAttentionBackend().forward(
            torch.zeros(1, 4, 128, HEAD_DIM), torch.zeros(1, 2, 128, HEAD_DIM),
            torch.zeros(1, 2, 128, HEAD_DIM), _cfg(), mask=None)


@pytest.mark.parametrize("bad", ("seq_len", "block_size"))
def test_mask_that_disagrees_with_the_cfg_is_rejected(bad):
    """A mask built for a different shape would measure a different sparsity
    than the cell claims to be measuring -- worse than an error, because the
    number produced looks entirely ordinary."""
    cfg = _cfg(seq_len=128, block_size=64)
    if bad == "seq_len":
        m = random_block_mask(seq_len=256, block_size=64, sparsity=0.5,
                              causal=True, identity_seed="x")
    else:
        m = random_block_mask(seq_len=128, block_size=128, sparsity=0.5,
                              causal=True, identity_seed="x")

    with pytest.raises(UnsupportedConfig, match="would measure a different"):
        FlexAttentionBackend().forward(
            torch.zeros(1, 4, 128, HEAD_DIM), torch.zeros(1, 2, 128, HEAD_DIM),
            torch.zeros(1, 2, 128, HEAD_DIM), cfg, mask=m)


def test_flex_claims_block_sparse_at_both_stage2_block_sizes():
    # bfloat16, not the float32 the numerical test uses: flex declares
    # dtypes=('bfloat16', 'float16'), and Stage 2 runs bf16 anyway. forward()
    # never consults claims_support, which is why the float32 numerical test
    # above is legitimate and this one still has to use a claimed dtype.
    for block_size in (64, 128):
        cfg = _cfg(block_size=block_size, dtype="bfloat16")
        ok, reason = FlexAttentionBackend.claims_support(cfg)
        assert ok, f"block_size={block_size} rejected: {reason}"


def test_flex_covers_the_block_size_bsa_cannot():
    """The gap this backend closes, stated as a test so it is noticed if BSA
    ever grows 64 support (at which point the redundancy is a choice, not an
    accident)."""
    from attnbench.backends.block_sparse import BlockSparseAttention

    cfg64 = _cfg(block_size=64, dtype="bfloat16")
    bsa_ok, _ = BlockSparseAttention.claims_support(cfg64)
    flex_ok, _ = FlexAttentionBackend.claims_support(cfg64)

    assert not bsa_ok, "BSA unexpectedly claims block_size=64"
    assert flex_ok, "flex must cover block_size=64 or Stage 2 has no sparse arm there"


def test_stage2_gains_block_sparse_cells_at_64():
    """The study-level claim, not just the capability flag: Stage 2's sweep
    must actually plan sparse cells at block_size=64.

    Before this backend was wired, `build_cells` produced 216 block_sparse
    cells at 128 (BSA) and ZERO at 64 -- half the block-size axis had no
    sparse arm, which is invisible in a results file because absent cells
    look the same as cells nobody asked for.
    """
    from collections import Counter

    from attnbench.backends.block_sparse import BlockSparseAttention
    from attnbench.config import SweepGrid
    from attnbench.sweep import build_cells

    cells = build_cells(SweepGrid(),
                        [FlexAttentionBackend(), BlockSparseAttention()],
                        mask_source="random")
    counts = Counter((c.backend_name, c.cfg.block_size)
                     for c in cells if c.cfg.mask == "block_sparse")

    assert counts[("flex", 64)] > 0, (
        "no sparse cells at block_size=64 -- the gap this backend closes")
    assert counts[("block_sparse", 64)] == 0, "BSA unexpectedly plans 64-wide cells"
    assert counts[("flex", 128)] > 0, "flex should also overlap BSA at 128"
