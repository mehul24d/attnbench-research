"""masks.py: determinism and nesting. Everything downstream (Stage 2 timing
comparisons, Stage 4's accuracy-vs-sparsity curve) depends on mask identity
being correct, so this validates it in isolation, on CPU, before any backend
consumes it.
"""

from __future__ import annotations

import torch

from attnbench.config import AttnConfig
from attnbench.masks import mask_for


def _sparse_cfg(sparsity: float, **overrides) -> AttnConfig:
    fields = dict(seq_len=1024, batch=2, n_heads_q=8, n_heads_kv=8, head_dim=64,
                  dtype="bfloat16", mask="block_sparse", pass_kind="fwd",
                  sparsity=sparsity, block_size=64, mask_source="random")
    fields.update(overrides)
    return AttnConfig(**fields)


def test_random_mask_is_deterministic():
    cfg = _sparse_cfg(0.9)
    m1 = mask_for(cfg)
    m2 = mask_for(cfg)
    assert torch.equal(m1.active, m2.active)
    assert m1.seed == m2.seed


def test_random_mask_nests_across_sparsity():
    """The 0.9 mask's active set must be a strict subset of the 0.75 mask's
    -- higher sparsity is a prefix of the same shuffle, not an independent
    resample, or accuracy-vs-sparsity would carry resampling noise on top
    of the sparsity effect itself."""
    sparse = mask_for(_sparse_cfg(0.9))
    looser = mask_for(_sparse_cfg(0.75))
    assert bool((sparse.active <= looser.active).all())
    assert sparse.active.sum() < looser.active.sum()


def test_importance_mask_nests_across_sparsity():
    cfg_09 = _sparse_cfg(0.9, mask_source="importance")
    cfg_075 = _sparse_cfg(0.75, mask_source="importance")
    n = -(-cfg_09.seq_len // cfg_09.block_size)
    g = torch.Generator().manual_seed(0)
    scores = torch.rand(n, n, generator=g)

    sparse = mask_for(cfg_09, importance_scores=scores)
    looser = mask_for(cfg_075, importance_scores=scores)
    assert bool((sparse.active <= looser.active).all())
    assert sparse.active.sum() < looser.active.sum()


def test_unrelated_field_change_does_not_collapse_identity():
    """Two configs differing only in an unrelated field (dtype) must not be
    treated as 'the same' mask identity -- each still generates
    deterministically, but independently."""
    a = mask_for(_sparse_cfg(0.9, dtype="bfloat16"))
    b = mask_for(_sparse_cfg(0.9, dtype="float16"))
    assert a.seed != b.seed


def test_diagonal_always_active():
    m = mask_for(_sparse_cfg(0.99))
    assert bool(m.active.diagonal().all())


def test_no_active_block_above_diagonal():
    """Causal is structural for block_sparse configs in this study: no
    separate causal mask is applied on top (see NaiveAttention.forward),
    so the generated pattern must never mark a strictly-upper-triangular
    block active."""
    m = mask_for(_sparse_cfg(0.0))   # sparsity=0 -> every legal block active
    n = m.active.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    assert not bool((m.active & upper).any())


def test_to_dense_bool_shape_and_dtype():
    cfg = _sparse_cfg(0.9)
    m = mask_for(cfg)
    dense = m.to_dense_bool(device="cpu")
    assert dense.shape == (cfg.seq_len, cfg.seq_len)
    assert dense.dtype == torch.bool


def test_causal_dense_mask_never_permits_attending_to_the_future():
    """A block grid cannot express the triangle inside its own diagonal
    block, so `to_dense_bool` must apply sub-block causality itself.

    Before this was fixed, a causal block-sparse mask at seq_len=256 /
    block_size=64 permitted 8064 (query, key) pairs where the key was
    strictly in the future -- up to 63 per query, all inside diagonal
    blocks. `backends/block_sparse.py` passes `is_causal` to the real kernel
    and so masked them, meaning the correctness oracle and the kernel under
    test computed different things by construction.
    """
    import torch
    from attnbench.masks import random_block_mask

    for block_size in (64, 128):
        m = random_block_mask(seq_len=512, block_size=block_size, sparsity=0.5,
                              causal=True, identity_seed="causal-check")
        dense = m.to_dense_bool(device="cpu")
        qi = torch.arange(512).unsqueeze(1)
        ki = torch.arange(512).unsqueeze(0)
        future = dense & (ki > qi)
        assert int(future.sum()) == 0, (
            f"block_size={block_size}: {int(future.sum())} positions let a "
            f"query attend to a strictly future key")


def test_non_causal_dense_mask_is_not_triangularised():
    """The tril applies only when the mask is causal -- a non-causal pattern
    must keep its full block structure, or bidirectional configs silently
    lose half their mask."""
    import torch
    from attnbench.masks import random_block_mask

    m = random_block_mask(seq_len=256, block_size=64, sparsity=0.5,
                          causal=False, identity_seed="noncausal-check")
    dense = m.to_dense_bool(device="cpu")
    qi = torch.arange(256).unsqueeze(1)
    ki = torch.arange(256).unsqueeze(0)
    assert int((dense & (ki > qi)).sum()) > 0, (
        "a non-causal mask should still permit attending to later positions")


def test_diagonal_block_is_still_active_after_triangularisation():
    """The fix must not remove the diagonal block -- every query needs to
    attend to at least itself, and the diagonal is granted for free rather
    than spent from the sparsity budget."""
    import torch
    from attnbench.masks import random_block_mask

    m = random_block_mask(seq_len=256, block_size=64, sparsity=0.9,
                          causal=True, identity_seed="diag-check")
    dense = m.to_dense_bool(device="cpu")
    assert bool(dense.diagonal().all()), "every query must attend to itself"
    assert int(dense.sum(1).min()) >= 1
