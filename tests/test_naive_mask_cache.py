"""`NaiveAttention` caches its dense block mask per config AND mask contents.

Until 27 Oct 2026 the cache key was `(cfg.key(), device)`. `cfg.key()` does
not include the `BlockSparseMask`, so one instance handed a second mask at the
same config (another layer's, say) returned the first mask's output, 0.51 max
difference against a fresh instance. Nothing banked went through one instance
per layer (`gates.check_correctness` builds a fresh one per call), but a
replay, a recorder or a toy-model test would.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from attnbench import masks
from attnbench.backends.impls import NaiveAttention
from attnbench.config import AttnConfig


def _cfg(seq_len=256, block=64):
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=2, head_dim=16,
                      dtype="float32", mask="block_sparse", block_size=block,
                      sparsity=0.5, mask_source="importance")


def _qkv(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, cfg.n_heads_q, cfg.seq_len, cfg.head_dim, generator=g)
    k = torch.randn(1, cfg.n_heads_kv, cfg.seq_len, cfg.head_dim, generator=g)
    v = torch.randn(1, cfg.n_heads_kv, cfg.seq_len, cfg.head_dim, generator=g)
    return q, k, v


def _mask(cfg, seed):
    n = masks._n_blocks(cfg.seq_len, cfg.block_size)
    scores = torch.rand(n, n, generator=torch.Generator().manual_seed(seed))
    return masks.mask_for(cfg, importance_scores=scores)


def test_a_second_mask_at_the_same_config_is_applied_not_the_first():
    cfg = _cfg()
    q, k, v = _qkv(cfg)
    m1, m2 = _mask(cfg, 1), _mask(cfg, 2)
    assert not torch.equal(m1.active, m2.active)

    one = NaiveAttention()
    first = one.forward(q, k, v, cfg, mask=m1)
    second = one.forward(q, k, v, cfg, mask=m2)
    fresh = NaiveAttention().forward(q, k, v, cfg, mask=m2)

    assert torch.equal(second, fresh)
    assert not torch.allclose(second, first, atol=1e-4)
    # And back again: the cache holds one entry, and returning to the first
    # mask rebuilds it rather than keeping the second.
    assert torch.equal(one.forward(q, k, v, cfg, mask=m1), first)


def test_equal_contents_are_one_cache_entry(monkeypatch):
    """The hoisting still holds: repeated calls with the same mask, or with a
    mask rebuilt from the same scores, build the dense mask once."""
    cfg = _cfg()
    q, k, v = _qkv(cfg)
    builds = []
    real = masks.BlockSparseMask.to_dense_bool

    def counted(self, *a, **kw):
        builds.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(masks.BlockSparseMask, "to_dense_bool", counted)
    one = NaiveAttention()
    for _ in range(3):
        one.forward(q, k, v, cfg, mask=_mask(cfg, 1))
    assert len(builds) == 1
    one.forward(q, k, v, cfg, mask=_mask(cfg, 2))
    assert len(builds) == 2


def test_a_mask_without_contents_to_key_on_is_never_cached():
    """A mask object with no `active` tensor cannot be keyed by content, so
    it is rebuilt on every call rather than keyed by `id()`."""
    cfg = _cfg()
    q, k, v = _qkv(cfg)
    real1, real2 = _mask(cfg, 1), _mask(cfg, 2)
    builds = []

    def opaque(real):
        def dense(device=None):
            builds.append(1)
            return real.to_dense_bool(device=device)
        return SimpleNamespace(seq_len=real.seq_len, block_size=real.block_size,
                               to_dense_bool=dense)

    one = NaiveAttention()
    a = one.forward(q, k, v, cfg, mask=opaque(real1))
    b = one.forward(q, k, v, cfg, mask=opaque(real2))
    assert len(builds) == 2
    assert torch.equal(a, NaiveAttention().forward(q, k, v, cfg, mask=real1))
    assert torch.equal(b, NaiveAttention().forward(q, k, v, cfg, mask=real2))
