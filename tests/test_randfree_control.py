"""The arbitrary-free-block arm: the control for forcing the attention sink.

Forcing the sink does two things at once -- it grants the sink, and it grants
one more block per row (+53.8% of active blocks at 2048/0.9). The accuracy
gain could come from either. This arm holds the block count identical and
changes only WHICH block is free, so the property that makes it a control is
`same per-row count, different block`, and that is what these assert.
"""
import torch
import pytest

from attnbench import masks
from attnbench.config import AttnConfig


def _scores(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, n, generator=g)


def _pair(seq_len=8192, block_size=128, sparsity=0.9, seed="k"):
    sc = _scores(seq_len // block_size)
    a = masks.importance_block_mask(seq_len, block_size, sparsity, sc,
                                    causal=True, identity_seed=seed)
    b = masks.importance_block_mask(seq_len, block_size, sparsity, sc,
                                    causal=True, identity_seed=seed,
                                    free_block="random")
    return a, b


@pytest.mark.parametrize("sparsity", [0.5, 0.75, 0.9])
@pytest.mark.parametrize("seq_len", [2048, 8192])
def test_control_has_the_same_block_count_per_row(seq_len, sparsity):
    a, b = _pair(seq_len=seq_len, sparsity=sparsity)
    assert (a.active.sum(1) == b.active.sum(1)).all()
    assert int(a.active.sum()) == int(b.active.sum())


def test_control_does_not_force_the_sink():
    a, b = _pair()
    n = a.active.shape[0]
    assert int(a.active[:, 0].sum()) == n          # study rule: every row
    assert int(b.active[:, 0].sum()) < n // 2      # control: only by chance


def test_control_still_grants_the_diagonal():
    _, b = _pair()
    assert bool(b.active.diagonal().all())


def test_control_is_deterministic_and_seed_dependent():
    _, b1 = _pair(seed="k")
    _, b2 = _pair(seed="k")
    _, b3 = _pair(seed="other")
    assert bool((b1.active == b2.active).all())
    assert not bool((b1.active == b3.active).all())


def test_control_is_labelled_so_it_cannot_be_read_as_the_oracle_arm():
    a, b = _pair()
    assert a.source == "importance" and b.source == "importance_randfree"


def test_mask_for_dispatches_the_control():
    cfg = AttnConfig(seq_len=2048, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=128, mask="block_sparse", sparsity=0.9,
                     block_size=128, mask_source="importance_randfree")
    sc = _scores(2048 // 128)
    m = masks.mask_for(cfg, importance_scores=sc)
    assert m.source == "importance_randfree"
    cfg_sink = AttnConfig(**{**cfg.__dict__, "mask_source": "importance"})
    assert masks.mask_for(cfg_sink, importance_scores=sc).source == "importance"


def test_control_rejects_an_unknown_free_block():
    with pytest.raises(ValueError, match="free_block"):
        masks.importance_block_mask(2048, 128, 0.9, _scores(16), causal=True,
                                    identity_seed="k", free_block="diagonal")


def test_the_two_arms_actually_differ():
    a, b = _pair()
    assert not bool((a.active == b.active).all())
