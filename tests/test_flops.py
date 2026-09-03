"""AttnConfig.issued_flops()/useful_flops(): no coverage existed before this
file, and a real bug (block_sparse configs not getting the causal halving
issued_flops() applies to plain "causal" configs) had shipped silently as a
result -- found via timing_probe.py's grid-hour extrapolation, which relies
on useful_flops() being an accurate figure for the work a real kernel
issues. See config.py's issued_flops() docstring for the full explanation.
"""

from __future__ import annotations

from attnbench.config import AttnConfig


def _cfg(**overrides) -> AttnConfig:
    fields = dict(seq_len=1024, batch=1, n_heads_q=8, n_heads_kv=8, head_dim=64,
                  mask="causal")
    fields.update(overrides)
    return AttnConfig(**fields)


def test_block_sparse_gets_the_same_causal_halving_as_causal():
    causal_cfg = _cfg(mask="causal")
    block_sparse_cfg = _cfg(mask="block_sparse", sparsity=0.0, block_size=128,
                             mask_source="importance")
    assert block_sparse_cfg.issued_flops() == causal_cfg.issued_flops()


def test_full_mask_is_not_halved():
    full_cfg = _cfg(mask="full")
    causal_cfg = _cfg(mask="causal")
    assert full_cfg.issued_flops() == 2 * causal_cfg.issued_flops()


def test_useful_flops_discounts_the_causal_issued_flops_not_the_full_form():
    """The regression this guards: useful_flops() = issued_flops() * (1 -
    sparsity) must discount off the *causal* baseline for a block_sparse
    config, since every block_sparse config in this study is structurally
    causal (masks.mask_for hardcodes it) -- discounting off the full
    non-causal form silently doubles the reported "useful" work."""
    block_sparse_cfg = _cfg(mask="block_sparse", sparsity=0.5, block_size=128,
                             mask_source="importance")
    causal_cfg = _cfg(mask="causal")
    assert block_sparse_cfg.useful_flops() == causal_cfg.issued_flops() // 2


def test_fwd_bwd_multiplier_applies_after_the_causal_halving():
    fwd_cfg = _cfg(mask="causal", pass_kind="fwd")
    fwd_bwd_cfg = _cfg(mask="causal", pass_kind="fwd_bwd")
    assert fwd_bwd_cfg.issued_flops() == int(fwd_cfg.issued_flops() * 3.5)
