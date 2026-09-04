"""A reference that declines the config is not an opinion about it. CPU-only.

2026-09-04, seg2f. With the abort bug fixed, the 66 previously-aborted cells
ran to completion — and all 66 still failed, now with real numbers:
`max_abs_err` 4.39–5.15 for block_sparse and naive above 2048.

The tell was in the detail line:

    disagreement beyond atol=0.04: {'fa2': 3.71685791015625,
      'sdpa_efficient': 3.71685791015625, 'sdpa_math': 3.71685791015625,
      'sdpa_flash': 3.71685791015625, 'sdpa_cudnn': 3.71685791015625}

Five independent implementations agreeing to six decimal places are not five
opinions. They are one computation — plain causal attention — because a dense
backend handed a `block_sparse` config ignores the `mask` argument and
succeeds. Every one of them answers `claims_support` with `(False, "no block
sparse")`, and `check_cross_backend` called `forward` on them anyway.

This is instance 1 (an oracle computing a different function than the kernel)
arriving through the reference path. The same guard was added to
`run_probe.py`'s choice of BACKEND on 2026-09-03, after it produced
`max_abs_err=4.81` and 114/126 failures — and not to the choice of REFERENCE.
The fix was applied to one side of the comparison.
"""

from __future__ import annotations

import pytest
import torch

from attnbench import gates
from attnbench.backends.base import AttentionBackend, Capability, UnsupportedConfig
from attnbench.config import AttnConfig
from attnbench.gates import CHECK_KIND_PAIR, check_cross_backend


def _cap(name, **kw):
    base = dict(name=name, family="dense_exact", min_compute_capability=(0, 0),
                supports_gqa=True, supports_causal=True, block_sizes=(64, 128),
                head_dims=(64, 128), dtypes=("float32", "bfloat16"))
    base.update(kw)
    return Capability(**base)


class _Dense(AttentionBackend):
    """Ignores the mask, exactly as a real fused dense kernel does."""

    def __init__(self, name):
        self.capability = _cap(name)

    @property
    def name(self):
        return self.capability.name

    def claims_support(self, cfg):
        if cfg.mask == "block_sparse":
            return False, "no block sparse"
        return True, ""

    def forward(self, q, k, v, cfg, mask=None):
        return v * 2.0            # a confident, wrong answer


class _Sparse(AttentionBackend):
    """Honours the mask, so it computes a different function than _Dense."""

    def __init__(self, name):
        self.capability = _cap(name, family="sparse")

    @property
    def name(self):
        return self.capability.name

    def claims_support(self, cfg):
        return True, ""

    def forward(self, q, k, v, cfg, mask=None):
        return v


def _sparse_cfg():
    return AttnConfig(seq_len=128, batch=1, n_heads_q=4, n_heads_kv=4,
                      head_dim=64, dtype="float32", mask="block_sparse",
                      sparsity=0.5, block_size=64, mask_source="random")


def _dense_cfg():
    return AttnConfig(seq_len=128, batch=1, n_heads_q=4, n_heads_kv=4,
                      head_dim=64, dtype="float32", mask="causal")


def test_a_declining_reference_is_never_asked():
    """The load-bearing assertion. A backend that says it cannot run this
    config must not be called anyway and have its answer counted."""
    called = []

    class _Spy(_Dense):
        def forward(self, q, k, v, cfg, mask=None):
            called.append(cfg)
            return v * 2.0

    check_cross_backend(_Sparse("under_test"), _sparse_cfg(),
                        [_Spy("a"), _Spy("b"), _Spy("c")], device="cpu")
    assert called == [], "a reference that declines the config was run anyway"


def test_the_sparse_arm_is_not_recorded_as_wrong():
    """What the bug actually cost: block_sparse recorded as disagreeing by
    ~4.9 with five kernels that were not computing block-sparse attention."""
    r = check_cross_backend(_Sparse("block_sparse"), _sparse_cfg(),
                            [_Dense("a"), _Dense("b"), _Dense("c")],
                            device="cpu")
    assert "disagreement" not in r.detail.lower(), (
        "a mask-ignoring reference was still counted as an opinion")


def test_the_shortfall_is_reported_honestly():
    """With no mask-honouring reference available, the honest answer is 'too
    few implementations could run this shape' -- not a pass, and not a verdict
    against the backend."""
    r = check_cross_backend(_Sparse("block_sparse"), _sparse_cfg(),
                            [_Dense("a"), _Dense("b"), _Dense("c")],
                            device="cpu")
    assert not r.passed
    assert "declines this config" in r.detail
    assert "no block sparse" in r.detail


def test_a_declining_reference_is_recorded_as_skipped_not_dropped():
    """It must appear in the row. Filtering it out of `usable` silently would
    make 'three references were offered' unrecoverable from the data."""
    r = check_cross_backend(_Sparse("block_sparse"), _sparse_cfg(),
                            [_Dense("a"), _Sparse("s1"), _Sparse("s2")],
                            device="cpu")
    assert r.passed
    assert "a" in r.detail and "declines" in r.detail


def test_losing_one_reference_to_a_decline_downgrades_to_a_pair():
    r = check_cross_backend(_Sparse("block_sparse"), _sparse_cfg(),
                            [_Dense("a"), _Dense("b"), _Sparse("s1")],
                            device="cpu")
    assert r.passed
    assert r.check_kind == CHECK_KIND_PAIR


def test_dense_configs_are_unaffected():
    """The guard must not cost the dense arm its references: on a causal
    config every dense backend genuinely does claim support."""
    r = check_cross_backend(_Dense("under_test"), _dense_cfg(),
                            [_Dense("a"), _Dense("b"), _Dense("c")],
                            device="cpu")
    assert r.passed and r.check_kind == "cross_backend"
    assert "declines" not in r.detail


def test_five_identical_errors_are_the_signature_worth_remembering():
    """Not a code path -- a reading habit, pinned so the next reader has it.

    Five independent kernels disagreeing with the backend under test by
    EXACTLY the same amount is not five confirmations. Genuine numerical
    disagreement between different implementations varies in the low bits;
    identical values to six decimal places mean one computation wearing five
    names, and the question to ask is what all five have in common.
    """
    errors = {"fa2": 3.71685791015625, "sdpa_efficient": 3.71685791015625,
              "sdpa_math": 3.71685791015625, "sdpa_flash": 3.71685791015625,
              "sdpa_cudnn": 3.71685791015625}
    assert len(set(errors.values())) == 1
