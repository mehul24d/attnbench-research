"""Stage 1 certifies three different things, and the column says which.

Until 2026-09-03 the correctness gate applied one check -- exact agreement
with a float64 naive softmax oracle -- to every backend regardless of what it
computes. Consequences, both caught by the Stage 2 pass-table gate rather than
by any test here:

- `gla` (family=linear) failed 6/6 at `max_abs_err=1.42e+01`. Not a bug: GLA
  is not an approximation of softmax attention, it is a different function.
- `block_sparse` (family=sparse) was never probed at all, because the probe
  supplied no masks and so the backend had no supported config.

Both would have been rejected by the pass table, and Stage 2 would have
measured only dense backends -- a clean-looking 666-cell segment missing the
study's entire subject matter.

These tests were also impossible to write before `device` was threaded through
the gates: `make_inputs` defaulted to CUDA, so `check_correctness` and
`check_structural` could not run in CI at all. That is why the suite stayed
green while the functions went uncovered.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.backends.base import AttentionBackend, Capability
from attnbench.backends.impls import NaiveAttention
from attnbench.config import AttnConfig
from attnbench.gates import (CorrectnessResult, check_correctness,
                             check_for_family, check_structural)

CPU = dict(device="cpu")


def _cfg(mask="causal", **kw):
    base = dict(seq_len=64, batch=1, n_heads_q=4, n_heads_kv=4, head_dim=64,
                dtype="float32", mask=mask)
    base.update(kw)
    return AttnConfig(**base)


def _cap(name, family, **kw):
    return Capability(name=name, family=family, min_compute_capability=(0, 0),
                      supports_gqa=True, supports_causal=True,
                      block_sizes=(64, 128), head_dims=(64, 128),
                      dtypes=("float32", "bfloat16", "float16"), **kw)


class _WellBehavedLinear(AttentionBackend):
    """A causal linear backend: output at t depends only on tokens <= t."""
    capability = _cap("stub_linear", "linear")

    def forward(self, q, k, v, cfg, mask=None):
        return torch.cumsum(v, dim=2) / torch.arange(
            1, v.shape[2] + 1, device=v.device, dtype=v.dtype).view(1, 1, -1, 1)


class _FutureLeaking(AttentionBackend):
    """Looks fine on every other axis and leaks the future."""
    capability = _cap("stub_leaky", "linear")

    def forward(self, q, k, v, cfg, mask=None):
        # reverse cumulative mean: position t sees everything AFTER t
        flipped = torch.flip(v, dims=[2])
        out = torch.cumsum(flipped, dim=2) / torch.arange(
            1, v.shape[2] + 1, device=v.device, dtype=v.dtype).view(1, 1, -1, 1)
        return torch.flip(out, dims=[2])


class _NonDeterministic(AttentionBackend):
    capability = _cap("stub_random", "linear")

    def forward(self, q, k, v, cfg, mask=None):
        return v + torch.randn_like(v) * 1e-3


class _NonFinite(AttentionBackend):
    capability = _cap("stub_nan", "linear")

    def forward(self, q, k, v, cfg, mask=None):
        out = v.clone()
        out[:, :, 0, 0] = float("nan")
        return out


# ---------------------------------------------------------------------------


def test_check_kind_has_no_default():
    """A default would let a cross_backend or structural pass be constructed
    as 'exact' and read that way forever after."""
    with pytest.raises(TypeError, match="check_kind"):
        CorrectnessResult("gla", "abc", True)


def test_dense_backend_gets_the_exact_check():
    r = check_correctness(NaiveAttention(), _cfg(), **CPU)
    assert r.passed and r.check_kind == "exact"
    assert r.max_abs_err < 1e-5


def test_family_dispatch_sends_dense_to_exact():
    assert check_for_family(NaiveAttention(), _cfg(), **CPU).check_kind == "exact"


def test_family_dispatch_sends_linear_to_structural():
    r = check_for_family(_WellBehavedLinear(), _cfg(), **CPU)
    assert r.check_kind == "structural", (
        "a linear backend must NOT be graded against an exact softmax oracle "
        "-- that is a category error, and it produced gla failing 6/6")
    assert r.passed


def test_sparse_config_without_a_mask_source_fails_legibly():
    """Rather than a cryptic ValueError from deep inside mask_for()."""
    cfg = _cfg(mask="block_sparse", block_size=64, sparsity=0.5)
    r = check_for_family(NaiveAttention(), cfg, **CPU)
    assert not r.passed and "mask_source=None" in r.detail


def test_structural_check_catches_a_causality_violation():
    """The load-bearing structural property.

    A linear-attention kernel that leaked future information would be badly
    broken in a way no throughput measurement reveals, and the corruption
    would reach Stage 3 as plausible accuracy numbers.
    """
    r = check_structural(_FutureLeaking(), _cfg(), **CPU)
    assert not r.passed
    assert "CAUSALITY VIOLATED" in r.detail
    assert r.check_kind == "structural"


def test_structural_check_catches_non_determinism():
    r = check_structural(_NonDeterministic(), _cfg(), **CPU)
    assert not r.passed and "non-deterministic" in r.detail


def test_structural_check_catches_non_finite_output():
    r = check_structural(_NonFinite(), _cfg(), **CPU)
    assert not r.passed and "non-finite" in r.detail


def test_structural_pass_says_what_it_does_not_certify():
    """A reader must not mistake a structural pass for numerical agreement."""
    r = check_structural(_WellBehavedLinear(), _cfg(), **CPU)
    assert r.passed
    assert "NOT numerical agreement" in r.detail


def test_sparse_family_is_given_a_mask_automatically():
    """block_sparse had zero correctness rows because the probe supplied no
    mask, so it had no supported config, so it had no pass -- and Stage 2
    would have dropped it wholesale."""
    cfg = _cfg(mask="block_sparse", block_size=64, sparsity=0.5,
               mask_source="random")
    r = check_for_family(NaiveAttention(), cfg, **CPU)
    assert r.check_kind == "masked_exact", (
        "a sparse config must be compared against the oracle given the SAME "
        "mask; comparing with no mask compares two different computations")
    assert r.passed


# ---------------------------------------------------------------------------
# Cross-backend agreement, where no float64 oracle can be allocated.
# ---------------------------------------------------------------------------

class _Dense(AttentionBackend):
    """A dense backend with a controllable bias, for agreement tests."""

    def __init__(self, name, bias=0.0):
        self.capability = _cap(name, "dense_exact")
        self._bias = bias

    @property
    def name(self):
        return self.capability.name

    def forward(self, q, k, v, cfg, mask=None):
        return v + self._bias


def test_cross_backend_requires_three_independent_implementations():
    """Two kernels sharing a bug is plausible -- a common upstream, a shared
    CUTLASS path. Three from different authors is much less so."""
    from attnbench.gates import check_cross_backend
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg, [_Dense("b")], **CPU)
    assert not r.passed
    assert "need 3" in r.detail and r.check_kind == "cross_backend"


def test_cross_backend_passes_when_three_agree_and_records_the_error():
    """The error is recorded per row, and every row carries its seq_len, so
    divergence as a function of length is readable as a result rather than
    collapsed into pass/fail."""
    from attnbench.gates import check_cross_backend
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg, [_Dense("b"), _Dense("c")], **CPU)
    assert r.passed and r.check_kind == "cross_backend"
    assert r.max_abs_err == pytest.approx(0.0)
    assert "NOT verified against a float64 oracle" in r.detail


def test_a_single_disagreeing_backend_is_a_finding_not_a_skip():
    from attnbench.gates import check_cross_backend
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c", bias=1.0)], **CPU)
    assert not r.passed
    assert "a finding, not a cell to skip" in r.detail
    assert r.max_abs_err == pytest.approx(1.0)


def test_long_sequences_dispatch_to_cross_backend_not_the_oracle():
    cfg = _cfg(seq_len=16384, batch=1, n_heads_q=32, n_heads_kv=32)
    r = check_for_family(_Dense("a"), cfg,
                         references=[_Dense("b"), _Dense("c")], **CPU)
    assert r.check_kind == "cross_backend"


def test_short_sequences_still_use_the_exact_oracle():
    cfg = _cfg(seq_len=64)
    assert check_for_family(NaiveAttention(), cfg, **CPU).check_kind == "exact"


def test_long_sequence_without_references_fails_rather_than_inheriting():
    """Inheriting a short-length pass upward would assume exactly what Stage 1
    exists to measure: that numerical error does not accumulate with length."""
    cfg = _cfg(seq_len=16384, batch=1, n_heads_q=32, n_heads_kv=32)
    r = check_for_family(_Dense("a"), cfg, **CPU)
    assert not r.passed and "no reference backends were supplied" in r.detail


def test_dense_backends_do_not_claim_block_sparse():
    """A dense backend handed a block_sparse config ignores the mask entirely
    and computes plain attention. It "succeeds", so probe() marks it
    supported -- and the correctness check then compares it against an oracle
    that DID apply the mask.

    Measured 2026-09-03: max_abs_err=4.81, 114/126 failures for fa2,
    sdpa_flash and sdpa_cudnn. Not a numerical defect; a config they never
    agreed to run. The gate must filter on the CLAIM.
    """
    from attnbench.backends.impls import FlashAttention2, SDPABackend

    cfg = _cfg(mask="block_sparse", block_size=128, sparsity=0.5,
               mask_source="random")
    for backend in (SDPABackend("flash"), FlashAttention2()):
        claimed, reason = backend.claims_support(cfg)
        assert not claimed and "block sparse" in reason


def test_naive_claims_block_sparse_because_it_is_the_oracle():
    """It has an explicit block_sparse branch in forward() and provides the
    ground truth for every sparse comparison. Declaring False made
    claims_support reject the configs it exists to serve."""
    cfg = _cfg(mask="block_sparse", block_size=128, sparsity=0.5,
               mask_source="random")
    claimed, _ = NaiveAttention.claims_support(cfg)
    assert claimed


def test_oracle_feasibility_depends_on_the_whole_config_not_just_seq_len():
    """An early version thresholded on seq_len<=4096 assuming batch=1, and
    OOM'd on the batch-16 configs: 4 GiB at batch 1 is 64 GiB at batch 16.

    All 29 Stage 1 "failures" on 2026-09-03 were this OOM, not a single
    numerical disagreement.
    """
    from attnbench.gates import exact_oracle_fits

    fits = _cfg(seq_len=4096, batch=1, n_heads_q=32, n_heads_kv=32)
    same_length_bigger_batch = _cfg(seq_len=4096, batch=16, n_heads_q=32,
                                    n_heads_kv=32)
    assert exact_oracle_fits(fits)
    assert not exact_oracle_fits(same_length_bigger_batch), (
        "batch must count: the oracle materialises (batch, heads, S, S)")


def test_an_infeasible_oracle_config_reports_the_size_it_would_need():
    cfg = _cfg(seq_len=16384, batch=1, n_heads_q=32, n_heads_kv=32)
    r = check_for_family(_Dense("a"), cfg, **CPU)
    assert not r.passed and "GiB for this config" in r.detail


def test_cross_backend_tolerance_comes_from_the_compute_dtype():
    """Not from the dtype the comparison is cast to.

    An earlier version used TOL['float32'] (atol=1e-4) because both sides are
    compared as float32, and every bf16 config "failed" at 0.004-0.023 --
    ordinary bf16 kernel variation reported as disagreement.
    """
    from attnbench.gates import CROSS_BACKEND_TOL_FACTOR, TOL, check_cross_backend

    # a disagreement that is fine for bf16 but far outside float32 tolerance
    bf16_ok = TOL["bfloat16"]["atol"]
    cfg = _cfg(seq_len=128, dtype="bfloat16")
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c", bias=bf16_ok)], **CPU)
    assert r.passed, (
        f"{bf16_ok} is within bf16 tolerance and must not be reported as "
        f"disagreement just because float32 would be stricter")


def test_cross_backend_is_looser_than_the_exact_check():
    """Both sides are approximate here, so each contributes its own error from
    truth: two kernels each within epsilon can differ by 2*epsilon."""
    from attnbench.gates import CROSS_BACKEND_TOL_FACTOR
    assert CROSS_BACKEND_TOL_FACTOR == 2.0


def test_cross_backend_still_catches_a_real_disagreement():
    """The looser tolerance must not make the check toothless."""
    from attnbench.gates import CROSS_BACKEND_TOL_FACTOR, TOL, check_cross_backend

    way_off = TOL["bfloat16"]["atol"] * CROSS_BACKEND_TOL_FACTOR * 10
    cfg = _cfg(seq_len=128, dtype="bfloat16")
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c", bias=way_off)], **CPU)
    assert not r.passed and "a finding, not a cell to skip" in r.detail
