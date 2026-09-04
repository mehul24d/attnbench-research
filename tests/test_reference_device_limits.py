"""A reference that cannot lower is one fewer opinion, not a verdict. CPU-only.

2026-09-04, seg2e. Stage 1 completed for the first time and reported 66
failures. All 66 read `reference flex raised InductorError`, in two variants:

  54  "No valid triton configs. OutOfMemoryError: out of resource:
       triton_tem_fused_0 Required: 114688 Hardware limit:101376"
  12  "Q and KV block size must be divisible by BLOCK_M and BLOCK_N.
       We got Q_BLOCK_SIZE=64 and KV_BLOCK_SIZE=64."

Both are flex's own documented limits on sm_89 (docs/limitations.md), and in
both cases flex was a REFERENCE, not the backend under test. The check treated
any non-OOM exception from a reference as a hard failure of the whole cell, so
a limit in one implementation was recorded as another implementation being
wrong. It cost block_sparse 42 of its 78 cells -- the sparse arm, at exactly
the lengths this study is about.

`UnsupportedConfig` and `OutOfMemoryError` from a reference were already
handled correctly, as "one fewer opinion". A device-capability limit is the
same thing arriving under a different exception type.

The hazard in fixing it is building a hatch that swallows real bugs, so these
tests are as much about what must STILL fail as about what must now be
skipped: matching is on the two known messages, never on the exception type.
"""

from __future__ import annotations

import pytest
import torch

from attnbench import gates
from attnbench.backends.base import AttentionBackend, Capability
from attnbench.config import AttnConfig
from attnbench.gates import CHECK_KIND_PAIR, device_limit_reason

SHARED_MEM = ("InductorError: RuntimeError: No valid triton configs. "
              "OutOfMemoryError: out of resource: triton_tem_fused_0 "
              "Required: 114688 Hardware limit:101376 Reducing block sizes "
              "or `num_stages` may help.")
BLOCK_DIVISIBILITY = ("InductorError: LoweringException: ValueError: Q and KV "
                      "block size must be divisible by BLOCK_M and BLOCK_N. "
                      "We got Q_BLOCK_SIZE=64 and KV_BLOCK_SIZE=64.")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [SHARED_MEM, BLOCK_DIVISIBILITY])
def test_the_two_observed_limits_are_recognised(msg):
    assert device_limit_reason(RuntimeError(msg)) is not None


def test_the_shared_memory_limit_is_named_for_what_it_is():
    assert "shared-memory" in device_limit_reason(RuntimeError(SHARED_MEM))


@pytest.mark.parametrize("msg", [
    "InductorError: CompilationError: at 7:11: unexpected token",
    "InductorError: AssertionError: expected 2 outputs, got 3",
    "InductorError: RuntimeError: Triton compilation failed",
    "RuntimeError: CUDA error: an illegal memory access was encountered",
    "ValueError: q must be contiguous",
])
def test_anything_else_is_not_a_device_limit(msg):
    """The load-bearing half. Matching on the EXCEPTION TYPE -- `except
    InductorError: skip` -- would silently downgrade the evidence behind every
    cell a genuinely broken reference touched, and there would be no trace:
    the cell would pass, with one fewer opinion and no reason to look."""
    assert device_limit_reason(RuntimeError(msg)) is None, (
        f"{msg[:50]!r} was classified as a hardware limit; it is not one")


def test_an_illegal_memory_access_is_never_a_mere_limit():
    """Explicit because it is the tempting one: it mentions memory, it comes
    from the device, and it is the single most destructive thing that can
    happen here -- an Xid 31 corrupts the whole CUDA context. Skipping past
    it as 'one fewer opinion' would mean continuing to run on a dead
    context."""
    assert device_limit_reason(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    ) is None


# ---------------------------------------------------------------------------
# Behaviour in the check
# ---------------------------------------------------------------------------

def _cap(name):
    return Capability(name=name, family="dense_exact",
                      min_compute_capability=(0, 0), supports_gqa=True,
                      supports_causal=True, block_sizes=(64, 128),
                      head_dims=(64, 128),
                      dtypes=("float32", "bfloat16", "float16"))


class _Dense(AttentionBackend):
    def __init__(self, name, bias=0.0, raises=None):
        self.capability = _cap(name)
        self._bias = bias
        self._raises = raises

    @property
    def name(self):
        return self.capability.name

    def forward(self, q, k, v, cfg, mask=None):
        if self._raises is not None:
            raise RuntimeError(self._raises)
        return v + self._bias


def _cfg():
    return AttnConfig(seq_len=128, batch=1, n_heads_q=4, n_heads_kv=4,
                      head_dim=64, dtype="float32", mask="causal")


def test_a_reference_at_a_device_limit_does_not_fail_the_cell():
    """The 66-cell bug, directly. flex could not lower a mask it never needed
    to, and block_sparse was recorded as wrong."""
    r = gates.check_cross_backend(
        _Dense("under_test"), _cfg(),
        [_Dense("flex", raises=SHARED_MEM), _Dense("b"), _Dense("c")],
        device="cpu")
    assert r.passed, r.detail
    assert r.check_kind == "cross_backend", (
        "two references still ran, so this is a full three-way agreement")
    assert "device limit" in r.detail, (
        "the lost reference should be named on the row, not silently dropped")


def test_the_lost_reference_is_recorded_as_skipped_not_as_disagreement():
    """It must be legible as the reference's problem. A cell that passes while
    quietly attributing the limit to the backend under test would be the same
    misattribution, just no longer fatal."""
    r = gates.check_cross_backend(
        _Dense("under_test"), _cfg(),
        [_Dense("flex", raises=BLOCK_DIVISIBILITY), _Dense("b")],
        device="cpu")
    assert "disagreement" not in r.detail.lower()
    assert "flex" in r.detail and "device limit" in r.detail


def test_losing_one_reference_to_a_limit_downgrades_to_a_pair():
    """Where cross_backend_pair finally earns its keep: two implementations
    still agree, and the row says the third was lost rather than pretending
    three opinions were had."""
    r = gates.check_cross_backend(
        _Dense("under_test"), _cfg(),
        [_Dense("flex", raises=SHARED_MEM), _Dense("b")],
        device="cpu")
    assert r.passed
    assert r.check_kind == CHECK_KIND_PAIR
    assert "WEAKER EVIDENCE" in r.detail


def test_a_real_reference_bug_still_fails_the_cell():
    """The hatch must not be open. An unrecognised error from a reference
    remains a hard failure -- erring toward failing loudly, because a
    wrongly-skipped reference weakens a pass with nothing to show for it."""
    r = gates.check_cross_backend(
        _Dense("under_test"), _cfg(),
        [_Dense("flex", raises="InductorError: AssertionError: broken"),
         _Dense("b"), _Dense("c")],
        device="cpu")
    assert not r.passed
    assert "raised" in r.detail


def test_every_reference_at_a_limit_leaves_too_few_and_says_so():
    r = gates.check_cross_backend(
        _Dense("under_test"), _cfg(),
        [_Dense("x", raises=SHARED_MEM), _Dense("y", raises=SHARED_MEM),
         _Dense("z", raises=SHARED_MEM)],
        device="cpu")
    assert not r.passed
    assert "device limit" in r.detail


def test_a_backend_under_test_at_a_device_limit_still_fails():
    """Asymmetric on purpose. A reference hitting its limit says nothing about
    the backend under test; the backend under test hitting one is exactly the
    kind of hardware-conditional behaviour this study reports."""
    r = gates.check_cross_backend(
        _Dense("under_test", raises=SHARED_MEM), _cfg(),
        [_Dense("b"), _Dense("c")], device="cpu")
    assert not r.passed
