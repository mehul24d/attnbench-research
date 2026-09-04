"""A kernel that faults the device is recorded, not omitted. CPU-only.

`sdpa_cudnn` reads unmapped memory above seq_len=8192 on sm_89. Observed
2026-09-04 on an L4 (driver 580.173.02, torch 2.9.1+cu129, CUDA 12.9): the
Stage 0 probe reached the 16384 band, nine backends completed all 84 configs
in it, cuDNN wrote zero, and the process died with

    torch.AcceleratorError: CUDA error: an illegal memory access
    NVRM: Xid 31 ... MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_T1_3

An illegal access corrupts the CUDA context for the entire process, so it
cannot be caught and recovered from -- `try/except` around the call is useless
because the damage is to the context, not the Python frame. The only safe
handling is not to launch it, which makes this a capability declaration rather
than an exception handler.

Two things must both hold, and they pull in opposite directions:

  * the faulting cells are never launched, and
  * they appear in the capability matrix as a FINDING, not as a gap.

A shipping cuDNN kernel that works at 8192 and faults the device at 16384 on
the same card is a result about a production kernel -- and the starkest
possible form of the hardware-conditional behaviour this study is about. It is
not an omission to be quietly skipped.
"""

from __future__ import annotations

import pytest

from attnbench.backends.base import FAULT_REASON_PREFIX
from attnbench.backends.impls import SDPABackend
from attnbench.config import AttnConfig

FAULTS_ABOVE = SDPABackend._CUDNN_FAULTS_ABOVE


def _cfg(seq_len: int) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=32, n_heads_kv=32,
                      head_dim=128, dtype="bfloat16", mask="causal",
                      pass_kind="fwd", regime="prefill")


@pytest.mark.parametrize("seq_len", [16384, 32768])
def test_cudnn_is_refused_above_the_observed_safe_length(seq_len):
    ok, reason = SDPABackend("cudnn").claims_support(_cfg(seq_len))
    assert not ok
    assert reason.startswith(FAULT_REASON_PREFIX)
    assert "Xid 31" in reason


@pytest.mark.parametrize("seq_len", [1024, 4096, 8192])
def test_cudnn_still_runs_at_lengths_that_were_observed_working(seq_len):
    """8192 is the highest length cuDNN actually completed (84/84 configs), so
    that is where the line sits -- not a round number chosen for tidiness."""
    ok, reason = SDPABackend("cudnn").claims_support(_cfg(seq_len))
    assert ok, reason


@pytest.mark.parametrize("kernel", ["flash", "efficient", "math"])
@pytest.mark.parametrize("seq_len", [16384, 32768])
def test_the_other_sdpa_kernels_are_untouched(kernel, seq_len):
    """The failure mode of a class-level declaration: all four SDPA variants
    share one class, so declaring the fault on the class would delete three
    backends' worth of long-context coverage to work around one. All three
    completed the 16384 band."""
    ok, reason = SDPABackend(kernel).claims_support(_cfg(seq_len))
    assert ok, f"{kernel} wrongly excluded at {seq_len}: {reason}"


def test_the_cap_matches_the_highest_length_observed_working():
    assert FAULTS_ABOVE == 8192, (
        "8192 is the longest band cuDNN completed (84/84). Raising this needs "
        "an observation at the new length, not a larger number.")


def test_a_faulting_cell_is_recorded_as_a_finding_not_as_unsupported():
    """The distinction the results table must not lose. 'unsupported' would say
    the kernel declines the config -- false, and much milder than the truth --
    and would bury it among fifty ordinary declines."""
    from attnbench import gates

    result = gates.probe(SDPABackend("cudnn"), _cfg(16384))
    assert result.actual == "illegal_memory_access"
    assert "Xid 31" in result.detail
    assert result.claimed is False
    assert result.claim_mismatch is False, (
        "claim and behaviour agree -- the backend says it faults here and it "
        "does; flagging a mismatch would make the real mismatches harder to see")


def test_probing_a_faulting_cell_launches_nothing(monkeypatch):
    """The load-bearing half. If the kernel were launched anyway, recording a
    nice status afterwards would be irrelevant: the process would be dead."""
    from attnbench import gates

    launched = []
    monkeypatch.setattr(SDPABackend, "run_once",
                        lambda self, cfg, mask=None: launched.append(cfg))
    gates.probe(SDPABackend("cudnn"), _cfg(16384))
    assert launched == [], "the faulting kernel was launched"


def test_a_safe_length_still_reaches_the_kernel(monkeypatch):
    """The other side: an exclusion that swallowed everything would be worse
    than the fault it prevents."""
    from attnbench import gates

    launched = []
    monkeypatch.setattr(SDPABackend, "run_once",
                        lambda self, cfg, mask=None: launched.append(cfg))
    monkeypatch.setattr("torch.cuda.synchronize", lambda *a, **k: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda *a, **k: None)
    gates.probe(SDPABackend("cudnn"), _cfg(8192))
    assert len(launched) == 1


def test_the_sweep_will_not_plan_faulting_cells():
    """Stage 2 filters on claims_support, so exclusion propagates there for
    free -- but 'for free' is exactly the kind of assumption worth asserting."""
    ok, _ = SDPABackend("cudnn").claims_support(_cfg(16384))
    assert not ok
