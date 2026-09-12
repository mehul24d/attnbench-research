from __future__ import annotations

import pandas as pd
import pytest
import torch

from attnbench import provenance
from attnbench.analysis import decode_confound
from attnbench.analysis.canary import canary_ratios
from attnbench.analysis.cross_arch import CrossArchError
from attnbench.backends.block_sparse import BlockSparseAttention
from attnbench.backends.impls import FlexAttentionBackend
from attnbench.backends.impls import SDPABackend
from attnbench.config import AttnConfig
from attnbench.gates import (check_correctness, check_cross_backend,
                             device_memory_bytes)
from attnbench.backends.base import UnsupportedConfig


class _NonFiniteBackend:
    def __init__(self, name):
        self.name = name

    def claims_support(self, cfg):
        return True, ""

    def make_inputs(self, cfg, device, seed):
        shape = (cfg.batch, cfg.n_heads_q, cfg.seq_len, cfg.head_dim)
        values = torch.zeros(shape, device=device, dtype=torch.float32)
        return values, values, values

    def forward(self, q, k, v, cfg, mask=None):
        return torch.full_like(q, float("nan"))


class _HealthyBackend(_NonFiniteBackend):
    """Finite, agreeing output -- a reference that is not itself broken."""

    def forward(self, q, k, v, cfg, mask=None):
        return torch.zeros_like(q)


class _WrongShapeBackend(_NonFiniteBackend):
    def forward(self, q, k, v, cfg, mask=None):
        return q[..., :-1]


def _gate_cfg():
    return AttnConfig(seq_len=64, batch=1, n_heads_q=1, n_heads_kv=1,
                      head_dim=64, dtype="float32", mask="causal")


def test_sh_rejects_nonzero_exit_even_with_stdout(monkeypatch):
    class FailedProcess:
        returncode = 1
        stdout = "HEAD\n"
        stderr = "fatal: bad revision\n"

    monkeypatch.setattr(provenance.shutil, "which", lambda _: "/bin/tool")
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: FailedProcess())

    assert provenance._sh(["git", "rev-parse", "HEAD"]) is None


def test_cross_backend_rejects_non_finite_outputs():
    """The backend under test emits NaN; the references are HEALTHY.

    This test used `_NonFiniteBackend` for the references too, which made it
    vacuous for the guard it names: deleting the backend-side non-finite
    check left it green, because the reference-side check caught the same
    NaN and failed the result for a different reason. Removing the guard
    then let a NaN-emitting backend through as

        passed=True, max_abs_err=0.0,
        "agrees with 2 independent implementations within atol=0.0"

    -- NaN != NaN, so every difference was dropped and zero comparisons were
    counted as full agreement. That is the failure the guard exists for, and
    only healthy references can express it. Verified by deletion on
    2026-09-12: with this version the test goes red.
    """
    backend = _NonFiniteBackend("a")
    result = check_cross_backend(
        backend, _gate_cfg(),
        [_HealthyBackend("b"), _HealthyBackend("c")], device="cpu")

    assert not result.passed
    assert "non-finite" in result.detail
    assert result.max_abs_err != 0.0 or "non-finite" in result.detail


def test_cross_backend_rejects_non_finite_reference():
    """The other polarity: a broken reference must not certify a good
    backend. Both sides are checked, and each needs its own test."""
    result = check_cross_backend(
        _HealthyBackend("a"), _gate_cfg(),
        [_HealthyBackend("b"), _NonFiniteBackend("c")], device="cpu")

    assert not result.passed
    assert "non-finite" in result.detail


def test_exact_gate_rejects_wrong_shape_without_raising():
    result = check_correctness(_WrongShapeBackend("wrong_shape"),
                               _gate_cfg(), device="cpu")

    assert not result.passed
    assert "shape" in result.detail


def test_flex_cache_distinguishes_masks_with_one_config():
    class FakeMask:
        seq_len = 64
        block_size = 64

        def __init__(self, value):
            self.active = torch.tensor([[value]], dtype=torch.bool)

        def to_flex_block_mask(self, device):
            return self.active.item()

    cfg = AttnConfig(seq_len=64, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=8, dtype="float32", mask="block_sparse",
                     block_size=64, sparsity=0.5, mask_source="random")
    backend = FlexAttentionBackend()

    first = backend._block_mask_for(cfg, FakeMask(True), "cpu")
    second = backend._block_mask_for(cfg, FakeMask(False), "cpu")

    assert first is True
    assert second is False


def test_flex_refuses_to_cache_a_mask_it_cannot_hash():
    """The F7 fallback branch was `mask_key = id(mask)`, which is wrong in
    both directions: CPython reuses an address once the object at it is
    collected, so a later mask inherits an earlier one's key and is served
    the earlier one's BlockMask; and a mask rebuilt from identical inputs
    lands at a new address, so the key is not a function of what it claims
    to identify. A cache key that is not derived from the value is exactly
    the bug F7 fixed, reintroduced in the branch that handles the case F7's
    hash cannot reach. A miss costs a rebuild; a collision costs a wrong
    measurement, so the fallback now declines to cache."""
    class UnhashableMask:
        seq_len = 64
        block_size = 64
        active = None                     # nothing to digest

        def __init__(self, value):
            self.value = value

        def to_flex_block_mask(self, device):
            return self.value

    cfg = AttnConfig(seq_len=64, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=8, dtype="float32", mask="block_sparse",
                     block_size=64, sparsity=0.5, mask_source="random")
    backend = FlexAttentionBackend()

    assert backend._block_mask_for(cfg, UnhashableMask("first"), "cpu") == "first"
    # Nothing was stored, so nothing can be served in place of the next one
    # -- including a mask that happens to land at a recycled address.
    assert "_bm_key" not in backend.__dict__
    assert backend._block_mask_for(cfg, UnhashableMask("second"), "cpu") == "second"


def test_sdpa_does_not_relabel_oom_as_unsupported(monkeypatch):
    def raise_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("out of memory")

    monkeypatch.setattr(torch.nn.functional,
                        "scaled_dot_product_attention", raise_oom)
    cfg = _gate_cfg()
    q = torch.zeros((1, 1, 64, 8))

    with pytest.raises(torch.cuda.OutOfMemoryError):
        SDPABackend("math").forward(q, q, q, cfg)


def test_device_memory_uses_requested_cuda_device(monkeypatch):
    class Device:
        def __init__(self, memory):
            self.total_memory = memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: Device(10 if device == 0 or str(device) == "cuda:0" else 20),
    )

    assert device_memory_bytes("cuda:1") == 20


def test_block_sparse_rejects_mismatched_mask_geometry_before_kernel_import():
    class WrongMask:
        seq_len = 32
        block_size = 64

    cfg = AttnConfig(seq_len=64, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=8, dtype="float16", mask="block_sparse",
                     block_size=64, sparsity=0.5, mask_source="random")
    q = torch.zeros((1, 1, 64, 8), dtype=torch.float16)

    with pytest.raises(UnsupportedConfig, match="mask is"):
        BlockSparseAttention().forward(q, q, q, cfg, mask=WrongMask())


def test_canary_rejects_non_finite_backend_latency():
    frame = pd.DataFrame([
        {"host": "host-1", "gpu_name": "NVIDIA L4", "backend": "sdpa_flash",
         "config_key": "c4096", "seq_len": 4096,
         "latency_ms_p50": 40.0, "ok": True},
        {"host": "host-1", "gpu_name": "NVIDIA L4", "backend": "gla",
         "config_key": "c4096", "seq_len": 4096,
         "latency_ms_p50": float("inf"), "ok": True},
    ])

    with pytest.raises(CrossArchError, match="finite and positive"):
        canary_ratios(frame)


def test_decode_correction_subtracts_n_minus_one_penalties():
    phases = pd.DataFrame([
        {"context_length": 2048, "sparsity": None, "phase": "prefill",
         "backend": "sdpa_flash", "ms_mean": 100.0},
        {"context_length": 2048, "sparsity": 0.75, "phase": "prefill",
         "backend": "block_sparse", "ms_mean": 90.0},
        {"context_length": 2048, "sparsity": None, "phase": "decode_step",
         "backend": "sdpa_flash", "ms_mean": 10.0},
        {"context_length": 2048, "sparsity": 0.75, "phase": "decode_step",
         "backend": "block_sparse", "ms_mean": 15.0},
    ])
    pareto = pd.DataFrame([
        {"task": "t", "context_length": 2048, "epsilon": 1.0,
         "backend": "sdpa_flash", "sparsity": float("nan"),
         "latency_ms": 140.0, "margin": 1.0,
         "is_dense_reference": True, "dominated_by_dense": False},
        {"task": "t", "context_length": 2048, "epsilon": 1.0,
         "backend": "block_sparse", "sparsity": 0.75,
         "latency_ms": 130.0, "margin": 1.0,
         "is_dense_reference": False, "dominated_by_dense": False},
    ])

    points = decode_confound.correct(
        pareto,
        phases,
        {
            ("sdpa_flash", "t", 2048, None): 4.0,
            ("block_sparse", "t", 2048, 0.75): 4.0,
        },
        arms_share_decode_backend=False,
    )
    sparse = next(point for point in points if point.backend == "block_sparse")

    # Penalty = 15 - 10 = 5 ms; four generated tokens cost three decode steps.
    assert sparse.decode_corrected_ms == pytest.approx(115.0)

# --- 2026-09-12: the phase-era pairing check ------------------------------
#
# `decode_corrected_ms` subtracts a penalty measured by a `phases` frame, and
# nothing tied that frame's decode era to the era of the rows it corrected.
# `results/stage6/decode_corrected.parquet` was built from
# `results/stage5_ols/phases.parquet` (both arms already on `sdpa_flash`,
# penalty 0.29-0.75 ms) applied to `results/stage3_s1b/` rows (sparse arm on
# `sdpa_math`, real penalty 8-22 ms). The column was a no-op wearing the name
# of a correction, and the confound it names survived it untouched.

def _era_frames(sparse_decode_ms):
    phases = pd.DataFrame([
        {"context_length": 2048, "sparsity": None, "phase": "prefill",
         "backend": "sdpa_flash", "ms_mean": 100.0},
        {"context_length": 2048, "sparsity": 0.75, "phase": "prefill",
         "backend": "block_sparse", "ms_mean": 90.0},
        {"context_length": 2048, "sparsity": None, "phase": "decode_step",
         "backend": "sdpa_flash", "ms_mean": 34.0},
        {"context_length": 2048, "sparsity": 0.75, "phase": "decode_step",
         "backend": "block_sparse", "ms_mean": sparse_decode_ms},
    ])
    pareto = pd.DataFrame([
        {"task": "t", "context_length": 2048, "epsilon": 1.0,
         "backend": "sdpa_flash", "sparsity": float("nan"),
         "latency_ms": 600.0, "margin": 1.0,
         "is_dense_reference": True, "dominated_by_dense": False},
        {"task": "t", "context_length": 2048, "epsilon": 1.0,
         "backend": "block_sparse", "sparsity": 0.75,
         "latency_ms": 700.0, "margin": 1.0,
         "is_dense_reference": False, "dominated_by_dense": False},
    ])
    n_gen = {("sdpa_flash", "t", 2048, None): 14.0,
             ("block_sparse", "t", 2048, 0.75): 14.0}
    return pareto, phases, n_gen


def test_correction_refuses_flash_era_phases_for_confounded_rows():
    pareto, phases, n_gen = _era_frames(34.3)      # +0.3 ms: same kernel
    with pytest.raises(decode_confound.PhaseEraMismatch, match="no-op"):
        decode_confound.correct(pareto, phases, n_gen,
                                arms_share_decode_backend=False)


def test_correction_refuses_math_era_phases_for_matched_rows():
    pareto, phases, n_gen = _era_frames(55.0)      # +21 ms: different kernels
    with pytest.raises(decode_confound.PhaseEraMismatch, match="not in the data"):
        decode_confound.correct(pareto, phases, n_gen,
                                arms_share_decode_backend=True)


def test_correction_accepts_each_matched_pairing():
    for sparse_ms, shared in ((55.0, False), (34.3, True)):
        pareto, phases, n_gen = _era_frames(sparse_ms)
        points = decode_confound.correct(pareto, phases, n_gen,
                                         arms_share_decode_backend=shared)
        sparse = next(p for p in points if p.backend == "block_sparse")
        assert sparse.decode_penalty_ms == pytest.approx(sparse_ms - 34.0)
