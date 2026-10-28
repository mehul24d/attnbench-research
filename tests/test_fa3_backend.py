"""The FA3 backend, tested on CPU with a stand-in for `flash_attn_interface`.

No test here needs an H100. The stand-in computes real attention in FA3's
(B, S, H, D) layout with the pinned source's mask semantics (`causal`, and a
`window_size=(left, right)` band), so a backend that forgets a transpose,
flips the causal flag, swaps the window or unpacks the wrong element returns
the wrong numbers, not just the wrong shape. The expected output is computed
independently in (B, H, S, D) with an explicit mask.

Each defect named above was introduced into `fa3.py` once and watched to fail
here (28 Oct 2026); see the docstrings of the tests that catch them.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from attnbench import dense_selection as ds
from attnbench.backends import all_backends, get
from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.fa3 import FlashAttention3
from attnbench.config import AttnConfig


def _attend_bshd(q, k, v, causal, window_size):
    """Reference FA3 semantics in (B, S, H, D), for equal Q and K lengths:
    query i sees keys in [i - left, i + right], -1 meaning unbounded; causal
    sets right to 0. GQA by head grouping."""
    B, S, Hq, D = q.shape
    g = Hq // k.shape[2]
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    left, right = window_size
    if causal:
        right = 0
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    keep = torch.ones(S, S, dtype=torch.bool)
    if left >= 0:
        keep &= j >= i - left
    if right >= 0:
        keep &= j <= i + right
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / D ** 0.5
    s = s.masked_fill(~keep, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v.float()).to(q.dtype)


class _StandIn:
    """`flash_attn_interface` with a recorder. `returns_tuple` switches between
    the pinned return (`out`) and the older `(out, softmax_lse)`."""

    def __init__(self, returns_tuple=False):
        self.calls = []
        self.returns_tuple = returns_tuple

    def flash_attn_func(self, q, k, v, softmax_scale=None, causal=False,
                        window_size=(-1, -1), **kw):
        self.calls.append(dict(shapes=(q.shape, k.shape, v.shape), causal=causal,
                               window_size=window_size, extra=kw))
        out = _attend_bshd(q, k, v, causal, window_size)
        if self.returns_tuple:
            lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1])
            return out, lse
        return out


@pytest.fixture
def standin(monkeypatch):
    def install(returns_tuple=False):
        rec = _StandIn(returns_tuple)
        mod = types.ModuleType("flash_attn_interface")
        mod.flash_attn_func = rec.flash_attn_func
        monkeypatch.setitem(sys.modules, "flash_attn_interface", mod)
        return rec
    return install


def _cfg(mask="causal", window=None, hq=4, hkv=2, seq=16, d=64, **kw):
    return AttnConfig(seq_len=seq, batch=2, n_heads_q=hq, n_heads_kv=hkv,
                      head_dim=d, dtype="float32", mask=mask, window=window, **kw)


def _inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda h: torch.randn(cfg.batch, h, cfg.seq_len, cfg.head_dim, generator=g)
    return mk(cfg.n_heads_q), mk(cfg.n_heads_kv), mk(cfg.n_heads_kv)


def _expected_bhsd(q, k, v, cfg):
    """Independent of the stand-in: (B, H, S, D) with an explicit mask."""
    g = cfg.n_heads_q // cfg.n_heads_kv
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    S = cfg.seq_len
    i, j = torch.arange(S)[:, None], torch.arange(S)[None, :]
    keep = {"full": torch.ones(S, S, dtype=torch.bool), "causal": j <= i,
            "sliding": (j <= i) & (j >= i - (cfg.window or 0))}[cfg.mask]
    s = (q @ k.transpose(-1, -2)) / cfg.head_dim ** 0.5
    return s.masked_fill(~keep, float("-inf")).softmax(-1) @ v


# --- registration and declarations ------------------------------------------------


def test_fa3_is_registered_as_a_dense_exact_hopper_kernel():
    assert all_backends()["fa3"] is FlashAttention3
    c = get("fa3").capability
    assert (c.family, c.min_compute_capability) == ("dense_exact", (9, 0))
    assert c.supports_gqa and c.supports_sliding and not c.supports_backward
    assert c.head_dims == (64, 96, 128, 192, 256)
    assert c.dtypes == ("bfloat16", "float16")


def test_the_declared_head_dims_are_the_pinned_builds_instantiations():
    """The capability names the head dims the build compiles. The build script
    disables no head dim; if it ever does, the capability must follow."""
    from pathlib import Path
    script = (Path(__file__).resolve().parent.parent / "scripts" / "build_fa3_wheel.sh").read_text()
    assert "DISABLE_HDIM" not in script
    for flag in ("FLASH_ATTENTION_DISABLE_BACKWARD=TRUE", "FLASH_ATTENTION_DISABLE_FP8=TRUE",
                 "FLASH_ATTENTION_DISABLE_SM80=TRUE"):
        assert flag in script, flag


@pytest.mark.parametrize("cfg,reason", [
    (_cfg(pass_kind="fwd_bwd"), "no backward pass"),
    (_cfg(d=32), "head dim 32 unsupported"),
    (_cfg(), "dtype float32 unsupported"),
])
def test_the_capability_declines_what_the_wheel_does_not_have(cfg, reason):
    assert FlashAttention3.claims_support(cfg) == (False, reason)


def test_bfloat16_forward_is_claimed():
    cfg = AttnConfig(seq_len=8192, batch=1, n_heads_q=32, n_heads_kv=8, head_dim=128)
    assert FlashAttention3.claims_support(cfg) == (True, "")


# --- availability -----------------------------------------------------------------


def test_unavailable_without_cuda(standin, monkeypatch):
    standin()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not FlashAttention3.is_available()


@pytest.mark.parametrize("cc,importable,expected", [
    ((8, 0), True, False),     # A100: the kernels are sm_90a only
    ((8, 9), True, False),     # L4
    ((9, 0), True, True),
    ((9, 0), False, False),    # H100 without the wheel
])
def test_available_only_on_hopper_with_the_wheel(cc, importable, expected, standin, monkeypatch):
    if importable:
        standin()
    else:
        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cc)
    assert FlashAttention3.is_available() is expected


def test_the_dense_selection_supplies_fa3_on_an_h100_once_it_is_available(standin, monkeypatch):
    standin()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    got = ds.registry_candidates("NVIDIA H100 80GB HBM3")
    assert isinstance(got[ds.FA3], FlashAttention3)
    assert ds.FA3 not in ds.registry_candidates("NVIDIA A100-SXM4-80GB")


# --- the forward call -------------------------------------------------------------


@pytest.mark.parametrize("returns_tuple", [False, True])
@pytest.mark.parametrize("cfg", [
    _cfg("causal"), _cfg("full"), _cfg("sliding", window=3), _cfg("causal", hq=4, hkv=4),
], ids=["causal_gqa", "full_gqa", "sliding_gqa", "causal_mha"])
def test_forward_matches_an_independent_reference(cfg, returns_tuple, standin):
    """Kills: a missing transpose in or out (wrong numbers with H != S), the
    causal flag flipped or always False, the window passed as (0, w), and
    `_unpack` taking element -1 (returns the lse under the old shape)."""
    standin(returns_tuple)
    q, k, v = _inputs(cfg)
    out = FlashAttention3().forward(q, k, v, cfg)
    assert out.shape == q.shape
    torch.testing.assert_close(out, _expected_bhsd(q, k, v, cfg), rtol=1e-5, atol=1e-5)


def test_the_kernel_receives_bshd_tensors_and_the_flags(standin):
    rec = standin()
    cfg = _cfg("sliding", window=5, seq=12)
    FlashAttention3().forward(*_inputs(cfg), cfg)
    (call,) = rec.calls
    B, S, D = cfg.batch, cfg.seq_len, cfg.head_dim
    assert call["shapes"] == ((B, S, 4, D), (B, S, 2, D), (B, S, 2, D))
    assert call["causal"] is False and call["window_size"] == (5, 0)
    assert call["extra"] == {}


def test_causal_and_full_pass_no_window(standin):
    rec = standin()
    for mask in ("causal", "full"):
        FlashAttention3().forward(*_inputs(_cfg(mask)), _cfg(mask))
    assert [(c["causal"], c["window_size"]) for c in rec.calls] == [(True, (-1, -1)),
                                                                     (False, (-1, -1))]


@pytest.mark.parametrize("cfg,match", [
    (_cfg("sliding"), "needs cfg.window"),
    (_cfg("block_sparse", sparsity=0.5, block_size=64), "not wired"),
])
def test_unsupported_configs_raise_with_the_reason(cfg, match, standin):
    rec = standin()
    with pytest.raises(UnsupportedConfig, match=match):
        FlashAttention3().forward(*_inputs(cfg), cfg)
    assert rec.calls == []
