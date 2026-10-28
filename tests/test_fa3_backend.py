"""The FA3 backend, tested on CPU with a stand-in for `flash_attn_interface`.

No test here needs an H100. The stand-in computes real attention in FA3's
(B, S, H, D) layout with the pinned source's causal semantics, so a backend
that forgets a transpose, flips the causal flag or unpacks the wrong element
returns the wrong numbers, not just the wrong shape. The expected output is
computed independently in (B, H, S, D) with an explicit mask.

The wheel is the restricted build (researcher's decision, 28 Oct 2026): head
dim 128, forward, bf16 and fp16. A restricted wheel given head dim 64 does not
fail; it runs its padded hdim-128 kernel (`hopper/flash_api.cpp` at the pin),
so the backend refuses first, and the tests feed it the dims it must refuse.

Each defect named in a test's docstring was introduced into `fa3.py` and
watched to fail here (28 Oct 2026).
"""

from __future__ import annotations

import dataclasses
import re
import sys
import types
from pathlib import Path

import pytest
import torch

from attnbench import dense_selection as ds
from attnbench.backends import all_backends, get
from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.fa3 import BUILD_FLAGS, BUILT_HEAD_DIMS, FlashAttention3
from attnbench.config import AttnConfig

REPO = Path(__file__).resolve().parent.parent


def _attend_bshd(q, k, v, causal):
    """Reference FA3 semantics in (B, S, H, D), equal Q and K lengths, GQA by
    head grouping."""
    B, S, Hq, D = q.shape
    g = Hq // k.shape[2]
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    i, j = torch.arange(S)[:, None], torch.arange(S)[None, :]
    keep = (j <= i) if causal else torch.ones(S, S, dtype=torch.bool)
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
        out = _attend_bshd(q, k, v, causal)
        if self.returns_tuple:
            return out, torch.zeros(q.shape[0], q.shape[2], q.shape[1])
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


def _cfg(mask="causal", window=None, hq=4, hkv=2, seq=16, d=128, **kw):
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
    keep = {"full": torch.ones(S, S, dtype=torch.bool), "causal": j <= i}[cfg.mask]
    s = (q @ k.transpose(-1, -2)) / cfg.head_dim ** 0.5
    return s.masked_fill(~keep, float("-inf")).softmax(-1) @ v


# --- registration, declarations and the build flags --------------------------------


def test_fa3_is_registered_as_the_restricted_hopper_kernel():
    assert all_backends()["fa3"] is FlashAttention3
    c = get("fa3").capability
    assert (c.family, c.min_compute_capability) == ("dense_exact", (9, 0))
    assert c.supports_gqa and not c.supports_sliding and not c.supports_backward
    assert c.head_dims == BUILT_HEAD_DIMS == (128,)
    assert c.dtypes == ("bfloat16", "float16")


def test_the_flags_remove_exactly_what_the_researcher_named():
    """Head dim 128, forward, bf16 and fp16: every other head dim, paged KV,
    append KV, split KV, softcap, local attention, FP8 and backward are off,
    plus the sm_80 kernels, and a source build is forced. fp16, PackGQA,
    varlen and clusters are not touched."""
    names = {f.split("=")[0].replace("FLASH_ATTENTION_", "") for f in BUILD_FLAGS}
    assert all(f.endswith("=TRUE") for f in BUILD_FLAGS)
    assert names == {"FORCE_BUILD", "DISABLE_BACKWARD", "DISABLE_FP8", "DISABLE_SM80",
                     "DISABLE_HDIM64", "DISABLE_HDIM96", "DISABLE_HDIM192",
                     "DISABLE_HDIM256", "DISABLE_PAGEDKV", "DISABLE_APPENDKV",
                     "DISABLE_SPLIT", "DISABLE_SOFTCAP", "DISABLE_LOCAL"}
    kept = {f"DISABLE_HDIM{d}" for d in BUILT_HEAD_DIMS}
    assert not names & kept


def test_the_build_script_takes_its_flags_from_the_backend():
    script = (REPO / "scripts" / "build_fa3_wheel.sh").read_text()
    assert "fa3_module fa3_wheel flags" in script
    assert not re.search(r"FLASH_ATTENTION_DISABLE_\w+=TRUE", script)


@pytest.mark.parametrize("cfg,reason", [
    (_cfg(pass_kind="fwd_bwd"), "no backward pass"),
    (_cfg(d=64), "head dim 64 unsupported"),
    (_cfg(d=256), "head dim 256 unsupported"),
    (_cfg(mask="sliding", window=4), "no sliding window"),
])
def test_the_capability_declines_what_the_wheel_does_not_have(cfg, reason):
    cfg = dataclasses.replace(cfg, dtype="bfloat16")
    assert FlashAttention3.claims_support(cfg) == (False, reason)


def test_bfloat16_forward_at_head_dim_128_is_claimed():
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
@pytest.mark.parametrize("cfg", [_cfg("causal"), _cfg("full"), _cfg("causal", hq=4, hkv=4)],
                         ids=["causal_gqa", "full_gqa", "causal_mha"])
def test_forward_matches_an_independent_reference(cfg, returns_tuple, standin):
    """Kills: a missing transpose in or out, the causal flag flipped or always
    False, and `_unpack` taking element -1 (the lse under the old shape)."""
    standin(returns_tuple)
    q, k, v = _inputs(cfg)
    out = FlashAttention3().forward(q, k, v, cfg)
    assert out.shape == q.shape
    torch.testing.assert_close(out, _expected_bhsd(q, k, v, cfg), rtol=1e-5, atol=1e-5)


def test_the_kernel_receives_bshd_tensors_and_no_window(standin):
    rec = standin()
    cfg = _cfg("causal", seq=12)
    FlashAttention3().forward(*_inputs(cfg), cfg)
    (call,) = rec.calls
    B, S, D = cfg.batch, cfg.seq_len, cfg.head_dim
    assert call["shapes"] == ((B, S, 4, D), (B, S, 2, D), (B, S, 2, D))
    assert call["causal"] is True and call["window_size"] == (-1, -1)
    assert call["extra"] == {}


@pytest.mark.parametrize("d", [32, 64, 96, 192, 256])
def test_a_head_dim_the_wheel_lacks_is_refused_before_the_kernel(d, standin):
    """Kills: dropping the head-dim check (the restricted kernel would run a
    padded hdim-128 instance and the stand-in would accept it)."""
    rec = standin()
    cfg = _cfg(d=d)
    with pytest.raises(UnsupportedConfig,
                       match=rf"fa3: head dim {d} is not in this wheel, which is built "
                             r"for head dim 128 only"):
        FlashAttention3().forward(*_inputs(cfg), cfg)
    assert rec.calls == []


@pytest.mark.parametrize("cfg,match", [
    (_cfg("sliding", window=3), "built with FLASH_ATTENTION_DISABLE_LOCAL"),
    (_cfg("block_sparse", sparsity=0.5, block_size=64), "mask block_sparse is not supported"),
])
def test_unsupported_masks_raise_with_the_reason(cfg, match, standin):
    """Kills: dropping the mask check (sliding would reach the kernel as full)."""
    rec = standin()
    with pytest.raises(UnsupportedConfig, match=match):
        FlashAttention3().forward(*_inputs(cfg), cfg)
    assert rec.calls == []
