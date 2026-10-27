"""T2.17: the asserts-removed XAttention arm and its bitwise gate.

On the CPU, a stand-in `xattn` package is written to disk with upstream's
layout (a namespace `src/`, `Xattention.py` star-importing `utils.py`) and
the one construct that matters: a tensor assert inside `try`/`except` that
repairs the mask when the assert fails, as `find_blocks_chunked` does at
commit `e379887`. `block_sparse_attn` is a stand-in whose output depends on
the mask. These tests are skipped where the real package is installed; the
CUDA test at the end is the plan's acceptance gate ("gate passes on the L4").
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
import types

import pytest
import torch

from attnbench import assert_strip as A
from attnbench.backends import get
from attnbench.backends.xattention import XAttentionBackend
from attnbench.backends.xattention_noassert import (AssertGate, GateRecord,
                                                    XAttentionNoAssertBackend)
from attnbench.config import AttnConfig

_REAL_XATTN = importlib.util.find_spec("xattn") is not None

UTILS = '''
import torch

def find_blocks_chunked(scores, threshold, causal):
    assert threshold is None or threshold > 0
    top = scores.amax(dim=-1, keepdim=True)
    mask = scores >= top - (1 - threshold) * top.abs()
    mask[..., 0] = True
    try:
        if causal:
            assert (~mask.triu(1)).all()
    except:
        mask = mask.tril()
    if causal:
        assert mask[..., 0].all()
    return mask
'''

XATTENTION = '''
from xattn.src.utils import *
import torch
import torch.nn.functional as F

def xattn_estimate(q, k, *, block_size, stride, norm, threshold, select_mode,
                   use_triton, causal, chunk_size, kdb, keep_sink, keep_recent):
    assert q.shape[1] == k.shape[1]
    _, h, s, d = q.shape
    nb = -(-s // block_size)
    pad = nb * block_size - s
    qb = F.pad(q, (0, 0, 0, pad)).view(1, h, nb, block_size, d).mean(3)
    kb = F.pad(k, (0, 0, 0, pad)).view(1, h, nb, block_size, d).mean(3)
    scores = qb @ kb.transpose(-1, -2)
    if threshold >= 0.5:
        scores = scores.masked_fill(torch.ones(nb, nb, dtype=torch.bool).triu(1),
                                    float("-inf"))
    return scores, find_blocks_chunked(scores, threshold, causal)
'''

S, H, D = 300, 4, 16


def _cfg():
    return AttnConfig(seq_len=S, batch=1, n_heads_q=H, n_heads_kv=2, head_dim=D,
                      dtype="bfloat16", mask="causal")


def _qkv(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, H, S, D, generator=g), torch.randn(1, 2, S, D, generator=g),
            torch.randn(1, 2, S, D, generator=g))


@pytest.fixture
def fake_xattn(tmp_path, monkeypatch):
    if _REAL_XATTN:
        pytest.skip("the real x-attention is installed; the CUDA gate covers it")
    root = tmp_path / "site"
    (root / "xattn" / "src").mkdir(parents=True)
    (root / "xattn" / "__init__.py").write_text("")
    (root / "xattn" / "src" / "utils.py").write_text(UTILS)
    (root / "xattn" / "src" / "Xattention.py").write_text(XATTENTION)
    monkeypatch.syspath_prepend(str(root))
    bsa = types.ModuleType("block_sparse_attn")

    def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming, mask,
                               sq, sk, p_dropout, deterministic, is_causal):
        return v * (1 + mask.float().sum())

    bsa.block_sparse_attn_func = block_sparse_attn_func
    monkeypatch.setitem(sys.modules, "block_sparse_attn", bsa)
    yield root
    A.uninstall()
    for n in [n for n in sys.modules if n == "xattn" or n.startswith("xattn.")]:
        del sys.modules[n]


# --- the arm --------------------------------------------------------------------

def test_registered_under_its_own_name_with_the_parents_capability():
    assert get("xattention_noassert") is XAttentionNoAssertBackend
    a, b = XAttentionBackend.capability, XAttentionNoAssertBackend.capability
    assert b.name == "xattention_noassert"
    assert b.notes.startswith(a.notes) and "optimize=1" in b.notes
    for field in ("family", "min_compute_capability", "supports_gqa", "supports_causal",
                  "supports_block_sparse", "block_sizes", "head_dims", "dtypes"):
        assert getattr(a, field) == getattr(b, field), field


def _estimate_tree(cls):
    fn = ast.parse(textwrap.dedent(inspect.getsource(cls.estimate))).body[0]
    fn.body = [s for s in fn.body if not (isinstance(s, ast.Expr)
                                          and isinstance(s.value, ast.Constant))]
    return fn


def test_estimate_is_the_parents_apart_from_where_it_imports_from():
    parent, child = _estimate_tree(XAttentionBackend), _estimate_tree(XAttentionNoAssertBackend)
    install, child.body = child.body[0], child.body[1:]
    assert ast.unparse(install) == "assert_strip.install()"
    imp = child.body[0]
    assert isinstance(imp, ast.ImportFrom) and imp.module == f"{A.ALIAS}.src.Xattention"
    imp.module = "xattn.src.Xattention"
    assert ast.dump(parent) == ast.dump(child)


def test_the_stripped_estimate_runs_the_stripped_package(fake_xattn):
    q, k, _ = _qkv(0)
    k_rep = k.repeat_interleave(2, dim=1)
    XAttentionBackend(threshold=0.9).estimate(q, k_rep)
    XAttentionNoAssertBackend(threshold=0.9).estimate(q, k_rep)
    shipped = sys.modules["xattn.src.utils"].find_blocks_chunked
    stripped = sys.modules[f"{A.ALIAS}.src.utils"].find_blocks_chunked
    assert (A.assertion_ops(shipped.__code__), A.assertion_ops(stripped.__code__)) == (3, 0)
    chk = A.session_check()
    assert chk.stripped_modules == ("xattn", "xattn.src.Xattention", "xattn.src.utils")
    assert chk.asserts_removed == chk.shipped_asserts == 4


@pytest.mark.parametrize("seed", range(5))
def test_masks_agree_where_every_assert_holds(fake_xattn, seed):
    q, k, _ = _qkv(seed)
    k_rep = k.repeat_interleave(2, dim=1)
    a = XAttentionBackend(threshold=0.9).estimate(q, k_rep)
    b = XAttentionNoAssertBackend(threshold=0.9).estimate(q, k_rep)
    assert torch.equal(a, b)


def test_masks_differ_where_the_assert_in_try_would_fail(fake_xattn):
    q, k, _ = _qkv(0)
    k_rep = k.repeat_interleave(2, dim=1)
    a = XAttentionBackend(threshold=0.3).estimate(q, k_rep)
    b = XAttentionNoAssertBackend(threshold=0.3).estimate(q, k_rep)
    assert not a.triu(1).any()       # the shipped arm repaired the mask
    assert b.triu(1).any()           # the stripped arm could not


# --- the gate -------------------------------------------------------------------

def _run_gate(threshold, layers=3):
    gate = AssertGate(XAttentionBackend(threshold=threshold),
                      XAttentionNoAssertBackend(threshold=threshold))
    outs = []
    for layer in range(layers):
        gate.set_layer(layer)
        q, k, v = _qkv(layer)
        outs.append((gate.forward(q, k, v, _cfg()), (q, k, v)))
    return gate, outs


def test_the_gate_passes_when_masks_and_outputs_agree(fake_xattn):
    gate, outs = _run_gate(0.9)
    assert gate.passed and gate.failures() == []
    assert gate.records == [GateRecord(i, True, True) for i in range(3)]
    ref = XAttentionBackend(threshold=0.9)
    for out, (q, k, v) in outs:
        assert torch.equal(out, ref.forward(q, k, v, _cfg()))   # the shipped output
    assert gate.stripped.layer_idx == gate.shipped.layer_idx == 2
    assert len(gate.last_layer_density) == 3


def test_the_gate_fails_on_the_divergent_input(fake_xattn):
    gate, outs = _run_gate(0.3)
    assert not gate.passed
    assert gate.failures() == [GateRecord(i, False, False) for i in range(3)]
    ref = XAttentionBackend(threshold=0.3)
    for out, (q, k, v) in outs:
        assert torch.equal(out, ref.forward(q, k, v, _cfg()))   # still the shipped output


def test_a_gate_with_no_records_has_not_passed():
    gate = AssertGate(XAttentionBackend(), XAttentionNoAssertBackend())
    assert not gate.passed


def test_the_gate_pairs_one_shipped_and_one_stripped_arm_with_equal_settings():
    with pytest.raises(TypeError, match="pairs the shipped arm"):
        AssertGate(XAttentionBackend(), XAttentionBackend())
    with pytest.raises(TypeError, match="pairs the shipped arm"):
        AssertGate(XAttentionNoAssertBackend(), XAttentionNoAssertBackend())
    with pytest.raises(ValueError, match="identical settings"):
        AssertGate(XAttentionBackend(threshold=0.9), XAttentionNoAssertBackend(threshold=0.8))
    with pytest.raises(ValueError, match="identical settings"):
        AssertGate(XAttentionBackend(stride=8), XAttentionNoAssertBackend(stride=16))


# --- on the GPU -----------------------------------------------------------------

_CUDA = torch.cuda.is_available() and _REAL_XATTN


@pytest.mark.skipif(not _CUDA, reason="needs CUDA + x-attention + block-sparse-attn")
@pytest.mark.parametrize("seq_len", [2048, 4096])
@pytest.mark.parametrize("threshold", [0.9, 0.95])
def test_bitwise_gate_on_cuda(seq_len, threshold):
    pytest.importorskip("block_sparse_attn")
    cfg = AttnConfig(seq_len=seq_len, batch=1, n_heads_q=12, n_heads_kv=2, head_dim=128,
                     dtype="bfloat16", mask="causal")
    gate = AssertGate(XAttentionBackend(threshold=threshold),
                      XAttentionNoAssertBackend(threshold=threshold))
    g = torch.Generator(device="cuda").manual_seed(seq_len)
    for layer in range(2):
        gate.set_layer(layer)
        q = torch.randn(1, 12, seq_len, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, 2, seq_len, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 2, seq_len, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        gate.forward(q, k, v, cfg)
    assert gate.passed, gate.failures()
    A.session_check()
