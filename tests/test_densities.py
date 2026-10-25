"""T2.20: mask density and executed density (`attnbench/densities.py`).

The executed density is taken from the tensor handed to the kernel, so these
tests drive real call sites (`kernels.bsa_prefill`, the XAttention backend's
forward) against a fake `block_sparse_attn` module and check that a union
inserted between the arm's mask and the kernel shows up as a difference.
CPU-only.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from attnbench import densities, kernels
from attnbench.backends.xattention import density_tensor
from attnbench.densities import (DENSE_FIELDS, KernelInputRecorder,
                                 block_density, density_problems)


def test_block_density_counts_causal_blocks_over_heads():
    m = torch.zeros(2, 4, 4, dtype=torch.bool)
    m[0] = torch.ones(4, 4, dtype=torch.bool).tril()     # head 0: all 10 valid
    m[1, :, 0] = True                                    # head 1: the sink column, 4
    assert float(block_density(m)) == pytest.approx(14 / 20)


def test_blocks_above_the_diagonal_are_not_counted():
    m = torch.ones(1, 4, 4, dtype=torch.bool)
    assert float(block_density(m)) == 1.0
    assert float(block_density(m, causal=False)) == 1.0
    upper = torch.ones(4, 4, dtype=torch.bool).triu(1).unsqueeze(0)
    assert float(block_density(upper)) == 0.0


def test_it_matches_the_xattention_rows_convention():
    """`realised_density` stays as it is; the new helper must agree with it
    so mask_density and realised_density are one number on XAttention rows."""
    g = torch.Generator().manual_seed(0)
    m = torch.rand(1, 3, 5, 5, generator=g) > 0.5
    assert float(block_density(m)) == pytest.approx(float(density_tensor(m, 5, 5)))


def test_a_query_chunk_aligns_to_the_last_key_block():
    m = torch.ones(1, 2, 4, dtype=torch.bool)            # 2 query blocks, 4 key blocks
    valid = torch.ones(2, 4, dtype=torch.bool).tril(diagonal=2)
    assert float(block_density(m)) == 1.0
    assert int(valid.sum()) == 7


def test_non_bool_masks_are_refused():
    with pytest.raises(ValueError):
        block_density(torch.ones(4, 4))


# --- the recorder at the kernel boundary ---------------------------------------

@pytest.fixture
def fake_bsa(monkeypatch):
    calls = []

    def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming,
                               base_blockmask, sq, sk, **kw):
        calls.append(base_blockmask)
        return q.clone()

    mod = types.ModuleType("block_sparse_attn")
    mod.block_sparse_attn_func = block_sparse_attn_func
    monkeypatch.setitem(sys.modules, "block_sparse_attn", mod)
    return mod, calls


def _qkv(h=2, s=512, d=8):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(1, h, s, d, generator=g) for _ in range(3)]


def _per_head_mask(h=2, nb=4):
    m = torch.zeros(1, h, nb, nb, dtype=torch.bool)
    m[:, :, :, 0] = True
    m[:, :, torch.arange(nb), torch.arange(nb)] = True   # sink and diagonal
    m[0, 0, 3, 1] = True                                  # one chosen block, head 0
    return m


def test_executed_equals_mask_when_nothing_unions(fake_bsa):
    q, k, v = _qkv()
    mask = _per_head_mask()
    with KernelInputRecorder() as rec:
        kernels.bsa_prefill(q, k, v, mask)
    mask_density = float(block_density(mask))
    assert rec.fields(mask_density=mask_density) == {
        "mask_density": mask_density, "executed_density": pytest.approx(mask_density)}


def test_a_union_before_the_kernel_shows_up(fake_bsa, monkeypatch):
    """The point of the field: a GQA-style union of the two heads' masks,
    inserted between the arm and the kernel, must make executed differ from
    mask. A formula over the arm's mask could not see it."""
    q, k, v = _qkv()
    mask = _per_head_mask()
    union = mask.any(dim=1, keepdim=True).expand_as(mask).contiguous()
    real = kernels.bsa_prefill
    monkeypatch.setattr(kernels, "bsa_prefill",
                        lambda q, k, v, m, **kw: real(q, k, v, union, **kw))
    with KernelInputRecorder() as rec:
        kernels.bsa_prefill(q, k, v, mask)
    assert rec.executed_density() == pytest.approx(float(block_density(union)))
    assert rec.executed_density() > float(block_density(mask))


def test_the_recorder_sees_the_xattention_backends_call(fake_bsa, monkeypatch):
    """The authors' call is not edited; the recorder must still see it. The
    fake estimate sets query block 0's whole row (keep_sink), which the
    kernel masks again under is_causal, so executed counts only the sink."""
    from attnbench.backends.xattention import XAttentionBackend
    from attnbench.config import AttnConfig
    nb = 4

    def xattn_estimate(q, k, **kw):
        m = torch.zeros(1, k.shape[1], nb, nb, dtype=torch.bool)
        m[:, :, :, 0] = True
        m[:, :, 0, :] = True
        return torch.zeros(1), m

    for name in ("xattn", "xattn.src"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    est = types.ModuleType("xattn.src.Xattention")
    est.xattn_estimate = xattn_estimate
    monkeypatch.setitem(sys.modules, "xattn.src.Xattention", est)
    q, k, v = _qkv(h=2, s=nb * 128)
    cfg = AttnConfig(seq_len=nb * 128, batch=1, n_heads_q=2, n_heads_kv=2,
                     head_dim=8, dtype="bfloat16", mask="causal")
    with KernelInputRecorder() as rec:
        XAttentionBackend(threshold=0.9).forward(q, k, v, cfg)
    assert len(rec.per_call) == 1
    assert rec.executed_density() == pytest.approx(4 / 10)


def test_the_kernel_is_restored_afterwards(fake_bsa):
    mod, _ = fake_bsa
    original = mod.block_sparse_attn_func
    with KernelInputRecorder():
        assert mod.block_sparse_attn_func is not original
    assert mod.block_sparse_attn_func is original


def test_the_kernel_is_restored_after_an_exception(fake_bsa):
    mod, _ = fake_bsa
    original = mod.block_sparse_attn_func
    with pytest.raises(RuntimeError):
        with KernelInputRecorder():
            raise RuntimeError("boom")
    assert mod.block_sparse_attn_func is original


def test_no_kernel_call_means_no_executed_density(fake_bsa):
    with KernelInputRecorder() as rec:
        pass
    with pytest.raises(RuntimeError, match="no block-sparse kernel call"):
        rec.executed_density()


def test_a_head_mask_type_other_than_all_ones_is_refused(fake_bsa):
    mod, _ = fake_bsa
    q = torch.zeros(4, 2, 8)
    mask = _per_head_mask()
    with KernelInputRecorder() as rec:
        mod.block_sparse_attn_func(q, q, q, None, None,
                                   torch.tensor([1, 0], dtype=torch.int32), None,
                                   mask, 512, 512, is_causal=True)
    with pytest.raises(ValueError, match="head_mask_type"):
        rec.executed_density()


def test_several_calls_are_averaged(fake_bsa):
    q, k, v = _qkv()
    full = torch.ones(1, 2, 4, 4, dtype=torch.bool).tril().contiguous()
    with KernelInputRecorder() as rec:
        kernels.bsa_prefill(q, k, v, _per_head_mask())
        kernels.bsa_prefill(q, k, v, full)
    expect = (float(block_density(_per_head_mask())) + 1.0) / 2
    assert rec.executed_density() == pytest.approx(expect)


# --- row checks -----------------------------------------------------------------

def test_dense_rows():
    assert density_problems(DENSE_FIELDS, dense=True) == []
    assert density_problems({"mask_density": 0.5, "executed_density": 1.0}, dense=True)
    assert density_problems({"mask_density": None, "executed_density": 0.9}, dense=True)


def test_sparse_rows():
    assert density_problems({"mask_density": 0.3, "executed_density": 0.35}, dense=False) == []
    assert density_problems({"mask_density": None, "executed_density": 0.35}, dense=False)
    assert density_problems({"mask_density": 0.3}, dense=False)
    assert density_problems({"mask_density": 0.0, "executed_density": 0.35}, dense=False)
    assert density_problems({"mask_density": float("nan"),
                             "executed_density": 0.35}, dense=False)


def test_the_module_names_the_kernel_every_call_site_imports():
    """The recorder patches one module attribute; if a call site imported the
    kernel under another name or at module load, it would miss that site."""
    from pathlib import Path
    root = Path(densities.__file__).resolve().parent
    for rel in ("kernels.py", "backends/xattention.py", "backends/block_sparse.py"):
        src = (root / rel).read_text()
        assert f"from {densities.KERNEL_MODULE} import {densities.KERNEL_NAME}" in src, rel
        top = [l for l in src.splitlines()
               if l.startswith(f"from {densities.KERNEL_MODULE}")]
        assert not top, f"{rel} imports the kernel at module load: {top}"
