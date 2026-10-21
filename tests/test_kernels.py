"""`bsa_prefill`, the frontier arms' one kernel call site (pre-registration
sec. 4.4). The library is replaced by a recorder here; gate G1b needs CUDA."""
from __future__ import annotations

import pytest
import torch

from attnbench import kernels, masks


def _inputs(h=3, s=300, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(1, h, s, d, generator=g) for _ in range(3))
    nb = -(-s // 128)
    m = masks.importance_block_mask_per_head(s, 128, 0.5, torch.rand(h, nb, nb, generator=g),
                                             causal=True, identity_seed="k").to_bsa().contiguous()
    return q, k, v, m


def test_the_call_carries_the_arguments_section_4_4_fixes():
    q, k, v, m = _inputs()
    seen = {}

    def kernel(*args, **kw):
        seen["args"], seen["kw"] = args, kw
        return args[0].clone()

    out = kernels.bsa_prefill(q, k, v, m, kernel=kernel)
    qf, kf, vf, cu_q, cu_k, head_mask_type, streaming, mask, max_q, max_k = seen["args"]
    assert seen["kw"] == dict(p_dropout=0.0, deterministic=True, is_causal=True,
                              exact_streaming=False)
    assert streaming is None and mask is m and (max_q, max_k) == (300, 300)
    assert head_mask_type.tolist() == [1, 1, 1] and head_mask_type.dtype == torch.int32
    assert cu_q.tolist() == cu_k.tolist() == [0, 300]
    assert qf.shape == (300, 3, 8) and torch.equal(qf, q[0].transpose(0, 1))
    assert torch.equal(kf, k[0].transpose(0, 1)) and torch.equal(vf, v[0].transpose(0, 1))
    assert out.shape == q.shape and torch.equal(out, q)        # the layout round-trips


@pytest.mark.parametrize("break_it", ["kv_not_repeated", "mask_int", "mask_shape", "batch",
                                      "mask_not_contiguous"])
def test_bad_inputs_are_refused_before_the_kernel_is_called(break_it):
    q, k, v, m = _inputs()
    if break_it == "kv_not_repeated":
        k, v = k[:, :1], v[:, :1]
    elif break_it == "mask_int":
        m = m.int()
    elif break_it == "mask_shape":
        m = m[:, :1]
    elif break_it == "batch":
        q, k, v = (t.expand(2, -1, -1, -1) for t in (q, k, v))
    else:
        m = m.transpose(2, 3)
        assert not m.is_contiguous()

    def never(*a, **kw):
        raise AssertionError("the kernel was called")

    with pytest.raises(ValueError):
        kernels.bsa_prefill(q, k, v, m, kernel=never)


def test_the_reference_is_dense_causal_attention_under_a_full_mask():
    q, k, v, m = _inputs()
    full = torch.ones_like(m).tril().contiguous()
    want = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(kernels.reference_prefill(q, k, v, full), want, atol=1e-5)


def test_the_reference_masks_per_head():
    """A block one head drops changes that head's output and no other's."""
    q, k, v, _ = _inputs()
    full = torch.ones(1, 3, 3, 3, dtype=torch.bool).tril().contiguous()
    cut = full.clone()
    cut[0, 1, 2, 1] = False
    a, b = kernels.reference_prefill(q, k, v, full), kernels.reference_prefill(q, k, v, cut)
    assert torch.equal(a[0, 0], b[0, 0]) and torch.equal(a[0, 2], b[0, 2])
    assert torch.equal(a[0, 1, :256], b[0, 1, :256]) and not torch.equal(a[0, 1, 256:], b[0, 1, 256:])


def _cuda_kernel():
    if not torch.cuda.is_available():
        return False
    try:
        import block_sparse_attn  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _cuda_kernel(), reason="needs CUDA + block-sparse-attn")
def test_the_kernel_matches_the_reference_on_cuda():
    q, k, v, m = (t.cuda() for t in _inputs(h=4, s=1024, d=128))
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    out = kernels.bsa_prefill(q, k, v, m)
    ref = kernels.reference_prefill(q, k, v, m)
    assert torch.allclose(out.float(), ref.float(), atol=3e-2)
