"""T2.4: the estimator's masks, captured at the kernel and banked
(`components.mask_bank`).

The kernel is a stand-in module with `block_sparse_attn_func` in the real
argument order (`kernels.py`), so the capture's patching, positional and
keyword arguments, substitution and restore are exercised as on the GPU.
Density is checked against `backends.xattention.density_tensor`, the
convention it claims to share.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from attnbench.backends.xattention import density_tensor
from attnbench.components import mask_bank as B
from attnbench.components.mask_bank import MaskBankError, MaskCapture
from attnbench.components.op_sequence import CodePath

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch",
              libraries=(("torch", "2.9.1"),))


def fake_kernel():
    """A module whose kernel records the mask it was handed."""
    mod = types.SimpleNamespace(seen=[])

    def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming_info,
                               base_blockmask, *rest, is_causal=False, **kw):
        mod.seen.append(base_blockmask.clone())
        return q
    mod.block_sparse_attn_func = block_sparse_attn_func
    return mod


def call(mod, mask, *, kw=False, hmt=None, causal=True):
    q = torch.zeros(2, 2)
    hmt = torch.ones(2, dtype=torch.int32) if hmt is None else hmt
    if kw:
        return mod.block_sparse_attn_func(q, q, q, None, None, head_mask_type=hmt,
                                          streaming_info=None, base_blockmask=mask,
                                          is_causal=causal)
    return mod.block_sparse_attn_func(q, q, q, None, None, hmt, None, mask, is_causal=causal)


def masks(n, shape=(1, 2, 3, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(shape, generator=g) > 0.5 for _ in range(n)]


def test_capture_sees_every_call_and_restores_the_kernel():
    mod = fake_kernel()
    original = mod.block_sparse_attn_func
    ms = masks(3)
    with MaskCapture(module=mod) as cap:
        call(mod, ms[0])
        call(mod, ms[1], kw=True)
        call(mod, ms[2], causal=False)
    assert mod.block_sparse_attn_func is original
    assert all(a is b for a, b in zip(cap.masks(), ms))      # references, no copy
    assert [c.causal for c in cap.calls] == [True, True, False]
    with pytest.raises(ValueError):
        with MaskCapture(module=mod):
            raise ValueError
    assert mod.block_sparse_attn_func is original


def test_copy_keeps_the_mask_as_it_was_at_the_call():
    mod = fake_kernel()
    buf = torch.zeros(1, 2, 3, 3, dtype=torch.bool)
    ms = masks(2)
    with MaskCapture(module=mod) as refs, MaskCapture(module=mod, copy=True) as copies:
        for m in ms:                     # one buffer reused across layers
            buf.copy_(m)
            call(mod, buf)
    assert all(torch.equal(r, ms[1]) for r in refs.masks())
    assert [torch.equal(c, m) for c, m in zip(copies.masks(), ms)] == [True, True]


def test_substitution_hands_the_kernel_the_banked_mask_and_keeps_the_live_one():
    mod = fake_kernel()
    live, banked = masks(2, seed=1), masks(2, seed=2)
    with MaskCapture(module=mod, substitute=banked) as cap:
        for m in live:
            call(mod, m)
    assert [torch.equal(s, b) for s, b in zip(mod.seen, banked)] == [True, True]
    assert all(a is b for a, b in zip(cap.live_masks(), live))
    assert all(a is b for a, b in zip(cap.masks(), banked))
    mod.seen.clear()
    with MaskCapture(module=mod, substitute=banked):
        call(mod, live[0], kw=True)
    assert torch.equal(mod.seen[0], banked[0])


def test_substitution_refuses_extra_calls_and_mismatched_masks():
    mod = fake_kernel()
    with pytest.raises(MaskBankError, match="no banked mask"):
        with MaskCapture(module=mod, substitute=masks(1)):
            call(mod, masks(1)[0])
            call(mod, masks(1)[0])
    with pytest.raises(MaskBankError, match=r"banked mask \(1, 2, 4, 4\)"):
        with MaskCapture(module=mod, substitute=masks(1, shape=(1, 2, 4, 4))):
            call(mod, masks(1)[0])
    with pytest.raises(MaskBankError, match="torch.uint8"):
        with MaskCapture(module=mod, substitute=[masks(1)[0].to(torch.uint8)]):
            call(mod, masks(1)[0])


def test_a_call_without_a_mask_is_refused():
    mod = fake_kernel()
    with pytest.raises(MaskBankError, match="without base_blockmask"):
        with MaskCapture(module=mod):
            mod.block_sparse_attn_func(1, 2, 3)


@pytest.mark.parametrize("nq,nk", [(4, 4), (3, 5), (7, 7)])
def test_density_is_the_xattention_convention(nq, nk):
    m = masks(1, shape=(1, 3, nq, nk), seed=nq * nk)[0]
    want = float(density_tensor(m, nq, nk))
    assert B.density(m.numpy(), causal=True) == pytest.approx(want, abs=1e-7)
    full = B.density(m.numpy(), causal=False)
    assert full == pytest.approx(float(m.float().mean()), abs=1e-7)


def record_of(ms, **kw):
    mod = fake_kernel()
    with MaskCapture(module=mod) as cap:
        for m in ms:
            call(mod, m, **kw)
    return B.from_capture(cap, code_path=CP, context_length=4096, batch=1, prompt_id="p0")


def test_the_bank_round_trips_bitwise(tmp_path):
    ms = masks(3, shape=(1, 3, 5, 7))             # 105 bits: not a whole byte count
    rec = record_of(ms)
    npz, man = B.save(rec, tmp_path / "masks")
    assert (npz.name, man.name) == ("masks.npz", "masks.json")
    back = B.load(tmp_path / "masks")
    assert back == rec and back.sha256 == rec.sha256
    assert [torch.equal(a, b) for a, b in zip(back.tensors(), ms)] == [True] * 3
    assert back.code_path == CP and back.prompt_id == "p0"
    with pytest.raises(MaskBankError, match="never overwritten"):
        B.save(rec, tmp_path / "masks")


def test_a_tampered_bank_is_refused(tmp_path):
    rec = record_of(masks(2))
    npz, man = B.save(rec, tmp_path / "masks")
    raw = bytearray(npz.read_bytes())
    raw[-30] ^= 0xFF
    npz.write_bytes(bytes(raw))
    with pytest.raises(MaskBankError, match="does not match its manifest"):
        B.load(tmp_path / "masks")


def test_a_manifest_naming_another_mask_is_refused(tmp_path):
    import json
    rec = record_of(masks(2))
    _, man = B.save(rec, tmp_path / "masks")
    meta = json.loads(man.read_text())
    meta["calls"][1]["sha256"] = "0" * 64
    meta["calls_sha256"] = B.calls_sha256(meta["calls"])    # the per-call check still fires
    man.write_text(json.dumps(meta))
    with pytest.raises(MaskBankError, match="call 1 does not match"):
        B.load(tmp_path / "masks")


def test_only_bool_masks_with_every_head_on_its_own_mask_are_banked():
    mod = fake_kernel()
    with MaskCapture(module=mod) as cap:
        call(mod, masks(1)[0], hmt=torch.tensor([1, 0], dtype=torch.int32))
    with pytest.raises(MaskBankError, match="head_mask_type"):
        B.from_capture(cap, code_path=CP, context_length=1, batch=1, prompt_id="p")
    with MaskCapture(module=mod) as cap:
        call(mod, masks(1)[0].to(torch.uint8))
    with pytest.raises(MaskBankError, match="bool"):
        B.from_capture(cap, code_path=CP, context_length=1, batch=1, prompt_id="p")
    with MaskCapture(module=mod) as cap:
        pass
    with pytest.raises(MaskBankError, match="no block-sparse kernel call"):
        B.from_capture(cap, code_path=CP, context_length=1, batch=1, prompt_id="p")


def test_agreement_is_per_call_and_bitwise():
    a = masks(2)
    b = [a[0].clone(), a[1].clone()]
    b[1][0, 0, 0, 0] = ~b[1][0, 0, 0, 0]
    r = B.agreement(a, b)
    assert r["identical"] is False
    assert r["per_call_agreement"][0] == 1.0
    assert r["per_call_agreement"][1] == pytest.approx(1 - 1 / 18)
    assert B.agreement(a, a)["identical"] is True
    with pytest.raises(MaskBankError):
        B.agreement(a, a[:1])
    with pytest.raises(MaskBankError, match="shapes"):
        B.agreement(a[:1], masks(1, shape=(1, 2, 4, 4)))


def test_the_packed_form_is_np_packbits():
    m = masks(1, shape=(1, 1, 3, 3))[0]
    rec = record_of([m])
    assert np.array_equal(rec.masks[0].packed, np.packbits(m.numpy().reshape(-1)))
