"""T2.9's HDBI diagnostic and T2.10's analytic arithmetic intensities.

Hand-computed numbers are worked in the comments: HDBI from TaxBreak Eq. 3
as decided (sum d / (sum d + sum h)), intensities from a toy shape whose
FLOPs and bytes can be counted on paper. The attention FLOPs are also
checked against the closed form 2 B h_q L^2 hd density, written out here
rather than taken from `config.AttnConfig`.
"""

from __future__ import annotations

import pytest

from attnbench.predict.hdbi import (BALANCED, DEVICE_BOUND, HOST_BOUND,
                                    classify, hdbi)
from attnbench.predict.intensity import (KERNEL_CLASSES, ModelShape,
                                         arithmetic_intensity, flops_and_bytes)
from attnbench.predict.recursion import (CostModelError, close_prefill,
                                         copy_op, kernel_op, price, sync_op)

from test_predict_recursion import COSTS


# --------------------------------------------------------------------------
# HDBI
# --------------------------------------------------------------------------

def test_hdbi_is_device_over_device_plus_host():
    # sum d = 2 + 1 = 3, sum h = 0.5 + 0.5 = 1: HDBI = 3 / 4.
    p = price([kernel_op(0.5, 2), kernel_op(0.5, 1)], COSTS)
    assert hdbi(p) == 0.75


def test_link_and_s0_are_outside_hdbi():
    # Adding a copy with l = 5 (and no host cost) and two synchronisations
    # changes neither sum d nor sum h, so HDBI stays 3/4.
    p = price(close_prefill([kernel_op(0.5, 2), copy_op(0, 5),
                             sync_op("item", payload_bytes=4),
                             kernel_op(0.5, 1)]), COSTS)
    assert sum(p.link) == 5 and sum(p.sync_cost) == 3
    assert hdbi(p) == 0.75


def test_hdbi_limits_and_refusal():
    assert hdbi(price([kernel_op(0, 4)], COSTS)) == 1.0
    assert hdbi(price([kernel_op(4, 0)], COSTS)) == 0.0
    with pytest.raises(CostModelError, match="undefined"):
        hdbi(price([kernel_op(0, 0)], COSTS))


@pytest.mark.parametrize("value,expected", [
    (0.15, HOST_BOUND), (0.29, HOST_BOUND), (0.3, BALANCED),
    (0.5, BALANCED), (0.7, BALANCED), (0.71, DEVICE_BOUND),
    (0.93, DEVICE_BOUND)])
def test_classify_uses_the_callers_thresholds(value, expected):
    assert classify(value, host_bound_below=0.3,
                    device_bound_above=0.7) == expected


def test_classify_has_no_default_thresholds():
    # TaxBreak names the two limits but no cut-offs between them.
    with pytest.raises(TypeError):
        classify(0.5)
    with pytest.raises(CostModelError, match="must not exceed"):
        classify(0.5, host_bound_below=0.8, device_bound_above=0.2)
    with pytest.raises(CostModelError, match=r"in \[0, 1\]"):
        classify(1.2, host_bound_below=0.3, device_bound_above=0.7)


# --------------------------------------------------------------------------
# Arithmetic intensity
# --------------------------------------------------------------------------

# d = 4, h_q = 2, h_kv = 1, hd = 2, I = 8, V = 10, 2 bytes per element.
TOY = ModelShape(hidden=4, n_heads_q=2, n_heads_kv=1, head_dim=2,
                 intermediate=8, vocab=10)


def test_flops_and_bytes_hand_computed():
    # L = 4, B = 1: T = 4.  qkv width (2 + 2*1) * 2 = 8.
    fb = flops_and_bytes(TOY, context_length=4, batch=1, density=1.0)
    # qkv: 2*4*4*8 = 256;  2*(4*8 + 4*4 + 4*8) = 160.
    assert fb["qkv_projection"] == (256, 160)
    # attention: 4*1*2*4*4*2 = 256, halved for causal = 128;
    #            bytes 2 * 4 * 2 * (2*2 + 2*1) = 96.
    assert fb["attention"] == (128, 96)
    # output: 2*4*2*2*4 = 128;  2*(2*2*4 + 4*2*2 + 4*4) = 96.
    assert fb["output_projection"] == (128, 96)
    # mlp: 3 * 2*4*4*8 + 2*4*8 = 768 + 64 = 832;  2*(3*4*8 + 3*4*4 + 3*4*8) = 480.
    assert fb["mlp"] == (832, 480)
    # norm: 4*4*4 = 64;  2*(2*4*4 + 4) = 72.
    assert fb["norm"] == (64, 72)
    # lm_head, last position only: 2*1*4*10 = 80;  2*(4*10 + 1*4 + 1*10) = 108.
    assert fb["lm_head"] == (80, 108)


def test_density_scales_only_the_attention_flops():
    # density 0.5: attention 128 * 0.5 = 64 FLOPs over the same 96 bytes.
    dense = flops_and_bytes(TOY, 4, 1, 1.0)
    half = flops_and_bytes(TOY, 4, 1, 0.5)
    assert half["attention"] == (64, 96)
    assert {c: half[c] for c in KERNEL_CLASSES if c != "attention"} == \
        {c: dense[c] for c in KERNEL_CLASSES if c != "attention"}
    ai = arithmetic_intensity(TOY, 4, 1, 0.5)
    assert ai["attention"] == pytest.approx(2 / 3)
    assert ai["qkv_projection"] == pytest.approx(1.6)


@pytest.mark.parametrize("L,B,density", [(1024, 1, 1.0), (2048, 4, 0.25),
                                         (4096, 2, 0.5)])
def test_attention_flops_match_the_closed_form(L, B, density):
    # 2 B h_q L^2 hd density: QK^T and PV at 2 FLOPs per multiply-add,
    # causal half, kept blocks only.
    shape = ModelShape(hidden=2048, n_heads_q=16, n_heads_kv=4, head_dim=128,
                       intermediate=8192, vocab=32000)
    flops, nbytes = flops_and_bytes(shape, L, B, density)["attention"]
    assert flops == pytest.approx(2 * B * 16 * L * L * 128 * density, rel=1e-12)
    assert nbytes == 2 * B * L * 128 * (2 * 16 + 2 * 4)


def test_dense_attention_intensity_doubles_with_context():
    # AI = 2 h_q L / (s (2 h_q + 2 h_kv)): linear in L.
    a = arithmetic_intensity(TOY, 64, 1, 1.0)["attention"]
    b = arithmetic_intensity(TOY, 128, 1, 1.0)["attention"]
    assert b == pytest.approx(2 * a)
    assert a == pytest.approx(2 * 2 * 64 / (2 * (4 + 2)))


@pytest.mark.parametrize("kw", [dict(hidden=0), dict(n_heads_kv=3),
                                dict(itemsize=True), dict(vocab=2.5)])
def test_model_shape_refusals(kw):
    base = dict(hidden=4, n_heads_q=2, n_heads_kv=1, head_dim=2,
                intermediate=8, vocab=10)
    with pytest.raises(CostModelError):
        ModelShape(**{**base, **kw})


@pytest.mark.parametrize("L,B,density", [(0, 1, 1.0), (4, 0, 1.0),
                                         (4, 1, 0.0), (4, 1, 1.5)])
def test_cell_refusals(L, B, density):
    with pytest.raises(CostModelError):
        flops_and_bytes(TOY, L, B, density)
