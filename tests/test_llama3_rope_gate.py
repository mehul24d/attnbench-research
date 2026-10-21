"""Llama-3.1 RoPE (`rope_type: llama3`) through the swappable-attention
wrapper (estimator-frontier pre-registration, gate G9 and sec. 4.9, added
2026-10-03).

Llama-3.1-8B-Instruct's config.json (G9, read by the researcher on
2026-10-03) sets `max_position_embeddings` 131072 with `rope_scaling`
{rope_type llama3, factor 8.0, low_freq_factor 1.0, high_freq_factor 4.0,
original_max_position_embeddings 8192} and `rope_theta` 500000.0. The
existing logits gate (`test_swappable_attention_model.py::test_a_...`) runs a
toy Llama with no rope scaling, so it cannot see a wrapper that drops it.

This file checks three things:

1. The installed transformers builds the llama3 inverse frequencies, and
   they match Meta's published formula, implemented independently here.
2. On a tiny random Llama with the same rope_scaling (original context
   scaled down to 16) run to 64 positions, wrapped logits equal unwrapped
   logits.
3. Sensitivity: dropping the scaling moves the logits by far more than the
   gate's tolerance, so a wrapper that lost it would fail (2).

All of this ran on transformers 5.18.0 (the project venv) and 4.46.0 (the GPU
images' pin) on 2026-10-03.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.backends.impls import NaiveAttention  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

# G9 values, verbatim from the researcher's report.
G9_ROPE_SCALING = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                   "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
G9_ROPE_THETA = 500000.0
G9_MAX_POSITIONS = 131072

TOY_ORIGINAL = 16          # stands in for 8192
TOY_SEQ = 64               # 4x past it, as 32768 is 4x past 8192
TOY_HEAD_DIM = 8


def _meta_llama3_inv_freq(theta, head_dim, factor, low, high, original):
    """Meta's `apply_scaling` (llama-models, Llama 3.1), in float64."""
    out = []
    low_wavelen, high_wavelen = original / low, original / high
    for i in range(0, head_dim, 2):
        f = 1.0 / theta ** (i / head_dim)
        wavelen = 2 * math.pi / f
        if wavelen < high_wavelen:
            out.append(f)
        elif wavelen > low_wavelen:
            out.append(f / factor)
        else:
            smooth = (original / wavelen - low) / (high - low)
            out.append((1 - smooth) * f / factor + smooth * f)
    return torch.tensor(out, dtype=torch.float64)


def _toy_llama(rope_scaling):
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=TOY_HEAD_DIM, max_position_embeddings=128,
        rope_theta=G9_ROPE_THETA, rope_scaling=rope_scaling,
        attn_implementation="eager",
        # At the default 0.02 the toy's attention is nearly uniform and RoPE
        # barely moves the logits (6.7e-4 between scaled and unscaled); at
        # 0.2 the gap is about 5, so the sensitivity check below has margin.
        initializer_range=0.2,
    )
    return LlamaForCausalLM(cfg).eval()


def _toy_scaling():
    return dict(G9_ROPE_SCALING, original_max_position_embeddings=TOY_ORIGINAL)


def _attn_cfg():
    return AttnConfig(seq_len=TOY_SEQ, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=TOY_HEAD_DIM, dtype="float32", mask="causal")


def test_the_g9_config_builds_llama3_frequencies_matching_meta():
    """The real geometry (head_dim 4096/32 = 128) and the real rope values,
    on the installed transformers. Only the rotary module is built."""
    cfg = LlamaConfig(hidden_size=4096, num_attention_heads=32,
                      num_key_value_heads=8, num_hidden_layers=32,
                      max_position_embeddings=G9_MAX_POSITIONS,
                      rope_theta=G9_ROPE_THETA, rope_scaling=dict(G9_ROPE_SCALING))
    rotary = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding(config=cfg)
    assert rotary.rope_type == "llama3"
    want = _meta_llama3_inv_freq(G9_ROPE_THETA, 128, 8.0, 1.0, 4.0, 8192)
    got = rotary.inv_freq.double()
    assert torch.allclose(got, want, rtol=1e-6, atol=0), (got - want).abs().max()
    unscaled = _meta_llama3_inv_freq(G9_ROPE_THETA, 128, 1.0, 1.0, 4.0, 8192)
    assert (got != unscaled).sum() > 0


def test_the_toy_config_rescales_every_frequency():
    """The toy must exercise the scaling, not sit in the band it leaves
    alone: at original 16, every one of the 4 frequencies is changed."""
    rotary = _toy_llama(_toy_scaling()).model.rotary_emb
    assert rotary.rope_type == "llama3"
    want = _meta_llama3_inv_freq(G9_ROPE_THETA, TOY_HEAD_DIM, 8.0, 1.0, 4.0, TOY_ORIGINAL)
    assert torch.allclose(rotary.inv_freq.double(), want, rtol=1e-6, atol=0)
    default = torch.tensor([1.0 / G9_ROPE_THETA ** (i / TOY_HEAD_DIM)
                            for i in range(0, TOY_HEAD_DIM, 2)], dtype=torch.float64)
    assert bool((~torch.isclose(want, default, rtol=1e-3)).all())


def test_wrapped_reproduces_unwrapped_logits_past_the_original_context():
    """The logits gate, run past `original_max_position_embeddings`."""
    model = _toy_llama(_toy_scaling())
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (1, TOY_SEQ))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    wrapped = SwappableAttentionModel(model, _attn_cfg(), model_id="toy/llama3-rope")
    out = wrapped.run_measured(ids, NaiveAttention(), cfg=_attn_cfg())
    diff = (out.logits - ref).abs()
    assert torch.allclose(out.logits, ref, atol=1e-5, rtol=1e-5), diff.max().item()
    assert diff[:, TOY_ORIGINAL:].max().item() <= 1e-5


def test_dropping_the_scaling_would_fail_the_gate():
    """Same weights, no rope_scaling. llama3 scaling is static (it rescales
    the frequencies, not the positions), so every position moves, not only
    those past the original context. The gate's own tolerance must reject
    the unscaled logits by a wide margin."""
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (1, TOY_SEQ))
    with torch.no_grad():
        scaled = _toy_llama(_toy_scaling())(input_ids=ids).logits
        plain = _toy_llama(None)(input_ids=ids).logits
    assert not torch.allclose(plain, scaled, atol=1e-5, rtol=1e-5)
    assert (scaled - plain)[:, TOY_ORIGINAL:].abs().max().item() > 1e4 * 1e-5
