"""Decode state: the KV cache, and the one line that must not be wrong.

Stage 3 generates 8-33 tokens per example. Without a cache each of those is a
full forward over the whole context, which turns a 13.35 h grid into ~97.7 h
(see docs/stage3_generation_decision.md). So the cache is not an optimisation
here, it is what makes the stage runnable.

`state_from_prefill` is deliberately separate from the pre-existing
`make_decode_state`, and the two return the same type, which is exactly why
the difference needs a test: `make_decode_state` invents its own inputs
(right for a Stage 2 decode sweep, which times decode against a state of the
correct shape and does not care what is in it) while Stage 3 needs the real
prompt's keys and values. Confusing them decodes fluent nonsense with no error
raised anywhere.

SDPA runs on CPU, so the contract is testable without a GPU. GLA needs `fla`
and CUDA; its bounded-state counterpart is asserted at the interface level
here and exercised on the instance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from attnbench.backends import KVCacheState, UnsupportedConfig, all_backends
from attnbench.backends.impls import SDPABackend
from attnbench.config import AttnConfig

CPU = dict(device="cpu")


def _cfg(seq_len=16, n_heads_q=4, n_heads_kv=4, head_dim=8, **kw):
    base = dict(seq_len=seq_len, batch=1, n_heads_q=n_heads_q,
                n_heads_kv=n_heads_kv, head_dim=head_dim, dtype="float32",
                mask="causal")
    base.update(kw)
    return AttnConfig(**base)


def _kv(cfg, seed=0, n=None):
    g = torch.Generator().manual_seed(seed)
    n = cfg.seq_len if n is None else n
    shape = (cfg.batch, cfg.n_heads_kv, n, cfg.head_dim)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def _q(cfg, seed=1, n=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((cfg.batch, cfg.n_heads_q, n, cfg.head_dim), generator=g)


# --- the correctness anchor -------------------------------------------------

def test_decoding_one_token_matches_a_full_causal_forward(): 
    """The test that makes every other decode number trustworthy.

    Prefill S tokens, decode the (S+1)th through the cache, and compare
    against running plain causal attention over all S+1 at once. The final
    position's output must agree to near machine precision in fp32 -- not to a
    loose kernel tolerance, because this is the SAME computation arranged two
    ways, not two kernels being compared.
    """
    backend = SDPABackend("math")
    cfg = _cfg(seq_len=16)
    k, v = _kv(cfg)
    k_new, v_new = _kv(cfg, seed=7, n=1)
    q_new = _q(cfg)

    state = backend.state_from_prefill(k, v, cfg)
    got, _ = backend.decode_step(q_new, k_new, v_new, state, cfg)

    full_k = torch.cat([k, k_new], dim=2)
    full_v = torch.cat([v, v_new], dim=2)
    full_q = torch.cat([torch.zeros_like(_q(cfg, n=cfg.seq_len)), q_new], dim=2)
    reference = F.scaled_dot_product_attention(full_q, full_k, full_v,
                                                is_causal=True)
    torch.testing.assert_close(got[:, :, -1, :], reference[:, :, -1, :],
                               atol=1e-5, rtol=1e-5)


def test_is_causal_on_a_single_query_would_see_only_position_zero():
    """The specific bug the decode path is written to avoid, demonstrated.

    With q_len=1 against kv_len=S+1, PyTorch aligns a causal mask to the
    TOP-LEFT, so is_causal=True masks everything except position 0. It raises
    nothing. It returns fluent, wrong values. This test exists so that if
    someone 'tidies' decode_step by making it match forward's
    `is_causal=(cfg.mask == 'causal')`, a test fails instead of the study.
    """
    cfg = _cfg(seq_len=16)
    k, v = _kv(cfg)
    k_new, v_new = _kv(cfg, seed=7, n=1)
    q_new = _q(cfg)
    full_k, full_v = torch.cat([k, k_new], 2), torch.cat([v, v_new], 2)

    wrong = F.scaled_dot_product_attention(q_new, full_k, full_v, is_causal=True)
    right = F.scaled_dot_product_attention(q_new, full_k, full_v, is_causal=False)
    only_first = F.scaled_dot_product_attention(
        q_new, full_k[:, :, :1], full_v[:, :, :1], is_causal=False)

    assert not torch.allclose(wrong, right, atol=1e-4)
    torch.testing.assert_close(wrong, only_first, atol=1e-5, rtol=1e-5)

    backend = SDPABackend("math")
    got, _ = backend.decode_step(q_new, k_new, v_new,
                                 backend.state_from_prefill(k, v, cfg), cfg)
    torch.testing.assert_close(got, right, atol=1e-5, rtol=1e-5)


def test_many_steps_agree_with_one_forward_over_the_whole_sequence():
    """Errors that compound across steps are invisible in a one-step test."""
    backend = SDPABackend("math")
    cfg = _cfg(seq_len=12)
    k, v = _kv(cfg)
    state = backend.state_from_prefill(k, v, cfg)

    outs, all_k, all_v, all_q = [], k, v, []
    for step in range(6):
        k_new, v_new = _kv(cfg, seed=100 + step, n=1)
        q_new = _q(cfg, seed=200 + step)
        out, state = backend.decode_step(q_new, k_new, v_new, state, cfg)
        outs.append(out)
        all_k = torch.cat([all_k, k_new], 2)
        all_v = torch.cat([all_v, v_new], 2)
        all_q.append(q_new)

    full_q = torch.cat([torch.zeros_like(_q(cfg, n=cfg.seq_len))] + all_q, dim=2)
    reference = F.scaled_dot_product_attention(full_q, all_k, all_v, is_causal=True)
    torch.testing.assert_close(torch.cat(outs, dim=2),
                               reference[:, :, cfg.seq_len:, :],
                               atol=1e-5, rtol=1e-5)


# --- the bounded / unbounded distinction ------------------------------------

def test_the_sdpa_cache_grows_and_stays_in_gqa_layout():
    """Payload size IS the measurement on the unbounded arm, so it must not be
    inflated by a convenience expansion. On Qwen2.5-1.5B (12 q heads, 2 kv
    heads) expanding here would report a 6x larger cache than the model
    actually holds."""
    backend = SDPABackend("math")
    cfg = _cfg(seq_len=16, n_heads_q=12, n_heads_kv=2)
    k, v = _kv(cfg)
    state = backend.state_from_prefill(k, v, cfg)
    assert state.payload["k"].shape[1] == 2, "expanded to query heads"

    before = state.payload["k"].shape[2]
    k_new, v_new = _kv(cfg, seed=7, n=1)
    _, state = backend.decode_step(_q(cfg), k_new, v_new, state, cfg)
    assert state.payload["k"].shape[2] == before + 1
    assert state.payload["k"].shape[1] == 2


def test_gla_declares_a_bounded_state_and_sdpa_does_not():
    """Interface-level, since GLA needs CUDA to run: the two arms must not
    both be reported as unbounded by a backend that simply forgot to
    implement state_from_prefill."""
    backends = all_backends()
    assert "state_from_prefill" in vars(backends["gla"]), \
        "gla must define its own bounded-state constructor"
    assert "state_from_prefill" in vars(backends["sdpa"])


# --- refusals ---------------------------------------------------------------

def test_state_from_another_backend_is_refused():
    backend = SDPABackend("math")
    cfg = _cfg()
    k, v = _kv(cfg)
    alien = KVCacheState(backend="gla", payload={"k": k, "v": v})
    with pytest.raises(ValueError, match="gla"):
        backend.decode_step(_q(cfg), *_kv(cfg, seed=7, n=1), alien, cfg)


def test_a_backend_without_decode_support_says_so():
    """Default raises UnsupportedConfig, so a missing decode path reports
    through the same claim/probe machinery as any other unsupported config
    rather than crashing mid-generation."""
    from attnbench.backends.impls import NaiveAttention

    cfg = _cfg()
    k, v = _kv(cfg)
    with pytest.raises(UnsupportedConfig, match="no decode support"):
        NaiveAttention().state_from_prefill(k, v, cfg)


def test_make_decode_state_and_state_from_prefill_are_not_the_same_thing():
    """Same return type, different contract. make_decode_state invents its
    inputs; a Stage 3 caller that reaches for it gets a state carrying
    somebody else's keys, and decodes fluent nonsense with no error."""
    backend = SDPABackend("math")
    cfg = _cfg(seq_len=16)
    k, v = _kv(cfg, seed=3)

    real = backend.state_from_prefill(k, v, cfg)
    synthetic = backend.make_decode_state(cfg, device="cpu", seed=3)

    assert real.payload["k"].shape == synthetic.payload["k"].shape
    assert not torch.allclose(real.payload["k"], synthetic.payload["k"]), (
        "if these ever coincide the test has stopped distinguishing them")
