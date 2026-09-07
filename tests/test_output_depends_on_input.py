"""Every backend's output must depend on its input.

The cheapest invariant in this repo, and the one that would have caught the
most expensive silent failure it has had.

On 2026-09-06, Stage 3 ran GLA against 300 distinct 2048-token prompts per
task and got **15 distinct predictions**. The dense arm got 300 of 300.
Nothing raised: every row had fluent text, a real latency, a real
stop_reason, a passing provenance stamp. The cause was that GLA's forget gate
is synthesized from `cfg.key()` -- correct for a throughput benchmark, where
gate VALUES cannot affect speed -- giving a memory horizon of 1.24 tokens, so
the model answered from the last token or two of a prompt whose tail is
identical across every example.

A backend that ignores its input is indistinguishable from one that is bad at
its job, unless you check. This checks.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.backends import all_backends
from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.impls import NaiveAttention, SDPABackend
from attnbench.config import AttnConfig

# Runnable without CUDA. The others are named in NOT_EXERCISED_ON_CPU below,
# the same convention tests/test_generation.py uses -- an unrunnable backend
# must not look like a clean one.
CPU_BACKENDS = {
    "naive": NaiveAttention,
    "sdpa_math": lambda: SDPABackend("math"),
}

NOT_EXERCISED_ON_CPU = {
    "fa2": "needs CUDA + flash-attn",
    "flex": "needs CUDA for inductor lowering",
    "gla": "needs CUDA + fla -- and is covered by the gate_source guard below",
    "block_sparse": "needs CUDA + block-sparse-attn",
    "sage": "needs CUDA + sageattention",
    "xformers": "needs CUDA + xformers",
}


def _cfg(seq_len=64):
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=4,
                      head_dim=64, dtype="float32", mask="causal")


def _qkv(seed, seq_len=64):
    g = torch.Generator().manual_seed(seed)
    shape = (1, 4, seq_len, 64)
    return tuple(torch.randn(shape, generator=g) for _ in range(3))


@pytest.mark.parametrize("name", sorted(CPU_BACKENDS))
def test_output_depends_on_input(name):
    """Two different prompts must not produce the same answer."""
    backend = CPU_BACKENDS[name]()
    cfg = _cfg()
    a = backend.forward(*_qkv(0), cfg)
    b = backend.forward(*_qkv(1), cfg)
    assert not torch.allclose(a, b, atol=1e-6), (
        f"{name} produced the same output for two different inputs -- it is "
        f"not reading its input, and every downstream number is meaningless")


@pytest.mark.parametrize("name", sorted(CPU_BACKENDS))
def test_output_depends_on_the_DISTANT_past_not_only_the_recent(name):
    """The specific shape of the GLA failure.

    Its output DID vary -- with the last token or two. What it had lost was
    everything before that, which is exactly what a long-context benchmark
    exists to measure. Changing only the FIRST half of the sequence must
    still change the final position's output.
    """
    backend = CPU_BACKENDS[name]()
    cfg = _cfg(seq_len=128)
    q, k, v = _qkv(0, seq_len=128)
    q2, k2, v2 = (t.clone() for t in (q, k, v))
    alt_q, alt_k, alt_v = _qkv(99, seq_len=128)
    q2[:, :, :64], k2[:, :, :64], v2[:, :, :64] = (
        alt_q[:, :, :64], alt_k[:, :, :64], alt_v[:, :, :64])

    last_a = backend.forward(q, k, v, cfg)[:, :, -1]
    last_b = backend.forward(q2, k2, v2, cfg)[:, :, -1]
    assert not torch.allclose(last_a, last_b, atol=1e-6), (
        f"{name}: rewriting the first half of the context did not change the "
        f"last position's output. A model that cannot see the start of its "
        f"prompt cannot answer a needle-in-a-haystack question.")


def test_gla_refuses_to_run_without_a_learned_gate():
    """The guard for the backend this test file exists because of. No CUDA
    needed -- the refusal happens before any kernel is reached."""
    from attnbench.backends.linear import GatedLinearAttention

    with pytest.raises(UnsupportedConfig, match="1.24 tokens"):
        GatedLinearAttention().forward(*_qkv(0), _cfg())
    with pytest.raises(UnsupportedConfig, match="learned forget gate"):
        GatedLinearAttention().state_from_prefill(*_qkv(0)[1:], _cfg())


def test_the_synthetic_gate_has_to_be_asked_for_by_name():
    """It produces a kernel-accurate latency and a meaningless output, so it
    must never be reachable by omission."""
    from attnbench.backends.linear import GatedLinearAttention

    assert GatedLinearAttention().gate_source == "learned"
    assert GatedLinearAttention(gate_source="synthetic").gate_source == "synthetic"
    with pytest.raises(ValueError):
        GatedLinearAttention(gate_source="whatever")


def test_every_registered_backend_is_either_exercised_or_named():
    registered = set(all_backends())
    covered = {n.split("_")[0] if n.startswith("sdpa") else n
               for n in CPU_BACKENDS} | set(NOT_EXERCISED_ON_CPU)
    assert registered <= covered, (
        f"{sorted(registered - covered)} are neither exercised here nor "
        f"listed as needing hardware")
