"""Greedy generation with a KV cache: the contract Stage 3 scores against.

Two failures live in this path and neither raises. Both produce fluent,
plausible, wrong text:

  1. `is_causal=True` on a single query token -- covered in
     tests/test_decode_state.py.
  2. A missing `position_ids` during decode, so RoPE rotates every generated
     token as though it were the start of the sequence. Covered here.

The anchor for both is the same: cached decode must reproduce, token for
token, what the unwrapped model produces re-running the whole sequence each
step. If that holds, the cache is not changing the answer -- it is only making
it affordable.

See docs/stage3_generation_decision.md for the stopping rule and caps.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench.accuracy.model import (
    GenerationResult,
    SwappableAttentionModel,
    UnsupportedModelArchitecture,
)
from attnbench.backends.impls import NaiveAttention, SDPABackend
from attnbench.config import AttnConfig


def _toy():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


def _cfg(seq_len=16):
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=8, dtype="float32", mask="causal")


def _ids(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (1, n), generator=g)


def _hf_greedy(model, ids, k):
    """Ground truth: no cache, whole sequence re-forwarded every step."""
    seq = list(ids[0].tolist())
    with torch.no_grad():
        for _ in range(k):
            out = model(input_ids=torch.tensor([seq]))
            seq.append(int(out.logits[0, -1].argmax()))
    return seq[ids.shape[-1]:]


# --- the anchor -------------------------------------------------------------

def test_cached_decode_reproduces_uncached_greedy_exactly():
    """Token identity, not a tolerance. Greedy decode is argmax over logits,
    so a cache that changed the logits even slightly would eventually flip a
    token, and a near-miss on one token is a completely different sentence
    afterwards."""
    model, ids = _toy(), _ids()
    expected = _hf_greedy(model, ids, 6)

    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    got = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                           max_new_tokens=6)
    assert got.token_ids == expected


def test_generate_supplies_increasing_absolute_positions():
    """Asserted directly, by watching what reaches the model, rather than
    inferred from the tokens that come out.

    The first version of this test compared generated tokens with and without
    position_ids and they were IDENTICAL -- on a 2-layer random-init toy
    model the RoPE difference is real but does not happen to flip an argmax
    over six steps. A test whose premise fails silently is worse than none,
    so this watches the input instead of the output.
    """
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    seen: list = []
    real_forward = wrapped.model.forward

    def spy(*args, **kwargs):
        if kwargs.get("input_ids") is not None and kwargs["input_ids"].shape[-1] == 1:
            pos = kwargs.get("position_ids")
            seen.append(None if pos is None else int(pos.flatten()[0]))
        return real_forward(*args, **kwargs)

    wrapped.model.forward = spy
    try:
        wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(), max_new_tokens=4)
    finally:
        wrapped.model.forward = real_forward

    prefill_len = ids.shape[-1]
    assert None not in seen, "a decode step reached the model with no position_ids"
    assert seen == [prefill_len + i for i in range(len(seen))], seen


def test_rope_actually_moves_the_logits_with_position():
    """The mechanism the test above protects. If this ever stops holding,
    that test is guarding nothing and should be removed rather than left
    green."""
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    wrapped.run_measured(ids, SDPABackend("math"), cfg=_cfg(),
                         logits_to_keep=1, decode_backend=SDPABackend("math"))
    wrapped._state.mode = "decode"
    try:
        step = torch.tensor([[7]])
        with torch.no_grad():
            at_zero = wrapped.model(input_ids=step, use_cache=False,
                                    position_ids=torch.tensor([[0]]),
                                    logits_to_keep=1).logits
        wrapped._state.decode_states = {}
        wrapped.run_measured(ids, SDPABackend("math"), cfg=_cfg(),
                             logits_to_keep=1, decode_backend=SDPABackend("math"))
        wrapped._state.mode = "decode"
        with torch.no_grad():
            at_sixteen = wrapped.model(input_ids=step, use_cache=False,
                                       position_ids=torch.tensor([[16]]),
                                       logits_to_keep=1).logits
    finally:
        wrapped._state.mode = "measured"

    assert not torch.allclose(at_zero, at_sixteen, atol=1e-4), (
        "RoPE position had no effect, so position_ids cannot be load-bearing "
        "on this model and the guard above is vacuous")


# --- stopping ---------------------------------------------------------------

def test_stops_on_eos_and_says_so():
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    first = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                             max_new_tokens=6).token_ids[0]
    r = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                         max_new_tokens=6, eos_token_ids=frozenset({first}))
    assert r.stop_reason == "eos"
    assert r.n_generated == 1
    assert not r.truncated


def test_stops_on_newline_which_is_the_one_that_will_actually_fire():
    """RULER prompts are completion-style and end mid-sentence, so an
    instruct model's EOS closes an assistant turn that never opened and is
    essentially never emitted. Without the newline stop every example exits
    on the cap and truncation rate carries no information about any
    backend."""
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    tokens = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                              max_new_tokens=6).token_ids
    r = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                         max_new_tokens=6,
                         newline_token_ids=frozenset({tokens[2]}))
    assert r.stop_reason == "newline"
    assert r.n_generated == 3


def test_the_cap_is_a_backstop_and_is_reported_as_truncation():
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    r = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(), max_new_tokens=3)
    assert r.stop_reason == "cap" and r.n_generated == 3 and r.truncated


def test_the_result_records_which_backend_did_what():
    """Sparsity is applied during prefill only, so prefill and decode can be
    different backends and the row has to say which were used."""
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    r = wrapped.generate(ids, NaiveAttention(), cfg=_cfg(), max_new_tokens=3,
                         decode_backend=SDPABackend("math"))
    assert isinstance(r, GenerationResult)
    assert r.prefill_backend == "naive"
    assert r.decode_backend == "sdpa_math"


def test_a_backend_without_decode_refuses_to_pick_one_silently():
    """block_sparse decodes densely BY DESIGN, and a silent fallback would
    make that indistinguishable from a bug. The caller names it."""
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    with pytest.raises(UnsupportedModelArchitecture, match="decode_backend"):
        wrapped.generate(ids, NaiveAttention(), cfg=_cfg(), max_new_tokens=3)


# --- determinism, per backend ----------------------------------------------

CPU_BACKENDS = {
    "naive": NaiveAttention,
    "sdpa_math": lambda: SDPABackend("math"),
}

# Backends that cannot be exercised here, and why. Named rather than skipped:
# an unrunnable backend must not look like a clean one, the same convention
# tests/test_timed_region_setup.py uses.
NOT_EXERCISED_ON_CPU = {
    "fa2": "needs CUDA + flash-attn",
    "flex": "needs CUDA for inductor lowering",
    "gla": "needs CUDA + fla",
    "block_sparse": "needs CUDA + block-sparse-attn",
    "sage": "needs CUDA + sageattention",
    "xformers": "needs CUDA + xformers",
}

# sdpa's flash / efficient / cudnn kernels are CUDA-only -- SDPA raises
# "No viable backend" for them on CPU -- so only the math variant is
# exercised here. The other three run under the same code path with a
# different kernel pinned, and are covered on the instance.


def _run_capturing_logits(wrapped, ids, backend, cfg, **kw):
    """Every logits tensor the model produced -- prefill and each decode
    step -- alongside the tokens. Bitwise comparison of these is what gives
    the determinism check any resolution at all; see the two tests below."""
    seen = []
    real_forward = wrapped.model.forward

    def spy(*args, **kwargs):
        out = real_forward(*args, **kwargs)
        seen.append(out.logits.detach().clone())
        return out

    wrapped.model.forward = spy
    try:
        result = wrapped.generate(ids, backend, cfg=cfg, **kw)
    finally:
        wrapped.model.forward = real_forward
    return result, seen


@pytest.mark.parametrize("name", sorted(CPU_BACKENDS))
def test_decode_is_deterministic_per_backend(name):
    """Per backend, not once.

    A single-backend determinism test passes while a backend with
    nondeterministic reductions silently poisons the comparison -- and the
    comparison is the entire study.

    Bitwise on the LOGITS, not only on the tokens, and the difference is not
    cosmetic. Measured on this toy model: perturbing a backend's output by
    1e-1 relative does not flip a single token over six steps, because a
    2-layer random-init model over a 64-token vocabulary has enormous logit
    gaps. A token-only check here would have had no resolution whatsoever
    against the failure it names -- green forever, guarding nothing, the same
    shape as the position_ids test above. The bitwise logits check resolves
    one fp32 ULP, and test_the_determinism_check_resolves_one_ulp holds it
    to that.
    """
    model, ids = _toy(), _ids()
    backend = CPU_BACKENDS[name]()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)
    a, logits_a = _run_capturing_logits(wrapped, ids, backend, _cfg(),
                                        max_new_tokens=6,
                                        decode_backend=SDPABackend("math"))
    b, logits_b = _run_capturing_logits(wrapped, ids, backend, _cfg(),
                                        max_new_tokens=6,
                                        decode_backend=SDPABackend("math"))
    assert a.token_ids == b.token_ids, f"{name} decoded differently twice"
    assert a.stop_reason == b.stop_reason
    assert len(logits_a) == len(logits_b) > 1, "prefill plus at least one step"
    for step, (x, y) in enumerate(zip(logits_a, logits_b)):
        assert torch.equal(x, y), f"{name} differed bitwise at step {step}"


class _Jitter(SDPABackend):
    """A backend that returns a slightly different answer each call -- the
    shape a nondeterministic reduction has, scaled relative to the values so
    it is a real ULP-level perturbation rather than an absolute nudge."""

    def __init__(self, rel: float):
        super().__init__("math")
        self._rel = rel
        self._sign = 1.0

    def next_call(self):
        self._sign = -self._sign

    def forward(self, q, k, v, cfg, mask=None):
        return super().forward(q, k, v, cfg, mask=mask) * (1.0 + self._sign * self._rel)


def test_the_determinism_check_resolves_one_ulp():
    """The companion to the test above: assert the mechanism, and separately
    assert the mechanism is observable.

    A determinism check that cannot detect a perturbation detects nothing,
    and there is no way to tell those apart from a passing test. This
    measures what the check actually resolves rather than assuming it
    resolves everything. Both halves matter:

      * 1e-7 relative (about one fp32 ULP) IS caught bitwise -- so the check
        above has real resolution against nondeterministic reductions.
      * 1e-1 relative -- a million times larger -- is NOT caught by comparing
        tokens, which is why the check above cannot rest on tokens alone.
    """
    model, ids = _toy(), _ids()
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)

    one_ulp = _Jitter(1e-7)
    one_ulp.next_call()
    _, a = _run_capturing_logits(wrapped, ids, one_ulp, _cfg(), max_new_tokens=6,
                                 decode_backend=SDPABackend("math"))
    one_ulp.next_call()
    _, b = _run_capturing_logits(wrapped, ids, one_ulp, _cfg(), max_new_tokens=6,
                                 decode_backend=SDPABackend("math"))
    assert any(not torch.equal(x, y) for x, y in zip(a, b)), (
        "a one-ULP perturbation went undetected, so the bitwise check in "
        "test_decode_is_deterministic_per_backend is not resolving anything")

    huge = _Jitter(1e-1)
    huge.next_call()
    t1 = wrapped.generate(ids, huge, cfg=_cfg(), max_new_tokens=6,
                          decode_backend=SDPABackend("math")).token_ids
    huge.next_call()
    t2 = wrapped.generate(ids, huge, cfg=_cfg(), max_new_tokens=6,
                          decode_backend=SDPABackend("math")).token_ids
    assert t1 == t2, (
        "tokens now resolve a 1e-1 perturbation on this toy model. Good "
        "news, but the comment in test_decode_is_deterministic_per_backend "
        "explaining why the bitwise check is load-bearing is then stale -- "
        "re-measure the token-level resolution and rewrite it.")


def test_every_registered_backend_is_either_exercised_or_named():
    """The list above must stay honest as backends are added -- otherwise a
    new one is exempt from the determinism check and nothing says so."""
    from attnbench.backends import all_backends

    registered = set(all_backends())
    covered = {n.split("_")[0] if n.startswith("sdpa") else n
               for n in CPU_BACKENDS} | set(NOT_EXERCISED_ON_CPU)
    assert registered <= covered, (
        f"{sorted(registered - covered)} are neither exercised on CPU nor "
        f"listed as needing hardware")
