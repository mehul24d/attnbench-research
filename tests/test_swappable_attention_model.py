"""SwappableAttentionModel: the mechanism Stage 3 depends on entirely.

All CPU, all on a tiny random-init Llama-family model (no download needed) --
this validates the wrapping/RoPE/GQA/scoring/caching *machinery*, not real
model accuracy, which needs a real model and a GPU.

Test (a) below matters more than every other test in this file combined: it
is the one guarantee that the wrapper reproduces the real model's math
exactly. Everything downstream -- Stage 3 accuracy numbers, Stage 4's
matched-accuracy points, Stage 6/7's decision tree -- inherits silently from
whatever this test would have caught. Treated the same way
test_gqa_agreement.py gates a rented session: run first, and a failure here
means don't trust anything else in this file.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench.accuracy.model import (  # noqa: E402
    SwappableAttentionModel,
    UnsupportedModelArchitecture,
    _scoring_forward_chunked,
    _self_attn_returns_3tuple,
    pool_scores_to_block_size,
)
from attnbench.accuracy import score_cache  # noqa: E402
from attnbench.backends.impls import NaiveAttention, SDPABackend  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402


def _toy_model(seed: int = 0, num_key_value_heads: int = 2):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=num_key_value_heads, head_dim=8,
        max_position_embeddings=64, attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _toy_cfg(**overrides) -> AttnConfig:
    base = dict(seq_len=16, batch=1, n_heads_q=4, n_heads_kv=2,
                head_dim=8, dtype="float32", mask="causal")
    base.update(overrides)
    return AttnConfig(**base)


class _ZeroBackend:
    """Deliberately wrong backend for test (c): proves the swap actually
    changes what runs, rather than the wrapper silently bypassing it."""

    def forward(self, q, k, v, cfg, mask=None):
        return torch.zeros_like(q)


def test_a_wrapped_reproduces_unwrapped_logits_near_machine_precision():
    """THE test. Wrapping with NaiveAttention (equivalent math to the
    model's own "eager" attention) must reproduce the unwrapped model's
    logits at near machine precision in fp32 -- not a loose kernel-
    comparison tolerance, since this checks the *same* computation, not two
    different kernels agreeing within numerical-method slop. A failure here
    means RoPE handling or KV layout is wrong and every downstream Stage 3
    number would be quietly invalid.
    """
    model = _toy_model()
    input_ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        reference_logits = model(input_ids=input_ids).logits

    cfg = _toy_cfg()
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama")
    out = wrapped.run_measured(input_ids, NaiveAttention(), cfg=cfg)

    assert torch.allclose(out.logits, reference_logits, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(out.logits - reference_logits).abs().max().item()}"
    )


def test_b_gqa_shapes_work():
    """num_key_value_heads=2 < num_attention_heads=4 is already the GQA
    case test (a) runs against; this just makes the GQA intent explicit and
    checks a second, different GQA ratio too."""
    model = _toy_model(num_key_value_heads=1)
    input_ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        reference_logits = model(input_ids=input_ids).logits

    cfg = _toy_cfg(n_heads_kv=1)
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama-gqa4to1")
    out = wrapped.run_measured(input_ids, NaiveAttention(), cfg=cfg)
    assert torch.allclose(out.logits, reference_logits, atol=1e-5, rtol=1e-5)


def test_c_different_backend_changes_output():
    """The swap must actually take effect -- a wrapper bug that silently
    keeps using the original attention would still pass every other test
    in this file."""
    model = _toy_model()
    input_ids = torch.randint(0, 64, (1, 16))
    cfg = _toy_cfg()
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama")

    real = wrapped.run_measured(input_ids, NaiveAttention(), cfg=cfg)
    zeroed = wrapped.run_measured(input_ids, _ZeroBackend(), cfg=cfg)
    assert not torch.allclose(real.logits, zeroed.logits)


def test_logits_to_keep_restricts_shape_and_matches_last_position():
    """Regression test for the lm_head OOM found on the first GPU
    validation session: computing logits over the full context at a ~150K
    vocab tried to allocate 11+ GiB at 32K tokens, for a call whose only
    use is the final generated token. `logits_to_keep=1` must both shrink
    the output to one position AND agree exactly with that same position
    from a full-sequence call -- restricting *which* positions are
    computed, not changing the computation itself."""
    model = _toy_model()
    input_ids = torch.randint(0, 64, (1, 16))
    cfg = _toy_cfg()
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama")

    full = wrapped.run_measured(input_ids, NaiveAttention(), cfg=cfg)
    last_only = wrapped.run_measured(input_ids, NaiveAttention(), cfg=cfg,
                                      logits_to_keep=1)

    assert last_only.logits.shape[1] == 1
    assert torch.allclose(last_only.logits[:, -1, :], full.logits[:, -1, :],
                          atol=1e-5, rtol=1e-5)


def test_d_unwrap_restores_original_behavior_exactly():
    model = _toy_model()
    input_ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        before = model(input_ids=input_ids).logits

    cfg = _toy_cfg()
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama")
    wrapped.run_measured(input_ids, NaiveAttention(), cfg=cfg)
    wrapped.unwrap()

    with torch.no_grad():
        after = model(input_ids=input_ids).logits
    assert torch.equal(before, after)


def test_unsupported_architecture_raises():
    class NotLlamaShaped:
        pass

    with pytest.raises(UnsupportedModelArchitecture):
        SwappableAttentionModel(NotLlamaShaped(), _toy_cfg(), model_id="not-a-model")


def test_e_chunked_matches_unchunked_scoring():
    """Chunking must not change the math, only the memory access pattern --
    chunk_blocks=1 (maximally chunked) vs a chunk_blocks large enough to
    cover everything in one shot (unchunked) must agree exactly."""
    torch.manual_seed(1)
    q = torch.randn(1, 4, 32, 8)
    k = torch.randn(1, 2, 32, 8)
    v = torch.randn(1, 2, 32, 8)

    out_chunked, scores_chunked = _scoring_forward_chunked(
        q, k, v, block_size=4, chunk_blocks=1)
    out_unchunked, scores_unchunked = _scoring_forward_chunked(
        q, k, v, block_size=4, chunk_blocks=1000)

    assert torch.allclose(out_chunked, out_unchunked, atol=1e-6)
    assert torch.allclose(scores_chunked, scores_unchunked, atol=1e-6)


def test_f_scoring_memory_bounded_not_quadratic():
    """Informal regression guard: peak memory during scoring on a larger
    CPU-feasible config should track chunk_size x seq_len, not seq_len^2.
    Not a hard proof, but catches a regression back to materialising the
    full matrix.
    """
    import tracemalloc

    torch.manual_seed(2)
    seq_len = 4096
    q = torch.randn(1, 4, seq_len, 8)
    k = torch.randn(1, 2, seq_len, 8)
    v = torch.randn(1, 2, seq_len, 8)

    tracemalloc.start()
    _scoring_forward_chunked(q, k, v, block_size=64, chunk_blocks=4)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    naive_matrix_bytes = seq_len * seq_len * 4  # one fp32 (S,S) matrix, one head
    assert peak < naive_matrix_bytes / 4, (
        f"peak {peak} bytes is too close to the O(seq_len^2) matrix size "
        f"{naive_matrix_bytes} bytes -- looks like chunking regressed"
    )


def test_g_score_cache_round_trip_hits_on_second_call():
    model = _toy_model()
    input_ids = torch.randint(0, 64, (1, 16))
    cfg = _toy_cfg()
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama",
                                       finest_block_size=4)

    calls = {"n": 0}
    import attnbench.accuracy.model as model_mod
    real_scoring_fn = model_mod._scoring_forward_chunked

    def counting_scoring_fn(*args, **kwargs):
        calls["n"] += 1
        return real_scoring_fn(*args, **kwargs)

    model_mod._scoring_forward_chunked = counting_scoring_fn
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as cache_dir:
            wrapped.compute_importance_scores(
                input_ids, task="niah_single_1", example_id="ex0",
                cache_dir=cache_dir)
            calls_after_first = calls["n"]
            assert calls_after_first == wrapped.n_layers  # one call per layer

            wrapped.compute_importance_scores(
                input_ids, task="niah_single_1", example_id="ex0",
                cache_dir=cache_dir)
            assert calls["n"] == calls_after_first, (
                "second call with the same key must hit the cache and skip "
                "the dense scoring pass entirely"
            )
    finally:
        model_mod._scoring_forward_chunked = real_scoring_fn


def test_h_coarser_block_size_derives_from_finest_cache():
    """A request for a coarser block_size must pool the cached finest
    tensor further, never trigger a separate cache miss/recompute."""
    torch.manual_seed(3)
    finest = torch.rand(2, 8, 8)  # (n_heads_kv, n_blocks@4, n_blocks@4)
    coarse = pool_scores_to_block_size(finest, finest_block_size=4,
                                        target_block_size=8)
    assert coarse.shape == (2, 4, 4)
    # Manually check one pooled cell against the four finest cells it covers.
    expected_00 = finest[:, 0:2, 0:2].mean(dim=(1, 2))
    assert torch.allclose(coarse[:, 0, 0], expected_00)


def test_i_cache_key_differs_when_only_model_id_changes():
    """The bug this exists to catch: model_id accepted as a parameter but
    never actually hashed, which would let a run against a different model
    silently load a wrong-shaped cached tensor."""
    k1 = score_cache.cache_key("model-a", "niah_single_1", "ex0", 1024)
    k2 = score_cache.cache_key("model-b", "niah_single_1", "ex0", 1024)
    assert k1 != k2


def test_j_cache_load_raises_on_shape_mismatch(tmp_path):
    key = score_cache.cache_key("model-a", "task", "ex0", 1024)
    wrong_shape = torch.rand(5, 3, 4, 4)  # e.g. saved by a different model
    score_cache.save(tmp_path, key, wrong_shape)

    with pytest.raises(ValueError):
        score_cache.load(tmp_path, key, expected_n_layers=2,
                          expected_n_heads_kv=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_k_scoring_forward_chunked_works_on_cuda():
    """Regression test for a real bug found on the first GPU validation
    session: `out`, `scores`, `row_idx`, and `col_idx` were allocated
    without `device=`, defaulting to CPU. On CPU-only testing (every other
    test in this file) that was invisible, since inputs were already CPU --
    it only surfaces as a device-mismatch RuntimeError at `masked_fill` once
    q/k/v are actually on CUDA. `out` must stay on q.device (it feeds the
    next transformer layer); `scores` is deliberately moved to CPU inside
    the function to match masks.py's CPU-only mask-generation convention,
    so it's asserted as CPU here specifically, not just "no crash".
    """
    device = "cuda"
    batch, n_heads_q, n_heads_kv, seq_len, head_dim = 1, 4, 2, 16, 8
    q = torch.randn(batch, n_heads_q, seq_len, head_dim, device=device)
    k = torch.randn(batch, n_heads_kv, seq_len, head_dim, device=device)
    v = torch.randn(batch, n_heads_kv, seq_len, head_dim, device=device)

    out, scores = _scoring_forward_chunked(q, k, v, block_size=4, chunk_blocks=2)

    assert out.device.type == "cuda"
    assert scores.device.type == "cpu"


def test_l_self_attn_tuple_arity_matches_known_transformers_versions():
    """Regression test for a second real bug found in the same GPU
    session: the decoder layer's unpacking of self_attn's return value
    differs between transformers major versions -- 4.46.0's
    Qwen2DecoderLayer unpacks 3 values, 5.16.1's LlamaDecoderLayer unpacks
    2. Pins both observed data points so a future transformers release
    changing this again fails a fast CPU test instead of a CUDA crash."""
    assert _self_attn_returns_3tuple("4.46.0") is True
    assert _self_attn_returns_3tuple("5.16.1") is False


# ---------------------------------------------------------------------------
# Batching: whether run_measured / compute_importance_scores support
# batch > 1, and whether doing so is accuracy-neutral -- checked directly
# rather than assumed, per the batching-lever investigation.
# ---------------------------------------------------------------------------

def test_run_measured_batches_cleanly_for_a_dense_backend():
    """Dense backends (no importance-scores dependency) must be safe to
    batch: each batch element's output should be identical to running that
    same sequence alone at batch=1 -- no cross-example contamination from
    q_proj/k_proj/v_proj/o_proj, RoPE, or NaiveAttention's own batched
    matmuls, all of which are standard per-example-independent operations.
    This is what actually licenses batching dense/gla/sage measured passes;
    it is not automatic just because nn.Linear happens to accept a batch
    dimension -- SwappedAttention's own plumbing (position_embeddings,
    cfg.batch, output reshape) has to carry it through correctly too.
    """
    model = _toy_model()
    seq_a = torch.randint(0, 64, (1, 16))
    seq_b = torch.randint(0, 64, (1, 16))
    cfg = _toy_cfg()

    wrapped_batched = SwappableAttentionModel(model, cfg, model_id="toy/llama-batch")
    batched_input = torch.cat([seq_a, seq_b], dim=0)  # (2, 16)
    batched_out = wrapped_batched.run_measured(batched_input, NaiveAttention(), cfg=cfg)
    wrapped_batched.unwrap()

    wrapped_a = SwappableAttentionModel(model, cfg, model_id="toy/llama-a")
    out_a = wrapped_a.run_measured(seq_a, NaiveAttention(), cfg=cfg)
    wrapped_a.unwrap()

    wrapped_b = SwappableAttentionModel(model, cfg, model_id="toy/llama-b")
    out_b = wrapped_b.run_measured(seq_b, NaiveAttention(), cfg=cfg)
    wrapped_b.unwrap()

    assert torch.allclose(batched_out.logits[0:1], out_a.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(batched_out.logits[1:2], out_b.logits, atol=1e-5, rtol=1e-5)


def test_compute_importance_scores_rejects_batch_greater_than_one():
    """The real architectural constraint the batching-lever investigation
    surfaced: the scoring pass is hard-coded to batch=1 (_scoring_forward_
    chunked's own check), and state.scores has no batch dimension at all
    (one (n_heads_kv, n_blocks, n_blocks) ranking per layer, not one per
    batch element). Removing the check alone would not make this safe --
    it would silently share one example's importance ranking across every
    other example in the batch. This test pins the check as a deliberate
    guard, not an oversight to "just remove" when batching block_sparse."""
    model = _toy_model()
    cfg = _toy_cfg()
    wrapped = SwappableAttentionModel(model, cfg, model_id="toy/llama-batch-scoring")
    batched_input = torch.randint(0, 64, (2, 16))

    with pytest.raises(ValueError, match="batch=1"):
        wrapped.compute_importance_scores(
            batched_input, task="niah_single", example_id="ex0", cache_dir="/tmp")


def test_cheap_scoring_pass_propagates_faithful_hidden_states(tmp_path):
    """The cheap (minference_meanpool) scoring branch must emit a REAL
    attention output, not zeros.

    It returned torch.zeros_like(q) on the theory that the scoring pass's
    output is unused. It is not: layer L's q and k are computed from the
    hidden states layers < L produced, so a zeroed output corrupts the
    activations every later layer is scored against -- while leaving the
    scores finite, normalised and entirely plausible. The cheap arm would
    have looked worse than it is, which in the oracle-vs-cheap experiment
    reads as "the oracle matters".

    The property asserted is the one that actually matters: the hidden
    states entering each layer during a CHEAP scoring pass must match those
    entering it during a DENSE scoring pass, because neither pass's
    attention output is supposed to differ. Against the zeros version this
    fails at layer 1 by ~1.8 in max-abs, not by a tolerance-sized amount.
    """
    tmpl = AttnConfig(seq_len=64, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=8, dtype="float32", mask="causal", block_size=8)
    ids = torch.randint(0, 64, (1, 64),
                        generator=torch.Generator().manual_seed(7))

    def hidden_states_per_layer(score_source: str):
        cache_dir = tmp_path / score_source
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, attn_implementation="eager",
        )
        model = LlamaForCausalLM(config)
        model.eval()
        swapped = SwappableAttentionModel(
            model, cfg_template=tmpl, model_id="toy",
            score_source=score_source)
        seen = []
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.register_forward_pre_hook(
                (lambda i: (lambda mod, args, kwargs: seen.append(
                    (i, (args[0] if args else kwargs["hidden_states"]).detach().clone())
                )))(idx),
                with_kwargs=True,
            )
        # Drive the PRODUCTION entry point, not `model(ids)` directly.
        #
        # This test called the raw HF model until 2026-09-21 and so never set
        # `use_cache=False` -- which compute_importance_scores sets
        # deliberately, because SwappedAttention returns no real KV cache and
        # the outer model then tries to convert a None to legacy format. The
        # comment at model.py:530 predicts that failure verbatim, and on the
        # measurement image (transformers 4.46.0) it is what happened:
        #     AttributeError: 'NoneType' object has no attribute 'to_legacy_cache'
        # while this file passed on the workstation's newer transformers,
        # which no longer takes that path. A test that builds a configuration
        # the production code never creates is testing something else -- and
        # here "something else" was the one keyword that matters.
        with torch.no_grad():
            swapped.compute_importance_scores(
                ids, task="t", example_id="e", cache_dir=str(cache_dir))
        return seen

    cheap = hidden_states_per_layer("minference_meanpool")
    dense = hidden_states_per_layer("dense_softmax_fp32")
    assert [i for i, _ in cheap] == [0, 1, 2, 3]

    for (i, a), (j, b) in zip(cheap, dense):
        assert i == j
        drift = (a - b).abs().max().item()
        assert drift < 1e-4, (
            f"layer {i} hidden states diverge by {drift:.3e} between the cheap "
            f"and dense scoring passes -- the cheap branch is not emitting a "
            f"real attention output"
        )
