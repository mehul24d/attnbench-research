"""Stage 5's measured body, driven against REAL objects at tiny shapes.

WHY THIS FILE EXISTS

Both failures of the 2026-09-07 Stage 5 session were signature errors:

  1. `cache_dir=None` -- not a supported "don't cache" sentinel;
     score_cache._path_for calls Path() on it.
  2. `masks.mask_for(cfg, importance_scores=scores[0])` -- one index too few;
     scores[0] is a layer's per-KV-head block (2, 16, 16) and the function
     wants one head's (16, 16).

`tests/test_phase_timing.py` was green through both. It covers the timing
protocol and the reconciliation arithmetic -- which are correct -- and
touches neither `SwappableAttentionModel` nor `masks`. "Tested on CPU" was a
narrower claim than it sounded.

A mock would not have helped: the thing that broke was the signature, and a
mock encodes the author's belief about the signature rather than the
signature. So this instantiates a real 2-layer LlamaForCausalLM at hidden
size 32, wraps it for real, and calls `phase_timing.measure_band` -- the same
body the script runs. One body, driven by both.

Each failure below is asserted to be caught, so the file cannot rot into a
smoke test that passes for the wrong reason.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from transformers import LlamaConfig, LlamaForCausalLM   # noqa: E402

from attnbench.accuracy.model import SwappableAttentionModel   # noqa: E402
from attnbench.accuracy.phase_timing import measure_band       # noqa: E402
from attnbench.backends.impls import SDPABackend               # noqa: E402
from attnbench.config import AttnConfig                        # noqa: E402


BAND = 32
BLOCK = 8


def _toy():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


def _cfg_for(band, mask, sparsity):
    return AttnConfig(seq_len=band, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=8, mask=mask, sparsity=sparsity,
                      block_size=BLOCK, dtype="float32",
                      mask_source="importance" if mask == "block_sparse" else None)


def _wrapped(model):
    return SwappableAttentionModel(model, _cfg_for(BAND, "causal", None),
                                    model_id="toy", finest_block_size=BLOCK)


def test_measure_band_runs_end_to_end_against_a_real_model(tmp_path):
    """The test that would have caught both failures, for free.

    It exercises the exact calls the harness makes: compute_importance_scores
    with a real cache_dir, run_measured with logits_to_keep, and generate with
    the stop sets disabled.
    """
    model = _toy()
    wrapped = _wrapped(model)
    ids = torch.randint(0, 64, (1, BAND))
    try:
        rows, recs, scoring_ms, prefill_ms = measure_band(
            wrapped, ids, band=BAND, arms=[("sdpa_math", None)],
            cfg_for=_cfg_for, scratch_dir=str(tmp_path),
            warmup=1, reps=2, scoring_reps=2, gen_lo=1, gen_hi=3,
            backend_factory=lambda name: SDPABackend(kernel="math"))
    finally:
        wrapped.unwrap()

    phases = {r.phase for r in rows}
    assert phases == {"scoring", "prefill", "decode_step"}, phases
    assert scoring_ms > 0.0
    assert prefill_ms[("sdpa_math", None)] > 0.0
    assert len(recs) == 1 and recs[0].n_generated == 3


def test_scoring_reps_each_miss_the_cache(tmp_path):
    """Failure 1, asserted rather than described.

    A shared example_id would hit after the first call and time a disk read.
    Every rep must write its own cache entry, so the file count is the proof.
    """
    model = _toy()
    wrapped = _wrapped(model)
    ids = torch.randint(0, 64, (1, BAND))
    try:
        measure_band(wrapped, ids, band=BAND, arms=[], cfg_for=_cfg_for,
                     scratch_dir=str(tmp_path), scoring_reps=4,
                     backend_factory=lambda n: SDPABackend(kernel="math"))
    finally:
        wrapped.unwrap()

    entries = list(tmp_path.glob("*.pt"))
    # 1 warmup + 4 timed + 1 final = 6 distinct ids, so 6 distinct entries.
    assert len(entries) == 6, (
        f"{len(entries)} cache entries: reps are sharing an id and timing a "
        f"cache read instead of a scoring pass")


def test_a_none_cache_dir_is_still_rejected_by_the_real_score_cache(tmp_path):
    """Failure 1's root cause, pinned so a future 'convenience' default of
    None reintroduces it loudly here rather than on an instance."""
    from attnbench.accuracy import score_cache
    with pytest.raises(TypeError):
        score_cache.load(None, "abc", expected_n_layers=2, expected_n_heads_kv=2)


def test_importance_scores_indexing_is_per_head_not_per_layer(tmp_path):
    """Failure 2, pinned as an interface fact.

    compute_importance_scores returns {layer: (n_heads_kv, n_blocks,
    n_blocks)}. masks.importance_block_mask wants ONE head's (n_blocks,
    n_blocks). `scores[layer]` is one index short -- that is exactly what
    crashed the sparse path on the instance, and the shapes are asserted here
    so the correct indexing is a checked fact rather than something to work
    out on a rented machine.
    """
    from attnbench import masks

    model = _toy()
    wrapped = _wrapped(model)
    ids = torch.randint(0, 64, (1, BAND))
    try:
        scores = wrapped.compute_importance_scores(
            ids, task="t", example_id="e", cache_dir=str(tmp_path))
    finally:
        wrapped.unwrap()

    n_blocks = BAND // BLOCK
    layer0 = scores[0]
    assert layer0.shape == (2, n_blocks, n_blocks), layer0.shape

    cfg = _cfg_for(BAND, "block_sparse", 0.5)
    with pytest.raises(ValueError, match="importance_scores shape"):
        masks.mask_for(cfg, importance_scores=layer0)

    # One more index. This is the call a mask_build phase would need.
    m = masks.mask_for(cfg, importance_scores=layer0[0])
    assert m is not None
