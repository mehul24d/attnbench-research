"""The inline estimator arm, from the CLI's score source to the row.

`run_measured(..., inline_estimator=...)` existed from 66c53b9 but nothing
called it: no score source selected it, so the arm could not be run or timed.
`score_source="minference_meanpool_inline"` now selects it, and these tests
pin the three things that make that wiring mean what it says:

  - no scoring pass runs (its cost would be outside the timer, and its
    ranking would replace the inline one);
  - every prefill layer gets a mask built from its OWN scores, inside the
    timed generate call;
  - a run cannot resume into another scorer's rows, which share every
    resume-key field with it.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench import masks
from attnbench.accuracy.generation import ModelGeometry, StopTokens, generate_one
from attnbench.accuracy.model import SwappableAttentionModel
from attnbench.accuracy.ruler import RulerExample
from attnbench.accuracy.runner import (ScoreSourceMismatch,
                                       check_score_source_continuity)
from attnbench.backends.impls import NaiveAttention
from attnbench.config import AttnConfig

VOCAB, PROMPT, BLOCK = 64, 32, 8


class _Tok:
    eos_token_id = 63

    def __len__(self):
        return VOCAB

    def __call__(self, text, return_tensors=None):
        ids = torch.randint(0, VOCAB, (1, PROMPT), generator=torch.Generator().manual_seed(1))
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join("\n" if i == 5 else f"t{i}" for i in ids)

    def batch_decode(self, batches, skip_special_tokens=False):
        return [self.decode(b) for b in batches]


class _Recording(NaiveAttention):
    def __init__(self):
        super().__init__()
        self.prefill_masks = []

    def forward(self, q, k, v, cfg, mask=None):
        if q.shape[2] > 1:
            self.prefill_masks.append(mask)
        return super().forward(q, k, v, cfg, mask=mask)


def _wrapped(score_source):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=256, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    geometry = ModelGeometry.from_config(model.config, "float32")
    template = geometry.onto(_cell(), seq_len=PROMPT)
    return SwappableAttentionModel(model, template, model_id="toy",
                                   finest_block_size=BLOCK,
                                   score_source=score_source), geometry


def _cell():
    return AttnConfig(seq_len=2048, batch=1, n_heads_q=1, n_heads_kv=1,
                      head_dim=128, mask="block_sparse", sparsity=0.5,
                      block_size=BLOCK, mask_source="importance")


def _run(wrapped, geometry, backend, tmp_path):
    ex = RulerExample(task="qa_1", example_id="e0", context="c", question="",
                      answer=["1"], context_length=PROMPT)
    tok = _Tok()
    return generate_one(wrapped, tok, cfg=_cell(), backend=backend, example=ex,
                        geometry=geometry, stop_tokens=StopTokens.from_tokenizer(tok),
                        score_cache_dir=str(tmp_path), device="cpu",
                        # Decode is dense and not under test; sdpa_math runs
                        # GQA on every CPU torch, flash does not.
                        pinned_fallback_decode="sdpa_math")


def test_inline_runs_no_scoring_pass_and_masks_every_layer_from_its_own_scores(tmp_path):
    wrapped, geometry = _wrapped("minference_meanpool_inline")
    wrapped.compute_importance_scores = lambda *a, **k: pytest.fail("scoring pass ran")
    wrapped._state.record_inline_scores = True
    backend = _Recording()
    _run(wrapped, geometry, backend, tmp_path)

    assert len(backend.prefill_masks) == wrapped.n_layers
    cfg = geometry.onto(_cell(), seq_len=PROMPT)
    for layer, mask in enumerate(backend.prefill_masks):
        assert mask is not None and mask.source == "importance"
        own = wrapped._state.inline_scores[layer]
        want = masks.importance_block_mask(
            PROMPT, BLOCK, 0.5, own.mean(dim=0).cpu(), causal=True,
            identity_seed=masks._mask_identity_key(cfg))
        assert torch.equal(mask.active.cpu(), want.active), layer


def test_the_oracle_source_still_takes_the_scoring_pass(tmp_path):
    wrapped, geometry = _wrapped("dense_softmax_fp32")
    called = []
    real = wrapped.compute_importance_scores
    wrapped.compute_importance_scores = lambda *a, **k: called.append(1) or real(*a, **k)
    _run(wrapped, geometry, _Recording(), tmp_path)
    assert called == [1]
    assert wrapped._state.inline_estimator is None


def test_the_scoring_pass_refuses_a_source_it_does_not_implement(tmp_path):
    """Called with the inline source, the pass would otherwise fall through to
    the dense oracle and cache that under the inline name."""
    wrapped, _ = _wrapped("minference_meanpool_inline")
    ids = torch.zeros(1, PROMPT, dtype=torch.long)
    with pytest.raises(ValueError, match="no scoring pass"):
        wrapped.compute_importance_scores(ids, task="t", example_id="e",
                                          cache_dir=str(tmp_path))


def _ckpt(tmp_path, sources):
    p = tmp_path / "accuracy.parquet"
    pd.DataFrame({"backend": ["x"] * len(sources), "score_source": sources}).to_parquet(p)
    return p


def test_score_source_guard(tmp_path):
    check_score_source_continuity(tmp_path / "none.parquet", score_source="dense_softmax_fp32")
    p = _ckpt(tmp_path, [None, "dense_softmax_fp32"])
    check_score_source_continuity(p, score_source="dense_softmax_fp32")
    with pytest.raises(ScoreSourceMismatch):
        check_score_source_continuity(p, score_source="minference_meanpool_inline")
    # dense-only rows say nothing about the scorer, so any arm may follow them
    check_score_source_continuity(_ckpt(tmp_path, [None, None]),
                                  score_source="minference_meanpool_inline")


def test_a_second_arm_into_the_first_arms_out_is_refused_before_it_skips(tmp_path):
    """The failure the guard exists for, end to end: an oracle run banks its
    sparse rows, then an inline run pointed at the same --out would find all
    of them "done". It must refuse instead, including on a dry run."""
    from dataclasses import replace as dc_replace

    from attnbench import provenance
    from attnbench.accuracy.runner import build_cells, run_accuracy
    from attnbench.accuracy.schema import Generated

    cfg = AttnConfig(seq_len=1024, batch=1, n_heads_q=8, n_heads_kv=8,
                     head_dim=64, mask="block_sparse", sparsity=0.5,
                     block_size=128, mask_source="importance")
    ex = RulerExample(task="qa_1", example_id="e0", context="c", question="",
                      answer=["1"], context_length=1024)
    cells = build_cells(configs_by_backend={"block_sparse": [cfg]},
                        examples_by_task_length={("qa_1", 1024): [ex]})
    prov = lambda: dc_replace(provenance.capture(), git_commit="a" * 40, git_dirty=False)
    kw = dict(out_dir=tmp_path, examples_by_id={("qa_1", "e0"): ex},
              generate_fn=lambda c, b, e: Generated("1", latency_ms=1.0),
              provenance_fn=prov)
    run_accuracy(cells, score_source="dense_softmax_fp32", **kw)
    for dry in (True, False):
        with pytest.raises(ScoreSourceMismatch):
            run_accuracy(cells, score_source="minference_meanpool_inline",
                         dry_run=dry, **kw)
