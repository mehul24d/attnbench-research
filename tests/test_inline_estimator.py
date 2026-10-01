"""The inline estimator arm (audit C1, 2026-10-01), on a toy Llama.

`run_measured(..., inline_estimator="minference_meanpool")` makes every layer
of the measured forward score itself from its own q and k, build its mask on
the device, and run the kernel -- one pass, as a deployed method would. What
must hold, and what each test pins:

  - at layer 0 both arms see the same input, so the inline mask equals the
    two-pass cheap arm's mask exactly;
  - at every layer the inline mask is the reference builder's mask for the
    scores that layer actually computed (wiring, not just the builder);
  - at later layers the inline scores DIFFER from the two-pass ones, because
    their inputs come from sparse layers -- if they did not, the arm would be
    reading the cache, i.e. not inline at all;
  - with nothing pruned, the inline forward reproduces dense causal attention;
  - the setting is per call and cannot leak into the next arm.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench import masks  # noqa: E402
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.backends.impls import NaiveAttention  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

SEQ, BLOCK = 64, 8


def _model(n_layers=4):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=n_layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    m = LlamaForCausalLM(cfg)
    m.eval()
    return m


def _cfg(sparsity=0.75, **kw):
    base = dict(seq_len=SEQ, batch=1, n_heads_q=4, n_heads_kv=2, head_dim=8,
                dtype="float32", mask="block_sparse", block_size=BLOCK,
                sparsity=sparsity, mask_source="importance")
    base.update(kw)
    return AttnConfig(**base)


class _Recording(NaiveAttention):
    """NaiveAttention that keeps the mask each layer was given, in call order."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, q, k, v, cfg, mask=None):
        self.seen.append(mask)
        return super().forward(q, k, v, cfg, mask=mask)


def _ids():
    return torch.randint(0, 64, (1, SEQ), generator=torch.Generator().manual_seed(3))


def _wrapped(model, score_source="dense_softmax_fp32"):
    return SwappableAttentionModel(model, _cfg(), model_id="toy",
                                   finest_block_size=BLOCK, score_source=score_source)


def test_layer_zero_matches_the_two_pass_cheap_arm(tmp_path):
    model, ids, cfg = _model(), _ids(), _cfg()
    two_pass = _wrapped(model, "minference_meanpool")
    scores = two_pass.compute_importance_scores(ids, task="t", example_id="e",
                                                cache_dir=str(tmp_path))
    rec_two = _Recording()
    two_pass.run_measured(ids, rec_two, cfg=cfg, layer_scores=scores)

    rec_inline = _Recording()
    two_pass.run_measured(ids, rec_inline, cfg=cfg, inline_estimator="minference_meanpool")
    assert torch.equal(rec_two.seen[0].active, rec_inline.seen[0].active.cpu())


def test_every_layer_mask_is_the_reference_rule_on_its_own_scores():
    model, ids, cfg = _model(), _ids(), _cfg()
    w = _wrapped(model)
    w._state.record_inline_scores = True
    rec = _Recording()
    w.run_measured(ids, rec, cfg=cfg, inline_estimator="minference_meanpool")
    assert len(rec.seen) == len(model.model.layers)
    layer_cfg = replace(cfg, seq_len=SEQ, batch=1)
    for layer, mask in enumerate(rec.seen):
        importance = w._state.inline_scores[layer].mean(dim=0).cpu()
        ref = masks.mask_for(layer_cfg, importance_scores=importance)
        assert torch.equal(ref.active, mask.active.cpu()), layer


def test_later_layers_see_sparse_inputs_not_the_cached_scores(tmp_path):
    model, ids, cfg = _model(), _ids(), _cfg(sparsity=0.9)
    w = _wrapped(model, "minference_meanpool")
    cached = w.compute_importance_scores(ids, task="t", example_id="e",
                                         cache_dir=str(tmp_path))
    w._state.record_inline_scores = True
    w.run_measured(ids, _Recording(), cfg=cfg, inline_estimator="minference_meanpool")
    inline = {i: t.cpu() for i, t in w._state.inline_scores.items()}
    # A cache miss returns the pass's own fp32 scores, so layer 0 -- same
    # input, same ops -- must agree to rounding, and the later layers must
    # stand well clear of that. On this toy model the later-layer gap is
    # ~1e-4 (random weights barely care which blocks they skip); the
    # contrast, not the size, is the property.
    first = (inline[0] - cached[0].float()).abs().max().item()
    later = [(inline[i] - cached[i].float()).abs().max().item() for i in range(1, 4)]
    assert first < 1e-7, first
    assert min(later) > 1e-5 and min(later) > 100 * max(first, 1e-9), (first, later)


def test_nothing_pruned_reproduces_dense_causal_attention():
    model, ids = _model(), _ids()
    w = _wrapped(model)
    dense = w.run_measured(ids, NaiveAttention(), cfg=_cfg(mask="causal", sparsity=None,
                                                          mask_source="random"))
    inline = w.run_measured(ids, NaiveAttention(), cfg=_cfg(sparsity=0.0),
                            inline_estimator="minference_meanpool")
    assert torch.allclose(dense.logits, inline.logits, atol=1e-5)


def test_the_setting_does_not_leak_into_the_next_call(tmp_path):
    model, ids, cfg = _model(), _ids(), _cfg()
    w = _wrapped(model)
    w.run_measured(ids, _Recording(), cfg=cfg, inline_estimator="minference_meanpool")
    assert w._state.inline_estimator == "minference_meanpool"
    # The next arm passes nothing: it must read scores, and with none loaded
    # that is a KeyError, not a silent inline mask.
    w._state.scores = {}
    with pytest.raises(KeyError):
        w.run_measured(ids, _Recording(), cfg=cfg)
    assert w._state.inline_estimator is None


def test_refusals():
    model, ids = _model(), _ids()
    w = _wrapped(model)
    with pytest.raises(ValueError, match="unknown inline_estimator"):
        w.run_measured(ids, _Recording(), cfg=_cfg(), inline_estimator="xattention")
    with pytest.raises(ValueError, match="mask_source"):
        w.run_measured(ids, _Recording(), cfg=_cfg(mask_source="random"),
                       inline_estimator="minference_meanpool")
