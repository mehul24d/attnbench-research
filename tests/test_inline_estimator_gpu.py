"""GPU variants of the two GQA tests in test_inline_estimator.py (T2.26).

UNRUN: written in a container without torch or a GPU. Same assertions and
bounds as the CPU tests, on the device and dtype `_gpu_env` reads from the
environment; they skip without that device. The bounds are kept: layer 0
compares one computation with itself on one device, and the later-layer
contrast only grows in half precision.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

import _gpu_env  # noqa: E402
from test_inline_estimator import BLOCK, SEQ, _Recording  # noqa: E402

pytestmark = _gpu_env.needs_gpu


def _model(n_layers=4):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=n_layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval().to(device=_gpu_env.device(),
                                            dtype=_gpu_env.dtype())


def _cfg(sparsity=0.75):
    return AttnConfig(seq_len=SEQ, batch=1, n_heads_q=4, n_heads_kv=2, head_dim=8,
                      dtype=_gpu_env.dtype_name(), mask="block_sparse",
                      block_size=BLOCK, sparsity=sparsity, mask_source="importance")


def _ids():
    ids = torch.randint(0, 64, (1, SEQ), generator=torch.Generator().manual_seed(3))
    return ids.to(_gpu_env.device())


def _wrapped(model, score_source):
    return SwappableAttentionModel(model, _cfg(), model_id="toy",
                                   finest_block_size=BLOCK, score_source=score_source)


def test_gpu_layer_zero_matches_the_two_pass_cheap_arm(tmp_path):
    model, ids, cfg = _model(), _ids(), _cfg()
    two_pass = _wrapped(model, "minference_meanpool")
    scores = two_pass.compute_importance_scores(ids, task="t", example_id="e",
                                                cache_dir=str(tmp_path))
    rec_two = _Recording()
    two_pass.run_measured(ids, rec_two, cfg=cfg, layer_scores=scores)
    rec_inline = _Recording()
    two_pass.run_measured(ids, rec_inline, cfg=cfg, inline_estimator="minference_meanpool")
    assert torch.equal(rec_two.seen[0].active.cpu(), rec_inline.seen[0].active.cpu())


def test_gpu_later_layers_see_sparse_inputs_not_the_cached_scores(tmp_path):
    model, ids, cfg = _model(), _ids(), _cfg(sparsity=0.9)
    w = _wrapped(model, "minference_meanpool")
    cached = w.compute_importance_scores(ids, task="t", example_id="e",
                                         cache_dir=str(tmp_path))
    w._state.record_inline_scores = True
    w.run_measured(ids, _Recording(), cfg=cfg, inline_estimator="minference_meanpool")
    inline = {i: t.float().cpu() for i, t in w._state.inline_scores.items()}
    first = (inline[0] - cached[0].float().cpu()).abs().max().item()
    later = [(inline[i] - cached[i].float().cpu()).abs().max().item() for i in range(1, 4)]
    assert first < 1e-7, first
    assert min(later) > 1e-5 and min(later) > 100 * max(first, 1e-9), (first, later)
