"""XAttention from a cell to a row (T4 XAttention phase, 2026-10-02).

The backend itself is pinned in tests/test_xattention_backend.py. What these
tests pin is everything between it and the parquet, run on a toy Llama with
the official package and the kernel replaced by stand-ins:

  - the row records the threshold that ran and the density the prefill
    produced, read off the backend instance, one entry per layer;
  - the row is a sparse arm (`backend_role="block_sparse"`, its scorer
    stamped), never the dense reference it is compared against;
  - a row that cannot be placed against the fixed-sparsity arms (no
    threshold, no density) cannot be built, and no other row can carry them;
  - a second threshold cannot resume into the first's rows.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace as dc_replace

import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench import provenance
from attnbench.accuracy.generation import ModelGeometry, StopTokens, generate_one
from attnbench.accuracy.model import SwappableAttentionModel
from attnbench.accuracy.ruler import RulerExample
from attnbench.accuracy.runner import (ThresholdMismatch, build_cells,
                                       check_xattn_threshold_continuity,
                                       run_accuracy)
from attnbench.accuracy.schema import AccuracyResult, Generated
from attnbench.accuracy.t4_pilot import XATTN_SCORE_SOURCE
from attnbench.backends.xattention import XAttentionBackend
from attnbench.config import AttnConfig

VOCAB, PROMPT = 64, 300          # 300 tokens -> 3 blocks of 128


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


@pytest.fixture
def fake_xattn(monkeypatch):
    """Stand-ins that keep the diagonal blocks only (density 3/6 of the
    causally valid blocks) and run dense attention for the output."""
    def xattn_estimate(q, k, **kw):
        nb = -(-q.shape[2] // 128)
        m = torch.zeros(1, k.shape[1], nb, nb, dtype=torch.bool)
        m[:, :, range(nb), range(nb)] = True
        return torch.zeros(1), m

    def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming,
                               mask, sq, sk, **kw):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        return out.transpose(0, 1).contiguous()

    pkg = types.ModuleType("xattn"); src = types.ModuleType("xattn.src")
    mod = types.ModuleType("xattn.src.Xattention"); mod.xattn_estimate = xattn_estimate
    bsa = types.ModuleType("block_sparse_attn"); bsa.block_sparse_attn_func = block_sparse_attn_func
    for name, m in (("xattn", pkg), ("xattn.src", src), ("xattn.src.Xattention", mod),
                    ("block_sparse_attn", bsa)):
        monkeypatch.setitem(sys.modules, name, m)


def _wrapped():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=512, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    geometry = ModelGeometry.from_config(model.config, "float32")
    template = geometry.onto(_cell(), seq_len=PROMPT)
    return SwappableAttentionModel(model, template, model_id="toy",
                                   score_source=XATTN_SCORE_SOURCE), geometry


def _cell():
    return AttnConfig(seq_len=2048, batch=1, n_heads_q=1, n_heads_kv=1,
                      head_dim=128, mask="causal")


def _generate(backend, tmp_path):
    wrapped, geometry = _wrapped()
    ex = RulerExample(task="qa_1", example_id="e0", context="c", question="",
                      answer=["1"], context_length=PROMPT)
    tok = _Tok()
    gen = generate_one(wrapped, tok, cfg=_cell(), backend=backend, example=ex,
                       geometry=geometry, stop_tokens=StopTokens.from_tokenizer(tok),
                       score_cache_dir=str(tmp_path), device="cpu",
                       pinned_fallback_decode="sdpa_math")
    return gen, wrapped


def test_the_row_carries_the_threshold_and_this_prefills_density(fake_xattn, tmp_path):
    backend = XAttentionBackend(threshold=0.9)
    backend.last_layer_density = [0.123] * 7          # a previous example's
    gen, wrapped = _generate(backend, tmp_path)
    assert len(backend.last_layer_density) == wrapped.n_layers
    assert gen.xattn_threshold == 0.9
    assert gen.realised_density == pytest.approx(3 / 6)
    assert gen.decode_backend == "sdpa_math"          # decode never reaches it


def test_a_dense_row_carries_neither(tmp_path):
    from attnbench.backends.impls import SDPABackend
    gen, _ = _generate(SDPABackend("math"), tmp_path)
    assert gen.xattn_threshold is None and gen.realised_density is None


def _row(**kw):
    base = dict(backend="xattention", backend_role="block_sparse", config_key="k",
                task="qa_1", example_id="e0", context_length=300, mask_source=None,
                sparsity=None, score_source=XATTN_SCORE_SOURCE, haystack_mode=None,
                predicted="1", expected="1", score=100.0, correct=True,
                xattn_threshold=0.9, realised_density=0.4)
    base.update(kw)
    return AccuracyResult(**base)


def test_the_schema_refuses_rows_that_cannot_be_placed():
    _row()
    with pytest.raises(ValueError, match="selects its own blocks"):
        _row(realised_density=None)
    with pytest.raises(ValueError, match="selects its own blocks"):
        _row(xattn_threshold=None)
    with pytest.raises(ValueError, match="does not select its own blocks"):
        _row(backend="sdpa_flash", backend_role="dense_reference", score_source=None)


def _cells():
    cfg = AttnConfig(seq_len=1024, batch=1, n_heads_q=8, n_heads_kv=8,
                     head_dim=64, mask="causal")
    ex = RulerExample(task="qa_1", example_id="e0", context="c", question="",
                      answer=["1"], context_length=1024)
    cells = build_cells(configs_by_backend={"sdpa_flash": [cfg], "xattention": [cfg]},
                        examples_by_task_length={("qa_1", 1024): [ex]})
    return cells, {("qa_1", "e0"): ex}


def _gen(cfg, backend_name, example, tau=0.9):
    if backend_name == "xattention":
        return Generated("1", latency_ms=1.0, xattn_threshold=tau, realised_density=0.4)
    return Generated("1", latency_ms=1.0)


def _run(tmp_path, tau, gen=_gen):
    cells, by_id = _cells()
    prov = lambda: dc_replace(provenance.capture(), git_commit="a" * 40, git_dirty=False)
    return run_accuracy(cells, out_dir=tmp_path, examples_by_id=by_id,
                        generate_fn=lambda c, b, e: gen(c, b, e, tau),
                        provenance_fn=prov, score_source=XATTN_SCORE_SOURCE,
                        xattn_threshold=tau)


def test_an_xattention_row_is_a_sparse_arm_not_the_reference(tmp_path):
    _run(tmp_path, 0.9)
    df = pd.read_parquet(tmp_path / "accuracy.parquet").set_index("backend")
    assert df.loc["xattention", "backend_role"] == "block_sparse"
    assert df.loc["xattention", "score_source"] == XATTN_SCORE_SOURCE
    assert df.loc["xattention", "xattn_threshold"] == 0.9
    assert df.loc["xattention", "realised_density"] == 0.4
    assert df.loc["sdpa_flash", "backend_role"] == "dense_reference"
    assert pd.isna(df.loc["sdpa_flash", "score_source"])


def test_a_second_threshold_cannot_resume_into_the_first(tmp_path):
    _run(tmp_path, 0.9)
    with pytest.raises(ThresholdMismatch):
        _run(tmp_path, 0.8)
    check_xattn_threshold_continuity(tmp_path / "accuracy.parquet", xattn_threshold=0.9)
    with pytest.raises(ThresholdMismatch):
        check_xattn_threshold_continuity(tmp_path / "accuracy.parquet", xattn_threshold=None)


def test_backend_instance_requires_and_confines_the_threshold():
    from attnbench.accuracy.grid_configs import backend_instance
    assert backend_instance("xattention", xattn_threshold=0.8).threshold == 0.8
    with pytest.raises(ValueError, match="pass its threshold"):
        backend_instance("xattention")
    with pytest.raises(ValueError, match="takes no threshold"):
        backend_instance("sdpa_flash", xattn_threshold=0.9)
