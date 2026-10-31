"""GPU variants of the two GQA tests in test_generation_wiring.py (T2.26).

UNRUN: written in a container without torch or a GPU. Same assertions as
the CPU tests, on the device and dtype `_gpu_env` reads from the
environment; they skip without that device.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench.accuracy.generation import (  # noqa: E402
    ModelGeometry, StopTokens, generate_one)
from attnbench.accuracy.grid_configs import DENSE_DECODE_BACKEND  # noqa: E402
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.accuracy.ruler import RulerExample  # noqa: E402
from attnbench.backends.impls import NaiveAttention  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

import _gpu_env  # noqa: E402
from test_generation_wiring import _FakeTokenizer  # noqa: E402

pytestmark = _gpu_env.needs_gpu


def _wrapped(prompt_len=16):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=256, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval().to(device=_gpu_env.device(),
                                             dtype=_gpu_env.dtype())
    geometry = ModelGeometry.from_config(model.config, _gpu_env.dtype_name())
    template = geometry.onto(_cell_cfg(), seq_len=prompt_len)
    return SwappableAttentionModel(model, template, model_id="toy",
                                   finest_block_size=8), geometry


def _cell_cfg(**over):
    base = dict(seq_len=2048, batch=1, n_heads_q=1, n_heads_kv=1,
                head_dim=128, mask="causal")
    base.update(over)
    return AttnConfig(**base)


def _example():
    return RulerExample(task="niah_single", example_id="ex0", context="the prompt",
                        question="", answer=["42"], context_length=2048)


def test_gpu_a_backend_with_no_decode_path_gets_the_dense_one_and_says_so():
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    got = generate_one(wrapped, tokenizer, cfg=_cell_cfg(),
                       backend=NaiveAttention(), example=_example(),
                       geometry=geometry, stop_tokens=StopTokens.from_tokenizer(tokenizer),
                       score_cache_dir="unused", device=_gpu_env.device())
    assert got.decode_backend == DENSE_DECODE_BACKEND
    assert got.decode_backend != NaiveAttention().name


def test_gpu_a_pinned_fallback_decodes_through_it_and_the_row_says_pinned():
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    kw = dict(cfg=_cell_cfg(), backend=NaiveAttention(), example=_example(),
              geometry=geometry, stop_tokens=StopTokens.from_tokenizer(tokenizer),
              score_cache_dir="unused", device=_gpu_env.device())
    pinned = generate_one(wrapped, tokenizer, pinned_fallback_decode="sdpa_math", **kw)
    assert pinned.decode_backend == "sdpa_math" and pinned.decode_pinned is True
    current = generate_one(wrapped, tokenizer, **kw)
    assert current.decode_backend == DENSE_DECODE_BACKEND and current.decode_pinned is False
