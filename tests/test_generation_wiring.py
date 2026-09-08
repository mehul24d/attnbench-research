"""The per-cell execution path, exercised on CPU.

This is the code that turns a grid cell into a result row, and it is where
the expensive mistakes live: a placeholder head geometry left un-substituted,
a cap taken from the wrong task, a decode backend chosen silently, a latency
that timed a kernel launch. None of those raise. All of them produce a full
parquet of plausible rows.

`generation.generate_one` exists as a module-level function rather than a
closure inside scripts/run_accuracy.py specifically so this file can run it
against a toy model and a fake tokenizer, before a rented hour is spent
finding out.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench.accuracy.generation import (
    ModelGeometry, StopTokens, generate_one)
from attnbench.accuracy.model import SwappableAttentionModel
from attnbench.accuracy.ruler import RulerExample
from attnbench.accuracy.schema import Generated
from attnbench.accuracy.grid_configs import DENSE_DECODE_BACKEND
from attnbench.backends.impls import NaiveAttention, SDPABackend
from attnbench.config import AttnConfig

VOCAB = 64
NEWLINE_ID = 5
SPACE_ID = 6


class _FakeTokenizer:
    """Character-level stand-in. A real tokenizer would need a download and
    would only prove the path works for one model; what these tests are
    about is the substitutions, which are tokenizer-independent."""

    def __init__(self, prompt_len=16):
        self._prompt_len = prompt_len
        self.eos_token_id = 63
        vocab = [f"t{i}" for i in range(VOCAB)]
        vocab[NEWLINE_ID] = "\n"
        vocab[SPACE_ID] = " "
        self._vocab = vocab

    def __len__(self):
        return VOCAB

    def __call__(self, text, return_tensors=None):
        g = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
        ids = torch.randint(0, VOCAB, (1, self._prompt_len), generator=g)
        return type("Enc", (), {"input_ids": ids})()

    def batch_decode(self, batches, skip_special_tokens=False):
        return ["".join(self._vocab[i] for i in b) for b in batches]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._vocab[i] for i in ids)


def _wrapped(prompt_len=16):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=256, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    geometry = ModelGeometry.from_config(model.config, "float32")
    template = geometry.onto(_cell_cfg(), seq_len=prompt_len)
    return SwappableAttentionModel(model, template, model_id="toy",
                                    finest_block_size=8), geometry


def _cell_cfg(**over):
    """A cell config as build_configs_by_backend really produces one --
    placeholder geometry included, because that is the input this path has
    to correct."""
    base = dict(seq_len=2048, batch=1, n_heads_q=1, n_heads_kv=1,
                head_dim=128, mask="causal")
    base.update(over)
    return AttnConfig(**base)


def _example(task="niah_single", example_id="ex0"):
    return RulerExample(task=task, example_id=example_id, context="the prompt",
                        question="", answer=["42"], context_length=2048)


def _stop_tokens(tokenizer):
    return StopTokens.from_tokenizer(tokenizer)


# --- geometry substitution --------------------------------------------------

def test_placeholder_geometry_is_replaced_by_the_real_models():
    """A cell's config says n_heads_q=1, head_dim=128. The model has 4 and 8.
    Running the cell's numbers would either crash or -- worse -- silently
    reshape into a different attention pattern."""
    geometry = ModelGeometry(n_heads_q=4, n_heads_kv=2, head_dim=8, dtype="float32")
    out = geometry.onto(_cell_cfg(mask="block_sparse", sparsity=0.9,
                                  block_size=128, mask_source="importance"),
                        seq_len=1900)
    assert (out.n_heads_q, out.n_heads_kv, out.head_dim) == (4, 2, 8)
    assert out.seq_len == 1900        # the REAL tokenized length, not the grid's
    # everything that makes this cell what it is survives
    assert out.mask == "block_sparse"
    assert out.sparsity == 0.9
    assert out.block_size == 128
    assert out.mask_source == "importance"


def test_the_real_tokenized_length_wins_over_the_grids_nominal_one():
    """Sizing fits contexts to a budget, so the two are close but not equal.
    Using the nominal length would build a mask for a sequence that isn't
    the one being attended."""
    tokenizer = _FakeTokenizer(prompt_len=23)
    wrapped, geometry = _wrapped(prompt_len=23)
    seen = {}
    real_generate = wrapped.generate

    def spy(input_ids, backend, **kw):
        seen["seq_len"] = kw["cfg"].seq_len
        return real_generate(input_ids, backend, **kw)

    wrapped.generate = spy
    generate_one(wrapped, tokenizer, cfg=_cell_cfg(seq_len=2048),
                 backend=SDPABackend("math"), example=_example(),
                 geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                 score_cache_dir="unused", device="cpu")
    assert seen["seq_len"] == 23


# --- caps and stopping ------------------------------------------------------

def test_each_task_gets_its_own_cap_not_a_shared_budget():
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    caps = {}
    real_generate = wrapped.generate

    def spy(input_ids, backend, **kw):
        caps[kw["cfg"].seq_len] = kw["max_new_tokens"]
        return real_generate(input_ids, backend, **kw)

    for task, expected in (("niah_single", 14), ("niah_multikey", 72), ("vt", 40)):
        wrapped.generate = spy
        caps.clear()
        generate_one(wrapped, tokenizer, cfg=_cell_cfg(),
                     backend=SDPABackend("math"), example=_example(task=task),
                     geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                     score_cache_dir="unused", device="cpu")
        assert list(caps.values()) == [expected], task


def test_the_stop_sets_come_from_the_vocabulary():
    tokens = _stop_tokens(_FakeTokenizer())
    assert NEWLINE_ID in tokens.newline
    assert SPACE_ID in tokens.whitespace
    assert tokens.eos == frozenset({63})
    # The two sets overlap and that is fine -- "\n".strip() is empty, so a
    # newline is whitespace-only by construction. What matters is that a
    # newline never counts as content: the arming rule excludes newline ids
    # first, so the overlap cannot arm the stop on a leading newline.
    assert NEWLINE_ID in tokens.whitespace
    from attnbench.accuracy.stopping import first_stop_index
    assert first_stop_index([NEWLINE_ID, SPACE_ID, NEWLINE_ID],
                            newline_ids=tokens.newline, eos_ids=tokens.eos,
                            whitespace_ids=tokens.whitespace) is None


# --- what lands on the row --------------------------------------------------

def test_generate_one_returns_a_populated_generated():
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    produced = {}
    real_generate = wrapped.generate

    def spy(input_ids, backend, **kw):
        result = real_generate(input_ids, backend, **kw)
        produced["token_ids"] = result.token_ids
        return result

    wrapped.generate = spy
    got = generate_one(wrapped, tokenizer, cfg=_cell_cfg(),
                       backend=SDPABackend("math"), example=_example(),
                       geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                       score_cache_dir="unused", device="cpu")
    assert isinstance(got, Generated)
    assert got.stop_reason in ("eos", "newline", "cap")
    assert 0 < got.n_generated == len(produced["token_ids"]) <= 14
    assert got.decode_backend == "sdpa_math"
    assert got.latency_ms is not None and got.latency_ms > 0
    # Text, not ids. ruler.score() compares strings, and a row carrying
    # "[32, 36, 6]" would score 0 against every answer while looking like a
    # model that simply got it wrong.
    assert got.text == tokenizer.decode(produced["token_ids"],
                                        skip_special_tokens=True)
    assert got.text != ""


def test_a_backend_with_no_decode_path_gets_the_dense_one_and_says_so():
    """Decision C on the row: sparsity is applied during prefill only, so
    the text of a sparse row is generated by a dense kernel over the cache.
    The row has to record that, or the comparison silently attributes decode
    behaviour to a backend that never decoded."""
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    got = generate_one(wrapped, tokenizer, cfg=_cell_cfg(),
                       backend=NaiveAttention(), example=_example(),
                       geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                       score_cache_dir="unused", device="cpu")
    # Against the constant, not a literal. The identity of the fallback is
    # grid_configs' decision (it changed sdpa_math -> sdpa_flash on
    # 2026-09-08); what this test owns is that a backend with no decode path
    # gets it and the row says so. A hardcoded name here was a fourth copy of
    # that rule, living in the test layer -- see silent_failure_patterns #23.
    assert got.decode_backend == DENSE_DECODE_BACKEND
    # ...and that it is not the backend under test, which is the property a
    # constant-valued assertion would otherwise stop checking.
    assert got.decode_backend != NaiveAttention().name


def test_latency_synchronizes_on_both_sides_of_the_timer():
    """Forgetting this does not raise -- it times a kernel launch instead of
    a kernel, in the direction that flatters whichever backend queues the
    most work. Asserted by counting calls, since on CPU there is nothing to
    synchronize and the omission would be invisible."""
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    calls = []
    generate_one(wrapped, tokenizer, cfg=_cell_cfg(),
                 backend=SDPABackend("math"), example=_example(),
                 geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                 score_cache_dir="unused", device="cpu",
                 synchronize=lambda: calls.append(1))
    assert len(calls) == 2


def test_a_task_with_no_measured_cap_stops_the_run():
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    with pytest.raises(KeyError, match="measured answer lengths"):
        generate_one(wrapped, tokenizer, cfg=_cell_cfg(),
                     backend=SDPABackend("math"), example=_example(task="cwe"),
                     geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                     score_cache_dir="unused", device="cpu")


def test_a_dense_cell_never_touches_the_score_cache():
    """The scoring pass is the most expensive thing in Stage 3 and a dense
    row has no use for it. Running it anyway would roughly double the
    dense arm's cost and nothing would look wrong."""
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    wrapped.compute_importance_scores = lambda *a, **k: pytest.fail(
        "a causal (dense) cell ran the importance-scoring pass")
    generate_one(wrapped, tokenizer, cfg=_cell_cfg(mask="causal"),
                 backend=SDPABackend("math"), example=_example(),
                 geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                 score_cache_dir="unused", device="cpu")


def test_a_block_sparse_cell_does_score_and_passes_the_scores_through():
    tokenizer = _FakeTokenizer()
    wrapped, geometry = _wrapped()
    sentinel = {"called": 0}
    real = wrapped.compute_importance_scores

    def counting(*a, **k):
        sentinel["called"] += 1
        return real(*a, **k)

    wrapped.compute_importance_scores = counting
    seen = {}
    real_generate = wrapped.generate

    def spy(input_ids, backend, **kw):
        seen["layer_scores"] = kw["layer_scores"]
        return real_generate(input_ids, backend, **kw)

    wrapped.generate = spy
    import tempfile
    with tempfile.TemporaryDirectory() as cache_dir:
        generate_one(wrapped, tokenizer,
                     cfg=_cell_cfg(mask="block_sparse", sparsity=0.5,
                                   block_size=8, mask_source="importance"),
                     backend=SDPABackend("math"), example=_example(),
                     geometry=geometry, stop_tokens=_stop_tokens(tokenizer),
                     score_cache_dir=cache_dir, device="cpu")
    assert sentinel["called"] == 1
    assert seen["layer_scores"] is not None
