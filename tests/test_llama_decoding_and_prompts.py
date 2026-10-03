"""Llama-3.1-8B decoding, stop set and prompt encoding (estimator-frontier
pre-registration, sec. 4.9, lock gate L7; added 2026-10-03).

The model's shipped generation_config.json samples (do_sample true,
temperature 0.6, top_p 0.9) and lists three EOS ids, while its tokenizer
reports one EOS and prepends a BOS. Each of those is a way for a Llama row to
mean something other than what the study says it means:

- **Greedy.** Every Llama arm decodes greedily. The wrapper forces it and
  asserts it, and the same prompt twice gives byte-identical output.
- **Stop set.** All three ids from generation_config.json. The tokenizer's
  128009 alone is refused, and the id that fired is recorded on the row.
- **Encoding.** The sizer counts exactly the ids fed to the model, per
  example. A double BOS is refused.

Everything here runs on CPU on a toy Llama with the G9 rope settings. The last
section checks the real pinned tokenizer, and skips unless it is in the
local HF cache.
"""
from __future__ import annotations

import hashlib
import inspect

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench.accuracy import ruler, stopping
from attnbench.accuracy.generation import ModelGeometry, StopTokens, generate_one
from attnbench.accuracy.model import (
    GREEDY_REQUIRED_MODELS, SwappableAttentionModel, assert_greedy, force_greedy)
from attnbench.accuracy.prompting import (
    DoubleBOSError, encode_prompt, prompt_ids, prompt_token_counter)
from attnbench.backends.impls import SDPABackend
from attnbench.config import AttnConfig

LLAMA = stopping.LLAMA_31_8B
LLAMA_EOS = frozenset({128001, 128008, 128009})
VOCAB = 64
BOS = 1
NEWLINE_ID = 5
SPACE_ID = 6


def _toy_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=1024, attn_implementation="eager",
                      rope_theta=500000.0,
                      rope_scaling={"rope_type": "llama3", "factor": 8.0,
                                    "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                    "original_max_position_embeddings": 64})
    model = LlamaForCausalLM(cfg).eval()
    # The shipped Llama-3.1-8B-Instruct generation_config.json, sampling fields
    # and all (read 2026-10-03 at revision 0e9e39f).
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p = True, 0.6, 0.9
    return model


def _cfg(seq_len=16):
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=8, dtype="float32", mask="causal")


def _wrapped(model_id=LLAMA):
    model = _toy_llama()
    return SwappableAttentionModel(model, _cfg(), model_id=model_id,
                                   finest_block_size=8)


def _ids(n=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, VOCAB, (1, n), generator=g)


class _WordTokenizer:
    """Word-level stand-in that behaves like Llama-3's tokenizer where it
    matters here: `add_special_tokens=True` prepends BOS, and the literal BOS
    text is parsed as the BOS id."""

    bos_token_id = BOS
    eos_token_id = 63
    BOS_TEXT = "<|begin_of_text|>"

    def __init__(self):
        vocab = [f"w{i}" for i in range(VOCAB)]
        vocab[NEWLINE_ID], vocab[SPACE_ID] = "\n", " "
        self._vocab = vocab

    def __len__(self):
        return VOCAB

    def _encode(self, text):
        ids = []
        while text.startswith(self.BOS_TEXT):
            ids.append(BOS)
            text = text[len(self.BOS_TEXT):]
        for w in text.split():
            ids.append(2 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (VOCAB - 3))
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = ([BOS] if add_special_tokens else []) + self._encode(text)
        out = torch.tensor([ids]) if return_tensors == "pt" else ids
        return type("Enc", (), {"input_ids": out})()

    def batch_decode(self, batches, skip_special_tokens=False):
        return [self.decode(b) for b in batches]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._vocab[i] for i in ids)


@pytest.fixture(autouse=True)
def _as_if_on_the_image(monkeypatch):
    """The Llama path refuses any transformers but the image's pin
    (accuracy/pins.py), and these tests run on the workstation's newer one.
    Every test here therefore says explicitly that it stands in for the
    image. The refusal itself is tested below with the real version."""
    import transformers
    from attnbench.accuracy.pins import PINNED_TRANSFORMERS
    monkeypatch.setattr(transformers, "__version__", PINNED_TRANSFORMERS[LLAMA])


# --- transformers pin ----------------------------------------------------------

def test_llama_generation_refuses_another_transformers(monkeypatch):
    import transformers
    from attnbench.accuracy.pins import TransformersVersionMismatch
    monkeypatch.setattr(transformers, "__version__", "5.18.0")
    with pytest.raises(TransformersVersionMismatch, match="4.46.0"):
        _wrapped().generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=3)
    # Other models are not pinned.
    _wrapped(model_id="toy").generate(_ids(), SDPABackend("math"), cfg=_cfg(),
                                      max_new_tokens=3)


def test_the_runner_refuses_to_score_llama_rows_off_the_pin(monkeypatch, tmp_path):
    import transformers
    from attnbench.accuracy import runner
    from attnbench.accuracy.pins import TransformersVersionMismatch
    from attnbench.accuracy.runner import AccuracyCell
    from attnbench.accuracy.schema import Generated
    ex = ruler.RulerExample(task="niah_single", example_id="niah_single_400_0",
                            context="c", question="", answer=["42"], context_length=1)
    cfg = AttnConfig(seq_len=400, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=128, mask="causal")
    cell = AccuracyCell(cfg=cfg, backend_name="sdpa_math", task="niah_single",
                        example_id=ex.example_id)
    gen = lambda *a: Generated(text="42", latency_ms=1.0)  # noqa: E731
    monkeypatch.setattr(transformers, "__version__", "5.18.0")
    with pytest.raises(TransformersVersionMismatch):
        runner.run_accuracy([cell], out_dir=tmp_path, examples_by_id={
            ("niah_single", ex.example_id): ex}, generate_fn=gen,
            allow_dirty=True, model_id=LLAMA)
    assert not (tmp_path / "accuracy.parquet").exists()


def test_every_row_records_the_transformers_version():
    from attnbench import provenance
    assert provenance.capture().transformers is not None
    assert "transformers" in provenance.capture().to_dict()


def test_banked_rows_load_the_new_columns_as_null(tmp_path):
    """Rows written before 2026-10-03 have neither column. The loader gives
    them typed nulls; new rows keep their integers."""
    import pandas as pd
    from attnbench.accuracy.schema import load_accuracy_parquet, normalise_accuracy_frame
    old = pd.DataFrame({"stop_reason": ["eos", "cap"], "task": ["vt", "vt"]})
    old.to_parquet(tmp_path / "old.parquet")
    df = load_accuracy_parquet(tmp_path / "old.parquet")
    assert str(df.stop_token_id.dtype) == "Int64" and df.stop_token_id.isna().all()
    assert df.transformers.isna().all()
    new = normalise_accuracy_frame(pd.DataFrame(
        {"stop_reason": ["eos", "cap"], "stop_token_id": [128009, None],
         "transformers": ["4.46.0", "4.46.0"]}))
    both = normalise_accuracy_frame(pd.concat([df, new], ignore_index=True))
    assert both.stop_token_id.tolist()[2] == 128009
    assert both.stop_token_id.isna().tolist() == [True, True, False, True]
    assert (both.stop_token_id == 128009).sum() == 1


# --- greedy -----------------------------------------------------------------

def test_the_wrapper_forces_greedy_on_llama():
    w = _wrapped()
    gc = w.model.generation_config
    assert gc.do_sample is False
    assert gc.temperature is None and gc.top_p is None and gc.top_k is None
    assert LLAMA in GREEDY_REQUIRED_MODELS


@pytest.mark.parametrize("field,value", [("do_sample", True), ("temperature", 0.6),
                                         ("top_p", 0.9)])
def test_a_sampling_parameter_reaching_a_llama_arm_is_refused(field, value):
    w = _wrapped()
    setattr(w.model.generation_config, field, value)
    with pytest.raises(RuntimeError, match="not greedy"):
        w.generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=3)


def test_generate_takes_no_sampling_arguments():
    params = inspect.signature(SwappableAttentionModel.generate).parameters
    assert not {"do_sample", "temperature", "top_p", "top_k"} & set(params)
    assert not any(p.kind is p.VAR_KEYWORD for p in params.values())


def test_without_the_override_the_shipped_config_is_refused():
    """Non-vacuity: the toy carries the shipped sampling config, so the
    guard has something to refuse."""
    model = _toy_llama()
    with pytest.raises(RuntimeError, match="not greedy"):
        assert_greedy(model)
    force_greedy(model)
    assert_greedy(model)


def test_llama_decoding_is_deterministic():
    """Same prompt twice, byte-identical: token ids, stop, and decoded text."""
    tok = _WordTokenizer()
    w = _wrapped()
    a = w.generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=12)
    b = w.generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=12)
    assert a.token_ids == b.token_ids and a.stop_reason == b.stop_reason
    assert tok.decode(a.token_ids).encode() == tok.decode(b.token_ids).encode()
    # A fresh wrapper over a freshly built model agrees too.
    c = _wrapped().generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=12)
    assert c.token_ids == a.token_ids


# --- stop set ----------------------------------------------------------------

def test_llama_stops_on_all_three_generation_config_ids():
    tok = _WordTokenizer()
    tok.eos_token_id = 128009
    gc = type("GC", (), {"eos_token_id": [128001, 128008, 128009]})()
    st = StopTokens.from_tokenizer(tok, gc, model_id=LLAMA)
    assert st.eos == LLAMA_EOS


@pytest.mark.parametrize("gc_eos", [None, [128009], [128001, 128009]])
def test_the_tokenizer_eos_alone_is_refused_for_llama(gc_eos):
    tok = _WordTokenizer()
    tok.eos_token_id = 128009
    gc = None if gc_eos is None else type("GC", (), {"eos_token_id": gc_eos})()
    with pytest.raises(ValueError, match="must stop on"):
        StopTokens.from_tokenizer(tok, gc, model_id=LLAMA)


def test_other_models_keep_the_union_rule():
    tok = _WordTokenizer()
    assert StopTokens.from_tokenizer(tok, None, model_id="toy").eos == {63}


def test_the_firing_id_is_recorded():
    w = _wrapped()
    first = w.generate(_ids(), SDPABackend("math"), cfg=_cfg(),
                       max_new_tokens=4).token_ids
    r = w.generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=4,
                   eos_token_ids=frozenset({first[0], 999}))
    assert (r.stop_reason, r.stop_token_id) == ("eos", first[0])
    r = w.generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=4,
                   newline_token_ids=frozenset({first[2]}))
    assert (r.stop_reason, r.stop_token_id) == ("newline", first[2])
    r = w.generate(_ids(), SDPABackend("math"), cfg=_cfg(), max_new_tokens=2)
    assert (r.stop_reason, r.stop_token_id) == ("cap", None)


# --- encoding: sizer == fed ids, and no double BOS -----------------------------

def _run(example, tok):
    w = _wrapped()
    fed = {}
    real = w.generate

    def spy(input_ids, backend, **kw):
        fed["ids"] = input_ids[0].tolist()
        return real(input_ids, backend, **kw)

    w.generate = spy
    geometry = ModelGeometry.from_config(w.model.config, "float32")
    cell = AttnConfig(seq_len=example.token_budget, batch=1, n_heads_q=1,
                      n_heads_kv=1, head_dim=128, mask="causal")
    st = StopTokens.from_tokenizer(tok, None, model_id="toy")
    gen = generate_one(w, tok, cfg=cell, backend=SDPABackend("math"),
                       example=example, geometry=geometry, stop_tokens=st,
                       score_cache_dir="unused", device="cpu")
    return fed["ids"], gen


@pytest.mark.parametrize("task", ["niah_single", "niah_multikey", "vt"])
def test_the_sizer_counts_exactly_the_ids_fed_to_the_model(task):
    tok = _WordTokenizer()
    budget = 400
    exs = ruler.generate_examples(task, [budget], n_per_length=3, seed=0,
                                  count_tokens=prompt_token_counter(tok),
                                  per_example_fit=True)
    for ex in exs:
        fed, gen = _run(ex, tok)
        assert len(fed) == ex.context_length <= budget, ex.example_id
        assert fed[0] == BOS and fed[1] != BOS
        assert gen.stop_reason in ("eos", "newline", "cap")


def test_a_sizer_that_skips_the_bos_is_caught_at_generation():
    """The failure the shared encoder exists for: counting with
    add_special_tokens=False undercounts every prompt by one."""
    tok = _WordTokenizer()
    undercount = lambda text: len(tok(text, add_special_tokens=False).input_ids)  # noqa: E731
    ex = ruler.generate_examples("niah_single", [400], n_per_length=1, seed=0,
                                 count_tokens=undercount, per_example_fit=True)[0]
    with pytest.raises(RuntimeError, match="are fed to the model"):
        _run(ex, tok)


def test_a_double_bos_is_refused():
    tok = _WordTokenizer()
    assert prompt_ids(tok, "hello world")[:2] == [BOS, prompt_ids(tok, "hello")[1]]
    with pytest.raises(DoubleBOSError):
        prompt_ids(tok, tok.BOS_TEXT + "hello world")
    with pytest.raises(DoubleBOSError):
        encode_prompt(tok, tok.BOS_TEXT + "hello world")
    # The one exception, the conditional arm I2c-B, says so explicitly.
    assert prompt_ids(tok, tok.BOS_TEXT + "hello world", allow_double_bos=True)[:2] == [BOS, BOS]


# --- the real pinned tokenizer, when cached --------------------------------------

REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
NEWLINE_DIGEST = ("dc24e2c3056c70b4", 2255)
WHITESPACE_DIGEST = ("445dc415bf20b800", 530)


@pytest.fixture(scope="module")
def llama_tok():
    from transformers import AutoTokenizer, GenerationConfig
    try:
        tok = AutoTokenizer.from_pretrained(LLAMA, revision=REVISION, local_files_only=True)
        gc = GenerationConfig.from_pretrained(LLAMA, revision=REVISION, local_files_only=True)
    except Exception:  # not cached: gated, and never fetched by a test
        pytest.skip("the pinned Llama-3.1-8B tokenizer is not in the local HF cache")
    return tok, gc


def _digest(ids):
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()[:16], len(ids)


def test_real_tokenizer_facts(llama_tok):
    tok, gc = llama_tok
    assert tok.bos_token_id == 128000 and tok.eos_token_id == 128009
    assert list(gc.eos_token_id) == [128001, 128008, 128009]
    assert prompt_ids(tok, "hello") == [128000, 15339]
    st = StopTokens.from_tokenizer(tok, gc, model_id=LLAMA)
    assert st.eos == LLAMA_EOS
    assert _digest(st.newline) == NEWLINE_DIGEST
    assert _digest(st.whitespace) == WHITESPACE_DIGEST


def test_the_authors_prompt_format_would_double_the_bos(llama_tok):
    """XAttention's RULER template and every text.json prompt begin with a
    literal <|begin_of_text|> and are tokenized with the default
    add_special_tokens=True. This study refuses that input."""
    tok, _ = llama_tok
    authors = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
               "Q?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    assert tok(authors).input_ids[:2] == [128000, 128000]
    with pytest.raises(DoubleBOSError):
        prompt_ids(tok, authors)
    assert prompt_ids(tok, authors[len("<|begin_of_text|>"):])[:2] == [128000, 128006]
