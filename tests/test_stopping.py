"""The stopping rule: caps, the newline set, and the two implementations of
the rule agreeing.

The rule lives in two places by necessity -- `stopping.first_stop_index` is
the pure statement of it, `SwappableAttentionModel.generate` is the version
that runs inside a decode loop and cannot be exercised without a model. Two
implementations of one rule is a drift hazard, so the last test here holds
them to the same answer on the same token stream rather than trusting them
to stay in step.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench.accuracy.stopping import (
    TASK_TOKEN_CAPS, eos_token_ids, first_stop_index, newline_token_ids,
    token_cap)


# --- caps -------------------------------------------------------------------

def test_caps_are_per_task_and_cover_the_grid():
    """A flat budget generous enough for niah_multikey would be ~10x
    niah_single's whole answer, which lets a rambling backend look identical
    to a terse one."""
    from attnbench.accuracy.config import load_grid
    grid = load_grid("configs/accuracy/stage3_grid.yaml")
    assert set(grid.tasks) <= set(TASK_TOKEN_CAPS)
    # Not one flat budget. This asserted every cap DISTINCT until 2026-10-01,
    # a proxy that fails on legitimately equal measurements: niah_multikey_1
    # answers one 7-digit number exactly as niah_single does (cap 14 both),
    # and niah_multivalue / niah_multiquery both answer four (62 both). That
    # each cap is 2x a measured max is pinned per task from the documented
    # table by tests/test_ruler_t4_tasks.py.
    assert len(set(TASK_TOKEN_CAPS.values())) > 1
    assert max(TASK_TOKEN_CAPS.values()) >= 4 * min(TASK_TOKEN_CAPS.values())


def test_an_unmeasured_task_raises_rather_than_borrowing_a_cap():
    """A default would truncate a longer-answered task silently, and a
    truncated answer scores identically to a wrong one."""
    with pytest.raises(KeyError, match="measured answer lengths"):
        token_cap("qa")


def test_llama_never_borrows_the_qwen_caps():
    """Estimator-frontier pre-registration, sec. 4.9 (2026-10-03): Llama-3.1-8B
    caps come from its own table, measured with its tokenizer. A Llama lookup
    returns that table's value, never Qwen's, and a task missing from it
    raises even where Qwen has a cap."""
    from attnbench.accuracy import stopping
    assert stopping.LLAMA_31_8B == "meta-llama/Llama-3.1-8B-Instruct"
    llama = stopping.LLAMA_31_8B_TASK_TOKEN_CAPS
    for task in TASK_TOKEN_CAPS:
        if task in llama:
            assert token_cap(task, stopping.LLAMA_31_8B) == llama[task]
        else:
            with pytest.raises(KeyError, match="not reused"):
                token_cap(task, stopping.LLAMA_31_8B)
    # The two tokenizers really differ: a borrowed table would be wrong here.
    assert any(llama[t] != TASK_TOKEN_CAPS[t] for t in llama)
    try:
        stopping.LLAMA_31_8B_TASK_TOKEN_CAPS.pop("qa_2")
        with pytest.raises(KeyError, match="not reused"):
            token_cap("qa_2", stopping.LLAMA_31_8B)
    finally:
        stopping.LLAMA_31_8B_TASK_TOKEN_CAPS["qa_2"] = 46


@pytest.mark.parametrize("task", sorted(
    __import__("attnbench.accuracy.stopping", fromlist=["x"]).LLAMA_31_8B_TASK_TOKEN_CAPS))
def test_every_llama_cap_is_twice_its_documented_measured_max(task):
    """Each Llama cap is 2x the max in the measured table in the
    pre-registration (sec. 4.9), read from the doc, not retyped."""
    import re
    from pathlib import Path
    from attnbench.accuracy import stopping
    doc = (Path(__file__).resolve().parents[1] / "docs"
           / "estimator_frontier_preregistration.md").read_text()
    m = re.search(rf"\| `{task}` \| 200 \|[^|]*\|[^|]*\|[^|]*\| (\d+) \| \*\*(\d+)\*\* \| Llama \|", doc)
    assert m, f"{task} missing from the Llama measured table"
    assert token_cap(task, stopping.LLAMA_31_8B) == 2 * int(m.group(1)) == int(m.group(2))


def test_other_models_keep_the_qwen_caps():
    """Qwen ids, the toy test models (LlamaForCausalLM, id "toy") and None
    all read TASK_TOKEN_CAPS, so banked-run behaviour is unchanged."""
    for model_id in (None, "Qwen/Qwen2.5-1.5B-Instruct", "toy", "toy/llama"):
        assert token_cap("qa_1", model_id) == TASK_TOKEN_CAPS["qa_1"]


def test_caps_are_at_least_twice_the_measured_maximum():
    """The rule the numbers came from, asserted so an edit has to keep it.
    Observed maxima over 200 examples/task: 7 / 36 / 20."""
    observed_max = {"niah_single": 7, "niah_multikey": 36, "vt": 20}
    for task, longest in observed_max.items():
        assert TASK_TOKEN_CAPS[task] >= 2 * longest, task


# --- deriving the token sets ------------------------------------------------

class _FakeTokenizer:
    """Stands in for a real BPE tokenizer. The point of these tests is that
    the newline set is DERIVED from the vocabulary rather than hardcoded, so
    a fake vocabulary is the right instrument -- a real one would only prove
    it works for one model."""

    def __init__(self, vocab: list[str], eos=None):
        self._vocab = vocab
        self.eos_token_id = eos

    def __len__(self):
        return len(self._vocab)

    def batch_decode(self, batches, skip_special_tokens=False):
        return ["".join(self._vocab[i] for i in b) for b in batches]


def test_newline_set_includes_every_token_containing_a_newline():
    """Not just the bare "\\n". A byte-level BPE vocabulary merges newlines
    onto neighbouring text, and missing those is the dangerous direction:
    the stop silently never fires and every example runs to its cap."""
    tok = _FakeTokenizer(["hello", "\n", ".\n", "\n\n", " ", "world", "a\nb"])
    assert newline_token_ids(tok) == frozenset({1, 2, 3, 6})


def test_a_vocabulary_with_no_newline_token_yields_an_empty_set_not_a_crash():
    tok = _FakeTokenizer(["a", "b", "c"])
    assert newline_token_ids(tok) == frozenset()


def test_eos_takes_the_union_of_tokenizer_and_generation_config():
    """They disagree in practice -- Qwen2.5-Instruct's tokenizer reports
    <|im_end|> while its generation_config lists <|endoftext|> too."""
    class _GenCfg:
        eos_token_id = [151645, 151643]
    tok = _FakeTokenizer(["a"], eos=151645)
    assert eos_token_ids(tok, _GenCfg()) == frozenset({151645, 151643})
    assert eos_token_ids(tok) == frozenset({151645})


def test_eos_survives_a_tokenizer_with_none():
    assert eos_token_ids(_FakeTokenizer(["a"])) == frozenset()


# --- the rule itself --------------------------------------------------------

NL = frozenset({9})
WS = frozenset({7})
EOS = frozenset({99})


def test_stops_at_the_first_newline_after_content():
    assert first_stop_index([1, 2, 9, 3], newline_ids=NL, eos_ids=EOS) == (2, "newline")


def test_eos_wins_over_everything():
    assert first_stop_index([99, 9], newline_ids=NL, eos_ids=EOS) == (0, "eos")


def test_nothing_stops_a_stream_with_no_stop_token():
    assert first_stop_index([1, 2, 3], newline_ids=NL, eos_ids=EOS) is None


def test_a_leading_newline_does_not_end_an_empty_answer():
    """The refinement to the confirmed rule. Both RULER templates end
    mid-sentence with a trailing space so this is unlikely -- but if it ever
    happened, the rule as literally written would stop having generated
    nothing, and every row of every backend would score 0 for a reason with
    nothing to do with attention."""
    assert first_stop_index([9, 7, 1, 9], newline_ids=NL, eos_ids=EOS,
                            whitespace_ids=WS) == (3, "newline")


def test_leading_whitespace_also_does_not_arm_the_stop():
    assert first_stop_index([7, 7, 9], newline_ids=NL, eos_ids=EOS,
                            whitespace_ids=WS) is None


def test_without_the_whitespace_set_a_leading_newline_still_stops_late():
    """The arming rule is about content, not about the whitespace set being
    supplied -- a newline is itself not content."""
    assert first_stop_index([9, 1, 9], newline_ids=NL, eos_ids=EOS) == (2, "newline")


# --- the two implementations agree ------------------------------------------

def _toy():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


def _cfg(seq_len=16):
    from attnbench.config import AttnConfig
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=2,
                      head_dim=8, dtype="float32", mask="causal")


@pytest.mark.parametrize("stop_at", [0, 1, 3])
def test_generate_and_the_pure_rule_agree_on_the_same_stream(stop_at):
    """Two implementations of one rule drift. This pins them together on a
    real decode loop: whatever tokens the model produces, declaring the
    token at `stop_at` a newline must stop generation exactly where
    first_stop_index says it should -- including at index 0, where the
    arming refinement means it must NOT stop."""
    from attnbench.accuracy.model import SwappableAttentionModel
    from attnbench.backends.impls import SDPABackend

    model = _toy()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 64, (1, 16), generator=g)
    wrapped = SwappableAttentionModel(model, _cfg(), model_id="toy",
                                      finest_block_size=8)

    baseline = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                                max_new_tokens=6).token_ids
    newline = frozenset({baseline[stop_at]})
    expected = first_stop_index(baseline, newline_ids=newline, eos_ids=frozenset())

    got = wrapped.generate(ids, SDPABackend("math"), cfg=_cfg(),
                           max_new_tokens=6, newline_token_ids=newline)

    if expected is None:
        assert got.stop_reason == "cap"
    else:
        index, reason = expected
        assert got.stop_reason == reason
        assert got.n_generated == index + 1
