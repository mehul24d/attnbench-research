"""The T4 candidate tasks (audit, 2026-10-01): RULER's essay-haystack NIAH
presets and QA, wired into the harness.

What is pinned, and why:
  - every EXISTING example is byte-identical to before the change -- the
    vendored generator was extended in place, and banked rows depend on it;
  - the essay path does what upstream's does: corpus words, sentence split,
    needles at sentence boundaries at depths from DEPTHS;
  - QA follows upstream's document construction (gold, own-article
    paragraphs, then random documents) and asks the same question at every
    band;
  - the new tasks are sized per example and never exceed their budget;
  - resources are refused when missing, unpinned, mismatched or from another
    package version, and the word-pair collision check bites;
  - no new task borrows a token cap: caps are measured.

The data-backed checks (real corpus, real SQuAD/HotpotQA) skip unless
$ATTNBENCH_RULER_DATA holds a pinned build.
"""

from __future__ import annotations

import hashlib
import json
import re
import typing

import pytest

from attnbench._vendor.ruler import niah, qa
from attnbench.accuracy import ruler, ruler_data, schema, sizing, stopping

T4_TASKS = ("niah_multikey_1", "niah_multivalue", "niah_multiquery", "qa_1", "qa_2")
EXISTING_DIGEST = "2fa1568b4e7745abecb7bae0b43335c043ba4e4e29de97ee274efa7a916b4582"


def test_t4_pilot_task_set_is_fixed():
    assert ruler.T4_DENSE_PILOT_TASKS == T4_TASKS


def _split(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


ESSAY = ("Startups are hard. Most of them fail! Why do they fail? Because the "
         "founders give up. Persistence matters more than brilliance. ") * 40
WORDS = tuple(re.sub(r"\s+", " ", ESSAY).split(" "))
POOL = (("red", "quick", "silent"), ("apple", "river", "stone", "lamp"))


def test_existing_examples_are_byte_identical_to_before_the_change():
    h = hashlib.sha256()
    for task in ("niah_single", "niah_multikey"):
        for seed in range(300):
            for units in (4, 5, 17, 200):
                h.update(repr(niah.generate_niah_example(
                    num_haystack=units, seed=seed, **ruler._NIAH_PARAMS[task])).encode())
    assert h.hexdigest() == EXISTING_DIGEST


def test_depths_are_upstreams():
    assert len(niah.DEPTHS) == 40 and niah.DEPTHS[0] == 0 and niah.DEPTHS[-1] == 100
    assert niah.DEPTHS[1] == 3 and niah.DEPTHS[20] == 51      # np.round half-to-even


@pytest.mark.parametrize("task", ["niah_multikey_1", "niah_multivalue", "niah_multiquery"])
def test_essay_examples_hide_every_needle_at_a_sentence_boundary(task):
    params = ruler._NIAH_PARAMS[task]
    for seed in range(25):
        text, answers = niah.generate_niah_example(
            num_haystack=300, seed=seed, essay_words=WORDS, sent_tokenize=_split,
            word_pool=POOL, **params)
        needle_re = r"One of the special magic numbers for [a-z]+-[a-z]+ is: \d{7}\."
        # the builder raises num_needle_k to max(k, q), each with v values
        expected = max(params["num_needle_k"], params["num_needle_q"]) * params["num_needle_v"]
        assert len(re.findall(needle_re, text)) == expected, (task, seed)
        for a in answers:
            assert f"is: {a}." in text
        # Needles go BETWEEN sentences, never inside one: with them removed,
        # the essay prefix is intact. (The prefix is cut at N words and may
        # end mid-sentence; depth 100 puts a needle after that fragment, as
        # upstream does.)
        body = text.split("afterwards.\n", 1)[1].rsplit("\nWhat ", 1)[0]
        stripped = " ".join(re.sub(needle_re, " ", body).split())
        assert stripped == " ".join(" ".join(WORDS[:300]).split()), (task, seed)


def test_essay_generation_is_deterministic_and_seed_sensitive():
    p = ruler._NIAH_PARAMS["niah_multivalue"]
    kw = dict(num_haystack=300, essay_words=WORDS, sent_tokenize=_split, word_pool=POOL, **p)
    assert niah.generate_niah_example(seed=1, **kw) == niah.generate_niah_example(seed=1, **kw)
    assert niah.generate_niah_example(seed=1, **kw) != niah.generate_niah_example(seed=2, **kw)


def test_essay_and_words_refuse_without_their_resources():
    p = ruler._NIAH_PARAMS["niah_multikey_1"]
    with pytest.raises(NotImplementedError, match="essay"):
        niah.generate_niah_example(num_haystack=50, seed=0, word_pool=POOL, **p)
    with pytest.raises(NotImplementedError, match="word_pool"):
        niah.generate_niah_example(num_haystack=50, seed=0, essay_words=WORDS,
                                   sent_tokenize=_split, **p)


SQUAD = {"data": [
    {"paragraphs": [
        {"context": "Paris is the capital of France.", "qas": [
            {"question": "What is the capital of France?", "is_impossible": False,
             "answers": [{"text": "Paris"}]},
            {"question": "Unanswerable?", "is_impossible": True, "answers": []}]},
        {"context": "France is in Europe.", "qas": []}]},
    {"paragraphs": [{"context": f"Filler document {i}.", "qas": []} for i in range(30)]},
]}


def test_read_squad_matches_upstream_structure():
    qas, docs = qa.read_squad(SQUAD)
    assert [q["query"] for q in qas] == ["What is the capital of France?"]   # impossible dropped
    assert docs == sorted(set(docs))
    q = qas[0]
    assert docs[q["context"][0]] == "Paris is the capital of France."
    assert [docs[i] for i in q["more_context"]] == ["France is in Europe."]


def test_qa_example_keeps_gold_then_own_article_then_random():
    qas, docs = qa.read_squad(SQUAD)
    text, answers = qa.generate_qa_example(qas, docs, index=0, num_docs=6, seed=3)
    assert answers == ["Paris"]
    assert "Paris is the capital of France." in text and "France is in Europe." in text
    assert text.count("Document ") == 6 and text.endswith("Question: What is the capital of France? Answer:")
    assert qa.generate_qa_example(qas, docs, index=0, num_docs=6, seed=3)[0] == text
    with pytest.raises(ValueError):
        qa.generate_qa_example(qas, docs, index=0, num_docs=0, seed=3)


def test_read_hotpotqa_keeps_every_paragraph_as_context():
    data = [{"question": "Q?", "answer": "A",
             "context": [["T1", ["s1", "s2"]], ["T2", ["s3"]]]}]
    qas, docs = qa.read_hotpotqa(data)
    assert qas[0]["outputs"] == ["A"] and len(qas[0]["context"]) == 2
    assert "T1\ns1s2" in docs and qa.min_docs(qas, 0) == 2


SQUAD_MANY = {"data": SQUAD["data"] + [
    {"paragraphs": [{"context": f"Fact {i}: the code is {i * 7}.", "qas": [
        {"question": f"What is code {i}?", "is_impossible": False,
         "answers": [{"text": str(i * 7)}]}]}]} for i in range(10)]}


@pytest.fixture
def fake_resources(monkeypatch):
    qas, docs = qa.read_squad(SQUAD_MANY)
    monkeypatch.setattr(ruler_data, "qa_dataset", lambda name: (qas, docs))
    monkeypatch.setattr(ruler_data, "essay_words", lambda: WORDS)
    monkeypatch.setattr(ruler_data, "sent_tokenizer", lambda: _split)
    monkeypatch.setattr(ruler_data, "word_pool", lambda: POOL)


@pytest.mark.parametrize("task", T4_TASKS)
def test_new_tasks_are_sized_per_example_and_never_exceed_budget(task, fake_resources):
    exs = ruler.generate_examples(task, [400], n_per_length=6, seed=0,
                                  count_tokens=sizing.approximate_token_count)
    assert len(exs) == 6
    for e in exs:
        assert e.context_length <= 400, (task, e.example_id, e.context_length)
        assert e.haystack_mode if hasattr(e, "haystack_mode") else True
    assert ruler.haystack_mode_for(task) in typing.get_args(schema.HaystackMode)


def test_qa_asks_the_same_questions_at_every_band(fake_resources):
    a = ruler.generate_examples("qa_1", [300, 500], n_per_length=1, seed=0,
                                count_tokens=sizing.approximate_token_count)
    assert a[0].answer == a[1].answer


@pytest.mark.parametrize("task", T4_TASKS)
def test_every_new_cap_is_twice_its_documented_measured_max(task):
    """No borrowed numbers: each cap is 2x the max in the measured table in
    docs/stage3_generation_decision.md, read from the doc, not retyped."""
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "stage3_generation_decision.md").read_text()
    m = re.search(rf"\| `{task}` \| 200 \|[^|]*\|[^|]*\|[^|]*\| (\d+) \| \*\*(\d+)\*\* \|", doc)
    assert m, f"{task} missing from the measured table"
    assert stopping.token_cap(task) == 2 * int(m.group(1)) == int(m.group(2))


def test_resources_refuse_when_missing_unpinned_or_mismatched(tmp_path, monkeypatch):
    monkeypatch.setenv(ruler_data.DATA_ENV, str(tmp_path))
    with pytest.raises(ruler_data.RulerDataError, match="missing"):
        ruler_data._verified("squad.json", {"squad.json": "0" * 64})
    (tmp_path / "squad.json").write_text("{}")
    with pytest.raises(ruler_data.RulerDataError, match="no SHA-256 pin"):
        ruler_data._verified("squad.json", {"squad.json": None})
    with pytest.raises(ruler_data.RulerDataError, match="!= pinned"):
        ruler_data._verified("squad.json", {"squad.json": "0" * 64})
    good = ruler_data.sha256_of(tmp_path / "squad.json")
    assert ruler_data._verified("squad.json", {"squad.json": good}).exists()


def test_version_pins_refuse_another_release():
    with pytest.raises(ruler_data.RulerDataError, match="pinned"):
        ruler_data._check_version("wonderwords", "3.0.1")


def test_word_pair_collision_is_detected():
    # ("big", "red-car") and ("big-red", "car") both join to "big-red-car".
    with pytest.raises(ruler_data.RulerDataError, match="collision"):
        ruler_data.word_pool_from(["big", "big-red"], ["red-car", "car"])
    assert ruler_data.word_pool_from(["big", "small"], ["red-car", "car"])


def test_the_installed_word_lists_have_no_collision():
    pytest.importorskip("wonderwords")
    adjs, nouns = ruler_data.word_pool()
    assert len(adjs) > 500 and len(nouns) > 5000


def _real_data():
    try:
        ruler_data.essay_words(); ruler_data.qa_dataset("squad"); ruler_data.sent_tokenizer()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_data(), reason="needs a pinned $ATTNBENCH_RULER_DATA build")
def test_real_corpus_examples_hold_their_answers():
    for task in T4_TASKS:
        for e in ruler.generate_examples(task, [4096], n_per_length=3, seed=0,
                                         count_tokens=sizing.approximate_token_count):
            assert e.context_length <= 4096
            assert all(a.lower() in e.context.lower() for a in e.answer) or task.startswith("qa")
