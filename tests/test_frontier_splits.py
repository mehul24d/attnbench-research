"""Gate G12 and split generation (estimator-frontier pre-registration sec. 5
and 6). The QA data is replaced by a small fake, so these run on a fresh
clone."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from attnbench.accuracy import frontier_splits as fs
from attnbench.accuracy import ruler, ruler_data, sizing
from attnbench.analysis import frontier_prereg as fp

REPO = Path(__file__).resolve().parents[1]
CAL = REPO / "scripts" / "calibrate_xattn_thresholds.py"


@pytest.fixture
def fake_squad(monkeypatch):
    """4,000 questions, one gold document each, all distinct."""
    docs = [f"Document number {i} opens with a sentence long enough to be probed, "
            f"about subject {i * 7919}." for i in range(4000)]
    qas = [{"query": f"What is the registered subject of record {i}?", "outputs": [str(i)],
            "context": [i], "more_context": []} for i in range(4000)]
    qas[3000]["query"] = "Where is it?"            # a short question, 12 characters
    monkeypatch.setattr(ruler_data, "qa_dataset", lambda name: (qas, docs))
    return qas, docs


def _script():
    spec = importlib.util.spec_from_file_location("cal", CAL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_g12_passes_on_texts_that_hold_nothing_held_out(fake_squad):
    out = fs.g12(["an unrelated calibration text " * 20])
    assert out["passed"] and out["n_indices"] == 300 + 32 + 100


@pytest.mark.parametrize("index,split", [(3007, "evaluation"), (1003, "selection"),
                                         (10, "t4_replication")])
def test_g12_refuses_a_held_out_question_or_its_gold_document(fake_squad, index, split):
    qas, docs = fake_squad
    for leak in (qas[index]["query"], docs[index]):
        with pytest.raises(fp.SplitLeak, match=split):
            fs.g12(["filler. " * 50, f"Document 3: {leak}\n\nmore filler"])


def test_g12_matches_a_short_question(fake_squad):
    """The case the 40-character floor skipped (2026-10-04)."""
    with pytest.raises(fp.SplitLeak, match="evaluation: 1 questions, 0 gold documents"):
        fs.g12(["Question: Where is it? Answer:"])


def test_g12_ignores_whitespace_and_a_calibration_index(fake_squad):
    qas, docs = fake_squad
    with pytest.raises(fp.SplitLeak):
        fs.g12([docs[3001].replace(" ", "\n  ")])
    assert fs.g12([docs[2003]])["passed"]          # calibration's own index is not held out


@pytest.mark.parametrize("texts", [[], None])
def test_g12_with_nothing_to_check_is_not_a_pass(fake_squad, texts):
    with pytest.raises(fp.SplitLeak):
        fs.g12(texts)


def test_g12_that_cannot_read_the_qa_data_raises(monkeypatch):
    def gone(name):
        raise ruler_data.RulerDataError("no data")
    monkeypatch.setattr(ruler_data, "qa_dataset", gone)
    with pytest.raises(ruler_data.RulerDataError):
        fs.g12(["text"])


def test_split_indices_are_the_registered_ranges():
    assert fs.split_indices("evaluation", "qa_1") == range(3000, 3300)
    assert fs.split_indices("evaluation", "niah_multivalue") == range(3000, 3050)
    assert fs.split_indices("selection", "qa_1") == range(1000, 1032)
    assert fs.split_indices("calibration", "qa_1") == range(2000, 2008)
    assert fs.split_indices("t4_replication", "qa_1") == range(0, 100)
    assert fs.split_indices("t4_replication", "niah_multiquery") == range(0, 50)


def test_generate_splits_cannot_return_without_calibration_texts(fake_squad):
    kw = dict(count_tokens=sizing.approximate_token_count, n_override={
        "calibration": 2, "selection": 2, "evaluation": 2})
    with pytest.raises(TypeError):
        fs.generate_splits(("niah_single",), (3000,), **kw)
    with pytest.raises(fp.SplitLeak):
        fs.generate_splits(("niah_single",), (3000,), calibration_texts=[], **kw)
    out = fs.generate_splits(("niah_single", "qa_1"), (3000,),
                             calibration_texts=["clean text " * 30], **kw)
    assert out["g12"]["passed"]
    ids = [e.example_id for e in out["evaluation"][("qa_1", 3000)]]
    assert ids == ["qa_1_3000_3000", "qa_1_3000_3001"]
    assert [e.example_id for e in out["selection"][("niah_single", 3000)]] == [
        "niah_single_3000_1000", "niah_single_3000_1001"]


def test_generate_splits_refuses_when_g12_fails(fake_squad):
    qas, docs = fake_squad
    with pytest.raises(fp.SplitLeak, match="G12"):
        fs.generate_splits(("qa_1",), (3000,), count_tokens=sizing.approximate_token_count,
                           calibration_texts=[docs[3100]],
                           n_override={"calibration": 2, "selection": 2, "evaluation": 2})


def test_the_calibration_script_runs_g12_before_it_returns_the_authors_texts(fake_squad):
    cal = _script()
    qas, docs = fake_squad
    with pytest.raises(SystemExit, match="STOP \\(G12\\)"):
        cal.g12_or_refuse([docs[5]])
    assert cal.g12_or_refuse(["clean " * 40])["passed"]
    # authors_texts() goes through authors_selection(), before its return.
    fn = next(n for n in ast.parse(CAL.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "authors_texts")
    calls = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "authors_selection"]
    returns = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert calls and max(calls) < min(returns)


def test_g12_that_cannot_run_stops_the_calibration_script(monkeypatch):
    def gone(name):
        raise ruler_data.RulerDataError("no data")
    monkeypatch.setattr(ruler_data, "qa_dataset", gone)
    with pytest.raises(SystemExit, match="has not passed"):
        _script().g12_or_refuse(["text"])


# --- the QA-text exclusion and the two authors' tables (2026-10-04) ----------

def _authors_like(fake_squad, n_qa=24, n_other=132):
    qas, docs = fake_squad
    qa_texts = [f"user {fs.QA_TEMPLATE_OPENING}. Only give me the answer.\n\nDocument 1: "
                f"{docs[3000 + k]}\n\nQuestion: {qas[k]['query']} Answer:" for k in range(n_qa)]
    other = [f"user A special magic number is hidden within the following text. {k} " * 5
             for k in range(n_other)]
    texts = other[:10] + qa_texts + other[10:]
    return texts, list(range(10, 10 + n_qa))


def test_the_claim_table_leaves_out_the_qa_texts_and_then_passes_g12(fake_squad, monkeypatch):
    cal = _script()
    texts, qa = _authors_like(fake_squad)
    assert fs.qa_text_indices(texts) == qa
    keep, record = cal.authors_selection(texts, fs.TEXT_JSON_SHA256, full=False)
    assert keep == [k for k in range(156) if k not in qa] and len(keep) == 132
    assert record["passed"] and record["qa_texts_excluded"] == qa
    assert record["needles"]["shared_key_value_pairs"] == 0


def test_the_full_set_table_is_made_and_records_that_g12_fails(fake_squad):
    cal = _script()
    texts, qa = _authors_like(fake_squad)
    keep, record = cal.authors_selection(texts, fs.TEXT_JSON_SHA256, full=True)
    assert keep == list(range(156)) and record["passed"] is False
    assert record["role"] == "descriptive"
    assert record["held_out_material"]["evaluation"]["gold_documents"] == 24
    assert record["held_out_material"]["t4_replication"]["questions"] == 24


def test_another_text_json_is_refused(fake_squad):
    cal = _script()
    texts, _ = _authors_like(fake_squad)
    with pytest.raises(SystemExit, match="not the text.json"):
        cal.authors_selection(texts, "0" * 64, full=False)
    with pytest.raises(SystemExit, match="not the text.json"):
        cal.authors_selection(texts[:-1], fs.TEXT_JSON_SHA256, full=True)
    fewer, _ = _authors_like(fake_squad, n_qa=23, n_other=133)
    with pytest.raises(SystemExit, match="23 QA texts"):
        cal.authors_selection(fewer, fs.TEXT_JSON_SHA256, full=False)


def test_a_leak_left_after_the_exclusion_still_stops_the_claim_table(fake_squad):
    cal = _script()
    qas, docs = fake_squad
    texts, _ = _authors_like(fake_squad)
    texts[0] = texts[0] + " " + docs[3100]         # a gold document in a non-QA text
    with pytest.raises(SystemExit, match="STOP \\(G12\\)"):
        cal.authors_selection(texts, fs.TEXT_JSON_SHA256, full=False)


def test_only_the_table_without_the_qa_texts_may_claim():
    from attnbench.accuracy import t4_pilot
    assert t4_pilot.XATTN_CALIBRATIONS == {"authors": "claim", "ruler_heldout": "descriptive",
                                           "authors_full": "descriptive"}


def test_a_needle_does_not_depend_on_the_haystack_length():
    """What scripts/check_g12_needles.py relies on."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("n", REPO / "scripts" / "check_g12_needles.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "niah_multikey" not in m.TASKS          # its haystack is made of needles
    for task in ("niah_single",):
        for ex in ruler.generate_examples(task, [3000], 3, seed=0,
                                          count_tokens=sizing.approximate_token_count,
                                          index_offset=3000):
            i = int(ex.example_id.rsplit("_", 1)[1])
            seed = ruler._example_seed(0, task, 3000, i)
            short, answer = ruler._render(task, seed, ruler._min_haystack_units(task) + 8, index=i)
            assert answer == ex.answer
            assert {v for _, v in m.needles(short)} >= set(ex.answer)
            assert {v for _, v in m.needles(ex.context)} >= set(ex.answer)
