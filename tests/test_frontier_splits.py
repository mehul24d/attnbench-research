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
    # authors_texts() calls it, and before its return.
    fn = next(n for n in ast.parse(CAL.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "authors_texts")
    calls = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "g12_or_refuse"]
    returns = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert calls and max(calls) < min(returns)


def test_g12_that_cannot_run_stops_the_calibration_script(monkeypatch):
    def gone(name):
        raise ruler_data.RulerDataError("no data")
    monkeypatch.setattr(ruler_data, "qa_dataset", gone)
    with pytest.raises(SystemExit, match="has not passed"):
        _script().g12_or_refuse(["text"])
