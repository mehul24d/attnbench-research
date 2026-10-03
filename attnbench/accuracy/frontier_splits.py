"""The estimator-frontier study's splits, and gate G12 on them
(pre-registration sec. 5 and 6).

`generate_splits` is the only place the study's calibration, selection and
evaluation examples are generated. It cannot return examples without having
run the four-identity disjointness check and gate G12 on the calibration
texts it is given: there is no path through it that skips either.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping, Optional, Sequence

from ..analysis.frontier_prereg import SPLIT_INDEX, SplitLeak, check_splits_disjoint
from . import ruler, ruler_data, sizing

# sec. 5: the seed of each split. The index offsets are SPLIT_INDEX.
SPLIT_SEED = {"calibration": 1, "selection": 2, "evaluation": 0, "t4_replication": 0}
# Examples per (task, band). Evaluation is 300 for the primary task and 50 for
# the others; calibration uses 8 of its reserved 48.
N_PER_BAND = {"calibration": 8, "selection": 32}
EVAL_N = {"qa_1": 300}
EVAL_N_DEFAULT = 50
# G12 probes a calibration text for a held-out question and for the opening
# of each of its gold documents.
GOLD_OPENING_CHARS = 160
G12_SPLITS = ("evaluation", "selection", "t4_replication")


def split_n(split: str, task: str) -> int:
    if split == "evaluation":
        return EVAL_N.get(task, EVAL_N_DEFAULT)
    if split == "t4_replication":
        return 100 if task == "qa_1" else 50
    return N_PER_BAND[split]


def split_indices(split: str, task: str) -> range:
    first, most = SPLIT_INDEX[split]
    n = split_n(split, task)
    if n > most:
        raise ValueError(f"{split} holds {most} indices; {task} asks for {n}")
    return range(first, first + n)


def identities(task: str, examples: Iterable[ruler.RulerExample]) -> list[dict]:
    """The four identities of sec. 5, per example."""
    out = []
    for ex in examples:
        row = {"example_id": ex.example_id,
               "context_sha256": hashlib.sha256(ex.context.encode()).hexdigest()}
        m = re.search(r"_(\d+)$", ex.example_id)
        if task.startswith("qa_") and m:
            row["qa_question"] = ((task, int(m.group(1))),)
        if task.startswith("niah"):
            row["niah_needles"] = [(task, a) for a in ex.answer]
        out.append(row)
    return out


# The opening of RULER's QA template (`_vendor/ruler/qa.py`). A calibration
# text that carries it is a QA prompt, and is excluded from the claim table
# (amendment of 2026-10-04: sec. 5 here, A8 in T4).
QA_TEMPLATE_OPENING = "Answer the question based on the given documents"
# The file the exclusion and the needle comparison were worked out on:
# x-attention's text.json at e379887. Another file is refused until both are
# redone for it.
TEXT_JSON_SHA256 = "d11899123f032af35abb23515eed685833b2a6853d483e04a6205c6b6296d50a"
TEXT_JSON_N_TEXTS = 156
TEXT_JSON_N_QA_TEXTS = 24


# The needle half of G12, run on a workstation with scripts/check_g12_needles.py
# against that file. Every needle in its 96 needle texts (18,361 distinct)
# was compared with every needle this harness generates for four needle
# tasks (not niah_multikey), in every split, at 16384 and 32768. No (key,
# value) pair is shared.
# Six 7-digit values recur under other keys, against 5.7 expected by chance.
NEEDLE_COMPARISON = {
    "checked": "2026-10-04", "text_json_sha256": TEXT_JSON_SHA256,
    "calibration_needles": 18361, "shared_key_value_pairs": 0,
    "shared_values": {"t4_replication": 2, "selection": 0, "calibration": 2, "evaluation": 2},
    "shared_values_expected_by_chance": {"t4_replication": 2.0, "selection": 1.3,
                                         "calibration": 0.3, "evaluation": 2.0},
    "rule": "pass if no (key, value) pair is shared",
}


def qa_text_indices(texts: Sequence[str]) -> list[int]:
    """Indices of the calibration texts that are QA prompts."""
    return [k for k, t in enumerate(texts) if QA_TEMPLATE_OPENING in _norm(t[:600])]


QUESTION_MIN_CHARS = 12
DOCUMENT_MIN_CHARS = 40


def _norm(text: str) -> str:
    return " ".join(text.split())


def g12_hits(calibration_texts: Sequence[str], *, tasks: Sequence[str] = ("qa_1",),
             splits: Sequence[str] = G12_SPLITS) -> list[dict]:
    """Every held-out QA index whose question, or the opening of one of whose
    gold documents, occurs in a calibration text. A question is matched at
    any length from 12 characters: until 2026-10-04 the shared 40-character
    floor skipped SQuAD question 0 (36 characters), which texts 0, 26, 52, 78,
    106 and 130 of the authors' `text.json` ask (sha256 TEXT_JSON_SHA256)."""
    texts = [_norm(t) for t in calibration_texts]
    hits = []
    for task in tasks:
        if not task.startswith("qa_"):
            continue
        qas, docs = ruler_data.qa_dataset(ruler._QA_PARAMS[task])
        found = {}                      # probe -> text indices, cached across splits

        def where(probe, floor):
            probe = _norm(probe)
            if len(probe) < floor:
                return []
            if probe not in found:
                found[probe] = [k for k, t in enumerate(texts) if probe in t]
            return found[probe]
        for split in splits:
            for i in split_indices(split, task):
                q = where(qas[i]["query"], QUESTION_MIN_CHARS)
                d = sorted({k for g in qas[i]["context"]
                            for k in where(docs[g][:GOLD_OPENING_CHARS], DOCUMENT_MIN_CHARS)})
                if q or d:
                    hits.append({"task": task, "split": split, "index": i,
                                 "question_in_texts": q, "gold_document_in_texts": d})
    return hits


def g12(calibration_texts: Sequence[str], *, tasks: Sequence[str] = ("qa_1",),
        splits: Sequence[str] = G12_SPLITS) -> dict:
    """Gate G12. Raises SplitLeak if a held-out question, or the opening of
    one of its gold documents, occurs in any calibration text. Returns what
    it checked, for the table that records it.

    It needs the QA data. Where that is missing it raises, and the caller
    stops: a gate that cannot run has not passed."""
    if calibration_texts is None or len(calibration_texts) == 0:
        raise SplitLeak("G12 was given no calibration texts; an empty check is not a pass")
    qa_tasks = [t for t in tasks if t.startswith("qa_")]
    if not qa_tasks:
        raise SplitLeak("G12 was given no QA task to probe")
    hits = g12_hits(calibration_texts, tasks=qa_tasks, splits=splits)
    if hits:
        by = {}
        for h in hits:
            c = by.setdefault(h["split"], [0, 0])
            c[0] += bool(h["question_in_texts"])
            c[1] += bool(h["gold_document_in_texts"])
        summary = "; ".join(f"{s}: {q} questions, {d} gold documents" for s, (q, d) in by.items())
        first = hits[0]
        raise SplitLeak(f"G12: the calibration texts contain held-out material ({summary}). "
                        f"First: {first['task']} index {first['index']} ({first['split']}).")
    return {"gate": "G12", "passed": True, "n_texts": len(calibration_texts),
            "n_indices": sum(len(split_indices(s, t)) for t in qa_tasks for s in splits),
            "splits": list(splits), "tasks": qa_tasks,
            "gold_opening_chars": GOLD_OPENING_CHARS}


def generate_splits(tasks: Sequence[str], bands: Sequence[int], *,
                    count_tokens: sizing.TokenCounter,
                    calibration_texts: Sequence[str],
                    n_override: Optional[Mapping[str, int]] = None) -> dict:
    """{split: {(task, band): [examples]}} for calibration, selection and
    evaluation, with `g12` in the result.

    `calibration_texts` (the `text.json` texts the Qwen tables are profiled
    on) is required and has no default: the splits are not handed back until
    G12 has passed on them and every pair of splits is disjoint by all four
    identities. `n_override` shrinks a split for a dry run."""
    out = {}
    for split in ("calibration", "selection", "evaluation"):
        first, _ = SPLIT_INDEX[split]
        out[split] = {}
        for task in tasks:
            n = (n_override or {}).get(split, split_n(split, task))
            for band in bands:
                out[split][(task, band)] = ruler.generate_examples(
                    task, [band], n, seed=SPLIT_SEED[split], count_tokens=count_tokens,
                    index_offset=first)
    check_splits_disjoint({s: [r for (task, _), exs in by.items() for r in identities(task, exs)]
                           for s, by in out.items()})
    out["g12"] = g12(calibration_texts, tasks=tasks)
    return out
