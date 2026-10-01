# Adapted from NVIDIA/RULER, scripts/data/synthetic/qa.py, commit
# c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, fetched 2026-10-01.
# https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/qa.py
#
# NOT a verbatim vendor, for the same reason as niah.py: the original parses
# argparse and loads its dataset at import time and samples from the global
# RNG. `read_squad`, `read_hotpotqa` and `generate_input_output` are kept
# line for line in what they compute; what changed:
#   - the readers take the parsed JSON instead of a path (loading, checksum
#     pinning and the data directory live in accuracy/ruler_data.py -- the
#     datasets are CC BY-SA 4.0 and are not committed to this repository);
#   - distractor sampling and the document shuffle use the example's local
#     random.Random(seed). Upstream samples distractors from the global RNG
#     and shuffles with random.Random(args.random_seed), a seed FIXED across
#     examples; here both follow the example seed, the convention niah.py
#     already uses;
#   - the question is chosen by `index`, as upstream does (examples 0..n-1 in
#     the reader's order), so every band asks the same questions and band
#     comparisons are not confounded by question difficulty.
#
# Template: RULER's `qa` template with its answer prefix appended, the same
# completion-style convention niah.py and variable_tracking.py use.
#
# Original license header, preserved from the source file:
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import random

DEFAULT_TEMPLATE = (
    "Answer the question based on the given documents. Only give me the "
    "answer and do not output any other words.\n\nThe following are given "
    "documents.\n\n{context}\n\nAnswer the question based on the given "
    "documents. Only give me the answer and do not output any other words."
    "\n\nQuestion: {query} Answer:"
)
DOCUMENT_PROMPT = "Document {i}:\n{document}"


def read_squad(data: dict) -> tuple[list[dict], list[str]]:
    """Upstream `read_squad`, on the parsed dev-v2.0 JSON."""
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append({
                        "query": qas["question"],
                        "outputs": [a["text"] for a in qas["answers"]],
                        "context": [total_docs_dict[p["context"]]],
                        "more_context": [idx for idx in more_docs
                                         if idx != total_docs_dict[p["context"]]],
                    })
    return total_qas, total_docs


def read_hotpotqa(data: list) -> tuple[list[dict], list[str]]:
    """Upstream `read_hotpotqa`, on the parsed dev distractor JSON."""
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append({
            "query": d["question"],
            "outputs": [d["answer"]],
            "context": [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]],
        })
    return total_qas, total_docs


def min_docs(qas: list[dict], index: int) -> int:
    """The gold documents of question `index`: fewer than this and upstream's
    `random.sample(curr_more, num_docs - len(curr_docs))` gets a negative
    count. SQuAD has one gold paragraph; a HotpotQA distractor question
    carries all of its paragraphs (gold and distractor) as context."""
    return len(qas[index]["context"])


def generate_qa_example(qas: list[dict], docs: list[str], *, index: int,
                        num_docs: int, seed: int,
                        template: str = DEFAULT_TEMPLATE) -> tuple[str, list[str]]:
    """Upstream `generate_input_output(index, num_docs)`: the question's gold
    documents, then its own article's other paragraphs (SQuAD), then random
    unrelated documents, shuffled, numbered, and formatted."""
    if not 0 <= index < len(qas):
        raise IndexError(f"question index {index} outside the {len(qas)} questions")
    curr_q = qas[index]["query"]
    curr_a = qas[index]["outputs"]
    curr_docs = qas[index]["context"]
    curr_more = qas[index].get("more_context", [])
    if num_docs < len(curr_docs):
        raise ValueError(f"num_docs={num_docs} < {len(curr_docs)} gold documents")
    rng = random.Random(seed)

    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(docs) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + rng.sample(
                addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + rng.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [docs[idx] for idx in all_docs]
    else:
        repeats = (num_docs + len(docs) - 1) // len(docs)
        all_docs = (docs * repeats)[:num_docs]

    rng.shuffle(all_docs)
    context = "\n\n".join(DOCUMENT_PROMPT.format(i=i + 1, document=d)
                          for i, d in enumerate(all_docs))
    return template.format(context=context, query=curr_q), list(curr_a)
