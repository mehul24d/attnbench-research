# Adapted from NVIDIA/RULER, scripts/data/synthetic/niah.py, commit
# c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, fetched 2026-09-02.
# https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/niah.py
#
# NOT a verbatim vendor -- the original is a CLI script that calls
# argparse.parse_args() at module import time and holds its config in a
# module-level `args` Namespace, so it cannot be imported as a library
# without a real command line. This file extracts the same needle/haystack
# construction algorithm (generate_random, generate_input_output) into a
# plain function taking explicit parameters instead of reading `args` and
# a module-level `random.seed()` call -- same math, restructured, no
# argparse or global RNG state.
#
# Also narrower than the original on two axes, both gated behind
# dependencies this project deliberately isn't adding right now (see
# attnbench/_vendor/ruler/VENDORED.md):
#   - haystack_mode: only "noise" and "needle" are implemented. The
#     original's "essay" haystack needs NLTK (sent_tokenize) plus a
#     separately-downloaded Paul Graham essay corpus. "essay" raises
#     NotImplementedError here rather than being silently substituted --
#     the seam is left for later, not wired.
#   - type_needle: only "numbers" and "uuids" are implemented (both
#     stdlib-only). The original's "words" needle type needs the
#     `wonderwords` package's noun/adjective word lists. Also raises
#     NotImplementedError rather than a silent substitution.
#
# EXTENDED 2026-10-01 (audit T4): both seams are now wired, with the
# resources INJECTED rather than imported here, so this module stays
# dependency-free and testable. `haystack_mode="essay"` takes the essay corpus
# as a word list plus a sentence splitter (production passes NLTK's
# `sent_tokenize`; see accuracy/ruler_data.py), and `type_needle="words"`
# takes the (adjectives, nouns) lists RULER draws from wonderwords. Without
# them both still raise NotImplementedError -- never a silent substitution.
# The essay path is upstream's generate_input_output essay branch, line for
# line, with the module-level RNG replaced by the example's local one.
#
# This means results from this module are RULER's task *construction*
# algorithm, not RULER's published benchmark -- see the accuracy-track
# README section on comparability.
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
import uuid
from typing import Callable, Optional, Sequence

import numpy as np

NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."

DEFAULT_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} "
    "afterwards.\n{context}\nWhat are all the special magic {type_needle_v} "
    "for {query} mentioned in the provided text? The special magic "
    "{type_needle_v} for {query} mentioned in the provided text are"
)

NOISE_SENTENCE = ("The grass is green. The sky is blue. The sun is yellow. "
                   "Here we go. There and back again.")

SUPPORTED_HAYSTACK_MODES = ("noise", "needle", "essay")
SUPPORTED_NEEDLE_TYPES = ("numbers", "uuids", "words")

# Upstream: `DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))`.
# Computed the same way (numpy's half-to-even round), not retyped.
DEPTHS = [int(d) for d in np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int)]


def _generate_random(type_needle: str, rng: random.Random,
                     word_pool: Optional[tuple[Sequence[str], Sequence[str]]] = None) -> str:
    if type_needle == "numbers":
        lower, upper = 10**6, 10**7 - 1
        return str(rng.randint(lower, upper))
    if type_needle == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    if type_needle == "words":
        if word_pool is None:
            raise NotImplementedError(
                "type_needle='words' needs wonderwords' (adjectives, nouns) "
                "lists passed as word_pool -- see accuracy/ruler_data.py")
        # Upstream: `random.choice(sorted(set(f"{adj}-{noun}" ...)))`, a 6.2M
        # string list. Choosing the adjective and the noun independently is
        # the same uniform distribution provided no two pairs join to the same
        # string; checked for wonderwords 2.2.0 (2026-10-01): 0 collisions,
        # the only way one can arise being a hyphenated noun whose head
        # completes another adjective. `ruler_data.word_pool` re-checks it.
        adjs, nouns = word_pool
        return f"{rng.choice(adjs)}-{rng.choice(nouns)}"
    raise ValueError(f"unknown type_needle: {type_needle!r}")


def generate_niah_example(*, num_haystack: int, seed: int,
                           num_needle_k: int = 1, num_needle_v: int = 1,
                           num_needle_q: int = 1,
                           type_needle_k: str = "uuids",
                           type_needle_v: str = "numbers",
                           haystack_mode: str = "noise",
                           template: str = DEFAULT_TEMPLATE,
                           essay_words: Optional[Sequence[str]] = None,
                           sent_tokenize: Optional[Callable[[str], list]] = None,
                           word_pool: Optional[tuple] = None,
                           ) -> tuple[str, list[str]]:
    """One NIAH example: (input_text, answers). Ported from RULER's
    niah.py generate_input_output() -- see module docstring for exactly
    what was restructured vs. what's out of scope.
    """
    if haystack_mode not in SUPPORTED_HAYSTACK_MODES:
        raise ValueError(f"unknown haystack_mode: {haystack_mode!r}")
    if haystack_mode == "essay" and (essay_words is None or sent_tokenize is None):
        raise NotImplementedError(
            "haystack_mode='essay' needs the essay corpus (essay_words) and a "
            "sentence splitter (sent_tokenize) -- see accuracy/ruler_data.py")

    rng = random.Random(seed)
    num_needle_k = max(num_needle_k, num_needle_q)

    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(_generate_random(type_needle_k, rng, word_pool))
        value = []
        for _ in range(num_needle_v):
            value.append(_generate_random(type_needle_v, rng, word_pool))
            needles.append(NEEDLE_TEMPLATE.format(
                type_needle_v=type_needle_v, key=keys[-1], value=value[-1]))
        values.append(value)
    rng.shuffle(needles)

    if haystack_mode == "essay":
        # Upstream's essay branch: num_haystack WORDS of the corpus (repeated
        # if the corpus is shorter), split into sentences, each needle placed
        # at a sentence boundary at a depth sampled from DEPTHS.
        if num_haystack <= len(essay_words):
            text = " ".join(essay_words[:num_haystack])
        else:
            repeats = (num_haystack + len(essay_words) - 1) // len(essay_words)
            text = " ".join((list(essay_words) * repeats)[:num_haystack])
        document_sents = sent_tokenize(text.strip())
        insertion_positions = ([0]
                               + sorted(int(len(document_sents) * (depth / 100))
                                        for depth in rng.sample(DEPTHS, len(needles)))
                               + [len(document_sents)])
        parts = []
        for i in range(1, len(insertion_positions)):
            parts.append(" ".join(document_sents[insertion_positions[i - 1]:insertion_positions[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)
    else:
        if haystack_mode == "noise":
            sentences = [NOISE_SENTENCE] * num_haystack
        else:  # "needle": filler needles as haystack, same shape as the real ones
            sentences = [NEEDLE_TEMPLATE.format(
                type_needle_v=type_needle_v,
                key=_generate_random(type_needle_k, rng, word_pool),
                value=_generate_random(type_needle_v, rng, word_pool),
            ) for _ in range(num_haystack)]

        indexes = sorted(rng.sample(range(num_haystack), min(len(needles), num_haystack)),
                          reverse=True)
        for index, needle in zip(indexes, needles):
            sentences.insert(index, needle)
        context = "\n".join(sentences)

    q_indices = rng.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in q_indices]
    answers = [a for i in q_indices for a in values[i]]
    query = (", ".join(queries[:-1]) + ", and " + queries[-1]
             if len(queries) > 1 else queries[0])

    type_needle_v_text = type_needle_v
    if num_needle_q * num_needle_v == 1:
        template = (template.replace("Some", "A").replace("are all", "is")
                    .replace("are", "is").replace("answers", "answer"))
        type_needle_v_text = type_needle_v_text[:-1]  # drop trailing "s"

    input_text = template.format(type_needle_v=type_needle_v_text,
                                  context=context, query=query)
    return input_text, answers
