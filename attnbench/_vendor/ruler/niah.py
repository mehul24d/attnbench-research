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

SUPPORTED_HAYSTACK_MODES = ("noise", "needle")
SUPPORTED_NEEDLE_TYPES = ("numbers", "uuids")


def _generate_random(type_needle: str, rng: random.Random) -> str:
    if type_needle == "numbers":
        lower, upper = 10**6, 10**7 - 1
        return str(rng.randint(lower, upper))
    if type_needle == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    if type_needle == "words":
        raise NotImplementedError(
            "type_needle='words' needs the wonderwords package's word "
            "lists, not vendored here (no new dependency added) -- use "
            "'numbers' or 'uuids'"
        )
    raise ValueError(f"unknown type_needle: {type_needle!r}")


def generate_niah_example(*, num_haystack: int, seed: int,
                           num_needle_k: int = 1, num_needle_v: int = 1,
                           num_needle_q: int = 1,
                           type_needle_k: str = "uuids",
                           type_needle_v: str = "numbers",
                           haystack_mode: str = "noise",
                           template: str = DEFAULT_TEMPLATE,
                           ) -> tuple[str, list[str]]:
    """One NIAH example: (input_text, answers). Ported from RULER's
    niah.py generate_input_output() -- see module docstring for exactly
    what was restructured vs. what's out of scope.
    """
    if haystack_mode not in SUPPORTED_HAYSTACK_MODES:
        if haystack_mode == "essay":
            raise NotImplementedError(
                "haystack_mode='essay' needs NLTK plus a downloaded essay "
                "corpus, not wired up -- use 'noise' or 'needle'. See "
                "VENDORED.md."
            )
        raise ValueError(f"unknown haystack_mode: {haystack_mode!r}")

    rng = random.Random(seed)
    num_needle_k = max(num_needle_k, num_needle_q)

    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(_generate_random(type_needle_k, rng))
        value = []
        for _ in range(num_needle_v):
            value.append(_generate_random(type_needle_v, rng))
            needles.append(NEEDLE_TEMPLATE.format(
                type_needle_v=type_needle_v, key=keys[-1], value=value[-1]))
        values.append(value)
    rng.shuffle(needles)

    if haystack_mode == "noise":
        sentences = [NOISE_SENTENCE] * num_haystack
    else:  # "needle": filler needles as haystack, same shape as the real ones
        sentences = [NEEDLE_TEMPLATE.format(
            type_needle_v=type_needle_v,
            key=_generate_random(type_needle_k, rng),
            value=_generate_random(type_needle_v, rng),
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
