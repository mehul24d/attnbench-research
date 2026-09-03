# Adapted from NVIDIA/RULER, scripts/data/synthetic/variable_tracking.py,
# commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, fetched 2026-09-02.
# https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/variable_tracking.py
#
# Same situation as niah.py in this directory: the original is an argparse
# CLI script, not an importable library. This extracts generate_chains() and
# the "noise"-haystack branch of generate_input_output() into plain
# functions with explicit parameters, using one local random.Random(seed)
# instead of module-level random.seed()/np.random calls. The "essay"
# haystack branch (needs NLTK) and the few-shot/ICL example machinery
# (add_fewshot, an orthogonal CLI convenience feature, not part of the core
# task construction) are both dropped -- see VENDORED.md.
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
import string

DEFAULT_TEMPLATE = (
    "[INST] Memorize and track the chain(s) of variable assignment hidden "
    "in the following text.\n\n{context}\nQuestion: Find all variables "
    "that are assigned the value {query} in the text above. [/INST] "
    "Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assgined the value {query}, they are: "
)

NOISE_SENTENCE = ("The grass is green. The sky is blue. The sun is yellow. "
                   "Here we go. There and back again.")


def _generate_chains(num_chains: int, num_hops: int, rng: random.Random,
                      ) -> tuple[list[list[str]], list[list[str]]]:
    k = 5
    vars_all = [''.join(rng.choices(string.ascii_uppercase, k=k))
                for _ in range((num_hops + 1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append(''.join(rng.choices(string.ascii_uppercase, k=k)))

    vars_ret, chains_ret = [], []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i:i + num_hops + 1]
        vars_ret.append(this_vars)
        this_chain = [f"VAR {this_vars[0]} = {rng.randint(10000, 99999)}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j+1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret


def generate_vt_example(*, num_noise: int, seed: int, num_chains: int = 1,
                         num_hops: int = 4, template: str = DEFAULT_TEMPLATE,
                         ) -> tuple[str, list[str]]:
    """One variable-tracking example: (input_text, answer_variable_names).
    Ported from RULER's variable_tracking.py generate_input_output(),
    noise-haystack branch only -- see module docstring.
    """
    rng = random.Random(seed)
    variables, chains = _generate_chains(num_chains, num_hops, rng)
    value = chains[0][0].split("=")[-1].strip()

    sentences = [NOISE_SENTENCE] * num_noise
    for chain in chains:
        positions = sorted(rng.sample(range(len(sentences)), len(chain)))
        for insert_pi, j in zip(positions, range(len(chain))):
            sentences.insert(insert_pi + j, chain[j])
    context = "\n".join(sentences).replace(". \n", ".\n")

    input_text = template.format(context=context, query=value,
                                  num_v=num_hops + 1)
    return input_text, variables[0]
