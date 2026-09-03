# Vendored verbatim from NVIDIA/RULER, scripts/eval/synthetic/constants.py,
# commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, fetched 2026-09-02.
# https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/eval/synthetic/constants.py
#
# Zero modifications: this file has no argparse, no module-level global
# state, and no external dependencies -- unlike the task generators (see
# niah.py / variable_tracking.py in this directory), it vendors exactly as
# the plan originally assumed every RULER file would. This is the part
# where a reimplementation would actually risk changing the number, so it
# is copied unmodified rather than reimplemented.
#
# Original license header follows, preserved from the source file:
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

"""
Add a new task:

TASK_NAME: {
    'metric_fn': the metric function with input (predictions: [str], references: [[str]]) to compute score.
}
"""


def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


TASKS = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'variable_tracking': {
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}
