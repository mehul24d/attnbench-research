#!/usr/bin/env python
"""Measure answer lengths per task with the real tokenizer, for token caps.

    python scripts/measure_answer_lengths.py [--tasks ...]

`stopping.TASK_TOKEN_CAPS` is 2x the longest expected answer over 200
examples (docs/stage3_generation_decision.md, section 1). The script that
produced the original three caps was not kept. This one is checked against
them before it is trusted: on the existing tasks (40 examples at each of the
five bands, as that section describes) it must reproduce the published
maxima -- niah_single 7, niah_multikey 36, vt 20 -- or it exits non-zero.

Answer text: the expected answers joined by a single space, no leading space
(all of them for NIAH/vt, which score `string_match_all`; for QA, scored
`string_match_part` against alternative gold answers, the LONGEST
alternative, since any one may be the one written). Chosen because it is the
one convention, of seven tried on 2026-10-01, that reproduces all three
published maxima: a leading space adds a token to every task (8 / 37 / 24
with ", "), and ", " adds the separators vt's published 20 does not count
(24). The first version of this script used " " + ", ".join(...) and failed
its own reproduction check.

New tasks: 200 examples at one band (4096; see NEW_TASK_BAND). Their answers do not depend on the band
by construction -- NIAH answers are needle payloads, and QA asks the same
questions at every band -- so 40 per band would repeat the same 40 QA
questions five times rather than measure 200.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench.accuracy import ruler  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BANDS = (2048, 4096, 8192, 16384, 32768)
PUBLISHED_MAX = {"niah_single": 7, "niah_multikey": 36, "vt": 20}
NEW_TASKS = ("niah_multikey_1", "niah_multivalue", "niah_multiquery", "qa_1", "qa_2")
# 4096, not 2048: a HotpotQA question carries all its paragraphs (gold and
# distractor), ~3,000 tokens, so qa_2 cannot be built at 2048 -- RULER's own
# smallest length is 4K. Answers do not depend on the band either way.
NEW_TASK_BAND = 4096


def answer_text(task: str, answers: list[str]) -> str:
    if ruler._TASK_CATEGORY[task] == "qa":
        return max(answers, key=len)
    return " ".join(answers)


def measure(task: str, tok, count_tokens) -> list[int]:
    if task in PUBLISHED_MAX:
        exs = ruler.generate_examples(task, list(BANDS), n_per_length=40, seed=0,
                                      count_tokens=count_tokens)
    else:
        exs = ruler.generate_examples(task, [NEW_TASK_BAND], n_per_length=200, seed=0,
                                      count_tokens=count_tokens)
    return [len(tok(answer_text(task, e.answer), add_special_tokens=False).input_ids)
            for e in exs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(PUBLISHED_MAX) + list(NEW_TASKS))
    ap.add_argument("--out", type=Path, default=None)
    # Another model's caps (2026-10-03, estimator-frontier pre-registration
    # sec. 4.9): Llama-3.1-8B is measured with its own tokenizer, never given
    # Qwen's caps. The method is validated only by reproducing the Qwen
    # maxima, so run without --model first; with it, that check is skipped
    # and the output says so.
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    validating = args.model == MODEL
    print(f"tokenizer: {args.model}"
          + (f" @ {args.revision}" if args.revision else "")
          + ("" if validating else
             "  (published-maxima check skipped: it validates the method on Qwen only)"))

    def count_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False).input_ids)

    rows, ok = {}, True
    for task in args.tasks:
        lens = sorted(measure(task, tok, count_tokens))
        row = dict(n=len(lens), min=lens[0], median=statistics.median(lens),
                   p95=lens[int(0.95 * (len(lens) - 1))], max=lens[-1], cap=2 * lens[-1])
        rows[task] = row
        check = ""
        if validating and task in PUBLISHED_MAX:
            match = lens[-1] == PUBLISHED_MAX[task]
            ok &= match
            check = f"  published max {PUBLISHED_MAX[task]}: {'REPRODUCED' if match else 'MISMATCH'}"
        print(f"{task:16s} n={row['n']} min={row['min']} median={row['median']} "
              f"p95={row['p95']} max={row['max']} cap={row['cap']}{check}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
    if not ok:
        print("method does not reproduce the published maxima; caps from it are not trusted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
