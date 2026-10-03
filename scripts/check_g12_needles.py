#!/usr/bin/env python
"""Gate G12, the needle half (estimator-frontier pre-registration sec. 5).

    check_g12_needles.py --text-json PATH/TO/x-attention/.../text.json

Compares every needle in the calibration texts ("One of the special magic
<type> for <key> is: <value>.") with every needle this harness generates
for four of its needle tasks, in every split, at 16384 and 32768. A needle's key and
value come from the example's seed alone (task, band, index), so each example
is rendered with a short haystack.

Exits 1 on any shared (key, value) pair. A shared value under different keys
is counted and set beside the number chance alone would give: the values are
random 7-digit numbers, and the calibration texts hold about 14,000 of them.
Prints counts only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attnbench.accuracy import frontier_splits, ruler, sizing  # noqa: E402

NEEDLE = re.compile(r"One of the special magic (\w+) for (.+?) is: (.+?)\.(?=\s|$)")
# The needle tasks whose needles depend on the example's seed alone.
# `niah_multikey` is left out: its haystack is itself made of needles, so
# which ones it holds changes with the haystack's length, and a short render
# is not the real prompt. It is not a task of the frontier study or of T4.
TASKS = ("niah_single", "niah_multikey_1", "niah_multivalue", "niah_multiquery")
BANDS = (16384, 32768)


def needles(text: str) -> set:
    return {(k.strip(), v.strip()) for _, k, v in NEEDLE.findall(text)}


def example_needles(task: str, split: str, band: int, index: int) -> set:
    """The needles of one harness example. Keys and values are drawn from the
    example's seed before the haystack is touched, so the shortest haystack
    the task can render carries the same ones as the full-length prompt
    (`tests/test_frontier_splits.py` checks that against `generate_examples`)."""
    seed = ruler._example_seed(frontier_splits.SPLIT_SEED[split], task, band, index)
    text, answer = ruler._render(task, seed, ruler._min_haystack_units(task) + 8, index=index)
    found = needles(text)
    if not {v for _, v in found} >= set(answer):
        raise SystemExit(f"{task} {band} {index}: the needle pattern misses an answer")
    return found


def harness_needles() -> dict:
    out = {}
    for split in ("t4_replication", "selection", "calibration", "evaluation"):
        got = set()
        for task in TASKS:
            for band in BANDS:
                for i in frontier_splits.split_indices(split, task):
                    got |= example_needles(task, split, band, i)
        out[split] = got
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text-json", type=Path, required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("cal", REPO / "scripts" / "calibrate_xattn_thresholds.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    texts = [cal.strip_chat_markers(t) for t in json.loads(a.text_json.read_bytes())]
    theirs = set()
    n_texts = 0
    for t in texts:
        found = needles(t)
        n_texts += bool(found)
        theirs |= found
    their_values = {v for _, v in theirs}
    print(f"calibration texts with a needle: {n_texts}; distinct needles {len(theirs)}, "
          f"distinct values {len(their_values)}")
    their_numbers = {v for v in their_values if v.isdigit()}
    bad = 0
    for split, mine in harness_needles().items():
        pairs = theirs & mine
        my_values = {v for _, v in mine}
        values = their_values & my_values
        chance = len({v for v in my_values if v.isdigit()}) * len(their_numbers) / 9e6
        bad += len(pairs)
        print(f"{split:15s} harness needles {len(mine):6d}: shared (key, value) pairs "
              f"{len(pairs)}; shared values {len(values)} (chance alone: {chance:.1f})")
    print("G12 needles:", "STOP" if bad else "pass (no shared pair)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
