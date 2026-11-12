#!/usr/bin/env python3
"""The prompt-set hash for T4.1's "prompt-set hash and tokenizer version"
(adviser's ruling 7, fourth set of 12 Nov 2026). The sessions write no prompt
file: each prompt is built at run time from the pinned RULER data and the
primary model's tokenizer (`shakedown_arms.Harness.input_ids`, task `qa_1`,
seed 0). This rebuilds the same prompts on a CPU, the cell prompt (index 0)
and the mask-dependence check's second prompt (index 1) at every grid length,
in the harness's own order (index 0 from a one-example build, index 1 from a
two-example build), and hashes them.

    sha256 of the canonical JSON list, one entry per (length, index):
      {"task", "length", "index", "example_id", "n_tokens", "ids_sha256"}

`--check-rows DIR` compares each length's index-0 token count with the
`context_length` of the ground-truth rows under DIR (`gt_L<length>_b1.json`),
which is what a session recorded.

    python3 scripts/prompt_set_hash.py --lengths 2048 4096 8192 16384 32768
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def entry(task: str, length: int, index: int, example_id: str, ids) -> dict:
    data = ",".join(str(int(t)) for t in ids).encode()
    return {"task": task, "length": length, "index": index, "example_id": example_id,
            "n_tokens": len(ids), "ids_sha256": hashlib.sha256(data).hexdigest()}


def set_sha256(entries: list) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def build(lengths, *, tokenizer=None) -> list:
    from transformers import AutoTokenizer
    from attnbench.accuracy.config import load_grid
    from attnbench.accuracy.grid_configs import build_examples_by_task_length
    from attnbench.accuracy.prompting import prompt_ids, prompt_token_counter
    from attnbench.components import shakedown_arms as SA
    grid = load_grid(SA.GRID)
    tok = tokenizer or AutoTokenizer.from_pretrained(grid.model_primary)
    out = []
    for L in lengths:
        for index in (0, 1):
            ex = build_examples_by_task_length(
                grid, SA.SEED, count_tokens=prompt_token_counter(tok), tasks=(SA.TASK,),
                seq_lens={L: index + 1})[(SA.TASK, L)][index]
            ids = prompt_ids(tok, ex.context)           # what `encode_prompt` sends
            out.append(entry(SA.TASK, L, index, ex.example_id, ids))
    return out


def check_rows(entries: list, rows_dir: Path) -> list:
    """(length, rebuilt tokens, recorded tokens) for every length that differs."""
    bad = []
    for e in entries:
        if e["index"] != 0:
            continue
        f = rows_dir / f"gt_L{e['length']}_b1.json"
        if not f.exists():
            bad.append((e["length"], e["n_tokens"], None))
            continue
        got = {r["context_length"] for r in json.loads(f.read_text())["rows"]}
        if got != {e["n_tokens"]}:
            bad.append((e["length"], e["n_tokens"], sorted(got)))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--check-rows", type=Path)
    args = ap.parse_args(argv)
    entries = build(args.lengths)
    import transformers
    print(json.dumps({"prompt_set_sha256": set_sha256(entries),
                      "transformers_here": transformers.__version__, "entries": entries},
                     indent=1))
    if args.check_rows:
        bad = check_rows(entries, args.check_rows)
        print("token counts match the recorded rows" if not bad else f"MISMATCH: {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
