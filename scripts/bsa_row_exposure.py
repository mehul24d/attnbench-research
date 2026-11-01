#!/usr/bin/env python3
"""How often the banked masks reach Block-Sparse-Attention's boundary read
(researcher's ruling, 2 Nov 2026, after the XAttention crash diagnostic).

The forward kernel reads a row of its int32 mask one entry past the row's end
when every key block in the row is selected (`attnbench/bsa_contract.py`):
the next row's first entry, or, for the last row of the last head, one entry
past the buffer.

Counts, per banked mask bank (one recorded prefill: an arm, a prompt, a
length), the (layer, head) pairs whose last query row is fully selected, the
layers where the last head's is (the read past the buffer), and the fully
selected rows anywhere (the reads inside it). Each bank is read with
`mask_bank.load`, which checks its hashes; banks with the same combined
sha256 count once. Reads `results/` and writes nothing there.

    python3 scripts/bsa_row_exposure.py [results] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attnbench.components import mask_bank  # noqa: E402


def exposure(masks) -> dict:
    """The counts for one prefill's masks, (B, H, R, C) bool, one per layer."""
    full_last, last_head, any_rows, pairs = [], [], 0, 0
    for layer, m in enumerate(masks):
        m = np.asarray(m, dtype=bool)
        full = m.all(axis=-1)                       # (B, H, R)
        pairs += full.shape[0] * full.shape[1]
        any_rows += int(full.sum())
        full_last += [(layer, h) for h in range(full.shape[1]) if full[:, h, -1].any()]
        if full[-1, -1, -1]:
            last_head.append(layer)
    return {"layers": len(masks), "pairs": pairs, "full_last_row_pairs": len(full_last),
            "full_last_row": full_last, "out_of_buffer_layers": last_head,
            "full_rows_any": any_rows}


def scan(results: Path) -> list[dict]:
    seen, rows = set(), []
    for npz in sorted(Path(results).rglob("masks.npz")):
        rec = mask_bank.load(npz.with_suffix(""))
        if rec.sha256 in seen:
            continue
        seen.add(rec.sha256)
        rows.append({"bank": str(npz.parent.relative_to(results)), "arm": rec.code_path.arm,
                     "implementation": rec.code_path.implementation,
                     "prompt_id": rec.prompt_id, "length": rec.context_length,
                     **exposure([m.array() for m in rec.masks])})
    return rows


def summary(rows: list[dict]) -> dict:
    by_arm: dict = {}
    for r in rows:
        a = by_arm.setdefault(r["arm"], {"banks": 0, "banks_exposed": 0,
                                         "banks_out_of_buffer": 0, "pairs": 0,
                                         "full_last_row_pairs": 0})
        a["banks"] += 1
        a["banks_exposed"] += r["full_last_row_pairs"] > 0
        a["banks_out_of_buffer"] += bool(r["out_of_buffer_layers"])
        a["pairs"] += r["pairs"]
        a["full_last_row_pairs"] += r["full_last_row_pairs"]
    return by_arm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="?", default=str(ROOT / "results"))
    ap.add_argument("--json", default=None, help="write every bank's counts here")
    args = ap.parse_args(argv)
    rows = scan(Path(args.results))
    print(f"{'arm':12s} {'impl':12s} {'length':>6s} {'prompt':16s} pairs full_last "
          f"rows_full oob_layers")
    for r in rows:
        print(f"{str(r['arm']):12s} {str(r['implementation']):12s} {r['length']:6d} "
              f"{str(r['prompt_id']):16s} {r['pairs']:5d} {r['full_last_row_pairs']:9d} "
              f"{r['full_rows_any']:9d} {r['out_of_buffer_layers']}")
    print(json.dumps(summary(rows), indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"banks": rows, "by_arm": summary(rows)},
                                              indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
