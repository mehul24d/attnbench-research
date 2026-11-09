#!/usr/bin/env python3
"""T2.3 on banked sequences: per cell, the stream check, the arms table's
synchronisation column and the trace flags; per cell, the strict "same
sequence" verdict against an earlier session's sequence at the same cell;
and, for each XAttention cell, the asserts attributed from its
asserts-removed partner (`op_sequence.attribute_asserts`).

Each cell is a directory named `<arm>_<density>_L<length>_b<batch>` holding
the session's `sequence.json` (`op_sequence.to_record`), as T3.1's step 6
banks them. Nothing here reads a trace or a duration.

    python3 -I scripts/t23_sync_report.py <cells dir> [--against <earlier cells dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attnbench.components import op_sequence as O  # noqa: E402

SHIPPED, STRIPPED = "xattention_", "xattention_noassert_"


def load(path: Path) -> O.OpSequence:
    return O.from_record(json.loads(Path(path).read_text()))


def report(cells: Path, against: Path | None = None) -> dict:
    seqs = {p.parent.name: load(p) for p in sorted(Path(cells).glob("*/sequence.json"))}
    out = {}
    for name, s in seqs.items():
        sc = O.stream_check(s)
        row = {"arm": s.code_path.arm, "implementation": s.code_path.implementation,
               "python_optimize": s.code_path.python_optimize,
               "context_length": s.context_length, "batch": s.batch,
               "operations": len(s.ops), "layers": len(s.layers),
               "single_stream": sc.single_stream, "streams": sc.message,
               "synchronisations": len(O.sync_map(s)),
               "sync_column": O.sync_column(s), "flags": O.flag_counts(s)}
        earlier = Path(against) / name / "sequence.json" if against else None
        if earlier is not None and earlier.exists():
            c = O.same_sequence(load(earlier), s)
            row["against_earlier"] = {"verdict": c.verdict, "reason": c.reason}
        out[name] = row
    asserts = {}
    for name, s in seqs.items():
        if not name.startswith(SHIPPED) or name.startswith(STRIPPED):
            continue
        partner = STRIPPED + name[len(SHIPPED):]
        if partner not in seqs:
            continue
        try:
            a = O.attribute_asserts(s, seqs[partner])
        except O.SequenceError as e:
            asserts[name] = {"attributed": False, "refusal": str(e)}
            continue
        asserts[name] = {"attributed": True, "assert_synchronisations": sum(
            1 for op in a.ops if op.composite == "assert" and op.kind is O.SeqKind.SYNC),
            "sync_column": O.sync_column(a)}
    return {"cells": out, "asserts": asserts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cells", type=Path)
    ap.add_argument("--against", type=Path, default=None)
    args = ap.parse_args(argv)
    json.dump(report(args.cells, args.against), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
