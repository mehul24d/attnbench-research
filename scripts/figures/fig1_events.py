"""Figure 1's events.csv from recorded trace pairs (C5).

Each --panel names a panel, a trace and the sequence record banked beside it
by T2.27's step 11 (`fig1_<k>_trace.json`, `fig1_<k>_sequence.json`). The
trace is parsed again with the record's code path and must give the banked
sequence; one layer's operations become the rows
(`attnbench/figures/fig1_events.py`). Writes one CSV and never overwrites.

    python scripts/figures/fig1_events.py --layer 14 --out events.csv \\
        --panel L4 fig1_a_trace.json fig1_a_sequence.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attnbench.figures import fig1_events as F  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", nargs=3, action="append", required=True,
                    metavar=("NAME", "TRACE", "SEQUENCE"))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        panels = [F.events_from_files(trace, seq, args.layer, panel=name)
                  for name, trace, seq in args.panel]
        path = F.write_events(panels, args.out)
    except (F.Fig1Error, FileExistsError, OSError, ValueError) as e:
        print(f"fig1_events: {e}", file=sys.stderr)
        return 2
    print(f"wrote {path} ({sum(map(len, panels))} rows, {len(panels)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
