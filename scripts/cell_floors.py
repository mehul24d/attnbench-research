"""The resolution floor per (context length, batch) by the amended rule
(Appendix A, 8 Nov 2026, fifth paste; `gate_a_repeats.cell_floors`), from
the ground truth of the sessions named. Since 9 Nov 2026 (Appendix A, sixth
paste) the rule's sessions are those whose timed ground-truth path is
byte-identical (`scripts/timed_path.py`); any other set is a sensitivity. Writes `<out>/cell_floors.json`, the
`scripts/gate_a.py --v2 --cell-floors` input ("<length>x<batch>" -> ms), and
`<out>/cell_floors_detail.json` with each cell's within- and between-session
spread and which one set it.

    python3 scripts/cell_floors.py --session T3.2a t32a/09_ground_truth \\
        --session T3.2b ... --session T3.2c ... --quantile 0.95 --provisional --out floors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from attnbench.analysis import gate_a as G  # noqa: E402
from attnbench.analysis import gate_a_repeats as GR  # noqa: E402
from attnbench.lock.floors import FloorsError  # noqa: E402
import gate_a as GA  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", nargs=2, action="append", required=True,
                    metavar=("NAME", "GROUND_TRUTH"))
    ap.add_argument("--quantile", type=float, required=True, help="the floors' quantile (a lock value)")
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    names = [n for n, _ in args.session]
    if len(set(names)) != len(names):
        print(f"cell_floors: a session is named twice: {names}", file=sys.stderr)
        return 2
    try:
        sessions = {n: GA.ground_truth(Path(d))[0] for n, d in args.session}
        fl = GR.cell_floors(sessions, quantile=args.quantile, pairing="P1")
    except (G.GateAError, FloorsError, OSError, KeyError, TypeError, ValueError) as e:
        print(f"cell_floors: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cell_floors.json").write_text(json.dumps({k: v["floor_ms"] for k, v in fl.items()},
                                                     indent=1))
    (out / "cell_floors_detail.json").write_text(json.dumps(
        {"sessions": names, "quantile": args.quantile, "provisional": args.provisional,
         "rule": "max(within-session, between-session) per (length, batch), over sessions with a "
                 "byte-identical timed ground-truth path; Appendix A, 8 and 9 Nov 2026",
         "cells": fl}, indent=1))
    for k, v in fl.items():
        b = "-" if v["between_ms"] is None else f"{v['between_ms']:.2f}"
        print(f"{k}: floor {v['floor_ms']:.2f} ms (within {v['within_ms']:.2f}, between {b}, "
              f"{v['source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
