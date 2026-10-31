"""Results tables for T3.1 and T3.3 from a committed prediction and its
ground truth (C5).

`--cells` is a JSON file: {"results": [{"file": RESULT_JSON, "arms": {ARM:
[MODEL, CODE_PATH, ARM, DENSITY, LENGTH, BATCH], ...}}, ...]}, result paths
relative to it. Each ground-truth result's arm rows are matched to the
committed cells it states (`attnbench/figures/results.py`). The noise floor
is a lock value and has no default. Writes CSV and Markdown tables and the
unmatched cells into --out-dir, never overwriting
(`attnbench/figures/t3_tables.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attnbench.figures import results as RS  # noqa: E402
from attnbench.figures import t3_tables as T  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
import _load  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--dense-arm", action="append", required=True)
    ap.add_argument("--relative-floor", type=float, required=True,
                    help="the noise floor, a lock value; say where a provisional one is used")
    ap.add_argument("--clock-skew-s", type=float, required=True)
    ap.add_argument("--banked-at", action="append", metavar="PAIRING=ISO")
    ap.add_argument("--out-dir", required=True)
    return ap


def _write(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def main(argv=None, store=None) -> int:
    args = parser().parse_args(argv)
    try:
        cells_path = Path(args.cells)
        spec = json.loads(cells_path.read_text(encoding="utf-8"))
        triples = []
        for r in spec["results"]:
            p = cells_path.parent / r["file"]
            triples.append((p.name, json.loads(p.read_text(encoding="utf-8")), r["arms"]))
        cp = _load.load(args.remote, args.local,
                        started_at=[t[1]["first_timed_block_started_at"] for t in triples],
                        clock_skew_s=args.clock_skew_s, banked_at=args.banked_at,
                        store=store)
        measured = RS.measured_cells(triples, statistic=cp.statistic)
        kw = dict(dense_arms=args.dense_arm, relative_floor=args.relative_floor)
        tables = {
            "t31_cells": T.cell_table(cp, measured, **kw),
            "t33_by_predictor": T.predictor_summary(cp, measured, by_arm_class=False, **kw),
            "t33_by_predictor_and_arm_class": T.predictor_summary(cp, measured,
                                                                  by_arm_class=True, **kw),
        }
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        caption = (f"{cp.pairing}, {cp.statistic}, noise floor {args.relative_floor} "
                   f"(prediction sha256 {cp.sha256[:12]})")
        for name, (cols, rows) in tables.items():
            _write(out / f"{name}.csv", T.to_csv(cols, rows))
            _write(out / f"{name}.md", T.to_markdown(cols, rows, caption=caption))
        _write(out / "unmatched.json",
               json.dumps(T.unmatched(cp, measured, **kw), indent=1, sort_keys=True) + "\n")
    except (T.TableError, RS.ResultsError, C.PredictionCommitError, FileExistsError,
            OSError, KeyError, ValueError) as e:
        print(f"t3_tables: {e}", file=sys.stderr)
        return 2
    print(f"wrote {len(tables)} tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
