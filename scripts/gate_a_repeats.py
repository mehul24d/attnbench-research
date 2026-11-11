"""Gate A over P1's repeat sessions (T3.3; `attnbench/analysis/gate_a_repeats.py`).

Each `--repeat` names one session's banked files, as `scripts/gate_a.py`
takes them: NAME REMOTE LOCAL GROUND_TRUTH CELLS RECORDINGS. The floors are
computed from every session's ground truth at `--floors-quantile` (a lock
value: before the lock pass `--floors-provisional`), and each session's
report is read at them. Writes `gate_a_repeats.json` and `gate_a_repeats.md`,
and each session's own `gate_a.json` and `gate_a.md` under `<out>/<NAME>/`.

    python3 scripts/gate_a_repeats.py --label "T3.2a and T3.2b" \\
        --repeat T3.2a gs://.../prediction_P1_<a>.json prediction_P1_<a>.json \\
            t32a/09_ground_truth t32a/cells.json t32a/recordings \\
        --repeat T3.2b ... --floors-quantile 0.95 --floors-provisional --out gate_a_repeats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from attnbench.analysis import gate_a as G  # noqa: E402
from attnbench.analysis import gate_a_repeats as GR  # noqa: E402
from attnbench.figures import results as RS  # noqa: E402
from attnbench.lock.floors import FloorTable, FloorsError  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict.commit import REQUIRED_SYNC_GAP_S  # noqa: E402
from attnbench.predict.recursion import CostModelError  # noqa: E402
import _load  # noqa: E402
import gate_a as GA  # noqa: E402

FIELDS = ("name", "remote", "local", "ground_truth", "cells", "recordings")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeat", nargs=len(FIELDS), action="append", required=True,
                    metavar=tuple(f.upper() for f in FIELDS))
    ap.add_argument("--dense-arm", default="dense")
    ap.add_argument("--floors-quantile", type=float, required=True,
                    help="the floors' quantile (a lock value)")
    ap.add_argument("--floors-provisional", action="store_true")
    ap.add_argument("--family-bound", type=float, default=None)
    ap.add_argument("--family-bound-provisional", action="store_true")
    ap.add_argument("--clock-skew-s", type=float, default=REQUIRED_SYNC_GAP_S)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None, store=None) -> int:
    args = parser().parse_args(argv)
    if args.family_bound is None and args.family_bound_provisional:
        print("gate_a_repeats: --family-bound-provisional needs --family-bound", file=sys.stderr)
        return 2
    repeats = [dict(zip(FIELDS, r)) for r in args.repeat]
    names = [r["name"] for r in repeats]
    if len(set(names)) != len(names):
        print(f"gate_a_repeats: a session is named twice: {names}", file=sys.stderr)
        return 2
    out = Path(args.out)
    try:
        gts = {r["name"]: GA.ground_truth(Path(r["ground_truth"])) for r in repeats}
        fl = GR.floors({s: res for s, (res, _) in gts.items()}, quantile=args.floors_quantile,
                       provisional=args.floors_provisional, pairing="P1")
        reports = {}
        for r in repeats:
            results, starts = gts[r["name"]]
            cp = _load.load(r["remote"], r["local"], started_at=starts,
                            clock_skew_s=args.clock_skew_s, banked_at=None, store=store)
            reports[r["name"]] = G.report(
                label=f"{args.label}: {r['name']}", cp=cp, results=results,
                dense_arm=args.dense_arm, relative_floor=fl["noise_floor_relative"],
                floors=FloorTable.from_ms(
                    {k: v["floor_ms"] for k, v in GR.cell_floors(
                        {s: res for s, (res, _) in gts.items()},
                        quantile=args.floors_quantile, pairing="P1").items()},
                    pairing="P1"),
                family_reports=GA.family_reports(Path(r["cells"]), Path(r["recordings"])),
                family_bound=args.family_bound,
                family_bound_provisional=args.family_bound_provisional)
            G.write(reports[r["name"]], out / r["name"])
        rep = GR.report(label=args.label, reports=reports, floors_=fl)
    except (G.GateAError, FloorsError, RS.ResultsError, C.PredictionCommitError,
            CostModelError, OSError, KeyError, TypeError, ValueError) as e:
        print(f"gate_a_repeats: {e}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_a_repeats.json").write_text(json.dumps(rep, indent=1, default=list))
    (out / "gate_a_repeats.md").write_text(GR.to_markdown(rep))
    print(f"wrote {out / 'gate_a_repeats.json'} and {out / 'gate_a_repeats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
