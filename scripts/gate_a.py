"""Gate A's inputs from one P1 session's banked files (T3.3;
`attnbench/analysis/gate_a.py`).

The committed prediction is read with `load_committed` against the store
(`--remote`, its local copy `--local`), the start being the earliest stamp
of the session's ground-truth groups (`--ground-truth`, a step-9 directory
with a `stamps.json` and `gt_*.json` per group). Step 6's priced cells
(`--cells`, `cells.json`) and, per cell, step 5's recording
(`--recordings`, a directory with one `<cell>/` per step-6 cell, named as
`t31_steps.cell_dir` names it, holding `trace.json` and `sequence.json`)
give leg 2. `--family-bound` is a lock value; before the lock pass
`--family-bound-provisional` with it, or leave it out and read the
distribution. Writes `gate_a.json` and `gate_a.md` into `--out`.

Gate A v2 (Appendix A, 8 Nov 2026): `--v2` with `--cell-floors` (a JSON
file, "<length>x<batch>" -> that cell's resolution floor in ms) exempts, in
each cell, a family below the cell's floor from leg 2, and reads leg 3 from
`--delay-injection` (step 12's `delay_injection.json`, slopes recomputed
from its raw repeats) against [1/sqrt 2, sqrt 2], every priced mechanism
listed.

    python3 scripts/gate_a.py --label "T3.2a alone, dry run" \\
        --remote gs://<bucket>/predictions/P1-T3.2a/<session>/prediction_P1_<t>.json \\
        --local prediction_P1_<t>.json --ground-truth 09_ground_truth \\
        --cells 06_device_components/cells.json --recordings 05_host_components \\
        --relative-floor 0 --resolution-floors cell_floors.json --out gate_a_t32a

`--resolution-floors` is the per-(length, batch) table the sign agreement
reads, '<length>x<batch>' -> ms (the `--cell-floors` format); one floor for
every length is no longer taken (T4.3 read A, finding 4; ruling 1a, 11 Nov
2026).
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
from attnbench.figures import results as RS  # noqa: E402
from attnbench.lock.floors import FloorTable, FloorsError  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict.commit import CLOCK_SKEW_ALLOWED_S  # noqa: E402
from attnbench.predict.recursion import CostModelError  # noqa: E402
import _load  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True, help="what this run is (shown on every output)")
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--dense-arm", default="dense")
    ap.add_argument("--relative-floor", type=float, required=True,
                    help="the noise floor, a relative error (a lock value)")
    ap.add_argument("--resolution-floors", required=True,
                    help="JSON, '<length>x<batch>' -> the resolution floor (ms) the sign "
                         "agreement reads per cell (a lock value)")
    ap.add_argument("--family-bound", type=float, default=None)
    ap.add_argument("--family-bound-provisional", action="store_true")
    ap.add_argument("--v2", action="store_true", help="Gate A v2 (Appendix A, 8 Nov 2026)")
    ap.add_argument("--cell-floors", default=None,
                    help="v2: JSON, '<length>x<batch>' -> resolution floor (ms)")
    ap.add_argument("--delay-injection", default=None,
                    help="step 12's delay_injection.json (leg 3)")
    ap.add_argument("--clock-skew-s", type=float, default=CLOCK_SKEW_ALLOWED_S)
    ap.add_argument("--out", required=True)
    return ap


def ground_truth(d: Path) -> tuple[list, list]:
    """([(name, result)], [start stamps]) from a step-9 directory."""
    results, starts = [], []
    for stamps in sorted(d.glob("*/stamps.json")) or sorted(d.glob("stamps.json")):
        s = json.loads(stamps.read_text())
        starts += [s["clock_not_before"], *s["first_clocked_block_started_at"]]
        for p in sorted(stamps.parent.glob("gt_*.json")):
            results.append((f"{stamps.parent.name}/{p.name}", json.loads(p.read_text())))
    if not results:
        raise G.GateAError(f"no ground truth under {d}")
    return results, starts


def family_reports(cells_json: Path, recordings: Path) -> dict:
    import t31_steps as st
    from attnbench.components import op_sequence as ops
    out = {}
    for e in json.loads(cells_json.read_text())["cells"]:
        name = st.cell_dir(Path(), *st.device_cell_key(e)).name
        rec = recordings / name
        seq = G.recorded_sequence(rec / "trace.json", rec / "sequence.json")
        trace_s = ops.device_durations(rec / "trace.json", seq)
        out[name] = G.replay_report(seq, trace_s, e["ops"])
    return out


def cell_floors_s(path: Path, names) -> dict:
    """Report name (a `t31_steps.cell_dir` name) -> its cell's floor, seconds."""
    import re
    raw = json.loads(Path(path).read_text())
    out = {}
    for n in names:
        m = re.search(r"_L(\d+)_b(\d+)$", n)
        k = f"{m.group(1)}x{m.group(2)}" if m else None
        if k not in raw:
            raise G.GateAError(f"{path} has no floor for {n} ({k})")
        out[n] = float(raw[k]) / 1e3
    return out


def leg3(path) -> dict | None:
    if path is None:
        return None
    from attnbench.components import delay_session as dls
    return dls.slopes_from_record(json.loads(Path(path).read_text()))


def main(argv=None, store=None) -> int:
    args = parser().parse_args(argv)
    if args.v2 and not args.cell_floors:
        print("gate_a: --v2 needs --cell-floors", file=sys.stderr)
        return 2
    if args.family_bound is None and args.family_bound_provisional:
        print("gate_a: --family-bound-provisional needs --family-bound", file=sys.stderr)
        return 2
    try:
        results, starts = ground_truth(Path(args.ground_truth))
        cp = _load.load(args.remote, args.local, started_at=starts,
                        clock_skew_s=args.clock_skew_s, banked_at=None, store=store)
        fams = family_reports(Path(args.cells), Path(args.recordings))
        from attnbench.predict.recursion import PRICED_MECHANISMS
        rep = G.report(label=args.label, cp=cp, results=results, dense_arm=args.dense_arm,
                       relative_floor=args.relative_floor,
                       floors=FloorTable.from_ms(
                           json.loads(Path(args.resolution_floors).read_text()), pairing="P1"),
                       family_reports=fams, family_bound=args.family_bound,
                       family_bound_provisional=args.family_bound_provisional,
                       measured_slopes=leg3(args.delay_injection),
                       mechanisms=PRICED_MECHANISMS if args.v2 else (),
                       slope_range=G.V2_SLOPE_RANGE if args.v2 else None,
                       cell_floors_s=(cell_floors_s(Path(args.cell_floors), fams)
                                      if args.v2 else None))
        j, md = G.write(rep, Path(args.out))
    except (G.GateAError, RS.ResultsError, C.PredictionCommitError, CostModelError,
            FloorsError, OSError, KeyError, TypeError, ValueError) as e:
        print(f"gate_a: {e}", file=sys.stderr)
        return 2
    print(f"wrote {j} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
