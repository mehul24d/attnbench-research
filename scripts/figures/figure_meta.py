"""A figure's meta.json: the committed prediction, ground truth and other
inputs it was drawn from, each with its sha256 (C5).

The prediction is read with `load_committed` from the store (it must have
been synced before the result's first timed block, K9) and the local copy
must hold the same bytes. Writes once and never overwrites
(`attnbench/figures/meta.py`).

    python scripts/figures/figure_meta.py --figure fig1 \\
        --remote gs://BUCKET/predictions/P1/prediction_P1_....json \\
        --local prediction_P1_....json --result ground_truth.json \\
        --cell MODEL CODE_PATH oracle 1.0 16384 1 --clock-skew-s 60 \\
        --input trace=fig1_a_trace.json --input events=events.csv --out meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attnbench.figures import meta as M  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
import _load  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--figure", required=True)
    ap.add_argument("--remote", required=True, help="the committed file in the store")
    ap.add_argument("--local", required=True, help="the same file on disk, sidecar beside it")
    ap.add_argument("--result", required=True, help="ground_truth.time_arms result JSON")
    ap.add_argument("--cell", nargs=6, required=True,
                    metavar=("MODEL", "CODE_PATH", "ARM", "DENSITY", "LENGTH", "BATCH"))
    ap.add_argument("--clock-skew-s", type=float, required=True)
    ap.add_argument("--banked-at", action="append", metavar="PAIRING=ISO")
    ap.add_argument("--input", action="append", default=[], metavar="ROLE=PATH")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None, store=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = json.loads(Path(args.result).read_text(encoding="utf-8"))
        cp = _load.load(args.remote, args.local,
                        started_at=[result["first_timed_block_started_at"]],
                        clock_skew_s=args.clock_skew_s, banked_at=args.banked_at,
                        store=store)
        m, a, arm, d, length, b = args.cell
        inputs = []
        for x in args.input:
            role, _, path = x.partition("=")
            inputs.append((role, path))
        meta = M.figure_meta(figure=args.figure, prediction=cp, prediction_path=args.local,
                             cell=(m, a, arm, float(d), int(length), int(b)),
                             result=result, result_path=args.result, inputs=inputs,
                             code=C.current_code_stamp())
        path = M.write_meta(meta, args.out)
    except (M.MetaError, C.PredictionCommitError, FileExistsError, OSError, KeyError,
            ValueError) as e:
        print(f"figure_meta: {e}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
