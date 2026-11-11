"""P1's training file for the fitted baselines of every later pairing
(`attnbench/predict/p1_training.py`; researcher's decision, 8 Nov 2026).

Each `--repeat` names one T3.2 session: LABEL REMOTE LOCAL GROUND_TRUTH LINK,
repeat a (the features) first: T3.2d then T3.2c for the primary, T3.2a then
T3.2b for the sensitivity (`p1_training.POOLS`).
REMOTE is the committed prediction in the store (read through
`load_committed` with its local copy LOCAL), GROUND_TRUTH the session's step-9
directory, LINK its `02_link/recursion_inputs.json` (the link figures; repeat
a's are the rows'). With repeat b failed, give repeat a alone and `--b-failed`.

    python3 scripts/p1_training.py \\
        --repeat T3.2d gs://.../predictions/P1-T3.2d/<s>/prediction_P1_<t>.json <local> \\
            <09_ground_truth> <recursion_inputs.json> \\
        --repeat T3.2c ... --out results/p1_training/p1_training.json.gz
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

from attnbench.figures import results as RS  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict import p1_training as PT  # noqa: E402
from attnbench.predict.commit import CLOCK_SKEW_ALLOWED_S  # noqa: E402
from attnbench.predict.training import HardwareFeatures  # noqa: E402
import _load  # noqa: E402
import gate_a as GA  # noqa: E402

FIELDS = ("label", "remote", "local", "ground_truth", "link")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repeat", nargs=len(FIELDS), action="append", required=True,
                    metavar=tuple(f.upper() for f in FIELDS))
    ap.add_argument("--b-failed", default=None, help="why T3.2b is not used (repeat a alone)")
    ap.add_argument("--clock-skew-s", type=float, default=CLOCK_SKEW_ALLOWED_S)
    ap.add_argument("--out", required=True)
    return ap


def repeat_session(r: dict, *, clock_skew_s: float, store=None) -> PT.RepeatSession:
    results, starts = GA.ground_truth(Path(r["ground_truth"]))
    cp = _load.load(r["remote"], r["local"], started_at=starts, clock_skew_s=clock_skew_s,
                    banked_at=None, store=store)
    link = json.loads(Path(r["link"]).read_text())
    if link.get("fake"):
        raise PT.TrainingRowsError(f"{r['link']} is a dry run's")
    return PT.RepeatSession(label=r["label"], remote=r["remote"], cp=cp, results=results,
                            hardware=HardwareFeatures(**link["hardware"]))


def main(argv=None, store=None) -> int:
    args = parser().parse_args(argv)
    reps = [dict(zip(FIELDS, r)) for r in args.repeat]
    if len(reps) > 2:
        print("p1_training: at most a pool's two repeats", file=sys.stderr)
        return 2
    try:
        sessions = [repeat_session(r, clock_skew_s=args.clock_skew_s, store=store) for r in reps]
        a, b = sessions[0], (sessions[1] if len(sessions) > 1 else None)
        out = PT.training_rows(a, b, b_failed=args.b_failed)
        rec = PT.to_record(out, a.hardware)
    except (PT.TrainingRowsError, RS.ResultsError, C.PredictionCommitError, OSError,
            KeyError, TypeError, ValueError) as e:
        print(f"p1_training: {e}", file=sys.stderr)
        return 2
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    PT.dump(rec, path)
    print(f"wrote {path}: {len(rec['rows'])} rows from {rec['repeats']}, sha256 {rec['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
