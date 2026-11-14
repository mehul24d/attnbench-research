"""P2's training file for P4's committed fits (`attnbench/predict/p2_training.py`;
adviser's ruling 1h of 11 Nov 2026).

REMOTE is P2's committed prediction, its path under the store's prefix as the
receipt names it (`predictions/P2/<session>/prediction_P2_<t>.json`), read
through `load_committed` with its local copy LOCAL, P1's training rows (the
lock's file) and P1's banking time from the pairing ledger. Each `--block` is
one of P2's block session directories; `block_drift.collect` keeps the passed
ones. P2 on Flex-start runs unblocked: `--session` names that one session
directory instead, read as P2's one block, its id the directory's name, once
every step of the session has passed, `late_check` among them (plan default,
flagged, 14 Nov 2026: neither card plans Spot blocks since the -r4 re-cut).
LINK is the session's `recursion_inputs.json` (the link figures).

    python3 scripts/p2_training.py --remote predictions/P2/<s>/prediction_P2_<t>.json \\
        --local <local copy> --link <recursion_inputs.json> \\
        --block results/t61_p2_<b1> --block results/t61_p2_<b2> \\
        --out results/p2_training/p2_training.json.gz
    python3 scripts/p2_training.py ... --session results/t61_p2_<s> --out ...

It reads the bucket (the prediction, its ledger entries); ask before running it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from attnbench.analysis import block_drift as BD  # noqa: E402
from attnbench.analysis.gate_a import GateAError  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict import p2_training as P2T  # noqa: E402
from attnbench.predict.intensity import shape_for  # noqa: E402
from attnbench.predict.ledger import Ledger, LedgerError, record_cells  # noqa: E402
from attnbench.predict.training import HardwareFeatures  # noqa: E402
import gate_a as GA  # noqa: E402
import t61_session as S  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--link", required=True)
    one = ap.add_mutually_exclusive_group(required=True)
    one.add_argument("--block", action="append")
    one.add_argument("--session")
    ap.add_argument("--clock-skew-s", type=float, default=C.CLOCK_SKEW_ALLOWED_S)
    ap.add_argument("--out", required=True)
    return ap


def unblocked(d: Path, gt_dir: str) -> tuple[list, list]:
    """One unblocked session as P2's one block: ([(id, results)], starts).
    Refused unless every step of the session's last attempts passed."""
    m = json.loads((d / "manifest.json").read_text())
    if m.get("block"):
        raise P2T.TrainingRowsError(f"{d} is a block's session: name it with --block")
    if m.get("dry_run"):
        raise P2T.TrainingRowsError(f"{d} is a dry run's session")
    if (d / BD.DISCARD).exists():
        raise P2T.TrainingRowsError(f"{d} carries {BD.DISCARD}")
    bad = BD._unpassed(m, {"steps": S.T61.names})
    if bad:
        raise P2T.TrainingRowsError(f"{d}: steps not passed: {bad}")
    results, starts = GA.ground_truth(d / gt_dir)
    return [(d.name, results)], starts


def session(args, store, prefix) -> P2T.PairingSession:
    gt_dir = S.step_dir(Path(), "ground_truth", S.T61).name
    if args.session:
        blocks, starts = unblocked(Path(args.session), gt_dir)
        got = {"discarded": [], "unpassed": []}
    else:
        got = BD.collect(args.block, ground_truth_dir=gt_dir)
        blocks, starts = [], []
        for d in args.block:
            m = json.loads((Path(d) / "manifest.json").read_text())
            if m["block"]["id"] not in got["blocks"]:
                continue
            results, st = GA.ground_truth(Path(d) / gt_dir)
            blocks.append((m["block"]["id"], results))
            starts += st
    p1 = Ledger(store, prefix, "P1").banked()
    if p1 is None:
        raise P2T.TrainingRowsError("P1 has no banked entry in the pairing ledger")
    files = S.training_files(S.T61)
    if p1[1]["training_file_sha256"] != files["P1"]:
        raise P2T.TrainingRowsError(f"the ledger banked P1's file "
                                    f"{p1[1]['training_file_sha256']}; "
                                    f"this reads {files['P1']}")
    # The pairing's earliest ground-truth start over the prediction's cells (ruling 1c).
    cells = record_cells(json.loads(Path(args.local).read_text()))
    first = Ledger(store, prefix, P2T.PAIRING).earliest_start(cells)
    start = C.ground_truth_start(starts)
    if first is not None:
        start = min(start, first)
    cp = C.load_committed(store, prefix + args.remote, ground_truth_started_at=start,
                          clock_skew=timedelta(seconds=args.clock_skew_s),
                          banked_at={"P1": p1[0]}, local=args.local,
                          training_rows=S.training_rows(S.T61), training_files=files)
    link = json.loads(Path(args.link).read_text())
    if link.get("fake"):
        raise P2T.TrainingRowsError(f"{args.link} is a dry run's")
    return P2T.PairingSession(remote=args.remote, cp=cp, blocks=blocks,
                              hardware=HardwareFeatures(**link["hardware"]),
                              model=shape_for("primary"),
                              unpassed=got["discarded"] + got["unpassed"])


def main(argv=None, store=None, prefix=None) -> int:
    args = parser().parse_args(argv)
    if store is None:
        store, prefix = C.GsutilStore(), S.results_bucket() + "/"
    try:
        s = session(args, store, prefix)
        rec = P2T.to_record(P2T.training_rows(s), s.hardware)
    except (P2T.TrainingRowsError, GateAError, BD.BlockDriftError, C.PredictionCommitError,
            LedgerError, S.StepFailed, OSError, KeyError, TypeError, ValueError) as e:
        print(f"p2_training: {e}", file=sys.stderr)
        return 2
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    P2T.dump(rec, path)
    print(f"wrote {path}: {len(rec['rows'])} rows from blocks {rec['blocks']}, "
          f"{len(rec['left_out'])} left out, sha256 {rec['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
