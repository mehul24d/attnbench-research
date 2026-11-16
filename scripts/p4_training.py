"""P4's training file for the later pairings' committed fits
(`attnbench/predict/p4_training.py`; the adviser's rulings dated 16 Nov 2026,
scoring wave, item 4: P2's script and rules, applied to P4).

REMOTE is P4's committed prediction, its path under the store's prefix as the
receipt names it (`predictions/P4/<session>/prediction_P4_<t>.json`), read
through `load_committed` with its local copy LOCAL, the training rows P4 was
fitted on (P1's file, the lock's, and P2's, the pairing ledger's P2 entry) and
P1's and P2's banking times from the pairing ledger. Each `--block` is one of
P4's block session directories (P4 ran in Spot blocks); `block_drift.collect`
keeps the passed ones. LINK is a block's `recursion_inputs.json` (the link
figures).

    python3 scripts/p4_training.py --remote predictions/P4/<s>/prediction_P4_<t>.json \\
        --local <local copy> --link <recursion_inputs.json> \\
        --p1-training <p1 file> --p2-training <p2 file> \\
        --block <t64_p4_..._b3> --block <t64_p4_..._b4> \\
        --out results/p4_training/p4_training.json.gz

It reads the bucket (the prediction, its ledger entries); ask before running it.
"""

from __future__ import annotations

import argparse
import hashlib
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
from attnbench.predict import p4_training as P4T  # noqa: E402
from attnbench.predict.intensity import shape_for  # noqa: E402
from attnbench.predict.ledger import Ledger, LedgerError, record_cells  # noqa: E402
from attnbench.predict.training import HardwareFeatures  # noqa: E402
import gate_a as GA  # noqa: E402
import t61_session as S  # noqa: E402
import t64_session as T  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--link", required=True)
    ap.add_argument("--p1-training", required=True)
    ap.add_argument("--p2-training", required=True)
    ap.add_argument("--block", action="append", required=True)
    ap.add_argument("--clock-skew-s", type=float, default=C.CLOCK_SKEW_ALLOWED_S)
    ap.add_argument("--out", required=True)
    return ap


def banked(store, prefix, files: dict) -> dict:
    """{pairing: banked_at} for P4's banked pairings, each ledger entry's
    training file the one read here."""
    out = {}
    for p in T.T64.banked:
        got = Ledger(store, prefix, p).banked()
        if got is None:
            raise P4T.TrainingRowsError(f"{p} has no banked entry in the pairing ledger")
        if got[1]["training_file_sha256"] != files[p]:
            raise P4T.TrainingRowsError(f"the ledger banked {p}'s file "
                                        f"{got[1]['training_file_sha256']}; this reads {files[p]}")
        out[p] = got[0]
    return out


def session(args, store, prefix) -> P4T.PairingSession:
    gt_dir = S.step_dir(Path(), "ground_truth", T.T64).name
    got = BD.collect(args.block, ground_truth_dir=gt_dir)
    blocks, starts = [], []
    for d in args.block:
        m = json.loads((Path(d) / "manifest.json").read_text())
        if m["block"]["id"] not in got["blocks"]:
            continue
        results, st = GA.ground_truth(Path(d) / gt_dir)
        blocks.append((m["block"]["id"], results))
        starts += st
    paths = {"P1": Path(args.p1_training), "P2": Path(args.p2_training)}
    files = {p: hashlib.sha256(f.read_bytes()).hexdigest() for p, f in paths.items()}
    banked_at = banked(store, prefix, files)
    # The pairing's earliest ground-truth start over the prediction's cells (ruling 1c).
    cells = record_cells(json.loads(Path(args.local).read_text()))
    first = Ledger(store, prefix, P4T.PAIRING).earliest_start(cells)
    start = C.ground_truth_start(starts)
    if first is not None:
        start = min(start, first)
    rows = S.training_rows(T.T64, path=paths["P1"], p2_path=paths["P2"])
    cp = C.load_committed(store, prefix + args.remote, ground_truth_started_at=start,
                          clock_skew=timedelta(seconds=args.clock_skew_s),
                          banked_at=banked_at, local=args.local,
                          training_rows=rows, training_files=files)
    link = json.loads(Path(args.link).read_text())
    if link.get("fake"):
        raise P4T.TrainingRowsError(f"{args.link} is a dry run's")
    return P4T.PairingSession(remote=args.remote, cp=cp, blocks=blocks,
                              hardware=HardwareFeatures(**link["hardware"]),
                              model=shape_for("primary"),
                              unpassed=got["discarded"] + got["unpassed"])


def main(argv=None, store=None, prefix=None) -> int:
    args = parser().parse_args(argv)
    if store is None:
        store, prefix = C.GsutilStore(), S.results_bucket() + "/"
    try:
        s = session(args, store, prefix)
        rec = P4T.to_record(P4T.training_rows(s), s.hardware)
    except (P4T.TrainingRowsError, GateAError, BD.BlockDriftError, C.PredictionCommitError,
            LedgerError, S.StepFailed, OSError, KeyError, TypeError, ValueError) as e:
        print(f"p4_training: {e}", file=sys.stderr)
        return 2
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    P4T.dump(rec, path)
    print(f"wrote {path}: {len(rec['rows'])} rows from blocks {rec['blocks']}, "
          f"{len(rec['left_out'])} left out, sha256 {rec['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
