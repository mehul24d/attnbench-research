"""Record a pairing as banked in its prediction ledger (ruling 1e, 11 Nov 2026).

Run once, when the pairing's ground truth is banked and its training file is
uploaded; writes `predictions/<PAIRING>/ledger/banked.json` in the results
bucket with the store's no-clobber put. The object's creation time is the
pairing's banking time every later prediction reads (`t61_session.banked_times`),
in place of a `--banked-at` typed by hand. A second run for the same pairing
refuses.

    python3 scripts/bank_pairing.py --pairing P1 --training-file results/p1_training/p1_training.json.gz \\
        --bucket-path banked/p1_training/p1_training_t32dc_20261109.json.gz

The training file's sha256 is computed here and must be the lock's
`p1_training.file_sha256` for P1. Nothing here is run by the session runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from attnbench.predict.ledger import Ledger, LedgerError  # noqa: E402


def bank(store, prefix: str, pairing: str, training_file: Path, bucket_path: str,
         lock: dict) -> dict:
    sha = hashlib.sha256(Path(training_file).read_bytes()).hexdigest()
    if pairing == "P1":
        want = (lock.get("p1_training") or {}).get("file_sha256")
        if sha != want:
            raise LedgerError(f"{training_file} has sha256 {sha}; the lock's P1 training "
                              f"file is {want}")
    entry = {"training_file_sha256": sha, "training_file_bucket_path": bucket_path,
             "banked_by": "scripts/bank_pairing.py"}
    Ledger(store, prefix, pairing).record_banked(entry)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairing", required=True)
    ap.add_argument("--training-file", required=True)
    ap.add_argument("--bucket-path", required=True)
    args = ap.parse_args(argv)
    import t31_steps as st
    import t61_session as S
    from attnbench.predict.commit import GsutilStore
    lock = json.loads(Path(st.LOCK_FILE).read_text())
    try:
        entry = bank(GsutilStore(), S.results_bucket() + "/", args.pairing,
                     Path(args.training_file), args.bucket_path, lock)
    except LedgerError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(entry, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
