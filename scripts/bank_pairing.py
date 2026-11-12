"""Record a pairing as banked in its prediction ledger (ruling 1e, 11 Nov 2026).

Run once, when the pairing's ground truth is banked and its training file is
uploaded; writes `predictions/<PAIRING>/ledger/banked.json` in the results
bucket with the store's no-clobber put, then reads it back and checks it byte
for byte. A second run for the same pairing refuses.

    python3 scripts/bank_pairing.py --pairing P1 \\
        --training-file results/p1_training/p1_training.json.gz \\
        --bucket-path banked/p1_training/p1_training_t32dc_20261109.json.gz \\
        --record <T3.2c session>/session_cost.txt --record <T3.2d session>/session_cost.txt

The banking time is read from the sessions' records, never typed (adviser's
ruling 2 dated 12 Nov 2026): each `--record` is a session's `session_cost.txt`
as the teardown writes it, and the pairing is banked at the latest
`teardown:` among them, when the last session's results reached the Mac. For
P1 the records must be one per repeat the training file names (`repeats`),
matched by instance name. Every later prediction reads this time
(`t61_session.banked_times`).

The training file's sha256 is computed here and must be the lock's
`p1_training.file_sha256` for P1; before the lock exists, the lock draft's
(`docs/lock_package_draft.json`), and the entry names which it was checked
against. Nothing here is run by the session runners.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from attnbench.predict.ledger import Ledger, LedgerError  # noqa: E402

LOCK_DRAFT = ROOT / "docs" / "lock_package_draft.json"


def read_record(path: Path) -> dict:
    """A session_cost.txt: its instance, its teardown time and its sha256."""
    data = Path(path).read_bytes()
    fields = dict(line.split(": ", 1) for line in data.decode().splitlines() if ": " in line)
    try:
        teardown = datetime.fromisoformat(fields["teardown"].replace("Z", "+00:00"))
    except (KeyError, ValueError) as e:
        raise LedgerError(f"{path} states no teardown time ({e})") from e
    if teardown.tzinfo is None or not fields.get("instance"):
        raise LedgerError(f"{path} needs an instance and a teardown time with its offset")
    return {"instance": fields["instance"], "teardown": teardown.isoformat(),
            "record_sha256": hashlib.sha256(data).hexdigest()}


def banked_at(records: list, repeats=None) -> datetime:
    """The latest teardown; with `repeats`, exactly one record per repeat."""
    if not records:
        raise LedgerError("the banking time is read from the sessions' records: give --record")
    if repeats is not None:
        tags = {r: r.lower().replace(".", "") for r in repeats}       # T3.2c -> t32c
        per = {r: [x for x in records if f"-{t}-" in x["instance"]] for r, t in tags.items()}
        named = sum(per.values(), [])
        if any(len(v) != 1 for v in per.values()) or len(named) != len(records):
            raise LedgerError(
                f"the training file's repeats are {list(repeats)}; the records are "
                f"{[x['instance'] for x in records]}: one record per repeat")
    return max(datetime.fromisoformat(x["teardown"]) for x in records)


def p1_sha_wanted(lock: dict) -> str:
    return (lock.get("p1_training") or {}).get("file_sha256")


def bank(store, prefix: str, pairing: str, training_file: Path, bucket_path: str,
         lock: dict, records: list, checked_against: str = "docs/lock_values.json") -> dict:
    data = Path(training_file).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    repeats = None
    if pairing == "P1":
        want = p1_sha_wanted(lock)
        if sha != want:
            raise LedgerError(f"{training_file} has sha256 {sha}; the lock's P1 training "
                              f"file is {want}")
        repeats = json.loads(gzip.decompress(data)).get("repeats")
        if not repeats:
            raise LedgerError(f"{training_file} names no repeats to match the records to")
    entry = {"training_file_sha256": sha, "training_file_bucket_path": bucket_path,
             "banked_at": banked_at(records, repeats).isoformat(),
             "banked_at_from": sorted(records, key=lambda x: x["teardown"]),
             "sha256_checked_against": checked_against,
             "banked_by": "scripts/bank_pairing.py"}
    led = Ledger(store, prefix, pairing)
    led.record_banked(entry)
    got = led.banked()
    if got is None or got[1] != {**entry, "pairing": pairing}:
        raise LedgerError(f"{pairing}'s banked entry read back as {got}, not as written")
    return got[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairing", required=True)
    ap.add_argument("--training-file", required=True)
    ap.add_argument("--bucket-path", required=True)
    ap.add_argument("--record", action="append", default=[],
                    help="a session's session_cost.txt (repeat per session)")
    args = ap.parse_args(argv)
    import t31_steps as st
    import t61_session as S
    from attnbench.predict.commit import GsutilStore
    lock_path = Path(st.LOCK_FILE)
    if not lock_path.exists() and args.pairing == "P1":
        lock_path = LOCK_DRAFT
    lock = json.loads(lock_path.read_text())
    try:
        entry = bank(GsutilStore(), S.results_bucket() + "/", args.pairing,
                     Path(args.training_file), args.bucket_path, lock,
                     [read_record(Path(r)) for r in args.record],
                     lock_path.relative_to(ROOT).as_posix())
    except LedgerError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(entry, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
