#!/usr/bin/env python3
"""The prediction store of P3 on AWS: `gsutil` as `attnbench.predict.commit.
GsutilStore` calls it, with the Mac as the store (docs/launch_cards/aws_p3_g6.md).

P3 has no GCS write path, and no S3 bucket or instance role is to be made
(adviser, 23 Nov 2026). The in-session `predict` step (`scripts/t31_steps.py`,
inside the timed-path pin, unchanged) syncs through `GsutilStore`, which runs
`gsutil` from PATH. On the instance, `scripts/aws_p3_instance.sh` puts a
`gsutil` first on PATH that runs this file, and sets ATTNBENCH_BUCKET to
`gs://p3-mac` (`BUCKET`), so the step's own code makes the same four calls it
makes on GCP:

  gsutil -q cp -n LOCAL REMOTE   spool LOCAL at REMOTE (never over an object
                                 already there), then wait until the Mac has
                                 witnessed it: pulled it over SSH, checked its
                                 sha256, kept its own copy, sent that copy
                                 back (`<object>.mac`) and written
                                 `<object>.witness` last. Exit 1 if the Mac's
                                 sha256 differs, or if no witness comes within
                                 ATTNBENCH_P3_STORE_WAIT_S (900 s): with SSH
                                 lost at that moment the session stops before
                                 any ground truth (plan default, the card)
  gsutil cat REMOTE              the Mac's copy, as it came back
  gsutil -q stat REMOTE          0 if witnessed; 1 and "No URLs matched" if not
  gsutil stat REMOTE             "Creation time:" the time the Mac's clock read
                                 when it held the whole object, in gsutil's form

So the creation time `sync_prediction` reads, and `hold` waits from, is set by
the Mac's clock, not by the instance writing the prediction. Standard library
only (the instance's /usr/bin/python3).

The Mac's side is `witness_store` in scripts/aws_p3_driver.sh. The spool is
ATTNBENCH_P3_STORE (default ~/p3_store): REMOTE gs://BUCKET/PATH is
`<spool>/BUCKET/PATH`, with `.request` (sha256, size, the instance's time),
`.mac` and `.witness` beside it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

BUCKET = "p3-mac"
WAIT_S = 900.0
POLL_S = 2.0


def spool() -> Path:
    return Path(os.environ.get("ATTNBENCH_P3_STORE", str(Path.home() / "p3_store")))


def local_path(remote: str) -> Path:
    if not remote.startswith(f"gs://{BUCKET}/"):
        raise SystemExit(f"aws_p3_gsutil: not a gs://{BUCKET}/ URI: {remote!r}")
    rel = remote[len("gs://"):]
    if ".." in Path(rel).parts or rel.endswith((".request", ".mac", ".witness")):
        raise SystemExit(f"aws_p3_gsutil: refusing the path {remote!r}")
    return spool() / rel


def sibling(p: Path, suffix: str) -> Path:
    return p.with_name(p.name + suffix)


def witness(p: Path) -> dict | None:
    w = sibling(p, ".witness")
    if not w.exists():
        return None
    return json.loads(w.read_text())


def cp(args: list[str]) -> int:
    if args[:1] != ["-n"] or len(args) != 3:
        print(f"aws_p3_gsutil: cp takes -n LOCAL REMOTE, got {args}", file=sys.stderr)
        return 2
    src, remote = Path(args[1]), args[2]
    dest = local_path(remote)
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not dest.exists():          # never clobbered: the read-back then finds any mismatch
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = sibling(dest, ".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        sibling(dest, ".request").write_text(json.dumps({
            "remote": remote, "sha256": digest, "bytes": len(data),
            "spooled_at_instance": datetime.now(timezone.utc).isoformat()}) + "\n")
    wait_s = float(os.environ.get("ATTNBENCH_P3_STORE_WAIT_S", WAIT_S))
    t0 = time.monotonic()
    while True:
        w = witness(dest)
        if w is not None:
            if w.get("sha256") != hashlib.sha256(dest.read_bytes()).hexdigest():
                print(f"aws_p3_gsutil: the Mac's copy of {remote} has sha256 {w.get('sha256')}, "
                      f"not the spooled object's", file=sys.stderr)
                return 1
            return 0
        if time.monotonic() - t0 > wait_s:
            print(f"aws_p3_gsutil: the Mac did not witness {remote} within {wait_s:.0f} s "
                  f"(SSH lost?); nothing is synced", file=sys.stderr)
            return 1
        time.sleep(float(os.environ.get("ATTNBENCH_P3_STORE_POLL_S", POLL_S)))


def cat(args: list[str]) -> int:
    p = local_path(args[0])
    mac = sibling(p, ".mac")
    if witness(p) is None or not mac.exists():
        print(f"No URLs matched: {args[0]}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(mac.read_bytes())
    return 0


def stat(args: list[str], quiet: bool) -> int:
    p = local_path(args[0])
    w = witness(p)
    if w is None:
        print(f"No URLs matched: {args[0]}", file=sys.stderr)
        return 1
    if not quiet:
        t = datetime.fromisoformat(w["received_utc"].replace("Z", "+00:00"))
        print(f"{args[0]}:\n    Creation time:          {format_datetime(t, usegmt=True)}\n"
              f"    Content-Length:         {p.stat().st_size}\n"
              f"    Hash (sha256):          {w['sha256']}")
    return 0


def main(argv: list[str]) -> int:
    quiet = bool(argv) and argv[0] == "-q"
    if quiet:
        argv = argv[1:]
    if not argv:
        print("aws_p3_gsutil: no command", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "cp":
        return cp(rest)
    if cmd == "cat" and len(rest) == 1:
        return cat(rest)
    if cmd == "stat" and len(rest) == 1:
        return stat(rest, quiet)
    print(f"aws_p3_gsutil: unsupported call {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
