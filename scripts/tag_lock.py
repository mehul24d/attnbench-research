#!/usr/bin/env python3
"""Check the lock commit and tag it locally (T4.4, Gate B).

Gate B: "All [lock] values fixed, two adversarial reads done, lock commit
tagged." This script checks what a script can check and tags HEAD; the
adversarial reads (T4.3) are the researcher's to confirm. It refuses unless:

  - the working tree is clean (rule 3: nothing measured or tagged from a
    dirty tree), untracked files included;
  - the lock file is tracked and committed at HEAD, unchanged;
  - every value T4.1 lists is fixed in the lock file, with its source, and
    the file names the plan it was written against (`attnbench.lock.checklist`);
  - the tag is new and named lock-YYYY-MM-DD.

It then makes an annotated tag on HEAD whose message records the lock file's
and the plan's sha256. It never pushes: pushing the tag is the researcher's
explicit yes (CLAUDE.md rule 11), and the command is printed for it.

Usage, from the repo root:
    python3 scripts/tag_lock.py --lock-file docs/lock_values.json --tag lock-2026-11-12
Check only, no tag:
    python3 scripts/tag_lock.py --lock-file docs/lock_values.json --tag lock-2026-11-12 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.lock import checklist  # noqa: E402

TAG_RE = re.compile(r"^lock-\d{4}-\d{2}-\d{2}$")
PLAN_PATH = "docs/RESEARCH_PLAN.md"


class Refused(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise Refused(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def check_lock_commit(repo: Path, lock_file: str, tag: str) -> dict:
    """Everything the tag needs, or `Refused`. Returns what the tag records."""
    if not TAG_RE.match(tag):
        raise Refused(f"the tag {tag!r} is not lock-YYYY-MM-DD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise Refused(f"the working tree is not clean:\n{status}")
    if _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}", check=False).strip():
        raise Refused(f"the tag {tag} already exists")
    tracked = _git(repo, "ls-files", "--", lock_file).strip()
    if tracked != lock_file:
        raise Refused(f"{lock_file} is not tracked; the lock values must be committed")
    head = _git(repo, "rev-parse", "HEAD").strip()
    lock_bytes = (repo / lock_file).read_bytes()
    plan_text = _git(repo, "show", f"HEAD:{PLAN_PATH}")
    try:
        lock = json.loads(lock_bytes)
    except json.JSONDecodeError as e:
        raise Refused(f"{lock_file} is not JSON: {e}") from e
    result = checklist.check(lock, plan_text)
    if not result.complete:
        raise Refused("the lock values are not complete:\n" + result.report())
    return {"head": head, "lock_file": lock_file,
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "plan_sha256": checklist.plan_sha256(plan_text),
            "n_values": len(result.items)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock-file", required=True, help="path relative to the repo root")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dry-run", action="store_true", help="check only; make no tag")
    args = ap.parse_args(argv)

    repo = Path(args.repo)
    try:
        rec = check_lock_commit(repo, args.lock_file, args.tag)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"{rec['n_values']} lock values complete at {rec['head'][:12]}; "
          f"lock {rec['lock_sha256'][:12]}, plan {rec['plan_sha256'][:12]}")
    if args.dry_run:
        print("dry run: no tag made")
        return 0
    msg = (f"Lock commit (T4.4).\n\nlock file: {rec['lock_file']}\n"
           f"lock sha256: {rec['lock_sha256']}\nplan sha256: {rec['plan_sha256']}\n"
           f"values: {rec['n_values']}\n")
    _git(repo, "tag", "-a", args.tag, "-m", msg, rec["head"])
    print(f"tagged {args.tag} locally. Push it only with the researcher's yes:\n"
          f"    git push origin {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
