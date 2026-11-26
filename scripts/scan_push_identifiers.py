#!/usr/bin/env python3
"""Scan what a push would publish for the account's identifiers (26 Nov 2026).

    python3 scripts/scan_push_identifiers.py [--values FILE] [REV ...]

REV defaults to `HEAD --not --remotes`: the commits no remote has. The lines
those commits add, and their messages, are searched for:

  account     the pinned account number (ATTNBENCH_AWS_ACCOUNT in
              ~/.attnbench/aws_env.sh) and that file's group id and address;
  value       any line of --values FILE (for example an organization or
              policy id read from a log, kept off the screen);
  id12        a 12-digit number standing alone;
  arn         an ARN with an account number (`arn:aws:<service>:<region>:<12 digits>:`);
  session     a role session (`assumed-role/<role>/<session>`, `botocore-session-`);
  keyid       an access-key or role id (AKIA, ASIA, AROA, AIDA + 16);
  org         an organization id (`o-` + 10 to 32) or SCP id (`policy/.../p-`).

The test fakes carry placeholder account numbers, ARNs, role sessions and
organization ids: an id12, arn, session or org hit in `tests/` is allowed when
the line does not hold the pinned account. The account and value classes (the
real values, matched exactly) and keyid are never allowed. The masking code's
own patterns (`arn:aws[...]`) carry no account, so they are not ARNs here.

Each hit is printed as commit, file, class and count, never the text matched.
Exit 0 when clean, 1 on any hit, 2 when git fails.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ENV = Path(os.environ.get("ATTNBENCH_AWS_ENV", Path.home() / ".attnbench" / "aws_env.sh"))
ENV_KEYS = ("ATTNBENCH_AWS_ACCOUNT", "AWS_SECURITY_GROUP_ID", "ATTNBENCH_SSH_FROM_IP")

ID12 = re.compile(r"(?<![0-9A-Za-z])[0-9]{12}(?![0-9A-Za-z])")
PATTERNS = {
    "id12": ID12,
    "arn": re.compile(r"arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:[0-9]{12}:", re.I),
    "session": re.compile(r"assumed-role/[A-Za-z0-9+=,.@_-]+/[A-Za-z0-9+=,.@_-]+|botocore-session-[0-9]"),
    "keyid": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AROA|AIDA)[A-Z0-9]{16}(?![A-Z0-9])"),
    "org": re.compile(r"(?<![A-Za-z0-9-])o-[a-z0-9]{10,32}(?![a-z0-9])|policy/[^\s]*p-[a-z0-9]{8,}"),
}
TESTS_MAY = {"id12", "arn", "session", "org"}


def env_values(path: Path) -> list[str]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Z_]+)=(.*)$", line)
        if m and m.group(1) in ENV_KEYS:
            v = m.group(2).strip().strip("'\"")
            v = v.split("/")[0] if m.group(1) == "ATTNBENCH_SSH_FROM_IP" else v
            if v:
                out.append(v)
    return out


def scan(revs: list[str], secrets: list[str], account: str | None, label: str = "account") -> Counter:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout

    hits: Counter = Counter()
    commits = git("rev-list", *revs).split()
    for c in commits:
        short = c[:7]
        lines = [("(message)", ln) for ln in git("log", "-1", "--format=%B", c).splitlines()]
        path = "?"
        for ln in git("show", "--format=", "--no-color", "--unified=0", c).splitlines():
            if ln.startswith("+++ "):
                path = ln[6:] if ln.startswith("+++ b/") else ln[4:]
            elif ln.startswith("+") and not ln.startswith("+++"):
                lines.append((path, ln[1:]))
        for path, text in lines:
            for v in secrets:
                if v in text:
                    hits[(short, path, label)] += 1
            for cls, rx in PATTERNS.items():
                n = len(rx.findall(text))
                if not n:
                    continue
                if cls in TESTS_MAY and path.startswith("tests/"):
                    if not account or account not in text:
                        continue
                hits[(short, path, cls)] += n
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--values", help="a file of further values to refuse, one per line")
    ap.add_argument("revs", nargs="*")
    a = ap.parse_args(argv)
    revs = a.revs or ["HEAD", "--not", "--remotes"]
    env = env_values(ENV)
    account = env[0] if env and re.fullmatch(r"[0-9]{12}", env[0]) else None
    secrets = list(env)
    extra = []
    if a.values:
        extra = [v.strip() for v in Path(a.values).read_text().splitlines() if v.strip()]
    try:
        hits = scan(revs, secrets, account)
        if extra:
            for (c, p, cls), n in scan(revs, extra, None, "value").items():
                if cls == "value":
                    hits[(c, p, cls)] += n
    except subprocess.CalledProcessError as e:
        print(f"scan_push_identifiers: git failed: {e.stderr.strip()}", file=sys.stderr)
        return 2
    n_commits = len(subprocess.run(["git", "rev-list", *revs], capture_output=True, text=True).stdout.split())
    if not hits:
        print(f"CLEAN: {n_commits} commit(s); account values checked: {len(env)}, "
              f"other values: {len(extra)}; classes: {', '.join(PATTERNS)}")
        return 0
    for (c, p, cls), n in sorted(hits.items()):
        print(f"HIT\t{c}\t{p}\t{cls}\t{n}")
    print(f"NOT CLEAN: {sum(hits.values())} hit(s) in {len({k[0] for k in hits})} commit(s); nothing printed but the class")
    return 1


if __name__ == "__main__":
    sys.exit(main())
