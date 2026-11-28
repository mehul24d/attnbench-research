"""Scan a directory for cloud and personal identifiers before it is offered
to anyone (the researcher's paper-quality wave of 28 Nov 2026, item 15).

    ARTIFACT_SCAN_LITERALS="$(...)" python3 scripts/artifact_scan.py <dir> [--report <md>]

Two kinds of check, over every file (gzip files and zip members, such as
.npz and .pt, are read decompressed):

  - literals: exact values passed in ARTIFACT_SCAN_LITERALS (one per line;
    the GCP project, the store name, the AWS account, a user name). They are
    never printed: the report names them "literal 1", "literal 2", ...;
  - patterns: ARNs, 12-digit numbers standing alone (the form of an AWS
    account ID), EC2 resource IDs, store URLs, GCP project paths, e-mail
    addresses and home directories.

The report gives counts and file paths only, never a matched value. The exit
code is 0 only if nothing matched.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import zipfile
from pathlib import Path

PATTERNS = {
    "aws_arn": rb"arn:aws[a-z-]*:",
    "aws_12_digit": rb"(?<![0-9A-Za-z.])[0-9]{12}(?![0-9A-Za-z.])",
    "ec2_resource_id": rb"\b(?:i|vol|snap|ami|sg|subnet|vpc|eni|igw|rtb)-[0-9a-f]{8,17}\b",
    # The artifact's own placeholder store name is allowed.
    "store_url": rb"(?:gs|s3)://(?!attnbench-store\b)",
    "gcp_project_path": rb"projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/",
    "email": rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    "home_dir": rb"/(?:Users|home)/(?!(?:ubuntu|user)\b)[A-Za-z0-9._-]+",
}


def blobs(path: Path):
    """(label, bytes) for one file: the file itself, and decompressed
    contents where it is a gzip file or a zip archive."""
    raw = path.read_bytes()
    yield "", raw
    if raw[:2] == b"\x1f\x8b":
        try:
            yield " (gunzipped)", gzip.decompress(raw)
        except OSError:
            pass
    if raw[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for n in z.namelist():
                    yield f" [{n}]", z.read(n)
        except zipfile.BadZipFile:
            pass


def scan(root: Path, literals: list[bytes]) -> dict:
    checks = {f"literal {i + 1}": re.compile(re.escape(v)) for i, v in enumerate(literals)}
    checks.update({k: re.compile(v) for k, v in PATTERNS.items()})
    hits = {k: {} for k in checks}
    files = 0
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        files += 1
        rel = p.relative_to(root).as_posix()
        for label, b in blobs(p):
            for k, rx in checks.items():
                n = len(rx.findall(b))
                if n:
                    hits[k][rel + label] = hits[k].get(rel + label, 0) + n
            for k, rx in checks.items():  # file names too
                if rx.search(rel.encode()):
                    hits[k][rel + " (path)"] = 1
    return {"files": files, "hits": hits}


def report(res: dict, root_name: str) -> str:
    L = [f"# Identifier scan: {root_name}", "",
         f"{res['files']} files scanned (gzip and zip members decompressed). "
         "Counts and paths only; no matched value is printed.", "",
         "| check | files with a hit | hits |", "|---|---|---|"]
    for k, v in res["hits"].items():
        L.append(f"| {k} | {len(v)} | {sum(v.values())} |")
    for k, v in res["hits"].items():
        if v:
            L += ["", f"## {k}", ""] + [f"- `{f}`: {n}" for f, n in sorted(v.items())]
    clean = not any(res["hits"].values())
    L += ["", f"**{'CLEAN: no identifier found.' if clean else 'NOT CLEAN.'}**", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    lits = [x.strip().encode() for x in os.environ.get("ARTIFACT_SCAN_LITERALS", "").splitlines()
            if len(x.strip()) >= 4]
    res = scan(a.root, lits)
    md = report(res, a.root.name)
    if a.report:
        a.report.write_text(md)
    print(md.split("\n\n## ")[0] if "\n\n## " in md else md)
    return 0 if not any(res["hits"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
