"""Prepare (never publish) the anonymised data artifact behind docs/paper/ (the
researcher's paper-quality wave of 28 Nov 2026, item 15). Nothing is
uploaded, pushed or shared; the directory is written locally and stays there
until the researcher decides.

    python3 scripts/build_artifact.py --banked <attnbench_backups dir> \\
        --s4 <Shakedown 4 results dir> --out <artifact dir>

The artifact keeps the banked directories' relative layout under data/, so
`scripts/paper_reproduce.py`, `scripts/null_kernel_bias.py` and
`scripts/pay_bound.py` run against it unchanged (`regenerate.sh`). Files are
APFS clones where the filesystem allows (no extra disk).

Anonymised: the scoring inputs name no store and no local path; the mirror's
store name is replaced in its index keys and its object directory (object
contents and their sha256 are unchanged). `scripts/artifact_scan.py` must then
report no hit before the artifact is offered to anyone.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = "attnbench-store"
INPUTS = "scoring_20261116/run/inputs.json"
MIRROR = "scoring_20261116/mirror"
# (relative path under --banked, glob of files to take, role); a trailing
# "/" takes the whole tree.
P2S = "bucket_20261117/attnbench-a100-p2e-20261114-0546/results/t61_p2_20261114T0022Z"
P4S = "bucket_20261117/attnbench-h100-p4s-20261116-170657-b3-a9/results/t64_p4_20261116T113724Z_b3"
HOSTFILES = "*/{host_profile,host_profile_null_pass,bare_prices}.json"
# What the loaders read of a session: its manifest and its ground truth.
SESSION = "{manifest.json,step_layout.json,%s/*.json,%s/*/gt_*.json,%s/*/stamps.json}"
TAKE = [
    ("scoring_20261116/bucket/attnbench-h100-p4s-20261116-170657-b3-a9/results/"
     "t64_p4_20261116T113724Z_b3", SESSION % (("13_ground_truth",) * 3), "P4 session b3 as scored"),
    ("scoring_20261116/bucket/attnbench-h100-p4s-20261116-184330-b4-a11/results/"
     "t64_p4_20261116T131401Z_b4", SESSION % (("13_ground_truth",) * 3), "P4 session b4 as scored"),
    (P2S, SESSION % (("12_ground_truth",) * 3), "P2 session as scored"),
    (P2S + "/07_host_components", HOSTFILES, "P2 host components as committed"),
    ("bucket_tierA_20261121/attnbench-a100-p2e-20261114-0546/results/t61_p2_20261114T0022Z/"
     "12_ground_truth/eager/traces/", None, "P2 ground-truth traces (profiled)"),
    ("p4_traces_20261116", "*/results/*/trace_*.json", "P4 ground-truth traces (profiled)"),
    (P4S + "/10_dense_select/", None, "P4 dense-selection microbenchmark"),
    (P4S + "/08_host_components", HOSTFILES, "P4 host components as committed"),
    ("aws_p3_g6_20261126T0906Z/session/05_host_components", HOSTFILES, "P3 host components"),
    ("aws_p3_g6_20261126T0906Z/session/09_ground_truth", "{result.json,eager/gt_*.json,eager/stamps.json}",
     "P3 ground truth (untraced)"),
    ("aws_p3_g6_20261126T0906Z/session/12_delay_injection", "*.json", "P3 delay injection (not analysed)"),
    ("p1_training/", None, "P1 training rows"),
    ("p2_training/", None, "P2 training rows"),
]
S4_CELL = "t31_rehearsal_20261101T0723Z/05_host_components/dense_d1_L2048_b1"
# A home directory in a copied file (not a hash-checked one) becomes /home/user.
HOME = re.compile(rb"/(Users|home)/(?!ubuntu\b)[A-Za-z0-9._-]+")
S4_FILES = ("host_profile.json", "recording/sequence.json", "recording/trace.json")


def clone(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        r = subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True)
        if r.returncode:
            shutil.copytree(src, dst)
    else:
        r = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
        if r.returncode:
            shutil.copy2(src, dst)


def expand(pattern: str) -> list[str]:
    if "{" not in pattern:
        return [pattern]
    head, rest = pattern.split("{", 1)
    alts, tail = rest.split("}", 1)
    return [p for a in alts.split(",") for p in expand(head + a + tail)]


def anonymised_inputs(spec: dict) -> dict:
    p1 = spec["training_files"]["P1"]
    old = p1[:p1.index("/p1_training/")]
    text = json.dumps(spec).replace(old, "<BANKED>")
    out = json.loads(text)
    out["bucket"] = f"gs://{STORE}"
    out["P2"]["local"] = None
    out["P2"]["sessions"] = ["<BANKED>/bucket_20261117/attnbench-a100-p2e-20261114-0546/results/"
                             "t61_p2_20261114T0022Z"]
    return out


def anonymise_mirror(src: Path, dst: Path, bucket: str) -> None:
    name = bucket.split("://")[-1].strip("/")
    idx = json.loads((src / "index.json").read_text())
    new = {k.replace(name, STORE): v for k, v in idx.items()}
    if json.dumps(new).count(name):
        raise SystemExit("the mirror index still names the store")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "index.json").write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    for top in (src / "objects").iterdir():
        clone(top, dst / "objects" / (STORE if top.name == name else top.name))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--banked", required=True, type=Path)
    ap.add_argument("--s4", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to write over it")
    data = a.out / "data"
    spec = json.loads((a.banked / INPUTS).read_text())
    roles, redacted = {}, []
    for rel, pattern, role in TAKE:
        src = a.banked / rel
        if pattern is None:
            clone(src, data / rel.rstrip("/"))
            roles[rel.rstrip("/")] = role
            continue
        hits = sorted({f for pat in expand(pattern) for f in glob.glob(str(src / pat))})
        if not hits:
            raise SystemExit(f"nothing under {rel} matches {pattern}")
        for f in hits:
            clone(Path(f), data / Path(f).relative_to(a.banked))
        roles[rel] = role
    for f in S4_FILES:
        dst = data / "s4" / S4_CELL / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = (a.s4 / S4_CELL / f).read_bytes()
        red = HOME.sub(rb"/\1/user", raw)
        dst.write_bytes(red)
        if red != raw:
            redacted.append(dst.relative_to(a.out).as_posix())
    roles["s4"] = "Shakedown 4 (L4, 1 Nov 2026): one cell's null pass and recording"
    anonymise_mirror(a.banked / MIRROR, data / MIRROR, spec["bucket"])
    roles[MIRROR] = "scoring wave mirror of the store (16 Nov 2026), store name replaced"
    inp = data / INPUTS
    inp.parent.mkdir(parents=True, exist_ok=True)
    inp.write_text(json.dumps(anonymised_inputs(spec), indent=1) + "\n")
    roles[INPUTS] = "scoring inputs, anonymised (no store, no local path)"
    rows = []
    for p in sorted(x for x in data.rglob("*") if x.is_file()):
        rel = p.relative_to(a.out).as_posix()
        role = next((r for k, r in roles.items() if rel.startswith("data/" + k.rstrip("/"))), "")
        if rel in redacted:
            role += " (home directory redacted to /home/user)"
        rows.append((rel, p.stat().st_size, sha256(p), role))
    (a.out / "MANIFEST.tsv").write_text("path\tbytes\tsha256\trole\n" + "".join(
        f"{r}\t{n}\t{h}\t{role}\n" for r, n, h, role in rows))
    (a.out / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, _, h, _ in rows))
    for f in ("DATA_DICTIONARY.md", "regenerate.sh"):
        shutil.copy2(ROOT / "docs/paper/artifact" / f, a.out / f)
    print(f"{len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
