"""Regenerate every paper table from banked data, and compare each with the
committed copy (the researcher's paper-quality wave of 28 Nov 2026, item 7).
CPU only and offline; nothing is launched and no store is contacted. Run it
from a clean clone:

    python3 scripts/paper_reproduce.py --banked <attnbench_backups dir> --out <dir>

What runs, in order (each step's output goes under --out, never into docs/):

  1. score_hypotheses.py (offline, from the scoring wave's mirror): H0-H3 as
     scored at lock-2026-11-14-r3, compared on "scores" with
     docs/scoring/hypotheses_scored_20261116.json;
  2. exploratory_wave.py: compared with docs/exploratory/results_20261116.json;
  3. exploratory_residual.py: compared with docs/exploratory/residual_20261122.json;
  4. paper_tables.py: compared with docs/paper/exploratory/paper_tables.json;
  5. t11_qa1_within_limit.py on the tracked T1.1 parquets: its own check
     refuses unless the certified all-rows bounds reproduce;
  6. paper_figures.py on steps 1 and 4's regenerated outputs: the paper's six
     tables and figures (paper skeleton wave, 29 Nov 2026), compared byte for
     byte with paper/generated/ and paper/figures/.

The scoring wave's inputs file names absolute paths on the Mac where it ran,
and its P2 prediction copy lived in a scratch directory since purged. This
script rewrites those paths under --banked and reads P2's prediction from the
mirror instead (the scorer checks it against its sha256 either way).

Not reproduced here, and said so in the report: P3's exploratory score (its
script is on the unmerged branch p3-g6-2026-11-26). The report is written to <out>/REPRODUCE.md. The exit
code is 0 only if every step ran and matched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2_SESSION = "bucket_20261117/attnbench-a100-p2e-20261114-0546/results/t61_p2_20261114T0022Z"
P2_TRACES = ("bucket_tierA_20261121/attnbench-a100-p2e-20261114-0546/results/"
             "t61_p2_20261114T0022Z/12_ground_truth/eager/traces/trace_*.json")
P4_TRACES = "p4_traces_20261116/*/results/*/trace_*.json"
P4_SELECT = ("bucket_20261117/attnbench-h100-p4s-20261116-170657-b3-a9/results/"
             "t64_p4_20261116T113724Z_b3/10_dense_select")
MIRROR = "scoring_20261116/mirror"
INPUTS = "scoring_20261116/run/inputs.json"


def repoint(banked: Path, out: Path) -> Path:
    """The scoring wave's inputs file with its paths under `banked`."""
    text = (banked / INPUTS).read_text()
    # The banked directory as the scoring wave's Mac named it: the prefix of
    # its P1 training file, which sits at <banked>/p1_training/.
    p1 = json.loads(text)["training_files"]["P1"]
    old = p1[:p1.index("/p1_training/")]
    spec = json.loads(text.replace(old, str(banked)))
    spec["P2"]["local"] = None
    spec["P2"]["sessions"] = [str(banked / P2_SESSION)]
    missing = [p for k in ("P2", "P4") for p in spec[k].get("sessions") or []
               if not Path(p).is_dir()]
    missing += [p for p in spec["training_files"].values() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"banked inputs missing: {missing}")
    path = out / "inputs_repointed.json"
    path.write_text(json.dumps(spec, indent=1) + "\n")
    return path


def strip_labels(o):
    """Block labels name the local directory a session was read from, which
    differs between Macs; everything else must match."""
    if isinstance(o, dict):
        return {k: strip_labels(v) for k, v in o.items() if k != "block"}
    if isinstance(o, list):
        return [strip_labels(v) for v in o]
    return o


def diff_paths(a, b, path=""):
    if type(a) is not type(b):
        return [path or "/"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}")
            else:
                out += diff_paths(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diff_paths(x, y, f"{path}[{i}]")]
    return [] if a == b else [path or "/"]


def compare_tree(got: Path, want: Path) -> str:
    """Every file the generator wrote must equal the committed copy, byte for byte."""
    files = sorted(f for f in got.rglob("*") if f.is_file())
    bad = [str(f.relative_to(got)) for f in files
           if not (want / f.relative_to(got)).is_file()
           or (want / f.relative_to(got)).read_bytes() != f.read_bytes()]
    if not files:
        return "FAILED: nothing generated"
    return (f"identical ({len(files)} files: 5 tables, 2 figures with their data)" if not bad
            else f"DIFFERS in {len(bad)} of {len(files)} files, first: {bad[:3]}")


def run(name: str, cmd: list, log: Path) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    log.write_text(f"$ {' '.join(cmd)}\n\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
    tail = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode, tail[:300]


def compare(got: Path, want: Path, key=None, labels=False) -> str:
    a, b = json.loads(got.read_text()), json.loads(want.read_text())
    if key:
        a, b = a[key], b[key]
    if labels:
        a, b = strip_labels(a), strip_labels(b)
    d = diff_paths(a, b)
    return "identical" if not d else f"DIFFERS at {len(d)} paths, first: {d[:3]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--banked", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    banked, out = a.banked.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    inputs = repoint(banked, out)
    py, m = sys.executable, str(banked / MIRROR)
    tr = ["--traces", f"P2={banked / P2_TRACES}", "--traces", f"P4={banked / P4_TRACES}"]
    steps = [
        ("H0-H3 scoring", [py, "scripts/score_hypotheses.py", "--inputs", str(inputs),
                           "--mirror", m, "--out", str(out / "scoring")],
         out / "scoring" / "hypotheses_scored.json",
         ROOT / "docs/scoring/hypotheses_scored_20261116.json", "scores", False),
        ("exploratory wave (16 Nov)", [py, "scripts/exploratory_wave.py", "--inputs", str(inputs),
                                       "--mirror", m, *tr, "--out", str(out / "results_20261116.json")],
         out / "results_20261116.json", ROOT / "docs/exploratory/results_20261116.json", None, True),
        ("residual split (22 Nov)", [py, "scripts/exploratory_residual.py", "--inputs", str(inputs),
                                     "--mirror", m, *tr, "--out", str(out / "residual_20261122.json")],
         out / "residual_20261122.json", ROOT / "docs/exploratory/residual_20261122.json", None, False),
        ("paper tables (28 Nov)", [py, "scripts/paper_tables.py", "--inputs", str(inputs),
                                   "--mirror", m, *tr, "--dense-select", f"P4={banked / P4_SELECT}",
                                   "--out-dir", str(out / "paper")],
         out / "paper" / "paper_tables.json", ROOT / "docs/paper/exploratory/paper_tables.json",
         None, False),
        ("qa_1/32768 within-limit sensitivity", [py, "scripts/t11_qa1_within_limit.py"],
         None, None, None, False),
        ("paper tables and figures (29 Nov)", [py, "scripts/paper_figures.py", "--scored",
                                               str(out / "scoring" / "hypotheses_scored.json"),
                                               "--tables", str(out / "paper" / "paper_tables.json"),
                                               "--out", str(out / "paper_artifacts")],
         out / "paper_artifacts", ROOT / "paper", "tree", False),
    ]
    rows, ok = [], True
    for i, (name, cmd, got, want, key, labels) in enumerate(steps, 1):
        code, tail = run(name, cmd, out / f"step{i}.log")
        if code != 0:
            res = f"FAILED (exit {code}): {tail}"
        elif got is None:
            res = "ran; its built-in check reproduced the certified all-rows bounds"
        elif key == "tree":
            res = compare_tree(got, want)
        elif not got.exists():
            res = f"FAILED: {got.name} not written"
        else:
            res = compare(got, want, key, labels)
            if labels and res == "identical":
                res = "identical except block labels (local directory names)"
        ok &= res.startswith(("identical", "ran;"))
        rows.append((i, name, res))
    rows += [(len(steps) + 1, "P3 exploratory score", "NOT RUN: scripts/score_p3_exploratory.py "
              "is on the unmerged branch p3-g6-2026-11-26 (584516a), not on this commit")]
    md = ["# Reproduction report", "", f"Commit `{head}`{' (DIRTY tree)' if dirty else ''}; "
          f"python {sys.version.split()[0]}. Banked data read from `--banked`; logs per step "
          "beside this file.", "", "| # | Step | Result |", "|---|---|---|"]
    md += [f"| {i} | {n} | {r} |" for i, n, r in rows]
    (out / "REPRODUCE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
