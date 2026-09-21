"""Paths a script's own usage block names must be paths that exist.

**The class of defect.** A usage line is the most-read documentation a script
has, and it is the least checked. `scripts/run_s1a_comparison.py` and
`scripts/check_dense_canary.py` both documented their input as
`results/s1a/accuracy.parquet`, a file the S1a session never produced -- it
wrote `accuracy_band2048.parquet`, `accuracy_bands2.parquet` and
`accuracy_all_bands.parquet`. Anyone pasting either line got "file not
found", which is the benign outcome; the malign one is a usage line that
names a file which exists and is the wrong one.

Two default *output* paths had drifted the same way and are not covered here,
because an output path that does not exist yet is the normal case:

  - `decide_gla_arm.py` defaulted to `results/stage3_s1/gla_arm_verdict.json`
    while the verdict that exists is in `stage3_s1b/`;
  - `run_vectorised_endtoend.py` defaulted to `results/s7_vec_endtoend` while
    the banked output is `results/s8_vec_endtoend`.

Both were corrected by hand on 2026-09-21. The asymmetry is the point and is
why this file checks only inputs: a wrong input path fails loudly the first
time, a wrong output path silently produces a second copy of something while
the first one goes on looking current.

Skips without `results/`, like the other banked-data tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# Only concrete artifact paths. A directory or a glob in a usage line is a
# placeholder; a named .parquet/.json is a claim about a file.
PATH_RE = re.compile(r"(results/[A-Za-z0-9_./-]+\.(?:parquet|json|csv))")


def documented_paths() -> list[tuple[str, str]]:
    """(script, path) for every results/ file named in a module docstring."""
    out = []
    for p in sorted(SCRIPTS.glob("*.py")):
        text = p.read_text()
        m = re.match(r'\s*(?:#!.*\n)?\s*r?"""(.*?)"""', text, re.S)
        if not m:
            continue
        for path in PATH_RE.findall(m.group(1)):
            out.append((p.name, path))
    return out


def test_some_scripts_document_a_results_path():
    """Anti-vacuity. If the docstring regex stops matching -- a script that
    opens with a comment block instead, say -- every check below passes over
    an empty list."""
    found = documented_paths()
    assert len(found) >= 4, (
        f"only {len(found)} documented results/ paths parsed from "
        f"scripts/*.py; the docstring extraction has probably broken")


@pytest.mark.skipif(not (REPO / "results").exists(),
                    reason="banked results/ not present in this checkout")
def test_every_documented_input_path_exists():
    missing = [(s, p) for s, p in documented_paths()
               if not (REPO / p).exists()]
    assert not missing, (
        "usage blocks name files that do not exist:\n  "
        + "\n  ".join(f"{s}: {p}" for s, p in missing)
        + "\nA usage line is the most-copied line in a script. Name the file "
          "the session actually wrote.")
