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

Skips when whole result stages are absent, like the other banked-data tests.
Not on bare `results/` existence -- it is gitignored but carries
force-committed evidence files, so it exists in every clone (instance #52) --
and not on a file count either, which a partial bucket sync clears while still
lacking the inputs (instance #53). See `_missing_stage_dirs()`.
"""

from __future__ import annotations

import pathlib
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


def _missing_stage_dirs(root: Path | None = None) -> list[str]:
    """The directories documented input paths live in that are absent here.

    The gate, and it is deliberately about DIRECTORIES rather than a file
    count (instance #53: a raw `> 20` file count is cleared by any partial
    bucket sync, which then FAILS this test instead of skipping it -- one
    partial-sync tree cleared it with 1,275 files and 16 missing inputs).

    Directories are the right granularity because of what this test is FOR.
    Every defect it was written for was a wrong FILENAME inside a directory
    that existed: `run_s1a_comparison.py` and `check_dense_canary.py` both
    named `results/s1a/accuracy.parquet` while `results/s1a/` held
    `accuracy_band2048.parquet`. So a missing directory means "this checkout
    does not have that stage" -- skip -- and a present directory missing the
    named file means "the usage line is wrong" -- fail. A file count cannot
    tell those apart; this can.
    """
    base = REPO if root is None else root
    return sorted({str(pathlib.Path(path).parent)
                   for _, path in documented_paths()
                   if not (base / path).parent.is_dir()})


@pytest.mark.skipif(bool(_missing_stage_dirs()),
                    reason=f"this checkout is missing whole result stages, so "
                           f"it cannot say whether a usage line names the "
                           f"right file within one: {_missing_stage_dirs()}")
def test_every_documented_input_path_exists():
    missing = [(s, p) for s, p in documented_paths()
               if not (REPO / p).exists()]
    assert not missing, (
        "usage blocks name files that do not exist:\n  "
        + "\n  ".join(f"{s}: {p}" for s, p in missing)
        + "\nA usage line is the most-copied line in a script. Name the file "
          "the session actually wrote.")


# --- the gate, break-tested in both directions ------------------------------
#
# Instance #53's lesson: a skip branch nobody has watched take is as unverified
# as an assertion nobody has watched fail. Both branches are exercised here
# against constructed trees, so neither depends on someone having built the
# right checkout by hand once.

def _stage_tree(root: Path, *, complete: bool) -> None:
    """A tree holding the directories every documented input path names.
    `complete=False` drops one stage, which is what a partial sync looks like.
    """
    dirs = sorted({pathlib.Path(path).parent for _, path in documented_paths()})
    if not complete:
        dirs = dirs[1:]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)


def test_the_gate_skips_a_checkout_missing_a_whole_stage(tmp_path):
    """The branch instance #53 proved nobody had watched."""
    _stage_tree(tmp_path, complete=False)
    missing = _missing_stage_dirs(tmp_path)
    assert missing, (
        "a tree missing an entire result stage reported nothing missing, so "
        "this test would RUN against a partial checkout -- instance #53")


def test_the_gate_runs_when_every_stage_is_present_even_with_files_absent(tmp_path):
    """Anti-vacuity, and the property that makes directories the right
    granularity: all stages present but the FILES absent must still RUN, since
    a wrong filename inside an existing stage is the defect this file exists
    for. A gate keyed on the files themselves could never fail."""
    _stage_tree(tmp_path, complete=True)
    assert _missing_stage_dirs(tmp_path) == [], _missing_stage_dirs(tmp_path)
    still_missing = [p for _, p in documented_paths()
                     if not (tmp_path / p).exists()]
    assert still_missing, (
        "the fixture created the files as well as the directories, so it no "
        "longer distinguishes 'stage absent' from 'filename wrong'")
