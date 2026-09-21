"""Every entry point must import `attnbench` from the working tree.

**The hazard.** `pip install .` (no `-e`) leaves a full copy of the package in
site-packages. It then shadows the working tree for any process that does not
put the repo on `sys.path` first, and it does so silently: the import
succeeds, the names are all there, and the code is whatever it was on the day
it was installed.

This is not hypothetical. The 2026-09-21 audit found exactly that in the
workstation venv used to run every analysis stage -- a non-editable copy dated
2026-09-10, twenty-nine files divergent, carrying a **pre-sink-fix
`masks.py`**, no `numerics.py`, and a `SwappableAttentionModel` with no
`score_source` parameter. Nothing was contaminated, because every
`scripts/*.py` derives the repo root from `__file__` and inserts it at
`sys.path[0]` before importing. That idiom was the only thing standing
between the project and silently running era-1 mask code, and nothing
asserted it.

**Two shapes, and only one of them is `__file__`-relative.**

  `python3 scripts/foo.py`        -- `__file__` is absolute, `.resolve()` is
                                     cwd-independent, so the insert works
                                     from any directory.
  `python3 -m attnbench.foo`      -- resolution happens BEFORE any of our
                                     code runs, from the CURRENT DIRECTORY.
                                     There is no `__file__` to derive from
                                     yet, so the caller's cwd decides which
                                     copy of the package is imported.

`scripts/build_flash_attn.sh` used the second shape and was the one call with
no protection. Verified from `/tmp` on 2026-09-21: it resolved to the stale
site-packages copy. Harmless as it stood -- `build_guards.py` was
byte-identical and imports nothing from the package -- and fixed by setting
`PYTHONPATH` from `BASH_SOURCE` rather than trusting cwd.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _first_attnbench_import(tree: ast.AST) -> int | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(n.split(".")[0] == "attnbench" for n in names):
            if best is None or node.lineno < best:
                best = node.lineno
    return best


def _first_syspath_insert(tree: ast.AST) -> int | None:
    best = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "insert"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "path"):
            if best is None or node.lineno < best:
                best = node.lineno
    return best


PY_SCRIPTS = sorted(SCRIPTS.glob("*.py"))


def test_there_are_scripts_to_check():
    """Anti-vacuity: the parametrisation below is over this glob."""
    assert len(PY_SCRIPTS) > 15, f"only {len(PY_SCRIPTS)} scripts found"
    importers = [p for p in PY_SCRIPTS
                 if _first_attnbench_import(ast.parse(p.read_text()))]
    assert len(importers) > 15, (
        f"only {len(importers)} scripts import attnbench; if that collapsed, "
        f"the import detection is broken and every check below is vacuous")


@pytest.mark.parametrize("script", PY_SCRIPTS, ids=lambda p: p.name)
def test_a_script_puts_the_repo_on_sys_path_before_importing_attnbench(script):
    tree = ast.parse(script.read_text())
    first_import = _first_attnbench_import(tree)
    if first_import is None:
        pytest.skip(f"{script.name} does not import attnbench")
    insert = _first_syspath_insert(tree)
    assert insert is not None, (
        f"{script.name} imports attnbench at line {first_import} and never "
        f"inserts the repo root on sys.path. Run from a venv holding a "
        f"non-editable install, it will import that copy instead of this "
        f"tree, silently. Add:\n"
        f"    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))")
    assert insert < first_import, (
        f"{script.name} inserts the repo root at line {insert}, AFTER its "
        f"first attnbench import at line {first_import}. The insert has no "
        f"effect on an import that already happened.")


def test_the_ordering_check_catches_an_import_before_the_insert(tmp_path):
    """Break-test. The subtle version of this defect is not a missing insert
    -- it is an insert that drifted below an import during an edit."""
    good = tmp_path / "good.py"
    good.write_text("import sys\nfrom pathlib import Path\n"
                    "sys.path.insert(0, 'x')\nfrom attnbench import masks\n")
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nfrom attnbench import masks\n"
                   "sys.path.insert(0, 'x')\n")
    none = tmp_path / "none.py"
    none.write_text("import sys\nfrom attnbench import masks\n")

    for f, insert_line, import_line in ((good, 3, 4), (bad, 3, 2), (none, None, 2)):
        tree = ast.parse(f.read_text())
        assert _first_attnbench_import(tree) == import_line, f.name
        assert _first_syspath_insert(tree) == insert_line, f.name


SH_SCRIPTS = sorted(SCRIPTS.glob("*.sh"))
MODULE_CALL = re.compile(r"python3?\s+-m\s+attnbench\.")


def test_no_shell_script_runs_python_m_attnbench_without_setting_pythonpath():
    """`-m` resolves the package from cwd, before any of our code can run.
    A launcher using it must put the repo root on PYTHONPATH itself,
    derived from its own location rather than from where it was called."""
    offenders = []
    for sh in SH_SCRIPTS:
        text = sh.read_text()
        for m in MODULE_CALL.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            # The PYTHONPATH assignment may prefix the call on the same line
            # or sit on the line above (a `\`-continued command).
            window = "\n".join(text.splitlines()[max(0, line_no - 3):line_no])
            if "PYTHONPATH" not in window:
                offenders.append(f"{sh.name}:{line_no}")
    assert not offenders, (
        "these run `python -m attnbench.…` without putting the repo root on "
        "PYTHONPATH, so the package resolves from the caller's working "
        "directory: " + ", ".join(offenders))


def test_the_shell_check_can_see_a_module_call():
    """Anti-vacuity: if the regex stops matching, the check above passes on
    an empty set forever."""
    assert MODULE_CALL.search("python3 -m attnbench.build_guards --x 1")
    assert MODULE_CALL.search("  python -m attnbench.foo")
    assert not MODULE_CALL.search("python3 -m pytest tests/")
    assert any(MODULE_CALL.search(sh.read_text()) for sh in SH_SCRIPTS), (
        "no shell script invokes `-m attnbench.…` any more. If that is "
        "deliberate, delete this check rather than leaving it to pass on "
        "nothing.")
