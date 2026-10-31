"""No hold, gap or wait on the prediction's sync is a literal.

`attnbench.predict.commit` holds the one number: `REQUIRED_SYNC_GAP_S` (60),
`SYNC_WAIT_MARGIN_S` and their sum `PLANNED_SYNC_WAIT_S` (90). A session runner
that adds its own "60 + 5" gets a gap the analysis does not know about: its
holds pass its own test and the analysis refuses the stamp, or the other way
round. The older check (`test_audit_fixes_commit`) only matched NAMED
assignments (`MIN_GAP_S = 60`), so `target = synced + 65` went through; this
one reads every statement in every runner and library module.

A statement is suspect when it touches sync, hold, gap or guard and holds a
number that is one of the gap's values or a plausible hand-rolled sum of them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from attnbench.predict import commit as C

ROOT = Path(__file__).resolve().parents[1]
THEME = re.compile(r"sync|hold|gap_s|skew|guard", re.IGNORECASE)

# The gap's own values and the sums a runner would write by hand (gap plus a
# guard of 1 to 30 s), as plain numbers in seconds.
BANNED = {float(C.REQUIRED_SYNC_GAP_S) + k for k in range(0, 61)} | {float(C.SYNC_WAIT_MARGIN_S)}


def sources():
    paths = sorted((ROOT / "scripts").glob("*.py"))
    paths += sorted(p for p in (ROOT / "attnbench").rglob("*.py")
                    if "_vendor" not in p.parts and p != Path(C.__file__).resolve())
    return paths


CONTEXT = (ast.BinOp, ast.Compare, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.keyword,
           ast.Return, ast.Expr)


def literal_hits(text: str, name: str = "<src>") -> list[str]:
    """`file:line: expression` for each banned number (bools excluded) whose
    nearest enclosing expression or assignment mentions the theme."""
    tree = ast.parse(text, name)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool) and float(node.value) in BANNED):
            continue
        ctx = parents.get(node)
        while ctx is not None and not isinstance(ctx, CONTEXT):
            ctx = parents.get(ctx)
        # A comparison or sum nested in a statement is judged with its statement's
        # target (`target = synced + 65`, `if t - synced < 60`).
        outer = ctx
        while outer is not None and not isinstance(outer, (ast.stmt, ast.keyword)):
            outer = parents.get(outer)
        seg = ast.unparse(ctx) if ctx is not None else ""
        if isinstance(outer, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            seg += " " + ast.unparse(outer.targets[0] if isinstance(outer, ast.Assign) else outer.target)
        if THEME.search(seg):
            hits.append(f"{name}:{node.lineno}: {seg[:90]}")
    return hits


@pytest.mark.parametrize("path", sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_sync_hold_is_a_literal(path):
    assert literal_hits(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))) == []


def test_the_scan_catches_the_literals_it_exists_for():
    # The scan is only worth anything if it fails on the shapes that went through.
    for src in ("target = epoch(r['synced_at']) + 65",
                "HOLD_S = 90.0",
                "time.sleep(hold_for_sync_margin + 60)",
                "if t - synced < 60: raise SystemExit"):
        assert literal_hits(src), src
    # And it is quiet on the name and on unrelated numbers.
    for src in ("target = epoch(r['synced_at']) + PLANNED_SYNC_WAIT_S",
                "timeout = 60",
                "gap = start - synced_at"):
        assert literal_hits(src) == [], src


def _session_runners():
    return [p for p in sorted((ROOT / "scripts").glob("*_session.py"))]


@pytest.mark.parametrize("path", _session_runners(), ids=lambda p: p.name)
def test_a_runner_that_holds_for_the_sync_reads_the_planned_wait(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mentions = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    holds = any(re.search(r"hold", m, re.I) for m in mentions) or "synced_at" in path.read_text()
    if holds and "t64" not in path.name:      # the P4 runner reuses the P2 runner's `S`
        assert "PLANNED_SYNC_WAIT_S" in mentions | names
