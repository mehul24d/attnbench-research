"""Static check that every script in scripts/ calls attnbench functions with
arguments their signatures actually accept.

**Why this exists.** Scripts are not imported by the test suite -- they are
entry points run by hand on a rented GPU instance. `py_compile` proves they
parse; it cannot prove that `build_examples_by_task_length(grid, seed=0)`
still matches a signature that has since grown a required keyword-only
`count_tokens`. That mismatch is a `TypeError` raised on the first line of
real work, which on this project means: after launching an instance, after
waiting for a model to download, on billed hardware.

It has already happened twice in one session -- both times a signature change
propagated to the library and its tests but missed a script call site, and
both times it was caught by eye rather than by anything automatic. Two misses
is enough.

This runs as part of the ordinary suite, which the session runbook already
executes as a prerequisite, so it costs no new discipline at session time.

**Deliberately conservative.** It only reports a mismatch it is certain
about, because a false positive here would train people to ignore it:

- calls using `*args`/`**kwargs` are skipped (arity is not statically known)
- only functions resolvable from a top-level `from attnbench... import ...`
  are checked
- decorated functions are skipped (the decorator may rewrite the signature)
- it checks for MISSING required parameters and UNKNOWN keyword names, not
  types
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(REPO.glob("scripts/*.py"))


def _imported_attnbench_names(tree: ast.AST) -> dict[str, tuple[str, str]]:
    """local name -> (module, original name), for attnbench imports only."""
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith("attnbench"):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                out[alias.asname or alias.name] = (node.module, alias.name)
    return out


def _resolve(module_name: str, attr: str):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr, None)


def _check_call(func, node: ast.Call) -> str | None:
    """Return a problem description, or None if the call looks fine."""
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if any(k.arg is None for k in node.keywords):   # **kwargs
        return None
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    params = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        # the callee itself accepts arbitrary args -- nothing to verify
        positional_ok = True
    else:
        positional_ok = False

    passed_kw = {k.arg for k in node.keywords}
    n_positional = len(node.args)

    positional_params = [p for p in params
                         if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    bound_positionally = {p.name for p in positional_params[:n_positional]}

    missing = []
    for p in params:
        if p.default is not p.empty:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in bound_positionally or p.name in passed_kw:
            continue
        if p.name == "self":
            continue
        missing.append(p.name)

    known = {p.name for p in params}
    unknown = sorted(passed_kw - known) if not positional_ok else []

    problems = []
    if missing:
        problems.append(f"missing required argument(s) {sorted(missing)}")
    if unknown:
        problems.append(f"unknown keyword argument(s) {unknown}")
    if not problems:
        return None
    return f"{'; '.join(problems)} -- signature is {sig}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_calls_match_attnbench_signatures(script: Path):
    """Every call a script makes into attnbench must satisfy that
    function's current signature.

    A failure here is a script that would raise TypeError on a rented
    instance, which is the expensive place to find out.
    """
    tree = ast.parse(script.read_text())
    imported = _imported_attnbench_names(tree)
    if not imported:
        pytest.skip(f"{script.name} imports nothing from attnbench")

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = imported.get(node.func.id)
        if target is None:
            continue
        func = _resolve(*target)
        if func is None or not callable(func):
            continue
        problem = _check_call(func, node)
        if problem:
            problems.append(f"{script.name}:{node.lineno} {node.func.id}(): {problem}")

    assert not problems, (
        "script call site(s) do not match current attnbench signatures -- "
        "these would raise TypeError at runtime, on billed hardware:\n  "
        + "\n  ".join(problems))


def test_the_checker_actually_detects_a_broken_call(tmp_path):
    """A checker nobody has watched fail is a guess -- the same reasoning as
    the build memory guard. Feed it a call that omits a required
    keyword-only argument and confirm it reports the problem.
    """
    broken = tmp_path / "broken_script.py"
    broken.write_text(
        "from attnbench.accuracy.grid_configs import build_examples_by_task_length\n"
        "build_examples_by_task_length(grid, seed=0)\n"   # no count_tokens
    )
    tree = ast.parse(broken.read_text())
    imported = _imported_attnbench_names(tree)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    func = _resolve(*imported["build_examples_by_task_length"])
    problem = _check_call(func, call)
    assert problem is not None and "count_tokens" in problem


def test_the_checker_accepts_a_correct_call(tmp_path):
    """The other half: it must not flag a valid call, or it becomes noise
    people learn to ignore."""
    ok = tmp_path / "ok_script.py"
    ok.write_text(
        "from attnbench.accuracy.grid_configs import build_examples_by_task_length\n"
        "build_examples_by_task_length(grid, seed=0, count_tokens=f)\n"
    )
    tree = ast.parse(ok.read_text())
    imported = _imported_attnbench_names(tree)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    assert _check_call(_resolve(*imported["build_examples_by_task_length"]), call) is None


def test_the_checker_ignores_calls_it_cannot_verify(tmp_path):
    """**kwargs forwarding is common and statically opaque; flagging it
    would be a false positive."""
    fwd = tmp_path / "fwd_script.py"
    fwd.write_text(
        "from attnbench.accuracy.grid_configs import build_examples_by_task_length\n"
        "build_examples_by_task_length(grid, **opts)\n"
    )
    tree = ast.parse(fwd.read_text())
    imported = _imported_attnbench_names(tree)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    assert _check_call(_resolve(*imported["build_examples_by_task_length"]), call) is None


def test_every_script_is_actually_covered():
    """Guards against the parametrisation silently collecting nothing --
    a passing-but-empty check is worse than no check."""
    assert len(SCRIPTS) >= 5, f"expected several scripts, found {SCRIPTS}"
