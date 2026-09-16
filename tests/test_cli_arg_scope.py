"""Every `args.<x>` must be in scope where it is read.

`scripts/run_accuracy.py` shipped `score_source=args.score_source` inside
`build_generate_fn`, which has no `args` parameter. It raised NameError on
the first real invocation -- after the model had loaded and the clocks had
locked, thirty seconds into a rented GPU session. Nothing on CPU caught it:
`--dry-run` substitutes `_dry_run_only` and never calls `build_generate_fn`
at all, so the dry run that was supposed to sanity-check CLI wiring passed
against the broken wiring. That is #32 (a test's coverage is bounded by its
fixture) reaching the one function the dry run exists to skip.

A static check is the right instrument here precisely because the dynamic
one needs a GPU and a 1.5B download. It is cheap, it covers every script at
once, and it cannot be fooled by a code path a fixture declines to enter.
"""
import ast
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_free_args_reference_outside_main(path: Path):
    tree = ast.parse(path.read_text())
    module_level = {
        n.targets[0].id
        for n in tree.body
        if isinstance(n, ast.Assign) and n.targets
        and isinstance(n.targets[0], ast.Name)
    }
    offenders = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if fn.name == "main":
            continue
        bound = {a.arg for a in (list(fn.args.args) + list(fn.args.kwonlyargs)
                                 + list(fn.args.posonlyargs))}
        if fn.args.vararg:
            bound.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            bound.add(fn.args.kwarg.arg)
        if "args" in bound or "args" in module_level:
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Name) and node.id == "args"
                    and isinstance(node.ctx, ast.Load)):
                # Assigned locally before use is fine.
                assigned = any(
                    isinstance(a, ast.Name) and a.id == "args"
                    for n2 in ast.walk(fn)
                    if isinstance(n2, ast.Assign)
                    for a in n2.targets
                )
                if not assigned:
                    offenders.append(f"{fn.name}:{node.lineno}")
    assert not offenders, (
        f"{path.name} reads `args` where it is not in scope: {offenders} -- "
        f"this raises NameError only on the real execution path"
    )
