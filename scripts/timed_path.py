#!/usr/bin/env python3
"""The timed ground-truth path's code hash at a commit (the training rule of
8 Nov 2026, fifth paste; Appendix A): the learned predictor's targets pool
every session whose timed ground-truth path is byte-identical.

The path, declared here before T3.2c exists:

  - every `attnbench` module in the static import closure of `ROOTS` (each
    `import` and `from` statement anywhere in a file, inside functions
    too, followed through `attnbench`), each file's bytes;
  - in `scripts/t31_steps.py`, `ENTRY` and every top-level function of that
    file it names, followed transitively, each function's source segment.

Read from git (`git show <commit>:<path>`), never from the working tree, so
two commits are compared without a checkout.

    python3 scripts/timed_path.py 3f50aab HEAD          # per-item table and verdict
    python3 scripts/timed_path.py 3f50aab HEAD --json
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("attnbench/ground_truth.py", "attnbench/components/shakedown_arms.py",
         "attnbench/dense_selection.py", "attnbench/clock_sampling.py",
         "attnbench/compiled_checks.py", "attnbench/backends/compiled.py")
RUNNER = "scripts/t31_steps.py"
ENTRY = "cmd_ground_truth_one"


@functools.lru_cache(maxsize=None)
def show(commit: str, path: str) -> str | None:
    p = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True,
                       text=True)
    return p.stdout if p.returncode == 0 else None


def _module_paths(commit: str, name: str) -> list[str]:
    rel = name.replace(".", "/")
    return [p for p in (f"{rel}.py", f"{rel}/__init__.py") if show(commit, p) is not None]


def _imports(path: str, src: str) -> set[str]:
    pkg = path[:-3].replace("/", ".")
    if pkg.endswith(".__init__"):
        pkg = pkg[: -len(".__init__")]
    else:
        pkg = pkg.rsplit(".", 1)[0]
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                up = parts[: len(parts) - node.level + 1]
                base = ".".join(up + ([base] if base else []))
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names}
    return {m for m in out if m == "attnbench" or m.startswith("attnbench.")}


def closure(commit: str) -> dict[str, str]:
    """{path: sha256 of its bytes} over the import closure of `ROOTS`."""
    commit = _resolve(commit)
    todo, seen = list(ROOTS), {}
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        src = show(commit, p)
        if src is None:
            raise SystemExit(f"{p} is not in {commit}")
        seen[p] = hashlib.sha256(src.encode()).hexdigest()
        for m in _imports(p, src):
            for q in _module_paths(commit, m):
                todo.append(q)
            parent = m.rsplit(".", 1)[0]
            for q in _module_paths(commit, parent):     # a package's __init__ runs on import
                todo.append(q)
    return dict(sorted(seen.items()))


def _resolve(commit: str) -> str:
    return subprocess.run(["git", "rev-parse", commit], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def runner_functions(commit: str) -> dict[str, str]:
    """{function: sha256 of its source} from `ENTRY` through `RUNNER`'s own
    top-level functions it names."""
    src = show(commit, RUNNER)
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    todo, seen = [ENTRY], {}
    while todo:
        f = todo.pop()
        if f in seen or f not in funcs:
            continue
        seg = ast.get_source_segment(src, funcs[f])
        seen[f] = hashlib.sha256(seg.encode()).hexdigest()
        todo += [n.id for n in ast.walk(funcs[f]) if isinstance(n, ast.Name) and n.id in funcs]
    return dict(sorted(seen.items()))


def path_hash(commit: str) -> dict:
    full = _resolve(commit)
    files, funcs = closure(full), runner_functions(full)
    items = {**{f"file {k}": v for k, v in files.items()},
             **{f"{RUNNER}::{k}": v for k, v in funcs.items()}}
    return {"commit": full, "items": items,
            "sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}


def compare(a: str, b: str) -> dict:
    ha, hb = path_hash(a), path_hash(b)
    keys = sorted(set(ha["items"]) | set(hb["items"]))
    diff = {k: [ha["items"].get(k), hb["items"].get(k)] for k in keys
            if ha["items"].get(k) != hb["items"].get(k)}
    return {"a": {"commit": ha["commit"], "sha256": ha["sha256"]},
            "b": {"commit": hb["commit"], "sha256": hb["sha256"]},
            "items": len(keys), "identical": not diff, "differing": diff}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    c = compare(args.a, args.b)
    if args.json:
        print(json.dumps(c, indent=1))
        return 0
    print(f"{c['a']['commit'][:7]} {c['a']['sha256'][:16]}  {c['b']['commit'][:7]} "
          f"{c['b']['sha256'][:16]}  {c['items']} items: "
          f"{'byte-identical' if c['identical'] else 'NOT identical'}")
    for k, (x, y) in c["differing"].items():
        print(f"  {k}: {(x or 'absent')[:12]} -> {(y or 'absent')[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
