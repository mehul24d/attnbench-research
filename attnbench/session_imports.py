"""The third-party modules a session's step scripts can import, found by a
static scan, and the check that each one imports on the instance.

The L4 rehearsal's first attempt ran ten steps and stopped at the eleventh
because `lightgbm` was not on the image (docs/launch_cards/t61_rehearsal_l4.md).
Nothing before that step imports it, so no earlier step could say so. This scan
starts from the session's scripts, follows every first-party import (the
`attnbench` package and the other files in `scripts/`) and every other script
that a scanned file names, and collects the modules that belong to neither the
standard library nor this repository. Imports are found wherever they stand,
inside functions too; the ones under `if TYPE_CHECKING:` are not runtime
imports and the ones under `try: ... except ImportError` are optional, so those
are reported apart and not required.

What a static scan cannot see: a module named in a string and loaded with
`importlib.import_module`, and a script reached only through a shell variable.
The names below cover the first kind that this repository uses (`DYNAMIC`)."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# Distribution names where the import name differs; the refusal names the
# package to install, not only the module that failed.
DISTRIBUTION = {"yaml": "pyyaml", "sklearn": "scikit-learn", "pynvml": "nvidia-ml-py",
                "PIL": "pillow", "bs4": "beautifulsoup4", "omegaconf": "hydra-core",
                "hydra": "hydra-core", "fla": "flash-linear-attention",
                "flash_attn": "flash-attn", "flashinfer": "flashinfer-python",
                "block_sparse_attn": "block-sparse-attn"}

# Imported by name from a string; none yet, kept so the first one is added here.
DYNAMIC: tuple = ()

# Names that are not installed packages: the assert-free XAttention copy that
# `assert_strip` builds at run time, and the XAttention checkout, which the
# session puts on sys.path (the preflight's own probe and step 4 exercise it).
GENERATED = frozenset({"attnbench_xattn_noassert", "xattn"})


def _stdlib() -> frozenset:
    return frozenset(sys.stdlib_module_names) | {"__future__"}


def _first_party(root: Path) -> frozenset:
    names = {p.stem for p in (root / "scripts").rglob("*.py")}     # scripts/figures/_load.py too
    names |= {p.name for p in root.iterdir() if (p / "__init__.py").exists()}
    names |= {p.stem for p in root.glob("*.py")}
    return frozenset(names)


def _module_file(root: Path, dotted: str) -> Path | None:
    """The repository file a first-party dotted name stands for."""
    head, _, rest = dotted.partition(".")
    for cand in sorted((root / "scripts").rglob(f"{head}.py")):
        return cand
    path = root.joinpath(head, *rest.split(".")) if rest else root / head
    for cand in (path.with_suffix(".py"), path / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _guards(tree: ast.AST) -> tuple[set, set]:
    """Node ids of the imports under `if TYPE_CHECKING:` and the ones in a
    `try:` body whose handler catches ImportError (or ModuleNotFoundError)."""
    typing_only, optional = set(), set()

    def under(nodes, into):
        for n in nodes:
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    into.add(id(sub))

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or \
                    (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"):
                under(node.body, typing_only)
        elif isinstance(node, ast.Try):
            # A handler that re-raises (learned.py's "lightgbm is not installed")
            # makes the import required; one that goes on without it, optional.
            caught = set()
            for h in node.handlers:
                if any(isinstance(x, ast.Raise) for x in ast.walk(h)):
                    continue
                if h.type is None:
                    caught.add("ImportError")
                for e in (ast.walk(h.type) if h.type is not None else ()):
                    if isinstance(e, ast.Name):
                        caught.add(e.id)
            if caught & {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}:
                under(node.body, optional)
    return typing_only, optional


def _imports(path: Path, root: Path) -> tuple[set, set, set, set]:
    """(runtime modules, optional modules, scripts named, first-party files
    to follow) of one file; the first two are dotted names."""
    text = path.read_text()
    tree = ast.parse(text)
    typing_only, optional_ids = _guards(tree)
    required, optional = set(), set()
    pkg = None
    try:
        rel = path.relative_to(root)
        if rel.parts[0] != "scripts":
            pkg = ".".join(rel.with_suffix("").parts[:-1] if rel.name != "__init__.py"
                           else rel.parts[:-1])
    except ValueError:
        pass
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in typing_only:
            continue
        into = optional if id(node) in optional_ids else required
        if isinstance(node, ast.Import):
            into.update(a.name for a in node.names)
            continue
        if node.level:                         # relative: stays inside the package
            if pkg is None:
                continue
            parts = pkg.split(".")
            base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
            mod = ".".join([*base, *([node.module] if node.module else [])])
        else:
            mod = node.module or ""
        into.add(mod)
        for a in node.names:                   # `from pkg import submodule`
            into.add(f"{mod}.{a.name}")
    # Scripts it runs: a string that is exactly a script's file name (the
    # `py("host_cpu_probe.py", ...)` and `script="t61_rehearsal.py"` forms),
    # not a file name inside a sentence or a docstring.
    named = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and re.fullmatch(r"[A-Za-z0-9_]+\.py", n.value)}
    return required, optional, {n for n in named if (root / "scripts" / n).exists()}, set()


def scan(entry_scripts, *, shell_files=(), root: Path = ROOT) -> dict:
    """{"required": {top-level module: [files that import it]}, "optional": {...},
    "files": [every repository file reached]} from `entry_scripts` (file names
    in scripts/) and the scripts the `shell_files` run."""
    stdlib, mine = _stdlib(), _first_party(root)
    queue = [root / "scripts" / s for s in entry_scripts]
    for sh in shell_files:
        queue += [root / "scripts" / n for n in
                  re.findall(r"scripts/([A-Za-z0-9_]+\.py)", Path(sh).read_text())
                  if (root / "scripts" / n).exists()]
    seen: set = set()
    req: dict = {}
    opt: dict = {}
    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        required, optional, named, _ = _imports(path, root)
        for into, mods in ((req, required), (opt, optional)):
            for m in mods:
                top = m.partition(".")[0]
                if not top or top in stdlib or top in GENERATED:
                    continue
                if top in mine:
                    target = _module_file(root, m)
                    if target is not None and target not in seen:
                        queue.append(target)
                    continue
                into.setdefault(top, set()).add(str(path.relative_to(root)))
        queue += [root / "scripts" / n for n in named]
    for m in DYNAMIC:
        req.setdefault(m, set()).add("DYNAMIC")
    rel = sorted(str(p.relative_to(root)) for p in seen)
    return {"required": {k: sorted(v) for k, v in sorted(req.items())},
            "optional": {k: sorted(v) for k, v in sorted(opt.items()) if k not in req},
            "files": rel}


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name.split("[")[0].split("=")[0].split("@")[0].split(">")[0]
                  .split("<")[0].strip().lower())


def extras(root: Path = ROOT) -> dict:
    """{extra group: [distribution names]} from pyproject.toml; the file is
    plain enough to read without tomllib, which Python 3.10 lacks."""
    text = (root / "pyproject.toml").read_text()
    block = text.split("[project.optional-dependencies]", 1)[-1].split("\n[", 1)[0]
    return {m.group(1): re.findall(r'"([^"]+)"', m.group(2))
            for m in re.finditer(r"^(\w+)\s*=\s*\[(.*?)^\]", block, re.S | re.M)}


def is_kernel(module: str, root: Path = ROOT) -> bool:
    """The `kernels` extra is installed per machine and per architecture, so a
    session need not have all of it; the probe (step 5) tries each arm that the
    session runs. These are reported, not required."""
    dists = {_normal(d) for d in extras(root).get("kernels", [])}
    mod = _normal(module)
    return mod in dists or _normal(DISTRIBUTION.get(module, "")) in dists or \
        any(mod.startswith(d) for d in dists)


_PROBE = (
    "import importlib, json, sys\n"
    "out = {}\n"
    "for m in json.loads(sys.argv[1]):\n"
    "    try:\n"
    "        importlib.import_module(m); out[m] = None\n"
    "    except BaseException as e:\n"
    "        out[m] = f'{type(e).__name__}: {e}'.splitlines()[0][:200]\n"
    "print(json.dumps(out))\n")


def import_problems(modules, *, python=None) -> list:
    """Each module imported in a fresh interpreter (`python` is a path or an
    argv prefix); one line per module that does not import, naming the package
    to install."""
    modules = sorted(modules)
    if not modules:
        return []
    argv = [python] if isinstance(python, str) else list(python or [sys.executable])
    got = subprocess.run([*argv, "-c", _PROBE, json.dumps(modules)],
                         capture_output=True, text=True)
    try:
        status = json.loads(got.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return [f"the import check itself failed (rc {got.returncode}): {got.stderr.strip()[-200:]}"]
    return [f"{DISTRIBUTION.get(m, m)} is missing or broken (import {m}: {why})"
            for m, why in status.items() if why]


def session_import_problems(entry_scripts, *, shell_files=(), python=None,
                            root: Path = ROOT) -> tuple[list, dict]:
    found = scan(entry_scripts, shell_files=shell_files, root=root)
    kernels = sorted(m for m in found["required"] if is_kernel(m, root))
    need = [m for m in found["required"] if m not in kernels]
    found["kernels"] = kernels                 # reported, not required
    found["kernel_problems"] = import_problems(kernels, python=python)
    return import_problems(need, python=python), found
