"""A module-scoped import hook that strips Python asserts from one package (T2.17).

The asserts-removed XAttention arm (plan Section 5.5) runs the authors'
unmodified `xattn` package with its Python `assert` statements and
`if __debug__` blocks removed, and nothing else changed. Whole-process
`python -O` is not used (Appendix A, 12 Oct 2026), and nothing is copied
(rule 8). So this module loads the package a second time, under an alias,
compiling each of its source files with `optimize=1`:

    install()                                    # once per process
    from attnbench_xattn_noassert.src.Xattention import xattn_estimate

  - **Only that package.** The finder answers only for the alias. Inside
    the stripped modules, `import xattn...` is redirected to the alias by a
    module-scoped `__import__`, so the stripped code never calls the
    shipped code. Everything else they import (torch, triton,
    `block_sparse_attn`, shared with mean-pool) is the process's own,
    unchanged, and `sys.flags.optimize` stays 0.
  - **Both arms in one process.** The shipped arm imports `xattn` as usual
    and keeps its asserts; the two are different module objects.
  - **No bytecode cache.** Each file is compiled from its source on import;
    nothing is read from or written to `__pycache__`, so stripped bytecode
    cannot reach the shipped arm through the cache.
  - **A log.** Every module the hook compiled is recorded with the sha256
    of its source.

`session_check` runs the checks the plan requires in every session:

  1. the shipped arm still carries its asserts: each of its live functions
     has the bytecode of an `optimize=0` compile of its source, and the
     package has at least one assert;
  2. the stripped code differs only in assert statements and `if __debug__`
     blocks: an `optimize=1` compile of each file equals an `optimize=0`
     compile of the same file with those statements deleted from its syntax
     tree, and each live stripped function has the `optimize=1` bytecode;
  3. the hook's log matches the package: every compiled file lies inside the
     shipped package's directory, every alias module was compiled by the
     hook (or is a namespace package), no stripped module holds a shipped
     function, class or module, and `sys.flags.optimize` is 0.

What the checks cannot see: an assert that guards a behaviour. Upstream's
`find_blocks_chunked` (commit `e379887`) wraps one tensor assert in
`try: ... except: mask[...] = False`. With the assert gone, the except
branch can no longer run, so on an input where that assert fails the two
arms differ. Check 2 passes, as it should, since only the assert was
removed. The bitwise mask and output gate on the recorded prompts
(`backends.xattention_noassert.AssertGate`) is what catches it.
"""

from __future__ import annotations

import ast
import builtins
import bisect
import dis
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

PACKAGE = "xattn"
ALIAS = "attnbench_xattn_noassert"


class StripError(RuntimeError):
    """A stripped package that is not the shipped one minus its asserts."""


# --- the reference: asserts deleted from the syntax tree ----------------------

def _is_debug_if(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
            and node.test.id == "__debug__")


class _DeleteAsserts(ast.NodeTransformer):
    def _stmts(self, stmts: list) -> list:
        out = []
        for s in stmts:
            if isinstance(s, ast.Assert):
                continue
            if _is_debug_if(s):
                # optimize=1 folds `__debug__` to the constant False before
                # compiling, so the reference does the same and the compiler
                # lays out both alike. No other use of `__debug__` is touched,
                # so one would show up as a difference.
                s.test = ast.copy_location(ast.Constant(False), s.test)
            out.append(self.visit(s))
        return out

    def generic_visit(self, node):
        for name, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                kept = self._stmts(value)
                if not kept:
                    kept = [ast.copy_location(ast.Pass(), value[0])]
                setattr(node, name, kept)
            elif isinstance(value, list):
                for x in value:
                    if isinstance(x, ast.AST):
                        self.visit(x)
            elif isinstance(value, ast.AST):
                self.visit(value)
        return node


def delete_asserts(source: str) -> ast.Module:
    """The source's syntax tree with every `assert` statement deleted (an
    emptied block becomes `pass`) and every `if __debug__:` test replaced by
    the constant False, which is what optimize=1 does to it."""
    tree = _DeleteAsserts().visit(ast.parse(source))
    return ast.fix_missing_locations(tree)


def count_asserts(source: str) -> int:
    return sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(source)))


# --- comparing bytecode --------------------------------------------------------

_JUMPS = frozenset(getattr(dis, "hasjump", ())) | frozenset(dis.hasjrel) | frozenset(dis.hasjabs)


def code_signature(code: types.CodeType) -> tuple:
    """Instructions with NOPs dropped, jumps as instruction indices and
    nested code compared recursively. Line numbers and byte offsets are left
    out: deleting a statement from the tree leaves a `pass` whose NOP moves
    every later offset without changing what runs."""
    kept = [i for i in dis.get_instructions(code) if i.opname != "NOP"]
    offsets = [i.offset for i in kept]
    out = []
    for i in kept:
        if isinstance(i.argval, types.CodeType):
            arg = ("code", code_signature(i.argval))
        elif i.opcode in _JUMPS and isinstance(i.argval, int):
            arg = ("jump", bisect.bisect_left(offsets, i.argval))
        else:
            arg = i.argrepr
        out.append((i.opname, arg))
    return (code.co_name, tuple(out))


def _codes(code: types.CodeType):
    yield code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            yield from _codes(c)


def assertion_ops(code: types.CodeType) -> int:
    """LOAD_ASSERTION_ERROR instructions in a code object and its children."""
    return sum(1 for co in _codes(code) for i in dis.get_instructions(co)
               if i.opname == "LOAD_ASSERTION_ERROR")


def check_only_asserts_removed(source: str, path: str) -> int:
    """Check 2 for one file. Returns its number of assert statements."""
    stripped = compile(source, path, "exec", dont_inherit=True, optimize=1)
    reference = compile(delete_asserts(source), path, "exec", dont_inherit=True, optimize=0)
    if code_signature(stripped) != code_signature(reference):
        raise StripError(
            f"{path}: compiling with optimize=1 changes more than its assert "
            f"statements and `if __debug__` blocks")
    return count_asserts(source)


# --- the hook -----------------------------------------------------------------

@dataclass(frozen=True)
class CompiledModule:
    alias_name: str
    real_name: str
    path: str
    source_sha256: str
    asserts_removed: int


class _StrippedLoader(importlib.abc.Loader):
    def __init__(self, finder: "StrippedFinder", real_name: str, path: str):
        self.finder, self.real_name, self.path = finder, real_name, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = Path(self.path).read_bytes()
        text = source.decode("utf-8")
        n = check_only_asserts_removed(text, self.path)
        code = compile(source, self.path, "exec", dont_inherit=True, optimize=1)
        module.__dict__["__builtins__"] = self.finder.builtins
        self.finder.log.append(CompiledModule(module.__name__, self.real_name, self.path,
                                              hashlib.sha256(source).hexdigest(), n))
        exec(code, module.__dict__)


class StrippedFinder(importlib.abc.MetaPathFinder):
    """Answers for `alias` and its submodules; no other name."""

    def __init__(self, package: str = PACKAGE, alias: str = ALIAS):
        if package == alias or alias.startswith(package + "."):
            raise ValueError("the alias must not lie inside the package")
        self.package, self.alias = package, alias
        self.log: list[CompiledModule] = []
        real_import = builtins.__import__

        def redirecting_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == package or name.startswith(package + ".")):
                name = alias + name[len(package):]
            return real_import(name, globals, locals, fromlist, level)

        self.builtins = dict(builtins.__dict__, __import__=redirecting_import)

    def real_name(self, alias_name: str) -> str:
        return self.package + alias_name[len(self.alias):]

    def _locate(self, real: str) -> Optional[tuple[Optional[str], Optional[list[str]]]]:
        """(source file, package directories) of a module of the shipped
        package, without importing any of it. A namespace portion has no
        file. Plain lists throughout: the import system's namespace paths
        look their parent up in sys.modules, where the shipped package need
        not be."""
        parent, _, last = real.rpartition(".")
        if not parent:
            # The top level may come from an editable install's own finder.
            spec = importlib.util.find_spec(real)
            if spec is None:
                return None
            dirs = (list(spec.submodule_search_locations)
                    if spec.submodule_search_locations is not None else None)
            origin = spec.origin if spec.origin not in (None, "namespace") else None
            return origin, dirs
        found = self._locate(parent)
        if found is None or found[1] is None:
            return None
        portions = []
        for d in found[1]:
            base = Path(d) / last
            if (base / "__init__.py").is_file():
                return str(base / "__init__.py"), [str(base)]
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                if (Path(d) / (last + suffix)).is_file():
                    return str(Path(d) / (last + suffix)), None
            if (Path(d) / (last + ".py")).is_file():
                return str(Path(d) / (last + ".py")), None
            if base.is_dir():
                portions.append(str(base))
        return (None, portions) if portions else None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.alias and not fullname.startswith(self.alias + "."):
            return None
        real = self.real_name(fullname)
        found = self._locate(real)
        if found is None:
            return None
        origin, dirs = found
        if origin is None:
            if dirs is None:
                raise ImportError(f"{real} has no source to strip")
            ns = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            ns.submodule_search_locations = dirs
            return ns
        if not origin.endswith(".py"):
            raise ImportError(f"{real} is {origin!r}; only Python source can be "
                              f"stripped, and nothing else may come from the package")
        out = importlib.util.spec_from_file_location(
            fullname, origin, loader=_StrippedLoader(self, real, origin),
            submodule_search_locations=dirs)
        out.cached = None
        return out


def installed(package: str = PACKAGE, alias: str = ALIAS) -> Optional[StrippedFinder]:
    for f in sys.meta_path:
        if isinstance(f, StrippedFinder) and (f.package, f.alias) == (package, alias):
            return f
    return None


def install(package: str = PACKAGE, alias: str = ALIAS) -> StrippedFinder:
    """Put the finder on `sys.meta_path` once; returns it."""
    found = installed(package, alias)
    if found is not None:
        return found
    finder = StrippedFinder(package, alias)
    sys.meta_path.insert(0, finder)
    return finder


def uninstall(package: str = PACKAGE, alias: str = ALIAS) -> None:
    """Remove the finder and every alias module (tests use this)."""
    sys.meta_path[:] = [f for f in sys.meta_path if not (
        isinstance(f, StrippedFinder) and (f.package, f.alias) == (package, alias))]
    for name in [n for n in sys.modules if n == alias or n.startswith(alias + ".")]:
        del sys.modules[name]


# --- the session checks --------------------------------------------------------

def _module_functions(module: types.ModuleType) -> list[types.FunctionType]:
    """Functions defined in this module's file: plain, in classes, or held
    by a wrapper (`triton.jit` keeps the function as `.fn`)."""
    path = getattr(module, "__file__", None)
    out = []

    def take(obj):
        fn = obj if isinstance(obj, types.FunctionType) else getattr(obj, "fn", None)
        if isinstance(fn, types.FunctionType) and fn.__code__.co_filename == path:
            out.append(fn)

    for obj in vars(module).values():
        take(obj)
        if isinstance(obj, type) and obj.__module__ == module.__name__:
            for v in vars(obj).values():
                take(getattr(v, "__func__", v))
    return out


def _check_live_code(module: types.ModuleType, optimize: int) -> int:
    """Every live function of `module` has the bytecode of an `optimize`
    compile of its file. Returns the number of functions checked."""
    path = module.__file__
    compiled = compile(Path(path).read_bytes(), path, "exec", dont_inherit=True,
                       optimize=optimize)
    by_key = {(c.co_name, c.co_firstlineno): c for c in _codes(compiled)}
    fns = _module_functions(module)
    for fn in fns:
        key = (fn.__code__.co_name, fn.__code__.co_firstlineno)
        if key not in by_key or code_signature(fn.__code__) != code_signature(by_key[key]):
            raise StripError(
                f"{module.__name__}.{fn.__qualname__} does not have the bytecode "
                f"of an optimize={optimize} compile of {Path(path).name}")
    return len(fns)


def _in_package(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


@dataclass(frozen=True)
class SessionCheck:
    optimize_flag: int
    stripped_modules: tuple[str, ...]
    asserts_removed: int
    shipped_modules: tuple[str, ...]
    shipped_asserts: int
    functions_checked: int


def session_check(finder: Optional[StrippedFinder] = None) -> SessionCheck:
    """The plan's per-session checks (module docstring, 1 to 3). Both arms
    must already be imported. Raises StripError on any failure."""
    finder = finder or installed()
    if finder is None:
        raise StripError("the hook is not installed")
    pkg, alias = finder.package, finder.alias
    if _optimize_flag() != 0:
        raise StripError(f"sys.flags.optimize is {_optimize_flag()}; the whole "
                         f"process must run unoptimised, with only {pkg} stripped")
    if not finder.log:
        raise StripError("the hook has compiled nothing; import the stripped arm first")
    shipped_root = _package_root(pkg)

    # 3. the log matches the package
    logged = {m.alias_name: m for m in finder.log}
    if len(logged) != len(finder.log):
        raise StripError("a module was compiled twice")
    for m in finder.log:
        if not _inside(Path(m.path), shipped_root):
            raise StripError(f"{m.alias_name} was compiled from {m.path}, outside "
                             f"{pkg}'s directory")
        if m.real_name != finder.real_name(m.alias_name):
            raise StripError(f"{m.alias_name} logged as {m.real_name}")
    alias_mods = {n: mod for n, mod in sys.modules.items() if _in_package(n, alias)}
    for name, mod in alias_mods.items():
        if name not in logged and getattr(mod, "__file__", None) is not None:
            raise StripError(f"{name} is loaded but the hook did not compile it")
        for key, obj in vars(mod).items():
            if key.startswith("__"):
                continue
            owner = obj.__name__ if isinstance(obj, types.ModuleType) else \
                getattr(obj, "__module__", None)
            if isinstance(owner, str) and _in_package(owner, pkg):
                raise StripError(f"{name}.{key} is the shipped arm's {owner}; the "
                                 f"stripped code must not call it")

    # 2. only asserts removed, and the live stripped code is that compile
    checked = 0
    removed = 0
    for m in finder.log:
        removed += check_only_asserts_removed(Path(m.path).read_text(), m.path)
        if m.alias_name in alias_mods:
            checked += _check_live_code(alias_mods[m.alias_name], optimize=1)

    # 1. the shipped arm still carries its asserts
    shipped = {n: mod for n, mod in sys.modules.items()
               if _in_package(n, pkg) and getattr(mod, "__file__", None)}
    for m in finder.log:
        if m.real_name not in shipped:
            raise StripError(f"the shipped arm has not imported {m.real_name}; "
                             f"import both arms before the check")
    shipped_asserts = 0
    for name in sorted(shipped):
        checked += _check_live_code(shipped[name], optimize=0)
        shipped_asserts += count_asserts(Path(shipped[name].__file__).read_text())
    if shipped_asserts == 0:
        raise StripError(f"the shipped {pkg} has no assert statements; the check "
                         f"would prove nothing")
    return SessionCheck(0, tuple(sorted(m.real_name for m in finder.log)), removed,
                        tuple(sorted(shipped)), shipped_asserts, checked)


def _optimize_flag() -> int:
    return sys.flags.optimize


def _package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise StripError(f"{package} is not an importable package")
    roots = [Path(p).resolve() for p in spec.submodule_search_locations]
    if len(roots) != 1:
        raise StripError(f"{package} spans {len(roots)} directories")
    return roots[0]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False
