"""T2.17: the module-scoped import hook that strips one package's asserts.

Every test builds its own package on disk under a unique name, so nothing
here touches the real `xattn`. The package mirrors the upstream layout that
matters: a regular top level, a namespace `sub/` with no `__init__.py`, a
star import and a plain import of a sibling by absolute name, a relative
import, a tensor assert inside `try`/`except`, and an `if __debug__` block.

The real package is checked by the last two tests, which run only where it
is installed (the GPU image), and by the CUDA gate in
tests/test_xattention_noassert.py.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

from attnbench import assert_strip as A

_ids = itertools.count()

INIT = '''
"""Top level."""
VERSION = 1
assert VERSION == 1
'''

HELPERS = '''
import torch
DEBUG_CALLS = []

def check_positive(x):
    assert (x > 0).all(), "not positive"
    return x

def guarded(mask):
    try:
        assert (~mask[1:]).all()
    except:
        mask = mask.clone()
        mask[1:] = False
    if __debug__:
        DEBUG_CALLS.append(1)
    return mask

class Box:
    def put(self, n):
        assert n >= 0
        return n
'''

MOD = '''
from {pkg}.sub.helpers import *
import {pkg}.sub.helpers as H
from {pkg} import VERSION

def run(x):
    assert x.dim() == 1
    return check_positive(x) * VERSION

def lazy():
    from {pkg}.sub.helpers import Box
    return Box
'''

REG_INIT = '''
from . import leaf
'''

LEAF = '''
def f(n):
    assert n != 3
    return n
'''


def make_package(tmp_path, monkeypatch, extra=None):
    pkg = f"t217pkg{next(_ids)}"
    root = tmp_path / "site"
    base = root / pkg
    (base / "sub").mkdir(parents=True)
    (base / "reg").mkdir()
    (base / "__init__.py").write_text(INIT)
    (base / "sub" / "helpers.py").write_text(HELPERS)
    (base / "sub" / "mod.py").write_text(MOD.format(pkg=pkg))
    (base / "reg" / "__init__.py").write_text(REG_INIT)
    (base / "reg" / "leaf.py").write_text(LEAF)
    for rel, text in (extra or {}).items():
        (base / rel).write_text(text)
    monkeypatch.syspath_prepend(str(root))
    alias = f"{pkg}_stripped"
    finder = A.install(pkg, alias)
    yield_ = (pkg, alias, base, finder)
    return yield_


@pytest.fixture
def package(tmp_path, monkeypatch):
    out = make_package(tmp_path, monkeypatch)
    yield out
    pkg, alias, _, _ = out
    A.uninstall(pkg, alias)
    for n in [n for n in sys.modules if n == pkg or n.startswith(pkg + ".")]:
        del sys.modules[n]


def both(pkg, alias, name="sub.mod"):
    return (importlib.import_module(f"{pkg}.{name}"),
            importlib.import_module(f"{alias}.{name}"))


# --- what stripping does ---------------------------------------------------------

def test_the_stripped_arm_skips_asserts_the_shipped_arm_keeps(package):
    pkg, alias, _, _ = package
    shipped, stripped = both(pkg, alias)
    bad = torch.tensor([1.0, -1.0])
    with pytest.raises(AssertionError, match="not positive"):
        shipped.run(bad)
    assert torch.equal(stripped.run(bad), bad)
    assert A.assertion_ops(shipped.run.__code__) == 1
    assert A.assertion_ops(stripped.run.__code__) == 0
    assert A.assertion_ops(shipped.H.Box.put.__code__) == 1
    assert A.assertion_ops(stripped.H.Box.put.__code__) == 0


def test_an_assert_inside_try_changes_behaviour_when_it_would_fail(package):
    """The case the session checks cannot see, and the gate must."""
    pkg, alias, _, _ = package
    shipped = importlib.import_module(f"{pkg}.sub.helpers")
    stripped = importlib.import_module(f"{alias}.sub.helpers")
    holds = torch.tensor([True, False, False])
    fails = torch.tensor([True, True, False])
    assert torch.equal(shipped.guarded(holds), stripped.guarded(holds))
    assert shipped.guarded(fails).tolist() == [True, False, False]
    assert stripped.guarded(fails).tolist() == [True, True, False]


def test_debug_blocks_go_too(package):
    pkg, alias, _, _ = package
    shipped = importlib.import_module(f"{pkg}.sub.helpers")
    stripped = importlib.import_module(f"{alias}.sub.helpers")
    m = torch.tensor([True, False])
    shipped.guarded(m)
    stripped.guarded(m)
    assert (len(shipped.DEBUG_CALLS), len(stripped.DEBUG_CALLS)) == (1, 0)


def test_imports_of_the_package_inside_it_resolve_to_the_alias(package):
    pkg, alias, _, _ = package
    _, stripped = both(pkg, alias)
    assert stripped.check_positive.__module__ == f"{alias}.sub.helpers"
    assert stripped.H.__name__ == f"{alias}.sub.helpers"
    assert stripped.lazy().__module__ == f"{alias}.sub.helpers"
    leaf = importlib.import_module(f"{alias}.reg").leaf
    assert leaf.__name__ == f"{alias}.reg.leaf" and leaf.f(3) == 3


def test_the_shipped_package_is_untouched_by_the_alias(package):
    pkg, alias, _, _ = package
    importlib.import_module(f"{alias}.sub.mod")
    assert not any(n == pkg or n.startswith(pkg + ".") for n in sys.modules)


def test_namespace_portions_load_without_a_file(package):
    pkg, alias, _, _ = package
    importlib.import_module(f"{alias}.sub.mod")
    ns = sys.modules[f"{alias}.sub"]
    assert getattr(ns, "__file__", None) is None and list(ns.__path__)


# --- the bytecode cache ------------------------------------------------------------

def test_nothing_is_written_to_the_bytecode_cache(package):
    pkg, alias, base, _ = package
    importlib.import_module(f"{alias}.sub.mod")
    importlib.import_module(f"{alias}.reg")
    assert not list(base.rglob("__pycache__"))


def test_a_planted_cache_file_is_never_read(tmp_path, monkeypatch):
    pkg, alias, base, _ = make_package(tmp_path, monkeypatch)
    try:
        tag = sys.implementation.cache_tag
        cache = base / "sub" / "__pycache__"
        cache.mkdir()
        for opt in ("", ".opt-1", ".opt-2"):
            (cache / f"helpers.{tag}{opt}.pyc").write_bytes(b"not bytecode")
        mod = importlib.import_module(f"{alias}.sub.helpers")
        assert mod.check_positive(torch.tensor([-1.0])).item() == -1.0
    finally:
        A.uninstall(pkg, alias)


# --- the log and the hook's scope ----------------------------------------------------

def test_the_log_names_every_compiled_file_with_its_hash_and_asserts(package):
    pkg, alias, base, finder = package
    importlib.import_module(f"{alias}.sub.mod")
    got = {(m.alias_name, m.real_name, Path(m.path).name, m.asserts_removed)
           for m in finder.log}
    # INIT 1, MOD 1, HELPERS 3 (check_positive, guarded, Box.put).
    assert got == {(alias, pkg, "__init__.py", 1),
                   (f"{alias}.sub.mod", f"{pkg}.sub.mod", "mod.py", 1),
                   (f"{alias}.sub.helpers", f"{pkg}.sub.helpers", "helpers.py", 3)}
    import hashlib
    for m in finder.log:
        assert m.source_sha256 == hashlib.sha256(Path(m.path).read_bytes()).hexdigest()


def test_the_finder_answers_for_its_alias_only(package):
    pkg, alias, _, finder = package
    assert finder.find_spec(pkg) is None
    assert finder.find_spec("torch") is None
    assert finder.find_spec(alias + "x") is None
    assert finder.find_spec(f"{alias}.missing") is None


def test_install_is_idempotent_and_uninstall_removes_everything(package):
    pkg, alias, _, finder = package
    assert A.install(pkg, alias) is finder
    assert sum(isinstance(f, A.StrippedFinder) and f.alias == alias
               for f in sys.meta_path) == 1
    importlib.import_module(f"{alias}.sub.mod")
    A.uninstall(pkg, alias)
    assert A.installed(pkg, alias) is None
    assert not any(n.startswith(alias) for n in sys.modules)


def test_an_alias_inside_the_package_is_refused():
    with pytest.raises(ValueError, match="must not lie inside"):
        A.StrippedFinder("xattn", "xattn.stripped")


def test_a_file_that_optimize_changes_beyond_asserts_is_refused_at_import(tmp_path, monkeypatch):
    pkg, alias, _, _ = make_package(tmp_path, monkeypatch,
                                    extra={"sub/flag.py": "ENABLED = __debug__\n"})
    try:
        with pytest.raises(A.StripError, match="changes more than"):
            importlib.import_module(f"{alias}.sub.flag")
        assert f"{alias}.sub.flag" not in sys.modules
    finally:
        A.uninstall(pkg, alias)


def test_a_compiled_extension_in_the_package_is_refused(tmp_path, monkeypatch):
    pkg, alias, base, _ = make_package(tmp_path, monkeypatch)
    try:
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        (base / "sub" / f"native{suffix}").write_bytes(b"")
        with pytest.raises(ImportError, match="only Python source can be stripped"):
            importlib.import_module(f"{alias}.sub.native")
    finally:
        A.uninstall(pkg, alias)


# --- check 2 on its own: only asserts removed ------------------------------------------

@pytest.mark.parametrize("src,n", [
    ("def f(c):\n    if c:\n        assert c\n    return c\n", 1),
    ("def f(c):\n    try:\n        assert c\n    except:\n        c = 0\n    return c\n", 1),
    ("def f(xs):\n    for x in xs:\n        assert x\n    while xs:\n        assert xs.pop()\n", 2),
    ("def f(c):\n    with c:\n        assert c\n    return 1\n", 1),
    ("def f(c):\n    if __debug__:\n        c += 1\n    try:\n        assert c\n    except:\n        c = 0\n    return c\n", 1),
    ("class K:\n    assert True\n    def m(self):\n        assert self, 'x'\n", 2),
])
def test_check_only_asserts_removed_accepts_these_layouts(src, n):
    assert A.check_only_asserts_removed(src, "l.py") == n


def test_check_only_asserts_removed_counts_and_passes():
    src = textwrap.dedent('''
        def f(x, c):
            if c:
                assert x
            try:
                assert x > 1, "msg"
            except AssertionError:
                x = 0
            if __debug__:
                x += 1
            else:
                x -= 1
            return x
        assert True
    ''')
    assert A.check_only_asserts_removed(src, "f.py") == 3


@pytest.mark.parametrize("src", [
    "x = __debug__\n",
    "if not __debug__:\n    y = 1\n",
])
def test_check_only_asserts_removed_refuses_anything_else_optimize_changes(src):
    with pytest.raises(A.StripError, match="changes more than"):
        A.check_only_asserts_removed(src, "g.py")


def test_delete_asserts_leaves_pass_in_an_emptied_block():
    import ast
    tree = A.delete_asserts("if c:\n    assert x\nelse:\n    y = 1\n")
    assert ast.unparse(tree) == "if c:\n    pass\nelse:\n    y = 1"
    tree = A.delete_asserts("if __debug__:\n    y = 1\nelse:\n    y = 2\n")
    assert ast.unparse(tree) == "if False:\n    y = 1\nelse:\n    y = 2"


def test_code_signature_ignores_lines_but_not_code():
    a = compile("def f(x):\n    return x + 1\n", "a.py", "exec")
    b = compile("\n\n\ndef f(x):\n\n    return x + 1\n", "b.py", "exec")
    c = compile("def f(x):\n    return x + 2\n", "a.py", "exec")
    assert A.code_signature(a) == A.code_signature(b)
    assert A.code_signature(a) != A.code_signature(c)


# --- the session check ----------------------------------------------------------------

def _import_both(pkg, alias):
    for name in ("sub.mod", "reg"):
        importlib.import_module(f"{pkg}.{name}")
        importlib.import_module(f"{alias}.{name}")


def test_session_check_passes_and_reports_counts(package):
    pkg, alias, _, finder = package
    _import_both(pkg, alias)
    chk = A.session_check(finder)
    names = (pkg, f"{pkg}.reg", f"{pkg}.reg.leaf", f"{pkg}.sub.helpers", f"{pkg}.sub.mod")
    assert chk.stripped_modules == names and chk.shipped_modules == names
    # INIT 1 + MOD 1 + HELPERS 3 + LEAF 1; REG_INIT has none.
    assert (chk.asserts_removed, chk.shipped_asserts, chk.optimize_flag) == (6, 6, 0)
    # Live functions: mod run, lazy; helpers check_positive, guarded, Box.put;
    # leaf f -> 6 per arm.
    assert chk.functions_checked == 12


def test_session_check_refuses_before_both_arms_are_imported(package):
    pkg, alias, _, finder = package
    with pytest.raises(A.StripError, match="compiled nothing"):
        A.session_check(finder)
    importlib.import_module(f"{alias}.sub.mod")
    with pytest.raises(A.StripError, match="has not imported"):
        A.session_check(finder)


def test_session_check_refuses_an_optimised_process(package, monkeypatch):
    pkg, alias, _, finder = package
    _import_both(pkg, alias)
    monkeypatch.setattr(A, "_optimize_flag", lambda: 1)
    with pytest.raises(A.StripError, match="sys.flags.optimize is 1"):
        A.session_check(finder)


def test_session_check_catches_a_shipped_arm_without_its_asserts(package, monkeypatch):
    pkg, alias, _, finder = package
    _import_both(pkg, alias)
    shipped = sys.modules[f"{pkg}.sub.helpers"]
    stripped = sys.modules[f"{alias}.sub.helpers"]
    monkeypatch.setattr(shipped.check_positive, "__code__", stripped.check_positive.__code__)
    with pytest.raises(A.StripError, match="optimize=0 compile"):
        A.session_check(finder)


def test_session_check_catches_a_stripped_arm_with_asserts(package, monkeypatch):
    pkg, alias, _, finder = package
    _import_both(pkg, alias)
    shipped = sys.modules[f"{pkg}.sub.helpers"]
    stripped = sys.modules[f"{alias}.sub.helpers"]
    monkeypatch.setattr(stripped.check_positive, "__code__", shipped.check_positive.__code__)
    with pytest.raises(A.StripError, match="optimize=1 compile"):
        A.session_check(finder)


def test_session_check_catches_shipped_code_inside_the_stripped_arm(package, monkeypatch):
    pkg, alias, _, finder = package
    _import_both(pkg, alias)
    monkeypatch.setitem(vars(sys.modules[f"{alias}.sub.mod"]), "check_positive",
                        sys.modules[f"{pkg}.sub.helpers"].check_positive)
    with pytest.raises(A.StripError, match="must not call it"):
        A.session_check(finder)


def test_session_check_catches_a_module_the_hook_did_not_compile(package, monkeypatch):
    pkg, alias, base, finder = package
    _import_both(pkg, alias)
    rogue = types.ModuleType(f"{alias}.sub.rogue")
    rogue.__file__ = str(base / "sub" / "mod.py")
    monkeypatch.setitem(sys.modules, rogue.__name__, rogue)
    with pytest.raises(A.StripError, match="did not compile it"):
        A.session_check(finder)


def test_session_check_catches_a_file_from_outside_the_package(package, tmp_path, monkeypatch):
    pkg, alias, _, finder = package
    _import_both(pkg, alias)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("x = 1\n")
    monkeypatch.setattr(finder, "log", finder.log + [A.CompiledModule(
        f"{alias}.sub.elsewhere", f"{pkg}.sub.elsewhere", str(outside), "0" * 64, 0)])
    with pytest.raises(A.StripError, match="outside"):
        A.session_check(finder)


def test_session_check_refuses_a_package_with_no_asserts(tmp_path, monkeypatch):
    pkg = f"t217bare{next(_ids)}"
    root = tmp_path / "bare"
    (root / pkg).mkdir(parents=True)
    (root / pkg / "__init__.py").write_text("def f():\n    return 1\n")
    monkeypatch.syspath_prepend(str(root))
    finder = A.install(pkg, pkg + "_s")
    try:
        importlib.import_module(pkg)
        importlib.import_module(pkg + "_s")
        with pytest.raises(A.StripError, match="would prove nothing"):
            A.session_check(finder)
    finally:
        A.uninstall(pkg, pkg + "_s")
        sys.modules.pop(pkg, None)


# --- the harness itself does not rely on assert ------------------------------------------

REPO = Path(__file__).resolve().parents[1]

# Every `assert` statement in harness code, by file, with why it is there.
# None is a safety check that a session relies on: the hook leaves
# sys.flags.optimize at 0, so these run, and whole-process -O is not used.
HARNESS_ASSERTS = {
    "attnbench/components/trace_recording.py": (1, "the mechanism zoo's assert on a GPU tensor, the wait under test"),
    "scripts/check_compile_graph_breaks.py": (6, "the assert forms whose graph breaks the script measures"),
    "scripts/check_score_dtype_asymmetry.py": (1, "a dtype sanity check in a one-off diagnostic"),
    "scripts/decide_gla_arm.py": (1, "a sanity check on a constant in a one-off decision script"),
    "scripts/derive_llama_oracle_cost.py": (1, "a key-set sanity check in a one-off derivation"),
    "scripts/repair_reconciliation_identity.py": (6, "a one-off data repair's checks; under -O they would vanish, so it must never run optimised"),
}


def test_every_assert_in_the_harness_is_listed():
    import ast
    found = {}
    for root in ("attnbench", "scripts", "tools"):
        for p in sorted((REPO / root).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if "/_vendor/" in rel:
                continue
            n = sum(isinstance(x, ast.Assert) for x in ast.walk(ast.parse(p.read_text())))
            if n:
                found[rel] = n
    assert found == {k: v[0] for k, v in HARNESS_ASSERTS.items()}


# --- the real package, where it is installed ---------------------------------------------

_HAS_XATTN = importlib.util.find_spec("xattn") is not None


@pytest.mark.skipif(not _HAS_XATTN, reason="x-attention is installed on the GPU image only")
def test_session_check_on_the_installed_xattn():
    pytest.importorskip("block_sparse_attn")
    import xattn.src.Xattention  # noqa: F401
    A.install()
    import attnbench_xattn_noassert.src.Xattention  # noqa: F401
    chk = A.session_check()
    assert "xattn.src.utils" in chk.stripped_modules and chk.shipped_asserts > 0


@pytest.mark.skipif(not _HAS_XATTN, reason="x-attention is installed on the GPU image only")
def test_the_prefill_tensor_asserts_are_gone_from_find_blocks_chunked():
    """Upstream e379887's find_blocks_chunked has four asserts (one on
    arguments, two on tensors in the prefill path, one decoding-only)."""
    pytest.importorskip("block_sparse_attn")
    import xattn.src.utils as shipped
    A.install()
    import attnbench_xattn_noassert.src.utils as stripped
    assert A.assertion_ops(shipped.find_blocks_chunked.__code__) == 4
    assert A.assertion_ops(stripped.find_blocks_chunked.__code__) == 0
