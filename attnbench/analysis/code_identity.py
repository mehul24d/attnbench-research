"""Was the same code timed? -- asked per backend, not per repository.

`cross_arch._check_commit_consistency` refuses to join segments measured at
different commits, which is the right default and the wrong resolution. The
override it offers (`allow_mixed_commits=True`) is all-or-nothing: it turns
off the check for every backend at once, on the strength of a judgement made
about one of them. This module supplies the judgement instead.

**The concrete case.** Four Stage 2 segments exist at four commits:

    L4-seg1  4e2a28e   L4-seg2  86322c0   L4-seg3  283526c   A100  d2d8ceb

Between seg1 and seg2, `FlexAttentionBackend.forward` was changed to hoist
`create_block_mask` out of the per-call path. That single change moved flex
at seq_len=1024/batch=1 from 4.22 to 40.72 useful TFLOPS -- a 10x measurement
error, documented in `tests/test_timed_region_setup.py`. `NaiveAttention.
forward` was rewritten to a blocked implementation over the same boundary.
Every other backend's timed region -- fa2, sdpa_*, gla, sage, xformers,
block_sparse -- is byte-identical in AST across all four commits.

So "may these segments be joined?" has no single answer. It is yes for six
backends and no for two, and a blanket flag cannot express that. Worse, the
blanket flag is granted on the basis of the backends the reader happened to
check: the flex numbers would have come along silently.

**What is fingerprinted.** Only the code that runs inside the timed region:
the backend class's `forward`, `timed_call` and `make_inputs`, plus
`timing._do_bench`, which is the loop the latency actually comes out of.
Deliberately NOT `timing.measure` -- it changed across this range (first-call
recording, the compile-fallback guard) without touching the steady-state
number, and a check that fires on that is a check that gets overridden.

Fingerprints are AST-structural, so reformatting, comment edits and docstring
rewrites do not trip it. A pure-comment change to a kernel is genuinely not a
different experiment, and a guard that says otherwise trains its reader to
pass the override.

**What this cannot see.** The pinned versions of torch, flash-attn, triton and
the CUDA driver -- which change the compiled kernel without changing a line of
this repository. Those live in each row's provenance stamp (`torch`,
`flash_attn`, `triton`, `driver`) and are checked separately; this module
answers only "did OUR code change", and says so rather than implying it has
cleared the whole question.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Backend row-name -> (source file, class). Row names are the registry keys
# from `backends.all_backends()`, except that SDPABackend registers once as
# "sdpa" and emits rows as sdpa_math / sdpa_flash / sdpa_efficient /
# sdpa_cudnn -- one class, four rows, resolved by prefix in `_class_for`.
BACKEND_SOURCES: dict[str, tuple[str, str]] = {
    "naive":        ("attnbench/backends/impls.py", "NaiveAttention"),
    "sdpa":         ("attnbench/backends/impls.py", "SDPABackend"),
    "fa2":          ("attnbench/backends/impls.py", "FlashAttention2"),
    "flex":         ("attnbench/backends/impls.py", "FlexAttentionBackend"),
    "block_sparse": ("attnbench/backends/block_sparse.py", "BlockSparseAttention"),
    "gla":          ("attnbench/backends/linear.py", "GatedLinearAttention"),
    "sage":         ("attnbench/backends/sage_attention.py", "SageAttention"),
    "xformers":     ("attnbench/backends/xformers_backend.py", "XFormersAttention"),
}

# Methods that execute per timed call, or that build what the timed call
# consumes. `make_inputs` is included because a change to the inputs is a
# change to the measurement even when the kernel is untouched.
TIMED_REGION_METHODS = ("forward", "timed_call", "make_inputs")

# The measurement loop itself, shared by every backend.
SHARED_FUNCTIONS: tuple[tuple[str, str], ...] = (("attnbench/timing.py", "_do_bench"),)


class CodeIdentityError(RuntimeError):
    """A backend whose timed region differs across the segments being joined."""


@dataclass(frozen=True)
class BackendFingerprint:
    backend: str
    commit: str
    digest: str
    # Method name -> per-method digest, so a mismatch can name WHICH method
    # moved instead of only that something did.
    parts: dict[str, str] = field(default_factory=dict)


def _git_show(repo: Path, commit: str, path: str) -> Optional[str]:
    proc = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                          capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def _ast_digest(node: ast.AST) -> str:
    """Structural hash: whitespace, comments and docstrings excluded.

    Docstrings are stripped explicitly -- `ast.dump` keeps them as the first
    statement, so a docstring rewrite would otherwise read as a code change.
    """
    node = _strip_docstrings(node)
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()[:12]


def _strip_docstrings(node: ast.AST) -> ast.AST:
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Module)):
            body = getattr(child, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                child.body = body[1:] or [ast.Pass()]
    return node


def _class_for(backend: str) -> tuple[str, str]:
    if backend in BACKEND_SOURCES:
        return BACKEND_SOURCES[backend]
    if backend.startswith("sdpa"):
        return BACKEND_SOURCES["sdpa"]
    raise CodeIdentityError(
        f"no source mapping for backend {backend!r}. Add it to "
        f"BACKEND_SOURCES -- an unmapped backend must not silently pass a "
        f"code-identity check it was never subjected to."
    )


def fingerprint(repo: Path | str, commit: str, backend: str) -> BackendFingerprint:
    """AST fingerprint of one backend's timed region at one commit."""
    repo = Path(repo)
    path, class_name = _class_for(backend)
    src = _git_show(repo, commit, path)
    if src is None:
        raise CodeIdentityError(
            f"{path} does not exist at commit {commit[:12]} -- the segment's "
            f"provenance points at a tree this backend was not in.")

    parts: dict[str, str] = {}
    tree = ast.parse(src)
    found_class = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            found_class = True
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name in TIMED_REGION_METHODS:
                    parts[f"{class_name}.{f.name}"] = _ast_digest(f)
    if not found_class:
        raise CodeIdentityError(
            f"class {class_name} not found in {path} at commit {commit[:12]}")

    for shared_path, func_name in SHARED_FUNCTIONS:
        shared_src = _git_show(repo, commit, shared_path)
        if shared_src is None:
            raise CodeIdentityError(f"{shared_path} missing at {commit[:12]}")
        for node in ast.parse(shared_src).body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                parts[f"{shared_path}:{func_name}"] = _ast_digest(node)

    combined = hashlib.sha256(
        "|".join(f"{k}={parts[k]}" for k in sorted(parts)).encode()).hexdigest()[:12]
    return BackendFingerprint(backend=backend, commit=commit, digest=combined,
                              parts=parts)


@dataclass(frozen=True)
class BackendDrift:
    """One backend measured at commits whose timed regions differ."""

    backend: str
    commits: tuple[str, ...]
    methods: tuple[str, ...]      # which timed-region methods actually moved

    def describe(self) -> str:
        return (f"{self.backend}: {', '.join(c[:8] for c in self.commits)} "
                f"differ in {', '.join(self.methods)}")


def backends_with_drift(df: pd.DataFrame, *, repo: Path | str,
                        ) -> list[BackendDrift]:
    """Backends in `df` whose timed region is not identical at every commit
    the frame's rows were measured at.

    Returns a list rather than raising, because the answer is per backend and
    the useful action is to drop or separate the affected ones -- not to
    abandon the join.
    """
    if "git_commit" not in df.columns:
        raise CodeIdentityError(
            "no git_commit column; code identity cannot be established")

    drifts: list[BackendDrift] = []
    for backend, group in df.groupby("backend"):
        commits = sorted({str(c) for c in group["git_commit"].dropna().unique()
                          if str(c) not in ("None", "nan", "HEAD", "")})
        if len(commits) < 2:
            continue
        prints = [fingerprint(repo, c, str(backend)) for c in commits]
        if len({p.digest for p in prints}) == 1:
            continue
        moved = sorted({name for name in prints[0].parts
                        if len({p.parts.get(name) for p in prints}) > 1})
        drifts.append(BackendDrift(backend=str(backend), commits=tuple(commits),
                                   methods=tuple(moved)))
    return drifts


def check_code_identity(df: pd.DataFrame, *, repo: Path | str,
                        allow: Iterable[str] = ()) -> list[BackendDrift]:
    """Raise if any backend's timed region drifted, except those in `allow`.

    `allow` names backends whose drift is accepted deliberately, at the call
    site, one backend at a time -- the granularity `allow_mixed_commits` does
    not have. Returns the allowed drifts so a caller can report them.
    """
    allow = set(allow)
    drifts = backends_with_drift(df, repo=repo)
    blocking = [d for d in drifts if d.backend not in allow]
    if blocking:
        raise CodeIdentityError(
            "the timed region changed between the commits these segments were "
            "measured at:\n  " + "\n  ".join(d.describe() for d in blocking) +
            "\n\nA latency measured before and after a change to the code being "
            "timed is two answers to two questions. The flex create_block_mask "
            "hoist (2026-09-04) moved one cell by 10x without changing any "
            "result's shape or status. Drop these backends from the join, put "
            "them in separate outputs, or name them in allow=[...] with a "
            "reason."
        )
    return drifts
