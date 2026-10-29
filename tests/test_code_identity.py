"""Joining segments measured at different commits is a per-backend question.

`cross_arch` refuses mixed commits and offers `allow_mixed_commits=True` as
the only way through -- a switch that clears every backend at once on the
strength of a judgement made about one. The real Stage 2 data is exactly the
case that breaks: four segments at four commits, where the timed region moved
for `flex` and `naive` and for nothing else.

The flex change is not a hypothetical difference. Hoisting `create_block_mask`
out of the per-call path moved one cell from 4.22 to 40.72 useful TFLOPS
(see tests/test_timed_region_setup.py). Under a blanket override those two
numbers would have been averaged into one architecture's mean.

These tests build throwaway git repositories rather than reading the project's
own history, so they keep working after the real commits age out of relevance.
`git` is deliberately not shimmed by conftest.py, which is what makes this
possible.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from attnbench.analysis.code_identity import (
    BACKEND_SOURCES,
    CodeIdentityError,
    backends_with_drift,
    check_code_identity,
    fingerprint,
)

REPO = Path(__file__).resolve().parents[1]
IMPLS = "attnbench/backends/impls.py"
TIMING = "attnbench/timing.py"


def _impls(fa2_body: str, naive_body: str = "return q") -> str:
    return textwrap.dedent(f"""
        class NaiveAttention:
            def make_inputs(self, cfg):
                return 1, 2, 3
            def forward(self, q, k, v, cfg, mask=None):
                {naive_body}

        class FlashAttention2:
            def make_inputs(self, cfg):
                return 1, 2, 3
            def forward(self, q, k, v, cfg, mask=None):
                {fa2_body}
    """)


def _timing(bench_body: str = "return 1.0, 0.9, 1.1",
            measure_body: str = "return None") -> str:
    return textwrap.dedent(f"""
        def _do_bench(fn, warmup, reps):
            {bench_body}

        def measure(backend, cfg, mask=None):
            {measure_body}
    """)


def _repo(tmp_path: Path, revisions: list[dict[str, str]]) -> tuple[Path, list[str]]:
    """Make a git repo, one commit per revision dict {path: content}."""
    repo = tmp_path / "repo"
    (repo / "attnbench" / "backends").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a],
                                    check=True, capture_output=True, text=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    run("config", "user.email", "t@t"); run("config", "user.name", "t")

    shas = []
    for rev in revisions:
        for path, content in rev.items():
            (repo / path).write_text(content)
        run("add", "-A")
        run("commit", "-q", "-m", f"rev{len(shas)}")
        shas.append(run("rev-parse", "HEAD").stdout.strip())
    return repo, shas


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"backend": b, "git_commit": c} for b, c in rows])


# --- the case the real data presents ---------------------------------------

def test_a_changed_forward_is_drift_and_names_the_method(tmp_path):
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k"), TIMING: _timing()},
        {IMPLS: _impls("return q @ k"), TIMING: _timing()},   # fa2 kernel changed
    ])
    drifts = backends_with_drift(
        _frame([("fa2", shas[0]), ("fa2", shas[1])]), repo=repo)
    assert len(drifts) == 1
    assert drifts[0].backend == "fa2"
    assert drifts[0].methods == ("FlashAttention2.forward",)


def test_drift_is_scoped_to_the_backend_that_moved(tmp_path):
    """The whole point: one backend changing must not disqualify the others."""
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k", naive_body="return q"), TIMING: _timing()},
        {IMPLS: _impls("return q + k", naive_body="return q * 2"), TIMING: _timing()},
    ])
    df = _frame([("naive", shas[0]), ("naive", shas[1]),
                 ("fa2", shas[0]), ("fa2", shas[1])])
    drifts = backends_with_drift(df, repo=repo)
    assert [d.backend for d in drifts] == ["naive"]


def test_allow_is_per_backend_not_all_or_nothing(tmp_path):
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k", naive_body="return q"), TIMING: _timing()},
        {IMPLS: _impls("return q @ k", naive_body="return q * 2"), TIMING: _timing()},
    ])
    df = _frame([("naive", shas[0]), ("naive", shas[1]),
                 ("fa2", shas[0]), ("fa2", shas[1])])
    with pytest.raises(CodeIdentityError) as exc:
        check_code_identity(df, repo=repo)
    assert "fa2" in str(exc.value) and "naive" in str(exc.value)

    with pytest.raises(CodeIdentityError, match="fa2"):
        check_code_identity(df, repo=repo, allow=["naive"])

    allowed = check_code_identity(df, repo=repo, allow=["naive", "fa2"])
    assert {d.backend for d in allowed} == {"naive", "fa2"}


# --- what must NOT trip it -------------------------------------------------

def test_a_docstring_rewrite_is_not_a_different_experiment(tmp_path):
    """A guard that fires on prose gets its override pasted in reflexively,
    and then it is not guarding anything."""
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls('return q + k'), TIMING: _timing()},
        {IMPLS: _impls('"""explains itself now."""\n                return q + k'),
         TIMING: _timing()},
    ])
    assert backends_with_drift(_frame([("fa2", shas[0]), ("fa2", shas[1])]),
                               repo=repo) == []


def test_a_change_to_measure_but_not_do_bench_is_not_drift(tmp_path):
    """The real seg1->seg2 change: first-call recording and the compile
    fallback guard were added to `measure`, around an unchanged `_do_bench`.
    The steady-state latency is produced by the loop, not by its caller."""
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k"), TIMING: _timing(measure_body="return None")},
        {IMPLS: _impls("return q + k"),
         TIMING: _timing(measure_body="return (0, 1, 2)")},
    ])
    assert backends_with_drift(_frame([("fa2", shas[0]), ("fa2", shas[1])]),
                               repo=repo) == []


def test_a_change_to_do_bench_is_drift_for_every_backend(tmp_path):
    """The shared measurement loop. If it moves, nothing measured before it
    is comparable to anything measured after."""
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k"), TIMING: _timing()},
        {IMPLS: _impls("return q + k"), TIMING: _timing(bench_body="return 2.0, 1.9, 2.1")},
    ])
    df = _frame([("fa2", shas[0]), ("fa2", shas[1]),
                 ("naive", shas[0]), ("naive", shas[1])])
    assert {d.backend for d in backends_with_drift(df, repo=repo)} == {"fa2", "naive"}


def test_a_single_commit_frame_has_nothing_to_compare(tmp_path):
    repo, shas = _repo(tmp_path, [{IMPLS: _impls("return q + k"), TIMING: _timing()}])
    assert backends_with_drift(_frame([("fa2", shas[0])]), repo=repo) == []


def test_make_inputs_counts_as_the_timed_region(tmp_path):
    """A kernel timed on different inputs is a different measurement even
    when the kernel itself is untouched."""
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k"), TIMING: _timing()},
        {IMPLS: _impls("return q + k").replace("return 1, 2, 3", "return 4, 5, 6"),
         TIMING: _timing()},
    ])
    drifts = backends_with_drift(_frame([("fa2", shas[0]), ("fa2", shas[1])]), repo=repo)
    assert len(drifts) == 1
    assert "make_inputs" in drifts[0].methods[0]


# --- refusals --------------------------------------------------------------

def test_an_unmapped_backend_refuses_rather_than_passing(tmp_path):
    """The silent direction: a backend with no source mapping would otherwise
    be reported as clean, having never been checked."""
    repo, shas = _repo(tmp_path, [
        {IMPLS: _impls("return q + k"), TIMING: _timing()},
        {IMPLS: _impls("return q @ k"), TIMING: _timing()},
    ])
    with pytest.raises(CodeIdentityError, match="no source mapping"):
        backends_with_drift(_frame([("mystery", shas[0]), ("mystery", shas[1])]),
                            repo=repo)


def test_sdpa_variants_resolve_to_the_one_registered_class(tmp_path):
    """SDPABackend registers once and emits four row names."""
    repo, shas = _repo(tmp_path, [{IMPLS: _impls("return q + k"), TIMING: _timing()}])
    for name in ("sdpa_math", "sdpa_flash", "sdpa_efficient", "sdpa_cudnn"):
        with pytest.raises(CodeIdentityError, match="SDPABackend"):
            # The stub repo has no SDPABackend class -- the point is that the
            # name resolved to it rather than falling through as unmapped.
            fingerprint(repo, shas[0], name)


def test_a_frame_without_git_commit_refuses(tmp_path):
    repo, _ = _repo(tmp_path, [{IMPLS: _impls("return q"), TIMING: _timing()}])
    with pytest.raises(CodeIdentityError, match="git_commit"):
        backends_with_drift(pd.DataFrame([{"backend": "fa2"}]), repo=repo)


XA = "attnbench/backends/xattention.py"
XA_NOASSERT = "attnbench/backends/xattention_noassert.py"


def _xa(forward_body: str, estimate_body: str = "return q") -> str:
    return textwrap.dedent(f"""
        class XAttentionBackend:
            def estimate(self, q, k):
                {estimate_body}
            def forward(self, q, k, v, cfg, mask=None):
                {forward_body}
    """)


def _xa_noassert(estimate_body: str = "return k") -> str:
    return textwrap.dedent(f"""
        class XAttentionNoAssertBackend(XAttentionBackend):
            def estimate(self, q, k):
                {estimate_body}
    """)


def test_an_inherited_forward_is_part_of_the_subclass_fingerprint(tmp_path):
    repo, shas = _repo(tmp_path, [
        {XA: _xa("return q"), XA_NOASSERT: _xa_noassert(), TIMING: _timing()},
        {XA: _xa("return q * 2"), XA_NOASSERT: _xa_noassert(), TIMING: _timing()},
        {XA: _xa("return q * 2", estimate_body="return v"),
         XA_NOASSERT: _xa_noassert(), TIMING: _timing()},
    ])
    fp = fingerprint(repo, shas[0], "xattention_noassert")
    assert set(fp.parts) == {"XAttentionNoAssertBackend.estimate",
                             "XAttentionBackend.forward", f"{TIMING}:_do_bench"}
    drifts = backends_with_drift(
        _frame([("xattention_noassert", shas[0]), ("xattention_noassert", shas[1])]),
        repo=repo)
    assert [d.methods for d in drifts] == [("XAttentionBackend.forward",)]
    # The parent's estimate is overridden, so its change is not the subclass's.
    assert fingerprint(repo, shas[1], "xattention_noassert").digest == \
        fingerprint(repo, shas[2], "xattention_noassert").digest
    assert fingerprint(repo, shas[1], "xattention").digest != \
        fingerprint(repo, shas[2], "xattention").digest


def test_every_registered_backend_has_a_source_mapping():
    """Static coverage check: a backend added to the registry without a
    BACKEND_SOURCES entry would be unmapped, and unmapped only shows up when
    that backend happens to appear in a mixed-commit join."""
    import attnbench.backends as backends
    for name, cls in backends.all_backends().items():
        assert name in BACKEND_SOURCES, f"{name} has no code-identity mapping"
        path, class_name = BACKEND_SOURCES[name]
        assert cls.__name__ == class_name, f"{name} maps to the wrong class"
        assert (REPO / path).exists(), f"{name} maps to a missing file {path}"


# --- build constants (fa3, 28 Oct 2026) ------------------------------------------

FA3 = "attnbench/backends/fa3.py"


def _fa3(flags: str = '("A=TRUE", "B=TRUE")', dims: str = "(128,)",
         forward: str = "return q", with_flags: bool = True) -> str:
    head = (f"BUILD_FLAGS = {flags}\n" if with_flags else "") + f"BUILT_HEAD_DIMS = {dims}\n"
    return head + textwrap.dedent(f"""
        class FlashAttention3:
            def forward(self, q, k, v, cfg, mask=None):
                {forward}
    """)


def test_the_build_flags_are_part_of_the_fa3_fingerprint(tmp_path):
    """A wheel built with other flags is another kernel behind the same forward:
    the rows must not be joined as the same code."""
    repo, shas = _repo(tmp_path, [
        {FA3: _fa3(), TIMING: _timing()},
        {FA3: _fa3(flags='("A=TRUE",)'), TIMING: _timing()},        # flag dropped
        {FA3: _fa3(flags='("A=TRUE",)', dims="(64, 128)"), TIMING: _timing()},
        {FA3: "# a comment\n" + _fa3(flags='("A=TRUE",)', dims="(64, 128)"),
         TIMING: _timing()},
    ])
    fp = fingerprint(repo, shas[0], "fa3")
    assert set(fp.parts) == {"FlashAttention3.forward", f"{FA3}:BUILD_FLAGS",
                             f"{FA3}:BUILT_HEAD_DIMS", f"{TIMING}:_do_bench"}
    drifts = backends_with_drift(_frame([("fa3", s) for s in shas[:2]]), repo=repo)
    assert [d.methods for d in drifts] == [(f"{FA3}:BUILD_FLAGS",)]
    drifts = backends_with_drift(_frame([("fa3", s) for s in shas[1:3]]), repo=repo)
    assert [d.methods for d in drifts] == [(f"{FA3}:BUILT_HEAD_DIMS",)]
    assert fingerprint(repo, shas[2], "fa3").digest == fingerprint(repo, shas[3], "fa3").digest


def test_a_missing_build_constant_refuses(tmp_path):
    repo, shas = _repo(tmp_path, [{FA3: _fa3(with_flags=False), TIMING: _timing()}])
    with pytest.raises(CodeIdentityError, match="BUILD_FLAGS not found"):
        fingerprint(repo, shas[0], "fa3")


def test_the_real_fa3_build_constants_are_fingerprinted():
    from attnbench.analysis.code_identity import BUILD_CONSTANTS
    from attnbench.backends import fa3
    for path, name in BUILD_CONSTANTS["fa3"]:
        assert path == "attnbench/backends/fa3.py" and hasattr(fa3, name)
