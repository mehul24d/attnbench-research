"""`score_source="dense_softmax_fp32"` must be enforced by the code that
claims it, not by torch happening to default `allow_tf32=False`.

Audit item S10. `model._scoring_forward_chunked` casts to fp32 and calls
`torch.matmul`, which on Ampere and later is TF32-eligible: torch may use a
10-bit mantissa. The flag that decides it is a global whose default moved
between torch 1.11 and 1.12, and `pyproject.toml` pins `torch>=2.6` with no
upper bound, so both behaviours are inside the supported range.

The banked runs were almost certainly true fp32 -- that is exactly the problem
these tests are about. A guarantee that holds because of a default is not a
guarantee, and nothing would have announced its loss.

CPU-only: these are plain globals and read/write identically without CUDA.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from attnbench import numerics
from attnbench.numerics import (FP32_SCORE_SOURCES, PrecisionNotEnforced,
                                assert_fp32_matmul, enforce_fp32_matmul,
                                fp32_matmul_state)

REPO = Path(__file__).resolve().parents[1]

# Every script that writes a result file computed from a matmul. Each must pin
# precision before it measures or scores -- a listed script that stops calling
# it fails here rather than silently inheriting whatever torch defaults to.
RESULT_PRODUCING_SCRIPTS = (
    "run_accuracy.py",
    "run_phase_timing.py",
    "run_sweep.py",
    "run_vectorised_endtoend.py",
    "check_score_dtype_asymmetry.py",
)


@pytest.fixture(autouse=True)
def _restore_globals():
    """These are process-wide. Restore them, or one test reconfigures the
    rest of the suite -- which is the shape of the bug being tested for."""
    before = fp32_matmul_state()
    yield
    torch.backends.cuda.matmul.allow_tf32 = before["allow_tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = before["allow_tf32_cudnn"]
    torch.set_float32_matmul_precision(before["float32_matmul_precision"])
    torch.backends.cudnn.benchmark = before["cudnn_benchmark"]


def test_enforce_sets_every_flag_it_claims_to():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    state = enforce_fp32_matmul()

    assert state == {
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
    }
    assert state == fp32_matmul_state(), "returned state must be the live state"


def test_enforce_is_idempotent():
    assert enforce_fp32_matmul() == enforce_fp32_matmul()


def test_the_guard_actually_fires_on_tf32():
    """A check nobody has watched fail is a guess. This puts torch into the
    state the audit found possible and requires the raise."""
    enforce_fp32_matmul()
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(PrecisionNotEnforced, match="allow_tf32"):
        assert_fp32_matmul("dense_softmax_fp32")


def test_the_guard_fires_on_a_lowered_matmul_precision():
    """The second route to the same loss: `set_float32_matmul_precision`
    downgrades fp32 matmuls without touching `allow_tf32`."""
    enforce_fp32_matmul()
    torch.set_float32_matmul_precision("medium")
    with pytest.raises(PrecisionNotEnforced, match="float32_matmul_precision"):
        assert_fp32_matmul("dense_softmax_fp32")


def test_the_guard_passes_once_enforced():
    torch.backends.cuda.matmul.allow_tf32 = True
    enforce_fp32_matmul()
    assert_fp32_matmul("dense_softmax_fp32")      # must not raise


def test_a_scorer_that_makes_no_fp32_claim_is_not_constrained():
    """`minference_meanpool` pools in fp32 too, but its name does not assert
    the precision. Widening the guard to every scorer would turn it into a
    global policy a deliberately-lower-precision estimator would have to
    fight, and this study's whole point is that cheap estimators are the
    interesting arm."""
    enforce_fp32_matmul()
    torch.backends.cuda.matmul.allow_tf32 = True
    assert_fp32_matmul("minference_meanpool")     # must not raise
    assert "minference_meanpool" not in FP32_SCORE_SOURCES


def test_the_scoring_path_asserts_before_it_computes():
    """The guard has to sit where the claim is applied. If it moved out of
    `compute_importance_scores`, a run could write rows stamped
    `dense_softmax_fp32` under TF32 and nothing would object."""
    src = (REPO / "attnbench" / "accuracy" / "model.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "compute_importance_scores")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "assert_fp32_matmul"]
    assert calls, ("compute_importance_scores no longer calls "
                   "numerics.assert_fp32_matmul -- the fp32 claim is "
                   "unenforced again")


@pytest.mark.parametrize("script", RESULT_PRODUCING_SCRIPTS)
def test_result_producing_scripts_pin_precision(script):
    """Asserted against the source rather than by running them, since they
    need a GPU. The failure this prevents is a new entry point that skips the
    call and inherits whatever the installed torch defaults to."""
    src = (REPO / "scripts" / script).read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "enforce_fp32_matmul"]
    assert calls, (
        f"scripts/{script} writes result rows but never calls "
        f"numerics.enforce_fp32_matmul(), so its fp32 matmuls run at "
        f"whatever precision the installed torch defaults to.")


def test_the_oracle_matmul_is_the_one_being_protected():
    """Anchors the guard to the code it exists for. If the scoring pass stops
    doing an fp32 matmul, this guard is either unnecessary or pointed at the
    wrong place, and either way someone should look."""
    src = (REPO / "attnbench" / "accuracy" / "model.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_scoring_forward_chunked")
    body = ast.unparse(fn)
    assert ".float()" in body and "torch.matmul" in body, (
        "_scoring_forward_chunked no longer casts to fp32 and matmuls; "
        "numerics.assert_fp32_matmul may now be guarding nothing")
