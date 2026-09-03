"""The compile-fallback guard. CPU-only -- the mechanism is device-independent.

The bug being guarded here was found in Stage 2 segment 1: Stage 1 certified
`flex` block-sparse 72/72 with real numerical agreement while Stage 2 could not
run a single one of those cells on the same machine at the same commit. The
cause was torch.compile's recompile-limit fallback silently running the
function eagerly, so the gate and the sweep exercised different code.

Test 2 is the one that matters. It drives a REAL dynamo fallback rather than
feeding the detector a hand-written string, because a detector validated only
against its own expected message tests the string, not the condition -- the
project's standing rule that a check nobody has watched fail is a guess.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch._dynamo as dynamo

from attnbench import compile_guard


@pytest.fixture(autouse=True)
def _clean_process_state():
    compile_guard.reset_process_state()
    yield
    compile_guard.reset_process_state()


FAIL_WIDTH = 64          # stands in for block_size=64: a shape that cannot lower


def _selective_backend(gm, example_inputs):
    """Compiles anything except FAIL_WIDTH, which it refuses -- the CPU analogue
    of inductor refusing to lower a 64-wide sparse block at head_dim=128."""
    if example_inputs[0].shape[-1] == FAIL_WIDTH:
        raise RuntimeError("LoweringException: cannot lower this shape")
    return gm.forward


def _fn(x):
    return x * 2 + 1


def test_configure_raises_the_recompile_ceiling():
    dynamo.config.recompile_limit = 8
    compile_guard.configure(limit=compile_guard.RECOMPILE_LIMIT)
    assert dynamo.config.recompile_limit == compile_guard.RECOMPILE_LIMIT
    assert compile_guard.RECOMPILE_LIMIT > 8


def test_the_fallback_is_real_and_the_guard_catches_it():
    """Identical call, opposite outcome, depending only on process history.

    This is the whole anomaly in eight lines: a fresh compile RAISES, and the
    same call after unrelated shapes have exhausted the recompile limit
    RETURNS a numerically correct answer from the eager path. Correctness is
    preserved, which is exactly why nothing looked wrong.
    """
    dynamo.reset()
    dynamo.config.recompile_limit = 8
    compiled = torch.compile(_fn, backend=_selective_backend, dynamic=False)
    x = torch.ones(FAIL_WIDTH)

    with pytest.raises(Exception):
        compiled(x)                       # fresh: compilation is attempted

    dynamo.reset()
    compiled = torch.compile(_fn, backend=_selective_backend, dynamic=False)
    with compile_guard.guard() as cg_clean:
        for w in (8, 16, 24, 32, 40, 48, 56):
            compiled(torch.ones(w))       # 7 successful compiles: under the limit
    assert not cg_clean.fell_back, "limit not yet reached; nothing to report"

    with compile_guard.guard() as cg:
        for w in (72, 80):                # crosses the limit
            compiled(torch.ones(w))
        out = compiled(x)                 # compilation SKIPPED -> eager

    assert torch.allclose(out, _fn(x)), (
        "the fallback returns a correct answer -- that is the trap, not a bug")
    assert cg.fell_back
    assert "recompile" in cg.detail or "cache_size" in cg.detail


def test_a_fallback_outside_any_guarded_region_is_still_recorded():
    """The gap `guard()` alone leaves. Stage 0's `probe()` is not wrapped, and
    torch emits its "called without torch.compile" warning through its own
    `_warn_once` set -- which `simplefilter("always")` cannot defeat. So if the
    FIRST crossing lands outside a guard, every later call is silent and the
    process flag would never be set. `configure()` attaches a permanent handler
    to the dynamo logger to close it."""
    compile_guard.configure()
    dynamo.reset()
    dynamo.config.recompile_limit = 4
    compiled = torch.compile(_fn, backend=_selective_backend, dynamic=False)

    for w in (8, 16, 24, 32, 40, 48):        # crossing, with nothing watching
        compiled(torch.ones(w))

    assert compile_guard.process_has_fallen_back()
    with compile_guard.guard() as cg:        # a later region inherits it
        pass
    assert cg.fell_back

    compile_guard.configure()                # restore the raised ceiling


def test_detection_is_sticky_across_regions():
    """The dynamo warning fires ONCE, at the crossing; every later call is
    silent. Per-region detection alone would clear the hundreds of contaminated
    cells that follow and flag only the one that happened to cross."""
    compile_guard._PROCESS_REASONS.append("recompile_limit hit (simulated)")
    with compile_guard.guard() as cg:
        pass                              # nothing happens in here at all
    assert cg.fell_back, "a later quiet region must still be refused"


def test_a_clean_region_reports_no_fallback():
    """Guards against the vacuous case: a detector that always fires is as
    useless as one that never does."""
    with compile_guard.guard() as cg:
        torch.ones(4) * 2
    assert not cg.fell_back
    assert not compile_guard.process_has_fallen_back()


def test_unrelated_warnings_are_re_emitted_not_swallowed():
    """The guard defeats the `once` dedup with simplefilter('always'), which
    also captures every other warning in the region. Those must be put back, or
    watching for one problem hides all the others from the session log."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with compile_guard.guard() as cg:
            warnings.warn("an unrelated deprecation", UserWarning)
    assert not cg.fell_back
    assert any("unrelated deprecation" in str(w.message) for w in seen)


def test_flex_eager_warning_is_treated_as_a_fallback():
    """torch's own per-call admission. It is the only signal that survives once
    the limit has already been crossed in an earlier process region."""
    with compile_guard.guard() as cg:
        warnings.warn("flex_attention called without torch.compile() - this "
                      "will be slow", UserWarning)
    assert cg.fell_back
    assert compile_guard.process_has_fallen_back()


def test_raise_if_fallen_back():
    rec = compile_guard.FallbackRecord(reasons=["recompile_limit (8)"])
    with pytest.raises(compile_guard.CompileFallbackError):
        compile_guard.raise_if_fallen_back(rec, context="cell x")
    compile_guard.raise_if_fallen_back(compile_guard.FallbackRecord())


def test_a_correctness_pass_earned_by_a_fallback_is_voided(monkeypatch):
    """Stage 1's verdict is what licenses Stage 2. A pass from code Stage 2
    will not run is worse than no pass: it reports the precondition satisfied
    while certifying a different implementation."""
    from attnbench import gates

    passing = gates.CorrectnessResult(
        backend="flex", config_key="deadbeef", passed=True,
        check_kind="masked_exact", max_abs_err=0.0157, detail="")

    def fake_dispatch(backend, cfg, **kw):
        warnings.warn("flex_attention called without torch.compile()",
                      UserWarning)
        return passing

    monkeypatch.setattr(gates, "_dispatch_for_family", fake_dispatch)
    out = gates.check_for_family(object(), object())

    assert out.passed is False, "a fallback must void the verdict"
    assert out.check_kind == "masked_exact", "what was attempted is still true"
    assert "VOID" in out.detail
    assert out.max_abs_err == pytest.approx(0.0157), (
        "the measured agreement is real and worth keeping -- it is the "
        "LICENCE that is withdrawn, not the number")


def test_a_clean_check_is_left_alone(monkeypatch):
    from attnbench import gates

    passing = gates.CorrectnessResult("fa2", "cafe", True, "exact",
                                      max_abs_err=1e-3)
    monkeypatch.setattr(gates, "_dispatch_for_family",
                        lambda backend, cfg, **kw: passing)
    out = gates.check_for_family(object(), object())
    assert out.passed is True and out.detail == ""
