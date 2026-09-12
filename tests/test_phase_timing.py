"""Stage 5: per-phase timing protocol and the reconciliation identity.

CPU-only. The GPU part is one orchestration function; everything that decides
what a number MEANS is here and runs against fake clocks.
"""

from __future__ import annotations

import math

import pytest

from attnbench.accuracy.phase_timing import (
    PER_CALL_PHASES, PHASES, RECONCILE_TOLERANCE, reconcile,
    scoring_overhead_ratio, summarize, time_repeated)


def test_warmup_samples_are_discarded_not_averaged_in():
    """The dense arm's first 50 rows at 8192 averaged 1911 ms against 1663
    for the rest. A warmup that is merely run and then INCLUDED would carry
    that first-touch effect into the reported mean.

    Modelled the way it actually happens: the work itself is expensive on its
    first few calls. Warmup reads no clock, so the cost has to live in `fn`.
    """
    state = {"t": 0.0, "calls": 0}

    def fn():
        state["calls"] += 1
        state["t"] += 1.0 if state["calls"] <= 3 else 0.010

    samples = time_repeated(fn, warmup=3, reps=5, clock=lambda: state["t"])

    assert state["calls"] == 8, "warmup must actually run"
    assert len(samples) == 5, "only the timed reps are reported"
    assert all(s == pytest.approx(10.0) for s in samples), samples
    # And the thing being defended against, shown rather than asserted in
    # prose: including the warmup reps would report a mean 38x higher.
    clean = sum(samples) / len(samples)
    contaminated = (3 * 1000.0 + 5 * 10.0) / 8
    assert contaminated / clean == pytest.approx(38.1, abs=0.1)


def test_reps_must_be_at_least_one():
    with pytest.raises(ValueError, match="reps must be"):
        time_repeated(lambda: None, reps=0)


def test_synchronize_is_called_inside_the_timed_region():
    """Not around the loop. A device that queues work would otherwise let
    every rep but the last report a kernel launch instead of a kernel."""
    order = []
    time_repeated(lambda: order.append("fn"), warmup=1, reps=2,
                  synchronize=lambda: order.append("sync"))
    # warmup fn, then one sync, then per rep: sync, fn, sync
    assert order == ["fn", "sync", "sync", "fn", "sync", "sync", "fn", "sync"]


def test_summarize_refuses_an_empty_sample():
    with pytest.raises(ValueError, match="nothing behind it"):
        summarize([], backend="x", sparsity=None, context_length=1,
                  phase="prefill", n_warmup=1, clocks_locked=True)


def test_summarize_carries_the_clock_lock_outcome():
    m = summarize([10.0, 12.0], backend="sdpa_flash", sparsity=None,
                  context_length=8192, phase="prefill", n_warmup=3,
                  clocks_locked=False)
    assert m.clocks_locked is False and m.n_reps == 2
    assert m.ms_mean == pytest.approx(11.0)


# --------------------------------------------------------------------------
# The identity this session exists to settle.
# --------------------------------------------------------------------------

def test_phases_that_compose_close():
    r = reconcile(prefill_ms=580.0, decode_step_ms=37.0, n_generated=29.6,
                  observed_total_ms=1677.0, context_length=8192,
                  backend="sdpa_flash")
    assert r.implied_total_ms == pytest.approx(580.0 + (29.6 - 1) * 37.0)
    assert r.closes, r.render()


def test_the_first_generated_token_comes_free_with_the_prefill():
    """The prefill forward emits the first token's logits, so n tokens cost
    one prefill plus n-1 decode steps -- not n.

    Found 2026-09-08 by the OLS fit's intercept cross-check (#27). The old
    `n * step` form overstated every implied total by exactly one step, and
    all 24 of the 2026-09-07 reconciliations were positive because of it --
    every one still inside the 10% tolerance."""
    a = reconcile(prefill_ms=600.0, decode_step_ms=40.0, n_generated=1,
                  observed_total_ms=600.0, context_length=8192, backend="x")
    # one generated token is the prefill and nothing else
    assert a.implied_total_ms == pytest.approx(600.0)
    b = reconcile(prefill_ms=600.0, decode_step_ms=40.0, n_generated=2,
                  observed_total_ms=640.0, context_length=8192, backend="x")
    assert b.implied_total_ms == pytest.approx(640.0)


def test_a_systematically_biased_identity_can_still_pass_a_loose_tolerance():
    """Why the reconciliation did not catch its own off-by-one, stated as a
    test so the lesson does not rest on a comment.

    One decode step (~35 ms) against an ~1150 ms total is 3% -- inside the
    10% tolerance, in the same direction, every time. `closes` is not
    evidence that an identity is right; it is evidence it is not badly
    wrong."""
    wrong = 600.0 + 14 * 40.0          # the retired n * step form
    right = 600.0 + (14 - 1) * 40.0
    r = reconcile(prefill_ms=600.0, decode_step_ms=40.0, n_generated=14,
                  observed_total_ms=right, context_length=8192, backend="x")
    assert r.implied_total_ms == pytest.approx(right)
    assert abs(wrong - right) / right < RECONCILE_TOLERANCE
    assert r.residual_frac == pytest.approx(0.0)


def test_the_open_discrepancy_is_reported_as_not_closing():
    """The 2026-09-07 numbers: 62.1 ms/token observed, a 55.5 ms decode step,
    and ~580 ms of implied prefill. They do not compose, and the harness must
    say so at the terminal rather than let it surface in analysis a day
    later."""
    r = reconcile(prefill_ms=580.0, decode_step_ms=55.5, n_generated=29.6,
                  observed_total_ms=1677.0, context_length=8192,
                  backend="sdpa_flash")
    assert not r.closes
    assert r.residual_frac > RECONCILE_TOLERANCE
    assert "DOES NOT CLOSE" in r.render()


def test_zero_decode_steps_is_refused():
    """An implied total with no decode steps is just the prefill, and its
    residual against an end-to-end number would be the entire decode phase
    reported as a discrepancy."""
    with pytest.raises(ValueError, match="zero decode steps"):
        reconcile(prefill_ms=1.0, decode_step_ms=1.0, n_generated=0,
                  observed_total_ms=100.0, context_length=2048, backend="x")


def test_mask_build_and_scoring_are_not_in_the_per_call_set():
    """Neither is paid per generated token -- mask build is per config, and
    scoring is excluded from every latency number in the study. Summing them
    into the identity would make it close for the wrong reason."""
    assert set(PER_CALL_PHASES) == {"prefill", "decode_step"}
    assert set(PER_CALL_PHASES) < set(PHASES)


# --------------------------------------------------------------------------
# The number that prices the study's biggest acknowledged caveat.
# --------------------------------------------------------------------------

def test_scoring_overhead_is_measured_against_what_sparsity_SAVED():
    """Not against the sparse prefill. The question is whether the estimator
    costs more than the thing it enables."""
    # dense 580, sparse 400 -> saved 180. Scoring 900 -> 5x what it saved.
    assert scoring_overhead_ratio(900.0, 580.0, 400.0) == pytest.approx(5.0)


def test_scoring_overhead_is_undefined_when_sparsity_saved_nothing():
    """Which is the 2026-09-07 situation: block_sparse prefill is not
    obviously cheaper than dense. NaN, not infinity -- there is no ratio to
    report, and a huge finite number would read as a measurement."""
    assert math.isnan(scoring_overhead_ratio(900.0, 400.0, 580.0))
    assert math.isnan(scoring_overhead_ratio(900.0, 400.0, 400.0))


# --------------------------------------------------------------------------
# The provenance stamp must not overwrite what the harness measured.
#
# 2026-09-07: Stage 5 ran with clocks locked, threaded clocks_locked=True onto
# every PhaseMeasurement, and then wrote clocks_locked=False to all 27 rows --
# `provenance.capture()` defaults the flag to False and the final merge loop
# assigned every stamp key over the frame. The field is GATED (cross_arch,
# canary read it), and the falsehood was plausible, so nothing downstream
# could have caught it.
#
# These drive the STAMPING path, not measure_band. The measurement was already
# correct; the bug lived entirely in the write, which no test reached.
# --------------------------------------------------------------------------

def test_stamp_fills_in_fields_the_rows_do_not_have():
    import pandas as pd
    from attnbench import provenance
    df = pd.DataFrame([{"phase": "prefill", "ms_mean": 1.0}])
    provenance.stamp_onto(df, {"git_commit": "abc123", "gpu_name": "NVIDIA L4"})
    assert df.git_commit.tolist() == ["abc123"]
    assert df.gpu_name.tolist() == ["NVIDIA L4"]


def test_stamp_does_not_overwrite_a_measured_field():
    """The exact 2026-09-07 loss: measured True, stamp default False."""
    import pandas as pd
    from attnbench import provenance
    df = pd.DataFrame([{"phase": "prefill", "clocks_locked": True}])
    with pytest.raises(ValueError, match="clocks_locked"):
        provenance.stamp_onto(df, {"clocks_locked": False})
    # and it is still True -- the raise happens before any assignment to it
    assert df.clocks_locked.tolist() == [True]


def test_stamp_is_silent_when_the_measured_value_agrees():
    """A caller that passes its measured value to capture() sees nothing.
    This is what makes raising safe at the end of a 19-minute GPU run: it
    cannot fire on a correct run."""
    import pandas as pd
    from attnbench import provenance
    df = pd.DataFrame([{"phase": "prefill", "clocks_locked": True}])
    provenance.stamp_onto(df, {"clocks_locked": True, "git_commit": "abc"})
    assert df.clocks_locked.tolist() == [True]
    assert df.git_commit.tolist() == ["abc"]


def test_capture_still_defaults_clocks_locked_to_False():
    """Pinning the default that made the overwrite plausible rather than
    loud. The default is not wrong -- most callers genuinely do not lock --
    but it means `capture()` ALWAYS has an opinion about a field some
    callers measure, which is why the merge has to refuse rather than
    trust the stamp."""
    from attnbench import provenance
    assert provenance.capture.__defaults__[0] is False


# --------------------------------------------------------------------------
# The decode step is fitted, not differenced (silent_failure_patterns #26).
# --------------------------------------------------------------------------

def test_a_two_point_fit_is_refused():
    """Two points fit a line exactly: R^2 is 1.0 whatever the data and the
    intercept cross-check cannot fail. A guard that cannot fail is the thing
    this project keeps finding, and a two-point 'fit' is that in numerical
    form."""
    from attnbench.accuracy.phase_timing import fit_decode_step
    with pytest.raises(ValueError, match="at least 3 points"):
        fit_decode_step({1: 700.0, 8: 940.0})


def test_the_fit_recovers_a_known_slope_and_intercept():
    from attnbench.accuracy.phase_timing import fit_decode_step
    f = fit_decode_step({k: 700.0 + 32.0 * k for k in (1, 2, 4, 8, 16)})
    assert f.slope_ms == pytest.approx(32.0)
    assert f.intercept_ms == pytest.approx(700.0)
    assert f.r_squared == pytest.approx(1.0)


def test_the_fitted_intercept_is_an_independent_estimate_of_prefill():
    """T(k) = prefill + k * step, so the intercept IS prefill. Checking it
    against the separately measured prefill is free and catches a linear
    model that does not hold -- which the two-point version could not do."""
    from attnbench.accuracy.phase_timing import fit_decode_step
    f = fit_decode_step({k: 700.0 + 32.0 * k for k in (1, 2, 4, 8, 16)})
    assert f.intercept_vs_prefill(700.0) == pytest.approx(0.0)
    assert f.intercept_vs_prefill(350.0) == pytest.approx(1.0)   # 100% off


def test_nonlinear_totals_show_up_as_a_poor_fit():
    """If per-token cost grows with position, there is no single decode step,
    and R^2 says so rather than the estimator returning a confident average
    of two regimes."""
    from attnbench.accuracy.phase_timing import fit_decode_step
    quadratic = {k: 700.0 + 32.0 * k + 2.0 * k * k for k in (1, 2, 4, 8, 16)}
    assert fit_decode_step(quadratic).r_squared < 0.995


def test_more_points_reduce_the_slopes_sensitivity_to_noise_in_one_point():
    """The reason for the change. Perturb the k=1 total -- the one carrying a
    full prefill -- and see how much of that lands in the slope."""
    from attnbench.accuracy.phase_timing import fit_decode_step

    def shift(ks, delta):
        base = {k: 700.0 + 32.0 * k for k in ks}
        a = fit_decode_step(base).slope_ms
        base[min(ks)] += delta
        return abs(fit_decode_step(base).slope_ms - a)

    two_ish = shift((1, 4, 8), 24.0)      # smallest allowed spread
    many = shift((1, 2, 4, 8, 16), 24.0)
    assert many < two_ish
    # 24 ms of prefill noise moved the old two-point estimate by 24/7 = 3.4
    assert many < 24.0 / 7


def test_the_fit_steps_default_spans_a_wide_range():
    """Sensitivity goes as 1/sqrt(sum((k - kbar)^2)), so the spread is what
    buys precision, not the count alone."""
    from attnbench.accuracy.phase_timing import DECODE_FIT_STEPS
    assert len(DECODE_FIT_STEPS) >= 3
    kbar = sum(DECODE_FIT_STEPS) / len(DECODE_FIT_STEPS)
    sxx = sum((k - kbar) ** 2 for k in DECODE_FIT_STEPS)
    old_sxx = 2 * (3.5 ** 2)              # the retired (1, 8)
    assert sxx > 4 * old_sxx


# --------------------------------------------------------------------------
# Same-sign residuals are bias by default (#27). The sign test is one line,
# so there is no reason for it not to be standing.
# --------------------------------------------------------------------------

def test_all_same_sign_residuals_are_flagged_as_bias():
    """The 2026-09-07 situation exactly: 24 of 24 positive, every cell inside
    the 10% tolerance, all reporting CLOSES."""
    from attnbench.accuracy.phase_timing import bias_warning, sign_test_p
    assert sign_test_p([1.0] * 24) < 1e-6
    w = bias_warning([50.0] * 24)
    assert w is not None and "OVERSTATES" in w and "24/24" in w


def test_balanced_residuals_are_not_flagged():
    from attnbench.accuracy.phase_timing import bias_warning, sign_test_p
    assert sign_test_p([1.0] * 12 + [-1.0] * 12) == pytest.approx(1.0)
    assert bias_warning([1.0] * 12 + [-1.0] * 12) is None


def test_the_understating_direction_is_named_too():
    from attnbench.accuracy.phase_timing import bias_warning
    assert "UNDERSTATES" in bias_warning([-50.0] * 24)


def test_zeros_are_dropped_not_split():
    """An exact zero is evidence for neither sign, so counting it as half of
    each would manufacture symmetry the data does not have."""
    from attnbench.accuracy.phase_timing import sign_test_p
    assert sign_test_p([1.0] * 24 + [0.0] * 100) == pytest.approx(sign_test_p([1.0] * 24))
    assert sign_test_p([0.0] * 10) == 1.0


def test_bias_is_detectable_even_when_every_cell_is_inside_tolerance():
    """The whole point. These residuals would each pass the per-cell check
    and the set is still a biased model."""
    from attnbench.accuracy.phase_timing import bias_warning
    small = [0.03] * 20                      # 3% each, tolerance is 10%
    assert all(abs(r) < RECONCILE_TOLERANCE for r in small)
    assert bias_warning(small) is not None


# --------------------------------------------------------------------------
# The sign test as a column, not a printed line
#
# `bias_warning` was stdout-only from 2026-09-07 to 2026-09-12, so it survived
# exactly as long as the terminal scrollback did. The one time it mattered --
# 24 of 24 same-sign residuals, an off-by-one in the identity, #27 -- the
# parquet a later reader opened said `closes=True` on every row with nothing
# in it to disagree.
#
# Two writers produce these files: `scripts/run_phase_timing.py` measures them
# and `scripts/repair_reconciliation_identity.py` rewrites them. The failure
# guarded here is one filename carrying two schemas depending on which touched
# it last.
# --------------------------------------------------------------------------

import subprocess                                                # noqa: E402
import sys                                                       # noqa: E402
from pathlib import Path                                         # noqa: E402

import pandas as pd                                              # noqa: E402

from attnbench.accuracy.phase_timing import (                    # noqa: E402
    BIAS_COLUMNS, add_bias_columns, bias_check, bias_warning)

REPO = Path(__file__).resolve().parents[1]


def _frame(residuals):
    return pd.DataFrame({
        "context_length": [2048] * len(residuals),
        "backend": ["sdpa_flash"] * len(residuals),
        "sparsity": [None] * len(residuals),
        "prefill_ms": [100.0] * len(residuals),
        "decode_step_ms": [10.0] * len(residuals),
        "n_generated": [14.0] * len(residuals),
        "implied_total_ms": [230.0] * len(residuals),
        "observed_total_ms": [230.0 - r for r in residuals],
        "residual_ms": list(residuals),
        "residual_frac": [r / 230.0 for r in residuals],
        "closes": [True] * len(residuals),
    })


def test_the_printed_line_and_the_column_cannot_disagree():
    """`bias_warning` is derived from `bias_check`, not a second computation
    of the same test. Two implementations of one rule is how this project got
    four copies of the generation identity."""
    for residuals in ([1.0] * 24, [1.0, -1.0] * 12, [0.0] * 5, [1.0] * 3):
        c = bias_check(residuals)
        assert (bias_warning(residuals) is not None) == c.biased


def test_a_biased_set_is_readable_from_the_parquet_alone():
    df = _frame([1.0] * 24)
    c = add_bias_columns(df)
    assert c.biased and c.direction == "over"
    assert set(BIAS_COLUMNS) <= set(df.columns)
    assert df.bias_detected.all()
    assert df.bias_n_positive.iloc[0] == 24 and df.bias_n_negative.iloc[0] == 0
    assert df.bias_sign_test_p.iloc[0] < 1e-6
    # Every row carries it: it is a property of the SET, stamped like
    # provenance, so no row can be read in isolation and look clean.
    assert df.bias_detected.nunique() == 1


def test_an_unbiased_set_says_so_rather_than_saying_nothing():
    df = _frame([1.0, -1.0] * 12)
    c = add_bias_columns(df)
    assert not c.biased and c.direction == "none"
    assert not df.bias_detected.any()
    assert set(BIAS_COLUMNS) <= set(df.columns)


def test_exact_zeros_are_dropped_not_split():
    """An exact zero is evidence for neither sign."""
    c = bias_check([0.0, 0.0, 1.0, 1.0, 1.0])
    assert c.n == 3 and c.n_positive == 3 and c.n_negative == 0


def test_bias_columns_refuse_a_frame_with_no_residuals():
    """Writing bias_detected=False onto a frame with nothing to test would
    assert something no measurement supports."""
    with pytest.raises(KeyError, match="residual_ms"):
        add_bias_columns(pd.DataFrame({"a": [1]}))


def test_neither_writer_recomputes_the_sign_test():
    """A second definition of one rule is how this project got four copies of
    the generation identity. Source-text only, and deliberately NOT used to
    assert that either writer *does* write the columns -- a substring proves
    the call is somewhere in the file, not that the write path reaches it.
    That is checked end to end below, on both branches."""
    for name in ("run_phase_timing.py", "repair_reconciliation_identity.py"):
        src = (REPO / "scripts" / name).read_text()
        assert "sign_test_p(" not in src, (
            f"{name} recomputes the sign test instead of calling the shared "
            f"writer")


def _run_repair(path):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" /
         "repair_reconciliation_identity.py"), str(path), "--apply"],
        capture_output=True, text=True, cwd=REPO)


@pytest.mark.parametrize("stale_identity", [False, True], ids=["annotate", "rewrite"])
def test_the_repair_adds_the_columns_on_both_of_its_branches(tmp_path,
                                                             stale_identity):
    """The repair has two paths and each writes the file.

    `annotate` -- arithmetic already correct, the file only gains columns.
    `rewrite`  -- the old `n * decode_step` identity, so every derived value
                  is recomputed.

    Both must emit the bias columns. Parametrised because a first version of
    this test exercised only the annotate branch: deleting `add_bias_columns`
    from the rewrite branch left it green, which is the vacuity this file
    exists to catch elsewhere.
    """
    df = _frame([1.0] * 12)
    if stale_identity:
        # Put the file back on the retired identity so the rewrite path runs.
        df["implied_total_ms"] = df.prefill_ms + df.n_generated * df.decode_step_ms
        df["residual_ms"] = df.implied_total_ms - df.observed_total_ms
        df["residual_frac"] = df.residual_ms / df.observed_total_ms
    dst = tmp_path / "reconciliation.parquet"
    df.to_parquet(dst, index=False)

    r1 = _run_repair(dst)
    assert r1.returncode == 0, r1.stderr
    assert ("rewritten" in r1.stdout)
    after = pd.read_parquet(dst)
    assert set(BIAS_COLUMNS) <= set(after.columns), (
        f"the {'rewrite' if stale_identity else 'annotate'} branch wrote the "
        f"file without the sign-test columns")
    assert "sign test:" in r1.stdout

    r2 = _run_repair(dst)
    assert r2.returncode == 0
    assert "stamped, sign-tested" in r2.stdout
    assert pd.read_parquet(dst).equals(after)          # idempotent
