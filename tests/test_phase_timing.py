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
    assert r.implied_total_ms == pytest.approx(580.0 + 29.6 * 37.0)
    assert r.closes, r.render()


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
