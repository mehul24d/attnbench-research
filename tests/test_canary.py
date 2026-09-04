"""The cross-session canary: drift nobody introduced on purpose.

Synthetic frames only. The thing under test is whether a ratio that moved
gets reported and a ratio that did not stays quiet -- and, most importantly,
whether the check can pass vacuously.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.analysis.canary import (CANARY_SEQ_LENS, CanaryDrift,
                                       assert_no_canary_drift, canary_ratios,
                                       canary_rows, check_canary_drift)
from attnbench.analysis import canary
from attnbench.analysis.cross_arch import CrossArchError

L4 = "NVIDIA L4"
H100 = "NVIDIA H100 80GB HBM3"


def _frame(host, gpu_name, *, gla_latency, seq_len=1024, ref_latency=100.0):
    """One session on one machine: the reference backend plus one other."""
    return pd.DataFrame([
        dict(host=host, gpu_name=gpu_name, backend="sdpa_flash",
             config_key=f"c{seq_len}", seq_len=seq_len,
             latency_ms_p50=ref_latency, ok=True),
        dict(host=host, gpu_name=gpu_name, backend="gla",
             config_key=f"c{seq_len}", seq_len=seq_len,
             latency_ms_p50=gla_latency, ok=True),
    ])


def test_identical_sessions_show_no_drift():
    a = _frame("host-1", L4, gla_latency=50.0)
    b = _frame("host-2", L4, gla_latency=50.0)
    assert check_canary_drift(a, b) == []
    assert_no_canary_drift(a, b)


def test_a_moved_ratio_is_reported():
    a = _frame("host-1", L4, gla_latency=50.0)      # ratio 2.00
    b = _frame("host-2", L4, gla_latency=40.0)      # ratio 2.50, +25%
    drifts = check_canary_drift(a, b)
    assert len(drifts) == 1
    assert drifts[0].backend == "gla"
    assert drifts[0].fractional_change == pytest.approx(0.25)

    with pytest.raises(CrossArchError, match="canary ratio"):
        assert_no_canary_drift(a, b)


def test_drift_within_tolerance_is_quiet():
    a = _frame("host-1", L4, gla_latency=50.0)      # 2.00
    b = _frame("host-2", L4, gla_latency=49.0)      # 2.041, ~2%
    assert check_canary_drift(a, b) == []


def test_different_machines_of_the_same_architecture_do_not_trip_it():
    """The point of using ratios: a machine that is uniformly 3x faster has
    the SAME ratios, so re-renting different hardware is not drift.

    Compared as raw latencies these two sessions differ by 200%.
    """
    a = _frame("host-1", L4, gla_latency=50.0, ref_latency=100.0)
    b = _frame("host-2", L4, gla_latency=150.0, ref_latency=300.0)
    assert check_canary_drift(a, b) == [], (
        "a uniformly slower machine changed no ratio and must not report drift")


def test_architectures_are_compared_separately():
    """An L4 and an H100 legitimately have different ratios -- that is the
    study's thesis, not a fault. Keys include gpu_name, so they never meet."""
    a = _frame("host-1", L4, gla_latency=50.0)
    b = _frame("host-2", H100, gla_latency=10.0)
    assert check_canary_drift(a, b) == [], (
        "different architectures share no key and cannot be compared as drift")


def test_a_key_present_in_only_one_session_is_not_drift():
    a = _frame("host-1", L4, gla_latency=50.0, seq_len=1024)
    b = _frame("host-2", L4, gla_latency=50.0, seq_len=4096)
    assert check_canary_drift(a, b) == []


def test_an_empty_canary_raises_instead_of_passing():
    """The failure this project keeps hitting: a check that reports success
    without having run. A frame with no canary lengths must not be silently
    treated as 'no drift found'."""
    off_grid = _frame("host-1", L4, gla_latency=50.0, seq_len=32768)
    assert off_grid[off_grid["seq_len"].isin(CANARY_SEQ_LENS)].empty

    with pytest.raises(CrossArchError, match="no usable canary rows"):
        canary_ratios(off_grid)


def test_missing_seq_len_column_raises():
    df = _frame("host-1", L4, gla_latency=50.0).drop(columns=["seq_len"])
    with pytest.raises(CrossArchError, match="seq_len"):
        canary_rows(df)


def test_ratios_are_never_formed_across_hosts():
    """Inherited from speedup_within_host, asserted here because the canary
    is the place someone would be tempted to relax it."""
    ref_only = pd.DataFrame([
        dict(host="host-1", gpu_name=L4, backend="sdpa_flash", config_key="c1024",
             seq_len=1024, latency_ms_p50=100.0, ok=True),
    ])
    other_only = pd.DataFrame([
        dict(host="host-2", gpu_name=L4, backend="gla", config_key="c1024",
             seq_len=1024, latency_ms_p50=50.0, ok=True),
    ])
    combined = pd.concat([ref_only, other_only], ignore_index=True)
    with pytest.raises(CrossArchError, match="was not measured on"):
        canary_ratios(combined)


def test_drift_string_names_what_moved():
    d = CanaryDrift(gpu_name=L4, backend="gla", config_key="c1024",
                    reference_ratio=2.0, observed_ratio=2.5)
    text = str(d)
    assert "gla" in text and "c1024" in text and "25.0%" in text


# --- deliberate baseline changes -------------------------------------------
#
# Added 2026-09-04. Hoisting mask construction out of flex's timed region
# moved seq_len=1024/batch=1 from 4.22 to 40.72 useful TFLOPS, so segment 2's
# canary would fire on flex against segment 1 -- correctly detecting a change,
# but one we made rather than environmental drift. A false alarm here is not
# harmless: it teaches a reader to ignore the check.

def _canary_frame(backend: str, latency: float, gpu="NVIDIA L4"):
    import pandas as pd
    rows = []
    for seq_len in canary.CANARY_SEQ_LENS:
        # The reference must clear CANARY_MIN_LATENCY_MS or the cell is not
        # resolvable at the tolerance and canary_rows excludes it. 1.0 ms was
        # fine when no floor existed; it now means "unmeasurable", which is
        # a different fixture than this test intends.
        for be, lat in (("sdpa_flash", 40.0), (backend, latency)):
            rows.append(dict(gpu_name=gpu, host="h1", backend=be,
                             config_key=f"cfg{seq_len}", seq_len=seq_len,
                             latency_ms_p50=lat, ok=True))
    return pd.DataFrame(rows)


def test_a_recorded_baseline_change_can_be_excluded():
    ref = _canary_frame("flex", 400.0)
    obs = _canary_frame("flex", 40.0)        # a 10x shift, as measured
    with pytest.raises(CrossArchError):
        canary.assert_no_canary_drift(ref, obs)
    canary.assert_no_canary_drift(ref, obs, rebased_backends=frozenset({"flex"}))


def test_excluding_a_backend_with_no_recorded_reason_is_refused():
    """The escape hatch must not be usable as 'silence whatever is firing'."""
    ref, obs = _canary_frame("gla", 10.0), _canary_frame("gla", 1.0)
    with pytest.raises(CrossArchError, match="no BaselineChange entry"):
        canary.assert_no_canary_drift(ref, obs,
                                      rebased_backends=frozenset({"gla"}))


def test_excluding_one_backend_does_not_silence_the_others():
    """The failure mode that would make this mechanism worse than useless."""
    import pandas as pd
    ref = pd.concat([_canary_frame("flex", 10.0), _canary_frame("gla", 4.0)])
    obs = pd.concat([_canary_frame("flex", 1.0), _canary_frame("gla", 1.0)])
    drifts = canary.check_canary_drift(ref, obs,
                                       rebased_backends=frozenset({"flex"}))
    assert {d.backend for d in drifts} == {"gla"}


def test_every_recorded_change_names_a_commit_and_a_reason():
    assert canary.BASELINE_CHANGES, "the registry documents real changes"
    for c in canary.BASELINE_CHANGES:
        assert len(c.commit) >= 7 and c.reason.strip() and c.date


# ---------------------------------------------------------------------------
# Resolution floor: a ratio is only as precise as its denominator.
# ---------------------------------------------------------------------------
#
# 2026-09-04. The canary fired on 12 ratios against segment 1 and none were
# environmental. Every one sat at a config where sdpa_flash -- the canary's
# own reference, and therefore the denominator of every ratio -- runs in about
# 2 ms and varies 4% run to run (max 17.8%). The slow backends at those same
# configs moved 0.2-1.5%, which is what a stable environment looks like.
#
# The fix is NOT a wider tolerance. It is refusing to build a ratio on a
# measurement that cannot resolve the tolerance in the first place.

def _res_frame(ref_latency, other_latency, backend="gla"):
    import pandas as pd
    rows = []
    for seq_len in canary.CANARY_SEQ_LENS:
        for be, lat in (("sdpa_flash", ref_latency), (backend, other_latency)):
            rows.append(dict(gpu_name="NVIDIA L4", host="h1", backend=be,
                             config_key=f"cfg{seq_len}", seq_len=seq_len,
                             latency_ms_p50=lat, ok=True))
    return pd.DataFrame(rows)


def test_a_cell_whose_reference_is_too_fast_to_measure_is_excluded():
    """2 ms against a 5% tolerance is 0.1 ms of resolution, which is below
    this instrument's run-to-run spread."""
    rows = canary.canary_rows(_res_frame(2.0, 50.0))
    assert rows.empty


def test_a_cell_with_a_slow_enough_reference_is_kept():
    rows = canary.canary_rows(_res_frame(40.0, 50.0))
    assert not rows.empty


def test_the_floor_looks_at_the_reference_not_the_backend():
    """The denominator is what limits the ratio. A numerator measured to 0.2%
    against a denominator measured to 18% still yields a ratio good to 18%."""
    # slow backend, fast reference -> still excluded
    assert canary.canary_rows(_res_frame(2.0, 500.0)).empty
    # fast backend, slow reference -> kept
    assert not canary.canary_rows(_res_frame(40.0, 0.5)).empty


def test_noise_at_an_unresolvable_cell_does_not_fire_as_drift():
    """The 2026-09-04 false alarm, reconstructed: a 20% swing in a 2 ms
    reference with everything else unchanged.

    It REFUSES rather than returning "no drift", and that is the stronger
    answer: "this cannot be checked" and "this was checked and is fine" are
    different claims, and reporting the second when the first is true is how
    a canary becomes decorative. The refusal names the floor and the observed
    median, so the reader can act on it.
    """
    ref = _res_frame(2.5, 50.0)
    obs = _res_frame(2.0, 50.0)          # reference alone moved 20%
    with pytest.raises(CrossArchError, match="no usable canary rows"):
        canary.check_canary_drift(ref, obs)


def test_real_drift_at_a_resolvable_cell_still_fires():
    """The floor must not become a way of not looking. Same 20% swing, at a
    config the instrument can actually measure."""
    ref = _res_frame(50.0, 50.0)
    obs = _res_frame(40.0, 50.0)
    assert canary.check_canary_drift(ref, obs)


def test_the_floor_is_justified_by_measurement_not_convenience():
    assert canary.CANARY_MIN_LATENCY_MS >= 10.0, (
        "lowering this re-admits cells whose run-to-run spread exceeds the "
        "drift tolerance; the numbers are in the constant's comment")
    assert canary.DRIFT_TOLERANCE == 0.05, (
        "the floor exists so this does NOT have to move")


def test_the_resolution_report_states_what_was_excluded():
    """A canary that quietly shrinks to two cells and reports 'no drift' is
    the exact failure this module exists to prevent."""
    msg = canary.canary_resolution_report(_res_frame(2.0, 50.0))
    assert "0 of" in msg and "cannot resolve" in msg
