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

    with pytest.raises(CrossArchError, match="no canary rows"):
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
