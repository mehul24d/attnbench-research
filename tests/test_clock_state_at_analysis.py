"""Unlocked clocks are checked where the harm lands, not where measured.

`provenance.py` has said since it was written that "an unlocked run must be
flagged in the results: unlocked clocks on a shared host produce variance that
is easy to mistake for a real effect." It was flagged. Nothing read it -- found
by the 2026-09-04 audit that followed instance 10, where `git_dirty` turned out
to have the same shape.

The decision, and why it is not a measurement-time gate: `nvidia-smi -lgc`
needs root and fails on most rental hosts, including the A100 this study rents
for its second architecture. Refusing to measure unlocked would delete that
architecture, and with it the hardware-conditional result the whole study
exists to produce -- a far larger loss than the variance. So the flag rides
onto every ratio and every drift line, and is loudest exactly where a reader
might otherwise over-claim.

The same principle as the token-sizing guard: put the check at the boundary
where a wrong number does damage, not at the point it is taken.

All CPU-only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.analysis import canary
from attnbench.analysis.cross_arch import (Speedup, compare_across_architectures,
                                           speedup_within_host)


def _rows(host, gpu, locked, *, base_ms=10.0, other_ms=5.0, seq_len=1024):
    common = dict(host=host, gpu_name=gpu, config_key=f"c{seq_len}",
                  seq_len=seq_len, ok=True, clocks_locked=locked)
    return [
        {**common, "backend": "sdpa_flash", "latency_ms_p50": base_ms},
        {**common, "backend": "fa2", "latency_ms_p50": other_ms},
    ]


def _df(*groups):
    return pd.DataFrame([r for g in groups for r in g])


# ---------------------------------------------------------------------------
# The flag reaches the ratio
# ---------------------------------------------------------------------------

def test_a_locked_pair_produces_a_locked_ratio():
    s = speedup_within_host(_df(_rows("h1", "L4", True)),
                            baseline_backend="sdpa_flash")
    assert [x.clocks_locked for x in s] == [True]


def test_an_unlocked_pair_produces_an_unlocked_ratio():
    s = speedup_within_host(_df(_rows("h1", "L4", False)),
                            baseline_backend="sdpa_flash")
    assert [x.clocks_locked for x in s] == [False]


def test_one_unlocked_side_makes_the_whole_ratio_unlocked():
    """A ratio is as noisy as its noisier half. 'Locked' cannot mean 'locked
    somewhere in the pair' -- the variance from the unlocked measurement is in
    the quotient either way."""
    rows = _rows("h1", "L4", True)
    rows[1]["clocks_locked"] = False          # the numerator only
    s = speedup_within_host(_df(rows), baseline_backend="sdpa_flash")
    assert s[0].clocks_locked is False


def test_a_missing_column_is_unknown_not_unlocked():
    """Rows predating the field must not be reported as unlocked. A flag that
    fires on absent data is ignored within a day, and then it is worth
    nothing when it fires on real data."""
    rows = _rows("h1", "L4", True)
    for r in rows:
        del r["clocks_locked"]
    s = speedup_within_host(_df(rows), baseline_backend="sdpa_flash")
    assert s[0].clocks_locked is None


def test_nan_is_unknown_not_locked():
    """`bool(nan)` is True, so the naive read reports a missing value as
    LOCKED -- the silent direction, quietly certifying a noisy comparison as
    clean."""
    rows = _rows("h1", "L4", True)
    rows[0]["clocks_locked"] = float("nan")
    s = speedup_within_host(_df(rows), baseline_backend="sdpa_flash")
    assert s[0].clocks_locked is None


# ---------------------------------------------------------------------------
# The flag survives into the cross-architecture claim
# ---------------------------------------------------------------------------

def _cmp(l4_locked, a100_locked, l4_ms=5.0, a100_ms=5.0):
    df = _df(_rows("h1", "L4", l4_locked, other_ms=l4_ms),
             _rows("h2", "A100", a100_locked, other_ms=a100_ms))
    return compare_across_architectures(
        speedup_within_host(df, baseline_backend="sdpa_flash"))[0]


def test_a_fully_locked_comparison_carries_no_caveat():
    assert _cmp(True, True).caveat() == ""
    assert _cmp(True, True).unlocked_architectures == []


def test_an_unlocked_architecture_is_named():
    c = _cmp(True, False)
    assert c.unlocked_architectures == ["A100"]
    assert "A100" in c.caveat()


def test_the_caveat_is_harshest_where_the_result_is_most_interesting():
    """A flip is this study's headline claim. A flip between 0.98 and 1.02
    produced by unlocked clocks is not a flip -- DRIFT_TOLERANCE is 5%, the
    same order as the effect being claimed."""
    # L4: baseline 10 / backend 5 = 2.0 (wins). A100: 10 / 20 = 0.5 (loses).
    c = _cmp(True, False, l4_ms=5.0, a100_ms=20.0)
    assert c.flips()
    assert "FLIPS" in c.caveat()
    assert "re-measure" in c.caveat()


def test_an_unlocked_host_taints_its_architectures_mean():
    """Ratios across hosts of one architecture are averaged; the lock state is
    REDUCED, not averaged. One unlocked host's variance is still in the mean."""
    df = _df(_rows("h1", "L4", True),
             _rows("h1b", "L4", False),
             _rows("h2", "A100", True))
    c = compare_across_architectures(
        speedup_within_host(df, baseline_backend="sdpa_flash"))[0]
    assert c.unlocked_architectures == ["L4"]


def test_the_comparison_is_still_produced():
    """The decision under test. Refusing here would delete the second
    architecture, since clock locking fails on most rental hosts -- the
    check flags, it does not veto."""
    c = _cmp(False, False)
    assert c.speedup_by_architecture, "an unlocked comparison was discarded"
    assert len(c.architectures) == 2


# ---------------------------------------------------------------------------
# The flag survives into the canary
# ---------------------------------------------------------------------------

def _canary_df(locked, other_ms):
    # Reference at 40 ms so the cells clear CANARY_MIN_LATENCY_MS: below the
    # floor they are excluded as unresolvable and the canary refuses outright,
    # which is a different behaviour than the one these tests are about.
    return _df(_rows("h1", "L4", locked, base_ms=40.0, other_ms=other_ms, seq_len=2048),
               _rows("h1", "L4", locked, base_ms=40.0, other_ms=other_ms, seq_len=4096))


def test_a_drift_on_unlocked_clocks_says_so():
    """DRIFT_TOLERANCE is 5%, the same order as unlocked-clock variance. A
    reader not told that will chase a phantom -- or learn to widen the
    tolerance, which makes the canary decorative."""
    drifts = canary.check_canary_drift(_canary_df(False, 5.0),
                                       _canary_df(False, 7.0))
    assert drifts
    assert all(d.clocks_locked is False for d in drifts)
    assert "CLOCKS UNLOCKED" in str(drifts[0])


def test_a_locked_drift_carries_no_such_note():
    drifts = canary.check_canary_drift(_canary_df(True, 5.0),
                                       _canary_df(True, 7.0))
    assert drifts
    assert "CLOCKS UNLOCKED" not in str(drifts[0])


def test_unlocked_in_either_session_taints_the_comparison():
    """Drift is a difference of two ratios and inherits the noise of both."""
    drifts = canary.check_canary_drift(_canary_df(True, 5.0),
                                       _canary_df(False, 7.0))
    assert drifts and drifts[0].clocks_locked is False


def test_the_canary_still_fires_on_unlocked_data():
    """Flagging must not become suppressing: an unlocked canary that stopped
    reporting drift would hide exactly what it exists to find."""
    drifts = canary.check_canary_drift(_canary_df(False, 5.0),
                                       _canary_df(False, 7.0))
    assert len(drifts) == 2


def test_clocks_locked_is_declared_gated():
    from attnbench.provenance import GATED_FIELDS, RECORDED_FIELDS
    assert "clocks_locked" in GATED_FIELDS
    assert "clocks_locked" not in RECORDED_FIELDS


def test_the_raw_clock_readings_stay_decorative():
    """Deliberate, and worth pinning: nothing can act on 1710 vs 1695 MHz, so
    gating on the readings would be a check with no decision behind it."""
    from attnbench.provenance import RECORDED_FIELDS
    assert {"sm_clock_mhz", "mem_clock_mhz", "persistence_mode"} <= RECORDED_FIELDS
