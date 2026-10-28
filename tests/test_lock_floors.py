"""The resolution and noise floors from banked repeats (Section 5.6), on
hand-worked fixtures."""

from __future__ import annotations

import pytest

from attnbench.lock.floors import (FloorsError, clean_series, inputs_sha256,
                                   noise_floor, resolution_floor)


def test_the_resolution_floor_is_a_quantile_of_pooled_deviations():
    # Series A: median 10, deviations 1, 0, 1. Series B: median 20, 0, 2, 0, 4
    # (median of 18, 20, 20, 24 is 20). Pool: 0, 0, 0, 1, 1, 2, 4.
    series = {("s1", "dense@16k"): [9.0, 10.0, 11.0],
              ("s1", "meanpool@16k"): [20.0, 18.0, 20.0, 24.0]}
    f = resolution_floor(series, quantile=1.0)
    assert f.floor_ms == 4.0
    assert (f.n_series, f.n_repeats) == (2, 7)
    # numpy's linear quantile at 0.5 of 0,0,0,1,1,2,4 is the 4th value, 1.
    assert resolution_floor(series, quantile=0.5).floor_ms == 1.0
    assert f.quantile == 1.0 and "pooled" in f.procedure


def test_the_noise_floor_is_a_quantile_of_between_session_spreads():
    sessions = {
        "s1": {"dense@16k": [10.0, 10.0, 10.0], "meanpool@16k": [20.0, 21.0, 22.0]},
        "s2": {"dense@16k": [11.0, 11.0], "meanpool@16k": [20.0]},
        "s3": {"dense@16k": [12.0], "only_here": [5.0]},
    }
    f = noise_floor(sessions, quantile=1.0)
    # dense medians 10, 11, 12: (12 - 10) / 11. meanpool 21, 20: 1 / 20.5.
    assert f.between_session_spread == pytest.approx(
        {"dense@16k": 2 / 11, "meanpool@16k": 1 / 20.5})
    assert f.floor_relative == pytest.approx(2 / 11)
    # A cell timed in one session only is not a between-session spread.
    assert "only_here" not in f.between_session_spread
    assert (f.n_cells, f.n_sessions) == (2, 3)
    # The within-session spread is reported beside it.
    assert f.within_session_spread[("s1", "meanpool@16k")] == pytest.approx(2 / 21)


def test_the_quantile_is_required_and_bounded():
    s = {("s", "c"): [1.0, 2.0]}
    with pytest.raises(TypeError):
        resolution_floor(s)
    for q in (0, 1.5, True, "0.9"):
        with pytest.raises(FloorsError, match="quantile"):
            resolution_floor(s, quantile=q)


@pytest.mark.parametrize("series, match", [
    ({("s", "c"): [1.0]}, "two repeats"),
    ({("s", "c"): [1.0, -2.0]}, "positive"),
    ({("s", "c"): [1.0, float("inf")]}, "positive"),
    ({}, "no series"),
])
def test_series_the_resolution_floor_cannot_use_are_refused(series, match):
    with pytest.raises(FloorsError, match=match):
        resolution_floor(series, quantile=0.9)


@pytest.mark.parametrize("sessions, match", [
    ({"s1": {"c": [1.0]}}, "two sessions"),
    ({"s1": {"c": [1.0]}, "s2": {"d": [1.0]}}, "two sessions"),
    ({"s1": {}, "s2": {"c": [1.0]}}, "no cells"),
    ({"s1": {"c": []}, "s2": {"c": [1.0]}}, "no repeats"),
])
def test_sessions_the_noise_floor_cannot_use_are_refused(sessions, match):
    with pytest.raises(FloorsError, match=match):
        noise_floor(sessions, quantile=0.9)


def test_the_inputs_hash_follows_the_data_and_not_the_order():
    a = {("s1", "c"): [1.0, 2.0], ("s2", "c"): [3.0, 4.0]}
    b = {("s2", "c"): [3.0, 4.0], ("s1", "c"): [1.0, 2.0]}
    assert inputs_sha256(a) == inputs_sha256(b)
    assert inputs_sha256(a) != inputs_sha256({("s1", "c"): [1.0, 2.5], ("s2", "c"): [3.0, 4.0]})
    assert resolution_floor(a, quantile=0.9).inputs_sha256 == inputs_sha256(a)


def test_clean_series_reads_a_ground_truth_result_and_refuses_a_traced_one():
    result = {"traced": False, "rows": [{"arm": "dense", "prefill_ms_samples": [1.0, 2.0]},
                                        {"arm": "meanpool", "prefill_ms_samples": [3.0]}]}
    assert clean_series(result) == {"dense": [1.0, 2.0], "meanpool": [3.0]}
    with pytest.raises(FloorsError, match="traced"):
        clean_series({**result, "traced": True})
