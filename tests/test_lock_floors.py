"""The resolution and noise floors from banked repeats (Section 5.6), on
hand-worked fixtures."""

from __future__ import annotations

import pytest

from attnbench.lock import floors as F
from attnbench.lock.floors import (FloorsError, clean_series, inputs_sha256,
                                   noise_floor, resolution_floor)
from attnbench.lock.hdbi_fit import HdbiRow, fit_threshold
from attnbench.predict import scoring


def test_the_resolution_floor_is_a_quantile_of_pooled_deviations():
    # Series A: median 10, deviations 1, 0, 1. Series B: median 20, 0, 2, 0, 4
    # (median of 18, 20, 20, 24 is 20). Pool: 0, 0, 0, 1, 1, 2, 4.
    series = {("s1", "dense@16k"): [9.0, 10.0, 11.0],
              ("s1", "meanpool@16k"): [20.0, 18.0, 20.0, 24.0]}
    f = resolution_floor(series, quantile=1.0, pairing="P1")
    assert f.floor_ms == 4.0
    assert (f.n_series, f.n_repeats) == (2, 7)
    # numpy's linear quantile at 0.5 of 0,0,0,1,1,2,4 is the 4th value, 1.
    assert resolution_floor(series, quantile=0.5, pairing="P1").floor_ms == 1.0
    assert f.quantile == 1.0 and "pooled" in f.procedure


def test_the_noise_floor_is_a_quantile_of_between_session_spreads():
    sessions = {
        "s1": {"dense@16k": [10.0, 10.0, 10.0], "meanpool@16k": [20.0, 21.0, 22.0]},
        "s2": {"dense@16k": [11.0, 11.0], "meanpool@16k": [20.0]},
        "s3": {"dense@16k": [12.0], "only_here": [5.0]},
    }
    f = noise_floor(sessions, quantile=1.0, pairing="P1")
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
        resolution_floor(s, pairing="P1")
    for q in (0, 1.5, True, "0.9"):
        with pytest.raises(FloorsError, match="quantile"):
            resolution_floor(s, quantile=q, pairing="P1")


@pytest.mark.parametrize("series, match", [
    ({("s", "c"): [1.0]}, "two repeats"),
    ({("s", "c"): [1.0, -2.0]}, "positive"),
    ({("s", "c"): [1.0, float("inf")]}, "positive"),
    ({}, "no series"),
])
def test_series_the_resolution_floor_cannot_use_are_refused(series, match):
    with pytest.raises(FloorsError, match=match):
        resolution_floor(series, quantile=0.9, pairing="P1")


@pytest.mark.parametrize("sessions, match", [
    ({"s1": {"c": [1.0]}}, "two sessions"),
    ({"s1": {"c": [1.0]}, "s2": {"d": [1.0]}}, "two sessions"),
    ({"s1": {}, "s2": {"c": [1.0]}}, "no cells"),
    ({"s1": {"c": []}, "s2": {"c": [1.0]}}, "no repeats"),
])
def test_sessions_the_noise_floor_cannot_use_are_refused(sessions, match):
    with pytest.raises(FloorsError, match=match):
        noise_floor(sessions, quantile=0.9, pairing="P1")


def test_the_inputs_hash_follows_the_data_and_not_the_order():
    a = {("s1", "c"): [1.0, 2.0], ("s2", "c"): [3.0, 4.0]}
    b = {("s2", "c"): [3.0, 4.0], ("s1", "c"): [1.0, 2.0]}
    assert inputs_sha256(a) == inputs_sha256(b)
    assert inputs_sha256(a) != inputs_sha256({("s1", "c"): [1.0, 2.5], ("s2", "c"): [3.0, 4.0]})
    assert resolution_floor(a, quantile=0.9, pairing="P1").inputs_sha256 == inputs_sha256(a)


def test_clean_series_reads_a_ground_truth_result_and_refuses_a_traced_one():
    result = {"traced": False, "rows": [{"arm": "dense", "prefill_ms_samples": [1.0, 2.0]},
                                        {"arm": "meanpool", "prefill_ms_samples": [3.0]}]}
    assert clean_series(result) == {"dense": [1.0, 2.0], "meanpool": [3.0]}
    with pytest.raises(FloorsError, match="traced"):
        clean_series({**result, "traced": True})


# --- researcher's decisions, 28 Oct 2026 ---------------------------------------------

def test_the_floors_come_from_p1_only():
    series = {("s1", "c"): [1.0, 2.0]}
    sessions = {"s1": {"c": [1.0]}, "s2": {"c": [1.1]}}
    assert resolution_floor(series, quantile=0.9, pairing="P1").pairing == "P1"
    assert noise_floor(sessions, quantile=0.9, pairing="P1").pairing == "P1"
    for p in ("P2", "P7", "p1", None):
        with pytest.raises(FloorsError, match="P1 sessions only"):
            resolution_floor(series, quantile=0.9, pairing=p)
        with pytest.raises(FloorsError, match="P1 sessions only"):
            noise_floor(sessions, quantile=0.9, pairing=p)
    with pytest.raises(TypeError):                   # no default source
        resolution_floor(series, quantile=0.9)
    with pytest.raises(TypeError):
        noise_floor(sessions, quantile=0.9)


def test_the_shakedown_quantile_is_provisional_and_never_a_default():
    assert F.SHAKEDOWN_QUANTILE == 0.95
    with pytest.raises(TypeError):
        resolution_floor({("s", "c"): [1.0, 2.0]}, pairing="P1")
    with pytest.raises(TypeError):
        noise_floor({"s1": {"c": [1.0]}, "s2": {"c": [1.0]}}, pairing="P1")


def _floor_of(ms: float):
    # Two repeats at 100 - ms and 100 + ms: deviations ms and ms, so the
    # floor is ms at any quantile.
    return resolution_floor({("s1", "c"): [100.0 - ms, 100.0 + ms]}, quantile=1.0,
                            pairing="P1")


def test_the_resolution_floor_converts_to_seconds_in_one_place():
    f = _floor_of(4.0)
    assert f.floor_ms == pytest.approx(4.0)
    assert f.floor_s == pytest.approx(0.004)
    # scoring takes seconds: a 3 ms measured difference is inside a 4 ms
    # floor, a 5 ms one is not.
    assert scoring.sign_agreement(1.0, 2.0, 1.000, 1.003, floor=f.floor_s) is None
    assert scoring.sign_agreement(1.0, 2.0, 1.000, 1.005, floor=f.floor_s) is True
    assert scoring.decision_error(1.0, 2.0, 1.000, 1.003, admissible=True,
                                  floor=f.floor_s) == scoring.NOT_INTERPRETED
    assert scoring.decision_error(1.0, 2.0, 1.000, 1.005, admissible=True,
                                  floor=f.floor_s) == scoring.CORRECT
    # The HDBI fit reads the same seconds.
    rows = [HdbiRow("P1", "a", 0.2, +0.003), HdbiRow("P1", "b", 0.8, -0.005)]
    fit = fit_threshold(rows, training_pairings=["P1"], floor=f.floor_s)
    assert (fit.n_scored, fit.n_inside_floor) == (1, 1)


def test_no_lock_module_converts_milliseconds_itself():
    # The conversion lives in ResolutionFloor.floor_s alone.
    import pathlib
    lock_dir = pathlib.Path(F.__file__).parent
    for path in lock_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "floors.py":
            assert text.count("/ MS_PER_S") == 1
            continue
        for bad in ("/ 1000", "/1000", "* 1e-3", "*1e-3", "/ 1e3", "/1e3", "* 0.001"):
            assert bad not in text, (path.name, bad)
