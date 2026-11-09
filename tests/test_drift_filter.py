"""The dense control's drift filter applied after the session
(`attnbench/analysis/drift_filter.py`; Appendix A, 9 Nov 2026)."""

from __future__ import annotations

import pytest

from attnbench.analysis import drift_filter as DF


def rnd(i, drift, *, reasons=()):
    return {"round": i, "dense_drift_ms": drift, "flag_reasons": list(reasons),
            "flagged": bool(reasons)}


def result():
    # warm-up 100 ms, absolute floor 5 ms: round 1 drifts -6 ms (flagged by the
    # floor, inside a 10% relative limit), round 2 +12 ms (flagged by both),
    # round 3 is throttled with no drift
    rounds = [rnd(0, -1.0), rnd(1, -6.0, reasons=["dense_drift_-6.000ms_vs_5ms_floor"]),
              rnd(2, 12.0, reasons=["dense_drift_+12.000ms_vs_5ms_floor"]),
              rnd(3, 0.5, reasons=["dense:throttled:hw_slowdown"])]
    rows = [{"arm": "dense", "reps": 4, "prefill_ms_samples": [99.0],
             "flagged_ms_samples": [94.0, 112.0, 100.5]},
            {"arm": "oracle", "reps": 4, "prefill_ms_samples": [50.0],
             "flagged_ms_samples": [51.0, 52.0, 53.0]}]
    return {"traced": False, "complete": False, "clean_rounds": 1, "rounds": rounds,
            "dense_control": {"reference_ms": 100.0}, "rows": rows}


def test_the_relative_filter_keeps_a_round_inside_the_limit_and_every_throttle_flag():
    out = DF.refilter(result(), mode=DF.RELATIVE, limit=0.10)
    assert [r["flagged"] for r in out["rounds"]] == [False, False, True, True]
    oracle = out["rows"][1]
    assert oracle["prefill_ms_samples"] == [50.0, 51.0]       # rounds 0 and 1, in order
    assert oracle["flagged_ms_samples"] == [52.0, 53.0]
    assert oracle["prefill_ms_median"] == pytest.approx(50.5)
    assert out["clean_rounds"] == 2 and out["complete"] is False
    assert DF.refilter(result(), mode=DF.RELATIVE, limit=0.10, min_clean=2)["complete"] is True


def test_no_filter_drops_only_the_throttled_round_and_recorded_changes_nothing():
    out = DF.refilter(result(), mode=DF.NONE)
    assert out["rows"][0]["prefill_ms_samples"] == [99.0, 94.0, 112.0]
    assert out["clean_rounds"] == 3
    rec = DF.refilter(result(), mode=DF.RECORDED)
    assert [r["flagged"] for r in rec["rounds"]] == [False, True, True, True]
    assert rec["rows"][1]["prefill_ms_samples"] == [50.0] and rec["clean_rounds"] == 1


def test_the_relative_limit_is_the_quantile_of_drift_over_warm_up_time():
    assert DF.relative_drifts(result()) == pytest.approx([0.01, 0.06, 0.12, 0.005])
    assert DF.relative_limit([result()], q=1.0) == pytest.approx(0.12)
    assert DF.relative_limit([result()], q=0.5) == pytest.approx(0.035)


def test_a_row_whose_samples_do_not_match_its_rounds_is_refused():
    bad = result()
    bad["rows"][1]["flagged_ms_samples"] = [51.0, 52.0]
    with pytest.raises(DF.DriftFilterError):
        DF.refilter(bad, mode=DF.NONE)
    with pytest.raises(DF.DriftFilterError):
        DF.refilter(result(), mode=DF.RELATIVE)
