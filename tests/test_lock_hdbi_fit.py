"""HDBI's one boundedness threshold, fitted by sign agreement, ties to the
lower value (Section 5.4; Appendix A, 25 Oct 2026). Hand-worked rows."""

from __future__ import annotations

import pytest

from attnbench.lock.hdbi_fit import (DEVICE_BOUND, HOST_BOUND, HdbiFitError, HdbiRow,
                                     boundedness, fit_threshold)
from attnbench.predict.hdbi import classify


def row(h, diff, cell=None, pairing="P1"):
    return HdbiRow(pairing=pairing, cell=cell or (h, diff), hdbi=h,
                   measured_difference=diff)


def test_a_clean_split_is_found_at_the_lowest_value_that_achieves_it():
    # Below 0.5 sparse is slower (+), from 0.5 up it is faster (-).
    rows = [row(0.2, +3), row(0.4, +1), row(0.5, -2), row(0.8, -5)]
    fit = fit_threshold(rows, training_pairings=["P1"], floor=0.5)
    # Candidates 0.2, 0.4, 0.5, 0.8, 1.0 give 2, 3, 4, 3, 2 agreements.
    assert fit.curve == ((0.2, 2), (0.4, 3), (0.5, 4), (0.8, 3), (1.0, 2))
    assert (fit.threshold, fit.agreements, fit.n_scored) == (0.5, 4, 4)
    assert fit.agreement_rate == 1.0


def test_ties_go_to_the_lower_value():
    # 0.3 (+) and 0.6 (-) with 0.45 (+) and 0.5 (-) swapped about the middle:
    # thresholds 0.45 and 0.6 both give 3 of 4.
    rows = [row(0.3, +1), row(0.45, -1), row(0.5, +1), row(0.6, -1)]
    fit = fit_threshold(rows, training_pairings=["P1"], floor=0.0)
    assert dict(fit.curve)[0.45] == dict(fit.curve)[0.6] == 3
    assert fit.threshold == 0.45


def test_cells_inside_the_floor_are_not_scored_and_are_counted():
    rows = [row(0.2, +3), row(0.3, 0.4), row(0.7, -0.4), row(0.8, -5)]
    fit = fit_threshold(rows, training_pairings=["P1"], floor=0.5)
    assert (fit.n_scored, fit.n_inside_floor) == (2, 2)
    # The floor is inclusive, as in scoring.sign_agreement.
    assert fit_threshold([row(0.2, 0.5), row(0.8, -1)], training_pairings=["P1"],
                         floor=0.5).n_inside_floor == 1


def test_all_host_bound_is_a_candidate():
    rows = [row(0.2, +1), row(0.9, +1)]
    fit = fit_threshold(rows, training_pairings=["P1"], floor=0.0)
    assert (fit.threshold, fit.agreements) == (1.0, 2)


def test_only_training_pairings_are_read():
    rows = [row(0.2, +1), row(0.9, -1, pairing="P2")]
    with pytest.raises(HdbiFitError, match="not a training pairing"):
        fit_threshold(rows, training_pairings=["P1"], floor=0.0)
    fit = fit_threshold(rows, training_pairings=["P2", "P1"], floor=0.0)
    assert fit.pairings == ("P1", "P2")


@pytest.mark.parametrize("rows, kw, match", [
    ([row(0.2, 1)], dict(training_pairings=[], floor=0.0), "training pairings"),
    ([row(1.2, 1)], dict(training_pairings=["P1"], floor=0.0), "HDBI"),
    ([row(0.2, float("nan"))], dict(training_pairings=["P1"], floor=0.0), "difference"),
    ([row(0.2, 1)], dict(training_pairings=["P1"], floor=-1.0), "floor"),
    ([row(0.2, 0.1)], dict(training_pairings=["P1"], floor=0.5), "outside the resolution"),
    ([row(0.2, 1, cell="c"), row(0.3, 1, cell="c")], dict(training_pairings=["P1"], floor=0.0),
     "twice"),
])
def test_rows_the_fit_cannot_use_are_refused(rows, kw, match):
    with pytest.raises(HdbiFitError, match=match):
        fit_threshold(rows, **kw)


def test_one_threshold_splits_as_classify_does_away_from_it():
    for v in (0.1, 0.49, 0.51, 0.9):
        mine = boundedness(v, 0.5)
        theirs = classify(v, host_bound_below=0.5, device_bound_above=0.5)
        assert (mine == HOST_BOUND) == (theirs == "host-bound")
        assert (mine == DEVICE_BOUND) == (theirs == "device-bound")
