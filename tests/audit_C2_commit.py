"""Audit C2: `predict/commit.py`, `scoring.py`, `intervals.py` and
`crossover.py` against plan Section 5.4 and the K9 late-sync rule
(Sections 5.1 and 5.6; T2.11, T2.19). Each test fails on main at c93d26d and
demonstrates one finding. The file is not named test_*.py, so the default
suite does not collect it; run it by name.

On main, `import attnbench` needs torch; this ran torch-free through a
scratch loader that skips the package __init__. None of it needs torch.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from attnbench.predict.commit import (LatePredictionError, PredictionCommitError,
                                      load_committed, verify_bytes, write_prediction)
from attnbench.predict.residuals import residuals

from test_predict_commit import (MODEL, CP, T0, Clock, MemoryStore, committed,
                                 p1_record)


# ---------------------------------------------------------------------------
# Finding 1 (high; plan ambiguity that changes a result). Section 5.1: "The
# prediction file and its sha256 are synced before the timed workload starts,
# and the workload starts at least 60 s after the sync; the analysis allows
# 60 s of clock skew between the store's clock and the instance's and refuses
# any prediction stamped later (Appendix A, 25 Oct 2026)."
# load_committed (commit.py:1052) refuses unless synced_at + skew < start, so
# the 60 s the session waits and the 60 s of skew the analysis allows cancel
# exactly: a session that waits the planned 60 s, with clocks that agree, is
# accepted only by the margin of its own read-back time, and is refused when
# that margin is consumed. GsutilStore.created (commit.py:742-755) returns the
# creation time rounded up by one second, which consumes it whenever the
# upload and read-back take under a second. A refused file loses every
# committed prediction of its pairing, after the workload has run.

PLAN_WAIT = timedelta(seconds=60)
PLAN_SKEW = timedelta(seconds=60)


class RoundedStore(MemoryStore):
    """Like GsutilStore: creation time printed in whole seconds, returned as
    the upper bound (+1 s)."""

    def created(self, remote):
        return super().created(remote).replace(microsecond=0) + timedelta(seconds=1)


def test_a_session_that_waits_the_planned_60_s_is_not_refused(tmp_path):
    # Synced at T0 + 0.4 s (true time, both clocks agree); sync_prediction
    # returns 0.2 s later, after the read-back; the session waits 60 s.
    store = RoundedStore(Clock(T0))
    sync_at = T0 + timedelta(milliseconds=400)
    _, receipt = committed(tmp_path, p1_record(), store, sync_at)
    start = sync_at + timedelta(milliseconds=200) + PLAN_WAIT
    load_committed(store, receipt.remote, ground_truth_started_at=start,
                   clock_skew=PLAN_SKEW)


def test_a_start_exactly_60_s_after_the_sync_is_not_refused(tmp_path):
    # "at least 60 s after the sync" includes 60 s.
    store = MemoryStore(Clock(T0))
    _, receipt = committed(tmp_path, p1_record(), store, T0)
    load_committed(store, receipt.remote, ground_truth_started_at=T0 + PLAN_WAIT,
                   clock_skew=PLAN_SKEW)


# ---------------------------------------------------------------------------
# Finding 2 (medium). Section 5.4, Fitting the baselines: "Each prediction for
# a pairing is trained only on pairings whose ground truth was banked before
# that prediction is written ... The training set is whatever has been banked
# when the prediction is committed, and each prediction file records it."
# load_committed (commit.py:1063-1074) checks that every pairing the file
# names was banked before the write, but never that every pairing banked
# before the write is named: `banked` is whatever the caller passed to
# build_record. A file that leaves a banked pairing out of its training set
# (and out of the nearest-pairing choice) is accepted, even when the analysis
# is told that pairing was banked first.

def test_a_pairing_banked_before_the_write_but_left_out_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    _, receipt = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    # P0's ground truth was banked an hour before P1's prediction was written,
    # yet the file says nothing was banked before it.
    with pytest.raises(PredictionCommitError):
        load_committed(store, receipt.remote,
                       ground_truth_started_at=T0 + timedelta(seconds=2),
                       clock_skew=timedelta(0),
                       banked_at={"P0": T0 - timedelta(hours=1)})


# ---------------------------------------------------------------------------
# Finding 3 (medium). Section 5.1: the analysis "refuses any prediction
# stamped later"; T2.11: "The analysis refuses a file synced at or after the
# timed workload started". The late check lives only in load_committed;
# verify_bytes returns the same CommittedPrediction type with synced_at=None
# (commit.py:1017), and the H1 scoring path, residuals (residuals.py:89),
# accepts it. So a prediction can be scored without the K9 check ever having
# run. (sensitivity.py says it takes either reader; that one is a
# recomputation, not a score.)

def test_scoring_refuses_a_prediction_whose_sync_was_never_checked(tmp_path):
    f = write_prediction(p1_record(), tmp_path)
    cp = verify_bytes(f.path.read_bytes(), f.sidecar.read_text(encoding="ascii"),
                      f.path.name)
    assert cp.synced_at is None
    measured = {(MODEL, CP, "xattn", 0.5, 4, 1): 20.0,
                (MODEL, CP, "dense", 1.0, 4, 1): 10.0}
    with pytest.raises(PredictionCommitError):
        residuals(cp, measured, statistic=cp.statistic, dense_arms=["dense"],
                  relative_floor=0.05)
