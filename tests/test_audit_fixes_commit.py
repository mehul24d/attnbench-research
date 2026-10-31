"""Fixes for audit C2 (findings 1 to 3): the sync gap, the banked pairings
and scoring only what `load_committed` read. Written before the fixes; each
failed on the code it fixes.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from attnbench.predict import commit as C
from attnbench.predict.commit import (LatePredictionError, PredictionCommitError,
                                      load_committed, verify_bytes, write_prediction)
from attnbench.predict.residuals import residuals

from test_predict_commit import (CP, MODEL, T0, Clock, MemoryStore, committed,
                                 locked, p1_record, p2_record)  # noqa: F401 (fixture)

GAP = timedelta(seconds=C.REQUIRED_SYNC_GAP_S)


class RoundedStore(MemoryStore):
    """Like GsutilStore: the creation time printed in whole seconds and
    returned as its upper bound (+1 s)."""

    def created(self, remote):
        return super().created(remote).replace(microsecond=0) + timedelta(seconds=1)


# --- finding 1: one required gap, a planned wait derived from it ---------------------

def test_the_planned_wait_is_the_required_gap_plus_30_s():
    assert C.REQUIRED_SYNC_GAP_S == 60.0
    assert C.SYNC_WAIT_MARGIN_S == 30.0
    assert C.PLANNED_SYNC_WAIT_S == C.REQUIRED_SYNC_GAP_S + C.SYNC_WAIT_MARGIN_S


@pytest.mark.parametrize("upload_ms, readback_ms", [(400, 200), (999, 999), (1, 1)])
def test_a_session_waiting_the_planned_time_passes(tmp_path, upload_ms, readback_ms):
    # True time; both clocks agree. The objects are created `upload_ms` in,
    # sync_prediction returns `readback_ms` later, and the session waits the
    # planned time from there. The store's whole-second round-up must not
    # eat the margin.
    store = RoundedStore(Clock(T0))
    created = T0 + timedelta(milliseconds=upload_ms)
    _, receipt = committed(tmp_path, p1_record(), store, created)
    start = created + timedelta(milliseconds=readback_ms) \
        + timedelta(seconds=C.PLANNED_SYNC_WAIT_S)
    load_committed(store, receipt.remote, ground_truth_started_at=start,
                   clock_skew=GAP, banked_at={})


@pytest.mark.parametrize("short", [timedelta(milliseconds=1), timedelta(seconds=1),
                                   timedelta(seconds=30)])
def test_a_start_below_the_required_gap_is_refused(tmp_path, short):
    store = MemoryStore(Clock(T0))
    _, receipt = committed(tmp_path, p1_record(), store, T0)
    with pytest.raises(LatePredictionError):
        load_committed(store, receipt.remote, ground_truth_started_at=T0 + GAP - short,
                       clock_skew=GAP, banked_at={})


# --- finding 2: the training set is exactly the pairings banked first ------------------

def test_a_pairing_banked_before_the_write_but_left_out_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    with pytest.raises(PredictionCommitError, match="leaves P0 out"):
        load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                       clock_skew=timedelta(0), banked_at={"P0": T0 - timedelta(hours=1)})


def test_a_pairing_named_but_banked_after_the_write_is_still_refused(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p2_record(), store, T0 + timedelta(seconds=1))
    with pytest.raises(LatePredictionError, match="P1"):
        load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                       clock_skew=timedelta(0), banked_at={"P1": T0 + timedelta(hours=1)})


def test_the_named_and_the_banked_sets_agreeing_is_accepted(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p2_record(), store, T0 + timedelta(seconds=1))
    # P3 was banked after the write: neither named nor required.
    load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                   clock_skew=timedelta(0),
                   banked_at={"P1": T0 - timedelta(hours=1), "P3": T0 + timedelta(days=1)})


def test_a_pairing_banked_within_the_skew_of_the_write_is_refused(tmp_path):
    # Whether it was banked first cannot be told, so neither set can be checked.
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    with pytest.raises(PredictionCommitError, match="cannot be ordered"):
        load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=40),
                       clock_skew=timedelta(seconds=10),
                       banked_at={"P0": T0 - timedelta(seconds=5)})


def test_the_banking_times_must_be_stated(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    with pytest.raises(TypeError):
        load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                       clock_skew=timedelta(0))
    with pytest.raises(PredictionCommitError, match="banked_at"):
        load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                       clock_skew=timedelta(0), banked_at=None)


def test_this_pairings_own_banking_time_is_not_a_training_pairing(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                   clock_skew=timedelta(0), banked_at={"P1": T0 + timedelta(hours=2)})


def test_this_pairings_own_ground_truth_banked_before_its_prediction_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    with pytest.raises(LatePredictionError, match="own ground truth"):
        load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                       clock_skew=timedelta(0), banked_at={"P1": T0 - timedelta(hours=2)})


# --- finding 3: H1 and residual scoring take only a loaded prediction ------------------

MEASURED = {(MODEL, CP, "xattn", 0.5, 4, 1): 20.0, (MODEL, CP, "dense", 1.0, 4, 1): 10.0}


def _unsynced(tmp_path):
    f = write_prediction(p1_record(), tmp_path)
    return verify_bytes(f.path.read_bytes(), f.sidecar.read_text(encoding="ascii"),
                        f.path.name)


def test_residuals_refuse_a_prediction_whose_sync_was_never_checked(tmp_path):
    cp = _unsynced(tmp_path)
    assert cp.synced_at is None
    with pytest.raises(PredictionCommitError, match="load_committed"):
        residuals(cp, MEASURED, statistic=cp.statistic, dense_arms=["dense"],
                  relative_floor=0.05)


def test_require_loaded_refuses_none_and_other_types(tmp_path):
    with pytest.raises(PredictionCommitError, match="load_committed"):
        C.require_loaded(_unsynced(tmp_path))
    with pytest.raises(PredictionCommitError, match="CommittedPrediction"):
        C.require_loaded({"synced_at": T0})


def test_residuals_accept_a_loaded_prediction(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    cp = load_committed(store, r.remote, ground_truth_started_at=T0 + timedelta(seconds=2),
                        clock_skew=timedelta(0), banked_at={})
    assert C.require_loaded(cp) is cp
    rs = residuals(cp, MEASURED, statistic=cp.statistic, dense_arms=["dense"],
                   relative_floor=0.05)
    assert len(rs.residuals) > 0


# --- finding 1, the runners: every one reads the gap from commit.py --------------------

import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ["scripts/t61_session.py", "scripts/t64_session.py",
           "scripts/t227_shakedown.py", "scripts/t227_steps.py", "scripts/t228_steps.py"]


def _t61():
    sys.path.insert(0, str(ROOT / "scripts"))
    import t61_session
    return t61_session


def test_the_p2_runner_takes_the_gap_and_the_wait_from_commit():
    S = _t61()
    assert S.MIN_GAP_S == C.REQUIRED_SYNC_GAP_S
    assert S.CLOCK_SKEW_S == C.REQUIRED_SYNC_GAP_S
    assert S.PLANNED_SYNC_WAIT_S is C.PLANNED_SYNC_WAIT_S


def test_the_p4_runner_uses_the_p2_runners_constants():
    sys.path.insert(0, str(ROOT / "scripts"))
    import t64_session
    assert t64_session.S.MIN_GAP_S == C.REQUIRED_SYNC_GAP_S
    assert t64_session.S.PLANNED_SYNC_WAIT_S == C.PLANNED_SYNC_WAIT_S


@pytest.mark.parametrize("path", RUNNERS)
def test_no_runner_hardcodes_a_60_s_gap(path):
    src = (ROOT / path).read_text(encoding="utf-8")
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    hits = re.findall(r"\b(?:MIN_GAP_S|CLOCK_SKEW_S|HOLD_MARGIN_S|[A-Z_]*(?:GAP|SKEW|WAIT)[A-Z_]*)"
                      r"\s*=\s*[0-9.]+", code)
    assert hits == [], f"{path} sets a sync gap from a literal: {hits}"


def test_the_runners_hold_is_the_planned_wait(tmp_path):
    S = _t61()
    session = tmp_path / "s"
    session.mkdir()
    t = S.now(session, True)
    S.write_json(session / S.RECEIPT, {"synced_at": S.iso(t)})
    opened = S.hold_for_sync(session, True)
    assert opened - t >= C.PLANNED_SYNC_WAIT_S
