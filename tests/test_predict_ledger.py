"""The pairing-level prediction ledger (ruling 1c, 11 Nov 2026)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import t61_session as S  # noqa: E402
from attnbench.predict import ledger as LG  # noqa: E402

T0 = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)


def store(tmp_path, clock=None):
    clock = clock or [T0]
    return S.DirStore(tmp_path / "store", lambda: clock[0].timestamp()), clock


def record(pred=1.0, written="2026-11-17T03:00:00+00:00"):
    return {"written_at": written,
            "cells": [{"arm": "oracle", "context_length": 16370, "batch": 1, "pred_s": pred},
                      {"arm": "dense", "context_length": 8150, "batch": 1, "pred_s": 2.0}]}


def entry(rec, block=None):
    return {"session": "s", "block": block, "sha256": "f" * 64,
            "predictions_digest": LG.predictions_digest(rec), "remote_path": "p/x.json",
            "store_prefix": S.DirStore.PREFIX, "receipt": {}, "t31_receipt": {}}


def test_cells_are_arm_budget_and_batch():
    assert LG.record_cells(record()) == ["dense_L8192_b1", "oracle_L16384_b1"]
    assert LG.budget_of(16384) == 16384 and LG.budget_of(16385) == 32768
    with pytest.raises(LG.LedgerError):
        LG.budget_of(0)


def test_the_digest_ignores_only_the_write_time():
    assert LG.predictions_digest(record()) == LG.predictions_digest(record(written="later"))
    assert LG.predictions_digest(record()) != LG.predictions_digest(record(pred=1.0 + 1e-12))


def test_a_second_prediction_of_a_cell_is_refused(tmp_path):
    st, _ = store(tmp_path)
    led = LG.Ledger(st, S.DirStore.PREFIX, "P2")
    rec = record()
    cells = LG.record_cells(rec)
    assert led.check_prediction(cells, LG.predictions_digest(rec), block=None) is None
    led.record_prediction(cells, entry(rec))
    with pytest.raises(LG.LedgerError, match="one prediction"):
        led.check_prediction(cells, LG.predictions_digest(rec), block=None)
    with pytest.raises(LG.LedgerError, match="one prediction"):     # an overlapping set too
        led.check_prediction(cells[:1], LG.predictions_digest(rec), block=None)
    with pytest.raises(LG.LedgerError, match="first"):
        led.record_prediction(cells, {**entry(rec), "session": "other"})


def test_a_redone_block_re_predicts_bit_for_bit_or_stops(tmp_path):
    st, _ = store(tmp_path)
    led = LG.Ledger(st, S.DirStore.PREFIX, "P4")
    rec = record()
    cells = LG.record_cells(rec)
    led.record_prediction(cells, entry(rec, block="b2"))
    led.record_ground_truth_start(cells, {"session": "s", "block": "b2"})
    got = led.check_prediction(cells, LG.predictions_digest(record(written="later")), block="b2")
    assert got["block"] == "b2" and got["predictions_digest"] == LG.predictions_digest(rec)
    with pytest.raises(LG.LedgerError, match="stop"):
        led.check_prediction(cells, LG.predictions_digest(record(pred=1.01)), block="b2")
    with pytest.raises(LG.LedgerError, match="one prediction"):      # another block may not
        led.check_prediction(cells, LG.predictions_digest(rec), block="b3")


def test_no_prediction_once_ground_truth_of_the_cell_has_started(tmp_path):
    st, _ = store(tmp_path)
    led = LG.Ledger(st, S.DirStore.PREFIX, "P2")
    led.record_ground_truth_start(["oracle_L16384_b1"], {"session": "a"})
    rec = record()
    with pytest.raises(LG.LedgerError, match="has started"):
        led.check_prediction(LG.record_cells(rec), LG.predictions_digest(rec), block=None)
    other = LG.Ledger(st, S.DirStore.PREFIX, "P4")                    # per pairing
    assert other.check_prediction(LG.record_cells(rec), LG.predictions_digest(rec),
                                  block=None) is None


def test_the_earliest_start_is_the_first_entry_over_the_cells(tmp_path):
    st, clock = store(tmp_path)
    led = LG.Ledger(st, S.DirStore.PREFIX, "P4")
    assert led.earliest_start(["dense_L8192_b1"]) is None
    led.record_ground_truth_start(["dense_L8192_b1"], {"session": "b1"})
    clock[0] = T0 + timedelta(hours=1)
    led.record_ground_truth_start(["dense_L8192_b1", "oracle_L16384_b1"], {"session": "b2"})
    assert led.get("dense_L8192_b1", LG.GT_STARTED)["session"] == "b1"   # the first stays
    assert led.earliest_start(["oracle_L16384_b1"]) == T0 + timedelta(hours=1)
    assert led.earliest_start(["dense_L8192_b1", "oracle_L16384_b1"]) == T0


def test_a_pairing_is_banked_once_at_the_store_s_time(tmp_path):
    st, clock = store(tmp_path)
    led = LG.Ledger(st, S.DirStore.PREFIX, "P1")
    assert led.banked() is None
    with pytest.raises(LG.LedgerError, match="64 hex"):
        led.record_banked({"training_file_sha256": "TBD"})
    led.record_banked({"training_file_sha256": "a" * 64})
    clock[0] = T0 + timedelta(days=1)
    with pytest.raises(LG.LedgerError, match="already banked"):
        led.record_banked({"training_file_sha256": "b" * 64})
    when, got = led.banked()
    assert when == T0 and got == {"training_file_sha256": "a" * 64, "pairing": "P1"}
    assert LG.Ledger(st, S.DirStore.PREFIX, "P2").banked() is None


def test_bank_pairing_refuses_a_p1_file_that_is_not_the_lock_s(tmp_path):
    import hashlib
    import bank_pairing as BP
    st, _ = store(tmp_path)
    f = tmp_path / "p1_training.json.gz"
    f.write_bytes(b"rows")
    sha = hashlib.sha256(b"rows").hexdigest()
    with pytest.raises(LG.LedgerError, match="the lock's P1 training file"):
        BP.bank(st, S.DirStore.PREFIX, "P1", f, "banked/x.json.gz",
                {"p1_training": {"file_sha256": "0" * 64}})
    with pytest.raises(LG.LedgerError, match="the lock's P1 training file"):
        BP.bank(st, S.DirStore.PREFIX, "P1", f, "banked/x.json.gz", {})
    entry = BP.bank(st, S.DirStore.PREFIX, "P1", f, "banked/x.json.gz",
                    {"p1_training": {"file_sha256": sha}})
    assert entry["training_file_sha256"] == sha
    assert LG.Ledger(st, S.DirStore.PREFIX, "P1").banked()[1]["training_file_bucket_path"] \
        == "banked/x.json.gz"
