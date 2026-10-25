"""T2.11: the prediction commit, and the analysis's refusal of late or
altered predictions (K9).

Hand-computed predictions are worked in the comments from Section 5.4. The
sidecar is checked with the system's own `shasum -a 256 -c` (or
`sha256sum -c`), which shares no code with hashlib. The store is an
in-memory fake with a settable clock; `GsutilStore` is tested against a fake
`gsutil` put first on PATH (tests/conftest.py blocks the real one).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from attnbench.predict import commit as C
from attnbench.predict import learned as L
from attnbench.predict.baselines import fit_fitted_sum
from attnbench.predict.commit import (
    COMPUTED, NOT_FITTED_REASON, CodeStamp, GsutilStore, LatePredictionError,
    PredictionCommitError, build_record, canonical_bytes, ground_truth_start,
    load_committed, sync_prediction, verify_bytes, write_prediction)
from attnbench.predict.learned import LearnedConfig, fit_learned
from attnbench.predict.recursion import (
    CostModelError, close_prefill, copy_op, kernel_op, price, recurse, sync_op)
from attnbench.predict.training import (
    Cell, GroundTruthRow, HardwareFeatures, committed_training_set,
    retrospective_training_set)

from test_predict_learned import TOY, MeanModel
from test_predict_recursion import COSTS

UTC = timezone.utc
T0 = datetime(2026, 11, 2, 3, 0, 0, tzinfo=UTC)
CLEAN = CodeStamp(git_commit="a" * 40, git_dirty=False)
HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)
CONFIG = LearnedConfig(target="seconds")
NO_SKEW = timedelta(0)

# K(h1,d10) S(item, 4 B) K(h1,d1), all layer 0, then the closing synchronise.
# Recursion (Q = 8 never binds): H = 1, 13, 14; D = 11, 13, 15 (as in
# test_predict_recursion); closing sync: max(14, 15) = 15, + s_fixed 1 = 16.
# Sum: h 2 + d 11 = 13.  Sum + s_0: 13 + 2 + 1 = 16.
# Step form: one step max(2, 11) = 11, tail 2 + 1 = 3: 14.
OPS = close_prefill([kernel_op(1, 10, layer=0),
                     sync_op("item", payload_bytes=4, layer=0),
                     kernel_op(1, 1, layer=0)])
# A second arm: K(h3,d1) K(h3,d1), closing sync.
# Recursion: H 3, 6; D 4, 7; closing: max(6, 7) = 7, + 1 = 8.
# Sum 6 + 2 = 8; Sum + s_0 9; step form max(6, 2) + 1 = 7.
OPS_HOST = close_prefill([kernel_op(3, 1, layer=0), kernel_op(3, 1, layer=0)])


def cell(pairing="P1", ops=OPS, arm="xattn", L=4, density=0.5, batch=1):
    return Cell(pairing=pairing, arm=arm, context_length=L, density=density,
                batch=batch, priced=price(ops, COSTS), hardware=HW, model=TOY)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class MemoryStore:
    def __init__(self, clock, no_clobber=False):
        self.objs, self.times, self.clock = {}, {}, clock
        self.no_clobber = no_clobber

    def put(self, local, remote):
        if self.no_clobber and remote in self.objs:
            return
        self.objs[remote] = Path(local).read_bytes()
        self.times[remote] = self.clock()

    def get(self, remote):
        return self.objs[remote]

    def created(self, remote):
        return self.times[remote]


def p1_record(**kw):
    args = dict(pairing="P1", cells=[cell(), cell(ops=OPS_HOST, arm="dense",
                                                  density=1.0)],
                queue_depth=8, banked=[], fitted_sum=None, learned=None,
                code=CLEAN, written_at=T0)
    args.update(kw)
    return build_record(**args)


def p1_rows():
    # P1 ground truth: 20 s for xattn, 10 s for dense.
    return [GroundTruthRow(cell=cell(), measured=20.0),
            GroundTruthRow(cell=cell(ops=OPS_HOST, arm="dense", density=1.0),
                           measured=10.0)]


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(L, "LOCKED_CONFIG", CONFIG)
    return CONFIG


def p2_parts(rows=None):
    rows = rows or p1_rows()
    ts = committed_training_set(rows, "P2", banked=["P1"])
    return (fit_fitted_sum(ts), fit_fitted_sum(ts, add_s0=True),
            fit_learned(ts, CONFIG, model_factory=MeanModel))


def p2_record(**kw):
    fs, fs0, lm = p2_parts()
    args = dict(pairing="P2", cells=[cell("P2"), cell("P2", ops=OPS_HOST,
                                                      arm="dense", density=1.0)],
                queue_depth=8, banked=["P1"], fitted_sum=fs, learned=lm,
                fitted_sum_plus_s0=fs0, code=CLEAN, written_at=T0)
    args.update(kw)
    return build_record(**args)


def committed(tmp_path, record, store, sync_at):
    store.clock.t = sync_at
    f = write_prediction(record, tmp_path)
    return f, sync_prediction(f, store, "gs://bucket/predictions/P")


# --------------------------------------------------------------------------
# What the record holds
# --------------------------------------------------------------------------

def test_p1_holds_the_hand_computed_predictions_and_no_fitted_baseline():
    rec = p1_record()
    x, d = rec["cells"]
    assert x["predictions"] == {"recursion": 16, "sum": 13, "sum_plus_s0": 16,
                                "step_form": 14}
    assert d["predictions"] == {"recursion": 8, "sum": 8, "sum_plus_s0": 9,
                                "step_form": 7}
    assert rec["training"] == {}
    assert rec["not_fitted"] == {n: NOT_FITTED_REASON for n in
                                 ("fitted_sum", "learned", "fitted_sum_plus_s0")}
    assert rec["banked_before"] == [] and rec["queue_depth"] == 8


def test_p2_names_the_pairings_each_baseline_was_trained_on(locked):
    rec = p2_record()
    for name in ("fitted_sum", "learned", "fitted_sum_plus_s0"):
        assert rec["training"][name]["trained_on"] == ["P1"]
        assert rec["training"][name]["kind"] == "committed"
    assert rec["training"]["learned"]["config"] == {"target": "seconds",
                                                    "params": {}}
    assert rec["not_fitted"] == {}
    for c, priced in zip(rec["cells"], (price(OPS, COSTS), price(OPS_HOST, COSTS))):
        t = rec["training"]["fitted_sum"]
        H, Lk, D = sum(priced.host), sum(priced.link), sum(priced.device)
        assert c["predictions"]["fitted_sum"] == pytest.approx(
            t["a"] * H + t["b"] * Lk + t["c"] * D)
        # MeanModel predicts the mean of P1's two times, (20 + 10) / 2.
        assert c["predictions"]["learned"] == 15


def test_p2_fitted_sum_follows_the_hand_fit(locked):
    # Rows (H, L, D; y): xattn (2, 0, 11; 20) and dense (6, 0, 2; 10). Two
    # rows, two identified terms: a 2 + c 11 = 20 and a 6 + c 2 = 10 solve
    # exactly with zero loss: c = 100/62, a = (10 - 2c) / 6.
    fs = p2_parts()[0]
    c = 100 / 62
    assert fs.c == pytest.approx(c) and fs.a == pytest.approx((10 - 2 * c) / 6)
    assert fs.unidentified == ("link",)


# --------------------------------------------------------------------------
# Round trip and the independent hash check
# --------------------------------------------------------------------------

def test_write_sync_load_round_trip(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    f, receipt = committed(tmp_path, p2_record(), store,
                           T0 + timedelta(seconds=5))
    assert receipt.synced_at == T0 + timedelta(seconds=5)
    cp = load_committed(store, receipt.remote,
                        ground_truth_started_at=T0 + timedelta(seconds=6),
                        clock_skew=NO_SKEW, banked_at={"P1": T0 - timedelta(days=1)},
                        local=f.path)
    assert cp.pairing == "P2" and cp.sha256 == f.sha256 and cp.queue_depth == 8
    assert cp.synced_at == receipt.synced_at and cp.written_at == T0
    x = cp.cell("xattn", 4, 0.5, 1)
    assert x.priced == price(OPS, COSTS)
    assert x.predictions["recursion"] == 16 and x.predictions["learned"] == 15


def test_the_sidecar_checks_with_the_system_sha256_tool(tmp_path):
    tool = (["shasum", "-a", "256", "-c"] if shutil.which("shasum")
            else ["sha256sum", "-c"] if shutil.which("sha256sum") else None)
    if tool is None:
        pytest.skip("no shasum or sha256sum on this host")
    f = write_prediction(p1_record(), tmp_path)
    ok = subprocess.run([*tool, f.sidecar.name], cwd=tmp_path,
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    f.path.write_bytes(f.path.read_bytes().replace(b'"P1"', b'"P9"'))
    bad = subprocess.run([*tool, f.sidecar.name], cwd=tmp_path,
                         capture_output=True, text=True)
    assert bad.returncode != 0


def test_the_file_is_canonical_json_with_a_final_newline(tmp_path):
    f = write_prediction(p1_record(), tmp_path)
    data = f.path.read_bytes()
    assert data.endswith(b"}\n") and b'": ' not in data
    assert canonical_bytes(json.loads(data)) == data
    assert f.path.name == "prediction_P1_20261102T030000000000Z.json"


def test_an_infinite_condition_number_survives_the_round_trip(tmp_path, locked):
    # One training row cannot separate a from c: the condition number is inf.
    rows = [GroundTruthRow(cell=cell(), measured=20.0)]
    ts = committed_training_set(rows, "P2", banked=["P1"])
    fs = fit_fitted_sum(ts)
    assert fs.condition_number == float("inf")
    rec = p2_record(fitted_sum=fs, fitted_sum_plus_s0=None,
                    learned=fit_learned(ts, CONFIG, model_factory=MeanModel))
    assert rec["training"]["fitted_sum"]["condition_number"] == "inf"
    f = write_prediction(rec, tmp_path)
    cp = verify_bytes(f.path.read_bytes(), f.sidecar.read_text(), f.path.name)
    assert cp.training["fitted_sum"]["condition_number"] == "inf"


@pytest.mark.parametrize("seed", range(25))
def test_random_sequences_round_trip_exactly(tmp_path, seed):
    rng = random.Random(9100 + seed)
    ops = []
    for _ in range(rng.randint(1, 25)):
        r = rng.random()
        if r < 0.6:
            ops.append(kernel_op(rng.random() * 3, rng.random() * 9, layer=0))
        elif r < 0.8:
            ops.append(copy_op(rng.random(), rng.random() * 4, layer=0))
        else:
            ops.append(sync_op(rng.choice(["item", "bool", "synchronize"]),
                               layer=0))
    ops = close_prefill(ops)
    Q = rng.randint(1, 6)
    rec = p1_record(cells=[cell(ops=ops)], queue_depth=Q)
    f = write_prediction(rec, tmp_path)
    cp = verify_bytes(f.path.read_bytes(), f.sidecar.read_text(), f.path.name)
    p = price(ops, COSTS)
    assert cp.cells[0].priced == p
    assert cp.cells[0].predictions["recursion"] == recurse(p, Q).end_to_end


# --------------------------------------------------------------------------
# What build_record refuses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("Q", [None, 0, -1, True, 2.0])
def test_the_committed_prediction_uses_a_measured_q(Q):
    with pytest.raises(PredictionCommitError, match="measured Q"):
        p1_record(queue_depth=Q)


@pytest.mark.parametrize("code", [
    CodeStamp("a" * 40, True), CodeStamp("a" * 40, None),
    CodeStamp("HEAD", False), CodeStamp("a" * 39, False)])
def test_a_dirty_or_unknown_commit_is_refused(code):
    with pytest.raises(PredictionCommitError):
        p1_record(code=code)


def test_a_naive_write_time_is_refused():
    with pytest.raises(PredictionCommitError, match="timezone-aware"):
        p1_record(written_at=datetime(2026, 11, 2, 3, 0))


def test_cells_must_be_this_pairings_unique_and_closed():
    with pytest.raises(PredictionCommitError, match="at least one cell"):
        p1_record(cells=[])
    with pytest.raises(PredictionCommitError, match="a cell of P2"):
        p1_record(cells=[cell("P2")])
    with pytest.raises(PredictionCommitError, match="appears twice"):
        p1_record(cells=[cell(), cell()])
    with pytest.raises(PredictionCommitError, match="closing synchronise"):
        p1_record(cells=[cell(ops=[kernel_op(1, 1, layer=0)])])
    # A last synchronisation that is not a payload-free synchronize is not
    # the closing one.
    with pytest.raises(PredictionCommitError, match="closing synchronise"):
        p1_record(cells=[cell(ops=[kernel_op(1, 1, layer=0),
                                   sync_op("item", layer=0)])])


def test_the_step_form_needs_layer_labels_at_commit():
    with pytest.raises(CostModelError, match="no layer label"):
        p1_record(cells=[cell(ops=close_prefill([kernel_op(1, 1)]))])


def test_a_pairing_cannot_be_banked_before_itself():
    with pytest.raises(PredictionCommitError, match="banked before its own"):
        p1_record(banked=["P1"])


def test_p1_refuses_a_baseline_fitted_on_nothing(locked):
    fs, fs0, lm = p2_parts()
    for kw in ({"fitted_sum": fs}, {"learned": lm}, {"fitted_sum_plus_s0": fs0}):
        with pytest.raises(PredictionCommitError, match="fit on nothing"):
            p1_record(**kw)


@pytest.mark.parametrize("drop", ["fitted_sum", "learned"])
def test_p2_commits_the_fitted_sum_and_the_learned_predictor_together(locked, drop):
    with pytest.raises(PredictionCommitError, match=f"missing: \\['{drop}'\\]"):
        p2_record(**{drop: None})


def test_the_fitted_sum_plus_s0_is_optional(locked):
    rec = p2_record(fitted_sum_plus_s0=None)
    assert "fitted_sum_plus_s0" not in rec["training"]
    assert "fitted_sum_plus_s0" not in rec["cells"][0]["predictions"]


def test_the_two_fitted_sums_cannot_swap_slots(locked):
    fs, fs0, _ = p2_parts()
    with pytest.raises(PredictionCommitError, match="labelled separately"):
        p2_record(fitted_sum=fs0)
    with pytest.raises(PredictionCommitError, match="labelled separately"):
        p2_record(fitted_sum_plus_s0=fs)


def test_a_retrospective_fit_is_not_a_committed_prediction(locked):
    rows = p1_rows() + [GroundTruthRow(cell=cell("P4"), measured=30.0)]
    ts = retrospective_training_set(rows, "P2")
    with pytest.raises(PredictionCommitError, match="retrospective"):
        p2_record(fitted_sum=fit_fitted_sum(ts), fitted_sum_plus_s0=None)
    with pytest.raises(PredictionCommitError, match="retrospective"):
        p2_record(learned=fit_learned(ts, CONFIG, model_factory=MeanModel))


def test_a_fit_for_another_target_is_refused(locked):
    ts = committed_training_set(p1_rows(), "P4", banked=["P1"])
    with pytest.raises(PredictionCommitError, match="fitted for P4"):
        p2_record(fitted_sum=fit_fitted_sum(ts))


def test_the_training_pairings_must_be_exactly_the_banked_ones(locked):
    rows = p1_rows() + [GroundTruthRow(cell=cell("P4"), measured=30.0)]
    with pytest.raises(PredictionCommitError, match="whatever was banked"):
        p2_record(banked=["P1", "P4"])          # fits name P1 only
    ts = committed_training_set(rows, "P2", banked=["P1", "P4"])
    with pytest.raises(PredictionCommitError, match="whatever was banked"):
        p2_record(fitted_sum=fit_fitted_sum(ts))  # banked names P1 only


def test_a_fit_to_another_statistic_is_refused(locked):
    rows = [dataclasses.replace(r, statistic="p90") for r in p1_rows()]
    fs, _, lm = p2_parts(rows)
    with pytest.raises(PredictionCommitError, match="fitted to the p90"):
        p2_record(fitted_sum=fs, learned=lm, fitted_sum_plus_s0=None)


def test_no_learned_prediction_is_committed_before_lock():
    assert L.LOCKED_CONFIG is None
    with pytest.raises(PredictionCommitError, match="fixed at lock"):
        p2_record()


def test_the_learned_config_must_be_the_locked_one(monkeypatch):
    monkeypatch.setattr(L, "LOCKED_CONFIG",
                        LearnedConfig(target="log_seconds"))
    with pytest.raises(PredictionCommitError, match="not the locked"):
        p2_record()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_non_positive_or_non_finite_prediction_is_refused(locked, bad):
    fs, fs0, lm = p2_parts()

    class Bad:
        def predict(self, X):
            return [bad for _ in X]

    with pytest.raises(PredictionCommitError, match="positive finite"):
        p2_record(learned=dataclasses.replace(lm, model=Bad()))


def test_an_unserialisable_learned_parameter_is_refused(monkeypatch):
    cfg = LearnedConfig(target="seconds", params={"x": object()})
    monkeypatch.setattr(L, "LOCKED_CONFIG", cfg)
    ts = committed_training_set(p1_rows(), "P2", banked=["P1"])
    rec = p2_record(learned=fit_learned(ts, cfg, model_factory=MeanModel))
    with pytest.raises(PredictionCommitError, match="canonical JSON"):
        canonical_bytes(rec)


# --------------------------------------------------------------------------
# Writing and syncing
# --------------------------------------------------------------------------

def test_a_committed_file_is_never_overwritten(tmp_path):
    f = write_prediction(p1_record(), tmp_path)
    before = f.path.read_bytes(), f.sidecar.read_bytes()
    # Same pairing and write stamp, so the same name, and other bytes.
    with pytest.raises(PredictionCommitError, match="never overwritten"):
        write_prediction(p1_record(cells=[cell()]), tmp_path)
    assert (f.path.read_bytes(), f.sidecar.read_bytes()) == before


def test_an_orphan_sidecar_blocks_the_write_before_anything_is_written(tmp_path):
    f = write_prediction(p1_record(), tmp_path)
    f.path.unlink()
    with pytest.raises(PredictionCommitError, match="never overwritten"):
        write_prediction(p1_record(), tmp_path)
    assert not f.path.exists()


def test_the_canonical_bytes_do_not_depend_on_key_order():
    rec = p1_record()
    reordered = dict(reversed(list(rec.items())))
    reordered["code"] = dict(reversed(list(rec["code"].items())))
    assert list(reordered) != list(rec)
    assert canonical_bytes(reordered) == canonical_bytes(rec)


def test_sync_refuses_a_file_changed_after_it_was_written(tmp_path):
    f = write_prediction(p1_record(), tmp_path)
    f.path.write_bytes(f.path.read_bytes() + b" ")
    with pytest.raises(PredictionCommitError, match="changed after"):
        sync_prediction(f, MemoryStore(Clock(T0)), "gs://b/p")


def test_sync_never_replaces_an_object_already_in_the_store(tmp_path):
    store = MemoryStore(Clock(T0), no_clobber=True)
    other = tmp_path / "other"
    other.mkdir()
    g = write_prediction(p1_record(cells=[cell()]), other)
    sync_prediction(g, store, "gs://b/p")
    f = write_prediction(p1_record(), tmp_path)      # same name, other bytes
    with pytest.raises(PredictionCommitError, match="never replaced"):
        sync_prediction(f, store, "gs://b/p")


def test_sync_reports_the_later_creation_time(tmp_path):
    store = MemoryStore(Clock(T0))
    f = write_prediction(p1_record(), tmp_path)

    class Late(MemoryStore):
        def put(self, local, remote):
            super().put(local, remote)
            self.clock.t += timedelta(seconds=3)

    late = Late(Clock(T0))
    r = sync_prediction(f, late, "gs://b/p/")
    assert r.remote == "gs://b/p/" + f.path.name
    assert r.synced_at == T0 + timedelta(seconds=3)   # the sidecar's
    assert store.objs == {}


# --------------------------------------------------------------------------
# The analysis's refusals (K9): the "Done when" of T2.11
# --------------------------------------------------------------------------

def load(store, receipt, start, skew=NO_SKEW, banked_at=None, **kw):
    return load_committed(store, receipt.remote, ground_truth_started_at=start,
                          clock_skew=skew, banked_at=banked_at, **kw)


def test_a_prediction_synced_before_the_workload_is_accepted(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    assert load(store, r, T0 + timedelta(seconds=2)).synced_at \
        == T0 + timedelta(seconds=1)


@pytest.mark.parametrize("start_after_sync", [timedelta(0), -timedelta(seconds=1),
                                              -timedelta(days=1)])
def test_a_prediction_synced_at_or_after_the_start_is_refused(tmp_path,
                                                              start_after_sync):
    store = MemoryStore(Clock(T0))
    sync_at = T0 + timedelta(seconds=10)
    _, r = committed(tmp_path, p1_record(), store, sync_at)
    with pytest.raises(LatePredictionError, match="refuses it"):
        load(store, r, sync_at + start_after_sync)


def test_the_clock_skew_only_makes_the_check_stricter(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    start = T0 + timedelta(seconds=2)
    load(store, r, start, skew=timedelta(milliseconds=999))
    with pytest.raises(LatePredictionError):
        load(store, r, start, skew=timedelta(seconds=1))
    with pytest.raises(PredictionCommitError, match="clock_skew"):
        load(store, r, start, skew=-timedelta(seconds=1))


def test_a_sidecar_synced_late_makes_the_prediction_late(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    store.times[r.remote_sidecar] = T0 + timedelta(seconds=9)
    with pytest.raises(LatePredictionError):
        load(store, r, T0 + timedelta(seconds=5))


def test_a_write_stamp_after_the_sync_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    # Written at T0 + 10 s by the instance's clock, created at T0 + 1 s by
    # the store's: one of them is wrong by more than the skew allowed.
    _, r = committed(tmp_path, p1_record(written_at=T0 + timedelta(seconds=10)),
                     store, T0 + timedelta(seconds=1))
    with pytest.raises(PredictionCommitError, match="clocks disagree"):
        load(store, r, T0 + timedelta(seconds=20))
    load(store, r, T0 + timedelta(seconds=20), skew=timedelta(seconds=9))


def test_a_naive_start_time_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    with pytest.raises(PredictionCommitError, match="timezone-aware"):
        load(store, r, datetime(2026, 11, 3))


def test_baselines_trained_on_a_pairing_banked_later_are_refused(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p2_record(), store, T0 + timedelta(seconds=1))
    start = T0 + timedelta(seconds=2)
    load(store, r, start, banked_at={"P1": T0 - timedelta(seconds=1)})
    with pytest.raises(LatePredictionError, match="P1's ground truth"):
        load(store, r, start, banked_at={"P1": T0})
    # Banked 1 s before the write by one clock, but 2 s of skew allowed.
    with pytest.raises(LatePredictionError, match="P1's ground truth"):
        load(store, r, T0 + timedelta(seconds=10), skew=timedelta(seconds=2),
             banked_at={"P1": T0 - timedelta(seconds=1)})
    with pytest.raises(PredictionCommitError, match="no banking time"):
        load(store, r, start, banked_at={})


def test_p1_needs_no_banking_times(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0 + timedelta(seconds=1))
    load(store, r, T0 + timedelta(seconds=2), banked_at=None)


def _tamper(store, r, edit, rehash=True):
    rec = json.loads(store.objs[r.remote])
    edit(rec)
    data = canonical_bytes(rec)
    store.objs[r.remote] = data
    if rehash:
        name = r.remote.rsplit("/", 1)[1]
        store.objs[r.remote_sidecar] = C.sidecar_text(
            C.sha256_hex(data), name).encode()


def test_altered_bytes_fail_the_hash(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    _tamper(store, r, lambda rec: rec.update(queue_depth=9), rehash=False)
    with pytest.raises(PredictionCommitError, match="not the one its sidecar"):
        load(store, r, T0 + timedelta(seconds=1))


@pytest.mark.parametrize("name", COMPUTED)
def test_a_prediction_that_does_not_follow_from_its_components_is_refused(
        tmp_path, name):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    _tamper(store, r, lambda rec: rec["cells"][0]["predictions"].update(
        {name: rec["cells"][0]["predictions"][name] * 0.9}))
    with pytest.raises(PredictionCommitError, match="does not follow"):
        load(store, r, T0 + timedelta(seconds=1))


def test_altered_components_are_refused_too(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    _tamper(store, r, lambda rec: rec["cells"][0]["components"]["device"]
            .__setitem__(0, 1.0))
    with pytest.raises(PredictionCommitError, match="does not follow"):
        load(store, r, T0 + timedelta(seconds=1))


@pytest.mark.parametrize("name", ["fitted_sum", "fitted_sum_plus_s0"])
def test_an_altered_fitted_coefficient_is_refused(tmp_path, locked, name):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p2_record(), store, T0)
    _tamper(store, r, lambda rec: rec["training"][name].update(a=0.5))
    with pytest.raises(PredictionCommitError,
                       match=f"committed {name} prediction .* does not follow"):
        load(store, r, T0 + timedelta(seconds=1),
             banked_at={"P1": T0 - timedelta(1)})


@pytest.mark.parametrize("edit, match", [
    (lambda rec: rec["training"].pop("learned"), "but no \\['learned'\\]"),
    (lambda rec: rec["training"]["learned"].update(trained_on=["P4"]),
     "does not name the committed"),
    (lambda rec: rec["training"]["learned"].update(kind="retrospective"),
     "does not name the committed"),
    (lambda rec: rec["training"].update(extra={}), "unknown baselines"),
    (lambda rec: rec["cells"][0]["predictions"].pop("learned"),
     "has predictions"),
    (lambda rec: rec.update(code={"git_commit": "a" * 40, "git_dirty": True}),
     "dirty"),
    (lambda rec: rec.update(schema="attnbench.prediction/0"), "schema"),
    (lambda rec: rec.update(queue_depth=0), "queue depth"),
    (lambda rec: rec.update(statistic="mean"), "statistic"),
])
def test_an_altered_record_is_refused_even_when_rehashed(tmp_path, locked, edit,
                                                         match):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p2_record(), store, T0)
    _tamper(store, r, edit)
    with pytest.raises(PredictionCommitError, match=match):
        load(store, r, T0 + timedelta(seconds=1),
             banked_at={"P1": T0 - timedelta(1)})


def test_p1_with_training_records_added_is_refused(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    rec2 = p2_record()
    _tamper(store, r, lambda rec: rec.update(training=rec2["training"]))
    with pytest.raises(PredictionCommitError, match="no pairing banked"):
        load(store, r, T0 + timedelta(seconds=1))


def test_a_pretty_printed_copy_is_not_canonical(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    data = json.dumps(json.loads(store.objs[r.remote]), indent=1).encode()
    store.objs[r.remote] = data
    store.objs[r.remote_sidecar] = C.sidecar_text(
        C.sha256_hex(data), r.remote.rsplit("/", 1)[1]).encode()
    with pytest.raises(PredictionCommitError, match="canonical form"):
        load(store, r, T0 + timedelta(seconds=1))


def test_a_renamed_file_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    new = r.remote.replace("P1_", "P2_")
    data = store.objs[r.remote]
    store.objs[new] = data
    store.objs[new + ".sha256"] = C.sidecar_text(
        C.sha256_hex(data), new.rsplit("/", 1)[1]).encode()
    store.times[new] = store.times[new + ".sha256"] = T0
    with pytest.raises(PredictionCommitError, match="does not match its own"):
        load_committed(store, new, ground_truth_started_at=T0 + timedelta(1),
                       clock_skew=NO_SKEW)


def test_a_sidecar_naming_another_file_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    _, r = committed(tmp_path, p1_record(), store, T0)
    store.objs[r.remote_sidecar] = C.sidecar_text(
        C.sha256_hex(store.objs[r.remote]), "other.json").encode()
    with pytest.raises(PredictionCommitError, match="names 'other.json'"):
        load(store, r, T0 + timedelta(seconds=1))


def test_a_local_copy_that_differs_from_the_store_is_refused(tmp_path):
    store = MemoryStore(Clock(T0))
    f, r = committed(tmp_path, p1_record(), store, T0)
    f.path.write_bytes(f.path.read_bytes() + b"\n")
    with pytest.raises(PredictionCommitError, match="differ"):
        load(store, r, T0 + timedelta(seconds=1), local=f.path)


# --------------------------------------------------------------------------
# Stamps
# --------------------------------------------------------------------------

def test_ground_truth_start_is_the_earliest_stamp():
    t = T0.timestamp()
    assert ground_truth_start([t + 5, t, T0 + timedelta(seconds=3)]) == T0
    with pytest.raises(PredictionCommitError, match="no ground-truth stamps"):
        ground_truth_start([])
    with pytest.raises(PredictionCommitError, match="timezone-aware"):
        ground_truth_start([datetime(2026, 1, 1)])


@pytest.mark.parametrize("state, ok", [
    (("b" * 40, False), True), (("b" * 40, True), False), ((None, None), False)])
def test_current_code_stamp_reads_provenance(monkeypatch, state, ok):
    import attnbench.provenance as prov
    monkeypatch.setattr(prov, "_git_state", lambda: state)
    if state[0] is None:
        with pytest.raises(PredictionCommitError, match="no usable git"):
            C.current_code_stamp()
        return
    stamp = C.current_code_stamp()
    assert stamp == CodeStamp(*state)
    if ok:
        p1_record(code=stamp)
    else:
        with pytest.raises(PredictionCommitError, match="dirty"):
            p1_record(code=stamp)


# --------------------------------------------------------------------------
# GsutilStore against a fake gsutil
# --------------------------------------------------------------------------

FAKE_GSUTIL = r"""#!/usr/bin/env python3
import os, sys, pathlib
root = pathlib.Path(os.environ["FAKE_GCS_ROOT"])
with open(root / "argv.log", "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
args = [a for a in sys.argv[1:] if a != "-q"]
def path(uri):
    assert uri.startswith("gs://")
    return root / "objects" / uri[5:]
if args[0] == "cp":
    noclobber = "-n" in args
    src, dst = [a for a in args[1:] if a != "-n"]
    p = path(dst)
    if noclobber and p.exists():
        sys.exit(0)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pathlib.Path(src).read_bytes())
elif args[0] == "cat":
    p = path(args[1])
    if not p.exists():
        sys.stderr.write("No URLs matched\n"); sys.exit(1)
    sys.stdout.buffer.write(p.read_bytes())
elif args[0] == "stat":
    print(args[1] + ":")
    print("    Creation time:          " + os.environ["FAKE_GCS_CREATED"])
    print("    Storage class:          STANDARD")
else:
    sys.exit(2)
"""


@pytest.fixture
def fake_gsutil(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "gsutil"
    exe.write_text(FAKE_GSUTIL)
    exe.chmod(0o755)
    root = tmp_path / "gcs"
    root.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GCS_ROOT", str(root))
    monkeypatch.setenv("FAKE_GCS_CREATED", "Mon, 02 Nov 2026 03:00:07 GMT")
    return root


def test_gsutil_store_round_trip(tmp_path, fake_gsutil):
    out = tmp_path / "out"
    out.mkdir()
    f = write_prediction(p1_record(), out)
    r = sync_prediction(f, GsutilStore(), "gs://bucket/predictions/P1")
    # gsutil prints whole seconds, so the store reports one second later.
    assert r.synced_at == datetime(2026, 11, 2, 3, 0, 8, tzinfo=UTC)
    calls = (fake_gsutil / "argv.log").read_text().splitlines()
    assert calls[0].startswith("-q cp -n ")
    cp = load_committed(GsutilStore(), r.remote,
                        ground_truth_started_at=datetime(2026, 11, 2, 3, 0, 9,
                                                         tzinfo=UTC),
                        clock_skew=NO_SKEW)
    assert cp.sha256 == f.sha256
    # Created at 03:00:07 and printed so; the workload starting at 03:00:08
    # could follow it by under a second, so it is refused.
    with pytest.raises(LatePredictionError):
        load_committed(GsutilStore(), r.remote,
                       ground_truth_started_at=datetime(2026, 11, 2, 3, 0, 8,
                                                        tzinfo=UTC),
                       clock_skew=NO_SKEW)


def test_gsutil_store_refuses_other_uris_and_reports_failures(tmp_path,
                                                             fake_gsutil):
    s = GsutilStore()
    with pytest.raises(PredictionCommitError, match="not a gs:// URI"):
        s.get("/local/path")
    with pytest.raises(PredictionCommitError, match="exited 1"):
        s.get("gs://bucket/missing")


def test_gsutil_store_needs_a_creation_time(tmp_path, fake_gsutil, monkeypatch):
    monkeypatch.setenv("FAKE_GCS_CREATED", "not a date")
    with pytest.raises(PredictionCommitError, match="unreadable creation time"):
        GsutilStore().created("gs://bucket/x")
