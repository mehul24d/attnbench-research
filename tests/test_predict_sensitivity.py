"""T2.15: the queue sensitivity and the residual analysis, both read from a
committed prediction file.

The queue example is worked by hand from Section 5.4; random sequences are
checked against the pending-launch oracle of test_predict_recursion, which
does not index D_(i-Q).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from attnbench.predict.commit import (
    CodeStamp, PredictionCommitError, build_record, verify_bytes,
    write_prediction)
from attnbench.predict.recursion import (
    close_prefill, copy_op, kernel_op, price, sync_op)
from attnbench.predict.residuals import (
    DENSE, SYNC_FREE, SYNCHRONISING, arm_class, residuals, summarise,
    synchronisations_in_step)
from attnbench.predict.sensitivity import (
    decision_sensitivity, queue_sensitivity)
from attnbench.predict.training import Cell, HardwareFeatures

from test_predict_recursion import COSTS, oracle

T0 = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
CLEAN = CodeStamp("a" * 40, False)
HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)

# Three device-heavy kernels, then one long host operation, then close.
#   No queue (and Q = 4, 8, 16): H 1, 2, 3, 13; D 5, 9, 13, 14; close 15.
#   Q = 1: H 1, 5, 9, 19; D 5, 9, 13, 20; the close waits to 20: 21.
#   Q = 2: H 1, 2, 5, 15; D 5, 9, 13, 16; the close waits to 16: 17.
QSEQ = close_prefill([kernel_op(1, 4, layer=0)] * 3 + [kernel_op(10, 1, layer=0)])
# K(h1, d14), close: 16 at every Q.
DENSE_SEQ = close_prefill([kernel_op(1, 14, layer=0)])
# K(h1,d10) S(item) K(h1,d1), close: 16 at every Q (test_predict_commit).
SYNC_SEQ = close_prefill([kernel_op(1, 10, layer=0),
                          sync_op("item", payload_bytes=4, layer=0),
                          kernel_op(1, 1, layer=0)])


def cell(ops, arm, L=4096, density=0.5, batch=1):
    return Cell(pairing="P1", arm=arm, context_length=L, density=density,
                batch=batch, priced=price(ops, COSTS), hardware=HW)


def committed(tmp_path, cells, Q=4, **kw):
    rec = build_record("P1", cells, Q, banked=[], fitted_sum=None, learned=None,
                       code=CLEAN, written_at=T0, **kw)
    f = write_prediction(rec, tmp_path)
    return verify_bytes(f.path.read_bytes(), f.sidecar.read_text(), f.path.name)


# --------------------------------------------------------------------------
# Queue sensitivity
# --------------------------------------------------------------------------

def test_queue_sensitivity_hand_computed(tmp_path):
    cp = committed(tmp_path, [cell(QSEQ, "xattn")])
    (s,) = queue_sensitivity(cp)
    assert s.measured_q == 4 and s.committed == 15
    got = [(p.multiple, p.queue_depth, p.predicted) for p in s.points]
    assert got == [(0.25, 1, 21), (0.5, 2, 17), (2, 8, 15), (4, 16, 15),
                   (None, None, 15)]
    assert s.at(0.25).relative_change == pytest.approx(6 / 15)
    assert s.at(None).relative_change == 0


def test_the_depths_are_rounded_to_nearest_not_floored(tmp_path):
    # Measured Q = 3: 0.75 -> 1, 1.5 -> 2 (half up; a floor gives 1), 6, 12.
    cp = committed(tmp_path, [cell(QSEQ, "xattn")], Q=3)
    (s,) = queue_sensitivity(cp)
    assert [p.queue_depth for p in s.points] == [1, 2, 6, 12, None]
    assert s.at(0.5).predicted == 17                   # Q = 2, from above


def test_every_multiple_is_reported_even_when_depths_coincide(tmp_path):
    # Measured Q = 1: 0.25x and 0.5x both round to 1, the measured depth.
    cp = committed(tmp_path, [cell(QSEQ, "xattn")], Q=1)
    (s,) = queue_sensitivity(cp)
    assert [p.queue_depth for p in s.points] == [1, 1, 2, 4, None]
    assert s.committed == 21 and s.at(0.5).predicted == 21
    assert s.at(None).predicted == 15


@pytest.mark.parametrize("seed", range(30))
def test_queue_sensitivity_matches_the_pending_launch_oracle(tmp_path, seed):
    rng = random.Random(7700 + seed)
    ops = []
    for _ in range(rng.randint(1, 20)):
        r = rng.random()
        ops.append(kernel_op(rng.randint(0, 6), rng.randint(0, 12), layer=0)
                   if r < 0.6 else copy_op(rng.randint(0, 4), rng.randint(0, 6),
                                           layer=0)
                   if r < 0.8 else sync_op("bool", payload_bytes=4, layer=0))
    ops = close_prefill(ops)
    Q = rng.randint(1, 8)
    cp = committed(tmp_path, [cell(ops, "x")], Q=Q)
    s0s = price(ops, COSTS).sync_cost
    for p in queue_sensitivity(cp)[0].points:
        assert p.predicted == oracle(ops, s0s, p.queue_depth)[-1]


def test_the_constant_host_cost_baseline_has_a_queue_term_too(tmp_path):
    # Constant 2 on QSEQ: K(2,4) x3, K(2,1). No queue: H 2, 4, 6, 8;
    # D 6, 10, 14, 15; close 16. Q = 1: H 2, 6, 10, 14; D 6, 10, 14, 15;
    # the close waits to 15: 16. Q = 2: H 2, 4, 6, 10; D 6, 10, 14, 15: 16.
    cp = committed(tmp_path, [cell(QSEQ, "xattn")], constant_host_cost=2.0)
    (s,) = queue_sensitivity(cp, "constant_host_cost")
    assert s.committed == 16 and {p.predicted for p in s.points} == {16}
    with pytest.raises(PredictionCommitError, match="no queue term"):
        queue_sensitivity(cp, "sum")
    (tmp_path / "b").mkdir()
    cp = committed(tmp_path / "b", [cell(QSEQ, "xattn")])
    with pytest.raises(PredictionCommitError, match="no constant-host-cost"):
        queue_sensitivity(cp, "constant_host_cost")


def test_a_committed_value_the_components_do_not_reproduce_is_refused(tmp_path):
    import dataclasses
    cp = committed(tmp_path, [cell(QSEQ, "xattn")])
    c = cp.cells[0]
    bad = dataclasses.replace(c, predictions={**c.predictions, "recursion": 14.0})
    with pytest.raises(PredictionCommitError, match="does not reproduce"):
        queue_sensitivity(dataclasses.replace(cp, cells=(bad,)))


def test_decision_sensitivity_names_the_depths_that_flip_the_sign(tmp_path):
    # Committed (Q = 4): xattn 15 < dense 16, sparse faster. At 0.25x it is
    # 21 and at 0.5x 17, both slower than dense: those two flip.
    cp = committed(tmp_path, [cell(QSEQ, "xattn"),
                              cell(DENSE_SEQ, "dense", density=1.0),
                              cell(QSEQ, "xattn", L=8192)])
    (d,) = decision_sensitivity(queue_sensitivity(cp), dense_arm="dense")
    assert d.sparse_key == ("xattn", 4096, 0.5, 1)
    assert d.dense_key == ("dense", 4096, 1.0, 1)
    assert d.committed_sparse_faster and d.flips_at == (0.25, 0.5)
    with pytest.raises(PredictionCommitError, match="no 'base' cells"):
        decision_sensitivity(queue_sensitivity(cp), dense_arm="base")


def test_decision_sensitivity_reads_dense_at_the_same_depth(tmp_path):
    # Dense is the queue-sensitive one now: dense 15 (Q = 4), 21 (Q = 1), 17
    # (Q = 2); the sparse arm is 16 at every depth. Committed: sparse slower.
    # At 0.25x and 0.5x dense is slower than 16, so sparse is faster: flips.
    cp = committed(tmp_path, [cell(DENSE_SEQ, "xattn"),
                              cell(QSEQ, "dense", density=1.0)])
    (d,) = decision_sensitivity(queue_sensitivity(cp), dense_arm="dense")
    assert not d.committed_sparse_faster and d.flips_at == (0.25, 0.5)


def test_a_cell_without_a_dense_partner_at_its_batch_is_left_out(tmp_path):
    cp = committed(tmp_path, [cell(QSEQ, "xattn"),
                              cell(QSEQ, "xattn", batch=4),
                              cell(DENSE_SEQ, "dense", density=1.0)])
    ds = decision_sensitivity(queue_sensitivity(cp), dense_arm="dense")
    assert [d.sparse_key for d in ds] == [("xattn", 4096, 0.5, 1)]


def test_the_dense_partner_is_the_one_at_the_same_batch(tmp_path):
    cp = committed(tmp_path, [cell(QSEQ, "xattn"),
                              cell(DENSE_SEQ, "dense", density=1.0),
                              cell(DENSE_SEQ, "dense", density=1.0, batch=4)])
    (d,) = decision_sensitivity(queue_sensitivity(cp), dense_arm="dense")
    assert d.dense_key == ("dense", 4096, 1.0, 1)


def test_decision_sensitivity_refuses_mixed_predictors(tmp_path):
    cp = committed(tmp_path, [cell(QSEQ, "xattn"),
                              cell(DENSE_SEQ, "dense", density=1.0)],
                   constant_host_cost=2.0)
    mixed = queue_sensitivity(cp) + queue_sensitivity(cp, "constant_host_cost")
    with pytest.raises(PredictionCommitError, match="mix of predictors"):
        decision_sensitivity(mixed, dense_arm="dense")


# --------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------

XK = ("xattn", 4096, 0.5, 1)
DK = ("dense", 4096, 1.0, 1)
MK = ("meanpool", 4096, 0.5, 1)


def three_arms(tmp_path):
    return committed(tmp_path, [cell(SYNC_SEQ, "xattn"),
                                cell(DENSE_SEQ, "dense", density=1.0),
                                cell(QSEQ, "meanpool")])


def test_synchronisation_count_leaves_out_the_closing_synchronise(tmp_path):
    cp = three_arms(tmp_path)
    assert [synchronisations_in_step(c) for c in cp.cells] == [1, 0, 0]
    assert arm_class("dense", 0, {"dense"}) == DENSE
    assert arm_class("dense", 3, {"dense"}) == DENSE
    assert arm_class("meanpool", 0, {"dense"}) == SYNC_FREE
    assert arm_class("xattn", 1, {"dense"}) == SYNCHRONISING


def test_residuals_hand_computed(tmp_path):
    cp = three_arms(tmp_path)
    # Recursion: xattn 16, dense 16, meanpool 15 (Q = 4).
    rs = residuals(cp, {XK: 20.0, DK: 16.8, MK: 15.0}, statistic="median",
                   dense_arms={"dense"}, relative_floor=0.1,
                   predictors=["recursion"])
    by = {r.arm: r for r in rs.residuals}
    assert by["xattn"].residual == -4 and by["xattn"].relative == -0.2
    assert by["xattn"].arm_class == SYNCHRONISING and not by["xattn"].inside_floor
    assert by["dense"].relative == pytest.approx(-0.8 / 16.8)
    assert by["dense"].inside_floor and by["dense"].arm_class == DENSE
    assert by["meanpool"].residual == 0 and by["meanpool"].arm_class == SYNC_FREE
    assert rs.unmeasured == () and rs.unpredicted == ()


def test_every_committed_predictor_by_default(tmp_path):
    cp = three_arms(tmp_path)
    rs = residuals(cp, {XK: 20.0}, statistic="median", dense_arms={"dense"},
                   relative_floor=0)
    assert sorted(r.predictor for r in rs.residuals) == sorted(
        cp.cells[0].predictions)
    assert rs.unmeasured == (DK, MK)


def test_unmeasured_and_unpredicted_cells_are_named(tmp_path):
    cp = three_arms(tmp_path)
    extra = ("xattn", 8192, 0.5, 1)
    rs = residuals(cp, {XK: 20.0, extra: 30.0}, statistic="median",
                   dense_arms={"dense"}, relative_floor=0.1)
    assert rs.unpredicted == (extra,) and rs.unmeasured == (DK, MK)


def test_the_floor_is_inclusive(tmp_path):
    cp = three_arms(tmp_path)
    rs = residuals(cp, {XK: 20.0}, statistic="median", dense_arms={"dense"},
                   relative_floor=0.2, predictors=["recursion"])
    assert rs.residuals[0].inside_floor           # |-0.2| <= 0.2


@pytest.mark.parametrize("kw, match", [
    ({"statistic": "p90"}, "predicts the median"),
    ({"relative_floor": -0.1}, "noise floor"),
    ({"relative_floor": float("nan")}, "noise floor"),
    ({"dense_arms": set()}, "dense arm"),
    ({"predictors": ["learned"]}, "no committed 'learned'"),
])
def test_residual_arguments_are_checked(tmp_path, kw, match):
    cp = three_arms(tmp_path)
    args = dict(statistic="median", dense_arms={"dense"}, relative_floor=0.1)
    args.update(kw)
    with pytest.raises(PredictionCommitError, match=match):
        residuals(cp, {XK: 20.0}, **args)


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), True])
def test_a_bad_measurement_is_refused(tmp_path, bad):
    with pytest.raises(PredictionCommitError, match="measured"):
        residuals(three_arms(tmp_path), {XK: bad}, statistic="median",
                  dense_arms={"dense"}, relative_floor=0.1)


def test_summaries_by_class_and_by_sync_count(tmp_path):
    cp = three_arms(tmp_path)
    rs = residuals(cp, {XK: 20.0, DK: 16.8, MK: 12.0}, statistic="median",
                   dense_arms={"dense"}, relative_floor=0.1,
                   predictors=["recursion", "sum"])
    # Sums: xattn h 2 + d 11 = 13; dense 1 + 14 = 15; meanpool h 13 + d 13
    # = 26.
    by_class = {g.group: g for g in summarise(rs.residuals,
                                              ["predictor", "arm_class"])}
    x = by_class[("sum", SYNCHRONISING)]
    assert x.n == 1 and x.mean_relative == pytest.approx((13 - 20) / 20)
    m = by_class[("recursion", SYNC_FREE)]      # meanpool: 15 vs 12
    assert m.median_abs_relative == pytest.approx(0.25)
    by_sync = summarise(rs.residuals, ["n_synchronisations"])
    assert [g.group for g in by_sync] == [(0,), (1,)]
    zero = by_sync[0]                            # dense and meanpool, 2 each
    # Relative errors: recursion dense -0.8/16.8, meanpool 3/12; sum dense
    # -1.8/16.8, meanpool 14/12. Only -0.8/16.8 (0.048) is inside 0.1.
    rels = [-0.8 / 16.8, 3 / 12, -1.8 / 16.8, 14 / 12]
    assert zero.n == 4 and zero.n_inside_floor == 1
    assert zero.mean_relative == pytest.approx(sum(rels) / 4)
    assert zero.max_abs_relative == pytest.approx(14 / 12)
    # Sorted |rel|: 0.048, 0.107, 0.25, 1.17; the median is (0.107 + 0.25) / 2.
    assert zero.median_abs_relative == pytest.approx((1.8 / 16.8 + 0.25) / 2)


def test_summarise_needs_known_fields():
    with pytest.raises(PredictionCommitError, match="group by"):
        summarise([], ["colour"])
    with pytest.raises(PredictionCommitError, match="group by"):
        summarise([], [])
    assert summarise([], ["arm"]) == []
