"""T2.10: the learned baseline's frozen features, its training rule and the
learning curve.

LightGBM is not a dependency of this repository yet, so the fits here use a
stub whose predictions can be worked by hand; the one LightGBM test skips
when it is not installed. The feature list is checked against the plan's
own sentence, not against `FEATURES`.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

from attnbench.predict.learned import (
    FEATURE_SOURCE, FEATURES, LOCKED_CONFIG, PENDING_FEATURES, LearnedConfig,
    features, fit_learned, learning_curve, lightgbm_factory)
from attnbench.predict.recursion import (
    CostModelError, close_prefill, copy_op, kernel_op, price, sync_op)
from attnbench.predict.training import (
    COMMITTED, RETROSPECTIVE, Cell, GroundTruthRow, HardwareFeatures,
    committed_training_set)

from test_predict_recursion import COSTS

PLAN = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_PLAN.md"
HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)


class MeanModel:
    """Predicts the mean of its (encoded) training targets."""

    def __init__(self, params):
        self.params = dict(params)

    def fit(self, X, y):
        self.width = len(X[0])
        self.mean = sum(y) / len(y)

    def predict(self, X):
        assert all(len(x) == self.width for x in X)
        return [self.mean for _ in X]


def cell(pairing="P1", ops=None, L=8192, density=0.25, batch=1, arm="xattn"):
    ops = ops or [kernel_op(1, 1)]
    return Cell(pairing=pairing, arm=arm, context_length=L, density=density,
                batch=batch, priced=price(ops, COSTS), hardware=HW)


def row(pairing, measured, **kw):
    return GroundTruthRow(cell=cell(pairing, **kw), measured=measured)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _learned_bullet():
    text = PLAN.read_text()
    m = re.search(r"- \*\*Learned \(CTFusion-style LightGBM\):\*\*(.*)", text)
    assert m, "the plan's learned-baseline bullet moved"
    return m.group(1)


def test_every_feature_the_plan_names_is_frozen_or_pending():
    bullet = _learned_bullet()
    # The plan's list, phrase by phrase, and where each one went.
    named = {
        "context length": ["context_length"],
        "density": ["density"],
        "batch": ["batch"],
        "operation and synchronisation counts": ["n_operations",
                                                 "n_synchronisations"],
        "host, link, device costs": ["sum_host", "sum_link", "sum_device"],
        "pinned and pageable bandwidth": ["pinned_bandwidth",
                                          "pageable_bandwidth"],
        "arithmetic intensity per operation class":
            ["arithmetic_intensity_per_operation_class"],
        "host-to-device time ratio": ["host_to_device_ratio"],
        "transfer demand relative to link bandwidth":
            ["transfer_demand_relative_to_link_bandwidth"],
        "synchronisations per operation": ["synchronisations_per_operation"],
    }
    for phrase, names in named.items():
        assert phrase in bullet, f"the plan no longer names {phrase!r}"
        for n in names:
            assert n in FEATURES or n in PENDING_FEATURES, n
    covered = {n for ns in named.values() for n in ns}
    extra = [f for f in FEATURES if f not in covered]
    assert extra == ["sum_sync_cost"]
    assert FEATURE_SOURCE["sum_sync_cost"] == "added"


def test_feature_order_and_sources_are_frozen():
    assert FEATURES == (
        "context_length", "density", "batch", "n_operations",
        "n_synchronisations", "sum_host", "sum_link", "sum_device",
        "pinned_bandwidth", "pageable_bandwidth", "host_to_device_ratio",
        "synchronisations_per_operation", "sum_sync_cost")
    assert set(FEATURE_SOURCE) == set(FEATURES)
    assert {n for n, s in FEATURE_SOURCE.items() if s != "plan"} == {
        "host_to_device_ratio", "sum_sync_cost"}
    assert not set(PENDING_FEATURES) & set(FEATURES)


def test_features_hand_computed():
    # K(h2,d3), S(item, 4 bytes: s0 2), C(h1,l1), closing synchronise (s0 1).
    # 4 operations, 2 synchronisations; sum h 3, l 1, d 3; h/(l+d) = 0.75;
    # syncs per op 0.5; sum s0 3.
    ops = close_prefill([kernel_op(2, 3), sync_op("item", payload_bytes=4),
                         copy_op(1, 1)])
    c = cell(ops=ops, L=16384, density=0.25, batch=4)
    assert features(c) == (16384, 0.25, 4, 4, 2, 3, 1, 3, 1024, 512, 0.75,
                           0.5, 3)


def test_host_to_device_ratio_with_no_device_time_is_refused():
    with pytest.raises(CostModelError, match="host_to_device_ratio"):
        features(cell(ops=[kernel_op(1, 0)]))


# --------------------------------------------------------------------------
# Fitting and the training rule
# --------------------------------------------------------------------------

ROWS = [row("P1", 1.0), row("P1", 4.0), row("P2", 2.0), row("P4", 3.0)]


def test_lock_has_not_happened():
    assert LOCKED_CONFIG is None


@pytest.mark.parametrize("target,expected", [
    ("seconds", 2.5),        # mean(1, 4)
    ("log_seconds", 2.0),    # exp(mean(log 1, log 4)) = exp(log 2)
])
def test_fit_on_a_committed_set_hand_computed(target, expected):
    ts = committed_training_set(ROWS, target="P2", banked=["P1"])
    m = fit_learned(ts, LearnedConfig(target=target), model_factory=MeanModel)
    assert m.predict([cell("P2")]) == [pytest.approx(expected)]
    assert m.kind == COMMITTED and m.trained_on == ("P1",) and m.n_rows == 2


def test_target_must_be_stated():
    ts = committed_training_set(ROWS, target="P2", banked=["P1"])
    with pytest.raises(CostModelError, match="target must be"):
        fit_learned(ts, LearnedConfig(target="ms"), model_factory=MeanModel)


def test_missing_lightgbm_is_a_clear_refusal(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(CostModelError, match="lightgbm is not installed"):
        lightgbm_factory({})


def test_lightgbm_fits_when_installed():
    pytest.importorskip("lightgbm")
    rows = [row("P1", 0.1 * k, ops=[kernel_op(k, k)]) for k in range(1, 30)]
    ts = committed_training_set(rows, target="P2", banked=["P1"])
    m = fit_learned(ts, LearnedConfig(target="log_seconds",
                                      params={"n_estimators": 5,
                                              "min_child_samples": 1,
                                              "verbose": -1}))
    (p,) = m.predict([cell("P2", ops=[kernel_op(5, 5)])])
    assert math.isfinite(p) and p > 0


# --------------------------------------------------------------------------
# Learning curve
# --------------------------------------------------------------------------

CURVE_ROWS = [row("T", 2.0), row("A", 1.0), row("B", 2.0), row("C", 4.0)]


def test_learning_curve_hand_computed():
    # Target T measures 2. MeanModel predicts the mean of the subset's rows.
    # size 1: A -> 1, |1-2|/2 = 0.5;  B -> 2, 0;  C -> 4, 1.
    # size 2: AB -> 1.5, 0.25;  AC -> 2.5, 0.25;  BC -> 3, 0.5.
    # size 3: ABC -> 7/3, (1/3)/2 = 1/6.
    pts = learning_curve(CURVE_ROWS, "T", [1, 2, 3],
                         LearnedConfig(target="seconds"),
                         model_factory=MeanModel)
    got = {(p.size, p.trained_on): p.median_abs_rel_error for p in pts}
    assert got == pytest.approx({
        (1, ("A",)): 0.5, (1, ("B",)): 0.0, (1, ("C",)): 1.0,
        (2, ("A", "B")): 0.25, (2, ("A", "C")): 0.25, (2, ("B", "C")): 0.5,
        (3, ("A", "B", "C")): 1 / 6})
    assert all(p.kind == RETROSPECTIVE and p.n_target_rows == 1 for p in pts)


def test_learning_curve_refuses_sizes_the_pool_cannot_supply():
    # "1 to 6" needs six other pairings; here there are three.
    with pytest.raises(CostModelError, match=r"\[4, 6\] are impossible"):
        learning_curve(CURVE_ROWS, "T", [1, 4, 6],
                       LearnedConfig(target="seconds"), model_factory=MeanModel)
    with pytest.raises(CostModelError, match="no ground-truth rows"):
        learning_curve(CURVE_ROWS, "P9", [1], LearnedConfig(target="seconds"),
                       model_factory=MeanModel)
