"""T2.10: the learned baseline's frozen features, its training rule, the
learning curve and the pinned LightGBM.

The hand-computable fits use a stub model; the LightGBM tests need the
optional `learned` extra and skip without it. The feature list is checked
against the plan's own sentence, not against `FEATURES`.
"""

from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path

import pytest

from attnbench.predict.intensity import KERNEL_CLASSES, ModelShape
from attnbench.predict.learned import (
    AI_FEATURES, FEATURE_SOURCE, FEATURES, LIGHTGBM_VERSION, LOCKED_CONFIG,
    PENDING_FEATURES, LearnedConfig, features, fit_learned, learning_curve,
    lightgbm_factory)
from attnbench.predict.recursion import (
    CostModelError, close_prefill, copy_op, kernel_op, price, sync_op)
from attnbench.predict.training import (
    COMMITTED, RETROSPECTIVE, Cell, GroundTruthRow, HardwareFeatures,
    committed_training_set)

from test_predict_recursion import COSTS

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs" / "RESEARCH_PLAN.md"
HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)
# The toy shape of test_predict_hdbi_intensity, whose intensities are
# worked by hand there.
TOY = ModelShape(hidden=4, n_heads_q=2, n_heads_kv=1, head_dim=2,
                 intermediate=8, vocab=10)


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


def cell(pairing="P1", ops=None, L=4, density=0.5, batch=1, arm="xattn",
         model=TOY):
    ops = ops or [kernel_op(1, 1)]
    return Cell(pairing=pairing, arm=arm, context_length=L, density=density,
                batch=batch, priced=price(ops, COSTS), hardware=HW, model=model)


def row(pairing, measured, statistic="median", **kw):
    return GroundTruthRow(cell=cell(pairing, **kw), measured=measured,
                          statistic=statistic)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _learned_bullet():
    text = PLAN.read_text()
    m = re.search(r"- \*\*Learned \(CTFusion-style LightGBM\):\*\*(.*)", text)
    assert m, "the plan's learned-baseline bullet moved"
    return m.group(1)


def test_every_feature_the_plan_names_is_frozen():
    bullet = _learned_bullet()
    named = {
        "context length": ["context_length"],
        "density": ["density"],
        "batch": ["batch"],
        "operation and synchronisation counts": ["n_operations",
                                                 "n_synchronisations"],
        "host, link, device costs": ["sum_host", "sum_link", "sum_device"],
        "pinned and pageable bandwidth": ["pinned_bandwidth",
                                          "pageable_bandwidth"],
        "arithmetic intensity per operation class": list(AI_FEATURES),
        "host-to-device time ratio": ["host_to_device_ratio"],
        "transfer demand relative to link bandwidth": ["transfer_demand"],
        "synchronisations per operation": ["synchronisations_per_operation"],
    }
    for phrase, names in named.items():
        assert phrase in bullet, f"the plan no longer names {phrase!r}"
        for n in names:
            assert n in FEATURES, n
    covered = {n for ns in named.values() for n in ns}
    assert [f for f in FEATURES if f not in covered] == ["sum_sync_cost"]


def test_feature_order_and_sources_are_frozen():
    assert FEATURES == (
        "context_length", "density", "batch", "n_operations",
        "n_synchronisations", "sum_host", "sum_link", "sum_device",
        "pinned_bandwidth", "pageable_bandwidth", "host_to_device_ratio",
        "synchronisations_per_operation", "sum_sync_cost", "transfer_demand",
        "ai_qkv_projection", "ai_attention", "ai_output_projection", "ai_mlp",
        "ai_norm", "ai_lm_head")
    assert set(FEATURE_SOURCE) == set(FEATURES)
    assert {n for n, s in FEATURE_SOURCE.items() if s == "decided"} == {
        "host_to_device_ratio", "sum_sync_cost"}
    assert {n for n, s in FEATURE_SOURCE.items() if s == "proposed"} == {
        "transfer_demand", *AI_FEATURES}
    assert set(PENDING_FEATURES) == {"ai_estimator_scoring"}


def test_features_hand_computed():
    # K(h2,d3), C(h1,l1, 2048 bytes), S(item, 4 bytes: s0 2), closing
    # synchronise (s0 1).  4 operations, 2 synchronisations; sum h 3, l 1,
    # d 3; h/(l+d) = 0.75; syncs per op 0.5; sum s0 3;
    # transfer demand (2048 + 4) / 1024 = 2.00390625 s.
    # Intensities at L = 4, B = 1, density 0.5 for the toy shape
    # (test_predict_hdbi_intensity): 256/160, 64/96, 128/96, 832/480,
    # 64/72, 80/108.
    ops = close_prefill([kernel_op(2, 3), copy_op(1, 1, payload_bytes=2048),
                         sync_op("item", payload_bytes=4)])
    got = features(cell(ops=ops, L=4, density=0.5, batch=1))
    assert got[:14] == (4, 0.5, 1, 4, 2, 3, 1, 3, 1024, 512, 0.75, 0.5, 3,
                        2.00390625)
    assert got[14:] == pytest.approx((256 / 160, 64 / 96, 128 / 96, 832 / 480,
                                      64 / 72, 80 / 108))
    assert len(KERNEL_CLASSES) == len(got) - 14


def test_feature_refusals():
    with pytest.raises(CostModelError, match="host_to_device_ratio"):
        features(cell(ops=[kernel_op(1, 0)]))
    with pytest.raises(CostModelError, match="no model shape"):
        features(cell(model=None))


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
    assert m.statistic == "median"


def test_target_must_be_stated():
    ts = committed_training_set(ROWS, target="P2", banked=["P1"])
    with pytest.raises(CostModelError, match="target must be"):
        fit_learned(ts, LearnedConfig(target="ms"), model_factory=MeanModel)


# --------------------------------------------------------------------------
# LightGBM, pinned and on one thread
# --------------------------------------------------------------------------

def test_the_pin_matches_the_optional_extra():
    text = (REPO / "pyproject.toml").read_text()
    m = re.search(r'^learned = \[\s*"lightgbm==([^"]+)",\s*\]', text, re.M)
    assert m, "pyproject.toml has no learned extra pinning lightgbm"
    assert m.group(1) == LIGHTGBM_VERSION


def test_missing_lightgbm_is_a_clear_refusal(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(CostModelError, match=r"\[learned\]"):
        lightgbm_factory({})


def test_another_lightgbm_version_is_refused(monkeypatch):
    fake = types.SimpleNamespace(__version__="9.9.9", LGBMRegressor=None)
    monkeypatch.setitem(sys.modules, "lightgbm", fake)
    with pytest.raises(CostModelError, match="pinned to"):
        lightgbm_factory({})


def test_more_than_one_thread_is_refused(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(__version__=LIGHTGBM_VERSION,
                                 LGBMRegressor=lambda **kw: calls.append(kw))
    monkeypatch.setitem(sys.modules, "lightgbm", fake)
    for key in ("num_threads", "n_jobs"):
        with pytest.raises(CostModelError, match="one thread"):
            lightgbm_factory({key: 4})
    lightgbm_factory({"n_jobs": 1, "n_estimators": 7})
    assert calls == [{"num_threads": 1, "n_estimators": 7}]


def test_lightgbm_fits_after_torch_on_one_thread():
    pytest.importorskip("lightgbm")
    import torch  # noqa: F401  -- the order that crashes a threaded fit
    rows = [row("P1", 0.1 * k, ops=[kernel_op(k, k)]) for k in range(1, 30)]
    ts = committed_training_set(rows, target="P2", banked=["P1"])
    m = fit_learned(ts, LearnedConfig(target="log_seconds",
                                      params={"n_estimators": 5,
                                              "min_child_samples": 1,
                                              "verbose": -1}))
    assert m.model.get_params()["num_threads"] == 1
    (p,) = m.predict([cell("P2", ops=[kernel_op(5, 5)])])
    assert math.isfinite(p) and p > 0


# --------------------------------------------------------------------------
# Learning curve: sizes 1 to N - 1
# --------------------------------------------------------------------------

CURVE_ROWS = [row("T", 2.0), row("A", 1.0), row("B", 2.0), row("C", 4.0)]


def test_learning_curve_hand_computed():
    # N = 4 pairings, so sizes 1 to 3. Target T measures 2; MeanModel
    # predicts the mean of the subset's rows.
    # size 1: A -> 1, |1-2|/2 = 0.5;  B -> 2, 0;  C -> 4, 1.
    # size 2: AB -> 1.5, 0.25;  AC -> 2.5, 0.25;  BC -> 3, 0.5.
    # size 3: ABC -> 7/3, (1/3)/2 = 1/6.
    pts = learning_curve(CURVE_ROWS, "T", LearnedConfig(target="seconds"),
                         model_factory=MeanModel)
    got = {(p.size, p.trained_on): p.median_abs_rel_error for p in pts}
    assert got == pytest.approx({
        (1, ("A",)): 0.5, (1, ("B",)): 0.0, (1, ("C",)): 1.0,
        (2, ("A", "B")): 0.25, (2, ("A", "C")): 0.25, (2, ("B", "C")): 0.5,
        (3, ("A", "B", "C")): 1 / 6})
    assert all(p.kind == RETROSPECTIVE and p.n_target_rows == 1 for p in pts)


def test_learning_curve_size_follows_the_pairings_available():
    # Two pairings: only size 1.
    pts = learning_curve(CURVE_ROWS[:2], "T", LearnedConfig(target="seconds"),
                         model_factory=MeanModel)
    assert [(p.size, p.trained_on) for p in pts] == [(1, ("A",))]


def test_learning_curve_refusals():
    with pytest.raises(CostModelError, match="no ground-truth rows"):
        learning_curve(CURVE_ROWS, "P9", LearnedConfig(target="seconds"),
                       model_factory=MeanModel)
    mixed = CURVE_ROWS + [row("A", 1.5, statistic="p90")]
    with pytest.raises(CostModelError, match="mix ground-truth statistics"):
        learning_curve(mixed, "T", LearnedConfig(target="seconds"),
                       model_factory=MeanModel)
    # The target's truth as p90 against medians everywhere else: every
    # training subset is uniform, so only the check over all rows sees it.
    odd_target = [row("T", 2.0, statistic="p90")] + CURVE_ROWS[1:]
    with pytest.raises(CostModelError, match="mix ground-truth statistics"):
        learning_curve(odd_target, "T", LearnedConfig(target="seconds"),
                       model_factory=MeanModel)
