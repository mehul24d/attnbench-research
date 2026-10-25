"""T2.19: bootstrap prediction intervals.

The resampling protocol is re-implemented here from the module's docstring
alone, so a change to the protocol that leaves old seeds unreproducible
fails. Coverage is checked by simulation against a known distribution.
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from attnbench.predict.intervals import (
    Interval, bootstrap, percentile_interval)
from attnbench.predict.recursion import (
    CostModelError, MechanismCost, close_prefill, kernel_op, predict_prefill,
    sync_op)


def test_one_repeat_per_measurement_gives_a_zero_width_interval():
    b = bootstrap({"h": [2.0], "d": [3.0]},
                  lambda v: {"t": v["h"] + v["d"]}, n_boot=50, seed=1)
    iv = b.interval("t", 0.9)
    assert (iv.point, iv.lo, iv.hi) == (5.0, 5.0, 5.0)


def test_the_point_is_the_full_sample_statistic_not_the_replicate_mean():
    b = bootstrap({"x": [1.0, 2.0, 10.0]}, lambda v: {"t": v["x"]},
                  n_boot=200, seed=3)
    assert b.point["t"] == 2.0                     # the median
    assert statistics.mean(b.replicates["t"]) != 2.0


def _protocol(samples, f, n_boot, seed, stat=statistics.median):
    rng = np.random.Generator(np.random.PCG64(seed))
    out = []
    for _ in range(n_boot):
        v = {}
        for k in sorted(samples):
            x = np.asarray(samples[k], dtype=float)
            v[k] = float(stat(x[rng.integers(0, x.size, size=x.size)]))
        out.append(f(v))
    return out


def test_replicates_follow_the_documented_protocol():
    samples = {"z": [5.0, 1.0, 3.0, 2.0], "a": [0.5, 0.25, 1.0]}
    f = lambda v: v["a"] * 10 + v["z"]
    b = bootstrap(samples, lambda v: {"t": f(v)}, n_boot=64, seed=12)
    assert list(b.replicates["t"]) == _protocol(samples, f, 64, 12)
    lo, hi = np.quantile(_protocol(samples, f, 64, 12), [0.05, 0.95])
    iv = b.interval("t", 0.9)
    assert (iv.lo, iv.hi) == (lo, hi)


def test_the_seed_reproduces_and_a_new_seed_differs():
    s = {"x": list(range(1, 20))}
    f = lambda v: {"t": float(v["x"])}
    assert bootstrap(s, f, n_boot=30, seed=5) == bootstrap(s, f, n_boot=30, seed=5)
    assert bootstrap(s, f, n_boot=30, seed=5).replicates != \
        bootstrap(s, f, n_boot=30, seed=6).replicates


def test_outputs_share_each_replicates_components():
    # a = x + y and b = x share x, so a - b is y in every replicate, and the
    # interval of the difference is the interval of y alone.
    rng = np.random.default_rng(0)
    s = {"x": rng.normal(10, 3, 25).tolist(), "y": rng.normal(1, 0.1, 25).tolist()}
    b = bootstrap(s, lambda v: {"a": v["x"] + v["y"], "b": v["x"], "y": v["y"]},
                  n_boot=300, seed=2)
    diff, y = b.difference("a", "b", 0.95), b.interval("y", 0.95)
    assert diff.lo == pytest.approx(y.lo) and diff.hi == pytest.approx(y.hi)
    # Far narrower than if a and b had been resampled apart.
    assert diff.hi - diff.lo < (b.interval("b", 0.95).hi - b.interval("b", 0.95).lo) / 5


def test_percentile_interval_coverage_is_near_nominal():
    # The mean of 30 normal draws, a 90% interval, 200 trials. The percentile
    # bootstrap of a mean under-covers slightly at this n; the band allows it.
    rng = np.random.default_rng(7)
    hits = 0
    trials = 200
    for t in range(trials):
        x = rng.normal(5.0, 2.0, 30).tolist()
        b = bootstrap({"x": x}, lambda v: {"m": v["x"]}, n_boot=300, seed=t,
                      statistic=lambda a: float(np.mean(a)))
        iv = b.interval("m", 0.9)
        hits += iv.lo <= 5.0 <= iv.hi
    assert 0.82 <= hits / trials <= 0.96


def test_a_recursion_prediction_bootstrapped_from_its_components():
    # h and d of one kernel resampled; s_fixed of the closing sync fixed.
    costs = {"synchronize": MechanismCost(s_fixed=1)}

    def f(v):
        ops = close_prefill([kernel_op(v["h"], v["d"])])
        return {"t": predict_prefill(ops, costs, None)}

    b = bootstrap({"h": [1.0, 1.0, 1.0], "d": [2.0, 3.0, 4.0]}, f, n_boot=400,
                  seed=0)
    assert b.point["t"] == 1 + 3 + 1
    iv = b.interval("t", 0.95)
    assert 4.0 <= iv.lo <= iv.point <= iv.hi <= 6.0
    assert set(b.replicates["t"]) <= {4.0, 5.0, 6.0}


def test_interval_dict_round_trip():
    iv = percentile_interval(1.0, [0.5, 1.0, 1.5], 0.5, n_boot=3, seed=9)
    assert Interval.from_dict(iv.to_dict()) == iv
    assert iv.method == "percentile"


@pytest.mark.parametrize("kw, match", [
    ({"n_boot": 0, "seed": 1}, "n_boot"), ({"n_boot": True, "seed": 1}, "n_boot"),
    ({"n_boot": 5, "seed": -1}, "seed"), ({"n_boot": 5, "seed": 1.0}, "seed")])
def test_bootstrap_arguments_are_checked(kw, match):
    with pytest.raises(CostModelError, match=match):
        bootstrap({"x": [1.0]}, lambda v: {"t": v["x"]}, **kw)


def test_bad_measurements_and_outputs_are_refused():
    with pytest.raises(CostModelError, match="at least one finite"):
        bootstrap({"x": []}, lambda v: {"t": 1.0}, n_boot=2, seed=0)
    with pytest.raises(CostModelError, match="at least one finite"):
        bootstrap({"x": [1.0, float("nan")]}, lambda v: {"t": 1.0}, n_boot=2, seed=0)
    with pytest.raises(CostModelError, match="not a finite number"):
        bootstrap({"x": [1.0]}, lambda v: {"t": float("inf")}, n_boot=2, seed=0)
    calls = []

    def changing(v):
        calls.append(1)
        return {"t": 1.0} if len(calls) == 1 else {"u": 1.0}

    with pytest.raises(CostModelError, match="same outputs"):
        bootstrap({"x": [1.0]}, changing, n_boot=2, seed=0)


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1, True])
def test_level_must_be_strictly_between_zero_and_one(level):
    with pytest.raises(CostModelError, match="level"):
        percentile_interval(1.0, [1.0], level, n_boot=1, seed=0)


def test_sync_costs_can_be_resampled_too():
    def f(v):
        costs = {"synchronize": MechanismCost(s_fixed=v["s"]),
                 "item": MechanismCost(s_fixed=v["s"])}
        return {"t": predict_prefill(close_prefill([kernel_op(1, 1),
                                                    sync_op("item")]), costs, 4)}
    # K: H 1, D 2; item: max(1, 2) = 2, + s; closing: + s.  t = 2 + 2s.
    b = bootstrap({"s": [0.5, 1.0, 1.5]}, f, n_boot=100, seed=4)
    assert b.point["t"] == 4.0
    assert set(b.replicates["t"]) <= {3.0, 4.0, 5.0}
