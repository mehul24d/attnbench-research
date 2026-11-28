"""EXPLORATORY host-price bias helpers (paper-quality wave, 28 Nov 2026,
item 10): hand-worked fixtures."""

import math

import pytest

from attnbench.exploratory import host_bias as HB

BS = dict(n_boot=2000, seed=0, level=0.95)


def test_op_classes():
    assert HB.op_class("aten::mm") == "gemm"
    assert HB.op_class("aten::_to_copy") == "copy"
    assert HB.op_class("aten::scalar_tensor") == "fill_and_create"
    assert HB.op_class(None) == "entry"
    assert HB.op_class("aten::unheard_of") == "other"
    every = [op for ops in HB.CLASSES.values() for op in ops]
    assert len(every) == len(set(every))


def test_host_parts_split_by_route():
    prof = {"host_s": {"0": 0.5, "1": 3.0, "2": 2.0, "3": 1.0},
            "residual_s": {"1": 1.0, "2": 0.5},
            "price_route": {"0": "isolated_replay", "1": "bare_leaf", "2": "bare_chain",
                            "3": "host_compute_null_pass"}}
    p = HB.host_parts(prof)
    assert p == {"total": 6.5, "bare": 3.5, "residual": 1.5, "other": 1.5}


def test_null_total_refuses_a_bare_profile():
    assert HB.null_total({"pricing": "null_pass", "host_s": {"1": 1.0, "2": 2.5}}) == 3.5
    with pytest.raises(ValueError):
        HB.null_total({"pricing": "bare_timer", "host_s": {"1": 1.0}})


def test_positions_skip_unpriced_synchronisations():
    # The call at 2 prices two positions, but 3 is a sync with no host price.
    bp = {"calls": [{"position": 1, "positions": 1, "operator": "aten::mm"},
                    {"position": 2, "positions": 2, "operator": "aten::index"}]}
    cls = HB.position_classes(bp, ["0", "1", "2", "4"])
    assert cls == {1: "gemm", 2: "mask_and_index", 4: "mask_and_index"}


def test_positions_refuse_a_call_past_the_end():
    bp = {"calls": [{"position": 1, "positions": 3, "operator": "aten::mm"}]}
    with pytest.raises(ValueError):
        HB.position_classes(bp, ["1", "2"])


def test_class_sums_like_for_like():
    bp = {"calls": [{"position": 1, "positions": 1, "operator": "aten::mm"},
                    {"position": 2, "positions": 1, "operator": "aten::add"},
                    {"position": 3, "positions": 1, "operator": "aten::mul"}]}
    null = {"host_s": {"1": 4.0, "2": 3.0, "3": 1.0}}
    com = {"host_s": {"0": 9.0, "1": 2.0, "2": 1.0, "3": 1.0}}
    s = HB.class_sums(bp, null, com)
    assert s == {"gemm": {"null_s": 4.0, "committed_s": 2.0, "positions": 1},
                 "elementwise": {"null_s": 4.0, "committed_s": 2.0, "positions": 2}}


def test_ratio_of_sums_point_and_interval():
    r = HB.ratio_of_sums([(2.0, 1.0), (6.0, 3.0)], **BS)
    assert r["ratio"] == 2.0 and r["lo"] == 2.0 and r["hi"] == 2.0 and r["clusters"] == 2
    r = HB.ratio_of_sums([(1.0, 1.0), (3.0, 1.0)], **BS)
    assert r["ratio"] == 2.0 and r["lo"] == 1.0 and r["hi"] == 3.0
    with pytest.raises(ValueError):
        HB.ratio_of_sums([], **BS)


def test_lower_bound_and_its_interval():
    lb = HB.lower_bound(3.0, 2.0, 1.5, 2.5)
    assert lb == {"ratio": 1.5, "lo": 3.0 / 2.5, "hi": 2.0}
    with pytest.raises(ValueError):
        HB.lower_bound(1.0, 2.0, 2.5, 1.5)


def test_layer_spans_and_null_by_layer():
    tr = {"traceEvents": [
        {"cat": "user_annotation", "name": "attnbench::prefill", "dur": 9e6},
        {"cat": "user_annotation", "name": "attnbench::layer.0", "dur": 2e6},
        {"cat": "user_annotation", "name": "attnbench::layer.1", "dur": 3e6},
        {"cat": "cpu_op", "name": "attnbench::layer.0", "dur": 1.0}]}
    assert HB.layer_spans(tr) == {0: 2.0, 1: 3.0}
    seq = {"ops": [{"layer": None}, {"layer": 0}, {"layer": 0}, {"layer": 1}]}
    prof = {"host_s": {"0": 5.0, "1": 1.0, "2": 2.0, "3": 4.0}}
    assert HB.null_by_layer(prof, seq) == {0: 3.0, 1: 4.0}
    tr["traceEvents"].append({"cat": "user_annotation", "name": "attnbench::layer.1", "dur": 1})
    with pytest.raises(ValueError):
        HB.layer_spans(tr)


def test_ratio_interval_is_deterministic():
    pairs = [(float(i + 1), 1.0) for i in range(10)]
    a, b = HB.ratio_of_sums(pairs, **BS), HB.ratio_of_sums(pairs, **BS)
    assert a == b and a["lo"] < a["ratio"] < a["hi"]
    assert math.isclose(a["ratio"], 5.5)
