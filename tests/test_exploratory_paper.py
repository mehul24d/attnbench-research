"""EXPLORATORY / POST HOC paper-quality tables (28 Nov 2026 wave, items 2-5):
every function behind a reported number, on fixtures worked out by hand in
the comments. Costs are small integers, so most comparisons are exact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from attnbench.exploratory import paper as PP
from attnbench.predict.recursion import (MechanismCost, close_prefill, kernel_op, price,
                                         recurse, sync_op)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_tables as PT  # noqa: E402

COSTS = {"synchronize": MechanismCost(s_fixed=1), "item": MechanismCost(s_fixed=2)}


def priced(ops):
    return price(close_prefill(ops), COSTS)


# A: kernels (h, d) = (1, 5), (1, 5), (3, 1), the closing synchronise (s_0 1),
# no queue. H 1, 2, 5; D 6, 11, 12. The close waits 5 -> 12 (7), + 1: 13.
A = [kernel_op(1, 5), kernel_op(1, 5), kernel_op(3, 1)]

# S: kernel (1, 10), an `.item()` (s_0 2), kernel (1, 1), the close.
# Full: k0 H 1, D 11. item: waits 1 -> 11 (10), + 2: H 13. k2: H 14, D 15.
#       close: waits 14 -> 15 (1), + 1: 16.
# No wait: item adds 2 at H 1: 3. k2: H 4, D = max(11, 4) + 1 = 12.
#       close: waits 4 -> 12 (8), + 1: 13.
# No s_0 (every s_0 0, the close's too; waits kept): item H 11. k2 H 12,
#       D 13. close waits 12 -> 13 (1), + 0: 13.
# No sync (s_0 0, no arm wait): item H 1. k2 H 2, D 12. close waits
#       2 -> 12 (10), + 0: 12.
# Host only (d = 0): k0 H 1, item H 3, k2 H 4, close H 5.
S = [kernel_op(1, 10), sync_op("item"), kernel_op(1, 1)]


def test_knockouts_remove_one_term_each():
    ko = PP.knockouts(priced(S), None)
    assert ko == {"recursion": 16, "no_queue": 16, "no_fixed_sync": 13, "no_wait": 13,
                  "no_sync": 12, "host_only": 5, "device_only": 11}


def test_knockouts_recursion_is_the_committed_recursion():
    p = priced(A)
    assert PP.knockouts(p, 1)["recursion"] == recurse(p, 1).end_to_end


def test_knockouts_no_queue_differs_where_the_queue_binds():
    # Q = 1, kernels (1, 5) twice: k1 waits for D0 = 6 (stall 4), D 11; the
    # close waits for D1 (stall 5), + 1: 12. Without Q: H 1, 2; D 6, 11; 12.
    # Same end time here, so the queue term moves nothing at Q = 1.
    ko = PP.knockouts(priced([kernel_op(1, 5), kernel_op(1, 5)]), 1)
    assert ko["recursion"] == 12 and ko["no_queue"] == 12
    # Kernels (3, 1), (1, 1) with Q = 1: k1 at H 4 waits for D0 = 4 (none);
    # host-bound, so no stall either way.
    ko = PP.knockouts(priced([kernel_op(3, 1), kernel_op(1, 1)]), 1)
    assert ko["recursion"] == ko["no_queue"]


def test_rel_and_summarise():
    assert PP.rel(13, 12) == pytest.approx(1 / 12)
    assert PP.rel(6, 12) == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        PP.rel(1, 0)
    s = PP.summarise([0.1, -0.3, 0.2])
    assert s == {"n": 3, "median_signed": pytest.approx(0.1),
                 "median_abs": pytest.approx(0.2), "max_abs": pytest.approx(0.3)}
    assert PP.summarise([])["median_abs"] is None


def test_arm_class_is_structural():
    assert PP.arm_class(priced(A), "dense") == "dense"
    assert PP.arm_class(priced(A), "meanpool") == "sync_free"     # only the close
    assert PP.arm_class(priced(S), "meanpool") == "synchronising"


def test_miss_split_is_exact():
    # A: predicted 13, chain 11, predicted idle 2. Measured 12 with busy 10:
    # measured idle 2. Error 1 = device part (11 - 10) 1 + idle part (2 - 2) 0.
    m = PP.miss_split(priced(A), None, 12.0, 10.0)
    assert (m["host"], m["fixed_sync"], m["queue"], m["sync_wait"]) == (5, 1, 0, 7)
    assert m["predicted"] == 13 and m["priced_device_chain"] == 11
    assert m["error_s"] == 1 and m["device_part_s"] == 1 and m["idle_part_s"] == 0
    # Measured 9 with busy 8: idle 1. Error 4 = (11 - 8) 3 + (2 - 1) 1.
    m = PP.miss_split(priced(A), None, 9.0, 8.0)
    assert (m["device_part_s"], m["idle_part_s"]) == (3, 1)
    assert m["rel"] == pytest.approx(4 / 9)


def test_miss_split_refuses_bad_inputs():
    with pytest.raises(ValueError):
        PP.miss_split(priced(A), None, 0.0, 1.0)
    with pytest.raises(ValueError):
        PP.miss_split(price([kernel_op(1, 5)], COSTS), None, 5.0, 5.0)   # no close


# B: kernels (4, 1) twice. With h times k (k >= 0.25): close at 8k + 2.
B = [kernel_op(4, 1), kernel_op(4, 1)]


def test_host_scaled():
    assert PP.host_scaled(priced(B), None, 1.0) == 10
    assert PP.host_scaled(priced(B), None, 0.5) == 6
    with pytest.raises(ValueError):
        PP.host_scaled(priced(B), None, -1)


def test_reprice_attention_scales_attention_positions_only():
    # Positions: attention (h 1, d 4), mm (1, 4), close. Ratio 0.5:
    # attention d 2. H 1, 2; D 3, 7; close waits 2 -> 7, + 1: 8 (was 10).
    p = priced([kernel_op(1, 4), kernel_op(1, 4)])
    fams = ["kernel:flash_attn_3::fwd", "kernel:aten::mm", "sync:synchronize"]
    rp, n = PP.reprice_attention(p, fams, 0.5)
    assert n == 1 and rp.device[:2] == (2, 4)
    assert recurse(rp, None).end_to_end == 8 and recurse(p, None).end_to_end == 10
    with pytest.raises(ValueError):
        PP.reprice_attention(p, fams[:2], 0.5)
    with pytest.raises(ValueError):
        PP.reprice_attention(p, fams, 0)


def test_median_interval_and_rel_interval():
    med, lo, hi = PP.median_interval([1.0, 1.0, 1.0, 1.0], n_boot=100, seed=0, level=0.95)
    assert (med, lo, hi) == (1.0, 1.0, 1.0)
    med, lo, hi = PP.median_interval([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=2000, seed=0,
                                     level=0.95)
    assert med == 3.0 and 1.0 <= lo <= 3.0 <= hi <= 5.0 and lo < hi
    # A larger measurement gives the smaller error: prediction 12 on [10, 15].
    assert PP.rel_interval(12, 10, 15) == pytest.approx((12 / 15 - 1, 12 / 10 - 1))
    with pytest.raises(ValueError):
        PP.median_interval([1.0], n_boot=10, seed=0, level=0.95)
    with pytest.raises(ValueError):
        PP.rel_interval(1, 0, 1)


def test_against_tolerance():
    assert PP.against_tolerance(-0.05, 0.09, 0.1) == "within"
    assert PP.against_tolerance(0.11, 0.2, 0.1) == "outside"
    assert PP.against_tolerance(-0.3, -0.2, 0.1) == "outside"
    assert PP.against_tolerance(0.05, 0.15, 0.1) == "straddles"
    assert PP.count_by([{"a": "x"}, {"a": "y"}, {"a": "x"}], "a") == {"x": 2, "y": 1}


def test_budget_of_maps_realised_lengths_to_grid_budgets():
    assert [PT.budget_of(n) for n in (1951, 4068, 8075, 16370, 32700)] == \
        [2048, 4096, 8192, 16384, 32768]
    with pytest.raises(ValueError):
        PT.budget_of(40000)


def test_fa3_ratios_reads_the_dense_selection_files(tmp_path):
    d = {"chosen": "fa3", "candidates": [{"name": "sdpa_flash", "median_ms": 2.0},
                                         {"name": "fa3", "median_ms": 0.8},
                                         {"name": "fa2", "median_ms": None}]}
    (tmp_path / "dense_choice_H100_16384_1.json").write_text(json.dumps(d))
    r = PT.fa3_ratios(tmp_path)
    assert r == {(16384, 1): {"fa3_ms": 0.8, "sdpa_flash_ms": 2.0, "ratio": 0.4,
                              "chosen": "fa3"}}
    d["candidates"] = d["candidates"][:1]
    (tmp_path / "dense_choice_H100_16384_1.json").write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        PT.fa3_ratios(tmp_path)


# --- scripts/paper_reproduce.py: the comparison it reports --------------------------

import paper_reproduce as RP  # noqa: E402


def test_reproduce_diff_paths_and_labels():
    a = {"x": 1, "y": [1, {"block": "b3", "v": 2.0}]}
    assert RP.diff_paths(a, a) == []
    assert RP.diff_paths(a, {"x": 2, "y": [1, {"block": "b3", "v": 2.0}]}) == ["/x"]
    b = {"x": 1, "y": [1, {"block": "p2_session", "v": 2.0}]}
    assert RP.diff_paths(a, b) == ["/y[1]/block"]
    assert RP.diff_paths(RP.strip_labels(a), RP.strip_labels(b)) == []
    assert RP.diff_paths({"x": [1]}, {"x": [1, 2]}) == ["/x (length 1 vs 2)"]
    assert RP.diff_paths({"x": 1}, {"z": 1}) == ["/x", "/z"]


def test_reproduce_repoints_the_banked_paths(tmp_path):
    old = "/old/mac/backups"
    run = tmp_path / "scoring_20261116" / "run"
    run.mkdir(parents=True)
    (tmp_path / RP.P2_SESSION).mkdir(parents=True)
    (tmp_path / "p4s").mkdir()
    (tmp_path / "p1_training").mkdir()
    (tmp_path / "p1_training" / "p1.json.gz").write_text("x")
    spec = {"bucket": "gs://b", "training_files": {"P1": f"{old}/p1_training/p1.json.gz"},
            "P2": {"local": "/purged/x.json", "sessions": ["/purged/s"]},
            "P4": {"local": None, "sessions": [f"{old}/p4s"]}}
    (run / "inputs.json").write_text(json.dumps(spec))
    got = json.loads(RP.repoint(tmp_path, tmp_path).read_text())
    assert got["training_files"]["P1"] == f"{tmp_path}/p1_training/p1.json.gz"
    assert got["P2"] == {"local": None, "sessions": [str(tmp_path / RP.P2_SESSION)]}
    assert got["P4"]["sessions"] == [f"{tmp_path}/p4s"]
    (tmp_path / "p4s").rmdir()
    with pytest.raises(SystemExit, match="missing"):
        RP.repoint(tmp_path, tmp_path)
