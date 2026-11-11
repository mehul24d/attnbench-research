"""Reader A's reproductions (T4.3 read A, 23f540e; imported 11 Nov 2026 unchanged except paths). Each test ASSERTS THE PLAN'S BEHAVIOUR, so a
test that FAILS here is a reproduced defect. Run in the scratch copy only."""
from __future__ import annotations

import math

import pytest

from attnbench.analysis import block_drift as BD
from attnbench.analysis import gate_a as GA
from attnbench.analysis import gate_a_repeats as GR
from attnbench.components import replay
from attnbench.predict import dispatcher as D
from attnbench.predict import crossover as X
from attnbench.predict import intervals as I
from attnbench.predict.dispatcher import Config, Point, build_points, evaluate, h6_line
from attnbench.predict.training import CellKey

M = "m"
S = Config("a", 0.5)
DENSE = Config("dense", 1.0)


def key(L, c, b=1):
    return CellKey(M, "cp", c.arm, c.density, L, b)


def kern(lengths_batches):
    return {key(L, c, b): v for (L, b) in lengths_batches
            for c, v in {DENSE: 40.0, S: 20.0}.items() if not (b > 1 and c != DENSE)}


def grid(lengths_batches, pred, kern=None):
    out = {}
    for (L, b) in lengths_batches:
        for c, v in pred.items():
            if b > 1 and c != DENSE:
                continue            # plan: sparse arms are batch 1 only
            out[key(L, c, b)] = v
    return out


# ---------------------------------------------------------------- F1: batch 4 is scored
def test_F1_batch_4_is_not_scored_for_h6():
    """Section 4 H6 / 5.9 / App. A 1 Nov: "at batch 4 every rule chooses dense, so
    batch 4 is not scored"."""
    pred = {DENSE: 100.0, S: 80.0}
    meas = {DENSE: 100.0, S: 80.0}
    pts = build_points(grid([(16384, 1), (16384, 4)], pred), dense_arms={"dense"},
                       kernel=kern([(16384, 1), (16384, 4)]))
    measured = {Point(M, 16384, 1): meas, Point(M, 16384, 4): {DENSE: 400.0}}
    out = evaluate(pts, measured, {"u": {S}}, 5.0)
    sc = out["u"]["always_dense"]
    # always-dense at the batch-1 point has regret 0.25; at batch 4 it is 0.
    # Batch 4 must not be in the mean: expected n_scored == 1, mean regret 0.25.
    assert sc.n_scored == 1, f"n_scored={sc.n_scored}: the batch-4 point is scored"


# ------------------------------------------------- F2: realised-token keys never match
def test_F2_h6_lengths_filter_matches_realised_token_keys():
    """App. A 9 Nov: prediction cells are keyed by REALISED token counts (T3.2a-d:
    1951..32700). H6_LENGTHS is the budget labels (16384, 32768)."""
    pred = {DENSE: 100.0, S: 80.0}
    pts = build_points(grid([(16370, 1), (32700, 1)], pred), dense_arms={"dense"},
                       kernel=kern([(16370, 1), (32700, 1)]))
    measured = {Point(M, 16370, 1): {DENSE: 100.0, S: 80.0},
                Point(M, 32700, 1): {DENSE: 400.0, S: 300.0}}
    out = evaluate(pts, measured, {"u": {S}}, 5.0)
    d = out["u"]["dispatcher"]
    assert d.n_scored == 2, f"scored {d.n_scored} points; unscored listed: {len(d.unscored)}"


def test_F2b_empty_score_reads_as_no_more_decision_errors():
    pred = {DENSE: 100.0, S: 80.0}
    pts = build_points(grid([(16370, 1)], pred, None), dense_arms={"dense"},
                       kernel={key(16370, DENSE): 40.0, key(16370, S): 20.0})
    measured = {Point(M, 16370, 1): {DENSE: 100.0, S: 130.0}}   # sparse is slower: rule errs
    out = evaluate(pts, measured, {"u": {S}}, 5.0)
    line = h6_line(out["u"]["dispatcher"], out["u"]["kernel_speedup_rule"])
    assert out["u"]["kernel_speedup_rule"].n_scored == 0
    assert line.no_more_decision_errors is not True or out["u"]["dispatcher"].n_scored > 0, \
        "H6 line says 'no more decision errors' over ZERO scored points"


# --------------------------------------- F3: one scalar floor for every (length, batch)
def test_F3_dispatcher_cannot_apply_a_per_length_floor():
    """App. A 8 Nov: 'the resolution floor is per (context length, batch)'. 16384 floor
    13.77 ms, 32768 floor 101.16 ms (9 Nov). A predicted 50 ms gain is resolvable at
    16384 and not at 32768."""
    p16 = {DENSE: 1.000, S: 0.950}
    p32 = {DENSE: 4.000, S: 3.950}
    pts = build_points({**grid([(16384, 1)], p16), **grid([(32768, 1)], p32)}, dense_arms={"dense"})
    for floor in (0.01377, 0.10116):
        ch = D.choices_of("dispatcher", pts, frozenset({S}), floor)
        got = (ch[Point(M, 16384, 1)] == S, ch[Point(M, 32768, 1)] == S)
        if got == (True, False):
            return
    pytest.fail("no single floor gives sparse@16384 and dense@32768 for the same 50 ms gain; "
                "evaluate/dispatcher/DispatchSpec take one scalar")


def test_F3b_crossover_takes_one_scalar_floor():
    import inspect
    sig = inspect.signature(X.crossover_brackets)
    assert "floors" in sig.parameters or "floor_of" in sig.parameters, \
        f"crossover_brackets takes {list(sig.parameters)}: one floor for every grid length"


# --------------------------------- F4: policies compared over different scored points
def test_F4_h6_line_compares_policies_over_the_same_points():
    pred = {DENSE: 100.0, S: 80.0, Config("b", 0.25): 70.0}
    pts = build_points(grid([(16384, 1), (32768, 1)], pred), dense_arms={"dense"},
                       kernel={key(L, c): v for L in (16384, 32768)
                               for c, v in {DENSE: 40.0, S: 20.0, Config("b", 0.25): 50.0}.items()})
    B = Config("b", 0.25)
    # ground truth: b (the dispatcher's pick) was NOT measured at 32768, where it would be awful
    measured = {Point(M, 16384, 1): {DENSE: 100.0, S: 80.0, B: 80.0},
                Point(M, 32768, 1): {DENSE: 100.0, S: 80.0}}
    out = evaluate(pts, measured, {"u": {S, B}}, 5.0)
    d, r = out["u"]["dispatcher"], out["u"]["kernel_speedup_rule"]
    assert d.n_scored == r.n_scored, f"dispatcher scored {d.n_scored} points, rule {r.n_scored}"


# ------------------------------------------------------------- F5: block drift
def _rows(medians, anchor_blocks=None):
    rows = []
    for b, m in medians.items():
        if anchor_blocks is None or b in anchor_blocks:
            rows.append({"block": b, "anchor": True, "arm": "dense", "nominal_density": 1.0,
                         "context_length": 8192, "batch": 1, "median_ms": m})
        rows.append({"block": b, "anchor": False, "arm": "oracle", "nominal_density": 0.5,
                     "context_length": 16384, "batch": 1, "median_ms": 2 * m})
    return rows


def test_F5a_a_block_without_an_anchor_row_is_not_silently_passed():
    rows = _rows({"b1": 100.0, "b2": 101.0, "b3": 130.0}, anchor_blocks={"b1", "b2"})
    dr = BD.drift(rows, order=["b1", "b2", "b3"])
    assert dr["passed"] is not True, f"b3 has no anchor row and drift says passed={dr['passed']}"


def test_F5b_nan_anchor_median_is_not_a_pass():
    rows = _rows({"b1": 100.0, "b2": float("nan")})
    dr = BD.drift(rows, order=["b1", "b2"])
    assert dr["passed"] is not True, f"a NaN anchor median passes: {dr}"


def test_F5c_reference_block_can_be_the_odd_one_out():
    # b1 is 8% away from both b2 and b3, which agree. The sensitivity run should drop b1.
    rows = _rows({"b1": 100.0, "b2": 108.0, "b3": 108.0})
    dr = BD.drift(rows, order=["b1", "b2", "b3"])
    assert "b1" in dr["above"], f"flagged {dr['above']}; the odd block b1 is the reference and cannot be flagged"


def test_F5d_nothing_calls_drift():
    import pathlib, re
    hits = []
    root = pathlib.Path(__file__).resolve().parent.parent
    for p in list((root / "attnbench").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        if p.name == "block_drift.py":
            continue
        if re.search(r"^\s*(from\s+\S*block_drift\s+import|import\s+\S*block_drift|from\s+\S+\s+import\s+.*\bblock_drift\b)", p.read_text(), re.M):
            hits.append(str(p))
    assert hits, "no module outside block_drift.py imports it: the flag is produced by nothing"


# ---------------------------------------------------- F6: check_passes degenerate inputs
def test_F6a_check_passes_nan_difference():
    rep = {"families": {"mm": {"n": 3, "trace_s": 1.0, "replay_s": float("nan"),
                               "relative_difference": float("nan")}},
           "positions_without_replay": [], "kernel_name_mismatch": None}
    ok, why = replay.check_passes(rep, 0.25)
    assert not ok, f"a NaN relative difference passes Gate A's family check: ok={ok}"


# ------------------------------------------------------- F7: Gate A v2 leg 3 in as_written
def test_F7_as_written_reads_a_v2_leg3():
    def rep(within):
        return {"predictors": {"all": {"recursion_below": {"sum": True, "step_form": True}}},
                "families": {"bound": 0.25, "bound_provisional": False, "failing_cells": {}},
                "slopes": {"status": "measured", "range": [0.707, 1.414],
                           "within_range": within, "missing": [], "slopes": {}},
                "prediction_sha256": "x"}
    ok = GR.as_written({"a": rep({"item": True}), "b": rep({"item": True})})
    bad = GR.as_written({"a": rep({"item": False}), "b": rep({"item": True})})
    assert ok["legs"]["leg3"]["verdict"] == GR.HOLDS, ok["legs"]["leg3"]
    assert bad["legs"]["leg3"]["verdict"] == GR.FAILS, bad["legs"]["leg3"]


def test_F7b_vacuous_v2_slope_leg():
    class Sl:
        slope = 5.0
    out = GA.slopes({"item": Sl()}, bound=None, mechanisms=(), slope_range=GA.V2_SLOPE_RANGE)
    assert out["within_range"], "mechanisms=() gives within_range == {} : all([]) is True"


# ------------------------------------------------------ F8: short-group rule bypass
def test_F8_missing_clean_rounds_is_not_complete_enough():
    assert GA.short_of_rounds({"traced": False, "complete": True, "rows": []}), \
        "a group with no clean_rounds field is treated as having 10 or more"


# -------------------------------------------- F12: a one-replicate bootstrap is 'determined'
def test_F12_degenerate_bootstrap_interval_decides_a_sign():
    bs = I.bootstrap({"x": [1.0]}, lambda v: {"d": v["x"] - 0.5}, n_boot=1, seed=0)
    iv = bs.difference("d", "d", 0.95) if False else bs.interval("d", 0.95)
    assert iv.lo < iv.hi, f"one repeat, one replicate: interval [{iv.lo}, {iv.hi}] has zero width"


# --------------------------- F6b: Gate A's leg 2 feeds check_passes no kernel-name evidence
def test_F6b_gate_a_report_carries_kernel_name_evidence():
    import copy, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import test_gate_a as TG
    seq, trace_s = TG._seq_and_trace()
    rep = GA.replay_report(seq, trace_s, [{"device": trace_s.get(i, 0.0)} for i in range(len(seq.ops))])
    # Section 5.6 'Replay mismatch': replayed kernel names AND durations are compared with the trace.
    assert rep["kernel_name_mismatch"] is not None, \
        "replay_report passes no Match, so check_passes' kernel-name/structure guard is never fed"


# ------------------------- F13: leg 1 HOLDS on any n; its verdict carries no cell count
def test_F13_leg1_verdict_carries_the_number_of_cells_read():
    def rep(n):
        return {"predictors": {"all": {"by_predictor": {"recursion": {"n": n}},
                                       "recursion_below": {"sum": True, "step_form": True}}},
                "families": {"bound": 0.25, "bound_provisional": False, "failing_cells": {}},
                "slopes": {"status": "not measured"}, "prediction_sha256": "x"}
    out = GR._leg1({"a": rep(1), "b": rep(1)})
    assert out["verdict"] != GR.HOLDS or any(k in out for k in ("n", "n_cells", "cells")), \
        f"leg 1 HOLDS on one cell per session and the verdict dict records no n: {out}"
