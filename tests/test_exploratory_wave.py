"""EXPLORATORY wave of 16 Nov 2026 (item 6): every function behind a reported
number, on fixtures with answers worked out by hand in the comments.

Costs are small integers or powers of two so the comparisons are exact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exploratory_wave as EW  # noqa: E402
from attnbench.exploratory import bound as BD  # noqa: E402
from attnbench.exploratory import breakdown as BR  # noqa: E402
from attnbench.exploratory import safe_dispatch as SD  # noqa: E402
from attnbench.exploratory import sign as SG  # noqa: E402
from attnbench.exploratory import traces as TR  # noqa: E402
from attnbench.lock.floors import FloorTable  # noqa: E402
from attnbench.predict import dispatcher as DP  # noqa: E402
from attnbench.predict import intensity as IN  # noqa: E402
from attnbench.predict.recursion import (MechanismCost, close_prefill, copy_op,  # noqa: E402
                                         kernel_op, price)
from attnbench.predict.training import CellKey  # noqa: E402

COSTS = {"synchronize": MechanismCost(s_fixed=1)}


def priced(ops):
    return price(close_prefill(ops), COSTS)


# --------------------------------------------------------------------------
# traces
# --------------------------------------------------------------------------

def test_trace_key_reads_length_batch_arm_and_density():
    assert TR.trace_key("a/trace_L8192_b1_0_meanpool_d0.25.json") == (8075, 1, "meanpool", 0.25)
    assert TR.trace_key("trace_L2048_b1_0_xattention_noassert.json") == \
        (1951, 1, "xattention_noassert", None)
    assert TR.trace_key("trace_L16384_b4_0_dense.json") == (16370, 4, "dense", None)
    with pytest.raises(TR.TraceError):
        TR.trace_key("result.json")


def _cell(arm, L, b=1, d=1.0):
    return SimpleNamespace(arm=arm, context_length=L, batch=b, density=d)


def test_cell_for_matches_density_only_where_the_name_carries_one():
    cells = [_cell("dense", 8075), _cell("meanpool", 8075, d=0.25), _cell("meanpool", 8075, d=0.5),
             _cell("xattention", 8075, d=0.666877), _cell("dense", 16370)]
    assert TR.cell_for(cells, "trace_L8192_b1_0_meanpool_d0.5.json") is cells[2]
    assert TR.cell_for(cells, "trace_L8192_b1_0_dense.json") is cells[0]
    assert TR.cell_for(cells, "trace_L8192_b1_0_xattention.json") is cells[3]
    with pytest.raises(TR.TraceError, match="0 cells"):
        TR.cell_for(cells, "trace_L8192_b4_0_dense.json")
    with pytest.raises(TR.TraceError, match="2 cells"):
        TR.cell_for(cells + [_cell("dense", 8075)], "trace_L8192_b1_0_dense.json")


def _ev(cat, name, ts, **args):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": 1.0, "pid": 1, "tid": 2,
            "args": args}


def test_window_is_the_embedding_and_the_first_synchronise_after_it():
    tj = {"traceEvents": [_ev("cuda_runtime", "cudaDeviceSynchronize", 50),
                          _ev("cpu_op", "aten::embedding", 100),
                          _ev("cuda_runtime", "cudaDeviceSynchronize", 400),
                          _ev("cuda_runtime", "cudaDeviceSynchronize", 300)]}
    e0, s = TR.window(tj)
    assert (e0["ts"], s["ts"]) == (100, 300)
    tj["traceEvents"].append(_ev("cpu_op", "aten::embedding", 200))
    with pytest.raises(TR.TraceError, match="2 embedding"):
        TR.window(tj)


def test_placeholders_cover_only_launches_in_the_window_without_a_device_record():
    ev = [_ev("kernel", "k1", 10, correlation=1, device=0, stream=7),
          _ev("kernel", "k2", 11, correlation=2, device=0, stream=7),
          _ev("kernel", "k9", 12, correlation=9, device=0, stream=9),
          _ev("cuda_runtime", "cudaLaunchKernel", 5, correlation=1),
          _ev("cuda_runtime", "cudaLaunchKernel", 6, correlation=2),
          _ev("cuda_runtime", "cudaLaunchKernel", 7, correlation=3),       # dropped record
          _ev("cuda_driver", "cuLaunchKernel", 8, correlation=4),          # dropped, driver API
          _ev("cuda_runtime", "cudaMemsetAsync", 8.5, correlation=6),      # dropped memset
          _ev("cuda_runtime", "cudaLaunchKernel", 99, correlation=5),      # outside the window
          _ev("cuda_runtime", "cudaFuncGetAttributes", 7.5, correlation=8)]
    add, corrs = TR.placeholders({"traceEvents": ev}, 0, 50)
    assert corrs == [3, 4, 6]
    assert [(a["cat"], a["tid"], a["dur"]) for a in add] == \
        [("kernel", 7, 0.0), ("kernel", 7, 0.0), ("gpu_memset", 7, 0.0)]


def test_align_folds_an_unpriced_fill_into_the_next_operation():
    fam = ["kernel:aten::mul", "kernel:aten::fill_", "kernel:flash_attn_3::fwd", "sync:closing"]
    f, d, n = TR.align(fam, [1.0, 0.5, 2.0, None], ["kernel", "kernel", "sync"])
    assert f == ["kernel:aten::mul", "kernel:flash_attn_3::fwd", "sync:closing"]
    assert d == [1.0, 2.5, None] and n == 1


def test_align_keeps_a_sequence_of_the_priced_length_untouched():
    fam = ["kernel:aten::fill_", "memset:aten::mm", "copy:aten::copy_", "sync:closing"]
    dev = [1.0, 2.0, 3.0, None]
    assert TR.align(fam, dev, ["kernel", "kernel", "copy", "sync"]) == (fam, dev, 0)


def test_align_refuses_a_kind_mismatch_a_length_mismatch_and_a_trailing_fill():
    with pytest.raises(TR.TraceError, match="kind differs"):
        TR.align(["kernel:aten::mul", "sync:closing"], [1.0, None], ["copy", "sync"])
    with pytest.raises(TR.TraceError, match="traced vs"):
        TR.align(["kernel:aten::mul", "kernel:aten::add", "sync:closing"], [1, 1, None],
                 ["kernel", "kernel", "kernel", "sync"])
    with pytest.raises(TR.TraceError, match="at the end"):
        TR.align(["kernel:aten::mul", "kernel:aten::fill_"], [1.0, 1.0], ["kernel"])


# --------------------------------------------------------------------------
# breakdown
# --------------------------------------------------------------------------

def test_component_of():
    assert BR.component_of("kernel:flash_attn_3::fwd") == "attention"
    assert BR.component_of("kernel:aten::_cudnn_attention_forward") == "attention"
    assert BR.component_of("kernel:flash_attn::_block_sparse_attn_forward") == "attention"
    assert BR.component_of("kernel:aten::mm") == "gemm"
    assert BR.component_of("memset:aten::addmm") == "gemm"
    assert BR.component_of("kernel:aten::bmm") == "other_device"
    assert BR.component_of("copy:aten::copy_") == "copy"
    assert BR.component_of("device_copy:aten::copy_") == "other_device"
    assert BR.component_of("sync:closing") is None


FAM = ["kernel:flash_attn::fwd", "kernel:aten::mm", "kernel:aten::add", "sync:closing"]


def _seq():
    # h = 1, 1, 3; d = 5, 5, 1; closing s_0 = 1.
    # H = 1, 2, 5. D1 = 1+5 = 6; D2 = max(2,6)+5 = 11; D3 = max(5,11)+1 = 12;
    # closing H = max(5,12)+1 = 13.
    return priced([kernel_op(1, 5), kernel_op(1, 5), kernel_op(3, 1)])


def test_split_replaces_one_component_at_a_time_then_all():
    traced = [2, 4, None, None]
    s = BR.split(_seq(), 1022, 10.0, traced, FAM)
    assert s.predicted == 13
    # attention: D1 = 1+2 = 3; D2 = max(2,3)+5 = 8; D3 = max(5,8)+1 = 9; end 10.
    assert s.replaced["attention"] == 10
    # gemm: D1 = 6; D2 = max(2,6)+4 = 10; D3 = 11; end 12.
    assert s.replaced["gemm"] == 12
    # other_device is unmeasured (None), copy absent: both unchanged.
    assert s.replaced["other_device"] == 13 and s.replaced["copy"] == 13
    # all: D1 = 3; D2 = max(2,3)+4 = 7; D3 = max(5,7)+1 = 8; end 9.
    assert s.all_replaced == 9
    assert s.device == {"attention": (5, 2), "gemm": (5, 4), "other_device": (0, 0),
                        "copy": (0, 0)}
    assert s.rel(s.predicted) == pytest.approx(0.3)


def test_substitute_puts_a_copy_in_link_and_leaves_unmeasured_positions():
    # copy: h = 1, l = 4. H = 1; D1 = 1+4 = 5; closing H = max(1,5)+1 = 6.
    p = priced([copy_op(1, 4)])
    q = BR.substitute(p, [2, None], ["copy:aten::copy_", "sync:closing"])
    assert q.link[0] == 2 and q.device == p.device
    assert BR.split(p, 1022, 4.0, [2, None], ["copy:aten::copy_", "sync:closing"]).all_replaced == 4
    with pytest.raises(ValueError, match="lengths differ"):
        BR.substitute(p, [2], ["copy:aten::copy_", "sync:closing"])


def test_median_abs_rel_and_summary():
    assert BR.median_abs_rel([(11, 10), (8, 10), (10, 10)]) == pytest.approx(0.1)
    a = BR.split(_seq(), 1022, 10.0, [2, 4, None, None], FAM)     # 13, 10/12, 9
    b = BR.split(_seq(), 1022, 13.0, [5, 5, 1, None], FAM)        # exact everywhere
    c = BR.split(_seq(), 1022, 26.0, [5, 5, 1, None], FAM)        # 13 against 26
    out = BR.summary([a, b, c])
    # |rel|: a 0.3, b 0, c 0.5 -> median 0.3. Attention replaced: a 0, b 0, c 0.5 -> 0.
    assert out["median_abs_rel"] == pytest.approx(0.3)
    assert out["replaced"]["attention"] == pytest.approx(0.0)
    assert out["replaced"]["gemm"] == pytest.approx(0.2)
    assert out["all_device_replaced"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        BR.median_abs_rel([])


# --------------------------------------------------------------------------
# safe dispatch
# --------------------------------------------------------------------------

M = "primary"
DENSE, MP, OR = DP.Config("dense", 1.0), DP.Config("meanpool", 0.25), DP.Config("oracle", 0.25)
FLOORS = FloorTable({(8192, 1): 1.0, (16384, 1): 1.0, (32768, 1): 1.0})
PRED = {8075: {DENSE: 100, MP: 90}, 16370: {DENSE: 100, MP: 95},
        32700: {DENSE: 100, MP: 70, OR: 50}}
MEAS = {8075: {DENSE: 100, MP: 105}, 16370: {DENSE: 100, MP: 70},
        32700: {DENSE: 100, MP: 60, OR: 40}}


def _key(L, c):
    return CellKey(M, "cp", c.arm, c.density, L, 1)


def _pts():
    return DP.build_points({_key(L, c): v for L, r in PRED.items() for c, v in r.items()},
                           dense_arms={"dense"})


def _meas():
    return SD.measured_by_point({_key(L, c): v for L, r in MEAS.items() for c, v in r.items()})


ADM = SD.admissible_of([_key(L, c) for L, r in PRED.items() for c in r])


def test_admissible_excludes_dense_and_the_oracle():
    assert ADM == frozenset({MP})


def test_margin_from_is_the_largest_advantage_among_losing_choices():
    # The dispatcher chooses meanpool everywhere (each advantage > the 1 s floor);
    # only 8075 measured slower than dense, at a predicted advantage of 10/100.
    m, bad = SD.margin_from(_pts(), _meas(), ADM, FLOORS)
    assert m == pytest.approx(0.10)
    assert [b[0].context_length for b in bad] == [8075]
    assert SD.margin_from(_pts(), _meas(), ADM, FLOORS, batches=(4,)) == (0.0, [])


def test_fallback_margin_is_one_sided():
    # predicted minus measured advantage: 8075 0.10-(-0.05) = +0.15;
    # 16370 0.05-0.30 = -0.25; 32700 0.30-0.40 = -0.10. One-sided max 0.15, not 0.25.
    assert SD.fallback_margin(_pts(), _meas(), ADM) == pytest.approx(0.15)
    meas = _meas()
    meas[DP.Point(M, 8075, 1)][MP] = 80      # 0.10 - 0.20 < 0: nothing overstated
    assert SD.fallback_margin(_pts(), meas, ADM) == 0.0


def test_abstaining_needs_more_than_the_margin():
    pts = {p.point.context_length: p for p in _pts()}
    assert SD.abstaining(pts[8075], ADM, FLOORS, 0.10) == DENSE     # 0.10 is not > 0.10
    assert SD.abstaining(pts[8075], ADM, FLOORS, 0.09) == MP
    assert SD.abstaining(pts[32700], ADM, FLOORS, 0.10) == MP
    with pytest.raises(ValueError):
        SD.abstaining(pts[8075], ADM, FLOORS, -0.01)


def test_score_regret_slowdown_and_wins_kept():
    pts, meas = _pts(), _meas()
    ab = {p.point: SD.abstaining(p, ADM, FLOORS, 0.10) for p in pts}     # D, D, MP
    s = SD.score(pts, ab, meas, ADM, FLOORS)
    # best (oracle excluded): 8075 dense, 16370 MP (70), 32700 MP (60).
    # regret: 0, 100/70 - 1, 0.  vs dense: 0, 0, -0.4.  wins: 16370, 32700; kept: 32700.
    assert s.n == 3
    assert s.worst_regret == pytest.approx(100 / 70 - 1)
    assert s.mean_regret == pytest.approx((100 / 70 - 1) / 3)
    assert s.worst_vs_dense == 0.0 and s.mean_vs_dense == pytest.approx(-0.4 / 3)
    assert (s.wins, s.wins_kept) == (2, 1)
    rec = {p.point: DP.dispatcher(p, ADM, FLOORS) for p in pts}          # MP everywhere
    r = SD.score(pts, rec, meas, ADM, FLOORS)
    assert r.worst_regret == pytest.approx(0.05) and r.worst_vs_dense == pytest.approx(0.05)
    assert (r.wins, r.wins_kept) == (2, 2)


def test_a_win_inside_the_floor_is_not_counted():
    floors = FloorTable({(8192, 1): 1.0, (16384, 1): 31.0, (32768, 1): 41.0})
    s = SD.score(_pts(), {p.point: DENSE for p in _pts()}, _meas(), ADM, floors)
    assert (s.wins, s.wins_kept) == (0, 0)


def test_policies_feed_the_step_form_its_own_times():
    cells = [SimpleNamespace(key=_key(L, c), predictions={"recursion": v,
                                                          "step_form": 100 if c == DENSE else 150})
             for L, r in PRED.items() for c, v in r.items()]
    pol = SD.policies(cells, ADM, FLOORS, 0.10)
    assert set(pol["bunga_recursion"][1].values()) == {MP}
    assert set(pol["bunga_step_form"][1].values()) == {DENSE}
    assert set(pol["always_dense"][1].values()) == {DENSE}
    assert sorted(v.arm for v in pol["bunga_abstain"][1].values()) == ["dense", "dense", "meanpool"]


# --------------------------------------------------------------------------
# bound
# --------------------------------------------------------------------------

L4 = "NVIDIA L4"
PC, PB = BD.PEAKS[L4]


def test_work_counts_every_layer_once_and_the_head_once():
    shape = IN.primary_shape()
    fb = IN.flops_and_bytes(shape, 1024, 1, 0.5)
    per_f = sum(v[0] for k, v in fb.items() if k != "lm_head")
    per_b = sum(v[1] for k, v in fb.items() if k != "lm_head")
    f1, b1 = BD.work(shape, 1, 1024, 1, 0.5)
    f3, b3 = BD.work(shape, 3, 1024, 1, 0.5)
    assert (f1, b1) == (per_f + fb["lm_head"][0], per_b + fb["lm_head"][1])
    assert (f3 - f1, b3 - b1) == (2 * per_f, 2 * per_b)


def test_fit_is_the_envelope_and_every_training_cell_is_at_or_above_it():
    a = BD.Obs(L4, flops=0.5 * PC, bytes=0.1 * PB, n_ops=50, measured=1.0)       # 0.5, 0.1, 0.02
    b = BD.Obs(L4, flops=0.1 * 2 * PC, bytes=0.3 * 2 * PB, n_ops=400, measured=2.0)  # 0.1, 0.3, 0.005
    f = BD.fit([a, b])
    assert (f.phi_c, f.phi_m, f.tau) == pytest.approx((0.5, 0.3, 0.005))
    r = BD.check(f, {"a": a, "b": b})
    assert min(v["ratio"] for v in r.values()) == pytest.approx(1.0)
    assert all(v["ratio"] >= 1 - 1e-12 for v in r.values())
    with pytest.raises(ValueError):
        BD.fit([BD.Obs("a GPU nobody tabled", 1, 1, 1, 1.0)])
    with pytest.raises(ValueError):
        BD.fit([])


def test_check_names_the_binding_term_and_the_status():
    f = BD.Bound(phi_c=0.5, phi_m=0.25, tau=0.01)
    # terms: compute F/(0.5 PC), memory B/(0.25 PB), host n * 0.01.
    o = lambda t: BD.Obs(L4, flops=0.5 * PC * 2.0, bytes=0.25 * PB * 1.0, n_ops=100, measured=t)
    r = BD.check(f, {"break": o(1.5), "close": o(2.1), "clear": o(3.0), "edge": o(2.0)})
    assert {k: v["binding"] for k, v in r.items()} == dict.fromkeys(r, "compute")
    assert r["break"]["bound_s"] == pytest.approx(2.0)
    assert [r[k]["status"] for k in ("break", "close", "clear", "edge")] == \
        ["break", "close", "clear", "close"]
    h = BD.Obs(L4, flops=0.0, bytes=0.0, n_ops=500, measured=4.0)        # host 5 s
    assert BD.check(f, {"h": h})["h"] == {"ratio": pytest.approx(0.8), "binding": "host",
                                          "bound_s": pytest.approx(5.0), "status": "break"}


# --------------------------------------------------------------------------
# sign
# --------------------------------------------------------------------------

def test_a_constant_difference_has_a_point_interval():
    d = [10.0, 11.0, 12.0, 13.0]
    s = SG.paired(d, [x - 1 for x in d], floor_s=0.5, n_boot=500)
    assert (s.diff_s, s.lo_s, s.hi_s, s.p_sparse_faster) == (-1.0, -1.0, -1.0, 1.0)
    assert (s.sign, s.beyond_floor) == ("sparse faster", "sparse faster")
    s2 = SG.paired(d, [x - 1 for x in d], floor_s=2.0, n_boot=500)
    assert (s2.sign, s2.beyond_floor) == ("sparse faster", "undecided")
    s3 = SG.paired(d, [x + 1 for x in d], floor_s=0.5, n_boot=500)
    assert (s3.sign, s3.p_sparse_faster) == ("dense faster", 0.0)


def test_pairing_by_round_removes_the_shared_drift():
    # A drift of 0..90 s shared by both arms, sparse 0.1 s faster every round.
    # Paired, every replicate is about -0.1; unpaired, medians of independent
    # resamples differ by tens of seconds and the interval straddles 0.
    drift = [10.0 * i for i in range(10)]
    d = [x + 0.01 * (i % 3) for i, x in enumerate(drift)]
    s = [x - 0.1 for x in d]
    out = SG.paired(d, s, floor_s=0.0)
    assert out.sign == "sparse faster" and out.hi_s < 0 and out.lo_s > -0.2


def test_same_seed_same_interval_and_bad_input_refused():
    d, s = [1.0, 2.0, 3.0, 5.0, 8.0], [1.5, 1.0, 3.5, 4.0, 9.0]
    assert SG.paired(d, s, floor_s=0, seed=3) == SG.paired(d, s, floor_s=0, seed=3)
    with pytest.raises(SG.PairingError):
        SG.paired([1.0, 2.0], [1.0], floor_s=0)
    with pytest.raises(ValueError):
        SG.paired(d, s, floor_s=0, level=1.0)


def test_pairable():
    row = {"prefill_ms_samples": [1.0] * 10}
    assert SG.pairable({"rounds_run": 10, "clean_rounds": 10, "rows": [row, row]}) is None
    assert "dropped round" in SG.pairable({"rounds_run": 11, "clean_rounds": 10, "rows": [row]})
    with pytest.raises(SG.PairingError):
        SG.pairable({"rounds_run": 10, "clean_rounds": 10,
                     "rows": [row, {"prefill_ms_samples": [1.0] * 9}]})


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _margin_with(monkeypatch, m0, fallback):
    monkeypatch.setattr(SD, "admissible_of", lambda keys: frozenset())
    monkeypatch.setattr(SD, "measured_by_point", lambda x: {})
    monkeypatch.setattr(SD, "points_for", lambda cells, p: [])
    monkeypatch.setattr(SD, "margin_from", lambda *a, **k: (m0, []))
    monkeypatch.setattr(SD, "fallback_margin", lambda *a, **k: fallback)
    D = {"cps": {"P1": SimpleNamespace(cells=[])}, "p1_measured": {},
         "locked": SimpleNamespace(floors=None)}
    return EW.margin(D)


def test_the_driver_dispatches_only_at_the_recorded_margin(monkeypatch):
    assert _margin_with(monkeypatch, 0.0, 0.022639)["margin"] == 0.022639
    assert _margin_with(monkeypatch, 0.0, 0.022639)["rule"].startswith("fallback")
    with pytest.raises(SystemExit, match="refusing"):
        _margin_with(monkeypatch, 0.0, 0.0300)
    with pytest.raises(SystemExit, match="refusing"):
        _margin_with(monkeypatch, 0.05, 0.022639)     # the primary rule wins when non-zero


def test_the_driver_refuses_output_naming_the_store():
    EW.refuse_store('{"a": 1}', "gs://some-bucket/")
    with pytest.raises(SystemExit):
        EW.refuse_store('{"a": "some-bucket/x"}', "gs://some-bucket/")
    with pytest.raises(SystemExit):
        EW.refuse_store('{"a": "gs://other"}', "gs://some-bucket/")


def test_device_sums_count_a_copy_from_link():
    p = priced([copy_op(1, 4), kernel_op(1, 3)])
    fam = ["copy:aten::copy_", "kernel:aten::add", "sync:closing"]
    assert BR.device_sums(p, [2, 1, None], fam) == {"attention": (0, 0), "gemm": (0, 0),
                                                    "other_device": (3, 1), "copy": (4, 2)}


def test_a_sparse_choice_inside_the_floor_is_not_a_kept_win():
    # 16370's win is MP at 70 (30 s); a second config B measured 99.5 is chosen.
    B = DP.Config("meanpool", 0.5)
    pts = DP.build_points({**{_key(L, c): v for L, r in PRED.items() for c, v in r.items()},
                           _key(16370, B): 94}, dense_arms={"dense"})
    meas = _meas()
    meas[DP.Point(M, 16370, 1)][B] = 99.5
    adm = frozenset({MP, B})
    ch = {p.point: (B if p.point.context_length == 16370 else DENSE) for p in pts}
    s = SD.score(pts, ch, meas, adm, FLOORS)
    assert (s.wins, s.wins_kept) == (2, 0)


def test_a_zero_difference_is_not_sparse_faster():
    d = [1.0, 2.0, 3.0]
    s = SG.paired(d, d, floor_s=0.0, n_boot=200)
    assert (s.p_sparse_faster, s.sign) == (0.0, "undecided")


def test_the_interval_has_the_stated_level():
    # 1001 rounds, sparse - dense = i/1000 - 0.5 per round, the rounds' median
    # difference is near 0, and the 95% interval is narrower than the 99% one.
    d = [0.0] * 1001
    s = [i / 1000 - 0.5 for i in range(1001)]
    a = SG.paired(d, s, floor_s=0, n_boot=2000, level=0.95)
    b = SG.paired(d, s, floor_s=0, n_boot=2000, level=0.50)
    assert a.lo_s < b.lo_s < 0 < b.hi_s < a.hi_s
    # the median of a resample of U(-0.5, 0.5) has sd about 1/sqrt(4n) = 0.0158
    assert a.hi_s - a.lo_s == pytest.approx(2 * 1.96 * 0.0158, rel=0.15)


# --------------------------------------------------------------------------
# the driver's composition, on fakes
# --------------------------------------------------------------------------

def _fake_cell(arm, L, density, seq, predictions=None):
    return SimpleNamespace(key=CellKey(M, "cp", arm, density, L, 1), arm=arm,
                           context_length=L, batch=1, density=density, code_path="cp",
                           priced=seq, predictions=predictions or {})


def test_item2_averages_a_cell_s_traces_and_lists_cells_without_one():
    c = _fake_cell("dense", 8075, 1.0, _seq())
    other = _fake_cell("meanpool", 8075, 0.25, _seq())
    cp = SimpleNamespace(cells=[c, other], queue_depth=1022)
    recs = [{"key": c.key, "family": FAM, "device": [1, 4, None, None]},
            {"key": c.key, "family": FAM, "device": [3, 4, 1, None]}]
    out = EW.item2(cp, {"seconds": {c.key: 10.0}}, recs)
    (cell,) = out["cells"]
    # averaged: [2, 4, None (one trace lacks it), None] -> the split test's 9.
    assert cell["rel_error_all_device_replaced"] == pytest.approx(-0.1)
    assert cell["traces"] == 2
    assert out["cells_without_trace"] == [list(other.key)]
    recs[1]["family"] = FAM[:2] + ["kernel:aten::mul", "sync:closing"]
    with pytest.raises(TR.TraceError, match="differ"):
        EW.item2(cp, {"seconds": {c.key: 10.0}}, recs)


def test_item5_pairs_within_a_file_and_skips_an_unpairable_one(monkeypatch):
    dk, sk = CellKey(M, "cp", "dense", 1.0, 8075, 1), CellKey(M, "cp", "meanpool", 0.25, 8075, 1)
    monkeypatch.setattr(EW.SH.G, "place_row", lambda cp, row, name: SimpleNamespace(key=row["k"]))
    rows = [{"k": dk, "prefill_ms_samples": [1.0] * 3}, {"k": sk, "prefill_ms_samples": [1.0] * 3}]
    good = {"rounds_run": 3, "clean_rounds": 3, "rows": rows}
    bad = {"rounds_run": 4, "clean_rounds": 3, "rows": rows}
    only_dense = {"rounds_run": 4, "clean_rounds": 3, "rows": rows[:1]}
    per = {EW.SH.BlockCell("b4", dk): [10.0, 11.0, 12.0], EW.SH.BlockCell("b4", sk): [9.0, 10.0, 11.0],
           EW.SH.BlockCell("b3", dk): [0.0] * 3, EW.SH.BlockCell("b3", sk): [99.0] * 3}
    D = {"locked": SimpleNamespace(floors=FLOORS, bootstrap=SimpleNamespace(n_boot=100, seed=0,
                                                                            level=0.95)),
         "cps": {p: None for p in EW.PAIRINGS},
         "ms": {p: {"per_block": per} for p in EW.PAIRINGS},
         "blocks": {p: [("b3", [("x", bad), ("y", only_dense)]), ("b4", [("z", good)])]
                    for p in EW.PAIRINGS}}
    out = EW.item5(D)["P2"]
    (name, v), = out["cells"].items()
    assert v["block"] == "b4" and v["diff_s"] == -1.0 and v["sign"] == "sparse faster"
    assert [s["file"] for s in out["skipped"]] == ["x"]


def test_item4_fits_on_p1_sparse_cells_only(monkeypatch):
    monkeypatch.setattr(BD, "layers", lambda *a: 1)
    seq = _seq()
    p1 = [_fake_cell("dense", 1951, 1.0, seq), _fake_cell("meanpool", 1951, 0.25, seq)]
    cp1 = SimpleNamespace(cells=p1, code_paths={"cp": {"gpu": L4}})
    D = {"cps": {"P1": cp1, **{p: cp1 for p in EW.PAIRINGS}},
         "p1_measured": {p1[0].key: 1e-6, p1[1].key: 1.0},
         "ms": {p: {"seconds": {p1[0].key: 1e-6, p1[1].key: 1.0}} for p in EW.PAIRINGS}}
    out = EW.item4(D)
    # the dense cell's 1 us would set tau to 0.25 us; the sparse cell alone gives 1/4 s.
    assert out["n_train"] == 1 and out["fit_on_P1"]["tau"] == pytest.approx(0.25)
    assert list(out["P2"]["cells"]) == ["meanpool|0.25|1951|1"]
    assert out["P2"]["cells"]["meanpool|0.25|1951|1"]["ratio"] == pytest.approx(1.0)


def test_item3_scores_the_four_policies_at_the_margin():
    cells = [SimpleNamespace(key=_key(L, c), predictions={"recursion": v, "step_form": v})
             for L, r in PRED.items() for c, v in r.items()]
    secs = {_key(L, c): v for L, r in MEAS.items() for c, v in r.items()}
    D = {"locked": SimpleNamespace(floors=FLOORS),
         "cps": {p: SimpleNamespace(cells=cells) for p in EW.PAIRINGS},
         "ms": {p: {"seconds": secs} for p in EW.PAIRINGS}}
    out = EW.item3(D, 0.10)["P4"]
    assert out["bunga_abstain"]["worst_regret"] == pytest.approx(100 / 70 - 1)
    assert out["bunga_recursion"]["worst_vs_dense"] == pytest.approx(0.05)
    assert out["always_dense"]["wins_kept"] == 0
    assert out["bunga_abstain"]["choices"]["32700"] == ["meanpool", 0.25]


def test_the_memory_term_reads_bandwidth_not_the_compute_peak():
    f = BD.Bound(phi_c=0.5, phi_m=0.25, tau=0.01)
    # memory: 0.25 * PB * 3 / (0.25 * PB) = 3 s; compute 0; host 0.01 s.
    o = BD.Obs(L4, flops=0.0, bytes=0.25 * PB * 3.0, n_ops=1, measured=6.0)
    r = BD.check(f, {"m": o})["m"]
    assert (r["binding"], r["bound_s"], r["ratio"]) == ("memory", pytest.approx(3.0), pytest.approx(2.0))


def test_the_observed_difference_is_of_medians():
    # medians 2 and 1 -> -1; means 2 and about 33.7 would give +31.7.
    s = SG.paired([1.0, 2.0, 3.0], [0.0, 1.0, 100.0], floor_s=0.0, n_boot=50)
    assert s.diff_s == -1.0
