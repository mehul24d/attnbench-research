"""EXPLORATORY: the H100 diagnostic (17 Nov 2026 rulings, item 2): the cell
rules, every function behind a number the session reports, and the runner's
refusals, on fixtures with answers worked out in the comments.

The Q4 recursion check needs P4's local data and is skipped without it
(H100_DIAG_Q4_INPUTS, H100_DIAG_Q4_MIRROR, H100_DIAG_Q4_TRACES).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import _nsys_fakes as NF  # noqa: E402
import h100_diag as D  # noqa: E402
from attnbench.exploratory import h100_diag as HD  # noqa: E402

CELLS = ROOT / "configs" / "exploratory" / "h100_diag_cells.json"
CARD = ROOT / "docs" / "exploratory" / "h100_diag_card.md"


def cell(id="c", arm="dense", backend="fa3", length=16384, batch=1, mode="eager"):
    return HD.Cell(id, arm, backend, length, batch, mode)


# --------------------------------------------------------------------------
# the cell list
# --------------------------------------------------------------------------

def test_the_proposed_list_passes_and_the_card_names_its_sha():
    cells, sha = HD.load_cells(CELLS)
    assert len(cells) == 8
    assert sha == hashlib.sha256(CELLS.read_bytes()).hexdigest()
    assert sha in CARD.read_text()


@pytest.mark.parametrize("cells, why", [
    ([cell(id=str(i)) for i in range(9)], "9 cells"),
    ([], "0 cells"),
    ([cell(), cell()], "repeat"),
    ([cell(batch=2)], "batch 2"),
    ([cell(length=4096)], "length 4096"),
    ([cell(mode="compiled")], "mode"),
    ([cell(backend=None)], "dense backend"),
    ([cell(backend="sdpa_math")], "dense backend"),
    ([cell(arm="xattention", backend="fa3")], "XAttention"),
    ([cell(arm="xattention", backend=None, mode="graph")], "XAttention"),
    ([cell(arm="meanpool")], "arm"),
    ([cell("a"), cell("g1", mode="graph"), cell("g2", mode="graph")], "2 CUDA-graph"),
    ([cell("a", length=8192), cell("g", mode="graph")], "eager twin"),
    ([cell("a", backend="sdpa_flash"), cell("g", mode="graph")], "eager twin"),
])
def test_cell_rules_refuse(cells, why):
    with pytest.raises(HD.DiagError, match=why):
        HD.check_cells(cells)


def test_load_refuses_another_schema(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"schema": "other", "cells": []}))
    with pytest.raises(HD.DiagError, match="schema"):
        HD.load_cells(p)


# --------------------------------------------------------------------------
# timelines
# --------------------------------------------------------------------------

def test_union_merges_overlaps_and_clips():
    # [0,4] u [2,6] = [0,6]; [8,12] clipped to [8,10]; [11,13] is outside
    assert HD._union([(2, 6), (0, 4), (8, 12), (11, 13)], 0, 10) == 8
    assert HD._union([(1, 2), (3, 4)], 0, 10) == 2
    assert HD._union([], 0, 10) == 0
    assert HD._union([(0, 1), (1.5, 2)], 0, 10) == 1.5      # a gap shorter than one unit
    assert HD._union([(0, 10), (2, 3)], 0, 20) == 10        # one interval inside another


def test_host_stats_counts_only_blocking_before_the_last_launch():
    # launches end at 2, 5, 9; a sync 3-4 (before the last launch) and the
    # closing sync 9-20 (after it); device busy 4-8 and 10-18 = 12
    s = HD.host_stats(0, 20, [(1, 2), (4, 5), (8, 9)], [(3, 4), (9, 20)], [(4, 8), (10, 18)])
    assert s == {"wall_s": 20, "device_busy_s": 12, "idle_s": 8, "idle_share": 0.4,
                 "submit_s": 9, "blocked_s": 1, "host_s": 8, "launches": 3}
    with pytest.raises(HD.DiagError):
        HD.host_stats(5, 5, [], [], [])


def test_host_stats_ignores_a_launch_before_the_start():
    s = HD.host_stats(10, 20, [(1, 15), (11, 12)], [], [])
    assert s["submit_s"] == 2 and s["launches"] == 1


def kineto(region="prefill"):
    ev = [{"name": region, "cat": "user_annotation", "ts": 100, "dur": 50},
          {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ts": 105, "dur": 5},
          {"name": "cuLaunchKernel", "cat": "cuda_driver", "ts": 120, "dur": 5},
          {"name": "cudaStreamSynchronize", "cat": "cuda_runtime", "ts": 112, "dur": 3},
          {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ts": 90, "dur": 5},  # before
          {"name": "k1", "cat": "kernel", "ts": 110, "dur": 20},
          {"name": "m1", "cat": "gpu_memcpy", "ts": 140, "dur": 20},
          {"name": "old", "cat": "kernel", "ts": 50, "dur": 10}]
    return {"traceEvents": ev}


def test_kineto_host_in_seconds_to_the_last_device_end():
    # region 100-150, last device end 160: wall 60 µs; busy 110-130 + 140-160
    # = 40; launches end 110, 125: submit 25, blocked 3 (112-115), host 22
    s = HD.kineto_host(kineto(), "prefill")
    assert s["wall_s"] == pytest.approx(60e-6) and s["device_busy_s"] == pytest.approx(40e-6)
    assert s["submit_s"] == pytest.approx(25e-6) and s["blocked_s"] == pytest.approx(3e-6)
    assert s["host_s"] == pytest.approx(22e-6) and s["launches"] == 2
    assert s["idle_share"] == pytest.approx(20 / 60)


def test_kineto_host_wants_one_region():
    with pytest.raises(HD.DiagError, match="0 'prefill'"):
        HD.kineto_host(kineto("other"), "prefill")
    t = kineto()
    t["traceEvents"].append(dict(t["traceEvents"][0]))
    with pytest.raises(HD.DiagError, match="2 'prefill'"):
        HD.kineto_host(t, "prefill")


def test_nsys_prefills_reads_the_synthetic_export(tmp_path):
    db = tmp_path / "x.sqlite"
    NF.write_db(db, 3)
    rows = HD.nsys_prefills(db)
    assert [r["range"] for r in rows] == [f"{NF.PREFIX}{i}" for i in range(3)]
    for i, r in enumerate(rows):
        e = NF.expected(i)
        for k, v in e.items():
            assert r[k] == (pytest.approx(v) if isinstance(v, float) else
                            ({n: pytest.approx(x) for n, x in v.items()} if isinstance(v, dict)
                             else v)), k


def test_nsys_prefills_takes_a_range_name_from_the_string_table(tmp_path):
    import sqlite3
    db = tmp_path / "x.sqlite"
    NF.write_db(db, 1)
    con = sqlite3.connect(db)
    con.execute("INSERT INTO StringIds VALUES (99, ?)", (f"{NF.PREFIX}9",))
    con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 99)", (5 * NF.PERIOD, 5 * NF.PERIOD + 10))
    con.commit()
    con.close()
    assert [r["range"] for r in HD.nsys_prefills(db)] == [f"{NF.PREFIX}0", f"{NF.PREFIX}9"]


def test_nsys_prefills_refuses_an_incomplete_export(tmp_path):
    import sqlite3
    db = tmp_path / "x.sqlite"
    NF.write_db(db, 1)
    with pytest.raises(HD.DiagError, match="no NVTX range"):
        HD.nsys_prefills(db, prefix="nope:")
    con = sqlite3.connect(db)
    con.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
    con.commit()
    con.close()
    with pytest.raises(HD.DiagError, match="RUNTIME"):
        HD.nsys_prefills(db)


def test_component_seconds_skips_syncs_and_counts_missing():
    fam = ["kernel:aten::_flash_attention_forward", "kernel:aten::mm", "sync:cudaDeviceSynchronize",
           "copy:aten::copy_", "kernel:aten::add", "kernel:aten::addmm"]
    c = HD.component_seconds(fam, {0: 4.0, 1: 2.0, 2: 100.0, 3: 1.0, 4: 0.5})
    assert c == {"attention": 4.0, "gemm": 2.0, "copy": 1.0, "other_device": 0.5, "total": 7.5,
                 "positions_without_duration": 1}


def test_median_of():
    assert HD.median_of([{"x": 1}, {"x": 5}, {"x": 2}], "x") == 2
    with pytest.raises(HD.DiagError):
        HD.median_of([], "x")


# --------------------------------------------------------------------------
# the outcome rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r, ok", [(1.05, True), (0.95, True), (1.2, False), (0.85, False)])
def test_q1(r, ok):
    q = HD.q1_priced_kernel(r, 1.0)
    assert q["confirmed"] == ok and q["ratio"] == pytest.approx(r)
    assert q["verdict"].startswith("confirmed" if ok else "not confirmed")


def test_q1_refuses_zero():
    with pytest.raises(HD.DiagError):
        HD.q1_priced_kernel(0.0, 1.0)


@pytest.mark.parametrize("g, start", [(1.05, "faithful"), (0.95, "faithful"),
                                      (2.0, "the graph replay alone"), (2.6, "the graph replay alone"),
                                      (1.5, "the graph replay inflates"),
                                      (0.8, "the graph replay deflates")])
def test_q2(g, start):
    r = HD.q2_graph(g, 1.0)
    assert r["verdict"].startswith(start) and r["faithful"] == start.startswith("faithful")
    assert r["ratio"] == pytest.approx(g)


def test_q2_refuses_zero():
    with pytest.raises(HD.DiagError):
        HD.q2_graph(1.0, 0.0)


def test_q3():
    assert HD.q3_comparable({"a": (1.05, 1.0), "b": (0.92, 1.0)})["comparable"]
    r = HD.q3_comparable({"a": (1.05, 1.0), "b": (1.2, 1.0)})
    assert not r["comparable"] and not r["rows"]["b"]["within"] and r["rows"]["a"]["within"]
    assert not HD.q3_comparable({"a": (0.85, 1.0)})["comparable"]
    assert not HD.q3_comparable({})["comparable"]


@pytest.mark.parametrize("pred_h, phi, start", [(2.0, 0.8, "the host costs are overstated"),
                                                (3.5, 0.5, "the host costs are overstated"),
                                                (4.0, 0.4, "partly"),
                                                (5.0, 0.2, "not the host costs"),
                                                (8.5, -0.5, "not the host costs")])
def test_q4(pred_h, phi, start):
    # residual 6 - 1 = 5 (5x the measurement); phi = (6 - pred_h) / 5, exact
    # at the two edges 0.5 and 0.2
    r = HD.q4_fraction(6.0, pred_h, 1.0)
    assert r["fraction"] == pytest.approx(phi) and r["verdict"].startswith(start)
    assert r["residual_rel"] == 5.0
    with pytest.raises(HD.DiagError):
        HD.q4_fraction(1.0, 0.9, 1.0)


# --------------------------------------------------------------------------
# the verdict over pulled manifests
# --------------------------------------------------------------------------

REF = D.p4_reference(ROOT / "docs" / "exploratory" / "results_20261116.json")


def test_p4_reference_holds_the_wave_numbers():
    d = REF[("dense", 16370)]
    assert d["attention_priced_s"] == pytest.approx(0.07415, abs=5e-5)
    assert d["attention_traced_s"] == pytest.approx(0.02943, abs=5e-5)
    assert d["measured_s"] == pytest.approx(0.1387, abs=5e-5)
    assert REF[("xattention", 8075)]["rel_error_all_device_replaced"] == pytest.approx(0.208, abs=5e-4)
    assert REF[("xattention", 32700)]["measured_s"] == pytest.approx(0.4961, abs=5e-5)


def eager(arm, backend, L, att, wall, host=None, nsys_host=None, status="passed"):
    m = {"status": status, "cell": vars(cell(f"{arm}{backend}{L}", arm, backend, L)),
         "untraced": {"median_s": wall},
         "torch_profiler": {"components_s": {"attention": att}, "host": {"host_s": host}},
         "nsys": {"available": True}}
    if nsys_host is not None:
        m["nsys"]["median"] = {"host_s": nsys_host}
    return m


def manifests(**over):
    """At P4's numbers: sdpa_flash at P4's priced attention, FA3 at the
    traced, the graph 1.5x its eager twin, walls at P4's."""
    r = REF
    man = {"off": eager("dense", "sdpa_flash", 16384, r[("dense", 16370)]["attention_priced_s"], 0.2),
           "g": {"status": "passed", "cell": vars(cell("g", mode="graph")),
                 "graph_components_s": {"attention": 1.5 * r[("dense", 16370)]["attention_traced_s"]}}}
    for L, n in D.TOKENS.items():
        man[f"d{L}"] = eager("dense", "fa3", L, r[("dense", n)]["attention_traced_s"],
                             r[("dense", n)]["measured_s"])
        man[f"x{L}"] = eager("xattention", None, L, 0.01, r[("xattention", n)]["measured_s"],
                             host=0.2, nsys_host=0.1)
    man.update(over)
    return man


def test_verdict_reads_every_rule():
    seen = []
    q4 = {n: (lambda h, n=n: seen.append((n, h)) or {"fraction": 0.6}) for n in D.TOKENS.values()}
    v = D.verdict(manifests(), REF, q4)
    assert v["Q1"]["confirmed"] and v["Q1"]["ratio"] == pytest.approx(1.0)
    assert v["Q2"]["ratio"] == pytest.approx(1.5)
    assert v["Q3"]["comparable"] and len(v["Q3"]["rows"]) == 9 and "note" not in v
    assert sorted(seen) == [(8075, 0.1), (16370, 0.1), (32700, 0.1)]
    assert v["Q4"]["16384"]["host_source"] == "nsys" and v["Q4"]["16384"]["fraction"] == 0.6


def test_verdict_falls_back_to_the_profiler_host_and_flags_q3():
    m = manifests()
    m["x16384"] = eager("xattention", None, 16384, 0.01, 0.5, host=0.2)   # 0.5 vs P4 0.2078
    q4 = {n: (lambda h: {"h": h}) for n in D.TOKENS.values()}
    v = D.verdict(m, REF, q4)
    assert v["Q4"]["16384"]["host_measured_s"] == 0.2
    assert v["Q4"]["16384"]["host_source"].startswith("torch.profiler")
    assert not v["Q3"]["rows"]["xattention_16384_wall"]["within"]
    assert not v["Q3"]["comparable"] and "Q3 failed" in v["note"]


def test_verdict_leaves_failed_or_missing_cells_unread():
    m = manifests(off={**manifests()["off"], "status": "failed"})
    del m["g"]
    v = D.verdict(m, REF, None)
    assert v["Q1"] == "not read" and v["Q2"] == "not read"
    assert v["Q4"] == {"8192": "not read", "16384": "not read", "32768": "not read"}


def test_verdict_records_a_rule_it_cannot_read():
    m = manifests(off=eager("dense", "sdpa_flash", 16384, 0.0, 0.2))
    assert D.verdict(m, REF, None)["Q1"].startswith("not read: Q1")


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

def fake_nsys(tmp_path, monkeypatch, fail=None):
    w = tmp_path / "nsys"
    w.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{ROOT / "tests" / "_nsys_fakes.py"}" "$@"\n')
    w.chmod(0o755)
    monkeypatch.setenv("H100_DIAG_NSYS", str(w))
    monkeypatch.setenv("FAKE_NSYS_NO_CHILD", "1")
    if fail:
        monkeypatch.setenv("FAKE_NSYS_FAIL", fail)


def nsys_args(tmp_path):
    return SimpleNamespace(cells=CELLS, warmup=1, prefills=4, cpu_boundary=True, nsys_timeout_s=60)


def test_nsys_pass_reads_the_export(tmp_path, monkeypatch):
    fake_nsys(tmp_path, monkeypatch)
    r = D.nsys_pass(cell("D16_fa3"), tmp_path, nsys_args(tmp_path))
    e = NF.expected()
    assert r["available"] and r["prefills"] == 4
    assert r["median"]["host_s"] == pytest.approx(e["host_s"])
    assert r["median"]["idle_share"] == pytest.approx(e["idle_s"] / e["wall_s"])
    assert r["attention_by_name_median_s"] == pytest.approx(e["kernels_s"]["flash_fwd_kernel"])


@pytest.mark.parametrize("step", ["profile", "export"])
def test_nsys_pass_records_a_failure(tmp_path, monkeypatch, step):
    fake_nsys(tmp_path, monkeypatch, fail=step)
    r = D.nsys_pass(cell("D16_fa3"), tmp_path, nsys_args(tmp_path))
    assert r["available"] and r["failed"].startswith(f"nsys {step}")


def test_nsys_pass_without_nsys(tmp_path, monkeypatch):
    monkeypatch.delenv("H100_DIAG_NSYS", raising=False)
    monkeypatch.setattr(D, "_nsys", lambda: None)
    assert D.nsys_pass(cell(), tmp_path, nsys_args(tmp_path)) == {"available": False}


def test_by_name_counts_attention_kernels():
    t = {"traceEvents": [{"cat": "kernel", "name": "flash_fwd_splitkv", "ts": 0, "dur": 3},
                         {"cat": "kernel", "name": "block_sparse_attn_fwd", "ts": 5, "dur": 2},
                         {"cat": "kernel", "name": "sm90_gemm", "ts": 9, "dur": 4},
                         {"cat": "cpu_op", "name": "flash", "ts": 0, "dur": 100}]}
    assert D.by_name(t, 0, float("inf")) == {"attention_s": pytest.approx(5e-6),
                                             "all_s": pytest.approx(9e-6)}


class Harness:
    def __init__(self, impl="xattn:triton"):
        self.impl = impl

    def prepare(self, arm, length, backend=None, batch=1):
        return SimpleNamespace(run_prefill=lambda: None, length=length, backend=object())

    def code_path(self, a):
        return SimpleNamespace(implementation=self.impl)

    def unwrap(self):
        pass


def run_args(tmp_path, **kw):
    return SimpleNamespace(cells=CELLS, out=tmp_path / "out", cpu_boundary=False, warmup=0,
                           prefills=1, reps=1, budget_min=55.0, nsys_timeout_s=60, **kw)


def test_run_refuses_xattention_off_triton(tmp_path):
    with pytest.raises(HD.DiagError, match="not Triton"):
        D.run_cell(Harness("xattn:eager"), cell("X", "xattention", None), tmp_path,
                   run_args(tmp_path))


def test_dense_backend_is_none_for_the_harness_backend_and_the_boundary():
    assert D.dense_backend(cell(backend="sdpa_flash"), False) is None
    assert D.dense_backend(cell(), True) is None
    assert D.dense_backend(cell(arm="xattention", backend=None), False) is None


def test_run_refuses_a_gpu_that_is_not_an_h100(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "environment", lambda a: {"gpu": "NVIDIA L4"})
    assert D.cmd_run(run_args(tmp_path)) == 2


def test_run_records_failures_and_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "environment", lambda a: {"gpu": "NVIDIA H100 80GB HBM3"})
    monkeypatch.setattr(D, "_harness", lambda a: Harness())
    order = []

    def run_cell(h, c, out, args):
        order.append(c.id)
        if c.id == "D16_off":
            raise RuntimeError("boom")
        if c.id == "X8":            # the budget runs out after the 4th cell
            args.budget_min = -1.0
        return {"cell": vars(c)}
    monkeypatch.setattr(D, "run_cell", run_cell)
    a = run_args(tmp_path)
    assert D.cmd_run(a) == 1
    assert order == ["D16_fa3", "D16_off", "D16_fa3_graph", "X8"]
    man = json.loads((a.out / "diag_manifest.json").read_text())
    st = {k: v["status"] for k, v in man["cells"].items()}
    assert st == {"D16_fa3": "passed", "D16_off": "failed", "D16_fa3_graph": "passed",
                  "X8": "passed", "X16": "skipped", "X32": "skipped", "D8_fa3": "skipped",
                  "D32_fa3": "skipped"}
    assert man["cells"]["D16_off"]["error"] == "RuntimeError: boom" and "finished" in man
    assert json.loads((a.out / "cell_D16_off_manifest.json").read_text())["status"] == "failed"
    assert man["cells_sha256"] == hashlib.sha256(CELLS.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Q4's recursion on P4's data (local only)
# --------------------------------------------------------------------------

Q4_ENV = ("H100_DIAG_Q4_INPUTS", "H100_DIAG_Q4_MIRROR", "H100_DIAG_Q4_TRACES")


@pytest.mark.skipif(not all(os.environ.get(k) for k in Q4_ENV), reason="P4's local data")
def test_q4_recursion_reproduces_p4_and_the_card_thresholds():
    fs = D.q4_functions(Path(os.environ[Q4_ENV[0]]), Path(os.environ[Q4_ENV[1]]),
                        os.environ[Q4_ENV[2]])
    assert set(fs) >= {8075, 16370, 32700}
    f = fs[8075]
    hp = f(1.0)["host_priced_s"]
    at = f(hp)
    assert at["fraction"] == pytest.approx(0.0, abs=1e-12)             # no change, nothing explained
    assert at["residual_rel"] == pytest.approx(0.208, abs=5e-4)        # P4's residual
    assert f(0.0985)["fraction"] == pytest.approx(0.5, abs=0.01)       # the card's thresholds
    assert f(0.1164)["fraction"] == pytest.approx(0.2, abs=0.01)
    assert fs[32700](0.2029)["fraction"] == pytest.approx(0.5, abs=0.01)
    assert re.search(r"0\.0985", CARD.read_text())
