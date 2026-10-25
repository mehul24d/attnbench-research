"""T2.7: trace checks flagged on synthetic traces
(`attnbench/analysis/trace_checks.py`, `scripts/check_trace.py`).

Kineto traces are written as chrome-trace JSON with hand-placed intervals;
Nsight Systems exports as sqlite files with page-fault tables. Each claim
the checks may make (Appendix A, 25 Oct 2026) is tested: Kineto at most "no
managed-memory use"; nsys "no page faults" only with a faulting control, and
only for the arms named. CPU-only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench.analysis import trace_checks as tc

REPO = Path(__file__).resolve().parents[1]


def ev(cat, name, ts, dur, stream=7, device=0):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur,
            "pid": device, "tid": stream, "args": {"device": device, "stream": stream}}


KERNEL = "void at::native::vectorized_elementwise_kernel<4>"
H2D = "Memcpy HtoD (Pinned -> Device)"


def test_serialised_copies_and_kernels_are_not_flagged():
    events = [ev("kernel", KERNEL, 0, 10), ev("gpu_memcpy", H2D, 10, 5),
              ev("kernel", KERNEL, 15, 3)]
    got = tc.copy_kernel_concurrency(events)
    assert not got["copy_kernel_concurrency"]
    assert got["cross_stream_overlaps"] == got["same_stream_overlaps"] == 0


def test_a_copy_on_another_stream_under_a_kernel_is_flagged():
    events = [ev("kernel", KERNEL, 0, 10, stream=7),
              ev("gpu_memcpy", H2D, 5, 3, stream=13)]
    got = tc.copy_kernel_concurrency(events)
    assert got["copy_kernel_concurrency"]
    assert got["cross_stream_overlaps"] == 1
    assert got["cross_stream_overlap_us_max"] == pytest.approx(3.0)
    assert got["cross_stream_pairs"] == ["13->7"]


def test_a_memset_counts_as_device_work():
    events = [ev("gpu_memset", "Memset (Device)", 0, 10, stream=7),
              ev("gpu_memcpy", H2D, 2, 3, stream=13)]
    assert tc.copy_kernel_concurrency(events)["copy_kernel_concurrency"]


def test_a_same_stream_overlap_is_a_timestamp_artifact_not_a_flag():
    events = [ev("kernel", KERNEL, 0, 10), ev("gpu_memcpy", H2D, 9.5, 3)]
    got = tc.copy_kernel_concurrency(events)
    assert not got["copy_kernel_concurrency"]
    assert got["same_stream_overlaps"] == 1
    assert got["same_stream_overlap_us_max"] == pytest.approx(0.5)


def test_overlap_against_several_kernels_is_summed():
    events = [ev("kernel", KERNEL, 0, 4, stream=7), ev("kernel", KERNEL, 6, 4, stream=7),
              ev("gpu_memcpy", H2D, 2, 6, stream=13)]           # [2, 8): 2 + 2
    got = tc.copy_kernel_concurrency(events)
    assert got["cross_stream_overlap_us_total"] == pytest.approx(4.0)


def test_other_devices_never_overlap():
    events = [ev("kernel", KERNEL, 0, 10, device=0),
              ev("gpu_memcpy", H2D, 2, 3, stream=13, device=1)]
    assert not tc.copy_kernel_concurrency(events)["copy_kernel_concurrency"]


def test_managed_memory_is_seen_in_copy_names_and_api_calls():
    clean = [ev("kernel", KERNEL, 0, 1), ev("gpu_memcpy", H2D, 1, 1)]
    assert tc.managed_memory_use(clean)["claim"] == "no managed-memory use"
    by_copy = clean + [ev("gpu_memcpy", "Memcpy HtoD (Managed -> Device)", 2, 1)]
    got = tc.managed_memory_use(by_copy)
    assert got["managed_memory_use"] and got["managed_copy_names"]
    by_api = clean + [ev("cuda_runtime", "cudaMallocManaged", 0, 1)]
    assert tc.managed_memory_use(by_api)["managed_api_calls"] == ["cudaMallocManaged"]


def test_a_kineto_check_never_claims_no_page_faults(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"traceEvents": [ev("kernel", KERNEL, 0, 1)]}))
    got = tc.check_kineto(p)
    assert "fault" not in got["claim"]


def test_a_trace_without_device_activity_is_refused(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"traceEvents": [ev("cpu_op", "aten::add", 0, 1)]}))
    with pytest.raises(ValueError, match="no device activity"):
        tc.check_kineto(p)


# --- nsys exports -------------------------------------------------------------------

def export(path, gpu_faults=0, cpu_faults=0, tables=True):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    if tables:
        for name, n in (("CUDA_UM_GPU_PAGE_FAULT_EVENTS", gpu_faults),
                        ("CUDA_UM_CPU_PAGE_FAULT_EVENTS", cpu_faults)):
            con.execute(f"CREATE TABLE {name} (start INTEGER, address INTEGER)")
            con.executemany(f"INSERT INTO {name} VALUES (?, ?)",
                            [(i, 4096 * i) for i in range(n)])
    con.commit()
    con.close()
    return path


def test_no_page_faults_needs_a_control_that_faulted(tmp_path):
    target = export(tmp_path / "t.sqlite")
    control = export(tmp_path / "c.sqlite", gpu_faults=40, cpu_faults=12)
    got = tc.check_page_faults(target, control, arms=["dense", "xattention"])
    assert got["page_faults"] == 0
    assert got["claim"] == "no page faults (arms traced: dense, xattention)"


def test_a_control_with_no_faults_leaves_the_answer_unknown(tmp_path):
    target = export(tmp_path / "t.sqlite")
    control = export(tmp_path / "c.sqlite")
    got = tc.check_page_faults(target, control, arms=["dense"])
    assert got["page_faults"] is None and got["claim"].startswith("unknown")


def test_faults_in_the_traced_pass_are_counted(tmp_path):
    target = export(tmp_path / "t.sqlite", gpu_faults=3, cpu_faults=1)
    control = export(tmp_path / "c.sqlite", gpu_faults=40)
    got = tc.check_page_faults(target, control, arms=["dense"])
    assert got["page_faults"] == 4 and got["claim"].startswith("page faults: 4")


def test_an_export_with_no_fault_tables_counts_as_zero_only_with_a_control(tmp_path):
    target = export(tmp_path / "t.sqlite", tables=False)
    assert tc.page_fault_counts(target) == {}
    control = export(tmp_path / "c.sqlite", gpu_faults=5)
    assert tc.check_page_faults(target, control, arms=["dense"])["page_faults"] == 0


def test_the_arms_must_be_named(tmp_path):
    e = export(tmp_path / "t.sqlite")
    with pytest.raises(ValueError, match="arms"):
        tc.check_page_faults(e, e, arms=[])


def test_nsys_is_looked_for_on_path_first():
    assert tc.nsys_available(which=lambda name: "/opt/x/bin/nsys") == "/opt/x/bin/nsys"


# --- the scripts -------------------------------------------------------------------------

def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                          text=True, timeout=300)


def test_the_kineto_subcommand_writes_a_stamped_row_and_signals_a_flag(tmp_path):
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"traceEvents": [ev("kernel", KERNEL, 0, 10),
                                                 ev("gpu_memcpy", H2D, 10, 5)]}))
    got = _run(REPO / "scripts/check_trace.py", "kineto", clean, "--out", tmp_path / "a")
    assert got.returncode == 0, got.stderr
    df = pd.read_parquet(tmp_path / "a" / "trace_check_kineto.parquet")
    assert len(df) == 1 and df["analysis_tool"][0] == "check_trace.kineto"
    flagged = tmp_path / "flagged.json"
    flagged.write_text(json.dumps({"traceEvents": [
        ev("kernel", KERNEL, 0, 10), ev("gpu_memcpy", H2D, 2, 3, stream=13)]}))
    got = _run(REPO / "scripts/check_trace.py", "kineto", flagged, "--out", tmp_path / "b")
    assert got.returncode == 1 and "FLAGGED" in got.stdout


def test_the_nsys_subcommand_prints_the_scoped_claim(tmp_path):
    t = export(tmp_path / "t.sqlite")
    c = export(tmp_path / "c.sqlite", gpu_faults=2)
    got = _run(REPO / "scripts/check_trace.py", "nsys-faults", t, "--control", c,
               "--arms", "dense", "--out", tmp_path)
    assert got.returncode == 0, got.stderr
    assert "no page faults (arms traced: dense)" in got.stdout


def test_the_fault_control_refuses_without_a_gpu():
    got = subprocess.run([sys.executable, str(REPO / "scripts/um_fault_control.py")],
                         capture_output=True, text=True, timeout=300,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert got.returncode == 2 and "CUDA" in got.stderr
