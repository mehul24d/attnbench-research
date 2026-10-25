"""T2.5: link microbenchmarks (`attnbench/components/link.py`,
`scripts/measure_link.py`).

A virtual device keeps one clock: every operation advances it by a cost
written in the test, so each s_fixed and bandwidth the code recovers is
checked against the number put in, and the readers' output is checked
against the recursion's own s_0. CPU-only; the real CUDA operations run in
the L4 shakedown (T2.27).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from attnbench.components import link
from attnbench.predict import recursion

REPO = Path(__file__).resolve().parents[1]

FIXED_NS = {"synchronize": 4_000, "item": 9_000, "bool": 10_000,
            "copy_to_host_pinned": 12_000, "copy_to_host_pageable": 15_000}
BW = {("h2d", "pinned"): 24e9, ("h2d", "pageable"): 8e9,
      ("d2h", "pinned"): 25e9, ("d2h", "pageable"): 6e9}
BLOCKING_BW = {"pinned": 20e9, "pageable": 5e9}


class VirtualDevice:
    """One clock in ns; operations advance it. `log` records the order."""

    def __init__(self):
        self.t = 0
        self.log: list[str] = []

    def clock(self) -> int:
        return self.t

    def ops(self) -> link.DeviceOps:
        def sync():
            self.log.append("sync")

        def mechanism(name):
            def op():
                self.log.append(name)
                self.t += FIXED_NS[name]
            return op

        def copy(direction, memory, n):
            return lambda: n / (BW[direction, memory])

        def blocking(memory, n):
            mech = f"copy_to_host_{memory}"

            def op():
                self.t += FIXED_NS[mech] + round(
                    n / (BLOCKING_BW[memory]) * 1e9)
            return op

        return link.DeviceOps(sync=sync, mechanism=mechanism, copy=copy,
                              blocking_d2h=blocking, device_time=lambda fn: fn())


def test_every_host_repetition_starts_from_an_idle_device():
    dev = VirtualDevice()
    got = link.time_host(dev.ops().mechanism("item"), 3, sync=dev.ops().sync,
                         clock=dev.clock)
    assert got == pytest.approx([9e-6] * 3)
    assert dev.log == ["sync", "item", "sync", "item", "sync", "item", "sync"]


def test_s_fixed_covers_exactly_the_recursions_mechanisms():
    dev = VirtualDevice()
    table = link.s_fixed_table(dev.ops(), 5, clock=dev.clock)
    assert tuple(table) == recursion.SYNC_MECHANISMS
    for m, ns in FIXED_NS.items():
        assert table[m]["median"] == pytest.approx(ns * 1e-9)


def test_bandwidths_are_recovered_by_hand():
    dev = VirtualDevice()
    fixed = link.s_fixed_table(dev.ops(), 3, clock=dev.clock)
    tables = link.bandwidth_tables(dev.ops(), [1 << 20, 1 << 24], 3, fixed,
                                   clock=dev.clock)
    for (direction, memory), bw in BW.items():
        assert link.headline(tables[f"{direction}_{memory}"]) == pytest.approx(bw)
    for memory, bw in BLOCKING_BW.items():
        # B = payload / (t - s_fixed): the fixed part is taken out, not fitted.
        assert link.headline(tables[f"blocking_d2h_{memory}"]) == pytest.approx(bw, rel=1e-6)


def test_a_blocking_copy_no_slower_than_s_fixed_has_no_bandwidth():
    dev = VirtualDevice()
    fixed = {m: {"median": 1.0} for m in recursion.SYNC_MECHANISMS}   # absurdly large
    tables = link.bandwidth_tables(dev.ops(), [1024], 1, fixed, clock=dev.clock)
    assert tables["blocking_d2h_pinned"][0]["bytes_per_s"] is None


def _row(**kw):
    dev = VirtualDevice()
    return link.measure(dev.ops(), sync_reps=3, copy_reps=2,
                        sizes=[1 << 20, 1 << 24], context_lengths=(2048,),
                        load="python_loop", clock=dev.clock, **kw)


def test_one_row_carries_every_field():
    row = _row()
    for m in recursion.SYNC_MECHANISMS:
        assert row[f"s_fixed_{m}_median"] == pytest.approx(FIXED_NS[m] * 1e-9)
    for key in ("h2d_pinned", "h2d_pageable", "d2h_pinned", "d2h_pageable",
                "blocking_d2h_pinned", "blocking_d2h_pageable"):
        assert row[f"bw_{key}_Bps"] > 0
        assert row[f"loaded_bw_{key}_2048_Bps"] > 0
    assert row["load"] == "python_loop" and row["load_iterations_2048"] is not None
    for k in ("gen_current", "gen_max", "width_current", "width_max"):
        assert f"pcie_{k}_under_copy" in row and f"pcie_{k}_idle" in row
    assert {"gpu_numa_node", "process_cpu_affinity", "affinity_on_gpu_node"} <= set(row)
    pd.DataFrame([row])            # one flat row a parquet can hold


def test_the_readers_give_the_recursion_its_s0():
    """s_0(m, payload) = s_fixed(m) + payload / B, from the row alone."""
    row = _row()
    costs = link.mechanism_costs(row, context_length=2048)
    assert set(costs) == set(recursion.SYNC_MECHANISMS)
    payload = 1 << 20
    s0 = recursion.s0(costs["copy_to_host_pinned"], payload, loaded=False)
    assert s0 == pytest.approx(12e-6 + payload / 20e9, rel=1e-6)
    assert costs["synchronize"].bandwidth is None
    with pytest.raises(recursion.CostModelError):
        recursion.s0(costs["item"], 8, loaded=False)     # item moves no payload here
    hw = link.hardware_features(row)
    assert (hw.pinned_bandwidth, hw.pageable_bandwidth) == pytest.approx((24e9, 8e9))


def test_the_readers_refuse_a_dry_run_row():
    row = _row(dry_run=True)
    with pytest.raises(ValueError, match="dry-run"):
        link.mechanism_costs(row)
    with pytest.raises(ValueError, match="dry-run"):
        link.hardware_features(row)


# --- the CPU load ------------------------------------------------------------------------

def test_the_load_runs_in_another_process_while_the_block_runs():
    with link.CpuLoad("python_loop", 2048, 0.5) as cpu:
        assert cpu._proc.is_alive() and cpu._proc.pid != os.getpid()
        time.sleep(1.0)
    assert cpu.iterations > 0
    assert not cpu._proc.is_alive()


def test_the_reference_builder_load_builds_a_mask():
    build = link._reference_builder_load(2048, 0.5)
    mask = build()
    assert mask.active.shape == (16, 16)


def test_an_unknown_load_is_refused():
    with pytest.raises(KeyError):
        link.CpuLoad("gpu_stress", 2048, 0.5)


# --- PCIe and NUMA -----------------------------------------------------------------------

def test_the_pcie_query_is_parsed(monkeypatch):
    from attnbench import provenance
    monkeypatch.setattr(provenance, "_sh_result", lambda cmd: (True, "4, 4, 16, 16\n"))
    assert link.query_pcie() == {"gen_current": 4, "gen_max": 4,
                                 "width_current": 16, "width_max": 16}
    monkeypatch.setattr(provenance, "_sh_result", lambda cmd: (False, ""))
    assert link.query_pcie() is None


def test_the_link_is_read_while_a_copy_runs():
    copies = []
    seen = []
    link.pcie_during(lambda: (copies.append(1), time.sleep(0.001)),
                     lambda: seen.append(len(copies)) or {}, settle_s=0.05)
    assert seen and seen[0] > 0


def _sysfs(tmp_path, node="0", cpulist="0-3,8-11"):
    d = tmp_path / "bus/pci/devices/0000:00:04.0"
    d.mkdir(parents=True)
    (d / "numa_node").write_text(node + "\n")
    n = tmp_path / f"devices/system/node/node{node}"
    if node != "-1":
        n.mkdir(parents=True)
        (n / "cpulist").write_text(cpulist + "\n")
    return str(tmp_path)


def test_numa_placement_inside_and_outside_the_gpus_node(tmp_path):
    sysfs = _sysfs(tmp_path)
    inside = link.numa_placement("00000000:00:04.0", sysfs=sysfs, affinity={0, 9})
    assert inside["gpu_numa_node"] == 0 and inside["affinity_on_gpu_node"] is True
    assert inside["numa_node_cpus"] == "0,1,2,3,8,9,10,11"
    outside = link.numa_placement("00000000:00:04.0", sysfs=sysfs, affinity={0, 4})
    assert outside["affinity_on_gpu_node"] is False


def test_a_host_without_numa_information_says_so(tmp_path):
    got = link.numa_placement("00000000:00:04.0", sysfs=_sysfs(tmp_path, "-1"),
                              affinity={0})
    assert got["gpu_numa_node"] == -1 and got["affinity_on_gpu_node"] is None


def test_the_bus_id_is_converted_to_sysfs_form():
    assert link._sysfs_bus_id("00000000:00:04.0") == "0000:00:04.0"
    assert link._sysfs_bus_id("00000001:3B:00.0") == "0001:3b:00.0"
    assert link._sysfs_bus_id("garbage") is None


# --- the script --------------------------------------------------------------------------

def _script(out, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "measure_link.py"), "--out", str(out),
         "--dry-run", "--sync-reps", "3", "--copy-reps", "2", "--sizes", "1024,4096",
         "--context-lengths", "2048", "--load", "python_loop", *extra],
        capture_output=True, text=True, timeout=300)


def test_the_dry_run_writes_one_stamped_row_and_refuses_a_second(tmp_path):
    first = _script(tmp_path)
    assert first.returncode == 0, first.stderr
    df = pd.read_parquet(tmp_path / "link.parquet")
    assert len(df) == 1 and bool(df["dry_run"][0])
    assert {"git_commit", "git_dirty", "python_optimize"} <= set(df.columns)
    assert "DRY RUN" in first.stdout
    again = _script(tmp_path)
    assert again.returncode == 2
