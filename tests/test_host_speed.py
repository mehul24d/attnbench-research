"""T2.1: the host-speed probe runs on CPU, and its GPU path is dry-run with
fakes (`attnbench/components/host_speed.py`, `host_cpu_probe.py speed`).

A fake clock makes every timing exact, so each figure the probe reports is
checked against a value worked out by hand rather than against itself.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench.components import host_speed

REPO = Path(__file__).resolve().parents[1]


class Clock:
    """perf_counter_ns that advances by the next step on every read."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.t = 0

    def __call__(self):
        self.t += self.steps.pop(0) if self.steps else 0
        return self.t


def test_the_loop_is_timed_per_iteration():
    # reads: start, end per rep -> durations 0+2000, 0+4000 ns over 1000 iterations
    clock = Clock([0, 2000, 0, 4000])
    got = host_speed.time_loop(2, 1000, clock=clock, loop=lambda n: None)
    assert got == pytest.approx([2e-9, 4e-9])


def test_the_loop_runs_with_the_collector_off_and_restores_it():
    seen = []
    host_speed.time_loop(1, 10, loop=lambda n: seen.append(gc.isenabled()))
    assert seen == [False] and gc.isenabled()


def test_the_loop_is_a_fixed_workload():
    assert host_speed.single_thread_loop(1000) == host_speed.single_thread_loop(1000)


def test_summarise_by_hand():
    s = host_speed.summarise([1.0, 2.0, 3.0, 4.0, 10.0])
    assert s["median"] == 3.0 and s["p10"] == 1.0 and s["p90"] == 10.0
    assert s["mean"] == 4.0
    assert s["stdev"] == pytest.approx(3.5355339, rel=1e-6)
    assert s["cv"] == pytest.approx(3.5355339 / 4.0, rel=1e-6)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_counts_are_refused(bad):
    with pytest.raises(ValueError):
        host_speed.time_loop(bad, 10)


# --- the GPU path, dry run ----------------------------------------------------------

class FakeDevice:
    def __init__(self):
        self.log = []

    def launch(self):
        self.log.append("launch")

    def sync(self):
        self.log.append("sync")


def test_a_burst_syncs_outside_the_timer_on_both_sides():
    dev = FakeDevice()
    clock = Clock([0, 3000, 0, 5000, 0, 4000])
    got = host_speed.null_kernel_burst(dev.launch, dev.sync, 3, clock=clock)
    assert got == pytest.approx([3e-6, 5e-6, 4e-6])
    assert dev.log == ["sync", "launch", "launch", "launch", "sync"]


def test_a_failing_launch_still_leaves_the_device_synchronised():
    dev = FakeDevice()

    def boom():
        raise RuntimeError("launch failed")

    with pytest.raises(RuntimeError):
        host_speed.null_kernel_burst(boom, dev.sync, 3)
    assert dev.log == ["sync", "sync"] and gc.isenabled()


def test_the_cuda_path_launches_the_null_kernel(monkeypatch):
    import torch
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "_sleep", lambda cycles: calls.append(("sleep", cycles)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(("sync",)))
    launch, sync = host_speed.cuda_null_kernel()
    host_speed.null_kernel_burst(launch, sync, 2)
    assert calls == [("sync",), ("sleep", 0), ("sleep", 0), ("sync",)]


def test_without_cuda_the_cuda_path_refuses(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        host_speed.cuda_null_kernel()


def test_a_probe_row_pools_every_launch_of_every_burst():
    dev = FakeDevice()
    row = host_speed.probe("start", loop_reps=2, loop_iterations=10, burst=4,
                           bursts=3, null_kernel=(dev.launch, dev.sync),
                           null_path="dry-run")
    assert row["null_launch_n"] == 12 and dev.log.count("launch") == 12
    assert dev.log.count("sync") == 6
    assert row["null_path"] == "dry-run" and row["null_burst"] == 4
    assert row["null_launch_s_median"] is not None


def test_a_probe_without_a_gpu_path_says_so():
    row = host_speed.probe("start", loop_reps=2, loop_iterations=10)
    assert row["null_path"] == "none" and row["null_launch_n"] == 0
    assert row["null_launch_s_median"] is None
    assert row["loop_s_per_iter_median"] > 0


# --- the script ------------------------------------------------------------------------

def _speed(out: Path, tag: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "host_cpu_probe.py"), "speed",
         "--out", str(out), "--tag", tag, "--loop-reps", "3",
         "--loop-iterations", "1000", "--burst", "2", "--bursts", "2", *extra],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_the_script_writes_one_stamped_row_and_never_reuses_a_tag(tmp_path):
    first = _speed(tmp_path, "start")
    assert first.returncode == 0, first.stderr
    df = pd.read_parquet(tmp_path / "host_speed_start.parquet")
    assert len(df) == 1 and df["tag"][0] == "start"
    assert {"git_commit", "python_optimize", "provisioning_model"} <= set(df.columns)
    assert df["null_path"][0] == "none"
    again = _speed(tmp_path, "start")
    assert again.returncode == 2 and "used once" in again.stderr


def test_the_session_runner_can_require_the_gpu_path(tmp_path):
    got = _speed(tmp_path, "start", "--require-gpu")
    assert got.returncode == 3
    assert not (tmp_path / "host_speed_start.parquet").exists()
