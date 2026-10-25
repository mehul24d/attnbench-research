"""T2.6: the launch-queue depth probe (`attnbench/components/launch_queue.py`,
`scripts/measure_launch_queue.py`).

Q is recovered from a simulated queue whose depth the test sets, and the
simulated queue is itself checked against the recursion's queue term, so
"depth" means the same thing in the probe and in the model. CPU-only; Q is
measured on the L4 in the shakedown (T2.27).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench.components import launch_queue as lq
from attnbench.predict import recursion

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("depth", [1, 2, 7, 64, 1024])
def test_the_probe_recovers_the_true_depth(depth):
    sim = lq.SimulatedQueue(depth)
    times = lq.trial(sim.ops(), "sleep0", launches=depth + 50, blocker_s=0.5,
                     clock=sim.clock)
    got = lq.depth_from_trial(times, 0.5)
    assert got["q"] == depth
    assert got["blocked_wait_s"] > 0.25 * 0.5
    assert got["fast_launch_median_s"] in (None, pytest.approx(5e-6))


def test_a_queue_deeper_than_the_launches_gives_only_a_bound():
    sim = lq.SimulatedQueue(500)
    times = lq.trial(sim.ops(), "sleep0", launches=100, blocker_s=0.5, clock=sim.clock)
    got = lq.depth_from_trial(times, 0.5)
    assert got["q"] is None and got["q_lower_bound"] == 101


def test_a_blocker_that_ends_too_soon_gives_only_a_bound():
    """1000 launches at 5 us outlast a 1 ms blocker: no launch waits long, and
    the probe must not report the depth it never reached."""
    sim = lq.SimulatedQueue(64)
    times = lq.trial(sim.ops(), "sleep0", launches=1000, blocker_s=0.001, clock=sim.clock)
    assert lq.depth_from_trial(times, 0.001)["q"] in (None, 64)
    sim = lq.SimulatedQueue(5000)
    times = lq.trial(sim.ops(), "sleep0", launches=1000, blocker_s=0.001, clock=sim.clock)
    assert lq.depth_from_trial(times, 0.001)["q"] is None


def test_the_simulated_depth_is_the_recursions_q():
    """The simulated queue and the recursion agree on when each launch is
    issued, for the probe's own sequence: one blocker, then null kernels."""
    depth, host, kernel, block = 4, 5e-6, 2e-6, 0.01
    ops = [recursion.kernel_op(host, block)] + [recursion.kernel_op(host, kernel)
                                                for _ in range(10)]
    timeline = recursion.recurse(recursion.price(ops, {}), depth)
    sim = lq.SimulatedQueue(depth, host_ns=5_000, kernel_ns=2_000)
    o = sim.ops()
    issued = []
    o.blocker(block)
    issued.append(sim.t)
    launch = o.launcher("sleep0")
    for _ in range(10):
        launch()
        issued.append(sim.t)
    assert [round(t * 1e9) for t in timeline.host_issued] == issued


def test_a_row_carries_every_kind_and_agrees_across_trials():
    sim = lq.SimulatedQueue(32)
    row = lq.measure(sim.ops(), trials=3, launches=100, blocker_s=0.2, clock=sim.clock)
    for kind in lq.KINDS:
        assert row[f"{kind}_q"] == 32 and row[f"{kind}_q_trials"] == "32,32,32"
    assert lq.queue_depth(row, "sleep0") == 32
    pd.DataFrame([row])


def test_queue_depth_refuses_what_the_probe_did_not_establish():
    sim = lq.SimulatedQueue(32)
    row = lq.measure(sim.ops(), kinds=["sleep0"], trials=2, launches=100,
                     blocker_s=0.2, clock=sim.clock)
    with pytest.raises(KeyError):
        lq.queue_depth(row, "add")
    bad = {**row, "sleep0_q": None, "sleep0_q_trials": "32,31"}
    with pytest.raises(ValueError, match="no agreed Q"):
        lq.queue_depth(bad, "sleep0")
    with pytest.raises(ValueError, match="dry-run"):
        lq.queue_depth({**row, "dry_run": True}, "sleep0")


def test_disagreeing_trials_record_no_q():
    class Flaky:
        def __init__(self):
            self.depths = iter([32, 31])
            self.sim = None

        def ops(self):
            def sync():
                if self.sim is not None:
                    self.sim.ops().sync()

            def blocker(s):
                self.sim = lq.SimulatedQueue(next(self.depths))
                self.sim.ops().blocker(s)
            return lq.QueueOps(sync=sync, blocker=blocker,
                               launcher=lambda k: (lambda: self.sim.ops().launcher(k)()))

    f = Flaky()
    row = lq.measure(f.ops(), kinds=["sleep0"], trials=2, launches=100,
                     blocker_s=0.2, clock=lambda: f.sim.t if f.sim else 0)
    assert row["sleep0_q"] is None and row["sleep0_q_trials"] == "32,31"


def test_the_device_is_left_idle_after_a_failed_launch():
    synced = []

    def boom():
        raise RuntimeError("launch failed")

    ops = lq.QueueOps(sync=lambda: synced.append(1), blocker=lambda s: None,
                      launcher=lambda k: boom)
    with pytest.raises(RuntimeError):
        lq.trial(ops, "sleep0", launches=3, blocker_s=0.1)
    assert len(synced) == 2


def test_the_dry_run_writes_one_stamped_row(tmp_path):
    cmd = [sys.executable, str(REPO / "scripts" / "measure_launch_queue.py"),
           "--out", str(tmp_path), "--dry-run", "--dry-run-depth", "48",
           "--trials", "2", "--launches", "200", "--blocker-s", "0.2"]
    got = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert got.returncode == 0, got.stderr
    df = pd.read_parquet(tmp_path / "launch_queue.parquet")
    assert len(df) == 1 and bool(df["dry_run"][0]) and df["sleep0_q"][0] == 48
    assert {"git_commit", "python_optimize"} <= set(df.columns)
    assert subprocess.run(cmd, capture_output=True, text=True, timeout=300).returncode == 2
