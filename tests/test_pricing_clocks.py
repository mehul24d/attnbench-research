"""NVML through the pricing steps, and the soak before step 6's priced replay
(adviser's rulings 3 and 4b of 9 Nov 2026, seventh paste)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from attnbench import pricing_clocks as PC  # noqa: E402
import t31_steps as T  # noqa: E402


def fake_factory(start: int = 1500):
    """A picklable reader factory for the sampler's process."""
    n = [0]

    def read():
        n[0] += 1
        return {"sm_mhz": start + n[0], "mem_mhz": 6251, "power_w": 70.0, "reasons": 4}
    return read


def test_the_sampler_runs_in_its_own_process_and_each_phase_is_summarised(tmp_path):
    with PC.PricingClocks(tmp_path, enabled=True, interval_s=0.01, factory=fake_factory) as pc:
        with pc.phase("soak"):
            import time
            time.sleep(0.15)
        with pc.phase("profiled_replay"):
            time.sleep(0.15)
    rec = json.loads((tmp_path / "pricing_clocks.json").read_text())
    assert rec["schema"] == PC.SCHEMA and rec["sampled"] and rec["problem"] is None
    for name in ("soak", "profiled_replay"):
        ph = rec["phases"][name]
        assert ph["samples"] >= 3 and ph["mem_mhz"]["median"] == 6251
        assert ph["power_w"]["max"] == 70.0 and ph["reasons_bits"] == 4
    assert rec["phases"]["profiled_replay"]["sm_mhz"]["min"] > \
        rec["phases"]["soak"]["sm_mhz"]["max"]                       # in time order
    assert (tmp_path / "pricing_clocks.jsonl").read_text().count("\n") >= 6


def test_without_nvml_the_phases_are_marked_and_nothing_is_sampled(tmp_path):
    with PC.PricingClocks(tmp_path, enabled=False) as pc:
        with pc.phase("record"):
            pass
    rec = json.loads((tmp_path / "pricing_clocks.json").read_text())
    assert rec["sampled"] is False and rec["phases"]["record"]["samples"] == 0
    assert not (tmp_path / "pricing_clocks.jsonl").exists()


def test_a_phase_holds_only_the_samples_inside_it():
    samples = [{"t": t, "sm_mhz": s, "mem_mhz": 1, "power_w": p, "reasons": r}
               for t, s, p, r in ((0.5, 900, 10.0, 1), (1.5, 1800, 70.0, 4), (2.5, 1200, 50.0, 8))]
    got = PC.summarise(samples, [{"name": "x", "t0": 1.0, "t1": 3.0}])["x"]
    assert got["samples"] == 2 and got["sm_mhz"] == {"median": 1500.0, "min": 1200, "max": 1800}
    assert got["reasons_bits"] == 12


class _Cap:
    def __init__(self, clock, per_replay):
        self.clock, self.per_replay, self.log = clock, per_replay, []
        self.backend = self

    def replay(self):
        self.log.append("replay")
        self.clock.t += self.per_replay

    def synchronize(self):
        self.log.append("sync")


class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize("per_replay, replays", [(1.0, 5), (3.0, 2), (10.0, 2)])
def test_the_soak_runs_back_to_back_synchronised_replays_for_its_seconds(per_replay, replays):
    clock = _Clock()
    cap = _Cap(clock, per_replay)
    got = T.soak(cap, seconds=5.0, min_replays=2, clock=clock)
    assert got["replays"] == replays
    assert cap.log == ["replay", "sync"] * replays      # never more than one replay in flight
    assert T.SOAK_S == 5.0 and T.SOAK_MIN_REPLAYS == 2
