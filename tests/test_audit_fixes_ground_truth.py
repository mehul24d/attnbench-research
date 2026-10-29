"""Fixes for audit C3 (ground_truth.py, dense_selection.py) and the
ground-truth `Controls` half of C3 finding 1. Written before the fixes.

`ground_truth` and `dense_selection` import torch at module level, so in a
torch-free container these tests skip. None of them computes with torch:
they were run here against an import-only stand-in for torch (kept out of
the repository), and must be run where torch is installed (UNRUN there).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from attnbench import clock_sampling as cs  # noqa: E402
from attnbench import ground_truth as gt  # noqa: E402

from _clock_fakes import shared_bits_reader  # noqa: E402


class _Counters:
    observes_transients = True

    def read(self):
        return cs.ClockSample(2040, 6250, 0, {"power": 0, "thermal": 0})


# --- C3 finding 1: Controls take only a sampler that sees inside the block -------------

def test_controls_refuse_a_counters_only_sampler():
    s = cs.BlockSampler(_Counters(), flag_bits=cs.RESEARCHER_FLAG_BITS)
    with pytest.raises(ValueError, match="counters-only"):
        gt.Controls(resolution_floor_ms=1.0, max_rerun_rounds=0, sampler=s)


def test_controls_refuse_a_counters_only_sampler_even_for_bits_it_could_count():
    s = cs.BlockSampler(_Counters(), flag_bits=cs.SW_THERMAL_SLOWDOWN | cs.HW_THERMAL_SLOWDOWN)
    with pytest.raises(ValueError, match="counters-only"):
        gt.Controls(1.0, 0, s)


@pytest.mark.parametrize("missing", [cs.HW_SLOWDOWN, cs.SW_THERMAL_SLOWDOWN,
                                     cs.HW_THERMAL_SLOWDOWN, cs.HW_POWER_BRAKE_SLOWDOWN],
                         ids=["hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"])
def test_controls_refuse_a_flag_set_without_the_researchers_bits(missing):
    s = cs.ProcessSampler(shared_bits_reader, (None, None),
                          flag_bits=cs.RESEARCHER_FLAG_BITS & ~missing)
    with pytest.raises(ValueError, match=cs.reason_names(missing)[0]):
        gt.Controls(1.0, 0, s)


def test_controls_take_the_process_sampler_and_an_in_block_thread():
    gt.Controls(1.0, 0, cs.ProcessSampler(shared_bits_reader, (None, None),
                                          flag_bits=cs.RESEARCHER_FLAG_BITS))
    gt.Controls(1.0, 0, cs.BlockSampler(_Counters(), flag_bits=cs.RESEARCHER_FLAG_BITS,
                                        interval_s=0.01))
