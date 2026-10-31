"""Fixes for audit C3, finding 1 (clock_sampling.py): hardware slowdown, the
thermal slowdowns and the hardware power brake flag a block; the software
power cap is recorded and never flags. Fakes only: the real NVML reader is
UNRUN here. Written before the fix. The ground-truth `Controls` half is in
test_audit_fixes_ground_truth.py (it imports torch).
"""

from __future__ import annotations

import threading
import time

import pytest

from attnbench import clock_sampling as cs

from _clock_fakes import shared_bits_reader

FLAG = cs.RESEARCHER_FLAG_BITS
INJECTED = [cs.HW_SLOWDOWN, cs.SW_THERMAL_SLOWDOWN, cs.HW_THERMAL_SLOWDOWN,
            cs.HW_POWER_BRAKE_SLOWDOWN]
IDS = ["hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"]


class _InBlockCard:
    """No counters. During the block (between `on()` and `off()`) the card
    reports `bits`; at the edges it is idle, as after the closing synchronise."""

    observes_transients = False

    def __init__(self, bits):
        self.bits, self.in_block = bits, threading.Event()

    def read(self) -> cs.ClockSample:
        b = self.bits if self.in_block.is_set() else cs.GPU_IDLE
        return cs.ClockSample(sm_mhz=1500, mem_mhz=6250, throttle_bits=b)


def _thread_block(bits):
    card = _InBlockCard(bits)
    s = cs.BlockSampler(card, flag_bits=FLAG, interval_s=0.001)
    s.begin()
    card.in_block.set()
    time.sleep(0.05)
    card.in_block.clear()
    return s.end()


def test_the_researchers_set_is_the_three_slowdown_kinds_and_not_the_cap():
    assert FLAG == (cs.HW_SLOWDOWN | cs.SW_THERMAL_SLOWDOWN | cs.HW_THERMAL_SLOWDOWN
                    | cs.HW_POWER_BRAKE_SLOWDOWN)
    assert not FLAG & cs.SW_POWER_CAP
    assert cs.RECORDED_ONLY == cs.SW_POWER_CAP


@pytest.mark.parametrize("bit", INJECTED, ids=IDS)
def test_an_injected_slowdown_flags_a_block_seen_inside_it(bit):
    block = _thread_block(bit)
    assert block.flagged and cs.reason_names(bit)[0] in block.flag_reasons


def test_the_software_power_cap_is_recorded_and_never_flags():
    block = _thread_block(cs.SW_POWER_CAP)
    assert not block.flagged
    assert "sw_power_cap" in block.to_dict()["throttle_reasons_seen"]


@pytest.mark.parametrize("make", [
    lambda bits: cs.BlockSampler(_InBlockCard(0), flag_bits=bits, interval_s=0.01),
    lambda bits: cs.ProcessSampler(shared_bits_reader, (None, None), flag_bits=bits),
], ids=["block", "process"])
def test_the_software_power_cap_cannot_be_chosen_to_flag(make):
    with pytest.raises(ValueError, match="software power cap"):
        make(FLAG | cs.SW_POWER_CAP)
    with pytest.raises(ValueError, match="software power cap"):
        make(cs.SW_POWER_CAP)


def test_a_sampler_says_whether_it_sees_inside_the_block():
    class Counters:
        observes_transients = True

        def read(self):
            return cs.ClockSample(1500, 6250, 0, {"power": 0, "thermal": 0})
    assert not cs.BlockSampler(Counters(), flag_bits=FLAG).sees_in_block
    assert cs.BlockSampler(Counters(), flag_bits=FLAG, interval_s=0.01).sees_in_block
    assert cs.ProcessSampler(shared_bits_reader, (None, None), flag_bits=FLAG).sees_in_block


@pytest.mark.parametrize("bit, counter_ns", [
    (cs.HW_SLOWDOWN, 0),                       # no counter integrates it
    (cs.HW_POWER_BRAKE_SLOWDOWN, 5_000_000),   # the power counter holds the cap too
], ids=["hw_slowdown", "hw_power_brake"])
def test_a_counters_only_sampler_cannot_see_these_bits(bit, counter_ns):
    """Why Controls refuses one (audit C3, finding 1): a braked block is clean
    to a counters-only sampler."""
    class Card:
        observes_transients = True

        def __init__(self):
            self.power = 0

        def read(self):
            return cs.ClockSample(2040, 6250, cs.GPU_IDLE, {"power": self.power, "thermal": 0})
    card = Card()
    s = cs.BlockSampler(card, flag_bits=FLAG)
    s.begin()
    card.power += counter_ns
    assert not s.end().flagged
    assert cs.counters_only_misses(FLAG) & bit
