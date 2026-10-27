"""T2.13's throttle detector: a sampler process reading the throttle-reason
bits at about 10 Hz (plan Section 5.6; Appendix A, 27 Oct 2026; module
`attnbench.clock_sampling`).

The card is a pair of shared values the test sets and the sampler process
reads, so what a test asserts about flags is scripted. The decision rule is
the researcher's: hardware slowdown, both thermal slowdowns and the hardware
power brake flag a block; the software power cap is recorded and never flags.
Real NVML is not exercised here (Shakedown 2 does that); the NVML reader's
call sequence is tested in `test_ground_truth_controls.py`.
"""

from __future__ import annotations

import multiprocessing
import time

import pytest

import _clock_fakes as fakes
from attnbench import clock_sampling as cs

INTERVAL = 0.01
SLOWDOWNS = cs.RESEARCHER_FLAG_BITS


@pytest.fixture
def card():
    ctx = multiprocessing.get_context("spawn")
    return ctx.Value("q", 0), ctx.Value("q", 2040)


@pytest.fixture
def sampler(card):
    with cs.ProcessSampler(fakes.shared_bits_reader, card, flag_bits=SLOWDOWNS,
                           interval_s=INTERVAL) as s:
        yield s


def _block(sampler, *, hold_s=0.08):
    sampler.begin()
    time.sleep(hold_s)
    return sampler.end()


def _settle(card, bits):
    card[0].value = bits
    time.sleep(6 * INTERVAL)           # several samples see the new state


# --- the researcher's set ----------------------------------------------------

def test_the_flag_set_is_the_researchers_and_excludes_the_software_power_cap():
    assert cs.RESEARCHER_FLAG_BITS == (cs.HW_SLOWDOWN | cs.SW_THERMAL_SLOWDOWN
                                       | cs.HW_THERMAL_SLOWDOWN
                                       | cs.HW_POWER_BRAKE_SLOWDOWN)
    assert not cs.RESEARCHER_FLAG_BITS & cs.SW_POWER_CAP
    assert not cs.RESEARCHER_FLAG_BITS & cs.NEVER_FLAGS


@pytest.mark.parametrize("bit", [cs.HW_SLOWDOWN, cs.SW_THERMAL_SLOWDOWN,
                                 cs.HW_THERMAL_SLOWDOWN, cs.HW_POWER_BRAKE_SLOWDOWN])
def test_each_of_the_four_slowdowns_flags_a_block(sampler, card, bit):
    _settle(card, bit)
    block = _block(sampler)
    assert block.flagged and block.flag_reasons == cs.reason_names(bit)


def test_the_software_power_cap_is_recorded_and_never_flags(sampler, card):
    _settle(card, cs.SW_POWER_CAP)
    block = _block(sampler)
    assert not block.flagged and block.flag_reasons == []
    assert "sw_power_cap" in block.to_dict()["throttle_reasons_seen"]


def test_a_cap_alongside_the_brake_flags_for_the_brake_only(sampler, card):
    _settle(card, cs.SW_POWER_CAP | cs.HW_POWER_BRAKE_SLOWDOWN)
    block = _block(sampler)
    assert block.flag_reasons == ["hw_power_brake_slowdown"]
    assert "sw_power_cap" in block.to_dict()["throttle_reasons_seen"]


def test_idle_and_the_session_clock_lock_never_flag(sampler, card):
    _settle(card, cs.GPU_IDLE | cs.APPLICATIONS_CLOCKS_SETTING)
    block = _block(sampler)
    assert not block.flagged
    assert "gpu_idle" in block.to_dict()["throttle_reasons_seen"]


def test_a_quiet_block_is_unflagged_and_carries_its_clocks(sampler, card):
    card[1].value = 1500
    time.sleep(5 * INTERVAL)
    block = _block(sampler)
    assert not block.flagged and block.samples >= 3
    assert block.sm_mhz_min == block.sm_mhz_max == 1500


# --- what the sampler sees ----------------------------------------------------

def test_a_throttle_that_began_inside_the_block_flags_it(sampler, card):
    sampler.begin()
    time.sleep(3 * INTERVAL)
    card[0].value = cs.HW_THERMAL_SLOWDOWN
    time.sleep(6 * INTERVAL)
    block = sampler.end()
    assert block.flagged and "hw_thermal_slowdown" in block.flag_reasons


def test_a_block_shorter_than_the_interval_is_still_bracketed():
    ctx = multiprocessing.get_context("spawn")
    bits, sm = ctx.Value("q", cs.HW_SLOWDOWN), ctx.Value("q", 2040)
    with cs.ProcessSampler(fakes.shared_bits_reader, (bits, sm), flag_bits=SLOWDOWNS,
                           interval_s=0.2) as s:
        s.begin()                       # no work at all
        block = s.end()
    assert block.samples >= 2 and block.flagged


def test_a_throttle_that_ended_long_before_the_block_does_not_flag_it(sampler, card):
    _settle(card, cs.HW_SLOWDOWN)
    _settle(card, 0)                    # the card recovered, with samples to prove it
    assert not _block(sampler).flagged


def test_the_reads_happen_in_another_process(card):
    s = cs.ProcessSampler(fakes.shared_bits_reader, card, flag_bits=SLOWDOWNS,
                          interval_s=INTERVAL)
    with s:
        assert s._proc.pid != multiprocessing.current_process().pid
        assert s._proc.is_alive()
    assert s._proc is None


# --- a sampler that is not working must stop the session -----------------------------

def test_a_dead_sampler_process_is_not_read_as_a_quiet_card(sampler, card):
    sampler.begin()
    sampler._proc.kill()
    sampler._proc.join()
    with pytest.raises(cs.ClockSamplingError, match="no longer running"):
        sampler.end()


def test_a_sampler_that_stalls_is_refused():
    with cs.ProcessSampler(fakes.sticks_after_first_read, (), flag_bits=SLOWDOWNS,
                           interval_s=INTERVAL, stall_s=0.3) as s:
        s.begin()
        with pytest.raises(cs.ClockSamplingError, match="stalled"):
            s.end()


def test_a_reader_that_cannot_be_built_is_reported_by_start():
    s = cs.ProcessSampler(fakes.unbuildable_reader, (), flag_bits=SLOWDOWNS,
                          interval_s=INTERVAL)
    with pytest.raises(cs.ClockSamplingError, match="no such device"):
        s.start()
    s.stop()


def test_the_interface_refusals(card):
    kw = dict(interval_s=INTERVAL)
    with pytest.raises(ValueError, match="not slowdowns"):
        cs.ProcessSampler(fakes.shared_bits_reader, card,
                          flag_bits=cs.GPU_IDLE | cs.HW_SLOWDOWN, **kw)
    with pytest.raises(ValueError, match="empty"):
        cs.ProcessSampler(fakes.shared_bits_reader, card, flag_bits=0, **kw)
    with pytest.raises(ValueError):
        cs.ProcessSampler(fakes.shared_bits_reader, card, flag_bits=SLOWDOWNS,
                          interval_s=0)
    with pytest.raises(TypeError):
        cs.ProcessSampler(fakes.shared_bits_reader, card)      # no default flag set
    s = cs.ProcessSampler(fakes.shared_bits_reader, card, flag_bits=SLOWDOWNS, **kw)
    with pytest.raises(cs.ClockSamplingError):
        s.begin()                                              # not started
    with s:
        with pytest.raises(cs.ClockSamplingError):
            s.end()                                            # end() without begin()
        s.begin()
        with pytest.raises(cs.ClockSamplingError):
            s.begin()
        s.abort()
        s.begin()
        s.end()


# --- in the harness ----------------------------------------------------------------------

def _rig(card, *, max_rerun=1):
    pytest.importorskip("transformers")
    from test_ground_truth_controls import Rig
    from attnbench import ground_truth as gt
    rig = Rig(max_rerun=max_rerun)
    proc = cs.ProcessSampler(fakes.shared_bits_reader, card, flag_bits=SLOWDOWNS,
                             interval_s=INTERVAL)
    rig.controls = gt.Controls(resolution_floor_ms=rig.controls.resolution_floor_ms,
                               max_rerun_rounds=max_rerun, sampler=proc)
    return rig, proc


def test_a_slowdown_flags_every_round_and_the_result_says_incomplete(card):
    rig, proc = _rig(card)
    card[0].value = cs.HW_THERMAL_SLOWDOWN
    with proc:
        time.sleep(5 * INTERVAL)
        out = rig.run(reps=2)
    assert not out["complete"] and out["clean_rounds"] == 0
    assert all(r["flagged"] for r in out["rounds"])
    assert "throttled" in out["rounds"][0]["flag_reasons"][0]
    assert out["controls"]["sampler_kind"] == "ProcessSampler"
    assert out["controls"]["flag_bits"] == cs.reason_names(SLOWDOWNS)


def test_a_power_capped_session_is_complete_and_the_cap_is_on_every_row(card):
    rig, proc = _rig(card)
    card[0].value = cs.SW_POWER_CAP
    with proc:
        time.sleep(5 * INTERVAL)
        out = rig.run(reps=2)
    assert out["complete"] and not any(r["flagged"] for r in out["rounds"])
    assert all("sw_power_cap" in r["throttle_reasons_seen"] for r in out["rows"])


# --- the counters of the in-process sampler: the cap must not flag through them ---

def test_the_power_counter_cannot_flag_when_it_holds_the_software_cap():
    """The power violation counter integrates the software cap and the hardware
    brake together. With only the brake in the flag set, a cap-only block must
    not flag through it; the brake is the process sampler's to see."""
    from _gt_fakes import FakeClock, ScriptedReader
    clock = FakeClock()
    reader = ScriptedReader(clock, windows=[(0.0, 10.0, "power")])
    brake_only = cs.BlockSampler(reader, flag_bits=cs.HW_POWER_BRAKE_SLOWDOWN)
    clock.t = 1.0
    brake_only.begin()
    clock.t = 2.0
    assert not brake_only.end().flagged
    both = cs.BlockSampler(reader, flag_bits=cs.SW_POWER_CAP | cs.HW_POWER_BRAKE_SLOWDOWN)
    clock.t = 3.0
    both.begin()
    clock.t = 4.0
    assert both.end().flagged            # a session that did ask for the cap gets it


def test_the_thermal_counter_still_flags_under_the_researchers_set():
    from _gt_fakes import FakeClock, ScriptedReader
    clock = FakeClock()
    reader = ScriptedReader(clock, windows=[(0.0, 10.0, "thermal")])
    s = cs.BlockSampler(reader, flag_bits=SLOWDOWNS)
    clock.t = 1.0
    s.begin()
    clock.t = 2.0
    assert s.end().flagged
