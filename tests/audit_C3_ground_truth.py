"""Audit C3: `ground_truth.py`, `clock_sampling.py`, `dense_selection.py` and
`precomputed_masks.py` against plan Section 5.3 and tasks T2.13, T2.16 and
T2.12. Each test fails on main at c93d26d and demonstrates one finding. The
file is not named test_*.py, so the default suite does not collect it; run
it by name.

Tests marked UNRUN need torch (and transformers) at run time: they were
written in a CPU container without torch and skip there through
`pytest.importorskip`; they must be run, and seen to fail, where torch is
installed. The clock-sampling test needs neither and ran.
"""

from __future__ import annotations

import pytest

from attnbench import clock_sampling as cs


# ---------------------------------------------------------------------------
# Finding 1 (medium). T2.13: "The session's detector is a separate sampler
# process reading the throttle-reason bits at about 10 Hz (`ProcessSampler`),
# because hardware slowdown has no counter and the power counter holds the
# software cap and the hardware brake together." Section 5.3 step 1: "GPU
# clocks and throttle reasons sampled during every timed block."
# ground_truth.Controls (ground_truth.py:328-340) still takes a counters-only
# BlockSampler (no in-block thread), and BlockSampler accepts the researcher's
# flag set with no thread (clock_sampling.py:224-244). Such a sampler reads
# the bits only at the block's edges, where the GPU is idle, and its counters
# cannot flag hardware slowdown (no counter) or the hardware power brake (the
# power counter also integrates the software cap, so end() never lets it
# flag, clock_sampling.py:306-312). Two of the four flag bits are then
# invisible, and a braked block counts as clean ground truth.

class _CounterReader:
    """A card with cumulative counters. During the block the throttle bits
    are `during`; at the edges the GPU is idle. The power counter moves by
    `power_ns` across the block (the brake is in the power policy)."""

    observes_transients = True

    def __init__(self, during: int, power_ns: int):
        self.during, self.power_ns, self.in_block = during, power_ns, False
        self.power = self.thermal = 0

    def read(self) -> cs.ClockSample:
        bits = self.during if self.in_block else cs.GPU_IDLE
        return cs.ClockSample(sm_mhz=1500 if self.in_block else 2040, mem_mhz=6250,
                              throttle_bits=bits,
                              violation_ns={"power": self.power, "thermal": self.thermal})


@pytest.mark.parametrize("bit, power_ns", [
    (cs.HW_SLOWDOWN, 0),                       # no counter integrates it
    (cs.HW_POWER_BRAKE_SLOWDOWN, 5_000_000),   # the power counter moves
], ids=["hw_slowdown", "hw_power_brake"])
def test_a_sampler_that_cannot_see_a_flag_bit_is_refused_or_sees_it(bit, power_ns):
    reader = _CounterReader(bit, power_ns)
    try:
        sampler = cs.BlockSampler(reader, flag_bits=cs.RESEARCHER_FLAG_BITS)
    except (cs.ClockSamplingError, ValueError):
        return                                  # refused: the fix's first form
    sampler.begin()
    reader.in_block = True                      # the timed block runs, braked
    reader.power += power_ns
    reader.in_block = False                     # closing synchronise: idle
    block = sampler.end()
    assert block.flagged, (
        f"a block under {cs.reason_names(bit)} was not flagged by a sampler "
        f"configured with the researcher's flag set")


# ---------------------------------------------------------------------------
# Finding 2 (medium). Section 5.3 step 7: "Mask density and executed density
# are recorded on every row." ground_truth.time_arms builds its rows
# (ground_truth.py:497-520) with timings, clocks and probe counts, but no
# mask density and no executed density, and it drops what run_measured
# returns. The ground-truth rows cannot show that an arm ran at the density
# its cell names, or that a backend fell back (K8).

def test_every_ground_truth_row_records_mask_and_executed_density():
    """UNRUN: needs torch and transformers (the harness's toy model)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_ground_truth_controls import Rig
    out = Rig().run()
    for row in out["rows"]:
        keys = set(row)
        assert any("mask_density" in k for k in keys), (row["arm"], sorted(keys))
        assert any("executed_density" in k for k in keys), (row["arm"], sorted(keys))


# ---------------------------------------------------------------------------
# Finding 3 (medium). Section 5.5, K2: "The *fastest* correct candidate has
# the lowest median untraced time per card, context length and batch ... The
# choice is recorded before the timed runs, and every speedup is stated
# against it." dense_selection.check_control (dense_selection.py:372-392)
# checks the backend name, completeness and the time, but takes no cell and
# no configuration: a record made for one context length or batch passes as
# the control for any other. The cuDNN fault above 8192 tokens is exactly the
# case where the choice differs by length.

def test_the_control_check_refuses_a_choice_recorded_for_another_cell():
    """UNRUN: needs torch (dense_selection imports gates and the backends)."""
    pytest.importorskip("torch")
    from attnbench import dense_selection as ds
    from test_dense_selection import BATCH, CARD, SEQ, _select
    ch = _select()                                   # recorded for (A100, 32, 1)
    other = ds.Cell(CARD, SEQ * 2, BATCH)
    with pytest.raises(ds.SelectionError):
        ds.check_control(ch, ch.chosen, ch.recorded_at + 1.0, cell=other)
