"""The P4 Spot blocks (researcher, 7 Nov 2026): numbers from the card, not typed in."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import t64_blocks as B  # noqa: E402

N = B.card_numbers(B.CARD.read_text())


def test_the_numbers_are_read_from_the_cards_tables():
    assert [n for n, _ in N["steps"]][:3] == ["prereqs", "preflight", "link"]
    assert len(N["steps"]) == 16 and N["setup_h"] == 0.6 and N["teardown_h"] == 0.15
    assert dict(N["steps"])["ground_truth"] == 71          # "1 + 70"
    assert sum(m for _, m in N["steps"]) == 195            # the card's "about 195 minutes"


def test_a_card_without_its_tables_is_refused():
    with pytest.raises(ValueError, match="not found"):
        B.card_numbers("nothing here")


def test_cap_is_rate_times_planned_hours_times_1_3_and_halt_is_1_5_times_the_hours():
    blocks = B.plan_blocks(N["steps"], N["setup_h"], N["teardown_h"], max_hours=2.2)
    for b in blocks:
        planned = N["setup_h"] + b.run_min / 60 + N["teardown_h"]
        assert b.cap_inr == math.ceil(668.04 * planned * 1.3)
        assert b.halt_min == math.ceil(1.5 * planned * 60)
        assert b.planned_h <= 2.2 + 0.2          # only the atomic group may run over a little
    assert [s for b in blocks for s in b.steps] == [n for n, _ in N["steps"]]


def test_predict_ground_truth_and_late_check_stay_in_one_block():
    for h in (2.0, 2.2, 2.5, 3.0):
        blocks = B.plan_blocks(N["steps"], N["setup_h"], N["teardown_h"], max_hours=h)
        holder = [b for b in blocks if "predict" in b.steps]
        assert len(holder) == 1 and {"ground_truth", "late_check"} <= set(holder[0].steps)


def test_the_halt_costs_more_than_the_cap_by_construction():
    """1.5 x the hours against 1.3 x the cost: a flagged inconsistency, not a hidden one."""
    b = B.plan_blocks(N["steps"], N["setup_h"], N["teardown_h"])[0]
    assert b.halt_exceeds_cap and b.cost_at_halt_inr / b.cap_inr == pytest.approx(1.5 / 1.3, abs=0.01)
    assert "ABOVE the cap" in B.report([b], 668.04)


def test_a_group_longer_than_a_block_is_refused():
    with pytest.raises(ValueError, match="more than a block"):
        B.plan_blocks(N["steps"], N["setup_h"], N["teardown_h"], max_hours=1.5)
    with pytest.raises(ValueError, match="exceed the block"):
        B.plan_blocks(N["steps"], 1.0, 1.0, max_hours=1.5)
