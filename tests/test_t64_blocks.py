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
    assert dict(N["steps"])["ground_truth"] == 49          # "2 + 28 + 5 + 14" (5 reruns)
    assert sum(m for _, m in N["steps"]) == 249            # the card's "249 minutes"


def test_a_card_without_its_tables_is_refused():
    with pytest.raises(ValueError, match="not found"):
        B.card_numbers("nothing here")


def plan(h=None, pairing="P4"):
    h = h or {"P2": 2.2, "P4": 2.3}[pairing]     # P4 at 2.2 h refuses (below)
    n = B.card_numbers(B.CARDS[pairing].read_text())
    return B.plan_blocks(n["steps"], n["setup_h"], n["teardown_h"], boot=B.BOOT[pairing],
                         rate=B.SPOT_RATES[pairing], max_hours=h, suite_min=n["suite_min"])


def test_the_spot_terms_are_cap_1_3_halt_1_3_rounded_down_and_delete_10_minutes_after():
    # researcher, 7 Nov 2026: halt = 1.3 x planned hours, hard delete = halt + 10 min
    for pairing in ("P4", "P2"):
        n, rate = B.card_numbers(B.CARDS[pairing].read_text()), B.SPOT_RATES[pairing]
        for i, b in enumerate(plan(pairing=pairing)):
            # the suite record (ruling 1h dated 10 Nov 2026) on the first boot only
            suite = n["suite_min"] if i == 0 else 0.0
            assert n["suite_min"] == 20
            planned = n["setup_h"] + (suite + b.run_min) / 60 + n["teardown_h"]
            assert b.cap_inr == math.ceil(rate * planned * 1.3)
            assert b.halt_min == math.floor(1.3 * planned * 60)
            assert b.delete_min == b.halt_min + 10
            assert b.cost_at_halt_inr == math.ceil(rate * b.halt_min / 60)


@pytest.mark.parametrize("h", [1.9, 2.0, 2.2, 2.5, 3.0])
@pytest.mark.parametrize("pairing", ["P4", "P2"])
def test_the_cost_at_the_halt_is_never_above_the_cap(h, pairing):
    # host_components, measured on T3.2d: 79 min; P4's boot steps take one more minute
    if h < {"P2": 2.1, "P4": 2.3}[pairing]:
        with pytest.raises(ValueError, match="host_components takes 79 min"):
            plan(h, pairing)
        return
    for b in plan(h, pairing):
        assert b.cost_at_halt_inr <= b.cap_inr, b
        assert not b.halt_exceeds_cap
        assert b.cost_at_delete_inr > b.cap_inr          # the backstop is past the cap


def test_p4_refuses_2_2_hour_blocks_and_is_four_blocks_at_2_3():
    with pytest.raises(ValueError, match="host_components takes 79 min, more than a block's 78"):
        plan(2.2)
    blocks = plan(2.3)
    assert len(blocks) == 4 and all(b.planned_h <= 2.3 + 1e-9 for b in blocks)
    assert [b.planned_h for b in blocks] == [1.85, 2.217, 2.3, 1.733]   # b1 + the 20 min suite record
    assert blocks[1].steps == ("host_components",)
    assert sum(b.ground_truth_min for b in blocks) == dict(N["steps"])["ground_truth"]
    gt = [b for b in blocks if "ground_truth" in b.steps]
    assert len(gt) == 2 and all({"late_check", "trace_checks"} <= set(b.steps) for b in gt)
    assert "predict" in gt[0].steps and "predict" not in gt[1].steps
    assert B.main(["--max-hours", "2.3", "--blocks", "4"]) == 0
    assert B.main(["--blocks", "4"]) == 0          # P4's default is the ruled 2.3 h
    assert B.main(["--max-hours", "2.3", "--blocks", "3"]) == 2


def test_every_step_but_the_boot_steps_is_planned_once_in_order():
    for pairing in ("P4", "P2"):
        names = [n for n, _ in B.card_numbers(B.CARDS[pairing].read_text())["steps"]]
        own = [s for b in plan(pairing=pairing) for s in b.steps
               if s not in ("ground_truth", "late_check", "trace_checks")]
        assert own == [n for n in names if n not in B.BOOT[pairing]
                       and n not in ("ground_truth", "late_check", "trace_checks")]


def test_each_block_pays_its_own_boot_steps_and_the_anchor():
    b = plan(2.3)[3]
    boot = sum(dict(N["steps"])[s] for s in B.BOOT["P4"])
    tail = B.ANCHOR_MIN + dict(N["steps"])["late_check"] + dict(N["steps"])["trace_checks"]
    assert b.run_min == pytest.approx(boot + b.ground_truth_min + tail)


def test_the_spec_is_the_lock_files_skeleton():
    import json
    s = B.spec(plan(2.3), max_hours=2.3)
    assert s["anchor"] is None and [b["id"] for b in s["blocks"]] == ["b1", "b2", "b3", "b4"]
    assert all(b["cells"] is None for b in s["blocks"] if "ground_truth" in b["steps"])
    assert "cells" not in s["blocks"][0]
    assert json.loads(json.dumps(s)) == s


def test_a_step_longer_than_a_block_is_refused():
    with pytest.raises(ValueError, match="more than a block"):
        B.plan_blocks(N["steps"], N["setup_h"], N["teardown_h"], max_hours=1.2)
    with pytest.raises(ValueError, match="fill the block"):
        B.plan_blocks(N["steps"], 1.0, 1.0, max_hours=1.5)
