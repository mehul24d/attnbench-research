"""The budget interlock (`scripts/budget_interlock.py`; adviser's ruling 1g
dated 10 Nov 2026): at most one of P2 and P4 on Spot, and no launch whose
headroom after every still-planned cap is negative."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import budget_interlock as BI  # noqa: E402

LEDGER = """\
### New-spend headroom against the ₹14,000 budget (test)

| row | est ₹ |
|---|---|
| a | 1,000 |
| b (estimate) | ~200 |
| **Total** | **~1,200** |

Headroom: ...
"""


def test_spent_sums_the_rows_and_checks_the_total():
    assert BI.spent(LEDGER) == 1200
    with pytest.raises(ValueError, match="sum to 1200"):
        BI.spent(LEDGER.replace("~1,200", "~1,300"))
    with pytest.raises(ValueError, match="no section"):
        BI.spent("nothing")


def test_the_real_ledger_and_rehearsal_card_read():
    s = BI.spent(BI.LEDGER.read_text())
    card, done = BI.rehearsal_cap(BI.REHEARSAL_CARD.read_text())
    assert s > 0 and card > 0 and 0 < done <= s


CAPS = {"P2": {"FLEX_START": 1000, "SPOT": 3000}, "P4": {"FLEX_START": 2000, "SPOT": 6000}}


def test_both_on_spot_is_never_allowed_even_with_room():
    cs = BI.cases(0, 0, CAPS)
    both = [c for c in cs if c.p2 == c.p4 == "SPOT"][0]
    assert both.headroom > 0 and not both.allowed
    assert all(c.allowed for c in cs if c is not both)


def test_a_launch_counts_the_other_pairing_at_its_dearest_allowed_provisioning():
    cs = BI.cases(4000, 250, CAPS)
    # P2 on Flex-start: P4 may still go Spot, so P4 is counted at 6000
    assert BI.launch_headroom("P2", "FLEX_START", cs) == 14000 - 4250 - 1000 - 6000
    # P2 on Spot: P4 is held to Flex-start, so 2000
    assert BI.launch_headroom("P2", "SPOT", cs) == 14000 - 4250 - 3000 - 2000
    assert BI.launch_headroom("P4", "SPOT", cs) == 14000 - 4250 - 1000 - 6000


def test_a_negative_headroom_refuses_the_launch():
    cs = BI.cases(7000, 250, CAPS)        # P2 Flex + P4 Spot: 14,250, over by 250
    assert BI.launch_headroom("P2", "FLEX_START", cs) == -250
    assert not BI.may_launch("P2", "FLEX_START", cs)
    assert BI.may_launch("P2", "SPOT", cs)
    assert not [c for c in cs if c.p2 == "FLEX_START" and c.p4 == "SPOT"][0].allowed


def test_the_cli_exits_2_on_a_refused_launch(monkeypatch, capsys):
    monkeypatch.setattr(BI, "RAN", {})       # the mechanics, before P2 ran
    monkeypatch.setattr(BI, "current", lambda: (7000, (250, 233), CAPS, BI.cases(7000, 250, CAPS)))
    assert BI.main(["--launch", "P2:FLEX_START"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert BI.main(["--launch", "P2:SPOT"]) == 0


def test_launch_4_is_counted_at_the_larger_reading(monkeypatch):
    monkeypatch.setattr(BI, "RAN", {})       # the mechanics, before P2 ran
    monkeypatch.setattr(BI, "pairing_caps", lambda: CAPS)
    monkeypatch.setattr(BI, "rehearsal_cap", lambda text: (100, 233))   # 403 - 233 = 170 > 100
    monkeypatch.setattr(BI, "rehearsal_closed", lambda text: None)
    s, _, _, cs = BI.current()
    assert cs[0].committed == s + 170 + 1000 + 2000


def test_both_cards_carry_the_current_interlock():
    for card, pairing in ((ROOT / "docs/launch_cards/t61_p2_a100.md", "P2"),
                          (ROOT / "docs/launch_cards/t64_p4_h100.md", "P4")):
        assert "\n".join(BI.section(pairing)) in card.read_text()


def test_a_closed_rehearsal_reserves_nothing_and_its_spend_is_the_ledgers(monkeypatch):
    # The adviser's ruling 6, fourth set of 12 Nov 2026: launch 4 A's Rs 71, Rs 304 in all.
    text = BI.REHEARSAL_CARD.read_text()
    assert BI.rehearsal_closed(text) == 304
    assert BI.rehearsal_closed(text.replace("The rehearsal closed on", "It may close on")) is None
    with pytest.raises(ValueError, match="not what launch 4 A used"):
        BI.rehearsal_closed(text.replace("launch 4 A used", "A used"))
    monkeypatch.setattr(BI, "pairing_caps", lambda: CAPS)
    s, _, _, cs = BI.current()
    assert BI.rehearsal_closed(text) <= s
    assert cs[0].committed == s + 0 + 2000      # nothing reserved; P2 ran (RAN)


# --- a ruled-out provisioning (the adviser's rulings dated 13 Nov 2026, item 6) ---------------

def test_a_ruled_out_provisioning_is_neither_launched_nor_counted():
    cs = BI.cases(4000, 250, CAPS, {("P4", "SPOT"): "x"})
    # P2 on Flex-start: P4 may no longer go Spot, so P4 is counted at Flex-start
    assert BI.launch_headroom("P2", "FLEX_START", cs) == 14000 - 4250 - 1000 - 2000
    assert not BI.possible("P4", "SPOT", cs) and not BI.may_launch("P4", "SPOT", cs)
    flex_spot = [c for c in cs if c.p2 == "FLEX_START" and c.p4 == "SPOT"][0]
    assert flex_spot.ruled_out and not flex_spot.allowed and flex_spot.headroom > 0
    assert BI.may_launch("P2", "FLEX_START", cs) and BI.may_launch("P4", "FLEX_START", cs)


def test_the_real_interlock_rules_out_p4_on_spot():
    assert set(BI.RULED_OUT) == {("P4", "SPOT")}
    _, _, _, cs = BI.current()
    assert not BI.possible("P4", "SPOT", cs) and BI.possible("P4", "FLEX_START", cs)


def test_the_cli_refuses_a_ruled_out_launch(monkeypatch, capsys):
    monkeypatch.setattr(BI, "RAN", {})       # the mechanics, before P2 ran
    cs = BI.cases(4000, 0, CAPS, BI.RULED_OUT)
    monkeypatch.setattr(BI, "current", lambda: (4000, (250, 233), CAPS, cs))
    assert BI.main(["--launch", "P4:SPOT"]) == 2
    assert "ruled out" in capsys.readouterr().out
    assert BI.main(["--launch", "P2:FLEX_START"]) == 0


def test_a_provisioning_a_card_has_no_plan_for_is_never_possible():
    # since the adviser's rulings dated 13 Nov 2026, second set, neither card plans Spot
    caps = {"P2": {"FLEX_START": 1000, "SPOT": None}, "P4": {"FLEX_START": 3000, "SPOT": None}}
    cs = BI.cases(4000, 0, caps)
    flex = next(c for c in cs if c.p2 == c.p4 == "FLEX_START")
    assert flex.planned and flex.possible and flex.headroom == 14000 - 8000
    assert [c.planned for c in cs if "SPOT" in (c.p2, c.p4)] == [False] * 3
    assert not BI.possible("P2", "SPOT", cs) and not BI.possible("P4", "SPOT", cs)
    assert BI.may_launch("P2", "FLEX_START", cs)
    assert BI.launch_headroom("P2", "FLEX_START", cs) == 6000


# --- a pairing that ran (P2, 14 Nov 2026) ------------------------------------------------------

def test_a_pairing_that_ran_reserves_nothing_and_is_not_launched_again(monkeypatch, capsys):
    assert BI.ran_caps(CAPS, {}) == CAPS
    got = BI.ran_caps(CAPS, {"P2": ("FLEX_START", "x")})
    assert got["P2"] == {"FLEX_START": 0, "SPOT": None} and got["P4"] == CAPS["P4"]
    assert set(BI.RAN) == {"P2"} and BI.RAN["P2"][0] == "FLEX_START"
    monkeypatch.setattr(BI, "pairing_caps", lambda: CAPS)
    s, _, caps, cs = BI.current()
    flex = next(c for c in cs if (c.p2, c.p4) == ("FLEX_START", "FLEX_START"))
    assert flex.committed == s + CAPS["P4"]["FLEX_START"]       # P2's cap not counted again
    assert not next(c for c in cs if (c.p2, c.p4) == ("SPOT", "FLEX_START")).possible
    assert BI.launch_headroom("P4", "FLEX_START", cs) == BI.BUDGET_INR - s - CAPS["P4"]["FLEX_START"]
    text = "\n".join(BI.section("P2"))
    assert "- **P2 ran** (" in text and "may launch" not in text
    assert "| Spot | Flex-start | — | — | no: P2 ran on Flex-start |" in text
    assert BI.main(["--launch", "P2:FLEX_START"]) == 2
    assert "REFUSED: P2 ran" in capsys.readouterr().out
