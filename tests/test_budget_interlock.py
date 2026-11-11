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
    monkeypatch.setattr(BI, "current", lambda: (7000, (250, 233), CAPS, BI.cases(7000, 250, CAPS)))
    assert BI.main(["--launch", "P2:FLEX_START"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert BI.main(["--launch", "P2:SPOT"]) == 0


def test_launch_4_is_counted_at_the_larger_reading(monkeypatch):
    monkeypatch.setattr(BI, "pairing_caps", lambda: CAPS)
    monkeypatch.setattr(BI, "rehearsal_cap", lambda text: (100, 233))   # 403 - 233 = 170 > 100
    s, _, _, cs = BI.current()
    assert cs[0].committed == s + 170 + 1000 + 2000


def test_both_cards_carry_the_current_interlock():
    for card, pairing in ((ROOT / "docs/launch_cards/t61_p2_a100.md", "P2"),
                          (ROOT / "docs/launch_cards/t64_p4_h100.md", "P4")):
        assert "\n".join(BI.section(pairing)) in card.read_text()
