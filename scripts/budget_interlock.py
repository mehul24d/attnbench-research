#!/usr/bin/env python3
"""The budget interlock (adviser's ruling 1g dated 10 Nov 2026): at most one
of P2 and P4 on Spot, and every launch card shows the headroom left after the
caps of all still-planned sessions; no launch if it is negative.

    headroom = BUDGET_INR - spent - (the cap of every still-planned session)

`spent` is the ledger's new-spend table (`docs/spend_ledger.md`, its rows
summed and checked against its own total). The still-planned sessions are the
L4 rehearsal's launch 4 (its card's cap), P2 and P4 (each card's cap, built
from its minutes file). A launch of one pairing on one provisioning counts the
other pairing at the dearest provisioning the one-Spot rule still allows it,
so P2 on Flex-start is counted with P4 on Spot, and P2 on Spot with P4 on
Flex-start only (P4 then cannot fall back to Spot).

    python3 scripts/budget_interlock.py                  # the table
    python3 scripts/budget_interlock.py --launch P2:FLEX_START   # exit 2 if refused
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

LEDGER = ROOT / "docs" / "spend_ledger.md"
REHEARSAL_CARD = ROOT / "docs" / "launch_cards" / "t61_rehearsal_l4.md"
BUDGET_INR = 14_000            # plan Section 5.2, new spend through 30 Nov 2026
REHEARSAL_CAP_RULED = 403      # the rehearsal's cap as ruling 1g names it (in all)
PROVISIONING = ("FLEX_START", "SPOT")
PAIRINGS = ("P2", "P4")
_HEAD = "### New-spend headroom against the ₹14,000 budget"
_ROW = re.compile(r"^\|\s*(?!\*\*Total)(.+?)\s*\|\s*~?([\d,]+)\s*\|\s*$")
_TOTAL = re.compile(r"^\|\s*\*\*Total\*\*\s*\|\s*\*\*~?([\d,]+)\*\*\s*\|\s*$")


def spent(text: str) -> int:
    """The ledger's new-spend rows, summed; refused unless they add to its total."""
    a = text.find(_HEAD)
    if a < 0:
        raise ValueError(f"the ledger has no section {_HEAD!r}")
    rows, total = [], None
    for line in text[a:].splitlines()[1:]:
        if line.startswith("#"):
            break
        if (m := _TOTAL.match(line)):
            total = int(m.group(1).replace(",", ""))
            break
        if (m := _ROW.match(line)) and not set(m.group(1)) <= set("-"):
            rows.append(int(m.group(2).replace(",", "")))
    if total is None or not rows:
        raise ValueError("the ledger's new-spend table has no rows or no total")
    if sum(rows) != total:
        raise ValueError(f"the ledger's new-spend rows sum to {sum(rows)}, its total says {total}")
    return total


def rehearsal_cap(text: str) -> tuple[int, int]:
    """(launch 4's cap, the rehearsal's spend so far) from the rehearsal card
    ("**₹254 for launch 4** ... **₹487 with attempts 1 and 2 and launch 3 (₹233)**")."""
    m = re.search(r"\*\*₹([\d,]+) for launch 4\*\*", text)
    n = re.search(r"\*\*₹[\d,]+ with attempts 1 and 2 and launch 3 \(₹([\d,]+)\)\*\*", text)
    if not m or not n:
        raise ValueError("the rehearsal card states no cap for launch 4, or no spend before it")
    return int(m.group(1).replace(",", "")), int(n.group(1).replace(",", ""))


def pairing_caps() -> dict:
    """{'P2': {'FLEX_START': cap, 'SPOT': caps summed}, 'P4': ...} from the cards' builders."""
    import t61_card
    import t64_card
    return {"P2": t61_card.caps(), "P4": t64_card.caps()}


@dataclass(frozen=True)
class Case:
    p2: str
    p4: str
    committed: int          # spent + every still-planned cap
    headroom: int

    @property
    def allowed(self) -> bool:
        """The one-Spot rule and a headroom of at least zero."""
        return not (self.p2 == self.p4 == "SPOT") and self.headroom >= 0


def cases(spent_inr: int, rehearsal_inr: int, caps: dict) -> list:
    out = []
    for p2, p4 in product(PROVISIONING, PROVISIONING):
        c = spent_inr + rehearsal_inr + caps["P2"][p2] + caps["P4"][p4]
        out.append(Case(p2, p4, c, BUDGET_INR - c))
    return out


def launch_headroom(pairing: str, provisioning: str, all_cases) -> int:
    """The headroom a launch of `pairing` on `provisioning` leaves, the other
    pairing at the dearest provisioning the one-Spot rule allows with it."""
    i = PAIRINGS.index(pairing)
    mine = [c for c in all_cases if (c.p2, c.p4)[i] == provisioning
            and not (c.p2 == c.p4 == "SPOT")]
    if not mine:
        raise ValueError(f"{pairing} on {provisioning} is allowed with nothing")
    return min(c.headroom for c in mine)


def may_launch(pairing: str, provisioning: str, all_cases) -> bool:
    return launch_headroom(pairing, provisioning, all_cases) >= 0


def current() -> tuple[int, tuple, dict, list]:
    """The ledger's spend, (launch 4's card cap, the rehearsal's spend), the
    pairings' caps and the cases. Launch 4 is counted at the larger of its
    card's cap and what ruling 1g's ₹403 leaves, so neither reading is undercounted."""
    s = spent(LEDGER.read_text(encoding="utf-8"))
    card, done = rehearsal_cap(REHEARSAL_CARD.read_text(encoding="utf-8"))
    caps = pairing_caps()
    r = max(card, REHEARSAL_CAP_RULED - done)
    return s, (card, done), caps, cases(s, r, caps)


def _p(prov: str) -> str:
    return {"FLEX_START": "Flex-start", "SPOT": "Spot"}[prov]


def section(pairing: str) -> list:
    """The card's interlock section, as Markdown lines."""
    s, (card, done), caps, cs = current()
    left = REHEARSAL_CAP_RULED - done
    out = ["## Budget interlock", "",
           "Adviser's ruling 1g dated 10 Nov 2026 (`scripts/budget_interlock.py`): **at most one "
           "of P2 and P4 on Spot**, and no launch if the headroom left after the caps of all "
           f"still-planned sessions is negative. Budget ₹{BUDGET_INR:,} (plan Section 5.2); "
           f"spent ₹{s:,} (the ledger's new-spend table, the rehearsal's ₹{done} in it); the L4 "
           f"rehearsal's launch 4 at ₹{max(card, left)}, the larger of its card's cap (₹{card}, "
           f"₹{card + done} in all) and what the ruling's rehearsal cap of ₹{REHEARSAL_CAP_RULED} "
           f"leaves (₹{left}); launch 4's own cap decision is still owed. P2 and P4 at their "
           "cards' caps: P2 Flex-start "
           f"₹{caps['P2']['FLEX_START']:,}, Spot ₹{caps['P2']['SPOT']:,}; P4 Flex-start "
           f"₹{caps['P4']['FLEX_START']:,}, Spot ₹{caps['P4']['SPOT']:,}. Not counted: reruns, "
           "a Spot block rerun after an interruption, image storage, egress.", "",
           "| P2 | P4 | Spent + caps ₹ | Headroom ₹ | Allowed |",
           "|---|---|---|---|---|"]
    for c in cs:
        why = ("no: both on Spot" if c.p2 == c.p4 == "SPOT" else
               "no: negative" if c.headroom < 0 else "yes")
        out.append(f"| {_p(c.p2)} | {_p(c.p4)} | {c.committed:,} | {c.headroom:,} | {why} |")
    out.append("")
    for prov in PROVISIONING:
        h = launch_headroom(pairing, prov, cs)
        out.append(f"- **{pairing} on {_p(prov)}: "
                   f"{'may launch' if h >= 0 else 'REFUSED'}**, headroom ₹{h:,} with the "
                   "other pairing at the dearest provisioning the rule allows.")
    return out + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--launch", help="PAIRING:PROVISIONING, e.g. P2:FLEX_START; exit 2 if refused")
    args = ap.parse_args(argv)
    if args.launch:
        pairing, prov = args.launch.split(":")
        _, _, _, cs = current()
        h = launch_headroom(pairing, prov, cs)
        print(f"{pairing} on {prov}: headroom Rs {h}; "
              f"{'may launch' if h >= 0 else 'REFUSED: negative headroom'}")
        return 0 if h >= 0 else 2
    print("\n".join(section("P2")[2:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
