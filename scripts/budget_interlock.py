#!/usr/bin/env python3
"""The budget interlock (adviser's ruling 1g dated 10 Nov 2026): at most one
of P2 and P4 on Spot, and every launch card shows the headroom left after the
caps of all still-planned sessions; no launch if it is negative.

    headroom = BUDGET_INR - spent - (the cap of every still-planned session)

`spent` is the ledger's new-spend table (`docs/spend_ledger.md`, its rows
summed and checked against its own total). The still-planned sessions are the
L4 rehearsal's launch 4 (its card's cap) until the rehearsal closes, P2 and P4 (each card's cap, built
from its minutes file). A launch of one pairing on one provisioning counts the
other pairing at the dearest provisioning the one-Spot rule still allows it,
so P2 on Flex-start is counted with P4 on Spot, and P2 on Spot with P4 on
Flex-start only (P4 then cannot fall back to Spot). A provisioning ruled out
(`RULED_OUT`) is neither launched nor counted: P4 on Spot since the adviser's
rulings dated 13 Nov 2026 (item 6), so P2 on Flex-start is counted with P4
on Flex-start.

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
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

LEDGER = ROOT / "docs" / "spend_ledger.md"
REHEARSAL_CARD = ROOT / "docs" / "launch_cards" / "t61_rehearsal_l4.md"
BUDGET_INR = 14_000            # plan Section 5.2, new spend through 30 Nov 2026
REHEARSAL_CAP_RULED = 403      # the rehearsal's cap as ruling 1g names it (in all)
PROVISIONING = ("FLEX_START", "SPOT")
PAIRINGS = ("P2", "P4")
# (pairing, provisioning) -> the ruling that took it off the table
RULED_OUT = {("P4", "SPOT"): "the adviser's rulings dated 13 Nov 2026, item 6: Spot no longer "
                             "fits the budget"}
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


def rehearsal_closed(text: str):
    """The rehearsal's spend in all if its card says it closed, else None. A
    closed rehearsal reserves nothing: its spend is in the ledger (adviser's
    ruling 6, fourth set of 12 Nov 2026, after it closed at launch 4 A)."""
    if "The rehearsal closed on" not in text:
        return None
    m = re.search(r"launch 4 A used ₹([\d,]+)", text)
    if not m:
        raise ValueError("the rehearsal card says it closed but not what launch 4 A used")
    return rehearsal_cap(text)[1] + int(m.group(1).replace(",", ""))


def pairing_caps() -> dict:
    """{'P2': {'FLEX_START': cap, 'SPOT': caps summed}, 'P4': ...} from the cards' builders."""
    import t61_card
    import t64_card
    return {"P2": t61_card.caps(), "P4": t64_card.caps()}


@dataclass(frozen=True)
class Case:
    p2: str
    p4: str
    committed: Optional[int]   # spent + every still-planned cap; None with no plan
    headroom: Optional[int]
    ruled_out: bool = False  # a provisioning in it is ruled out (`RULED_OUT`)

    @property
    def planned(self) -> bool:
        """Whether both cards plan their provisioning (a card's cap is None
        where it has no plan)."""
        return self.committed is not None

    @property
    def possible(self) -> bool:
        """The one-Spot rule, both planned, and no provisioning ruled out."""
        return not (self.p2 == self.p4 == "SPOT") and self.planned and not self.ruled_out

    @property
    def allowed(self) -> bool:
        """Possible, and a headroom of at least zero."""
        return self.possible and self.headroom >= 0


def cases(spent_inr: int, rehearsal_inr: int, caps: dict, ruled_out=()) -> list:
    out = []
    for p2, p4 in product(PROVISIONING, PROVISIONING):
        a, b = caps["P2"][p2], caps["P4"][p4]
        c = None if a is None or b is None else spent_inr + rehearsal_inr + a + b
        out.append(Case(p2, p4, c, None if c is None else BUDGET_INR - c,
                        ruled_out=("P2", p2) in ruled_out or ("P4", p4) in ruled_out))
    return out


def launch_headroom(pairing: str, provisioning: str, all_cases) -> int:
    """The headroom a launch of `pairing` on `provisioning` leaves, the other
    pairing at the dearest provisioning the one-Spot rule allows with it."""
    i = PAIRINGS.index(pairing)
    mine = [c for c in all_cases if (c.p2, c.p4)[i] == provisioning and c.possible]
    if not mine:
        raise ValueError(f"{pairing} on {provisioning} is allowed with nothing")
    return min(c.headroom for c in mine)


def possible(pairing: str, provisioning: str, all_cases) -> bool:
    """Whether any case the rules allow has `pairing` on `provisioning`."""
    i = PAIRINGS.index(pairing)
    return any((c.p2, c.p4)[i] == provisioning and c.possible for c in all_cases)


def may_launch(pairing: str, provisioning: str, all_cases) -> bool:
    return possible(pairing, provisioning, all_cases) and \
        launch_headroom(pairing, provisioning, all_cases) >= 0


def current() -> tuple[int, tuple, dict, list]:
    """The ledger's spend, (launch 4's card cap, the rehearsal's spend), the
    pairings' caps and the cases. Launch 4 is counted at the larger of its
    card's cap and what ruling 1g's ₹403 leaves, so neither reading is undercounted."""
    s = spent(LEDGER.read_text(encoding="utf-8"))
    text = REHEARSAL_CARD.read_text(encoding="utf-8")
    card, done = rehearsal_cap(text)
    caps = pairing_caps()
    r = 0 if rehearsal_closed(text) is not None else max(card, REHEARSAL_CAP_RULED - done)
    return s, (card, done), caps, cases(s, r, caps, RULED_OUT)


def _cap(inr) -> str:
    return "no plan" if inr is None else f"₹{inr:,}"


def _p(prov: str) -> str:
    return {"FLEX_START": "Flex-start", "SPOT": "Spot"}[prov]


def section(pairing: str) -> list:
    """The card's interlock section, as Markdown lines."""
    s, (card, done), caps, cs = current()
    left = REHEARSAL_CAP_RULED - done
    closed = rehearsal_closed(REHEARSAL_CARD.read_text(encoding="utf-8"))
    rehearsal = (f"spent ₹{s:,} (the ledger's new-spend table, the L4 rehearsal's ₹{closed} in "
                 f"it); the rehearsal closed, so nothing more is reserved for it"
                 if closed is not None else
                 f"spent ₹{s:,} (the ledger's new-spend table, the rehearsal's ₹{done} in it); "
                 f"the L4 rehearsal's launch 4 at ₹{max(card, left)}, the larger of its card's "
                 f"cap (₹{card}, ₹{card + done} in all) and what the ruling's rehearsal cap of "
                 f"₹{REHEARSAL_CAP_RULED} leaves (₹{left}); launch 4's own cap decision is "
                 f"still owed")
    out = ["## Budget interlock", "",
           "Adviser's ruling 1g dated 10 Nov 2026 (`scripts/budget_interlock.py`): **at most one "
           "of P2 and P4 on Spot**, and no launch if the headroom left after the caps of all "
           f"still-planned sessions is negative. Budget ₹{BUDGET_INR:,} (plan Section 5.2); "
           f"{rehearsal}. P2 and P4 at their cards' caps, the ground truth's worst case of 5 "
           "reruns at every cell in each: P2 Flex-start "
           f"₹{caps['P2']['FLEX_START']:,}, Spot {_cap(caps['P2']['SPOT'])}; P4 Flex-start "
           f"₹{caps['P4']['FLEX_START']:,}, Spot {_cap(caps['P4']['SPOT'])}. Not counted: a Spot "
           "block rerun after an interruption, image storage, egress.", "",
           "| P2 | P4 | Spent + caps ₹ | Headroom ₹ | Allowed |",
           "|---|---|---|---|---|"]
    for c in cs:
        why = ("no: both on Spot" if c.p2 == c.p4 == "SPOT" else
               "no: no Spot plan" if not c.planned else
               "no: ruled out" if c.ruled_out else
               "no: negative" if c.headroom < 0 else "yes")
        num = (lambda x: "—" if x is None else f"{x:,}")  # noqa: E731
        out.append(f"| {_p(c.p2)} | {_p(c.p4)} | {num(c.committed)} | {num(c.headroom)} | "
                   f"{why} |")
    out.append("")
    for (pp, prov), why in RULED_OUT.items():
        out.append(f"- **{pp} on {_p(prov)}: ruled out** ({why}).")
    for prov in PROVISIONING:
        if not possible(pairing, prov, cs):
            continue
        h = launch_headroom(pairing, prov, cs)
        out.append(f"- **{pairing} on {_p(prov)}: "
                   f"{'may launch' if h >= 0 else 'REFUSED'}**, headroom ₹{h:,} with the "
                   "other pairing at the dearest provisioning the rules allow.")
    return out + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--launch", help="PAIRING:PROVISIONING, e.g. P2:FLEX_START; exit 2 if refused")
    args = ap.parse_args(argv)
    if args.launch:
        pairing, prov = args.launch.split(":")
        _, _, _, cs = current()
        if not possible(pairing, prov, cs):
            print(f"{pairing} on {prov}: REFUSED: ruled out ({RULED_OUT.get((pairing, prov))})")
            return 2
        h = launch_headroom(pairing, prov, cs)
        print(f"{pairing} on {prov}: headroom Rs {h}; "
              f"{'may launch' if h >= 0 else 'REFUSED: negative headroom'}")
        return 0 if h >= 0 else 2
    print("\n".join(section("P2")[2:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
