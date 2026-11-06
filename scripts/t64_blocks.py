#!/usr/bin/env python3
"""The P4 timed session on Spot, planned as short blocks (researcher, 7 Nov 2026).

A Spot VM can be preempted, no cell-level resume crosses instances, and an
interrupted session's rows are discarded in full, so a preemption late in one
long session loses all of it. The session is planned as 2 to 3 blocks of about
two hours or less. The prediction is committed once at lock.

For each block this prints the numbers the card needs, from the step table of
`docs/launch_cards/t64_p4_h100.md` (the plan's minutes per step, its setup and
teardown hours):

    planned hours   setup + the block's steps + teardown
    cap (Rs)        Spot rate x planned hours x CAP_FACTOR   (1.3)
    halt (min)      HALT_FACTOR x planned hours               (1.5)

`predict`, `ground_truth` and `late_check` stay together in one block as the
runner is built (ground truth follows the sync of the prediction, and a second
prediction in a session is refused): splitting ground truth across blocks needs
block-aware cells that do not exist yet.

    python3 scripts/t64_blocks.py [--rate 668.04] [--max-hours 2.0]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "docs" / "launch_cards" / "t64_p4_h100.md"
SPOT_RATE_INR_HR = 668.04      # plan Section 5.2, the higher Spot region
CAP_FACTOR = 1.3
HALT_FACTOR = 1.5
MAX_BLOCK_HOURS = 2.0
ATOMIC = ("predict", "ground_truth", "late_check")


@dataclass(frozen=True)
class Block:
    steps: tuple
    run_min: float
    planned_h: float
    cap_inr: int
    halt_min: int
    cost_at_halt_inr: int

    @property
    def halt_exceeds_cap(self) -> bool:
        return self.cost_at_halt_inr > self.cap_inr


def card_numbers(text: str) -> dict:
    """{'steps': [(name, minutes)], 'setup_h', 'teardown_h'} from the card's tables."""
    steps = []
    for m in re.finditer(r"^\|\s*\d+\s*\|\s*`(\w+)`\s*\|.*\|\s*([\d +]+?)\s*\|\s*$", text, re.M):
        steps.append((m.group(1), float(sum(int(x) for x in m.group(2).split("+")))))
    setup = re.search(r"^\|\s*Setup:[^|]*\|\s*([\d.]+)\s*\|", text, re.M)
    tear = re.search(r"^\|\s*Teardown:[^|]*\|\s*([\d.]+)\s*\|", text, re.M)
    if not steps or not setup or not tear:
        raise ValueError("the card's step table, setup row or teardown row was not found")
    return {"steps": steps, "setup_h": float(setup.group(1)), "teardown_h": float(tear.group(1))}


def plan_blocks(steps, setup_h: float, teardown_h: float, *, rate: float = SPOT_RATE_INR_HR,
                max_hours: float = MAX_BLOCK_HOURS) -> list:
    """Greedy blocks in step order, each at most `max_hours` planned hours."""
    room = max_hours * 60 - (setup_h + teardown_h) * 60
    if room <= 0:
        raise ValueError("setup and teardown alone exceed the block length")
    groups, i = [], 0
    while i < len(steps):
        if steps[i][0] == ATOMIC[0]:
            j = i
            while j < len(steps) and steps[j][0] in ATOMIC:
                j += 1
            groups.append(steps[i:j])
            i = j
        else:
            groups.append([steps[i]])
            i += 1
    blocks, cur = [], []
    for g in groups:
        gmin = sum(m for _, m in g)
        if gmin > room:
            raise ValueError(f"{[n for n, _ in g]} take {gmin:.0f} min, more than a block's {room:.0f}")
        if sum(m for _, m in cur) + gmin > room:
            blocks.append(cur)
            cur = []
        cur += g
    blocks.append(cur)
    out = []
    for b in blocks:
        run = sum(m for _, m in b)
        planned = setup_h + run / 60 + teardown_h
        out.append(Block(tuple(n for n, _ in b), run, round(planned, 3),
                         math.ceil(rate * planned * CAP_FACTOR),
                         math.ceil(HALT_FACTOR * planned * 60),
                         math.ceil(rate * HALT_FACTOR * planned)))
    return out


def report(blocks, rate: float) -> str:
    lines = [f"Spot rate Rs {rate}/h; cap = rate x planned hours x {CAP_FACTOR}; "
             f"halt = {HALT_FACTOR} x planned hours", ""]
    for n, b in enumerate(blocks, 1):
        lines.append(f"block {n}: {b.run_min:.0f} min of steps, planned {b.planned_h:.2f} h, "
                     f"cap Rs {b.cap_inr}, halt {b.halt_min} min (Rs {b.cost_at_halt_inr} if it "
                     f"runs to the halt{', ABOVE the cap' if b.halt_exceeds_cap else ''}); "
                     f"steps: {', '.join(b.steps)}")
    tot = sum(b.planned_h for b in blocks)
    lines += ["", f"{len(blocks)} blocks, {tot:.2f} planned hours, expected Rs {math.ceil(tot * rate)}, "
              f"sum of caps Rs {sum(b.cap_inr for b in blocks)}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=SPOT_RATE_INR_HR)
    ap.add_argument("--max-hours", type=float, default=MAX_BLOCK_HOURS)
    args = ap.parse_args(argv)
    n = card_numbers(CARD.read_text())
    print(report(plan_blocks(n["steps"], n["setup_h"], n["teardown_h"], rate=args.rate,
                             max_hours=args.max_hours), args.rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
