#!/usr/bin/env python3
"""A pairing's timed session on Spot, planned as short blocks (researcher,
7 Nov 2026), and the block spec it gives the lock file.

A Spot VM can be preempted, no cell-level resume crosses instances, and an
interrupted session's rows are discarded, so the session runs in blocks of
about two hours or less, each on its own boot (`attnbench.lock.blocks`). For
P4 the ruling is 3 blocks of at most 2.2 hours; P2's card is planned the same
way. From the card's step table (its minutes per step, its setup and
teardown hours) each block is:

    planned hours   setup + the boot steps + its steps + teardown
    cap (Rs)        Spot rate x planned hours x CAP_FACTOR                (1.3)
    halt (min)      HALT_FACTOR x planned hours, rounded down             (1.3)
    hard delete     the halt + DELETE_AFTER_HALT_MIN                      (10)

so the cost at the halt is never above the cap (rounded down, the halt's
minutes cost at most rate x planned hours x 1.3); `plan_blocks` refuses a
block where it would be. The hard delete is the backstop if the halt fails.

Every block runs the boot steps (`prereqs`, P4's `fa3_import`, `preflight`,
the end host-speed probe), and the steps between them in order. Ground
truth is split across blocks by its minutes: each block with ground truth
also measures the anchor cell (`ANCHOR_MIN`, an estimate: no anchor has been
timed) and runs the late check and the trace checks after it; `predict` is in
the block where ground truth starts. The block spec (`--spec`) lists each
block's steps and minutes; its cells and the anchor are fixed at lock with
the grid, so the skeleton leaves them null.

    python3 scripts/t64_blocks.py [--pairing P4] [--max-hours 2.2] [--blocks 3] [--spec]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = {"P4": ROOT / "docs" / "launch_cards" / "t64_p4_h100.md",
         "P2": ROOT / "docs" / "launch_cards" / "t61_p2_a100.md"}
CARD = CARDS["P4"]
SPOT_RATES = {"P4": 668.04,    # plan Section 5.2, the higher H100 Spot region
              "P2": 350.14}    # plan Section 5.2, A100 Spot in asia-southeast1
SPOT_RATE_INR_HR = SPOT_RATES["P4"]
BOOT = {"P4": ("prereqs", "fa3_import", "preflight", "host_speed_end"),
        "P2": ("prereqs", "preflight", "host_speed_end")}
CAP_FACTOR = 1.3
HALT_FACTOR = 1.3
DELETE_AFTER_HALT_MIN = 10
MAX_BLOCK_HOURS = 2.2
BLOCK_HOURS = {"P2": MAX_BLOCK_HOURS,
               "P4": 2.3}   # P4's Spot fallback (adviser's ruling 5, 10 Nov 2026): 2.2 h refuses
ANCHOR_MIN = 2.0               # the anchor cell's ground truth per block: an estimate
SPLIT_FOLLOWERS = ("late_check", "trace_checks")


@dataclass(frozen=True)
class Block:
    steps: tuple                # its own steps (no boot step), in order
    ground_truth_min: float     # its share of the ground truth, the anchor excluded
    run_min: float              # every minute of steps on the instance: boot, own, anchor
    planned_h: float
    cap_inr: int
    halt_min: int
    delete_min: int
    cost_at_halt_inr: int
    cost_at_delete_inr: int

    @property
    def halt_exceeds_cap(self) -> bool:
        return self.cost_at_halt_inr > self.cap_inr


def card_numbers(text: str) -> dict:
    """{'steps': [(name, minutes)], 'setup_h', 'teardown_h', 'suite_min'} from the
    card's tables; 'suite_min' is 0 where the card has no suite-record row."""
    steps = []
    for m in re.finditer(r"^\|\s*\d+\s*\|\s*`(\w+)`\s*\|.*\|\s*([\d +]+?)\s*\|\s*$", text, re.M):
        steps.append((m.group(1), float(sum(int(x) for x in m.group(2).split("+")))))
    setup = re.search(r"^\|\s*Setup:[^|]*\|\s*([\d.]+)\s*\|", text, re.M)
    tear = re.search(r"^\|\s*Teardown:[^|]*\|\s*([\d.]+)\s*\|", text, re.M)
    suite = re.search(r"^\|\s*Suite record:[^|]*\|\s*([\d.]+) min\s*\|", text, re.M)
    if not steps or not setup or not tear:
        raise ValueError("the card's step table, setup row or teardown row was not found")
    return {"steps": steps, "setup_h": float(setup.group(1)), "teardown_h": float(tear.group(1)),
            "suite_min": float(suite.group(1)) if suite else 0.0}


def _block(steps, gt_min, run_min, setup_h, teardown_h, rate, suite_min: float = 0.0) -> Block:
    planned = setup_h + (suite_min + run_min) / 60 + teardown_h
    halt = math.floor(HALT_FACTOR * planned * 60)
    b = Block(tuple(steps), gt_min, run_min, round(planned, 3),
              math.ceil(rate * planned * CAP_FACTOR), halt, halt + DELETE_AFTER_HALT_MIN,
              math.ceil(rate * halt / 60), math.ceil(rate * (halt + DELETE_AFTER_HALT_MIN) / 60))
    if b.halt_exceeds_cap:
        raise ValueError(f"a block's halt costs Rs {b.cost_at_halt_inr}, above its cap "
                         f"Rs {b.cap_inr}")
    return b


def plan_blocks(steps, setup_h: float, teardown_h: float, *, boot=BOOT["P4"],
                rate: float = SPOT_RATE_INR_HR, max_hours: float = MAX_BLOCK_HOURS,
                anchor_min: float = ANCHOR_MIN, suite_min: float = 0.0) -> list:
    """Greedy blocks in step order, each at most `max_hours` planned hours,
    ground truth split across them by its minutes. The suite record
    (`suite_min`, the bootstrap's `--suite-record`) runs on the first boot
    only: the image, the commit and the GPU model are the same on every
    block, so it is one record per pairing (a plan default, 11 Nov 2026)."""
    mins = dict(steps)
    boot_min = sum(mins.get(s, 0.0) for s in boot)
    room0 = max_hours * 60 - (setup_h + teardown_h) * 60 - boot_min
    followers = [s for s in SPLIT_FOLLOWERS if s in mins]
    gt_extra = anchor_min + sum(mins[s] for s in followers)
    if room0 - suite_min <= gt_extra:
        raise ValueError("setup, teardown and the boot steps alone fill the block")
    order = [n for n, _ in steps if n not in boot and n not in followers]
    blocks, cur, used = [], [], 0.0
    gt_used = 0.0

    def room():
        return room0 - (suite_min if not blocks else 0.0)

    def close():
        nonlocal cur, used, gt_used
        if cur:
            blocks.append(_block(cur, gt_used, boot_min + used, setup_h, teardown_h, rate,
                                 suite_min if not blocks else 0.0))
        cur, used, gt_used = [], 0.0, 0.0

    for name in order:
        m = mins[name]
        if name != "ground_truth":
            if m > room0:
                raise ValueError(f"{name} takes {m:.0f} min, more than a block's {room0:.0f}")
            if used + m > room():
                close()
            cur.append(name)
            used += m
            continue
        left = m
        while left > 1e-9:
            free = room() - used - gt_extra
            if free <= 1e-9:
                if "predict" in cur and cur[-1] == "predict":
                    raise ValueError("predict's block has no room for its ground truth")
                close()
                continue
            take = min(left, free)
            cur += ["ground_truth", *followers]
            used += take + gt_extra
            gt_used, left = take, left - take
            if left > 1e-9:
                close()
    close()
    over = [n for n, b in enumerate(blocks, 1) if b.planned_h > max_hours + 1e-6]
    if over:
        raise ValueError(f"block(s) {over} exceed {max_hours} h (the suite record's "
                         f"{suite_min:.0f} min with the first block's steps)")
    return blocks


def spec(blocks, *, max_hours: float) -> dict:
    """The lock file's block spec for these blocks (`attnbench.lock.blocks`);
    cells and the anchor null until the grid is locked."""
    out = []
    for n, b in enumerate(blocks, 1):
        e = {"id": f"b{n}", "steps": list(b.steps), "planned_hours": b.planned_h}
        if "ground_truth" in b.steps:
            e.update(ground_truth_minutes=round(b.ground_truth_min, 1), cells=None)
        out.append(e)
    return {"max_hours": max_hours, "anchor": None, "blocks": out}


def report(blocks, rate: float) -> str:
    lines = [f"Spot rate Rs {rate}/h; cap = rate x planned hours x {CAP_FACTOR}; halt = "
             f"{HALT_FACTOR} x planned hours (rounded down); hard delete = the halt + "
             f"{DELETE_AFTER_HALT_MIN} min", ""]
    for n, b in enumerate(blocks, 1):
        gt = f", ground truth {b.ground_truth_min:.0f} min + the anchor" if b.ground_truth_min else ""
        lines.append(f"block b{n}: {b.run_min:.0f} min of steps{gt}, planned {b.planned_h:.2f} h, "
                     f"cap Rs {b.cap_inr}, halt {b.halt_min} min (Rs {b.cost_at_halt_inr}), "
                     f"hard delete {b.delete_min} min (Rs {b.cost_at_delete_inr}); "
                     f"steps: {', '.join(b.steps)}")
    tot = sum(b.planned_h for b in blocks)
    lines += ["", f"{len(blocks)} blocks, {tot:.2f} planned hours, expected Rs {math.ceil(tot * rate)}, "
              f"sum of caps Rs {sum(b.cap_inr for b in blocks)}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairing", default="P4", choices=sorted(CARDS))
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="default: the pairing's block length, BLOCK_HOURS")
    ap.add_argument("--blocks", type=int, default=None, help="refuse unless the plan has this many")
    ap.add_argument("--spec", action="store_true", help="print the lock file's block spec")
    args = ap.parse_args(argv)
    rate = args.rate or SPOT_RATES[args.pairing]
    args.max_hours = args.max_hours or BLOCK_HOURS[args.pairing]
    n = card_numbers(CARDS[args.pairing].read_text())
    try:
        blocks = plan_blocks(n["steps"], n["setup_h"], n["teardown_h"], boot=BOOT[args.pairing],
                             rate=rate, max_hours=args.max_hours, suite_min=n["suite_min"])
    except ValueError as e:
        print(f"REFUSING: no blocks at {args.max_hours} h: {e}", file=sys.stderr)
        return 2
    if args.blocks is not None and len(blocks) != args.blocks:
        print(f"REFUSING: {len(blocks)} blocks at {args.max_hours} h, not {args.blocks}",
              file=sys.stderr)
        return 2
    print(json.dumps({args.pairing: spec(blocks, max_hours=args.max_hours)}, indent=1)
          if args.spec else report(blocks, rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
