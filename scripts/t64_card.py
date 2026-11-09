#!/usr/bin/env python3
"""The P4 launch card's step table and its cost comparison, built from what
they describe (adviser's ruling 7 of the eleventh 9 Nov 2026 paste): the
session's steps in `t64_session.T64`'s order with each step's minutes and
basis (`configs/t64/p4_card_minutes.json`), then Flex-start, Spot blocks
and on-demand side by side, each with its expected cost, its cap and its
interruption risk, the overhead every Spot block pays on its own boot
included. The measurement protocol is the same under each; only the
provisioning differs.

Only the region between the card's two markers is built; the rest of
`docs/launch_cards/t64_p4_h100.md` is prose and stays hand-written.

    python3 scripts/t64_card.py [--date 2026-11-09]
    python3 scripts/t64_card.py --check     # exit 1 if the card's built region is stale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import t64_blocks as B  # noqa: E402
from t64_session import T64  # noqa: E402

MINUTES = ROOT / "configs" / "t64" / "p4_card_minutes.json"
CARD = B.CARDS["P4"]
FLEX_RATE = 462.42           # plan Section 5.2, H100 Flex-start, us-central1, all-in
SPOT_RATE = B.SPOT_RATES["P4"]
SPOT_RATE_LOW = 629.94       # plan Section 5.2, H100 Spot, us-central1, all-in
BEGIN = "<!-- built by scripts/t64_card.py from configs/t64/p4_card_minutes.json: rebuild, do not edit -->"
END = "<!-- end of the built region -->"


def load_minutes(path=MINUTES) -> dict:
    """The minutes file, refused unless it names exactly T64's steps."""
    m = json.loads(Path(path).read_text())
    names = [s.name for s in T64.steps]
    if set(m["steps"]) != set(names):
        raise ValueError(f"{path} names {sorted(m['steps'])}; T64's steps are {names}")
    return m


def spot_blocks(steps, setup, tear):
    """The blocks at the ruled length, or, if a step does not fit one, the
    plan at the shortest length that fits (0.1 h steps), with the refusal."""
    try:
        return B.MAX_BLOCK_HOURS, B.plan_blocks(steps, setup, tear, boot=B.BOOT["P4"],
                                                rate=SPOT_RATE), None
    except ValueError as e:
        refusal = str(e)
    for tenth in range(round(B.MAX_BLOCK_HOURS * 10) + 1, 61):
        try:
            h = tenth / 10
            return h, B.plan_blocks(steps, setup, tear, boot=B.BOOT["P4"], rate=SPOT_RATE,
                                    max_hours=h), refusal
        except ValueError:
            continue
    raise ValueError(f"no block length up to 6 h fits: {refusal}")


def build(minutes: dict, on: str) -> str:
    steps = [(s.name, float(sum(minutes["steps"][s.name]["minutes"]))) for s in T64.steps]
    setup, tear = minutes["setup_h"]["value"], minutes["teardown_h"]["value"]
    run_min = sum(m for _, m in steps)
    flex = B._block([n for n, _ in steps], 0.0, run_min, setup, tear, FLEX_RATE)
    hours, blocks, refusal = spot_blocks(steps, setup, tear)
    mins = dict(steps)
    boot_min = sum(mins[s] for s in B.BOOT["P4"])
    per_boot_h = setup + boot_min / 60 + tear
    spot_h = sum(b.planned_h for b in blocks)
    spot_exp = math.ceil(spot_h * SPOT_RATE)
    worst = max(range(len(blocks)), key=lambda i: blocks[i].cap_inr)
    measured = [n for n, _ in steps if "measured" in minutes["steps"][n]["basis"]]
    out = [BEGIN, "",
           f"Built {on}. {len(measured)} of {len(steps)} steps' minutes are measured on P1's "
           "L4 (T3.2d); the rest are estimates. None has run on an H100.", "",
           "| # | Step | Task | What it does | Basis | Expected minutes |",
           "|---|---|---|---|---|---|"]
    for i, s in enumerate(T64.steps, 1):
        e = minutes["steps"][s.name]
        out.append(f"| {i} | `{s.name}` | {s.task} | {s.what} | {e['basis']} | "
                   f"{' + '.join(str(int(x)) for x in e['minutes'])} |")
    out += [
        "",
        f"Steps total {run_min:.0f} minutes.",
        "",
        "## Time and cost: Flex-start, Spot blocks, on-demand",
        "",
        f"Cap = rate x planned hours x {B.CAP_FACTOR}; halt = {B.HALT_FACTOR} x planned hours "
        f"(rounded down); hard delete = the halt + {B.DELETE_AFTER_HALT_MIN} min. Rates are "
        "the plan's all-in planning rates (Section 5.2). Every boot pays the setup, the boot "
        f"steps ({', '.join(f'`{x}`' for x in B.BOOT['P4'])}: {boot_min:.0f} min) and the "
        f"teardown, {per_boot_h:.2f} h; a Spot block with ground truth also measures the "
        f"anchor cell ({B.ANCHOR_MIN:g} min, an estimate) and runs `late_check` and "
        "`trace_checks` again.",
        "",
        "| | Hours | ₹ |",
        "|---|---|---|",
        f"| Setup: write gate, deploy, preflight, bootstrap with XAttention | {setup} | "
        f"{math.ceil(setup * FLEX_RATE)} |",
        f"| The steps ({run_min:.0f} minutes) | {run_min / 60:.2f} | "
        f"{math.ceil(run_min / 60 * FLEX_RATE)} |",
        f"| Teardown: logs and manifests pull, delete, cleanup check | {tear} | "
        f"{math.ceil(tear * FLEX_RATE)} |",
        "",
        "| | Flex-start, one session | Spot, in blocks | On-demand |",
        "|---|---|---|---|",
        f"| Rate | ₹{FLEX_RATE}/h (`us-central1`) | ₹{SPOT_RATE}/h (`asia-southeast1`, the "
        f"higher; `us-central1` ₹{SPOT_RATE_LOW}/h) | not offered for `a3-highgpu-1g` |",
        f"| Boots | 1 | {len(blocks)} | |",
        f"| Overhead paid per boot | {per_boot_h:.2f} h | {per_boot_h:.2f} h x {len(blocks)}, "
        "plus the anchor and the trace steps in each ground-truth block | |",
        f"| Planned hours | {flex.planned_h:.2f} | {spot_h:.2f} | |",
        f"| Expected, nothing fails | ₹{math.ceil(flex.planned_h * FLEX_RATE)} | ₹{spot_exp} | |",
        f"| Cap | ₹{flex.cap_inr} (halt {flex.halt_min} min, hard delete {flex.delete_min} min) | "
        f"₹{sum(b.cap_inr for b in blocks)}, the caps summed | |",
        "| Interruption risk | None once granted: a Flex-start VM runs up to seven days. The "
        "risk is the grant: a waiting request may not be granted, and costs nothing while it "
        "waits (`docs/launch_cards/h100_flex_attempts.md`) | Any block can be preempted at "
        "any time. Its rows are discarded and it is rerun on a new instance, so one "
        f"interruption adds up to that block's cap (at most ₹{blocks[worst].cap_inr}, b{worst + 1}) "
        "| Compute Engine offers A3 High with fewer than 8 GPUs only as Spot or Flex-start; "
        "`a3-highgpu-8g` would need 8 GPUs against a quota of 1 |",
        "",
    ]
    if refusal:
        out += [f"**At the ruled {B.MAX_BLOCK_HOURS} h the blocks are refused:** {refusal} "
                f"(setup, teardown and the boot steps take {per_boot_h * 60:.0f} of a block's "
                f"{B.MAX_BLOCK_HOURS * 60:.0f} minutes). The Spot plan below is at {hours} h, the "
                "shortest length that fits; it is not the ruled length, and using it is a "
                "decision.", ""]
    out += [f"**Spot blocks at {hours} h:**", "",
            "| Block | Steps | Planned h | Expected ₹ | Cap ₹ | Halt min | Hard delete min |",
            "|---|---|---|---|---|---|---|"]
    for n, b in enumerate(blocks, 1):
        out.append(f"| b{n} | {', '.join(f'`{x}`' for x in b.steps)} | {b.planned_h:.2f} | "
                   f"{math.ceil(b.planned_h * SPOT_RATE)} | {b.cap_inr} | {b.halt_min} | "
                   f"{b.delete_min} |")
    out += ["", END]
    return "\n".join(out)


def region(text: str) -> tuple[int, int]:
    a, b = text.find(BEGIN), text.find(END)
    if a < 0 or b < a:
        raise ValueError(f"{CARD.name} has no built region")
    return a, b + len(END)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless the card's built region is this build (its date kept)")
    args = ap.parse_args(argv)
    text = CARD.read_text(encoding="utf-8")
    a, b = region(text)
    if args.check:
        on = text[a:b].split("Built ", 1)[1].split(".", 1)[0]
        same = text[a:b] == build(load_minutes(), on)
        print("the card is current" if same else "the card is stale: rebuild it")
        return 0 if same else 1
    CARD.write_text(text[:a] + build(load_minutes(), args.date) + text[b:], encoding="utf-8")
    print(f"wrote {CARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
