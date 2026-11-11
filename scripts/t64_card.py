#!/usr/bin/env python3
"""The P4 launch card's step table and its cost comparison, built from what
they describe (adviser's ruling 7 of the eleventh 9 Nov 2026 paste): the
session's steps in `t64_session.T64`'s order with each step's minutes and
basis (`configs/t64/p4_card_minutes.json`), then Flex-start, Spot blocks
and on-demand side by side, each with its expected cost, its cap and its
interruption risk, the overhead every Spot block pays on its own boot
included. The measurement protocol is the same under each; only the
provisioning differs. The order is ruled (adviser's ruling 5 of the 10 Nov
2026 paste): Flex-start first, requested right after the lock; Spot in four
blocks of 2.3 h if no grant comes by about 22 Nov. A plan that is not those
four blocks is refused, not printed.

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
from t61_card import prelaunch_pass  # noqa: E402

MINUTES = ROOT / "configs" / "t64" / "p4_card_minutes.json"
CARD = B.CARDS["P4"]
FLEX_RATE = 462.42           # plan Section 5.2, H100 Flex-start, us-central1, all-in
SPOT_RATE = B.SPOT_RATES["P4"]
SPOT_RATE_LOW = 629.94       # plan Section 5.2, H100 Spot, us-central1, all-in
SPOT_HOURS, SPOT_BLOCKS = 2.3, 4   # the fallback as ruled, 10 Nov 2026
FLEX_REQUEST = "right after the lock (14 Nov 2026)"
SPOT_FROM = "about 22 Nov 2026"
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
    """The fallback's blocks at the ruled 2.3 h, refused unless they are the
    ruled four; with why the general 2.2 h length refuses, if it does."""
    blocks = B.plan_blocks(steps, setup, tear, boot=B.BOOT["P4"], rate=SPOT_RATE,
                           max_hours=SPOT_HOURS)
    if len(blocks) != SPOT_BLOCKS:
        raise ValueError(f"{len(blocks)} blocks at {SPOT_HOURS} h, not the ruled "
                         f"{SPOT_BLOCKS}: the fallback needs a new ruling")
    try:
        B.plan_blocks(steps, setup, tear, boot=B.BOOT["P4"], rate=SPOT_RATE)
        refusal = None
    except ValueError as e:
        refusal = str(e)
    return blocks, refusal


def build(minutes: dict, on: str) -> str:
    steps = [(s.name, float(sum(minutes["steps"][s.name]["minutes"]))) for s in T64.steps]
    setup, tear = minutes["setup_h"]["value"], minutes["teardown_h"]["value"]
    run_min = sum(m for _, m in steps)
    flex = B._block([n for n, _ in steps], 0.0, run_min, setup, tear, FLEX_RATE)
    blocks, refusal = spot_blocks(steps, setup, tear)
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
        prelaunch_pass("p4"),
        "",
        "## Time and cost: Flex-start, Spot blocks, on-demand",
        "",
        f"**Provisioning, in this order** (adviser's ruling 5 of the 10 Nov 2026 paste): "
        f"Flex-start first, its request submitted {FLEX_REQUEST}; if no grant has come by "
        f"{SPOT_FROM}, Spot in {SPOT_BLOCKS} blocks of {SPOT_HOURS} h (below). On-demand "
        "is not offered. The measurement protocol is the same under each.",
        "",
        f"Cap = rate x planned hours x {B.CAP_FACTOR}; halt = {B.HALT_FACTOR} x planned hours "
        f"(rounded down); hard delete = the halt + {B.DELETE_AFTER_HALT_MIN} min. Rates are "
        "the plan's all-in planning rates (Section 5.2). Every boot pays the setup, the boot "
        f"steps ({', '.join(f'`{x}`' for x in B.BOOT['P4'])}: {boot_min:.0f} min) and the "
        f"teardown, {per_boot_h:.2f} h; a Spot block with ground truth also measures the "
        f"anchor cell ({B.ANCHOR_MIN:g} min, an estimate) and runs `late_check` and "
        "`trace_checks` again.",
        "",
        "**No suite record in the setup** (adviser's ruling of 10 Nov 2026, reversing the "
        "ruling h that would have added it): the bootstrap is run without `--suite-record`, "
        "and the 20 minutes it would have added are in neither the setup row nor any cap "
        "below. T2.26's five GPU variants still run in every preflight. The whole suite on a "
        "GPU host fails 27 tests that assume a CPU-only host "
        "(`docs/gpu_suite_failures_20261110.md`), and `scripts/gpu_suite_record.sh` is not "
        "run until it selects GPU-safe tests.",
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
        "| | Flex-start, one session (first) | Spot, in blocks (the fallback) | On-demand |",
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
        out += [f"**Why {SPOT_HOURS} h, not the {B.MAX_BLOCK_HOURS} h of 7 Nov:** at "
                f"{B.MAX_BLOCK_HOURS} h the blocks are refused: {refusal} (setup, teardown and "
                f"the boot steps take {per_boot_h * 60:.0f} of a block's "
                f"{B.MAX_BLOCK_HOURS * 60:.0f} minutes).", ""]
    out += [f"**The fallback, Spot blocks at {SPOT_HOURS} h:**", "",
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
