#!/usr/bin/env python3
"""The P4 launch card's step table and its cost comparison, built from what
they describe (adviser's ruling 7 of the eleventh 9 Nov 2026 paste): the
session's steps in `t64_session.T64`'s order with each step's minutes and
basis (`configs/t64/p4_card_minutes.json`), then Flex-start, Spot blocks
and on-demand side by side, each with its expected cost, its cap and its
interruption risk, the overhead every Spot block pays on its own boot
included. The measurement protocol is the same under each; only the
provisioning differs. Since the researcher's decision of 15 Nov 2026,
Flex-start and Spot are requested in parallel and the first RUNNING grant
runs P4 (before it, adviser's ruling 5 of the 10 Nov 2026 paste: Flex-start
first, Spot after about 22 Nov); Spot in four blocks of 2.3 h, every block
after the first on the short preflight (`t64_blocks.SHORT_PREFLIGHT_MIN`).
A plan that is not those four blocks is refused, not printed.

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

import budget_interlock as BI  # noqa: E402
import t64_blocks as B  # noqa: E402
from t64_session import T64  # noqa: E402
from t61_card import deselected_tests, prelaunch_pass  # noqa: E402

MINUTES = ROOT / "configs" / "t64" / "p4_card_minutes.json"
CARD = B.CARDS["P4"]
FLEX_RATE = 462.42           # plan Section 5.2, H100 Flex-start, us-central1, all-in
SPOT_RATE = B.SPOT_RATES["P4"]
SPOT_RATE_LOW = B.P4_SPOT_RATES["asia-southeast1"]   # the lower region, 15 Nov 2026
SPOT_HOURS, SPOT_BLOCKS = 2.3, 4   # the fallback as ruled, 10 Nov 2026
SHORT = B.SHORT_PREFLIGHT_MIN["P4"]
BEGIN = "<!-- built by scripts/t64_card.py from configs/t64/p4_card_minutes.json: rebuild, do not edit -->"
END = "<!-- end of the built region -->"


def load_minutes(path=MINUTES) -> dict:
    """The minutes file, refused unless it names exactly T64's steps."""
    m = json.loads(Path(path).read_text())
    names = [s.name for s in T64.steps]
    if set(m["steps"]) != set(names):
        raise ValueError(f"{path} names {sorted(m['steps'])}; T64's steps are {names}")
    return m


def spot_blocks(steps, setup, tear, suite=0.0):
    """The fallback's blocks at the ruled 2.3 h, refused unless they are the
    ruled four; with why the general 2.2 h length refuses, if it does. No
    blocks, and why, when the planner refuses 2.3 h as well (as it did from the
    adviser's rulings dated 13 Nov 2026, second set, when the full preflight
    on every boot left no block room for `host_components`, until the short
    preflight on later boots of 15 Nov 2026)."""
    try:
        blocks = B.plan_blocks(steps, setup, tear, boot=B.BOOT["P4"], rate=SPOT_RATE,
                               max_hours=SPOT_HOURS, suite_min=suite, short_preflight_min=SHORT)
    except ValueError as e:
        return [], str(e)
    if len(blocks) != SPOT_BLOCKS:
        raise ValueError(f"{len(blocks)} blocks at {SPOT_HOURS} h, not the ruled "
                         f"{SPOT_BLOCKS}: the fallback needs a new ruling")
    try:
        B.plan_blocks(steps, setup, tear, boot=B.BOOT["P4"], rate=SPOT_RATE, suite_min=suite,
                      short_preflight_min=SHORT)
        refusal = None
    except ValueError as e:
        refusal = str(e)
    return blocks, refusal


def costs(minutes: dict):
    """(steps, setup h, teardown h, suite min, the Flex-start session, the Spot
    blocks, why 2.2 h refuses)."""
    steps = [(s.name, float(sum(minutes["steps"][s.name]["minutes"]))) for s in T64.steps]
    setup, tear = minutes["setup_h"]["value"], minutes["teardown_h"]["value"]
    suite = float(minutes["suite_record_min"]["value"])
    run_min = sum(m for _, m in steps)
    flex = B._block([n for n, _ in steps], 0.0, run_min, setup, tear, FLEX_RATE, suite)
    blocks, refusal = spot_blocks(steps, setup, tear, suite)
    return steps, setup, tear, suite, flex, blocks, refusal


def caps(minutes: dict | None = None) -> dict:
    """P4's cap on each provisioning, for the budget interlock."""
    _, _, _, _, flex, blocks, _ = costs(minutes or load_minutes())
    return {"FLEX_START": flex.cap_inr, "SPOT": sum(b.cap_inr for b in blocks) if blocks else None}


def build(minutes: dict, on: str) -> str:
    steps, setup, tear, suite, flex, blocks, refusal = costs(minutes)
    run_min = sum(m for _, m in steps)
    mins = dict(steps)
    boot_min = sum(mins[s] for s in B.BOOT["P4"])
    per_boot_h = setup + boot_min / 60 + tear
    spot_h = sum(b.planned_h for b in blocks)
    spot_exp = math.ceil(spot_h * SPOT_RATE)
    worst = max(range(len(blocks)), key=lambda i: blocks[i].cap_inr) if blocks else None
    no = "no plan"
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
        "**Provisioning: Flex-start and Spot in parallel** (the researcher's decision of 15 Nov "
        "2026, which replaces the order of the adviser's ruling 5 of the 10 Nov 2026 paste): "
        "the first RUNNING grant runs P4, every other request is cancelled at once, and a "
        f"second grant is deleted unused; on Spot, in {SPOT_BLOCKS} blocks of {SPOT_HOURS} h "
        "(below). On-demand is not offered. The measurement protocol is the same under each.",
        "",
        f"Cap = rate x planned hours x {B.CAP_FACTOR}; halt = {B.HALT_FACTOR} x planned hours "
        f"(rounded down); hard delete = the halt + {B.DELETE_AFTER_HALT_MIN} min. Rates are "
        "the plan's all-in planning rates (Section 5.2). The suite record "
        f"(adviser's ruling 1h dated 10 Nov 2026; {suite:.0f} min) runs on the first boot "
        "only, so on Spot in block b1. Every boot pays the setup, the boot "
        f"steps ({', '.join(f'`{x}`' for x in B.BOOT['P4'])}: {boot_min:.0f} min) and the "
        f"teardown, {per_boot_h:.2f} h. A Spot block after the first runs the short preflight "
        f"({SHORT:g} min, an estimate, for the preflight's {mins['preflight']:.0f}): provenance, "
        "imports, the library check and the kernel contract; the GPU variants and the smoke "
        "checks ran on b1's boot, whose instance type, GPU, CPU, zone, image, driver and "
        "bootstrap every later block must share. A Spot block with ground truth also measures the "
        f"anchor cell ({B.ANCHOR_MIN:g} min, an estimate) and runs `late_check` and "
        "`trace_checks` again.",
        "",
        "**The suite record is in the setup** (adviser's ruling 1h dated 10 Nov 2026, which "
        "supersedes the earlier 10 Nov ruling that left it out): the first boot's bootstrap "
        "passes `--suite-record` (on Spot, block b1 only), and its "
        f"{suite:.0f} minutes are in the row below and in every cap. "
        f"`scripts/gpu_suite_record.sh` deselects the {deselected_tests()} tests that assume a CPU-only host "
        "(`tests/gpu_host_deselect.txt`). **Risk:** the selection has not run on any GPU "
        "host; a test that fails on the H100 for a reason the L4 did not show stops the "
        "bootstrap before any step, at the cost of the setup.",
        "",
        "| | Hours | ₹ |",
        "|---|---|---|",
        f"| Setup: write gate, deploy, preflight, bootstrap with XAttention | {setup} | "
        f"{math.ceil(setup * FLEX_RATE)} |",
        f"| Suite record: `--suite-record`, first boot only | {suite:.0f} min | "
        f"{math.ceil(suite / 60 * FLEX_RATE)} |",
        f"| The steps ({run_min:.0f} minutes) | {run_min / 60:.2f} | "
        f"{math.ceil(run_min / 60 * FLEX_RATE)} |",
        f"| Teardown: logs and manifests pull, delete, cleanup check | {tear} | "
        f"{math.ceil(tear * FLEX_RATE)} |",
        "",
        "| | Flex-start, one session (first) | Spot, in blocks (the fallback) | On-demand |",
        "|---|---|---|---|",
        f"| Rate | ₹{FLEX_RATE}/h (`us-central1`) | ₹{SPOT_RATE}/h (`us-central1`, the "
        f"higher; `asia-southeast1` ₹{SPOT_RATE_LOW}/h; the catalog read 15 Nov 2026) | "
        "not offered for `a3-highgpu-1g` |",
        f"| Boots | 1 | {len(blocks) if blocks else no} | |",
        f"| Overhead paid per boot | {per_boot_h:.2f} h | "
        + (f"{per_boot_h:.2f} h on b1, {per_boot_h - (mins['preflight'] - SHORT) / 60:.2f} h on "
           f"each of the {len(blocks) - 1} later, plus the anchor and the trace steps in each "
           "ground-truth block" if blocks else no) + " | |",
        f"| Planned hours | {flex.planned_h:.2f} | {f'{spot_h:.2f}' if blocks else no} | |",
        f"| Expected, nothing fails | ₹{math.ceil(flex.planned_h * FLEX_RATE)} | "
        f"{f'₹{spot_exp}' if blocks else no} | |",
        f"| Cap | ₹{flex.cap_inr} (halt {flex.halt_min} min, hard delete {flex.delete_min} min) | "
        + (f"₹{sum(b.cap_inr for b in blocks)}, the caps summed" if blocks else no) + " | |",
        "| Interruption risk | None once granted: a Flex-start VM runs up to seven days. The "
        "risk is the grant: a waiting request may not be granted, and costs nothing while it "
        "waits (`docs/launch_cards/h100_flex_attempts.md`) | "
        + ("Any block can be preempted at any time. Its rows are discarded and it is rerun on "
           "a new instance, so one interruption adds up to that block's cap (at most "
           f"₹{blocks[worst].cap_inr}, b{worst + 1})" if blocks else no)
        + " | Compute Engine offers A3 High with fewer than 8 GPUs only as Spot or Flex-start; "
        "`a3-highgpu-8g` would need 8 GPUs against a quota of 1 |",
        "",
    ]
    if not blocks:
        out += [f"**The fallback, Spot blocks at {SPOT_HOURS} h: no plan.** The planner refuses "
                f"them: {refusal} (setup, teardown and the boot steps take "
                f"{per_boot_h * 60:.0f} of a block's {SPOT_HOURS * 60:.0f} minutes). Spot is "
                "ruled out for P4 (the adviser's rulings dated 13 Nov 2026, item 6).", "",
                *BI.section("P4"), END]
        return "\n".join(out)
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
    out += ["", *BI.section("P4"), END]
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
