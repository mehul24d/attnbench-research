#!/usr/bin/env python3
"""The P2 launch card, built from what it describes (adviser's ruling 9 of
the tenth 9 Nov 2026 paste): the session's steps in `t61_session.T61`'s
order, each step's minutes and their basis (`configs/t61/p2_card_minutes.json`),
the blocks `t64_blocks.plan_blocks` gives at P2's Spot rate, a single
Flex-start session under the same 1.3 x rule, and every task `prereqs`
checks with its status in the plan as it stands. A hand-written card went
stale within days (28 Oct 2026); this one is rebuilt instead of edited.

The card is a proposal, never a launch: it names no project, bucket or
instance; the launch yes names those.

    python3 scripts/t61_card.py [--date 2026-11-09] [--out docs/launch_cards/t61_p2_a100.md]
    python3 scripts/t61_card.py --check     # exit 1 if the card on disk is not this build
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

import t61_session as S  # noqa: E402
import t64_blocks as B  # noqa: E402

MINUTES = ROOT / "configs" / "t61" / "p2_card_minutes.json"
CARD = B.CARDS["P2"]
PLAN = ROOT / "docs" / "RESEARCH_PLAN.md"
FLEX_RATE = 297.47          # plan Section 5.2, A100 Flex-start, all-in planning rate
SPOT_RATE = B.SPOT_RATES["P2"]
ZONE = "asia-southeast1-c"  # the only zone offering a2-ultragpu-1g (plan Section 5.2)


def load_minutes(path=MINUTES) -> dict:
    """The minutes file, refused unless it names exactly T61's steps."""
    m = json.loads(Path(path).read_text())
    names = [s.name for s in S.T61.steps]
    if set(m["steps"]) != set(names):
        raise ValueError(f"{path} names {sorted(m['steps'])}; T61's steps are {names}")
    return m


def status_head(status: str) -> str:
    """A plan status cell's first clause: 'Partly built (9 Nov 2026)'."""
    return status.split(":")[0].split(";")[0].strip() if status else "no row in Section 8"


def build(minutes: dict, plan_text: str, on: str) -> str:
    steps = [(s.name, float(sum(minutes["steps"][s.name]["minutes"]))) for s in S.T61.steps]
    setup, tear = minutes["setup_h"]["value"], minutes["teardown_h"]["value"]
    run_min = sum(m for _, m in steps)
    flex = B._block([n for n, _ in steps], 0.0, run_min, setup, tear, FLEX_RATE)
    blocks = B.plan_blocks(steps, setup, tear, boot=B.BOOT["P2"], rate=SPOT_RATE)
    statuses = S.plan_statuses(plan_text)
    unmet = {r["task"] for r in S.unmet(S.T61.requirements, statuses)}
    measured = [n for n, _ in steps if "measured" in minutes["steps"][n]["basis"]]
    out = [
        "# Launch card: the P2 session (T6.1, Cascade Lake + A100)",
        "",
        f"Built {on} by `scripts/t61_card.py` from the session runner's step list, "
        "`configs/t61/p2_card_minutes.json` and the plan's Section 8 statuses. "
        "Rebuild it; do not edit it by hand. **Not launched.** The researcher's launch yes "
        "names the instance, zone, commit and cap. Nothing in this card has run on an A100. "
        f"{len(measured)} of {len(steps)} steps' minutes are measured on P1's L4 (T3.2d); "
        "the rest are estimates, and each row names its basis.",
        "",
        "## What runs",
        "",
        "`python3 scripts/t61_session.py run --banked-at P1=<when P1 was banked>` on the "
        "instance from the repo root, after deploy, preflight, the GCS write gate and "
        "`scripts/gcp_bootstrap_instance.sh`. Each step runs in its own process through "
        "`scripts/run_phase.sh`, and the first failure stops the session. On Spot it runs "
        "in blocks (`--block`), each on its own boot, and an interrupted block's rows are "
        "discarded.",
        "",
        "| # | Step | Task | What it does | Basis | Expected minutes |",
        "|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(S.T61.steps, 1):
        e = minutes["steps"][s.name]
        out.append(f"| {i} | `{s.name}` | {s.task} | {s.what} | {e['basis']} | "
                   f"{' + '.join(str(int(x)) for x in e['minutes'])} |")
    out += [
        "",
        f"Steps total {run_min:.0f} minutes. The compiled groups are not in: their ground "
        "truth is unresolved, and they come in only if it is resolved by 14 Nov 2026 "
        "(adviser's ruling 2 of the tenth 9 Nov 2026 paste). If they come in, this card "
        "is rebuilt with their host time added to `ground_truth`, and each cap below rises "
        "with it.",
        "",
        "## Staged on the instance before `run`",
        "",
        f"- `{S.P1_MASKS}/`: P1's banked XAttention masks, copied from T3.2d's step 5 "
        "(`<cell>/recording/masks.{npz,json}` for every `xattention` and "
        "`xattention_noassert` cell), for the mask gate (T2.23).",
        "- The P1 training file at `results/p1_training/p1_training.json.gz`, its sha256 "
        "the lock's.",
        "- `docs/lock_values.json`, the tagged lock (T4.4). Without it the session's values "
        "and the mask-gate bound are unset, and the steps refuse.",
        "- `--banked-at P1=<ISO time>`, the time P1 was banked.",
        "",
        "## Time and cost",
        "",
        f"Cap = rate x planned hours x {B.CAP_FACTOR}; halt = {B.HALT_FACTOR} x planned hours "
        f"(rounded down); hard delete = the halt + {B.DELETE_AFTER_HALT_MIN} min. Rates are "
        "the plan's all-in planning rates (Section 5.2); `GCP_RATE_INR_HR` takes the rate "
        "of the model actually created.",
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
        f"| **Flex-start (₹{FLEX_RATE}/h), one session, planned** | **{flex.planned_h:.2f}** | "
        f"**{math.ceil(flex.planned_h * FLEX_RATE)}** |",
        f"| In-guest halt ({flex.halt_min} min) | {flex.halt_min / 60:.2f} | "
        f"{flex.cost_at_halt_inr} |",
        f"| Hard delete ({flex.delete_min} min) | {flex.delete_min / 60:.2f} | "
        f"{flex.cost_at_delete_inr} |",
        f"| **The cap** | | **{flex.cap_inr}** |",
        "",
        f"**On Spot (₹{SPOT_RATE}/h), in {len(blocks)} blocks of at most {B.MAX_BLOCK_HOURS} h** "
        "(`scripts/t64_blocks.py --pairing P2`, which reads this card):",
        "",
        "| Block | Steps | Planned h | Cap ₹ | Halt min | Hard delete min |",
        "|---|---|---|---|---|---|",
    ]
    for n, b in enumerate(blocks, 1):
        out.append(f"| b{n} | {', '.join(f'`{x}`' for x in b.steps)} | {b.planned_h:.2f} | "
                   f"{b.cap_inr} | {b.halt_min} | {b.delete_min} |")
    tot = sum(b.planned_h for b in blocks)
    out += [
        "",
        f"{tot:.2f} planned hours on Spot, ₹{math.ceil(tot * SPOT_RATE)} expected, the caps "
        f"summing to ₹{sum(b.cap_inr for b in blocks)}. Each block also runs the boot steps "
        f"({', '.join(f'`{x}`' for x in B.BOOT['P2'])}) and, with ground truth, the anchor "
        f"cell ({B.ANCHOR_MIN:g} min, an estimate). Zone `{ZONE}`, `a2-ultragpu-1g`. P2 and P4 "
        "cannot overlap: the GPU quota across regions is 1.",
        "",
        "## What `prereqs` checks",
        "",
        f"The step refuses a real session unless every task below reads Done in the deployed "
        f"plan. **As the plan stands, {len(unmet)} of {len(S.T61.requirements)} do not**, so "
        "the session would stop at step 1.",
        "",
        "| Task | Why the session needs it | Status in the plan | Done |",
        "|---|---|---|---|",
    ]
    for r in S.T61.requirements:
        out.append(f"| {r.task} | {r.why} | {status_head(statuses.get(r.task, ''))} | "
                   f"{'no' if r.task in unmet else 'yes'} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--out", default=str(CARD))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless the card on disk is this build (its date kept)")
    args = ap.parse_args(argv)
    text = PLAN.read_text(encoding="utf-8")
    if args.check:
        disk = Path(args.out).read_text()
        on = disk.split("Built ", 1)[1].split(" ", 1)[0] if "Built " in disk else args.date
        same = disk == build(load_minutes(), text, on)
        print("the card is current" if same else "the card is stale: rebuild it")
        return 0 if same else 1
    Path(args.out).write_text(build(load_minutes(), text, args.date))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
