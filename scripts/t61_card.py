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

import budget_interlock as BI  # noqa: E402
import t61_session as S  # noqa: E402
import t64_blocks as B  # noqa: E402

MINUTES = ROOT / "configs" / "t61" / "p2_card_minutes.json"
CARD = B.CARDS["P2"]


def deselected_tests() -> int:
    """The tests `scripts/gpu_suite_record.sh` deselects on a GPU host
    (`tests/gpu_host_deselect.txt`, one per line, comments aside)."""
    lines = (ROOT / "tests" / "gpu_host_deselect.txt").read_text().splitlines()
    return sum(1 for x in lines if x.strip() and not x.lstrip().startswith("#"))
PLAN = ROOT / "docs" / "RESEARCH_PLAN.md"
FLEX_RATE = 297.47          # plan Section 5.2, A100 Flex-start, all-in planning rate
SPOT_RATE = B.SPOT_RATES["P2"]
ZONE = "asia-southeast1-c"  # the only zone offering a2-ultragpu-1g (plan Section 5.2)



def prelaunch_pass(config: str) -> str:
    """The card's pre-launch CPU pass paragraph (adviser's ruling 4 of the
    11 Nov 2026 paste); the P4 card prints it too."""
    return (f"**Pre-launch CPU pass, before the yes is asked** (adviser's ruling 4 of the 11 Nov "
            f"2026 paste). On this card's commit, from a clean tree: `ATTNBENCH_PRELAUNCH_MASKS=<T3.2d's "
            f"step 5 XAttention masks, staged> scripts/prelaunch_cpu_pass.sh "
            f"{config} results/p1_training/p1_training.json.gz` (the masks: the third lock-wave "
            f"rulings of 12 Nov 2026, item 2). It runs the session's real config "
            "and name through the real start checks (the exit wrapper, a clean tree, the "
            "timed-path pin, the banking times) and every step a CPU can run with its real body, "
            "prereqs and the staged inputs first; only the GPU steps, the device and the bucket "
            "are mocked (`tests/test_prelaunch_cpu_pass.py`). The yes is asked only after it "
            "prints PASSED at the commit the yes names, and the request cites its record "
            "(`prelaunch_cpu_pass.json`). Until the lock commit it refuses at the start: no "
            "`docs/lock_values.json`, so the timed path has no pinned value.")

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


def costs(minutes: dict):
    """(steps, setup h, teardown h, suite min, the Flex-start session, the Spot blocks)."""
    steps = [(s.name, float(sum(minutes["steps"][s.name]["minutes"]))) for s in S.T61.steps]
    setup, tear = minutes["setup_h"]["value"], minutes["teardown_h"]["value"]
    suite = float(minutes["suite_record_min"]["value"])
    run_min = sum(m for _, m in steps)
    flex = B._block([n for n, _ in steps], 0.0, run_min, setup, tear, FLEX_RATE, suite)
    blocks = B.plan_blocks(steps, setup, tear, boot=B.BOOT["P2"], rate=SPOT_RATE,
                           suite_min=suite)
    return steps, setup, tear, suite, flex, blocks


def caps(minutes: dict | None = None) -> dict:
    """P2's cap on each provisioning, for the budget interlock."""
    *_, flex, blocks = costs(minutes or load_minutes())
    return {"FLEX_START": flex.cap_inr, "SPOT": sum(b.cap_inr for b in blocks)}


def build(minutes: dict, plan_text: str, on: str) -> str:
    steps, setup, tear, suite, flex, blocks = costs(minutes)
    run_min = sum(m for _, m in steps)
    statuses = S.plan_statuses(plan_text)
    appendix = S.appendix_a_dates(plan_text)
    unmet = {r["task"] for r in S.unmet(S.T61.requirements, statuses, appendix, S.T61.pairing)}
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
        "`python3 scripts/t61_session.py run --session results/t61_p2_<UTC>` on the "
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
        "- Nothing for P1's banking time: a real run reads it, and the training file's "
        "sha256, from the pairing ledger (`predictions/P1/ledger/banked.json`) and refuses "
        "`--banked-at` (ruling 1e of 11 Nov 2026).",
        "",
        "**Cheap checks first** (adviser's ruling 2 of the 10 Nov 2026 paste: a failure costs "
        "minutes). `prereqs`, step 1, checks every staged input a later step reads "
        "(`t61_session.p2_staged_inputs`): the lock is complete and its mask-gate bound is set, "
        f"and `{S.P1_MASKS}/` reads back (both read by step 6); the training file's sha256 is "
        "the lock's `p1_training.file_sha256` and the learned configuration is set (both read "
        "by step 11, after about three hours). Before 10 Nov 2026 the sha256 was checked "
        "nowhere in the runner. On Spot, a block checks only what its own steps read. "
        "Then `preflight` (3 min) checks the instance, the GPU variants and lightgbm; the "
        "first long step is `probe`.",
        "",
        prelaunch_pass("p2"),
        "",
        "## Not exercised before P2 (documented risks)",
        "",
        "The L4 rehearsal's launch 4 is the last rehearsal launch (adviser's ruling 2 of the "
        "10 Nov 2026 paste). At best it runs steps 1 to 15 of P2's runner on P2-R once, "
        "with two blocks and a simulated reset. These are still not exercised on a GPU "
        "before P2's own session, and each is a risk P2 carries:",
        "",
        "- The two-path mask verdict (T2.23, `real_mask_gate`): the L4 has no Triton path, "
        "so the rehearsal compares one path against P1's banked masks only.",
        "- The A100 instance: `a2-ultragpu-1g`'s provenance, Flex-start and Spot provisioning, "
        "Spot's own preemption (the rehearsal resets an on-demand instance) and P2's four "
        "Spot blocks (the rehearsal runs two).",
        "- The lock: the session's values from `docs/lock_values.json` "
        "(`t31_steps.fill_lock_values`), the lock's mask-gate bound and learned "
        "configuration. The rehearsal runs P1's working values and configuration 4 of "
        "`docs/learned_cv_p1_20261109.json`. Step 1 checks that they are present, on a CPU only.",
        "- P2's full grid: lengths above 8192, batches above 1, and the Kineto trace sizes, "
        "disk and memory there. The rehearsal's grid is 2048 and 8192 at batch 1.",
        "- The compiled groups beyond 2048 at batch 1 (scope v), and all of them if launch 4's "
        "compiled stage does not run clean. If the compiled path is not clean by 14 Nov 2026, "
        "the compiled group is dropped from P2 and P4 and the limitation is written in the "
        "plan (Section 8 calendar; ruling 2 of the tenth 9 Nov 2026 paste).",
        "- Step 1's refusal on the plan rows not Done (the rehearsal records them and does not "
        "refuse), and the teardown and pull from the A100's zone.",
        "",
        "## Time and cost",
        "",
        f"Cap = rate x planned hours x {B.CAP_FACTOR}; halt = {B.HALT_FACTOR} x planned hours "
        f"(rounded down); hard delete = the halt + {B.DELETE_AFTER_HALT_MIN} min. Rates are "
        "the plan's all-in planning rates (Section 5.2); `GCP_RATE_INR_HR` takes the rate "
        "of the model actually created.",
        "",
        "**The suite record is in the setup** (adviser's ruling 1h dated 10 Nov 2026, which "
        "supersedes the earlier 10 Nov ruling that left it out): the bootstrap runs "
        "`gcp_bootstrap_instance.sh \"$NAME\" \"$ZONE\" --suite-record` on the first boot "
        f"(on Spot, block b1 only), and its {suite:.0f} minutes are in the row below and in "
        f"every cap. `scripts/gpu_suite_record.sh` deselects the {deselected_tests()} tests that assume a "
        "CPU-only host (`tests/gpu_host_deselect.txt`, from "
        "`docs/gpu_suite_failures_20261110.md`). **Risk:** the selection has not run on any "
        "GPU host; a test that fails on the A100 for a reason the L4 did not show stops the "
        "bootstrap, before any step, at the cost of the setup. P2's relaunches run "
        "without it (the researcher's ruling of 12 Nov 2026, after launch 1 stopped in it); "
        "its minutes stay in every cap as margin.",
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
        *BI.section("P2"),
        "## What `prereqs` checks",
        "",
        "The step refuses a real session unless every task below reads Done in the deployed "
        f"plan, or, for a task that is not a gate, \"{S.WAIVER}\" with a citation of a dated "
        "Appendix A line that exists (adviser's ruling 1b dated 10 Nov 2026). "
        + (f"**As the plan stands, {len(unmet)} of {len(S.T61.requirements)} are not met**, so "
           "the session would stop at step 1." if unmet else
           f"As the plan stands, all {len(S.T61.requirements)} are met."),
        "",
        "| Task | Why the session needs it | Gate | Status in the plan | Met |",
        "|---|---|---|---|---|",
    ]
    for r in S.T61.requirements:
        s = statuses.get(r.task, '')
        met = "no" if r.task in unmet else ("waived" if S.waived(r, s, appendix, "P2") else "yes")
        out.append(f"| {r.task} | {r.why} | {'yes' if r.gate else ''} | {status_head(s)} | {met} |")
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
