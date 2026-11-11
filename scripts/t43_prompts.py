#!/usr/bin/env python3
"""T4.3's two adversarial-read prompts, A and B, built from one text so they
cannot drift apart (adviser's rulings dated 10 Nov 2026, item 2): both readers
cover all four areas, reader B in reverse order (area 4 first); the commit is
pinned when the researcher starts them; the report is `T43_read_<A|B>.md` in
the reader's scratchpad, printed to the user; deadline Fri 13 Nov 2026 12:00 IST.

    python3 scripts/t43_prompts.py                    # write docs/t43_reader_prompts.md
    python3 scripts/t43_prompts.py --print A --commit <sha>   # one prompt, pinned
    python3 scripts/t43_prompts.py --check            # exit 1 if the doc is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "t43_reader_prompts.md"
REPO = "https://github.com/mehul24d/attnbench-research"
PIN = "<COMMIT>"
DEADLINE = "Friday 13 Nov 2026, 12:00 IST"

# The plan's headings each area's questions rest on; the reader checks each
# number against the heading itself (`## 4. Hypotheses` and so on).
SECTIONS = {
    "4": "Hypotheses", "5.1": "Definitions", "5.3": "Per-pairing session protocol",
    "5.4": "Cost model", "5.6": "Confounders and controls", "5.7": "Quality measures",
    "6": "Operating rules (both Claudes, every session)", "7": "Decision gates",
}

AREAS = [
    ("Scoring and gate arithmetic",
     "Sections 4 (hypotheses), 5.1 (definitions), 5.7 (quality measures: H4's scoring and the "
     "Holm correction) and 7 (decision gates)",
     ["attnbench/predict/scoring.py", "attnbench/predict/residuals.py",
      "attnbench/predict/intervals.py", "attnbench/predict/naive.py",
      "attnbench/predict/baselines.py", "attnbench/predict/hdbi.py",
      "attnbench/lock/floors.py", "attnbench/lock/hdbi_fit.py", "attnbench/lock/holm.py",
      "attnbench/analysis/gate_a.py", "attnbench/analysis/gate_a_repeats.py",
      "attnbench/analysis/decision.py", "attnbench/analysis/block_drift.py",
      "attnbench/analysis/exact_noninferiority.py",
      "attnbench/components/mask_gate.py (and `mask_gate_bound` in `scripts/t61_session.py`)"],
     ["For each hypothesis H0 to H6 and each gate, does the code compute exactly the statistic, "
      "tolerance, aggregation (median, quantile, per pairing or pooled) and multiple-comparison "
      "correction the plan states? Name every mismatch.",
      "Which cells or rows can silently drop out of a score: the resolution floor (\"not "
      "interpreted\"), missing rounds, flagged clocks, density mismatches, the 16384-only label, "
      "or an empty group? Does a drop ever make a pass easier? Is each drop counted and reported?",
      "Can a gate pass on too few observations, or on a degenerate input such as all rows equal, "
      "one block, one session, a zero denominator or a NaN?",
      "Is the mask gate's verdict the two code paths on one card, at the bound the lock holds, "
      "with agreement reported only, as Appendix A (9 and 11 Nov 2026) states? Is the block "
      "drift bound applied to the anchor cell the plan names, before blocks are pooled?",
      "Are the bootstrap and the Holm correction seeded, sized and applied in the order the plan "
      "requires, over H4's family of m = 18?",
      "Is any number the analysis reports produced by nothing that could contradict it, such as "
      "a constant, a default or a value read back from its own output?"]),
    ("The commit machinery",
     "Sections 5.3 (the session protocol), 6 (operating rules) and 7 (Gate B)",
     ["attnbench/predict/commit.py (`build_record`, `write_prediction`, `sync_prediction`, "
      "`load_committed`)", "attnbench/predict/session_order.py",
      "scripts/t61_session.py: the `predict`, `ground_truth` and `late_check` steps, the hold, "
      "the strict stamp, blocks and their imports, the interrupted-block discard, `prereqs` "
      "(`unmet`, `waived`, `appendix_a_dates`)",
      "scripts/t31_steps.py: `build_prediction` and the ground-truth step",
      "scripts/timed_path.py and the timed-path identity (Appendix A, 10 Nov 2026)"],
     ["Can ground truth start, or any of its rows be read, before the prediction is synced plus "
      "the plan's gap and skew? Consider clocks moving, a resumed session, a second attempt, "
      "blocks on different instances and the store's creation time.",
      "Can a committed prediction be replaced, edited or re-synced after ground truth begins "
      "without `load_committed` refusing it? Consider the hash, the sidecar, a dirty tree, the "
      "code stamp and a store that clobbers.",
      "Are blocks of one pairing pooled only when they ran one commit, one spec and one instance "
      "identity, and is an interrupted block discarded in full?",
      "Can `prereqs` pass a session that the plan's Section 8 statuses say is not ready: a "
      "waiver on a gate, a waiver citing an Appendix A date that has no line, or a waiver for a "
      "pairing other than P2?"]),
    ("Design leakage",
     "Sections 5.3 and 5.4 (the cost model: which pairings each prediction is trained on)",
     ["attnbench/predict/training.py (`committed_training_set`)",
      "attnbench/predict/p1_training.py", "attnbench/predict/learned.py",
      "scripts/t61_session.py: `p2_staged_inputs`"],
     ["No P2 or P4 data exist yet, so \"leakage\" here means design leakage: a path by which, "
      "once those data exist, a prediction could see them. Can the training set for a pairing "
      "include that pairing's own rows, a later pairing's rows, or rows banked after its "
      "prediction was written?",
      "Can the learned predictor's configuration, features or hyperparameters be chosen, or "
      "re-chosen, using the pairing's own data?",
      "Is any input the session reads not hashed or checked against the lock: the training file, "
      "the banked masks, the configs?"]),
    ("The lock tooling",
     "Section 8's Phase 4 rows (T4.1 to T4.4) and Appendix A (dated decisions; a later line "
     "overrides an earlier one)",
     ["attnbench/lock/checklist.py", "attnbench/lock/blocks.py",
      "scripts/t31_steps.py: `fill_lock_values` and `LOCK_FILE` (`docs/lock_values.json`), the "
      "lock-file loader P2's session reads", "scripts/tag_lock.py",
      "docs/lock_package_draft.json (a draft: every value is null, and a test keeps it from "
      "reading as a lock)"],
     ["Can `checklist.py` accept a lock file with a missing, null or wrongly-typed value, or one "
      "whose `plan_sha256` is not the plan's?",
      "Can a session runner read a value the lock does not fix, or fall back to a working value "
      "when the lock is absent?",
      "Does every item T4.1 lists have a key in the checklist and the draft, and is any value "
      "the draft marks decided (H0, H1 to H3, H4's bound and m = 18, the ±0.01 density "
      "tolerance, the timed-path identity) chosen after data it could have seen?"]),
]


def prompt(reader: str, commit: str = PIN) -> str:
    order = list(range(4)) if reader == "A" else list(range(3, -1, -1))
    pin = (f"run `git checkout {commit}`" if commit != PIN else
           f"run `git checkout {PIN}`, the full sha of `main`'s tip that the researcher gives "
           "you when starting this session")
    lines = [
        f"You are independent adversarial reader {reader} for a pre-registered benchmarking "
        "study. Each hardware pairing's latency prediction is written, hashed and synced to a "
        "store before any of that pairing's ground truth is measured, and is trained only on "
        "pairings banked earlier. Your job is to find the ways the code and the written plan "
        "could let a result through that the plan would not allow: a pass that should be a "
        "fail, a number nothing could contradict, a prediction that could see its own ground "
        "truth, or a lock accepted while incomplete. You did not write this code. Assume it is "
        "wrong somewhere and find where.",
        "",
        f"**Repository.** `{REPO}`. Clone it into your session's scratchpad directory, {pin}, "
        "and read only that commit. The plan is `docs/RESEARCH_PLAN.md`, the single source of "
        "truth; Appendix A holds dated decisions, and a later line overrides an earlier one.",
        "",
        "**Section numbers.** This prompt cites the plan by section number. Before relying on "
        "one, check it against the plan's own headings (`grep -n '^#' docs/RESEARCH_PLAN.md`). "
        "These are the headings it expects: "
        + "; ".join(f"{k} {v}" for k, v in SECTIONS.items())
        + ". Report any mismatch as a finding.",
        "",
        "**Your scope: all four areas below, in this order"
        + (" (reverse order: area 4 first)" if reader == "B" else "") + ".** Read each area's "
        "files and their tests under `tests/`, then answer its questions with evidence.",
        "",
    ]
    for pos, i in enumerate(order, 1):
        name, sections, files, questions = AREAS[i]
        lines += [f"### {pos}. Area {i + 1}: {name}", "", f"Plan: {sections}.", "", "Files:", ""]
        lines += [f"- `{f.split(' ')[0]}`{f[len(f.split(' ')[0]):]}" for f in files]
        lines += ["", "Questions:", ""]
        lines += [f"{n}. {q}" for n, q in enumerate(questions, 1)]
        lines += [""]
    lines += [
        "**Rules.**",
        "",
        "- Read code and run CPU tests only: `python3 -m pytest tests/ -q` works on a CPU with "
        "no GPU. A CPU dry run is allowed: "
        "`python3 scripts/t61_session.py run --dry-run --session <a scratch dir>`.",
        "- Do not run any `scripts/gcp_*` script or any cloud, `gcloud`, `gsutil` or `aws` "
        "command, and do not read any bucket. Read the scripts if you need to.",
        "- Do not push, open a pull request or change files in the repository.",
        "- Do not read the other reader's report (`T43_read_A.md` or `T43_read_B.md`, whichever is "
        "not yours) or any other session's files, and do not contact another session. The two "
        "reads run in parallel and must stay independent.",
        "- `docs/t43_first_read.md` is an earlier read by the agent that wrote much of the code. "
        "Read it only after your own findings are written, then add a short section on where you "
        "agree, where you disagree and what it missed.",
        "",
        f"**Deliverable.** One Markdown file, `T43_read_{reader}.md`, saved in your session's "
        "scratchpad directory (not in the repository), and its full text printed to the user "
        "at the end of your session. It holds:",
        "",
        "- the commit you read (full sha);",
        "- your findings, most severe first; for each, the area, the file and line, the plan "
        "sentence it conflicts with (quoted, with its section), a concrete failure scenario (the "
        "inputs and the wrong result), and whether you verified it by running code or only "
        "suspect it;",
        "- any section number that did not match the plan's headings;",
        "- what you did not read;",
        "- the section comparing your findings with `docs/t43_first_read.md`.",
        "",
        f"Keep the report under 2,500 words. Deadline: {DEADLINE}.",
    ]
    return "\n".join(lines)


def doc() -> str:
    head = [
        "# T4.3: the two adversarial reads, prompts A and B (11 Nov 2026)",
        "",
        "Built by `scripts/t43_prompts.py`; rebuild it, do not edit it. Adviser's rulings dated "
        "10 Nov 2026 (item 1e and item 2): the two reads run in parallel, superseding the "
        "sequence of 31 Oct 2026 for calendar reasons; both readers cover all four areas, reader "
        "B in reverse order; after both reads the findings are fixed, then the suite and the "
        "mutation checks are rerun, with no third read (Appendix A, 11 Nov 2026). The researcher "
        "starts each in a fresh session, one prompt per session, and pins the commit: "
        "`python3 scripts/t43_prompts.py --print A --commit <main's full sha>` prints prompt A "
        f"with `{PIN}` filled in. Deadline: {DEADLINE}.",
        "",
        "Not launched by Claude Code.",
        "",
    ]
    return "\n".join(head + ["---", "", "## Prompt A", "", prompt("A"), "", "---", "",
                             "## Prompt B", "", prompt("B"), ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print", dest="reader", choices=("A", "B"))
    ap.add_argument("--commit", default=PIN)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.reader:
        print(prompt(args.reader, args.commit))
        return 0
    if args.check:
        same = DOC.read_text(encoding="utf-8") == doc()
        print("the prompts are current" if same else "the prompts are stale: rebuild them")
        return 0 if same else 1
    DOC.write_text(doc(), encoding="utf-8")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
