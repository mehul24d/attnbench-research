# T4.3: the two adversarial reads, prompts A and B (10 Nov 2026)

These are the prompts for T4.3's two independent adversarial reads (Section 8). The researcher
starts each read in a fresh session, one prompt per session, and the two run in parallel
(adviser's ruling of 9 Nov 2026). **Flag:** the researcher's decision of 31 Oct 2026 runs the
reads in sequence, the second after T4.1 has written the lock values. The plan has not
reconciled the two (Appendix A, 10 Nov 2026). Prompt B therefore reads the lock *draft* and
the lock tooling, not written lock values.

Both reads are pinned to `main` at `ef43165` (`ef43165f10ed8baa382e46b141a2b23d7363ec22`). Deadline: Friday 13 Nov 2026, 12:00 IST. Each prompt is
self-contained, and a reader is given only its own prompt.

---

## Prompt A: scoring and gate arithmetic

You are an independent adversarial reader for a pre-registered benchmarking study. Your job
is to find the ways its scoring code and decision gates could report a result that the
written plan would not allow: a pass that should be a fail, a number that cannot be
contradicted, or a gap between what the plan says and what the code computes. You did not
write this code. Assume it is wrong somewhere and find where.

**Repository.** `https://github.com/mehul24d/attnbench-research`. Clone it, run
`git checkout ef43165`, and read only that commit. The plan is `docs/RESEARCH_PLAN.md`, the
single source of truth. Read Sections 4 (hypotheses), 5.1, 5.3 and 5.4 (definitions, the
session protocol, the cost model), 7 (decision gates) and Appendix A (dated decisions; a later
line overrides an earlier one).

**Your scope: scoring and gate arithmetic.**

- `attnbench/predict/scoring.py`, `residuals.py`, `intervals.py`, `naive.py`, `baselines.py`,
  `hdbi.py`
- `attnbench/lock/floors.py`, `hdbi_fit.py`, `holm.py`
- `attnbench/analysis/gate_a.py`, `gate_a_repeats.py`, `decision.py`, `block_drift.py`,
  `exact_noninferiority.py`
- the tests for each, under `tests/`

**Questions to answer, each with evidence:**

1. For each hypothesis H0 to H6 and each gate, does the code compute exactly the statistic,
   tolerance, aggregation (median, quantile, per pairing or pooled) and multiple-comparison
   correction the plan states? Name every mismatch.
2. Which cells or rows can silently drop out of a score: the resolution floor ("not
   interpreted"), missing rounds, flagged clocks, density mismatches, the 16384-only label, or
   an empty group? Does a drop ever make a pass easier? Is each drop counted and reported?
3. Can a gate pass on too few observations, or with a degenerate input such as all rows equal,
   one block, one session, a zero denominator or a NaN?
4. Is the block drift bound (`block_drift.py`) applied to the anchor cell the plan names, at
   the bound the plan names, before blocks are pooled?
5. Are the bootstrap and the Holm correction seeded, sized and applied in the order the plan
   requires?
6. Is any number that the analysis reports produced by nothing that could contradict it, such
   as a constant, a default or a value read back from its own output?

**Rules.**

- Read code and run CPU tests only: `python3 -m pytest tests/ -q` works on a CPU with no GPU.
- Do not run any `scripts/gcp_*` script or any cloud, `gcloud`, `gsutil` or `aws` command.
  Read the scripts if you need to. Do not read any bucket.
- Do not push, open a pull request or change files in the repository. Write your findings to
  a new file outside the repository.
- `docs/t43_first_read.md` is an earlier read by the agent that wrote much of the code.
  Read it only after your own findings are written. Then add a short section on where you
  agree, where you disagree, and what it missed.

**Deliverable.** One Markdown file, `t43_read_A_<your name or session>.md`, with:

- your findings, most severe first; for each, the file and line, the plan sentence it
  conflicts with (quoted, with its section), a concrete failure scenario (the inputs and the
  wrong result), and whether you verified it by running code or only suspect it;
- a list of what you did not read;
- the section comparing your findings with `docs/t43_first_read.md`.

Keep the report under 2,000 words. Deadline: Friday 13 Nov 2026, 12:00 IST.

---

## Prompt B: commit machinery, leakage, and the lock tooling

You are an independent adversarial reader for a pre-registered benchmarking study. Each
hardware pairing's latency prediction is written, hashed and synced to a store *before* any
of that pairing's ground truth is measured, and it is trained only on pairings banked earlier.
Your job is to find any way that rule can be broken without the code refusing: a prediction
that sees its own ground truth, a training set that leaks the pairing's own or later rows, a
committed file that can be swapped, or a lock that can be accepted while incomplete or while
holding values chosen after the confirmatory data. You did not write this code. Assume it is
wrong somewhere and find where.

**Repository.** `https://github.com/mehul24d/attnbench-research`. Clone it, run
`git checkout ef43165`, and read only that commit. The plan is `docs/RESEARCH_PLAN.md`, the
single source of truth. Read Sections 5.3 and 5.4 (the session protocol, the cost model), 6
(operating rules), 7 (Gate B), Section 8's Phase 4 rows (T4.1 to T4.4) and Appendix A (dated
decisions; a later line overrides an earlier one).

**Your scope: the commit machinery, leakage and the lock tooling.**

- `attnbench/predict/commit.py` (`build_record`, `write_prediction`, `sync_prediction`,
  `load_committed`), `session_order.py`, `training.py` (`committed_training_set`),
  `p1_training.py`, `learned.py`
- `scripts/t61_session.py`: the `predict`, `ground_truth` and `late_check` steps, the hold, the
  strict stamp, blocks and their imports, and the interrupted-session discard
- `scripts/t31_steps.py`: `build_prediction`, `fill_lock_values`, the ground-truth step
- `attnbench/lock/checklist.py`, `blocks.py`; `docs/lock_package_draft.json` (a draft: every
  value is null, and a test keeps it from reading as a lock)
- `scripts/timed_path.py` and the timed-path identity in Appendix A, 10 Nov 2026
- the tests for each, under `tests/`

**Questions to answer, each with evidence:**

1. Can ground truth start, or any of its rows be read, before the prediction is synced plus the
   plan's gap and skew? Consider clocks moving, a resumed session, a second attempt, blocks run
   on different instances, and the store's creation time.
2. Can a committed prediction be replaced, edited or re-synced after ground truth begins
   without `load_committed` refusing it? Consider the hash, the sidecar, a dirty tree, the code
   stamp, and a store that clobbers.
3. Can the training set for a pairing include that pairing's own rows, a later pairing's rows,
   or rows banked after its prediction was written? Can the learned predictor's configuration
   or features be chosen using the pairing's data?
4. Can the lock checker (`checklist.py`) accept a lock file with a missing, null or
   wrongly-typed value? Can the session runners read a value that the lock does not fix?
5. Are blocks of one pairing pooled only when they ran one commit, one spec and one instance
   identity, and is an interrupted block discarded in full?
6. Is there any input the session reads that is not hashed or checked against the lock: the
   training file, the banked masks or the configs?

**Rules.**

- Read code and run CPU tests only: `python3 -m pytest tests/ -q` works on a CPU with no GPU.
  A CPU dry run of a session is allowed:
  `python3 scripts/t61_session.py run --dry-run --session /tmp/<dir>`.
- Do not run any `scripts/gcp_*` script or any cloud, `gcloud`, `gsutil` or `aws` command.
  Read the scripts if you need to. Do not read any bucket.
- Do not push, open a pull request or change files in the repository. Write your findings to
  a new file outside the repository.
- `docs/t43_first_read.md` is an earlier read by the agent that wrote much of the code.
  Read it only after your own findings are written. Then add a short section on where you
  agree, where you disagree, and what it missed.

**Deliverable.** One Markdown file, `t43_read_B_<your name or session>.md`, with:

- your findings, most severe first; for each, the file and line, the plan sentence it
  conflicts with (quoted, with its section), a concrete failure scenario (the inputs and the
  wrong result), and whether you verified it by running code or only suspect it;
- a list of what you did not read;
- the section comparing your findings with `docs/t43_first_read.md`.

Keep the report under 2,000 words. Deadline: Friday 13 Nov 2026, 12:00 IST.
