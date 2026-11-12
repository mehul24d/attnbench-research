# What the lock still waits on (12 Nov 2026)

The adviser's ruling 5, second set of 12 Nov 2026: the lock (T4.4) moves to Sat 14 Nov 2026.
`docs/lock_values_prep.json` holds every T4.1 value with evidence in hand, typed as
`attnbench/lock/checklist.SCHEMA` requires: 23 of 40. Each entry names its source and reads
"decided" (a dated ruling or the researcher's decision) or "evidence" (measured or recorded,
nothing left to decide). The other 17 are null, with the draft's proposal beside them. The
file is not the lock. `docs/lock_values.json` does not exist yet, so P2 and P4 still refuse.

**A runner reads nothing from an incomplete lock.** `t31_steps.fill_lock_values` and
`t61_session.mask_gate_bound` refuse a lock the checker does not find complete. So P2
cannot start until all 40 are set, the scored pairings are stated, and the plan's sha256
is the lock commit's.

## Named by the ruling

| What | Item | Needs |
|---|---|---|
| User item 31 | Flagged-memset sensitivity bound | The researcher: a rule and a reported shift (0 s over both P1 repeats), not a number. The schema types it as a number (≥ 0), so the value is 0 with the rule as its source, if confirmed |
| User item 18 | HDBI boundedness threshold | The researcher: each row dropped by its own cell's floor (a small code change in `hdbi_fit.fit_threshold`, mutation-tested), then the fit on T3.2d at the lock |
| User item 28 | Budget cut order | The researcher: the decision point's "fewer blocks" as an exception to the 10 Nov 2026 rule that P4's Spot fallback drops no cell |
| Flex-start rates | Not a T4.1 value | The researcher supplies them. They go into P2's and P4's cards and the budget interlock (Appendix A, 11 Nov 2026, ruling 1i: re-check owed) |
| Compiled decision | Decided: dropped (the adviser's ruling 4, third set of 12 Nov 2026) | P2 and P4 run the eager group only. T4.1 items 29 and 33 are moot, and their values stand |

## The 14 values with no evidence, for one ruling (12 Nov 2026)

The adviser's ruling 6, third set of 12 Nov 2026: one table, ruled in one pass. **R** marks a
value that needs the researcher. "Not read" means no P2 or P4 runner reads the item
(`t31_steps.LOCK_SOURCED`), and neither pairing tests its hypothesis: P2 tests H0 to H2 (T6.1),
and P4 tests H1 and H3 (T6.4). **Conflict:** the 10 Nov 2026 rule allows a waiver only for a
task that feeds no lock value. The six tasks in rows 1 to 6 each feed one, so none of them can
be waived as that rule is written.

| # | T4.1 item (row) | Proposed value | Basis | Read by P2/P4 | R |
|---|---|---|---|---|---|
| 1 | H6 regret bound (2) | Not fixed by this lock; fixed at a second lock before H6 is scored. H6 has no scored pairing | T7.6 not built; neither pairing tests H6 | Not read | |
| 2 | Power target and n (6) | α 0.025, power 0.90; n fixed at the second lock | lock table row 6; T1.5 (p) not run | Not read | |
| 3 | Primary cell n, admissibility n (7) | 200 and 200, raised at the second lock if 1,300 × p is larger | lock table row 7 | Not read | |
| 4 | Admissibility Holm family (8) | At the second lock, from T1.8 | T1.8 not built | Not read | |
| 5 | Cross-card dense task-score bound (12) | At the second lock, after the A100 dense control | audit B1 open | Not read | |
| 6 | Head-type criterion (22) | At the second lock, from T1.3 | T1.3 not decided | Not read | R (T1.3 is the researcher's decision) |
| 7 | Null-kernel floor multiple (14) | Extracted on a CPU on 13 Nov from T2.2's banked null pass, at the floors quantile 0.95 | T2.2's null pass | Not read | |
| 8 | Probe bounds (15) | Extracted from the P1 repeats' probe records | Not among the Mac's pulls: a bucket read | Not read | R (bucket read yes) |
| 9 | Chain-check tolerance (32) | 0.10 (aggregate) | Working value since 1 Nov 2026; every T3.2 session passed under it | **Read** | |
| 10 | Residual-spread tolerance (34) | 0.10 | Working value; T3.2c's and T3.2d's spreads under the idle-device residual are banked, not extracted | **Read** | |
| 11 | Pricing-mismatch share (35) | 0.05 | The researcher's working value of 6 Nov 2026; no session stops on it | **Read** | R (confirm) |
| 12 | Rerun budget (23) | `max_rerun_rounds` 5, `min_clean_rounds` 10 | Launch 4 A: 3 of 10 rounds over 6.31% at 2048. T3.2d: 2 at 2048 × 1, 9 at 8192 × 1. **Flag:** the runner never reads `min_clean_rounds`: a group is complete only at `reps` clean rounds (`ground_truth.time_arms`). Any value below 10 needs a code change. Depends on the drift-flag ruling below | **Read** | |
| 13 | Bootstrap settings (38) | level 0.95, `n_boot` 10,000, seed 0 | Plan default (T4.3 read A, finding 5) | Not read | |
| 14 | Prompt-set hash and tokenizer (26) | The sha256 of T3.2d's prompt file; transformers 4.46.0 | On the image. The prompts are not among the Mac's pulls: a bucket read | Not read | R (bucket read yes) |

Rows 1 to 6 need the lock to allow a stated "fixed at a second lock" form for items that no
pairing reads. The checker has no such form: it requires every item typed. Building one is a
small change to `checklist.py`, with mutation tests, once the ruling is given.

**Also needed, not T4.1 values:**

- **The scored pairings:** H0 [P2], H1 [P2, P4], H2 [P2], H3 [P4], H4 to H6 []. P1 is never
  scored (T6.1, T6.4; ruling 3 of the ninth 9 Nov 2026 paste).
- **`pairing_values`:** P2 `gt_groups` {eager: the five arms}, `reps` 10, `dense_rounds` 5,
  `sampler_interval_s` 0.1. P4 the same, with the groups its blocks run. The compiled groups
  are dropped (ruling 4, third set of 12 Nov 2026).
- **The drift flag for P2 and P4** (`docs/rehearsal_l4_launch4_gt.md`). As built, it reads the
  lock's per-cell resolution floor: 1.39 ms at 2048 × 1, which would flag 10 of 10 of launch 4
  A's rounds. Proposed: flag in session by the 6.31% relative limit of 9 Nov 2026. This changes
  `time_arms` and the timed-path pin, and P2's 2048 cells may still come up short. No GPU is
  needed to verify it: the rule's input is the per-round record.

**The researcher's own items:** 31 (flagged-memset bound, 0 by the rule), 18 (the HDBI
threshold), 28 (the budget cut order: the compiled group is now dropped, so it leaves the
list), and the Flex-start rates.

## At the lock commit

1. Fill the remaining values. Commit the file as `docs/lock_values.json`, with `plan_sha256` of the plan at that commit.
2. `scripts/tag_lock.py --dry-run` must report the lock complete.
3. Re-check the timed-path pin at the lock commit against the lock's item 37 (`3aea782a…` at `d124c6d` unless the timed path changes again). Also check that `analysis_commit` is the full sha of `14e4419`.
4. Re-check P1's ledger entry (`predictions/P1/ledger/banked.json`): its training-file sha256 must equal the lock file's `p1_training.file_sha256`, not the draft's (adviser's ruling 5, second set of 12 Nov 2026). The entry was checked against the draft (`de7f0f44…`).
5. Run `scripts/prelaunch_cpu_pass.sh p2` and `p4` on the lock commit.
6. Tag the commit; pushing the tag needs the researcher's yes. Then build P2's Flex card from the lock commit, with the headroom table.
