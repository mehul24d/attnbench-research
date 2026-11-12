# What the lock still waits on (12 Nov 2026, after the fourth set)

The adviser's ruling 5, second set of 12 Nov 2026: the lock (T4.4) moves to Sat 14 Nov 2026.
`docs/lock_values_prep.json` holds every T4.1 value with evidence in hand, typed as
`attnbench/lock/checklist.SCHEMA` requires: 25 of 40 (the rerun budget and the prompt-set hash added by the fourth set). Each entry names its source and reads
"decided" (a dated ruling or the researcher's decision) or "evidence" (measured or recorded,
nothing left to decide). The other 15 are null, with the draft's proposal beside them. The
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
| 8 | Probe bounds (15) | Proposed, not set: loop CV ≤ 0.01; loop median within 0.05 and null-launch median within 0.10 of the source pairing's | Read 12 Nov 2026 (ruling 7, fourth set): one `host_speed_end` probe per P1 repeat. Loop median 124.22 to 125.69 ns, CV ≤ 0.0027; null median 3.92 to 4.04 µs, CV 1.2 to 1.5, too heavy-tailed for a CV bound; between sessions 1.18% (loop) and 2.93% (null) | Not read | R (the bounds) |
| 9 | Chain-check tolerance (32) | 0.10 (aggregate) | Working value since 1 Nov 2026; every T3.2 session passed under it | **Read** | |
| 10 | Residual-spread tolerance (34) | 0.10 | Working value; T3.2c's and T3.2d's spreads under the idle-device residual are banked, not extracted | **Read** | |
| 11 | Pricing-mismatch share (35) | 0.05 | The researcher's working value of 6 Nov 2026; no session stops on it | **Read** | R (confirm) |
| 12 | Rerun budget (23) | **Decided:** `max_rerun_rounds` 5, `min_clean_rounds` 10 (ruling 2, fourth set) | The runner now reads `min_clean_rounds` (`Controls.min_clean_rounds`); the caps carry 5 reruns at every cell | **Read** | |
| 13 | Bootstrap settings (38) | level 0.95, `n_boot` 10,000, seed 0 | Plan default (T4.3 read A, finding 5) | Not read | |
| 14 | Prompt-set hash and tokenizer (26) | **Evidence:** set sha256 `cee769a2…`; transformers 4.46.0 | No session writes a prompt file: rebuilt on a CPU (`scripts/prompt_set_hash.py`), token counts equal to every P1 repeat's rows. **Flag:** rebuilt under 5.16.1; re-checked under 4.46.0 on the image | Not read | |

**Rows 1 to 5: the staged lock is built** (ruling 4, fourth set). Each takes a typed deferral,
`{"deferred": {"rule", "input", "fixed_by"}}`, with `fixed_by` naming the second lock, which
lands before P4's prediction. P2 runs on a staged lock; P4 refuses one. **Owed: the five rule
texts.** None is written yet, and a value whose rule is not written by 14 Nov cannot be
deferred. **Flag:** the ruling names P2's banked entry as the input. The table's inputs for
rows 2 to 5 are T1.5's pilot rows, T1.8 and the A100 dense control. **Row 6, the head-type
criterion, cannot be deferred:** the criterion is itself the rule. It needs the researcher's
T1.3 decision by 14 Nov.

**Also needed, not T4.1 values:**

- **The scored pairings:** H0 [P2], H1 [P2, P4], H2 [P2], H3 [P4], H4 to H6 []. P1 is never
  scored (T6.1, T6.4; ruling 3 of the ninth 9 Nov 2026 paste). **Flag:** the checker requires
  every hypothesis to name at least one pairing, so H4 to H6 [] fails it. Either the empty list
  is allowed (a code change, mutation-tested) or H4 to H6 name a pairing.
- **`pairing_values`:** P2 `gt_groups` {eager: the five arms}, `reps` 10, `dense_rounds` 5,
  `sampler_interval_s` 0.1. P4 the same, with the groups its blocks run. The compiled groups
  are dropped (ruling 4, third set of 12 Nov 2026).
- **The drift flag for P2 and P4: decided** (ruling 1, fourth set). Relative, 6.31% of the
  warm-up time, checked in session. Applied to P1's records, it refuses T3.2d's 8192 × 1 (1
  clean of 10); Appendix A, 12 Nov 2026 records the sensitivity.
- **Waivers** (ruling 5, fourth set): T2.21 now reads "Waived for P2/P4". P2's check lacks only
  T4.4, and P4's lacks T4.4 and T6.1.
- **P4's Flex-start capacity request,** as soon as the lock lands (ruling 8, fourth set). It
  is a GCP action and needs the researcher's own-words yes.

**The researcher's own items:** 31 (flagged-memset bound, 0 by the rule), 18 (the HDBI
threshold), 28 (the budget cut order: the compiled group is now dropped, so it leaves the
list), and the Flex-start rates.

## At the lock commit

1. Fill the remaining values. Commit the file as `docs/lock_values.json`, with `plan_sha256` of the plan at that commit.
2. `scripts/tag_lock.py --dry-run` must report the lock complete.
3. Re-check the timed-path pin at the lock commit against the lock's item 37 (`a7ce881e…` at `43b5963` unless the timed path changes again). Also check that `analysis_commit` is the full sha of `14e4419`.
4. Re-check P1's ledger entry (`predictions/P1/ledger/banked.json`): its training-file sha256 must equal the lock file's `p1_training.file_sha256`, not the draft's (adviser's ruling 5, second set of 12 Nov 2026). The entry was checked against the draft (`de7f0f44…`).
5. Run `scripts/prelaunch_cpu_pass.sh p2` and `p4` on the lock commit.
6. Tag the commit; pushing the tag needs the researcher's yes. Then build P2's Flex card from the lock commit, with the headroom table.
