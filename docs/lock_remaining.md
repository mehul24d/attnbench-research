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
| Compiled decision | P2's `gt_groups`; T4.1 item 33 moot if dropped | Launch 4 (`docs/launch_cards/t61_rehearsal_l4.md`). Clean means `compiled/result.json` reads passed, both groups are complete, there is no recompile after warm-up, and the twin and mask-dependence checks pass. Otherwise the compiled groups are dropped from P2 on 14 Nov 2026 and the limitation is written |

## Not in the ruling's list (flagged)

Each of these is null in the prep file, and the lock cannot be complete without it.

**Owed by tasks not done before 14 Nov.** No data exists. Each needs a ruling on what the
lock fixes instead, for example the rule with the number written when the task runs, or
the item marked out of scope with a dated waiver:

- H6 regret bound (T7.6)
- power target and n: α 0.025 and power 0.90 are proposed, but n needs T1.5's p
- primary cell n and admissibility n: 200, raised if 1,300 × p is larger (T1.5)
- admissibility Holm family (T1.8)
- cross-card dense task-score bound (the A100 dense control, audit B1)
- head-type criterion (T1.3)

**The data is banked, but no value has been extracted from it.** These can be done on CPU
before 14 Nov if the adviser asks for it. Otherwise each needs a ruling to lock its working
value:

- null-kernel floor multiple (T2.2's null pass)
- probe bounds (the P1 repeats' probe records)
- chain-check tolerance: working 10%, from T3.1 and T3.2's chain records
- residual spread tolerance: working 10%, from T3.2c and T3.2d's spreads under the idle-device residual
- pricing mismatch share: working 5%, from T3.2's pricing-gate records

**A proposal that no ruling has confirmed:**

- rerun budget: 1 extra round and 3 clean rounds
- bootstrap settings: level 0.95, `n_boot` 10,000, seed 0; a plan default from T4.3 read A, finding 5
- prompt-set hash and tokenizer version: transformers 4.46.0 on the image. The hash comes from T3.2d's prompts, which are not among the local pulls; reading them from the bucket needs a yes

**Not a T4.1 value, but the lock needs it:**

- the scored pairings for H0 to H6. P1 is refused in H1's, H3's and H6's lists, and H0's and H2's lists follow T2.3's classification
- `pairing_values` for P2 and P4. The draft proposes P1's working values: `reps` 10, `dense_rounds` 5, `sampler_interval_s` 0.1, and an eager group of all five arms. P4's groups are its blocks'

## At the lock commit

1. Fill the remaining values. Commit the file as `docs/lock_values.json`, with `plan_sha256` of the plan at that commit.
2. `scripts/tag_lock.py --dry-run` must report the lock complete.
3. Re-check the timed-path pin at the lock commit against the lock's item 37 (`84edb24d…` at `56e21eb` unless the timed path changes again). Also check that `analysis_commit` is the full sha of `14e4419`.
4. Re-check P1's ledger entry (`predictions/P1/ledger/banked.json`): its training-file sha256 must equal the lock file's `p1_training.file_sha256`, not the draft's (adviser's ruling 5, second set of 12 Nov 2026). The entry was checked against the draft (`de7f0f44…`).
5. Run `scripts/prelaunch_cpu_pass.sh p2` and `p4` on the lock commit.
6. Tag the commit; pushing the tag needs the researcher's yes. Then build P2's Flex card from the lock commit, with the headroom table.
