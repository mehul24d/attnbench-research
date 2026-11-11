# T4.3 read B: commit machinery, leakage, lock tooling

Commit `23f540e` (`origin/main` when the read started, 11 Nov 2026). Reproductions are in `repro_B/test_read_b.py`: 19 tests, each asserting what the plan requires, and all 19 fail on `23f540e`. They import the reviewed clone unmodified. Mutations were made in a separate copy. Full CPU suite on the clean clone: 6241 passed, 157 skipped, 0 failed (13 min). The clone's tree stayed clean.

## Findings, most severe first

### 1. P2 clocks the dense arm's whole prefill before the prediction is written, then stops (High; verified)
`scripts/t31_steps.py:940` (`step_dense_select`) calls `derive_session_floor` (`:960`) whenever the config's `resolution_floor_ms` is null. That function times `run_prefill()` on the dense arm, with a synchronise, at every length and batch, and saves the series. P2's config has the floor null. The lock cannot fill it, because `LOCK_KEYS` (`:122`) does not include it. `validate()` puts `dense_select` before `predict`.
- Plan: "the stamp is the start of the first clocked block of any kind, the dense control's clocked warm-up included" (Appendix A, 27 Oct 2026). Also "Predictions are written and hashed before the data they predict" (Section 6, rule 6), and the resolution floor is a T4.1 lock value.
- Scenario: P2's configuration with `reps=10`. The test counts 72 dense end-to-end prefills over 6 cells before `predict`. Then `floors.resolution_floor` raises `FloorsError: the floors come from P1 sessions only ... got 'P2'`. On the A100 this is P2 stopping at step 10 after its component steps. If the refusal were relaxed to get past it, the timed dense workload would precede the commit.
- No CPU dry run reaches this, because the dry `dense_select` returns early. The L4 rehearsal used P1's config, whose pairing is P1.

### 2. The analysis pools ground-truth rows that the runner refused (High; verified)
`scripts/t61_session.py:1074` writes `ground_truth.json` before `check_stamp` and the completeness check. `attnbench/analysis/block_drift.py:68` (`collect`) reads every non-discarded block's `ground_truth.json`. It never looks at the manifest's step statuses or at `late_check`.
- Plan: the 7 Nov 2026 block ruling ("checked before anything runs ... every step passed"); K9 ("the analysis refuses any prediction stamped later", Section 5.1).
- Scenario: block b2's harness returns a stamp from before the attempt opened (a clock step or a stale stamp), or an incomplete result. The step fails and the session stops. `collect([b1, b2])` still returns b2's 2 rows. A block preempted after ground truth but before `late_check`, and never resumed, also has no discard marker, so its rows are pooled too.

### 3. No pairing-level record: a later session can re-predict after earlier ground truth (High; verified)
`body_predict` (`t61_session.py:830`) refuses a second prediction only within one session directory. `late_check` and every analysis caller pass their own `ground_truth_started_at` and their own prediction path to `load_committed`. Nothing lists a pairing's predictions or takes the earliest ground truth across its sessions.
- Plan: Section 6, rule 6; Section 5.4 ("Each prediction for a pairing ... hashed and synced ... before the workload is timed").
- Scenario: P2 session a runs to the end. Session b starts 10 minutes later in its own directory, predicts, and runs its ground truth. b's `late_check` passes. b's prediction was written more than 500 s after a's ground truth began, so the analysis can score either file. The same path exists when an interrupted predict block is redone: the new b2 predicts after the old b2's ground truth began, and only which directories are passed to `collect` decides what is scored.

### 4. The lock checker accepts values the runner never reads, and the runner reads values the checker never checks (High; verified)
`attnbench/lock/checklist.py:117`: `_unset` rejects only None and empty containers. So `"TBD"`, `False`, `[None]` and `{"2048": None}` all count as fixed (4 tests). The session values come from `pairing_values.<P>` (`t31_steps.py:334`), which `check` never examines. With a lock that `check` calls complete:
- `max_rerun_rounds` missing from `pairing_values` gives a rerun budget of 0 (`:650`), while `values.rerun_budget` is 2.
- `pairing_values.P2.densities` differs from `values.ground_truth_density_set`, and the runner uses the first with no comparison.
- The throttle set, grid, warmup and `not_measured` come from `configs/t61/p2_session.json`, never from the lock. Both the throttle set and the grid are T4.1 items.
- Plan: Gate B, "All [lock] values fixed" (Section 7). Q4 of the brief.

### 5. A committed learned prediction and its training set cannot be checked (Medium-High; verified)
`load_committed` recomputes the computed predictions and the fitted sums from their recorded coefficients. The `learned` values are only required to exist (`commit.py:1010`), and the recorded `config` is never compared with `LOCKED_CONFIG`.
- The record names training pairings as labels (`trained_on`) and holds no hash of the training rows or the training file.
- `training_rows` (`t61_session.py:788`) re-reads the training file at `predict` with no lock check: any file that matches its own sha256 is accepted. The lock's sha256 is compared only at step 1.
- `--banked-at` times are typed on the command line and checked against nothing.
- Plan: Section 5.4, "each prediction file records it [its training set]"; "features and hyperparameters fixed at lock".
- Scenario: a record whose `learned` values are all 42.0 and whose config is `log_seconds`/`num_leaves 63` loads cleanly. Within the code path, rows still come only from the P1 file, so no P2 or P4 leak path exists today. The gap is that the file cannot show this.

### 6. A block resumed after a fix commit keeps its first commit (Medium; verified)
`manifest["code_commit"]` is set once, at block start (`t61_session.py:1717`). `--from STEP` after a committed fix (the documented recovery) runs later steps at the new HEAD without refusing or recording it. `collect` then reports one commit.
- Plan: "the same spec and code commit" (Appendix A, 7 Nov 2026); rule 3.
- Scenario: b2's ground truth fails at X. A fix is committed as Y, and `--from ground_truth` passes at Y. `collect` reports commit X.

### 7. The timed-path identity misses parts of the timed path (Medium; verified by mutation)
`scripts/timed_path.py:112` follows only top-level functions referenced by bare name, and hashes no data files. The hash stayed `8716d7db…` under three local commits:
- meanpool removed from `BATCH1_ONLY` (which arms run at batch 4);
- `TraceExport.__call__` edited;
- `configs/t61/p2_session.json` warmup set to 1 and the throttle set to one reason.

`lock.checklist` (the lock fill) and the hold guard (`session_order.ready_to_clock`) are also outside the closure.
- Plan: the timed-path identity for P2 and P4 (Appendix A, 10 Nov 2026).

### 8. The block spec file and the lock file are not tied together (Medium; verified)
`block_run` (`t61_session.py:1742`) takes the spec from `--lock FILE`. It never runs `checklist.check` on that file and never compares it with `docs/lock_values.json`, which every other reader uses. A real-mode block runs from a file that holds nothing but a block spec.

Separately, `p2_staged_inputs` (`:1241`) reads `lock["p1_training"]`, but the draft package holds `banked_files.p1_training`. A lock written from the draft is refused at step 1. This fails closed, but it invites a second, unchecked copy of the hash.

### 9. Block identity compares three fields and passes on missing ones (Medium-Low; verified)
`BLOCK_IDENTITY` (`:139`) is the instance type, GPU name and CPU model. These match the 7 Nov text, but `None == None` passes, so a row without `cpu_model` matches anything. The boot image, driver and bootstrap hash are recorded or recordable and never compared. The zone is not recorded at all.
- Scenario: blocks on image v6 with driver 550 and on v7 with driver 560 are pooled.

### 10. The real-time gap can be about 2 s inside the allowed skew (Low; verified)
`commit.py:1104` requires `start > synced + 60` across two clocks, using one 60 s for both the gap and the skew.
- Plan: "the workload starts more than 60 s after the sync ... the analysis allows 60 s of clock skew" (Section 5.1).
- Scenario: the instance clock is 59 s fast, the store records the object at T, and ground truth starts at instance time T+61, which is real time T+2. The file is accepted. The runner's 90 s hold keeps the real gap at 31 s or more, but the analysis does not enforce the gap.

### 11. P4's prediction is not built, and P4 learns this late (Low; verified)
`training_rows` raises `NotBuilt` for P2's rows (`:780`), and P4 has no staged check. P4 would fail at `predict` after its component steps. This fails closed; it is a cost risk.

## Unconfirmed (read, not reproduced)
- P1's masks (`banked_masks`) are checked against their own hashes only. The lock holds no hash or cell list for them, and P2's mask-gate prompts come from that staged bank.
- `gsutil cp -n` is check-then-write, not a generation-match precondition, so two concurrent writers could race. The read-back comparison should catch a mismatch. Not testable without the store.
- The prediction is protected by the store's clock, but the ground-truth start it is compared with comes from instance-clock files (`ground_truth_opened.json`, `stamps.json`) that carry no hash. An edited stamp file would move the start. This is tampering, outside the brief's "without the code refusing" in spirit.

## Known open items, confirmed
Every lock value in the draft is null (37 of 37). Nothing at session start reads `timed_path` (no reference in `t61_session.py`, `t64_session.py` or `t31_steps.py`).

## Not read
- `ground_truth.py` internals beyond the stamp.
- `dispatcher.py`'s commit path; P2 commits no dispatch section today.
- `t64_blocks.py`, `t61_card.py`, `session_exit.sh`, the `gcp_*` scripts, and the AWS tooling.
- The secondary-pass files beyond the notes below.

## Mutations of existing guards (copy `mut_B`, each restored after)
All 22 were caught, each by the test named. In every case the tests failed when the guard was removed.

Guards removed (each caught): the late check, its skew, the write-after-sync check, own-pairing-banked-first, the sync read-back, the locked learned config, target-in-banked, `--from` before a passed predict, the second receipt, stamp before open, the 60 s gap in `check_stamp`, import of a discarded block, the import commit, block identity, the staged training sha, `ready_to_clock` in the ground-truth process, `collect`'s discard skip, `interrupted()`, lock-fill completeness, `_unset`, the training file's self-sha and the recomputed predictions. The catching tests are in `tests/test_predict_commit.py`, `test_audit_fixes_commit.py`, `test_predict_baselines.py`, `test_t61_session.py`, `test_t61_blocks.py`, `test_t31_session.py`, `test_lock_fill.py`, `test_lock_checklist.py` and `test_p1_training.py`; the script is `mutate_B.py`.

Timed-path mutations: 3 local commits, the hash unchanged each time (Finding 7).

## Against `docs/t43_first_read.md` (read after the above)
**Agree.**
- Its item 9: the training file's integrity rests on the lock's sha256. I add that this sha256 is checked only at step 1, never at `predict`, and is not in the record (Finding 5).
- Its item 8: P2 cannot commit without the learned configuration. This still holds: `LOCKED_CONFIG` is None.
- Its fixes 6 and 7 (`row_key`, `median_ms`): they hold; the dry block runs pass.

**Disagree.**
- It says the block path was read and finds only keying defects. `collect` still pools rows from refused or unfinished ground truth (Finding 2), and it records one commit for a resumed block (Finding 6).
- It treats the lock list as covered by `checklist.py`. The checker does not type-check, and the runner reads `pairing_values` and the repo config, which the checker never sees (Finding 4).

**What it missed.**
- Findings 1, 3, 5, 7 and 8.
- That its own item 2's per-length floor is not representable by the scalar-floor APIs (secondary note 1).

## Secondary, lightly checked (reader A's scope, about 30 minutes)
1. The resolution floor is per context length and batch since 8 Nov 2026 (T4.1), but `scoring.score_cells`, `dispatcher.dispatcher`, `gate_a.sanity` and `figures/paper_tables.py:174,223` each take one scalar floor. `paper_tables` groups cells across lengths, so one floor is applied to every length.
2. `block_drift.drift` defaults to `WORKING_BOUND` (6.77%). A caller that omits the bound gets the working value silently, not the lock's (reader A's question 6).
3. Holm (`lock/holm.py`): step-down at α/(m−i) over p-values from bisection, family checked equal to the declared set. I saw nothing wrong.
4. `scoring.predicted_profitable` uses no floor on the predicted side while the dispatcher uses one. This is documented as intended. Reader A should confirm that H1's sign agreement wants the unfloored predicted sign.
