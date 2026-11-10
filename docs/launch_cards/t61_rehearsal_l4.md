# L4 rehearsal card: the P2 runner before an A100 session is spent

Prepared 10 Nov 2026 from the adviser's ruling of 10 Nov 2026. **Launchable
only on the researcher's yes, in their own words, naming the instances, the
zone, the commit and the cap. Nothing here runs on its own. The first yes (commit `2053377`, ₹403) is spent: attempt 1 used it. The second (commit `10e1fe1`, ₹296) is spent: attempt 2 used it. The third (commit `59cc607`, ₹254, ₹422 in all) is spent: launch 3 used it.**

**Exploratory. No row of this rehearsal enters training, the floors, Gate A,
the lock or any claim.** Every session it writes says so in `manifest.json`
(`exploratory`) and in `EXPLORATORY.md`. Its prediction goes to
`predictions/P2-R/`, not P2's prefix. The pairing is `P2-R`.

| | |
|---|---|
| Commit | **`40b6df5`** (`40b6df5e2be77e486f9db9ce454e55024d74fb14`, branch `timed-path-pin-2026-11-10` off `main` `ac3f563`). It is `main` plus four things: the timed-path identity pinned (`8316263`, docs and a lock test), the prediction record named for the session's pairing (`3057240`, launch 3's stop), the remote start commands backgrounded inside braces (`e7162cc`), launch 3's records (`d42b0b0`) and the suite counts. Deployed from a fresh clone, detached, clean tree. A later commit updates this card to name it |
| Instances | A: `attnbench-l4-reh-a-<YYYYMMDD-HHMM>`; then B: `attnbench-l4-reh-b-<YYYYMMDD-HHMM>`. Both `g2-standard-8` with one L4, on-demand (STANDARD) |
| Zone | `asia-northeast1-a` |
| Image | `attnbench-env-v6-20260917` |
| Rate | ₹108.64/h (planning rate, all-in) |
| **Cap** | **₹254 for launch 4** (A ₹163 + B ₹91), **₹487 with attempts 1 and 2 and launch 3 (₹233)**: over the ₹422 the researcher allowed on 10 Nov 2026, so launch 4 needs a cap decision as well as its yes. Without (v): ₹231 and ₹464 |

A runs first. B is created only after A is deleted, because the GPU quota is 1.

## Attempt 1 (10 Nov 2026, commit `2053377`) and the relaunch

Instance A ran on the researcher's yes of 10 Nov. It stopped itself at step 11 of 15 (`predict`) after 55 min (₹101), was torn down from the bucket, and nothing is left billing. **Instance B was not started.**

- **Cause:** `lightgbm` is not on image v6 and no launch path installed it, so the learned predictor could not fit (`CostModelError`). P2's and P4's A100 sessions would have stopped at the same step, after their setup and ten steps. Fixed in `5c9175a`: the bootstrap installs `lightgbm` at `learned.LIGHTGBM_VERSION` (4.6.0) and checks the import; `real_preflight` refuses without it, so the stop now comes in step 2.
- **Passed on A:** `prereqs`, `preflight` (T2.26's five variants included), `link`, `launch_queue`, `probe`, `mask_gate` (4 cells, 112 of 112 layers identical, largest density difference 0.0), `host_components`, `device_components`, `dense_select`, `host_speed_mid`, and T2.17's `assert_gate` (3 prompts, 84 layer records).
- **Not run:** ground truth, trace sizes, the second block, the interruption and the compiled stage: scopes (i), (iii) and (v) are still open.
- **The suite record (T2.26 in the bootstrap) returned rc 1:** 27 failed, 6,123 passed, 215 skipped in 9:41. All 27 assume a CPU-only host (`a CUDA device is present; the CPU boundary is for a CPU`, CPU dry runs that reach an installed GPU kernel in float32, a fake capture that is soaked because a CUDA device exists); none is a GPU-path defect (`docs/gpu_suite_failures_20261110.md`). The record is banked (phase `s9_suite`). The relaunch does not pass the flag, and neither do P2's or P4's cards.
- **Measured minutes (A, steps 1 to 10):** `link` 2.8, `probe` 5.0, `mask_gate` 0.2, `host_components` 7.7, `device_components` 5.1, `dense_select` 0.4, the rest under 1; 21.4 of work in 26.9 wall. The first card's table used T3.2d's per-cell scaling and overstated `host_components` (23.2) and `device_components` (14.4); "Time and cost" below is rebuilt from these.

**Attempt 1's relaunch (attempt 2) ran at `10e1fe1`, cap ₹296; see the next section.**
- **The 27 suite failures, classified (no new GPU time):** all 27 are tests that assume a CPU-only host; none is a defect in the GPU path (`docs/gpu_suite_failures_20261110.md`). The S8 run of the timed-region test, separate from the 27, found one thing: `block_sparse` converts its mask on every call (about 3 µs on the Mac, views only). It is allowlisted, not hoisted (adviser's ruling 2 of 10 Nov 2026, `a115619`): the same bytes are in T3.2c's and T3.2d's timed path, `4e78a962…`.
- **New in `10e1fe1`:** the preflight scans the session's scripts for every third-party import and refuses, naming the package, if one does not import (`attnbench/session_imports.py`). The lightgbm version check stays beside it.

## Attempt 2 (10 Nov 2026, commit `10e1fe1`) and launch 3

Instance A `attnbench-l4-reh-a-20261110-0446` ran on the researcher's second yes after two stockouts in `asia-northeast1-a` (the same zone, created on the third try). It stopped itself at step 11 again after 37 min (₹67), was torn down from the bucket, and nothing is left billing. **B was not started.**

- **Cause:** `predict` failed with `CostModelError: P1/dense has no model shape`. Nothing on the real path set `Cell.model`, neither the P1 training rows nor the session's cells; only test fixtures and the dry run's fake record did. Fixed in `89e1f6f`: `intensity.shape_for` gives both the primary model's pinned shape. A test reproduces the failure, and a trained prediction is built over the session's cells.
- **Passed on A:** setup without the suite record (5.3 min), steps 1 to 10, the import check (no false refusal; lightgbm installed and verified) and T2.17's `assert_gate`. The compiled stage started after the failed session and ended in 1 s, so it measured nothing.
- **Still not run:** ground truth and trace sizes (i), the second block and the interruption (iii), the compiled groups (v), and steps 12 to 15 have never run for real.

**Launch 3 (ran 10 Nov 2026; see the next section):** commit **`59cc607`**; instance A `attnbench-l4-reh-a-<YYYYMMDD-HHMM>`, then B; same zone, image and rate. The caps are built from attempt 2's measured durations, each plan × 1.3 (see "Time and cost"):
- A plans 62 min, `GCP_HALT_MINUTES=80`, `GCP_MAX_RUN=1h30m`, cap **₹163**.
- B plans 31 min, halt 32, delete `50m`, cap **₹91**.

Launch 3's cap is **₹254**. With the ₹168 already spent, the rehearsal totals at most **₹422**, expected ₹338. **This departs from the adviser's "cut (v) first if over about ₹400"**: the researcher chose on 10 Nov 2026 to keep (v) and raise the total to ₹422. Without (v), the total is ₹399. Launch 3 does not pass `--suite-record`.

## Launch 3 (10 Nov 2026, commit `59cc607`) and launch 4

Instance A `attnbench-l4-reh-a-20261110-0653` ran on the researcher's third yes. It was created at 01:47:34Z after ten stockouts in `asia-northeast1-a` (the same zone, retried from 01:23Z). It stopped itself at step 11 again after 36 min (₹65), was torn down from the bucket (112 of 112 logs and manifests), and nothing is left billing. **B was not started:** it imports b1's ground truth, which never ran.

- **Cause:** `predict` failed with `PredictionCommitError: P1 is listed as banked before its own prediction`. `t31_steps.build_prediction` named the record after its components config's `pairing`. The rehearsal's config is P1's (`configs/t61/rehearsal_l4.json`), while the session is `P2-R` and is trained on banked P1. Fixed in `3057240`: `real_record` passes the session's pairing. Two tests reproduce the stop, and each was watched to fail with its half of the fix undone. P2's own config says `P2`, so P2 would not have stopped here.
- **The start ssh hung for 27 min** (the session ran; the ssh was killed and the watcher started at 02:18Z). `cd X && nohup cmd ... &` backgrounds the whole list, and that subshell holds ssh's stdout until the session ends. The `&` now sits inside braces in both start commands below and in the interrupt script's boot 2 command (`e7162cc`). The stub test now waits for stdout EOF as sshd does, and fails at 30 s with the old form.
- **Passed on A:** setup in 3.9 min, steps 1 to 10 in 26.5 min (attempt 2: 27.3; T2.26's five variants passed), T2.17's `assert_gate`. The compiled stage refused (rc 2: no b1 prediction).
- **Still not run:** steps 12 to 15, ground truth and trace sizes (i), the second block and the interruption (iii), and the compiled groups (v).

**Launch 4 (needs its own yes and a cap decision):** commit **`40b6df5`**, same instances, zone, image, rate, halts and deletes as launch 3. The caps are unchanged: launch 3's measured minutes are within attempt 2's. A is ₹163 and B ₹91, so launch 4 is ₹254 and the total ₹487, or ₹231 and ₹464 without (v). The 14 Nov rule still stands, and launch 4 is the compiled path's last chance before it.

## What it rehearses (adviser's scope, by priority)

| Scope | Where | What is measured or checked |
|---|---|---|
| (i) the real ground-truth harness on a reduced grid | A (block b1, 2048 and the anchor), B (block b2, 8192) | P2's `ground_truth` step (`t61_session.real_harness`, T3.1's eager group) on the L4. `trace_sizes.json` gives each Kineto trace's bytes, by arm and cell |
| (ii) P2's mask-gate path against P1's saved masks | A, `mask_gate` | P2's recorder, `mask_gate.compare_banks` and `mask_gate.gate` at **±0.01 density** (`DENSITY_FLAG_TOLERANCE`). The inputs are T3.2d's four XAttention cells (two arms, 2048 and 8192) against the same prompts recorded again on this card's PyTorch path. Same hardware and same path, so the masks should agree; a failure is a code defect. Block agreement is descriptive |
| (iii) the block-aware runner, two blocks | A, then B | The block spec (`configs/t61/rehearsal_lock.json`), the common anchor (dense, 8192, batch 1) measured in both blocks, b2 importing b1 (`--import-block`), the between-block drift check (`block_drift`, working bound 6.77%), the halt and self-halt, and a **simulated interruption**: B is reset after b2's ground truth opens; the resume must be refused with `INTERRUPTED_DISCARD.md`; b2 then runs again in a new session, and the drift check reads b1, the discarded b2 (skipped) and the new b2 |
| (iv) T2.17's assert gate and T2.26's GPU suite | A | `t227_steps.py assert_gate` at 2048 over 3 prompts. T2.26's five GPU variants in every preflight: a skip, a failure or a count other than five is a problem. The whole-suite record is not run (ruling of 10 Nov 2026; attempt 1 ran it and 27 CPU-assuming tests failed, `docs/gpu_suite_failures_20261110.md`) |
| (v) compiled ground truth on a reduced grid | A, after b1 (`ATTNBENCH_REH_COMPILED=1`) | `compiled_default` and `compiled_capture` (mean-pool, XAttention) at 2048 through P2's harness. The harness's own recompile gate and `compiled_checks` (the Kineto twin and mask-dependence checks) run with the recompile limit raised to 256 |

**Grid and protocol.** The grid is 2048 and 8192 at batch 1, five arms, with oracle and mean-pool at density 0.25 only: the working 0.5 is dropped to fit (v) under ₹400. That gives 10 cells, plus 5 compiled cells at 2048. The measurement protocol is the working values', unchanged: 10 reps, 3 warm-up, 5 dense rounds, the sampler, the hold and the strict stamp (`tests/test_t61_rehearsal.py` checks reps against P1's config). T2.17's gate uses 3 prompts, not 5: fewer reps of the same check.

**Recompile handling, verified on CPU first (10 Nov 2026).** 91 tests pass across `test_compile_guard`, `test_compiled_checks`, `test_compiled_variants` and `test_shakedown6_cpu`. Two mutations were each watched to fail:
- making `CompiledBackend.compiled_region_targets` return `()` kills 8 tests, among them `test_no_probe_wrapper_is_traced_by_dynamo`, which recompiles per call under the old probe;
- setting `RECOMPILE_LIMIT = 8` kills `test_configure_raises_the_recompile_ceiling`.

## What differs from P2, and why it is not the code under test

- **The card is the L4**, so the provenance checks expect `g2-standard-8`, on-demand.
- **Prereqs:** P2's unmet plan tasks are recorded (`p2_would_refuse`), not refused on. The rehearsal is what T2.17 and T2.26 wait for.
- **No Triton path on the L4.** XAttention takes Triton only on a device whose name contains "100". So P2's own verdict, Triton against PyTorch on one card (`p2_mask_gate`), **is not rehearsed**. The rehearsal runs the same recorder, comparison and gate on one path against P1's bank. P1's bank is this card's path, and the gate refuses one code path compared with itself, so the new recording is labelled `xattn:torch@rehearsal`. Only the label changes, never a mask.
- **The learned predictor's configuration** is a lock value and unset, so `predict` fits the working one: configuration 4 of `docs/learned_cv_p1_20261109.json`. The step's findings say so.
- **The rerun after the interruption runs on the same instance after its reset.** Real P2 or P4 starts a new instance; the runner's discard rule is the same either way.

## Defects found and gaps closed while building it

1. **`block_config` let a block visit a length without its dense control.** T3.1 times the dense control at every length and batch it visits. So such a block would have clocked control rows and then failed `tag_rows` after the time was spent. It is now refused before anything is clocked, and the grid is cut to the block's own lengths and batches.
2. **T2.26 was not called from the preflight.** `real_preflight` now runs the five GPU variants and reads their JUnit record.
3. **Two scripts failed to parse under bash 3.2** (an apostrophe inside `${VAR:?message}`). A parse test now covers all four rehearsal scripts.
4. **T2.26's suite record was in no launch path.** `gcp_bootstrap_instance.sh --suite-record` now runs it on the instance after the installs and before the tree check. A suite that does not pass, or reaches the bound, stops the bootstrap. The flag is off by default, so no other card's setup changes.

Items 1 to 3 were fixed in `dda1c39` and item 4 in `2053377`.

## Time and cost (launches 3 and 4: attempt 2's measured durations × 1.3, ₹108.64/h)

Cap = rate × (planned minutes × 1.3, rounded down, + 10) ÷ 60, rounded up. The in-guest halt is at 1.3 × the plan, and the hard delete comes 10 minutes after it. Attempt 2 measured the setup, steps 1 to 10, the assert gate and the exit (the phase logs, `session_cost.txt`):

| Measured in attempt 2 (A) | Minutes | From |
|---|---|---|
| Boot to the session start, without the suite record | 5.3 | Created 23:25:54Z, `prereqs` began 23:31:09Z (deploy, preflight, write gate, bootstrap, no SSH retry). Attempt 1's 14.0 included an SSH retry and is not used; a retry is covered by the margin |
| Steps 1 to 10, wall time, to the start of `predict` | 27.3 | 23:31:09Z to 23:58:26Z. Work: `link` 2.8, `probe` 5.1, `host_components` 7.8, `device_components` 5.2, the rest under 1 |
| The sync between steps | 0.3 to 0.7 | 17 s while the session is small; 42 to 44 s once the components are written |
| T2.17 `assert_gate`, with the sync before it | 0.9 | 23:58:29Z to 23:59:21Z |
| Compiled stage end to the halt | 2.6 | 00:00:05Z to 00:02:41Z (exit, sync, halt) |

Not measured on this card, so estimates: `predict` (it failed after 3 s), the hold (the runner's 90 s), `ground_truth` (T3.2d step 9: 28 min over 34 cells, scaled to 5 cells and the anchor), `late_check`, `trace_checks`, `host_speed_end`, the compiled stage (never timed clean on a GPU), fetching b1 on B and the reset. Syncs after ground truth carry its traces, whose size is (i) and unknown.

**Instance A**

| | Minutes | Basis |
|---|---|---|
| Setup | 5.3 | measured |
| Steps 1 to 10, with the sync, to `predict` | 27.3 | measured |
| `predict` | 2 | estimate |
| The hold | 1.5 | the runner's 90 s |
| `ground_truth`, 5 cells + the anchor | 4.9 | T3.2d step 9, 28 / 34 per cell × 6 |
| `late_check`, `trace_checks`, `host_speed_end` | 4.5 | estimates: 1 + 3 + 0.5 |
| Sync between steps 11 to 15 | 2.9 | 4 × 0.72 (measured, the later syncs) |
| T2.17 `assert_gate`, with its sync | 0.9 | measured |
| (v) the compiled groups, 2 × 3 cells at 2048, with the sync | 10.5 | **estimate; never timed clean on a GPU** |
| Exit, sync, halt | 2.6 | measured |
| **Planned** | **62** | 62.4; without (v): 52 (51.9) |

**Instance B**

| | Minutes | Basis |
|---|---|---|
| Boot 1: setup 5.3, fetch b1 1, `prereqs` and `preflight` 1, the hold 1.5, ground truth opening 1 | 9.8 | measured setup and preflight; estimates |
| Reset, SSH back | 2 | estimate |
| Boot 2: the refused resume 1; then b2 again: `prereqs` and `preflight` 1, hold 1.5, ground truth 4.9, `late_check`, `trace_checks`, `host_speed_end` 4.5, sync 3.6 (5 × 0.72), drift 0.5, exit 2.6 | 19.6 | as above |
| **Planned** | **31** | 31.4. If the reset never comes, boot 1 runs b2 to its end: 5.3 + 1 + 18.6 = 25 min, the longest single boot |

| Limit | A | B |
|---|---|---|
| In-guest halt, from **each** boot (`GCP_HALT_MINUTES`) | 80 (1.3 × 62) | 32 (1.3 × 25, the longest boot) |
| Hard delete (`GCP_MAX_RUN`, DELETE) | 90 min = `1h30m` (halt + 10) | 50 min = `50m` (1.3 × 31 rounded down, 40, + 10) |
| Cap, at the later bound | **₹163** (90 min) | **₹91** (50 min) |

**The interrupt window is now 10 minutes, not 20.** Boot 1 reaches ground truth about 4.5 min after the session starts. With `--within 10`, the reset comes at most about 16 min after B boots. If the reset restarts the run timer, boot 2 halts by 16 + 32 = 48 min, which is inside the 50-minute delete. With 20, it could have run to 58 min, past the cap.

**Launch 3's cap is ₹254** (A ₹163 + B ₹91). Expected ₹170: A 62 min ₹113, B 31 min ₹57. With attempts 1 and 2 (₹101 + ₹67 = ₹168, `docs/spend_ledger.md`): **₹422**, expected ₹338. That is over the original ₹403, on the researcher's choice of 10 Nov 2026 (keep (v), raise the total to ₹422). Launch 3 spent ₹65 of it. **Launch 4's cap is the same ₹254**, which brings the total to **₹487** (₹464 without (v)): over ₹422, so it is the researcher's decision.

Without (v), with `ATTNBENCH_REH_COMPILED` unset: A plans 52 min, halts at 67 and is deleted at 77 (`1h17m`, ₹140). That gives launch 3 a cap of **₹231** and a total of **₹399**. The protocol is not touched either way.

After attempt 2, ₹4,707 of ₹14,000 is spent (`docs/spend_ledger.md`, counted from its rows); launch 3's cap leaves at least **₹9,039**.

## Named bucket reads (part of the launch yes)

- **On A:** T3.2d's eight XAttention mask files:
  - from `attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components/`;
  - for each of `xattention` and `xattention_noassert` at `_dcal_L2048_b1` and `_dcal_L8192_b1`;
  - `recording/masks.{npz,json}` for each.

  Also the P1 training file, `banked/p1_training/p1_training_t32dc_20261109.json.gz`. Its sha256 is checked against `de7f0f44…`, and the chain stops on a mismatch.
- **On B:** A's b1 session, `<A>/results/t61r_l4_<UTC>_b1/`.
- **On the Mac, afterwards:** each instance's logs and manifests, through `gcp_teardown_session.sh`'s bucket fallback.

## The commands

On the Mac, in a fresh clone at `40b6df5`:

```bash
git clone <repository> reh && cd reh && git checkout --detach 40b6df5
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917 GCP_RATE_INR_HR=108.64
ZONE=asia-northeast1-a
```

**Instance A**

```bash
export GCP_HALT_MINUTES=80 GCP_MAX_RUN=1h30m     # without (v): 67 and 1h17m, and leave ATTNBENCH_REH_COMPILED unset
A=attnbench-l4-reh-a-$(date +%Y%m%d-%H%M)
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$A"     # type 'launch'
bash scripts/gcp_deploy_source.sh "$A" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$A" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$A" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$A" "$ZONE"      # no --suite-record (ruling of 10 Nov 2026)
# if it stops on an SSH timeout or a failed scp, rerun it: it is idempotent. That is pre-decided.
B1=results/t61r_l4_$(date -u +%Y%m%dT%H%MZ)_b1
gcloud compute ssh "$A" --zone "$ZONE" --command "cd ~/attnbench_scaffold && { \
  ATTNBENCH_REH_COMPILED=1 nohup bash scripts/session_exit.sh --log ~/reh_a.log -- \
  bash scripts/t61_rehearsal_a.sh $B1 >/dev/null 2>&1 < /dev/null & }" < /dev/null   # the & inside the braces, or ssh does not return
bash scripts/t61_rehearsal_watch.sh "$A" "$ZONE" > ~/reh_watch_a.log 2>&1   # until TERMINATED or gone
bash scripts/gcp_session_elapsed.sh "$A" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$A" "$ZONE" "$B1"
bash scripts/gcp_cleanup_check.sh
```

**Instance B** (after A is deleted)

```bash
export GCP_HALT_MINUTES=32 GCP_MAX_RUN=50m
B=attnbench-l4-reh-b-$(date +%Y%m%d-%H%M)
# launch, deploy, preflight, write gate, bootstrap: as for A, with "$B"
X=results/t61r_l4_$(date -u +%Y%m%dT%H%MZ)_b2_interrupted; Y=${X%_interrupted}
gcloud compute ssh "$B" --zone "$ZONE" --command "cd ~/attnbench_scaffold && { \
  nohup bash scripts/session_exit.sh --log ~/reh_b1.log -- \
  bash scripts/t61_rehearsal_b.sh boot1 $A $B1 $X >/dev/null 2>&1 < /dev/null & }" < /dev/null
bash scripts/t61_rehearsal_watch.sh "$B" "$ZONE" > ~/reh_watch_b.log 2>&1 &
bash scripts/t61_rehearsal_interrupt.sh "$B" "$ZONE" "$A" "$B1" "$X" "$Y" --within 10
wait                                                     # the watcher: TERMINATED or gone
bash scripts/gcp_session_elapsed.sh "$B" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$B" "$ZONE" "$Y"
bash scripts/gcp_cleanup_check.sh
```

No `ssh` in the interrupt script reads stdin (`225bed9`, the ruling of 10 Nov 2026): every call gets `/dev/null`, and so does the boot2 command. `test_the_interrupt_starts_boot2_and_no_ssh_in_it_reads_stdin` runs the script against a `gcloud` stub that reads its stdin. The stdin is never closed, and `INTERRUPT_DONE` prints; with the local redirect removed, the script hangs. If ground truth has not opened within 10 minutes, the interrupt script resets nothing and prints `INTERRUPT_NOT_REHEARSED`. Boot 1's b2 then runs on and halts itself, and the report says the interruption was not rehearsed.

**While it bills.** No blocking question is asked. The watcher is a poll loop that runs until the instance is TERMINATED or gone. It outlives any question and changes nothing; the self-halt is what ends each session.

**Afterwards.** Pull the logs and manifests from the bucket, add the ledger row (`docs/spend_ledger.md`, marked exploratory and not T6.1 data), and report in under 300 words.

## What it reports

- Each step's status per block.
- The trace bytes per arm and cell.
- The mask-gate verdict, the largest density difference and the minimum block agreement.
- Whether the interrupted b2 was refused and discarded, and the anchor's drift between b1 and b2.
- T2.17's gate.
- T2.26's variants and the suite record's rc.
- The compiled groups' completeness, recompiles after warm-up, and the twin and mask-dependence checks.
- Minutes per step against this card's table, which is the input for rebuilding P2's card with measured minutes.

## Limitations and the 14 Nov rule

- P2's two-path mask verdict is not rehearsed on the L4 (no Triton). It first runs on P2's A100.
- The learned configuration is the working one. P2's prerequisites are recorded, not enforced.
- **No suite record.** The adviser's ruling of 10 Nov 2026 reverses the 10 Nov ruling h: P2's and P4's cards leave `--suite-record` off and carry no minutes for it (their setup rows are unchanged, 0.5 h and 0.6 h; each card says so). T2.26's five GPU variants still run in every preflight. The step is still in the bootstrap as an opt-in, and `gpu_suite_record.sh` needs a GPU-safe selection before any card uses it.
- The compiled minutes are estimates.
- **The 14 Nov rule:** the compiled path is clean when `compiled/result.json` reads passed on the relaunch, both groups complete, no recompile after warm-up, and the Kineto twin and mask-dependence checks passed. **If it is not clean by 14 Nov 2026, the compiled group is dropped** (the cut order of Appendix A, 9 Nov 2026, item 10) **and the limitation is written**: the compiled variants' ground truth was not measured, and no claim about compiled arms rests on this study. P2's and P4's cards are built without the compiled groups today and stay so. This rehearsal is the only GPU run of the compiled path before then. Attempts 1 and 2 never reached it, so launch 3 is the one attempt left: results by Thu 13 Nov leave one day.
