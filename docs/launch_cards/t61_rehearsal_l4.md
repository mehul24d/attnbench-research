# L4 rehearsal card: the P2 runner before an A100 session is spent

Prepared 10 Nov 2026 from the adviser's ruling of 10 Nov 2026. **Launchable
only on the researcher's yes, in their own words, naming the instances, the
zone, the commit and the cap. Nothing here runs on its own. The first yes (commit `2053377`, ₹403) is spent: attempt 1 used it.**

**Exploratory. No row of this rehearsal enters training, the floors, Gate A,
the lock or any claim.** Every session it writes says so in `manifest.json`
(`exploratory`) and in `EXPLORATORY.md`. Its prediction goes to
`predictions/P2-R/`, not P2's prefix. The pairing is `P2-R`.

| | |
|---|---|
| Commit | **`10e1fe1`** (`10e1fe1b728f34b82078e3953a4d64cf2ef140ae`, branch `l4-rehearsal-lightgbm-2026-11-10`: attempt 1's `2053377` plus the lightgbm fix, the import preflight and the P2 and P4 cards' suite-record note), from a fresh clone, detached, clean tree. A later commit holds this card and the ledger; it is not deployed |
| Instances | A: `attnbench-l4-reh-a-<YYYYMMDD-HHMM>`; then B: `attnbench-l4-reh-b-<YYYYMMDD-HHMM>`. Both `g2-standard-8` with one L4, on-demand (STANDARD) |
| Zone | `asia-northeast1-a` |
| Image | `attnbench-env-v6-20260917` |
| Rate | ₹108.64/h (planning rate, all-in) |
| **Cap** | **₹296 for the relaunch** (A ₹183 + B ₹113), **₹397 with attempt 1's ₹101**, inside the ₹403; expected ₹299 in all. Without scope (v): ₹273 and ₹374 |

A runs first. B is created only after A is deleted, because the GPU quota is 1.

## Attempt 1 (10 Nov 2026, commit `2053377`) and the relaunch

Instance A ran on the researcher's yes of 10 Nov. It stopped itself at step 11 of 15 (`predict`) after 55 min (₹101), was torn down from the bucket, and nothing is left billing. **Instance B was not started.**

- **Cause:** `lightgbm` is not on image v6 and no launch path installed it, so the learned predictor could not fit (`CostModelError`). P2's and P4's A100 sessions would have stopped at the same step, after their setup and ten steps. Fixed in `5c9175a`: the bootstrap installs `lightgbm` at `learned.LIGHTGBM_VERSION` (4.6.0) and checks the import; `real_preflight` refuses without it, so the stop now comes in step 2.
- **Passed on A:** `prereqs`, `preflight` (T2.26's five variants included), `link`, `launch_queue`, `probe`, `mask_gate` (4 cells, 112 of 112 layers identical, largest density difference 0.0), `host_components`, `device_components`, `dense_select`, `host_speed_mid`, and T2.17's `assert_gate` (3 prompts, 84 layer records).
- **Not run:** ground truth, trace sizes, the second block, the interruption and the compiled stage: scopes (i), (iii) and (v) are still open.
- **The suite record (T2.26 in the bootstrap) returned rc 1:** 27 failed, 6,123 passed, 215 skipped in 9:41. All 27 assume a CPU-only host (`a CUDA device is present; the CPU boundary is for a CPU`, CPU dry runs that reach an installed GPU kernel in float32, a fake capture that is soaked because a CUDA device exists); none is a GPU-path defect (`docs/gpu_suite_failures_20261110.md`). The record is banked (phase `s9_suite`). The relaunch does not pass the flag, and neither do P2's or P4's cards.
- **Measured minutes (A, steps 1 to 10):** `link` 2.8, `probe` 5.0, `mask_gate` 0.2, `host_components` 7.7, `device_components` 5.1, `dense_select` 0.4, the rest under 1; 21.4 of work in 26.9 wall. The first card's table used T3.2d's per-cell scaling and overstated `host_components` (23.2) and `device_components` (14.4); "Time and cost" below is rebuilt from these.

**Relaunch (needs its own yes):** commit **`10e1fe1`**; instance A `attnbench-l4-reh-a-<YYYYMMDD-HHMM>` again, then B; same zone, image and rate. The caps below are built from attempt 1's measured durations, each plan × 1.3 (see "Time and cost"): A plans 70 min, `GCP_HALT_MINUTES=91`, `GCP_MAX_RUN=1h41m`, cap **₹183**; B plans 39 min, halt 41, delete `60m`, cap **₹113**. Relaunch **₹296**; with attempt 1's ₹101 the rehearsal totals at most **₹397**, inside the original ₹403 (₹6 to spare). Expected ₹299 in all. The relaunch does not pass `--suite-record`.
- **The 27 suite failures, classified (no new GPU time):** all 27 are tests that assume a CPU-only host; none is a defect in the GPU path (`docs/gpu_suite_failures_20261110.md`). The S8 run of the timed-region test, separate from the 27, found one thing: `block_sparse` converts its mask on every call (about 3 µs on the Mac, views only). It is a finding for the researcher, not changed here.
- **New in `10e1fe1`:** the preflight scans the session's scripts for every third-party import and refuses, naming the package, if one does not import (`attnbench/session_imports.py`). The lightgbm version check stays beside it.


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

## Time and cost (the relaunch: attempt 1's measured durations × 1.3, ₹108.64/h)

Cap = rate × (planned minutes × 1.3, rounded down, + 10) ÷ 60: the in-guest halt is 1.3 × the plan and the hard delete is 10 minutes after it. Attempt 1 measured the setup and steps 1 to 10 (`session_cost.txt`, the phase logs, the suite log):

| Measured in attempt 1 (A) | Minutes | From |
|---|---|---|
| Boot to the session start, without the suite record and the S8 run | 14.0 | Booted 20:14:12Z. The suite began 20:26:12Z (12.0 min: deploy, preflight, write gate, bootstrap with one SSH retry); 2.0 min more to the session start at 20:38:48Z. The suite (9.7) and S8 (0.8) are taken out |
| Steps 1 to 10, wall time | 26.9 | 20:38:48Z to 21:05:39Z. The work is 21.4 min; the sync between steps took 16 to 47 s each (mean 31 s) |
| T2.17 `assert_gate` | 0.2 | 9 s |
| Exit, sync, halt | 2.6 | T3.2d, measured (attempt 1: 1.8) |

Not measured on this card, so estimates: `predict`, the hold (the runner's 90 s), `ground_truth` (T3.2d step 9: 28 min over 34 cells, scaled to 5 cells and the anchor), `late_check`, `trace_checks`, `host_speed_end`, the compiled stage (never timed clean on a GPU), fetching b1 on B and the reset.

**Instance A**

| | Minutes | Basis |
|---|---|---|
| Setup | 14.0 | measured |
| Steps 1 to 10 | 26.9 | measured, with the sync |
| `predict` | 2 | estimate |
| The hold | 1.5 | the runner's 90 s |
| `ground_truth`, 5 cells + the anchor | 4.9 | T3.2d step 9, 28 / 34 per cell × 6 |
| `late_check`, `trace_checks`, `host_speed_end` | 4.5 | estimates: 1 + 3 + 0.5 |
| Sync between steps 11 to 15 | 2.5 | 5 × 0.5 (measured mean 31 s) |
| T2.17 `assert_gate`, with its sync | 0.7 | 9 s measured + 0.5 |
| (v) the compiled groups, 2 × 3 cells at 2048, with the sync | 10.5 | **estimate; never timed clean on a GPU** |
| Exit, sync, halt | 2.6 | measured |
| **Planned** | **70** | 70.1; without (v): 59.6 |

**Instance B**

| | Minutes | Basis |
|---|---|---|
| Boot 1: setup 14.0, fetch b1 1, `prereqs` and `preflight` 1, the hold 1.5, ground truth opening 1 | 18.5 | measured setup; estimates |
| Reset, SSH back | 2 | estimate |
| Boot 2: the refused resume 1; then b2 again: `prereqs` and `preflight` 1, hold 1.5, ground truth 4.9, `late_check`, `trace_checks`, `host_speed_end` 4.5, sync 2.0, drift 0.5, exit 2.6 | 18.0 | as above, 5 cells × 28 / 34 |
| **Planned** | **39** | 38.5. If the reset never comes, boot 1 runs b2 to its end: 14 + 1 + 17 = 32 min, the longest single boot |

| Limit | A | B |
|---|---|---|
| In-guest halt, from **each** boot (`GCP_HALT_MINUTES`) | 91 (1.3 × 70) | 41 (1.3 × 32, the longest boot) |
| Hard delete (`GCP_MAX_RUN`, DELETE) | 101 min = `1h41m` (halt + 10) | 60 min = `60m` (1.3 × 39 rounded down, 50, + 10) |
| Cap, at the later bound | **₹183** (101 min) | **₹113** (62 min: the reset at about 19 min, then a halted boot 2 at 41 min, if the reset restarts the run timer; 60 if not) |

**Relaunch cap ₹296** (A ₹183 + B ₹113); expected ₹198 (A 70 min ₹127, B 39 min ₹71). With attempt 1's ₹101 (`docs/spend_ledger.md`): **₹397 against the ₹403 cap**, expected ₹299. Without (v), with `ATTNBENCH_REH_COMPILED` unset: A plans 60 min, halts at 78 and is deleted at 88 (`1h28m`, ₹160), a relaunch cap of **₹273** and a total of **₹374**. If a plan moves the total over about ₹400, (v) is cut first and the protocol is not touched. After attempt 1, ₹4,640 of ₹14,000 is spent (`docs/spend_ledger.md`, counted from its rows); the relaunch's cap leaves at least **₹9,060**.

## Named bucket reads (part of the launch yes)

- **On A:** T3.2d's eight XAttention mask files:
  - from `attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components/`;
  - for each of `xattention` and `xattention_noassert` at `_dcal_L2048_b1` and `_dcal_L8192_b1`;
  - `recording/masks.{npz,json}` for each.

  Also the P1 training file, `banked/p1_training/p1_training_t32dc_20261109.json.gz`. Its sha256 is checked against `de7f0f44…`, and the chain stops on a mismatch.
- **On B:** A's b1 session, `<A>/results/t61r_l4_<UTC>_b1/`.
- **On the Mac, afterwards:** each instance's logs and manifests, through `gcp_teardown_session.sh`'s bucket fallback.

## The commands

On the Mac, in a fresh clone at `10e1fe1`:

```bash
git clone <repository> reh && cd reh && git checkout --detach 10e1fe1
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917 GCP_RATE_INR_HR=108.64
ZONE=asia-northeast1-a
```

**Instance A**

```bash
export GCP_HALT_MINUTES=91 GCP_MAX_RUN=1h41m     # without (v): 78 and 1h28m, and leave ATTNBENCH_REH_COMPILED unset
A=attnbench-l4-reh-a-$(date +%Y%m%d-%H%M)
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$A"     # type 'launch'
bash scripts/gcp_deploy_source.sh "$A" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$A" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$A" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$A" "$ZONE"      # no --suite-record (ruling of 10 Nov 2026)
# if it stops on an SSH timeout or a failed scp, rerun it: it is idempotent. That is pre-decided.
B1=results/t61r_l4_$(date -u +%Y%m%dT%H%MZ)_b1
gcloud compute ssh "$A" --zone "$ZONE" --command "cd ~/attnbench_scaffold && \
  ATTNBENCH_REH_COMPILED=1 nohup bash scripts/session_exit.sh --log ~/reh_a.log -- \
  bash scripts/t61_rehearsal_a.sh $B1 >/dev/null 2>&1 < /dev/null &"   # < /dev/null, or ssh does not return
bash scripts/t61_rehearsal_watch.sh "$A" "$ZONE" > ~/reh_watch_a.log 2>&1   # until TERMINATED or gone
bash scripts/gcp_session_elapsed.sh "$A" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$A" "$ZONE" "$B1"
bash scripts/gcp_cleanup_check.sh
```

**Instance B** (after A is deleted)

```bash
export GCP_HALT_MINUTES=41 GCP_MAX_RUN=60m
B=attnbench-l4-reh-b-$(date +%Y%m%d-%H%M)
# launch, deploy, preflight, write gate, bootstrap: as for A, with "$B"
X=results/t61r_l4_$(date -u +%Y%m%dT%H%MZ)_b2_interrupted; Y=${X%_interrupted}
gcloud compute ssh "$B" --zone "$ZONE" --command "cd ~/attnbench_scaffold && \
  nohup bash scripts/session_exit.sh --log ~/reh_b1.log -- \
  bash scripts/t61_rehearsal_b.sh boot1 $A $B1 $X >/dev/null 2>&1 &"
bash scripts/t61_rehearsal_watch.sh "$B" "$ZONE" > ~/reh_watch_b.log 2>&1 &
bash scripts/t61_rehearsal_interrupt.sh "$B" "$ZONE" "$A" "$B1" "$X" "$Y" --within 20
wait                                                     # the watcher: TERMINATED or gone
bash scripts/gcp_session_elapsed.sh "$B" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$B" "$ZONE" "$Y"
bash scripts/gcp_cleanup_check.sh
```

If ground truth has not opened within 20 minutes, the interrupt script resets nothing and prints `INTERRUPT_NOT_REHEARSED`. Boot 1's b2 then runs on and halts itself, and the report says the interruption was not rehearsed.

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
- **The 14 Nov rule:** the compiled path is clean when `compiled/result.json` reads passed on the relaunch, both groups complete, no recompile after warm-up, and the Kineto twin and mask-dependence checks passed. **If it is not clean by 14 Nov 2026, the compiled group is dropped** (the cut order of Appendix A, 9 Nov 2026, item 10) **and the limitation is written**: the compiled variants' ground truth was not measured, and no claim about compiled arms rests on this study. P2's and P4's cards are built without the compiled groups today and stay so. This rehearsal is the only GPU run of the compiled path before then, so there is one attempt: results by Thu 13 Nov leave one day.
