# L4 rehearsal card: the P2 runner before an A100 session is spent

Prepared 10 Nov 2026 from the adviser's ruling of 10 Nov 2026. **Launchable
only on the researcher's yes, in their own words, naming the instances, the
zone, the commit and the cap. Nothing here runs on its own.**

**Exploratory. No row of this rehearsal enters training, the floors, Gate A,
the lock or any claim.** Every session it writes says so in `manifest.json`
(`exploratory`) and in `EXPLORATORY.md`. Its prediction goes to
`predictions/P2-R/`, not P2's prefix. The pairing is `P2-R`.

| | |
|---|---|
| Commit | **`2053377`** (`2053377e3edf3e9d13758e1e80339474a948ec81`, branch `l4-rehearsal-2026-11-10`: `main` at `f1934b3` merged with this branch), from a fresh clone, detached, clean tree |
| Instances | A: `attnbench-l4-reh-a-<YYYYMMDD-HHMM>`; then B: `attnbench-l4-reh-b-<YYYYMMDD-HHMM>`. Both `g2-standard-8` with one L4, on-demand (STANDARD) |
| Zone | `asia-northeast1-a` |
| Image | `attnbench-env-v6-20260917` |
| Rate | ₹108.64/h (planning rate, all-in) |
| **Cap** | **₹403** (A ₹301 + B ₹102); expected ₹279. Without scope (v): ₹379 |

A runs first. B is created only after A is deleted, because the GPU quota is 1.

## Attempt 1 (10 Nov 2026, commit `2053377`) and the relaunch

Instance A ran on the researcher's yes of 10 Nov. It stopped itself at step 11 of 15 (`predict`) after 55 min (₹101), was torn down from the bucket, and nothing is left billing. **Instance B was not started.**

- **Cause:** `lightgbm` is not on image v6 and no launch path installed it, so the learned predictor could not fit (`CostModelError`). P2's and P4's A100 sessions would have stopped at the same step, after their setup and ten steps. Fixed in `5c9175a`: the bootstrap installs `lightgbm` at `learned.LIGHTGBM_VERSION` (4.6.0) and checks the import; `real_preflight` refuses without it, so the stop now comes in step 2.
- **Passed on A:** `prereqs`, `preflight` (T2.26's five variants included), `link`, `launch_queue`, `probe`, `mask_gate` (4 cells, 112 of 112 layers identical, largest density difference 0.0), `host_components`, `device_components`, `dense_select`, `host_speed_mid`, and T2.17's `assert_gate` (3 prompts, 84 layer records).
- **Not run:** ground truth, trace sizes, the second block, the interruption and the compiled stage: scopes (i), (iii) and (v) are still open.
- **The suite record (T2.26 in the bootstrap) returned rc 1:** 27 failed, 6,123 passed, 215 skipped in 9:41. Most failures are tests that assume a CPU-only host (`a CUDA device is present; the CPU boundary is for a CPU`, the CPU dry runs) failing on a CUDA host; the diagnosis is not finished. The full suite therefore cannot pass as a GPU record, and `gpu_suite_record.sh` needs a CPU-only deselection or a GPU-only selection before any card passes `--suite-record` again. The record is banked (phase `s9_suite`). The relaunch does not pass the flag.
- **Measured minutes (A, steps 1 to 10):** `link` 2.8, `probe` 5.0, `mask_gate` 0.2, `host_components` 7.7, `device_components` 5.1, `dense_select` 0.4, the rest under 1; 21.4 in all. The table below used T3.2d's per-cell scaling and overstated `host_components` (23.2) and `device_components` (14.4).

**Relaunch (needs its own yes):** commit **`5c9175a`**; instance A `attnbench-l4-reh-a-<YYYYMMDD-HHMM>` again, then B; same zone and image. Planned A is **60 min** (setup 12, steps 1 to 10 at 21.4, predict 2, hold 1.5, ground truth 4.9, late and end checks 1, trace checks 3, assert gate 0.1, compiled 10 (estimate), exit 2.6): `GCP_HALT_MINUTES=78`, `GCP_MAX_RUN=1h28m`, cap **₹160**. B is unchanged (₹102). Cap for the relaunch: ₹160 + ₹102 = **₹262**; with attempt 1's ₹101 the rehearsal totals at most **₹363**, inside the original ₹403. Expected ₹272 in all.


## What it rehearses (adviser's scope, by priority)

| Scope | Where | What is measured or checked |
|---|---|---|
| (i) the real ground-truth harness on a reduced grid | A (block b1, 2048 and the anchor), B (block b2, 8192) | P2's `ground_truth` step (`t61_session.real_harness`, T3.1's eager group) on the L4. `trace_sizes.json` gives each Kineto trace's bytes, by arm and cell |
| (ii) P2's mask-gate path against P1's saved masks | A, `mask_gate` | P2's recorder, `mask_gate.compare_banks` and `mask_gate.gate` at **±0.01 density** (`DENSITY_FLAG_TOLERANCE`). The inputs are T3.2d's four XAttention cells (two arms, 2048 and 8192) against the same prompts recorded again on this card's PyTorch path. Same hardware and same path, so the masks should agree; a failure is a code defect. Block agreement is descriptive |
| (iii) the block-aware runner, two blocks | A, then B | The block spec (`configs/t61/rehearsal_lock.json`), the common anchor (dense, 8192, batch 1) measured in both blocks, b2 importing b1 (`--import-block`), the between-block drift check (`block_drift`, working bound 6.77%), the halt and self-halt, and a **simulated interruption**: B is reset after b2's ground truth opens; the resume must be refused with `INTERRUPTED_DISCARD.md`; b2 then runs again in a new session, and the drift check reads b1, the discarded b2 (skipped) and the new b2 |
| (iv) T2.17's assert gate and T2.26's GPU suite | A | `t227_steps.py assert_gate` at 2048 over 3 prompts. T2.26's five GPU variants in every preflight: a skip, a failure or a count other than five is a problem. And the suite record (`gpu_suite_record.sh`), run by `gcp_bootstrap_instance.sh --suite-record` before anything is measured, bounded at 25 min |
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

## Time and cost (measured L4 durations × 1.3, ₹108.64/h)

Per-cell minutes are T3.2d's (`docs/spend_ledger.md`, 9 Nov 2026). Each is divided over its 34 measured cells, which include the 16384 and 32768 cells, so the scaling overstates 2048 and 8192.

**Instance A**

| | Minutes | Basis |
|---|---|---|
| Deploy, preflight, write gate, bootstrap, staging | 6 | T3.2d: 199 boot-to-halt − 191 session − 2.6 exit-to-halt = 5.4, + staging |
| b1: `prereqs`, `preflight`, `link`, `launch_queue`, `dense_select`, `host_speed_mid`, `predict`, `late_check`, `host_speed_end` | 20 | T3.2d's session less its steps 4, 5, 6, 9 and 12 (191 − 171.5), measured as a whole |
| b1: T2.26's variants in the preflight | 2 | estimate |
| b1: `probe`, 2 of 5 lengths | 5.2 | T3.2d step 4, 13 × 2/5 |
| b1: `mask_gate`, 4 cells, untimed recording | 4 | estimate |
| b1: `host_components`, 10 cells | 23.2 | T3.2d step 5, 79 / 34 per cell |
| b1: `device_components`, 10 cells | 14.4 | T3.2d step 6, 49 / 34 per cell |
| b1: the hold | 1.5 | the runner's 90 s |
| b1: `ground_truth`, 5 cells + the anchor | 4.9 | T3.2d step 9, 28 / 34 per cell |
| b1: `trace_checks` | 3 | estimate |
| T2.17 `assert_gate` | 3 | estimate |
| (v) the compiled groups, 2 × 3 cells at 2048 | 10 | **estimate; never timed clean on a GPU** |
| The suite record, in the bootstrap (`--suite-record`) | 20 | estimate (the Mac suite takes 11 min); bounded at 25 |
| Exit, sync, halt | 2.6 | T3.2d, measured |
| **Planned** | **120** | b1 alone 78 min (`planned_hours` 1.31) |

**Instance B**

| | Minutes | Basis |
|---|---|---|
| Boot 1: setup, fetch b1, `prereqs`, `preflight` up to b2's ground truth | 12 | as above |
| Reset, SSH back | 2 | estimate |
| Boot 2: the refused resume; then b2 again (`prereqs`, `preflight`, ground truth on 4 cells + the anchor, `late_check`, `trace_checks`, `host_speed_end`); drift; exit | 20 | as above, 5 cells × 28 / 34 |
| **Planned** | **34** | b2 alone 16 min (`planned_hours` 0.27). The longest single boot is 22 min, if the reset never comes |

| Limit | A | B |
|---|---|---|
| In-guest halt, from **each** boot (`GCP_HALT_MINUTES`) | 156 (1.3 × 120) | 28 (1.3 × 22, the longest boot) |
| Hard delete (`GCP_MAX_RUN`, DELETE) | 166 min = `2h46m` (halt + 10) | 54 min = `54m` (1.3 × 34 + 10) |
| Cap, at the later bound | **₹301** (166 min) | **₹102** (56 min: two halted boots, if a reset restarts the run timer; 54 if not) |

**Cap ₹403; expected ₹279 (A ₹217, B ₹62).** Without (v), with `ATTNBENCH_REH_COMPILED` unset: A plans 110 min, halts at 143 and is deleted at 153 (`2h33m`, ₹277), for a total cap of **₹379**. Before the rehearsal, ₹4,539 of ₹14,000 is spent (`docs/spend_ledger.md`, counted from its rows), leaving at least ₹9,058 after it.

## Named bucket reads (part of the launch yes)

- **On A:** T3.2d's eight XAttention mask files:
  - from `attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components/`;
  - for each of `xattention` and `xattention_noassert` at `_dcal_L2048_b1` and `_dcal_L8192_b1`;
  - `recording/masks.{npz,json}` for each.

  Also the P1 training file, `banked/p1_training/p1_training_t32dc_20261109.json.gz`. Its sha256 is checked against `de7f0f44…`, and the chain stops on a mismatch.
- **On B:** A's b1 session, `<A>/results/t61r_l4_<UTC>_b1/`.
- **On the Mac, afterwards:** each instance's logs and manifests, through `gcp_teardown_session.sh`'s bucket fallback.

## The commands

On the Mac, in a fresh clone at `2053377`:

```bash
git clone <repository> reh && cd reh && git checkout --detach 2053377
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917 GCP_RATE_INR_HR=108.64
ZONE=asia-northeast1-a
```

**Instance A**

```bash
export GCP_HALT_MINUTES=156 GCP_MAX_RUN=2h46m     # relaunch: 78 and 1h28m (see above)
A=attnbench-l4-reh-a-$(date +%Y%m%d-%H%M)
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$A"     # type 'launch'
bash scripts/gcp_deploy_source.sh "$A" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$A" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$A" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$A" "$ZONE" --suite-record   # attempt 1 only; the relaunch omits the flag
# if it stops at 7/8 (the suite): the record is in the bucket (phase s9_suite);
# rerun it without --suite-record, then go on. That is pre-decided, so no question is asked.
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
export GCP_HALT_MINUTES=28 GCP_MAX_RUN=54m
B=attnbench-l4-reh-b-$(date +%Y%m%d-%H%M)
# launch, deploy, preflight, write gate, bootstrap: as for A, with "$B" and no --suite-record
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
- The suite record is an opt-in step of the shared bootstrap. **P2's and P4's cards do not pass `--suite-record` yet.** Turning it on adds about 20 min of setup to each card, and that is a decision to take with those cards' caps.
- The compiled minutes and the suite minutes are estimates. A suite that reaches the 25-minute bound reports rc 124, and its record is partial.
- **The 14 Nov rule:** if the compiled path is not clean by 14 Nov 2026 (both groups complete, no recompile after warm-up, both compiled checks passed), the compiled group is dropped per the plan (Appendix A, 9 Nov 2026, item 10's cut order) and the limitation is written. This rehearsal is the only GPU run of the compiled path before then.
