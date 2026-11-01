# Launch card: T3.1, the P1 session

Prepared 28 Oct 2026; rewritten 1 Nov 2026 after Shakedown 6 and the
researcher's rulings that followed it. Deploys from `main` at the merge of
`shakedown6-fixes-2026-11-01` or later. Launchable only on the researcher's
yes; nothing here runs on its own. **Stop at the first failure:** the session
runs once, the runner stops at the first step that fails, and the instance
is torn down. Nothing is fixed and resumed inside this session; a fix and a
new session need their own yes.

## What runs

`scripts/t31_session.py`, steps in `scripts/t31_steps.py`, values in
`configs/t31/p1_session.json` with its null working values filled from
`docs/lock_value_table.md` (marked "working, pre-lock", P1 only). Eleven
steps, in order, each in its own process with its own log
(`results/t31_p1_<UTC>/<NN>_<step>/step.log`), each synced to GCS as it ends
(`run_phase.sh`, phase `t31_<NN>_<step>`).

The grid: lengths 2048, 4096, 8192, 16384 and 32768 at batch 1, and 16384 at
batch 4 for dense only; dense, the oracle and mean-pool at 0.25 and 0.5, and
the two XAttention arms at their calibrated density. That is 36 cells. The
ground truth runs three compile groups: eager (all five arms), compiled
default and compiled capture (mean-pool and XAttention, each with the eager
dense control). It runs 10 clean rounds per cell and 5 dense-selection
rounds; the rerun budget is 0 and the floor is derived in the session.

Minutes are from `scripts/t31_projection.py` (low / central / high), which
scales Shakedown 6's measurements to this grid. Steps 5 to 11 have not run on
a GPU, so their minutes are the least certain.

| # | Step | Task | Minutes |
|---|---|---|---|
| 1 | `preflight` | T2.1 probe `start`, T2.20 fields, config of record | 1 / 1 / 1 |
| 2 | `link` | T2.5 row, T2.6 Q, mechanism table per length | 5 / 5 / 5 |
| 3 | `dense_select` | T2.16 record per (length, batch); the session floor | 6 / 6 / 6 |
| 4 | `probe` | T2.18, every arm, at every grid length (durations discarded) | 19 / 19 / 19 |
| 5 | `host_components` | T2.2 per cell: the recording, the null pass (instrumented), the bare prices with the per-layer residual, which the prediction reads | 36 / 48 / 84 |
| 6 | `device_components` | T2.4 replay per cell; the priced cells | 9 / 10 / 15 |
| 7 | `predict` | T2.11: write, sync, read back; the receipt | 1 |
| 8 | `hold` | until sync + 90 s (`PLANNED_SYNC_WAIT_S`), on the instance clock | 1.5 |
| 9 | `ground_truth` | T2.13, one process per compile group, under the recompile gate | 55 / 77 / 124 |
| 10 | `verify` | `load_committed` against the earliest ground-truth stamp | 1 |
| 11 | `postflight` | T2.1 probe `end` | 2 |

Step 5's minutes, central: recording 4, isolated replay 4, null passes 11,
bare pass 23, residual 7. The bare pass is mostly device time: at least
1000 repetitions of each signature, after a synchronise.

## What changed since Shakedown 6

- **Step 5, the bare timer's entry.** A custom op's re-issues were counted
  as calls, which refused the oracle in Shakedown 6. They are no longer
  counted (`5375b07`); on the instance, with the fix applied, the oracle
  aligned at 2048 and passed the chain gate at 5.5%.
- **Step 5, once-only host pricing.** XAttention's null pass was refused at a
  `where` with a Python number. Each pass call is now priced once, inside the
  recorded call that nests it. This was tested on Shakedown 6's own sequence,
  with a conservation test, and the CPU dry run now reproduces the nesting.
- **Step 5, the per-layer residual.** The arm's own Python between calls,
  measured per layer from the prefill's span under the tape with real kernels
  and added back to the bare prices. A layer whose spread is over T4.1 item
  34 (working 10%), or where the device fell behind, labels the cell. It is
  never refused.
- **Step 5, speed.** The isolated trace is read by bisection: the oracle's
  8192-token trace took 4 s, down from over 27 minutes, and that alone ended
  Shakedown 6's zoo. A null pass takes about half its former time.
- **Step 9, the kernel interface outside Dynamo for every arm.** Compiled
  mean-pool recompiled per call under the default settings in Shakedown 6.
  The oracle, mean-pool and XAttention now call the kernel through
  `block_sparse.kernel_interface`, one declared break under both settings.
  The recompile gate (0 compile events after warm-up) refuses the cell
  otherwise, which stops this session.

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` (32 GB RAM) |
| Zone | `asia-northeast1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917` (set `ATTNBENCH_IMAGE`; the launcher defaults to v5) |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention |
| Rate | ₹108.64/h all-in planning rate (`GCP_RATE_INR_HR=108.64`) |
| Launch window | by 9 Dec 2026; within the credit by 30 Nov 2026 |

While this VM exists, the H100 retry loop running on the Mac skips its rounds
(one GPU session at a time). It resumes by itself after the delete.

## Time and cost

| | Hours | ₹ at 108.64/h |
|---|---|---|
| Setup: launch, deploy, preflight, write gate, bootstrap (Shakedown 6: 6 min) | 0.1 | 11 |
| Steps 1 to 11, central (low 2.3 h, high 4.3 h) | 2.9 | 312 |
| Teardown: results pull, delete, cleanup check | 0.13 | 14 |
| **Expected, central** (low 2.5 h ₹274, high 4.6 h ₹495) | **3.1** | **336** |
| In-guest halt (`GCP_HALT_MINUTES=420`): GPU billing stops | 7.0 | 760 |
| Hard delete (`GCP_MAX_RUN=7h30m`) | 7.5 | disk only after the halt |

**Cost cap: ₹760**, the halt. The projection's high case is inside the
expected 5 h. There is no "fixed and resumed" row: the session stops at the
first failure.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=420 GCP_MAX_RUN=7h30m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t31-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean"
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`, once:

```bash
python3 scripts/t31_session.py          # stops at the first failure; no --from in this session
```

Back on the Mac, when it ends (passed or stopped):

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t31_p1
bash scripts/gcp_cleanup_check.sh
```

## The prediction and the clock

- Steps 1 to 6 time microbenchmarks, kernels, null passes, bare calls and
  the tape span only, never the composed workload (plan Section 5.1). The
  residual's span runs the arm's prefill under the tape before the
  prediction is written; it is a component measurement, as the null pass is.
- Step 7 writes the file and its sha256, uploads both with no-clobber,
  reads them back, and records the store's creation time (`synced_at`).
- Step 8 waits until the instance clock reads `synced_at + 90 s`.
- Each ground-truth process checks the same condition itself before its
  first forward and records `clock_not_before` there.
- Step 10 runs the analysis's own refusal in the session, so a late
  prediction is found before teardown.

## Before launch, on the Mac

The CPU dry run, which must pass. It needs x-attention cloned outside the
repo at `XATTN_COMMIT`, and transformers 4.46.0 (the L4's) in a venv over
the system torch:

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
python3 scripts/t228_steps.py density --cpu-boundary --out /tmp/density --length 1024
python3 scripts/t228_steps.py pricing_zoo --cpu-boundary --out /tmp/zoo --length 1024
python3 scripts/t227_steps.py graph-breaks-one --cpu-boundary --arms meanpool --setting default --out /tmp/gb --gb-length 1024
python3 scripts/t31_projection.py
```

## The CPU dry run (1 Nov 2026)

On the Mac, behind `attnbench/cpu_boundary.py` (transformers 4.46.0,
x-attention at `XATTN_COMMIT`), at the launch branch's head, everything
passed:
- **The rehearsal, steps 1 to 8.** In step 5 every cell was recorded,
  null-passed, bare-priced and given its residual. Each XAttention cell
  priced its `scalar_tensor` calls inside their `where` (10 per cell, 8 for
  the assert-free variant), the nesting that refused Shakedown 6.
- **`density`.** Both compiled groups made 0 compile events in 12 prefills
  after warm-up.
- **`pricing_zoo`.**
- **The graph-break reports**, each arm under both settings. Mean-pool and
  the oracle show exactly the declared break, at `block_sparse.py
  kernel_interface`, and dense shows none. The XAttention arms show the
  declared break and one more at `Xattention.py:58`: Dynamo cannot trace the
  stub device's properties object there. The L4's own report in Shakedown 6
  had no such break.

The CPU's timings, residuals, ratios and labels mean nothing: a CPU call runs
its compute.

## Known before launch

- **Steps 5 to 11 have not run on a GPU for a sparse arm.** In Shakedown 6
  step 5 priced dense at 2048 and the oracle with the entry fix applied at
  run time. Once-only pricing, the residual and the speed-ups are tested on
  the CPU and on Shakedown 6's banked sequence and traces, not on the L4.
  The first failure stops the session.
- **The recompile gate is the proof of the kernel exclusion.** Behind the
  CPU boundary the stub sizes a tensor from its data, so a traced interface
  shows its break and is refused as another code path. The L4's per-call
  recompile itself cannot be reproduced on a CPU.
- **The oracle's host work grows with length.** It builds its mask on the
  host: 21,507 host operators per prefill at 2048 and 125,751 at 8192. At
  that rate it reaches about 735,000 at 32768, which makes recording and
  isolated-replay traces of over 1 GB. The 32 GB machine is expected to hold
  them, but no session has loaded one that size.
- **The residual's device check.** A layer where the device fell behind the
  host is labelled, and its span may hold launch-queue waits. Expect labels
  at 32768.
- **The density figures and the floor are working, pre-lock values**, and
  the rows are P1's pre-lock data (plan Section 5.3).
- **The H100 loop** on the Mac is Flex only (it was started from an older
  checkout) and skips its rounds while this VM exists.
