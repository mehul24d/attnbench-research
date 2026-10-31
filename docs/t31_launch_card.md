# Launch card: T3.1, the P1 session

Prepared 28 Oct 2026. Not launched, and not launchable yet: four steps
refuse on a GPU until their Phase 2 work is done (below). The researcher
launches it; nothing here runs on its own.

## What runs

`scripts/t31_session.py`, steps in `scripts/t31_steps.py`, values in
`configs/t31/p1_session.json`. Eleven steps, in order, each in its own process
with its own log (`results/t31_p1_<UTC>/<NN>_<step>/step.log`), stopping at
the first failure, each synced to GCS as it ends (`run_phase.sh`, phase
`t31_<NN>_<step>`). Resume with `--from STEP --session DIR`.

| # | Step | Task | Expected minutes | Built? |
|---|---|---|---|---|
| 1 | `preflight` | T2.1 probe `start`, T2.20 fields, config of record (T2.27's step) | 3 | yes |
| 2 | `link` | T2.5 row, T2.6 Q, mechanism table per length (T2.27's steps) | 10 | yes |
| 3 | `dense_select` | T2.16 record per (length, batch): 6 cells | 12 | yes |
| 4 | `probe` | T2.18, every arm, durations discarded | 15 | no |
| 5 | `host_components` | T2.2 null pass, every arm and cell | 30 | no |
| 6 | `device_components` | T2.4 replay, every arm, density and cell; priced cells | 60 | no |
| 7 | `predict` | T2.11: write, sync, read back; the receipt | 1 | yes |
| 8 | `hold` | until sync + 90 s (PLANNED_SYNC_WAIT_S), instance clock | 1 | yes |
| 9 | `ground_truth` | T2.13, one process per compile group (eager, default, capture) | 135 | no |
| 10 | `verify` | `load_committed` against the earliest ground-truth stamp | 1 | yes |
| 11 | `postflight` | T2.1 probe `end` | 2 | yes |

`python3 scripts/t31_session.py --list` prints what each unbuilt step still
needs.

## The prediction and the clock

- Steps 1 to 6 time microbenchmarks, kernels and null passes only, never
  the composed workload (plan Section 5.1).
- Step 7 writes the file and its sha256, uploads both with no-clobber,
  reads them back, and records the store's creation time (`synced_at`).
- Step 8 waits until the instance clock reads `synced_at + 90 s` (`PLANNED_SYNC_WAIT_S`: the 60 s gap plus a 30 s margin).
  The analysis refuses at `synced_at + 60 s` and later, so the guard makes
  the release strictly past it.
- Each ground-truth process checks the same condition itself before its
  first forward of any kind, and records `clock_not_before` there. This
  covers the strict rule (Appendix A, 27 Oct 2026) until `time_arms` stamps
  the warm-up itself.
- Step 10 runs the analysis's own refusal in the session, so a late
  prediction is found before teardown.
- Once ground truth has started, the runner refuses a resume from `predict`
  or earlier: start a new session instead.
- The receipt holds bucket-relative paths. The bucket is found as
  `run_phase.sh` finds it, so no banked file names the project.

## The instance

| | |
|---|---|
| Card | 1 × L4, `g2-standard-8`, `asia-northeast1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917` via `ATTNBENCH_IMAGE`, then `gcp_bootstrap_instance.sh` with XAttention |
| Rate | ₹108.64/h, all-in planning rate. Teardown needs `GCP_RATE_INR_HR=108.64` |
| Launch window | by 9 Dec 2026; within the credit by 30 Nov 2026 |

## Time and cost (estimates; no step has run on an L4)

| | Hours | ₹ at 108.64/h |
|---|---|---|
| Setup: launch, deploy, preflight, write gate, bootstrap | 0.35 | 38 |
| Steps 1 to 8 | 2.2 | 239 |
| Step 9 (20 clean rounds, about 14 arm-density configs, 6 cells; scales with reps × densities) | 2.25 | 244 |
| Steps 10, 11; teardown | 0.2 | 22 |
| **Expected** | **5.0** | **543** |
| One failure, fixed and resumed | 6.0 | 652 |
| Halt (`GCP_HALT_MINUTES=420`) | 7.0 | 760 (the cap) |
| Hard delete (`GCP_MAX_RUN=7h30m`) | 7.5 | disk only after the halt |

The cost cap is the halt: ₹760 at ₹108.64/h.

## Commands (when every step is built and the values are decided)

On the Mac, from the repo root, on the commit to deploy:

```bash
export ATTNBENCH_GCP_PROJECT=...           # never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=420 GCP_MAX_RUN=7h30m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t31-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh
bash scripts/gcp_launch_l4.sh "$NAME"
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
python3 scripts/t31_session.py
python3 scripts/t31_session.py --from STEP --session results/t31_p1_<UTC>   # after a fix
```

A code fix is committed on the Mac and redeployed, because the runner
refuses a dirty tree. So are the session values: they live in the committed
config, never edited on the instance.

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t31_p1
bash scripts/gcp_cleanup_check.sh
```

## Owed before launch

**Values** (null in `configs/t31/p1_session.json`; a real run refuses
while any is null): the density set per arm, the compile groups' arm
lists, reps, the rerun budget, the resolution floor and the dense-selection
rounds. These are lock values used provisionally on P1, before the lock.

**Phase 2** (T3.1 depends on every Phase 2 task except T2.14's AWS half,
T2.22 and T2.23 to T2.26; Appendix A, 25 Oct 2026):

| Task | Status | Must finish before T3.1 |
|---|---|---|
| T2.1, T2.7, T2.20 | Done | nothing |
| T2.8, T2.9, T2.10 | Done | nothing |
| T2.11, T2.15, T2.19, T2.28 | Built | row to Done; T2.28's deltas wait for T2.23, so it is computed after T3.1 |
| T2.2 | Partly built | per-cell arm runner; L4 smoke run (Shakedown 2); compiled arms have no host profile (T2.21) |
| T2.3 | Partly built | GPU traces of two code paths (Shakedown 1); the looser "same sequence"; the arms table's synchronisation column |
| T2.4 | Partly built | pricing substituted positions; GPU run (Shakedown 1) |
| T2.5 | Partly built | s_fixed for a blocking host-to-device copy and an event synchronise; the load decision |
| T2.6 | Partly built | Q measured on the L4 (Shakedown 1) |
| T2.12 | Partly built | real-kernel run on the L4 (Shakedown 2) |
| T2.13 | Partly built | the cell join (step 9); strict warm-up stamp in `time_arms`; real NVML reader; L4 dry run (Shakedown 2) |
| T2.14 | Not started | the GCP half: decide whether v6 plus `gcp_bootstrap_instance.sh` counts, or build the image |
| T2.16 | Partly built | L4 dry run (Shakedown 2); `TOL` is a lock value |
| T2.17 | Partly built | bitwise gate on the L4 (Shakedown 1) |
| T2.18 | Partly built | chunked-arm lengths; full-length structure; L4 test (Shakedown 1) |
| T2.21 | Partly built | real XAttention report and L4 dry run (Shakedown 1) |
| T2.27 | Running | every step passed, or its failure and fix recorded |

**Not in this runner:** the delay-injection slope check (T2.25, injector
not built) and the profiler-overhead pass (Section 5.3 step 8). Gate A
needs the first. Either is added as a step after `verify` once built.

## Dry run

`python3 scripts/t31_session.py --dry-run --session /tmp/t31_dry` passes on
a Mac in about 40 s. It uses synthetic priced cells for steps 4 to 6 and a
dense-only ground truth on the two-layer random model through the real
`ground_truth.time_arms`. Steps 7 to 10 run the real code against a local
store. Skew is 2 s and guard 1 s, so it does not wait 90 s. A real run
refuses less than the planned 90 s. `tests/test_t31_session.py` covers the order, the
resume rules, the refusals and the GPU wiring (autospec fakes).
