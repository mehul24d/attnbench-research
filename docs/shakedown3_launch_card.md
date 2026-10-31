# Launch card: Shakedown 3 (one L4 session, three runs)

Prepared 31 Oct 2026. Not launched. The researcher launches it; nothing here
runs on its own. The step minutes are estimates. Shakedown 2's failed steps
ended early, so their times on 31 Oct (47 to 265 s) understate a passing run.

## What runs

Only the steps Shakedown 2 failed, then T3.1 rehearsal steps 1 to 8. Every
runner runs each step in its own process, syncs it to GCS as it ends
(`run_phase.sh`), and with `--keep-going` records a failure and goes on.
`--only` (new) runs the named steps in checklist order in a new session,
numbered as in the full checklist; the other steps are neither run nor
reported.

| Run | Command | Steps | What Shakedown 2 found, and the fix | Expected min |
|---|---|---|---|---|
| A | `t227_shakedown.py --keep-going --only t21_t220,replay,probe,figure1` | `t21_t220` | Compared a row name with a launch name. One vocabulary module now (`attnbench/provisioning.py`) | 1 |
| | | `replay` | Pageable host-to-device copy of a scalar during graph capture. The replayed `setitem` builds its value on the device | 3 |
| | | `probe` | See "Known before launch": two arms' fixes need rulings | 6 |
| | | `figure1` | The first kernel after the profiler starts had no device record. The recorder now runs one profiler warm-up step | 3 |
| B | `t228_steps.py run --keep-going --only t216,t213,density,t212,t22` | `t216` | Passed on 31 Oct. Rerun because `t213` reads the K2 record from its own session | 1 |
| | | `t213`, `t212` | The oracle and mean-pool backends keep no per-layer density. The density rule: measured from the executed mask in an untimed eager twin pass | 16 |
| | | `density` | The same, plus `compiled_default` read an overwritten CUDA-graph output. The recorder now copies and is never traced; a compiled arm's masks must equal its eager twin's | 10 |
| | | `t22` | The null entry replaced the extension's Python wrapper, so the tape took a shorter path. The entry nulls the extension and keeps the wrapper | 10 |
| C | `t31_session.py --rehearsal` | T3.1 steps 1 to 8, `preflight` to `hold` | Stopped at `preflight` on the vocabulary defect. Results discarded, never banked | 25 |

The `queue` step is not rerun. Shakedown 2 gave the add 1022 warm in every
setting, and the warm probe is now the method of record (Appendix A, 31 Oct
2026); `launch_queue` in run A is not rerun either.

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` |
| Zone | `asia-northeast1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917` (set `ATTNBENCH_IMAGE`; the launcher defaults to v5) |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention |
| Rate | ₹108.64/h all-in planning rate (`GCP_RATE_INR_HR=108.64`) |

## Time and cost

| | Minutes | ₹ at 108.64/h |
|---|---|---|
| Boot to the runs starting (Shakedown 2: 5 min) | 6 | 11 |
| Run A | 13 | 24 |
| Run B | 37 | 67 |
| Run C | 25 | 45 |
| Teardown: results pull, delete, cleanup check | 8 | 14 |
| **Expected** | **89** | **161** |
| In-guest halt (`GCP_HALT_MINUTES=90`): GPU billing stops | 90 | 163 |
| Hard delete (`GCP_MAX_RUN=2h`) | 120 | disk only after the halt |

**Cost cap: ₹200.** The halt's ₹163 plus the disk to the hard delete. The
expected session fills the 90-minute halt, so there is no margin. If the halt
fires mid-run, every finished step is already synced and the session ends
there. The order puts the runs that fix the blocking defects first; run C's
later steps are the ones a halt would cut.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=90 GCP_MAX_RUN=2h GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-s3-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean"
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
python3 scripts/t227_shakedown.py --keep-going --only t21_t220,replay,probe,figure1
python3 scripts/t228_steps.py run --keep-going --only t216,t213,density,t212,t22
python3 scripts/t31_session.py --rehearsal              # results discarded
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/shakedown3_l4
bash scripts/gcp_cleanup_check.sh
```

## Known before launch

- **The probe has three open items, and they reach run C.** Without rulings
  (proposed 31 Oct 2026), the `probe` step is expected to fail again on
  dense, oracle and XAttention. T3.1's own step 4 records XAttention at its
  three probe lengths, so the rehearsal is expected to stop at step 4, and
  steps 5 to 8 would not run:
  1. XAttention's three-length rule asks for 34816 tokens, which is over the
     32768 position limit. Below 32768 XAttention has only two chunk counts
     (3 and 4), so `chunked_probe_lengths(limit=32768)` refuses.
  2. XAttention's 84 `cudaMemsetAsync` calls (3 per layer, under `addmm`) have
     no device record. The hypothesis is zero-work memsets; whether to
     tolerate them is open.
  3. Dense and oracle: cuBLAS issues a `Memset` before `addmm` at 1951 tokens
     but not at 4068 or 8075 (a GEMM algorithm regime). The proposal is probe
     lengths inside one regime.
- **XAttention may fail the compiled equality check.** On 31 Oct its eager
  mask density was 0.66688 and its `compiled_capture` density 0.66582. The
  density rule refuses a compiled arm whose masks differ from its eager twin,
  so `density` is expected to refuse compiled XAttention. That is the rule
  working; what to do about it is open.
- **The density figures are a rehearsal's.** Working, pre-lock values from
  `docs/lock_value_table.md`.
- **The L4 rows are on-demand.** `t21_t220` checks the row name `on-demand`
  against the launch name STANDARD through the one map.
