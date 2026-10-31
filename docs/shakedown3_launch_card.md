# Launch card: Shakedown 3 (one L4 session, three runs)

Prepared 31 Oct 2026; revised the same day for the researcher's four
rulings (probe verdicts, probe lengths, flagged memsets, compiled density).
Not launched. Deploys from `main` at the merge of `rulings-2026-10-31` or
later. The researcher launches it; nothing here
runs on its own. The step minutes are estimates. Shakedown 2's failed steps
ended early, so their times on 31 Oct (47 to 265 s) understate a passing run.

## What runs

Only the steps Shakedown 2 failed, and T3.1 rehearsal steps 1 to 8, in the
order A, C, B: if the halt cuts the session, it cuts B's last step, not the
rehearsal. Every
runner runs each step in its own process, syncs it to GCS as it ends
(`run_phase.sh`), and with `--keep-going` records a failure and goes on.
`--only` (new) runs the named steps in checklist order in a new session,
numbered as in the full checklist; the other steps are neither run nor
reported.

| Run | Command | Steps | What Shakedown 2 found, and the fix | Expected min |
|---|---|---|---|---|
| A | `t227_shakedown.py --keep-going --only t21_t220,replay,probe,figure1` | `t21_t220` | Compared a row name with a launch name. One vocabulary module now (`attnbench/provisioning.py`) | 1 |
| | | `replay` | Pageable host-to-device copy of a scalar during graph capture. The replayed `setitem` builds its value on the device | 3 |
| | | `probe` | Dense and oracle: "different" (cuBLAS memset at 1951 tokens) is now a verdict, and the full recording is the structure. XAttention: memsets with no device record are flagged, not refused; probe points 20480, 28672 and a full recording at 32768. Meanpool: the profiler warm-up | 8 |
| | | `figure1` | The first kernel after the profiler starts had no device record. The recorder now runs one profiler warm-up step | 3 |
| B | `t228_steps.py run --keep-going --only t216,t213,density,t212,t22` | `t216` | Passed on 31 Oct. Rerun because `t213` reads the K2 record from its own session | 1 |
| | | `t213`, `t212` | The oracle and mean-pool backends keep no per-layer density. The density rule: measured from the executed mask in an untimed twin pass | 16 |
| | | `density` | The same, plus `compiled_default` read an overwritten CUDA-graph output. The recorder now copies and is never traced. A compiled row's density is its own compiled twin's; two compiled passes must agree exactly; the difference from eager is reported, flagged beyond ±0.01 | 10 |
| | | `t22` | The null entry replaced the extension's Python wrapper, so the tape took a shorter path. The entry nulls the extension and keeps the wrapper | 10 |
| C | `t31_session.py --rehearsal` | T3.1 steps 1 to 8, `preflight` to `hold` | Stopped at `preflight` on the vocabulary defect. Now also: XAttention's probe at 32768, and the memset zoo in `device_components`. Results discarded, never banked | 28 |

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
| Run A | 15 | 27 |
| Run C | 28 | 51 |
| Run B | 37 | 67 |
| Teardown: results pull, delete, cleanup check | 8 | 14 |
| **Expected** | **94** | **163** (GPU billing stops at the halt) |
| In-guest halt (`GCP_HALT_MINUTES=90`): GPU billing stops | 90 | 163 |
| Hard delete (`GCP_MAX_RUN=2h`) | 120 | disk only after the halt |

**Cost cap: ₹200.** The halt's ₹163 plus the disk to the hard delete. The
runs are expected to end at about 86 minutes, 4 minutes before the halt, so
there is almost no margin; the teardown may straddle the halt, which is
harmless, since every step is synced as it ends and the teardown deletes a
stopped VM. If the halt fires mid-run, the session ends there. The order is
A, C, B, so a halt cuts B's last step (`t22`), not the rehearsal.

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
python3 scripts/t31_session.py --rehearsal              # results discarded
python3 scripts/t228_steps.py run --keep-going --only t216,t213,density,t212,t22
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/shakedown3_l4
bash scripts/gcp_cleanup_check.sh
```

## Known before launch

- **The four rulings are in the code** (Appendix A, 31 Oct 2026). The
  `probe` step and T3.1's step 4 pass on a verdict with its evidence;
  "different" is recorded with the full structure. XAttention's memsets
  with no device record are counted (`flag_counts`); `predict` writes
  `memset_sensitivity.json` with the upper-bound shift. Compiled XAttention
  is no longer refused for differing from eager; it is refused only if two
  compiled passes disagree.
- **The memset zoo and the bound's operations are unrun.** They run first
  in the rehearsal's `device_components` (step 6), on the L4. The zoo calls
  `cudaMemsetAsync` through the CUDA runtime library torch loads; if that
  library is not found, step 6 fails and says so.
- **The H6 bitwise mask gate is unchanged.** If compiled XAttention's masks
  are not bitwise equal to eager, the arm loses L4 admissibility on that
  path. That is a result.
- **The density figures are a rehearsal's.** Working, pre-lock values from
  `docs/lock_value_table.md`.
- **The L4 rows are on-demand.** `t21_t220` checks the row name `on-demand`
  against the launch name STANDARD through the one map.
