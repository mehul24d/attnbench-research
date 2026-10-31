# Launch card: Shakedown 2 (one L4 session, three runs)

Prepared 28 Oct 2026. Revised 31 Oct 2026 to the full scope. Not launched.
The researcher launches it; nothing here runs on its own. Every number in the
time and cost tables is an estimate. Only Shakedown 1's four passed steps
have run on a card.

## What runs

The session runs three runners one after another, all from the same deployed
commit (`gcp_deploy_source.sh` bundles the local HEAD and confirms the commit
and a clean tree on the instance). Each runner runs each step in its own
process, with a log and a `result.json` per step. Each step is synced to GCS
as it ends (`run_phase.sh`). With `--keep-going`, a failed step is recorded
and the next step runs. There is no in-session fix and redeploy: a failure is
a finding for the next session.

**Run A: Shakedown 1 again.** `python3 scripts/t227_shakedown.py --keep-going`
runs all eleven T2.27 steps. Seven of them failed on 27 Oct, all on the
exact-prompt-length defect fixed in PR #20: `t21_t220`, `nsys`, `replay`,
`probe`, `assert_gate`, `graph_breaks` and `figure1`. The four that passed
(`zoo`, `link`, `launch_queue`, `null_kernel`) run again too; they take
minutes and replicate 27 Oct on the new commit. nsys is called by absolute
path (`trace_checks.nsys_available` looks in `/usr/local/cuda*/bin`, where
image v6 has it).

**Run B: Shakedown 2.** `python3 scripts/t228_steps.py run --keep-going`.

| # | Step | Task | What it decides | Expected minutes |
|---|---|---|---|---|
| 1 | `nvml` | the real NVML reader | The throttle bits decode. The sampler process keeps at least half of 10 Hz, idle and under load, and the loaded block has at least 3 samples; otherwise the step fails. Which bits an L4 shows (the software power cap included) is a finding | 2 |
| 2 | `t216` | T2.16 | K2 at 8192, batch 1: each planned candidate is declined, checked or timed. The record is written once with its sha256, then read back. Fails if no candidate was correct. Runs before `t213`, so the record precedes its first clocked block | 5 |
| 3 | `t213` | T2.13 dry run | Dense, oracle, meanpool and XAttention (authors table) on one RULER `qa_1` prompt, in three passes: no sampler; sampler and tracer; no sampler again. Fails on a refused timed region, a stamp outside the run, a block seen by fewer than two samples, or a sampler under half its rate. Findings: the sampler's cost per arm (with its noise), throttled rounds, the traced overhead, and `check_control` against `t216` | 10 |
| 4 | `density` | T2.20 density probe | Each arm's backend goes through `kernel_densities` at the working, pre-lock densities, one process per compile setting (eager, `compiled_default`, `compiled_capture`). Fails on a backend that keeps no per-layer mask density, or on a row `density_problems` refuses. Each arm's mask density against its nominal density is a finding | 10 |
| 5 | `t212` | T2.12 | Meanpool and XAttention masks are recorded, replayed through `PrecomputedMaskBackend` with the real kernel, compared on logits with the source, then timed together. Fails on a different executed density, or on an XAttention replay that is not bitwise equal (the plan expects it to be; mean-pool is not known to be, so its mismatch is a finding) | 6 |
| 6 | `t22` | T2.2 smoke | The null-kernel host profile of meanpool's prefill from its tape (5 passes), then the reference builder under `cProfile` and `torch.profiler` at 2048 to 32768 on this host. Fails if a pass is refused or the builder profile is not written. The 32768 row is read against the plan's unbanked 94% | 10 |
| 7 | `queue` | T2.6 Q record | Q for the null kernel and the one-element add, in six probe settings: Shakedown 1's own (cold, 2 s, 0.25), the same warm, then the blocker at 1 s and 4 s and the threshold at 0.1 and 0.5. Records the add's Q and lower bound in each setting, and whether they match the null kernel's. Fails only if the null kernel's Q is not agreed in Shakedown 1's setting | 3 |

**Run C: T3.1 rehearsal.** `python3 scripts/t31_session.py --rehearsal`
runs T3.1 steps 1 to 8 (`preflight` to `hold`) and stops before
`ground_truth`, so nothing is clocked. Its config is the P1 config with
`"rehearsal": true`, the working values filled in, the grid cut to 2048 and
16384 (16384 at batch 1 and 4), and predictions written under
`predictions/P1-rehearsal/`. `prediction_dir` refuses a rehearsal under any
other prefix, and refuses that prefix for anything but a rehearsal. The
results are discarded: they are kept in the session tree for debugging and
are never banked.

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` (the target host of the L4 rows) |
| Zone | `asia-northeast1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917`. The launcher defaults to v5, so set `ATTNBENCH_IMAGE` |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention |
| Rate | ₹108.64/h, the all-in planning rate. Teardown needs `GCP_RATE_INR_HR=108.64` |
| Launch window | by 9 Dec 2026 (`_launch_policy.sh` enforces it) |

## Time and cost

| | Hours | ₹ at 108.64/h |
|---|---|---|
| Setup: launch to write gate, deploy, preflight, bootstrap with XAttention | 0.35 | 38 |
| Run A: eleven T2.27 steps (Shakedown 1 took 23 min end to end, with seven steps failing early) | 0.45 | 49 |
| Run B: seven steps (46 min) | 0.75 | 81 |
| Run C: T3.1 steps 1 to 8 on 19 cells, the hold included | 0.45 | 49 |
| Teardown: serial log, results pull, delete, cleanup check | 0.15 | 16 |
| **Expected** | **2.15** | **234** |
| In-guest halt (`GCP_HALT_MINUTES=150`): GPU billing stops here | 2.5 | 272 |
| Hard delete (`GCP_MAX_RUN=3h`) | 3.0 | disk only after the halt |

**Cost cap: ₹300** for the session (the halt's ₹272 plus the disk to the hard
delete). The expected session is 129 minutes against a 150-minute halt.
That margin is about a sixth, much thinner than the 28 Oct card's. If the halt
fires mid-run, every step that has ended has already been synced, and the
session stops there. The step minutes are the least certain part.

## The commands

On the Mac, from the repo root, with the launch commit checked out:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project (never the gcloud default)
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=150 GCP_MAX_RUN=3h GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t228-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean"
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
python3 scripts/t227_shakedown.py --keep-going        # run A
python3 scripts/t228_steps.py run --keep-going        # run B
python3 scripts/t31_session.py --rehearsal            # run C (results discarded)
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t228_l4
bash scripts/gcp_cleanup_check.sh
```

## Known before launch

- **`nvidia-ml-py` on v6 is not on record.** It is a declared dependency
  (`pyproject.toml`), but no run has used it. If step B1 reports that
  `pynvml` is missing, `pip install nvidia-ml-py` on the instance, note the
  version in the session log, and rerun `--from nvml` in the same session
  directory.
- **The resolution floor is provisional.** `--floor-ms` defaults to 1.0 for
  this dry run only. `t213` records the dense control's spread for T3.1's
  floor. Rounds flagged against a provisional floor are findings, not a
  verdict on the harness.
- **`check_control` may refuse in `t213` by design.** The shakedown's dense
  control is `sdpa_flash`. If `t216` chooses another kernel, the check says
  so, and that is recorded as a finding.
- **The add's Q in Shakedown 1 was `1,1022,1022,1022,1022`.** Its first trial's
  first launch waited, which looks like a one-off kernel load, not a queue of
  one. The probe has no warm launch, so the `queue` step records both the
  cold method (the T2.6 method of record) and a warm one. Which one is the
  method of record afterwards is the researcher's call. The model's Q stays
  the null kernel's either way (Appendix A, 25 Oct 2026).
- **The density probe uses working values.** The densities and groups come
  from `docs/lock_value_table.md`'s working block (P1 only, marked "working,
  pre-lock"). The `density` step's numbers are a rehearsal's.
- **Real throttling cannot be forced.** Step B1 shows the reader and the
  process on a quiet card. A hardware slowdown or the power brake is
  validated only if the card happens to throttle.
- **The sampler is a second process on an 8-vCPU host.** `t213` measures
  what it costs each arm.
- **XAttention runs its torch path on the L4**, so `t212` replays its masks
  from the torch estimator; the A100's Triton path is not exercised here.
- **The rehearsal does not cover 32768.** Its grid is cut to keep the session
  inside the halt, so T3.1's 32768 cells (and XAttention's three-length probe
  at 32768) first run in T3.1 itself.
