# Launch card: Shakedown 2 (T2.13, T2.16, T2.12, T2.2 and the real NVML reader)

Prepared 28 Oct 2026. Not launched. The researcher launches it; nothing here
runs on its own. Every number in the time and cost tables is an estimate: none
of these five steps has run on a card.

## What runs

`python3 scripts/t228_steps.py run` from branch `t228-followups-2026-10-28`
(`gcp_deploy_source.sh` bundles the local HEAD and confirms the commit and a
clean tree on the instance). It is the T2.27 runner with a different step
list: each step in its own process, stopping at the first failure, a log per
step (`results/t228_shakedown_<UTC>/<NN>_<step>/step.log`), each synced to GCS
as it ends (`run_phase.sh`, phase `t228_<NN>_<step>`).

| # | Step | Task | What it decides | Expected minutes |
|---|---|---|---|---|
| 1 | `nvml` | the real NVML reader | the throttle bits decode; the sampler process keeps at least half of 10 Hz, idle and under load; the loaded block has at least 3 samples. Fails otherwise. Which bits an L4 shows (the software power cap included) is a finding | 2 |
| 2 | `t216` | T2.16 | K2 at 8192, batch 1: each planned candidate declined, checked or timed; the record written once, with its sha256, and read back. Fails if no candidate was correct. Runs before `t213` so the record precedes its first clocked block | 5 |
| 3 | `t213` | T2.13 dry run | dense, oracle, meanpool, XAttention (authors table) on one RULER `qa_1` prompt. Passes: no sampler, then sampler and tracer, then no sampler again. Fails on a refused timed region, a stamp outside the run, a block seen by fewer than two samples, or a sampler under half its rate. Findings: the sampler's cost per arm (with its noise), throttled rounds, the traced overhead, `check_control` against `t216` | 10 |
| 4 | `t212` | T2.12 | meanpool and XAttention masks recorded, replayed through `PrecomputedMaskBackend` with the real kernel, logits compared with the source, then timed together. Fails on a different executed density, or an XAttention replay that is not bitwise equal (the plan expects it; mean-pool is not known to be, so its mismatch is a finding) | 6 |
| 5 | `t22` | T2.2 | the null-kernel host profile of meanpool's prefill from its tape (5 passes), and the reference builder under `cProfile` and `torch.profiler` at 2048 to 32768 on this host. Fails if a pass is refused or the builder profile is not written. The 32768 row is read against the plan's unbanked 94% | 10 |

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` (the target host of the L4 rows) |
| Zone | `asia-northeast1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917`; the launcher's default is v5, so set `ATTNBENCH_IMAGE` |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention |
| Rate | ₹108.64/h, the all-in planning rate. Teardown needs `GCP_RATE_INR_HR=108.64` |
| Launch window | by 9 Dec 2026 (`_launch_policy.sh` enforces it) |

## Time and cost

| | Hours | ₹ at 108.64/h |
|---|---|---|
| Setup: launch to write gate, deploy, preflight, bootstrap with XAttention | 0.35 | 38 |
| The five steps (33 min) | 0.55 | 60 |
| Teardown: serial log, results pull, delete, cleanup check | 0.15 | 16 |
| **Expected, nothing fails** | **1.05** | **114** |
| One failure, fixed on the Mac, redeployed and resumed with `--from` | 1.75 | 190 |
| In-guest halt (`GCP_HALT_MINUTES=150`): GPU billing stops here | 2.5 | 272 |
| Hard delete (`GCP_MAX_RUN=3h`) | 3.0 | disk only after the halt |

**Cost cap: ₹300** for the session (the halt's ₹272 plus the disk to the hard
delete). The estimates for step minutes are the least certain part; the halt
is set at about two and a half times the expected session.

## The commands

On the Mac, from the repo root, with the branch checked out:

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
python3 scripts/t228_steps.py run
# after a failure and its fix (a code fix is committed on the Mac and redeployed):
python3 scripts/t228_steps.py run --from STEP --session results/t228_shakedown_<UTC>
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t228_l4
bash scripts/gcp_cleanup_check.sh
```

## Known before launch

- **`nvidia-ml-py` on v6 is not on record.** It is a declared dependency
  (`pyproject.toml`), but no run has used it. If step 1 reports that `pynvml`
  is missing, `pip install nvidia-ml-py` on the instance, note the version in
  the step's log, and resume `--from nvml`.
- **The resolution floor is provisional.** `--floor-ms` defaults to 1.0 for
  this dry run only; `t213` records the dense control's spread so the lock can
  set the real one. Rounds flagged against a provisional floor are findings,
  not a verdict on the harness.
- **`check_control` may refuse in `t213` by design.** The shakedown's dense
  control is `sdpa_flash`; if `t216` chooses another kernel the check says so,
  and that is recorded as a finding. A choice recorded after the stamp is not
  expected: `t216` runs first.
- **Real throttling cannot be forced.** Step 1 shows the reader and the
  process on a quiet card; whether a hardware slowdown or the power brake is
  seen is validated only if the card happens to throttle. The sampler's flag
  rule is tested on a scripted card, not an L4.
- **The sampler is a second process on an 8-vCPU host.** `t213` measures what
  it costs each arm (the two no-sampler passes bracket the sampler pass).
- **XAttention runs its torch path on the L4**, so `t212` replays its masks
  from the torch estimator; the A100's Triton path is not exercised here.
- **Batch 1, one prompt, length 8192** (`--length`). Other cells are not part
  of this session.
