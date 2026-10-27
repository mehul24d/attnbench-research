# Launch card: T2.27, the L4 shakedown

Prepared 27 Oct 2026. Not launched. The researcher launches it; nothing here
runs on its own.

## What runs

`scripts/t227_shakedown.py` from branch `t227-shakedown-2026-10-27`
(`gcp_deploy_source.sh` bundles the local HEAD and confirms the commit and a
clean tree on the instance). The branch is main `615a522`, which already
holds T2.4, T2.17, T2.18 and T2.21, plus the runner and this card.

The eleven steps, in order, stopping at the first failure, each with its own
log (`results/t227_shakedown_<UTC>/<NN>_<step>/step.log`) and each synced to
GCS as it ends (`run_phase.sh`, phase `t227_<NN>_<step>`):

| # | Step | Task | Expected minutes |
|---|---|---|---|
| 1 | `t21_t220` | T2.1 host-speed probe; T2.20 fields (g2-standard-8, STANDARD, L4, optimize 0, microcode, clean tree); executed density from the kernel's input | 2 |
| 2 | `zoo` | T2.3 mechanism zoo, one stream (each wait's answer, and which alternative held) and two (must be flagged) | 1 |
| 3 | `link` | T2.5 link row | 8 |
| 4 | `launch_queue` | T2.6 Q, null kernel and one-element add | 1 |
| 5 | `nsys` | T2.7: Kineto checks per arm; nsys present?; control `um_fault_control.py` and each arm under the same nsys options; `nsys-faults` per export | 12 (+10 if nsys must be installed) |
| 6 | `replay` | T2.4 on meanpool at 8192: record, CUDA graph, mask agreement, timing, match, replay-vs-trace, isolated replay, cache effect | 5 |
| 7 | `probe` | T2.18, each of the five arms: 2048 and 4096, extended to 8192, checked against 8192 | 12 |
| 8 | `assert_gate` | T2.17: x-attention checkout is `e379887` and clean; session check; bitwise gate on 5 real `qa_1` prompts at 8192 | 3 |
| 9 | `graph_breaks` | T2.21 report, 5 arms × 2 settings, each in its own process, all 28 layers | 25 (the least certain figure) |
| 10 | `null_kernel` | 64 null kernels under the profiler, one launch and one kernel each | 1 |
| 11 | `figure1` | oracle at 16384 (`--fig1-length`), `host_compute()` around the mask build, recorded twice, layer 14 cut out of each; pair must be the same sequence. The L4 half only: the A100 counterpart is not part of this session | 5 |

The arms: dense (`sdpa_flash`), oracle (block-sparse, reference builder on
the host, scores before the prefill), meanpool (block-sparse, inline
estimator, device builder), xattention (calibrated `authors` table, stride
8), xattention_noassert. Sparsity 0.75. Prompts are real RULER `qa_1`
examples sized by the Qwen tokenizer.

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` |
| Zone | `asia-northeast1-a` (the launcher's default and the plan's pin) |
| Provisioning | STANDARD (on-demand); the launcher tags it, T2.20 checks the tag |
| Image | `attnbench-env-v6-20260917` (torch 2.9.1+cu129, transformers 4.46.0, flash_attn 2.8.3). The launcher's default is still v5, so set `ATTNBENCH_IMAGE` |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention (v6 lacks the RULER data, wonderwords, nltk and the XAttention install) |
| Rate | ₹108.64/h, the all-in planning rate (Tokyo, 20 Oct 2026 catalogue, with the boot disk). Teardown needs `GCP_RATE_INR_HR=108.64` |
| Launch window | by 9 Dec 2026 (plan rule 14; `_launch_policy.sh` enforces it) |

## Time and cost

| | Hours | ₹ at 108.64/h |
|---|---|---|
| Setup: launch to write gate (v6: about 2 min), deploy, preflight, bootstrap with XAttention | 0.35 | 38 |
| The eleven steps (75 min) | 1.25 | 136 |
| Teardown: serial log, results pull (the traces are large), delete, cleanup check | 0.15 | 16 |
| **Expected, nothing fails** | **1.75** | **190** |
| One failure, fixed on the instance, resumed with `--from` | 2.5 | 272 |
| In-guest halt (`GCP_HALT_MINUTES=180`): GPU billing stops here | 3.0 | 326 |
| Hard delete (`GCP_MAX_RUN=3h30m`) | 3.5 | disk only after the halt |

The step minutes are estimates, not measurements: no step has run on an L4.
The graph-break report is the widest: XAttention's estimator is traced by
Dynamo once per layer per setting. If it runs long, rerun step 9 with
`--gb-layers 0,14,27` and record that the report covers three layers.

## The commands

On the Mac, from the repo root, with the branch checked out:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project (never the gcloud default)
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=180 GCP_MAX_RUN=3h30m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t227-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean"
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine   # v6 ships a results/ tree
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
python3 scripts/t227_shakedown.py
# after a failure and its fix:
python3 scripts/t227_shakedown.py --from STEP --session results/t227_shakedown_<UTC>
```

A fix made on the instance dirties the tree, and the runner then refuses
(every row must be banked with `git_dirty = False`). So a code fix is
committed on the Mac and redeployed (`gcp_deploy_source.sh`), then the
session resumes; an environment fix (installing nsys) needs no redeploy. The
failed attempt stays beside the new one (`<NN>_<step>.failed.<time>`).

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64   # during the session, any time
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t227_l4
bash scripts/gcp_cleanup_check.sh
```

## Known before launch

- **nsys on v6 is not on record.** If step 5 finds none it stops, by design.
  Install Nsight Systems on the instance, note the command and version in
  the step's log, and resume `--from nsys`. The plan asks for exactly this.
- **XAttention runs its torch path on the L4.** Upstream uses Triton only on
  a device whose name contains "100", so the T2.17 gate here checks the
  torch path, and every XAttention sequence is recorded as `xattn:torch`.
- **T2.4 is a dry run.** Positions that the tape substitutes (`nonzero` to
  `nonzero_static`) get no device cost, so an incomplete match is recorded,
  not failed; the masks must still agree bitwise. The check of record is
  P1's, in T3.1.
- **T2.18 at 8192 is extendable for XAttention** (one chunk at 2048, 4096
  and 8192); 32768 is not, and is not attempted.
- The plan's checklist also lists T2.13 and T2.16 dry runs and the T2.2
  profiler. This session runs the list as asked for on 27 Oct 2026:
  T2.2 is not built, so step 10 profiles the null kernel and step 11 profiles
  the reference mask builder on the L4's host.
- **Figure 1 is the L4 half.** Step 11 records the pair at 16384 tokens on
  the L4. The A100 counterpart (same arm, length and layer) is not part of
  this session and is not costed here.
- The dry run on a Mac CPU passes steps 1, 3, 4 and 9 (dense arm only) and
  skips the seven that need CUDA; their wiring is tested with signature-checked
  fakes (`tests/test_t227_shakedown.py`).
