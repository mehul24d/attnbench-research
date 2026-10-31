# Launch card: Shakedown 4 (one L4 session, three runs)

Prepared 1 Nov 2026. **Not launchable yet.** The researcher's condition is
that the CPU dry run of the T3.1 rehearsal (`t31_session.py
--cpu-boundary`, steps 1 to 8 on the real arms behind the stubbed GPU
boundary) passes first, and it does not: it stops at step 5 (see "Known
before launch"). Deploys from `main` at the merge of
`shakedown4-prep-2026-10-31` or later. The researcher launches it; nothing
here runs on its own. The step minutes are estimates from Shakedown 3.

## What runs

The scope the researcher set: T3.1 rehearsal steps 5 to 8 with the memset
zoo, compiled XAttention's density and its eager comparison, the replay,
T2.12 and T2.2. Steps 5 to 8 read steps 1 to 4's outputs from their own
session, so the rehearsal runs from step 1 (Shakedown 3: 12 minutes for
steps 1 to 4). Order C, A, B: if the halt cuts the session, it cuts B's
last step, not the rehearsal.

| Run | Command | Steps | What Shakedown 3 found, and what changed | Expected min |
|---|---|---|---|---|
| C | `t31_session.py --rehearsal` | T3.1 steps 1 to 8, `preflight` to `hold` | Step 5 refused dense's 1.0 as a density setting; dense now takes none. The batch-4 cells hold the dense control only (the oracle and mean-pool refuse batch > 1, as XAttention does). The embedding's gather is priced by an isolated replay. Step 6 runs the memset zoo, and on a mask mismatch writes `mask_diagnosis.json`. Results discarded, never banked | 40 |
| A | `t227_shakedown.py --keep-going --only replay` | `replay` | All-False live masks in all 28 layers. Same gate; now also `mask_diagnosis.json`: live/live, live/bank, the tape without a graph, and the graph's masks before a replay, after one, and copied inside it | 5 |
| B | `t228_steps.py run --keep-going --only density,t212,t22` | `density` | The compiled wrapper claimed a density record its arm (mean-pool) lacks; it now forwards one only where the arm keeps one. Compiled XAttention's density is its compiled twin's, with the eager comparison | 10 |
| | | `t212` | Mean-pool's source density record is empty, so the check compared nothing. Same gate; `mask_diagnosis_<arm>.json` and the record lengths are now in the findings | 8 |
| | | `t22` | Refused at the embedding's gather outside the layers. Priced by the isolated replay with a cost-source flag (ruling, 31 Oct 2026). The next refusal is expected at `aten::arange` (see below) | 10 |

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
| Boot to the runs starting (Shakedown 3: 6 min) | 6 | 11 |
| Run C | 40 | 72 |
| Run A | 5 | 9 |
| Run B | 28 | 51 |
| Teardown: results pull, delete, cleanup check | 8 | 14 |
| **Expected** | **87** | **157** |
| In-guest halt (`GCP_HALT_MINUTES=90`): GPU billing stops | 90 | 163 |
| Hard delete (`GCP_MAX_RUN=2h`) | 120 | disk only after the halt |

**Cost cap: ₹200.** The halt's ₹163 plus the disk to the hard delete. Run C
is the estimate least known: steps 5 and 6 have never run on a GPU (15
cells, each recorded, profiled five times, captured, replayed and priced).
If the halt fires mid-run, the session ends there; every step is synced as
it ends.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=90 GCP_MAX_RUN=2h GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-s4-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean"
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
python3 scripts/t31_session.py --rehearsal              # results discarded
python3 scripts/t227_shakedown.py --keep-going --only replay
python3 scripts/t228_steps.py run --keep-going --only density,t212,t22
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/shakedown4_l4
bash scripts/gcp_cleanup_check.sh
```

## Before launch, on the Mac

The CPU dry run, which must pass (x-attention cloned outside the repo at
`XATTN_COMMIT`; transformers 4.46.0, the L4's, in a venv over the system
torch):

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
python3 scripts/t228_steps.py density --cpu-boundary --out /tmp/s4_density --length 1024
python3 scripts/t228_steps.py t212 --cpu-boundary --out /tmp/s4_t212 --length 1024 --reps 2
python3 scripts/t228_steps.py t22 --cpu-boundary --out /tmp/s4_t22 --length 1024 --profile-reps 1
python3 scripts/t227_steps.py replay --cpu-boundary --out /tmp/s4_replay --length 1024
```

## Known before launch

- **The CPU dry run stops at step 5.** The null pass (T2.2) refuses at the
  first launching call it does not see, as the ruling has it ("refuse all
  other unhandled ops"). Past the embedding that is `aten::arange` before
  the layers: a factory call with no tensor input, so the pass never sees
  it touch the device. Behind it are the composite calls, where the
  dispatcher shows the pass `aten::_to_copy`, `aten::constant_pad_nd`,
  `aten::masked_scatter` or `aten::clone` and the recording names their
  kernels' `copy_`, `fill_` or `masked_scatter_` (Shakedown 3's mean-pool
  recording has the same calls in every layer). The L4 would refuse where
  the CPU does. A ruling is owed before this card can launch.
- **The replay's mask check reads before any replay.** `step_replay` and
  T3.1's `device_components` call `mask_agreement` straight after
  `capture`. A CUDA graph's capture records kernels and runs none, so the
  live masks it holds were never written: Shakedown 3 read all-False masks
  in all 28 layers (7380 of 49152 kept in each bank layer, agreement
  1 - 7380/49152). On the CPU's eager path every comparison agrees. Run A's
  diagnosis reads the graph's masks before and after a replay; the gate is
  unchanged until it is ruled on.
- **T2.12's mean-pool density check compares an empty record.**
  `BlockSparseAttention` keeps no `last_layer_density`, so the source side
  has no layers. On the CPU the masks agree in every layer and the logits
  are bitwise equal. The gate is unchanged until it is ruled on.
- **The memset zoo and the bound's operations are unrun** on a GPU. They run
  in step 6 if a cell's sequence holds flagged memsets.
- **The density figures are a rehearsal's.** Working, pre-lock values.
