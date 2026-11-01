# Launch card: Shakedown 4 (one L4 session, three runs)

Prepared 1 Nov 2026, revised the same day for the researcher's five rulings
of 1 Nov 2026. The researcher's condition, that the CPU dry run of the T3.1
rehearsal (`t31_session.py --cpu-boundary`, steps 1 to 8 on the real arms
behind the stubbed GPU boundary) passes first, is met: all eight steps
passed in one run on `shakedown4-rulings-2026-11-01` (see "The CPU dry
run"). Deploys from `main` at the merge of `shakedown4-rulings-2026-11-01`
or later. Launchable only on the researcher's yes; nothing here runs on its
own. The step minutes are estimates from Shakedown 3.

## What runs

The scope the researcher set: T3.1 rehearsal steps 5 to 8 with the memset
zoo, compiled XAttention's density and its eager comparison, the replay,
T2.12 and T2.2. Steps 5 to 8 read steps 1 to 4's outputs from their own
session, so the rehearsal runs from step 1 (Shakedown 3: 12 minutes for
steps 1 to 4). Order C, A, B: if the halt cuts the session, it cuts B's
last step, not the rehearsal.

| Run | Command | Steps | What Shakedown 3 found, and what changed | Expected min |
|---|---|---|---|---|
| C | `t31_session.py --rehearsal` | T3.1 steps 1 to 8, `preflight` to `hold` | Step 5 refused dense's 1.0 as a density setting; dense now takes none. The batch-4 cells hold the dense control only (open, see below). Step 5 now nulls composite calls at their recorded chain and factory calls by their device argument, and writes each cell's `chain_check.json` (recorded, not gated, in a rehearsal). Step 6 checks the live masks on a graph of its own after one replay, runs the memset zoo, and on a mask failure writes `mask_diagnosis.json`. Results discarded, never banked | 40 |
| A | `t227_shakedown.py --keep-going --only replay` | `replay` | All-False live masks in all 28 layers: read before any replay. Now read after one replay of a check graph, from buffers marked "never written", and refused where one is still unwritten; `mask_diagnosis.json` has the check graph's buffers after the capture alone and after one replay | 5 |
| B | `t228_steps.py run --keep-going --only density,t212,t22` | `density` | The compiled wrapper claimed a density record its arm (mean-pool) lacks; it now forwards one only where the arm keeps one. Compiled XAttention's density is its compiled twin's, with the eager comparison | 10 |
| | | `t212` | Mean-pool's source density record was empty, so the check compared nothing. Both sides' executed density is now the kernel-input recorder's, and a comparison over no layers is refused | 8 |
| | | `t22` | Refused at the embedding's gather outside the layers, then (CPU) at `aten::arange`. Gather by the isolated replay; composites by chain, factories by device argument, each flagged; the chain check is in the findings | 10 |

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
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu   # passed, 1 Nov 2026
python3 scripts/t228_steps.py density --cpu-boundary --out /tmp/s4_density --length 1024
python3 scripts/t228_steps.py t212 --cpu-boundary --out /tmp/s4_t212 --length 1024 --reps 2
python3 scripts/t228_steps.py t22 --cpu-boundary --out /tmp/s4_t22 --length 1024 --profile-reps 1
python3 scripts/t227_steps.py replay --cpu-boundary --out /tmp/s4_replay --length 1024
```

## The CPU dry run (1 Nov 2026)

On the Mac, the real arms behind `attnbench/cpu_boundary.py`, transformers
4.46.0, x-attention at `XATTN_COMMIT`: all eight rehearsal steps passed in
one run (15 cells: five arms at 2048 and 4096, batch 2 dense only at 4096).
Fixed on the way, each with a test on the real classes:

- the null pass: composites met at their chain, `arange` by its device
  argument, SDPA's in-place `add_` met out of place under the pass's own
  dispatch mode (flagged `out_of_place`), and `aten::promote_types` (no
  `device` argument) not taken for a factory;
- the boundary itself: the trace shim gave a launch to `promote_types`,
  `detach_` and wholly CompositeImplicitAutograd leaves (`dropout` at p=0),
  and the tape took XAttention's `.to("cuda")` on a device tensor for a copy
  to the host; the replay now runs on a graph stand-in whose capture writes
  nothing (`GraphLikeBackend`), which is how Shakedown 3's unwritten masks
  would have shown on the CPU;
- the mask check's copies, inside the timed graph at first, were graph
  nodes with no position: the check has a graph of its own;
- the CPU config's prediction prefix lacked `-rehearsal` (step 7).

The chain check reads every composite call (33 to 211 per cell) and fails
everywhere on the CPU, as expected: a CPU kernel runs inside its call, so the
direct time holds the compute. It is recorded there, not gated.

XAttention's extension test fails on the CPU: layer 0 holds 234, 294 and
287 operations at 20480, 28672 and 32768 (3, 4 and 4 chunks), base 54 and
body 60 from the first two, so 294 predicted at 32768. The 7 missing are a
partial last chunk's padding (`aten::pad`; 32768 is four whole chunks of
8192), and the chunks do not repeat in one order. The no-assert copy: base
65, body 50, 258 recorded against 265. Both take the fallback, the
full-length recording per grid cell.

## Known before launch

- **The chain check's tolerance is a working guess** (T4.1 row 32: 0.25 of
  the direct time plus 5 us per call). Run C records it per cell and goes
  on; `chain_check.json` is what sets it. No GPU figure exists yet.
- **Batch 4 holds dense only, and the plan says otherwise.** Section 5.4
  ("Batch 1 and 4 at 16384 on the L4"), H6 and Section 5.9 choose a sparse
  arm per batch. The researcher's default holds the rehearsal; the scope
  line waits for a decision.
- **XAttention's structure is its full-length recording** at every grid
  cell unless the L4's probe passes the extension test the CPU's fails.
- **The memset zoo and the bound's operations are unrun** on a GPU: a CPU
  trace has no memset without a device record, so step 6 never reached them
  in the dry run.
- **The check graph doubles step 6's captures**: one extra capture and
  replay per sparse cell, its graph dropped before the timed one.
- **The density figures are a rehearsal's.** Working, pre-lock values.
