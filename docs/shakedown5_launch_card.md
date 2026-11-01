# Launch card: Shakedown 5 (one L4 session, three runs), the last before T3.1

Prepared 1 Nov 2026 for the researcher's rulings of 1 Nov 2026 after
Shakedown 4. Deploys from `main` at the merge of
`shakedown5-rulings-2026-11-01` or later. Launchable only on the
researcher's yes; nothing here runs on its own. Step minutes are estimates
from Shakedowns 3 and 4; steps 5 to 8 have not yet run on a GPU.

**The rule after this session** (researcher, 1 Nov 2026). This is the last
shakedown before T3.1. Failures in steps 6 to 8 are fixed inline, and T3.1
goes ahead with its own preflight. There is no sixth shakedown unless a
blocker affects T3.1's ground truth.

## What runs

The scope the researcher set covers six things:
- T3.1 rehearsal steps 5 to 8, with the memset zoo;
- compiled XAttention's density with the output-masks A/B;
- the replay;
- T2.12;
- T2.2.

Steps 5 to 8 read the outputs of steps 1 to 4 from their own session, so
the rehearsal runs from step 1 (12 minutes in Shakedown 4). The order is C,
A, B: if the halt cuts the session, it cuts B's last step (`t22`), not the
rehearsal.

| Run | Command | Steps | What Shakedown 4 found, and what changed | Expected min |
|---|---|---|---|---|
| C | `t31_session.py --rehearsal` | T3.1 steps 1 to 8, `preflight` to `hold` | Step 5 stopped at XAttention 2048: the null pass nulled a `fill_` on an empty slice that CUDA never launches. Three changes. Empty calls launch nothing, and each cell's profile flags them. Calls are aligned by recorded call instance. Every recording on the GPU checks the CPU stand-in's predicted launches against its own trace and refuses a cell on any mismatch (`launch_check.json`). The chain check is gated in aggregate (working 10%), and a rehearsal records it without gating. Results are discarded, never banked | 45 |
| A | `t227_shakedown.py --keep-going --only replay` | `replay` | Passed in Shakedown 4. Rerun because recordings now carry call instances and shapes | 5 |
| B | `t228_steps.py run --keep-going --only density,mask_ab,t212,t22` | `density` | Both compiled groups failed: the read hit a CUDA-graph output a later replay had overwritten. Compiled XAttention now returns its mask from the compiled function. Its twin pass marks each step and clones the mask at once | 10 |
| | | `mask_ab` (new) | Compiled XAttention timed with and without the mask output, interleaved at 8192 (`--length`), 10 rounds each. Writes `mask_ab.json` with B − A and the spread. A finding for the researcher's ruling | 6 |
| | | `t212` | Passed in Shakedown 4; rerun on the new recordings | 8 |
| | | `t22` | Refused at mean-pool 8192: three `addmm` calls launched 2, 2 and 1 kernels, which an even split cannot divide. Now aligned by instance. The boolean `x[mask] = v` runs as written in the pass, without the tape's stand-in, which had inflated `index_put_` 2.0 to 2.6× | 10 |

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` |
| Zone | `asia-northeast1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917` (set `ATTNBENCH_IMAGE`; the launcher defaults to v5) |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention |
| Rate | ₹108.64/h all-in planning rate (`GCP_RATE_INR_HR=108.64`) |

While this VM exists, the H100 retry loop skips its rounds, because it allows
one GPU session at a time. It resumes by itself after the delete.

## Time and cost

| | Minutes | ₹ at 108.64/h |
|---|---|---|
| Boot to the runs starting (Shakedown 4: 5 min) | 6 | 11 |
| Run C | 45 | 81 |
| Run A | 5 | 9 |
| Run B | 34 | 62 |
| Teardown: results pull, delete, cleanup check | 8 | 14 |
| **Expected** | **98** | **177** |
| In-guest halt (`GCP_HALT_MINUTES=90`): GPU billing stops | 90 | 163 |
| Hard delete (`GCP_MAX_RUN=2h`) | 120 | disk only after the halt |

**Cost cap: ₹200.** That is the halt's ₹163 plus the disk until the hard
delete. The runs are expected to finish about 90 minutes after boot, so the
halt may cut `t22`, the last step of B. If it does, `t22`'s Shakedown 4
partial stands, and the rest is banked as each step ends.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=90 GCP_MAX_RUN=2h GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-s5-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

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
python3 scripts/t228_steps.py run --keep-going --only density,mask_ab,t212,t22
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/shakedown5_l4
bash scripts/gcp_cleanup_check.sh
```

## Before launch, on the Mac

The CPU dry run, which must pass. It needs x-attention cloned outside the
repo at `XATTN_COMMIT`, and transformers 4.46.0 (the L4's) in a venv over
the system torch:

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
python3 scripts/t228_steps.py mask-ab-one --cpu-boundary --out /tmp/s5_ab --length 1024 --reps 3 --warmup 1
```

## The CPU dry run (1 Nov 2026)

On the Mac, behind `attnbench/cpu_boundary.py` (transformers 4.46.0,
x-attention at `XATTN_COMMIT`), all eight rehearsal steps passed in one
run. It covered 15 cells: five arms at 2048 and 4096, plus dense at batch 2
at 4096. The `mask-ab-one` child also ran end to end; its CPU timings mean
nothing.

In step 5 each XAttention cell met its empty `fill_` as an empty call, at
layer 0, flagged. Two things were found and fixed on the way, each with a
test:
- The first run stopped at step 6 (oracle d0.5, 2048): the replay's kernel
  order differed from the recording's from layer 1 on. The stand-in placed
  a launch beside children (a `sort` after its copy) in the parent's widest
  uncovered gap, and which side of the children that gap fell on varied with
  timing. The launch now goes after the children.
- The stand-in now reads shapes. An `out=` call (`arange`) gets an empty
  output and resizes it, so a call with a `resize_` child is never empty.

The chain check is recorded, not gated, on the CPU: a CPU kernel runs inside
its call.

## Known before launch

- **The launch check is new on the GPU.** On Shakedown 4's banked L4 traces,
  read without shapes, it passes dense (4,231 calls) and mean-pool (11,847)
  with no mismatch. Oracle's remaining 168 mismatches sit in the oracle's
  host-side mask path, now one named function (`_mask_from_scores`) that
  the check leaves out; the banked traces predate the name. XAttention has
  no banked L4 recording. A mismatch refuses that cell in step 5 and names
  the operator, which is what the check is for.
- **The host-cost instrument's overhead is open** (researcher's ruling
  pending; Appendix A, 1 Nov 2026). On Shakedown 4's cells, the null pass
  priced leaf calls at 2.1 to 2.5 times the recorded prefill's own host time
  per call. The isolated replay's direct time is inflated as well, so the
  chain/direct ratio is the same for leaf calls as for composites. The
  overhead affects the predictions T3.1 tests, not its ground truth. This
  session measures nothing new on it.
- **The aggregate chain gate would refuse oracle.** On Shakedown 4's cells
  at 10%, dense passes (4.8%) and mean-pool is about 8.3% once the
  `index_put_` stand-in is removed. Oracle is at 22 to 26%, mostly the
  isolated replay's inflation of its host-side mask build. Run C records the
  check without gating; T3.1 would gate it.
- **XAttention's structure is its full-length recording at every grid
  cell.** On the L4 the extension test is short by one operation at 32700
  tokens, where the last chunk's tail slice is empty.
- **The memset zoo and the bound's operations have not run** on a GPU.
- **The density figures are a rehearsal's**, from working, pre-lock values.
