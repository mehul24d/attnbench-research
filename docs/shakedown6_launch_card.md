# Launch card: Shakedown 6 (one L4 session, three runs)

Prepared 1 Nov 2026 for the researcher's rulings of 1 Nov 2026 after
Shakedown 5. Deploys from `main` at the merge of
`shakedown5-fixes-2026-11-01` or later. Launchable only on the researcher's
yes; nothing here runs on its own. Step minutes are estimates from
Shakedowns 4 and 5; steps 5 to 8 of the rehearsal have not yet run on a GPU.

## What runs

The scope the researcher set covers three things:
- T3.1 rehearsal steps 4 to 8;
- compiled density;
- the pricing zoo: composites, `index_put_`, SDPA flash, and every arm,
  the oracle included.

Steps 4 to 8 read steps 1 to 3's outputs from their own session, so the
rehearsal runs from step 1 (about 5 minutes in Shakedown 5). The order is
C, B, Z: if the halt cuts the session, it cuts the zoo's last arms, and the
oracle is the zoo's second arm.

| Run | Command | Steps | What Shakedown 5 found, and what changed | Expected min |
|---|---|---|---|---|
| C | `t31_session.py --rehearsal` | T3.1 steps 1 to 8, `preflight` to `hold` | Step 4 stopped: the launch check refused 84 of 33,772 calls in XAttention's 20480-token probe, a boolean `index_put_` that launches beside its children, which the stand-in now predicts; the check took 4 minutes per trace and now takes seconds. Step 5 now prices every call with the bare timer (below), the null pass beside it as the instrumented comparison; the chain gate is evaluated on the bare prices and a cell that fails it is labelled with its fallback, never stopped. Results are discarded, never banked | 44 |
| B | `t228_steps.py run --keep-going --only density` | `density` | Compiled XAttention failed under `compiled_default` twice: a cached threshold row built inside the graph, now built outside it; and a resume frame in the kernel interface that recompiled per call, now outside Dynamo. Each compiled arm (mean-pool and XAttention, both settings) now runs 3 warm-up prefills, then 12 more under the recompile gate (0 compile events after warm-up), then its density | 6 |
| Z | `t228_steps.py run --keep-going --only pricing_zoo` | `pricing_zoo` (new) | The bare timer on the card: SDPA forced to flash (causal), a boolean `index_put_` and a `_to_copy` at the model's shapes, each call with its own bare time and its children's; then dense, oracle, mean-pool, XAttention and XAttention without the assert at 8192, each recorded, null-passed and bare-priced, with its chain gate, fallback label and instrumented/bare ratio per operator. The oracle's row is its re-evaluation | 8 |

**The bare timer**, as built (`attnbench/components/bare_timer.py`). Each
call the null pass meets, leaf or composite, is re-issued alone at its
prefill inputs at least 1000 times, each after a synchronise outside the
timer, and priced at the median less the timer's own overhead. Prices are
cached by signature within a process, so a shape shared by arms is timed
once. Every repetition runs the call's device work, which is what the
minutes above pay for: at 16384 tokens about 4 minutes of the L4 per
batch-1 cell set.

**The rehearsal's batch above 1 is at 2048** (`REHEARSAL_GRID`, a default:
any values do for a rehearsal, researcher, 31 Oct 2026). At 16384 tokens
and batch 4 the bare timer would spend about 11 minutes on one dense cell,
which would put the session past its halt. T3.1's own batch-4 cell stays
at 16384.

**Proposed, not in the scope set:** T2.21's graph-break report
(`t227_shakedown.py --keep-going --only graph_breaks`, 3 minutes in
Shakedown 2), which now checks each compiled variant's breaks against its
declared code path (ruling, 1 Nov 2026). Its declared set for XAttention,
the kernel interface outside Dynamo, has been seen only on a CPU. It runs
only if the researcher adds it.

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
| Boot to the runs starting (Shakedown 5: 5 min) | 6 | 11 |
| Run C | 44 | 80 |
| Run B | 6 | 11 |
| Run Z | 8 | 14 |
| Teardown: results pull, delete, cleanup check | 8 | 14 |
| **Expected** | **72** | **130** |
| In-guest halt (`GCP_HALT_MINUTES=85`): GPU billing stops | 85 | 154 |
| Hard delete (`GCP_MAX_RUN=2h`) | 120 | disk only after the halt |

**Cost cap: ₹160.** That is the halt's ₹154 plus the disk until the hard
delete. Step 5's minutes are the least certain: it has never run on a GPU,
and each cell now takes a recording, an isolated replay, five null passes
and a bare pass. If the halt cuts the session, it cuts the zoo's last arms.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=85 GCP_MAX_RUN=2h GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-s6-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

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
python3 scripts/t228_steps.py run --keep-going --only density
python3 scripts/t228_steps.py run --keep-going --only pricing_zoo
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/shakedown6_l4
bash scripts/gcp_cleanup_check.sh
```

## Before launch, on the Mac

The CPU dry run, which must pass. It needs x-attention cloned outside the
repo at `XATTN_COMMIT`, and transformers 4.46.0 (the L4's) in a venv over
the system torch:

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
python3 scripts/t228_steps.py density --cpu-boundary --out /tmp/s6_density --length 1024
python3 scripts/t228_steps.py pricing_zoo --cpu-boundary --out /tmp/s6_zoo --length 1024
```

## The CPU dry run (1 Nov 2026)

(Filled in after the dry run below completes.)

## Known before launch

- **The bare timer is new on the GPU.** Behind the CPU boundary a bare call
  runs its CPU compute, which the null pass skips, so the CPU's
  instrumented/bare ratios (0.02 to 0.4) and its gate shares mean nothing.
  On the L4 the ratio is the finding: Shakedown 4's null pass priced leaf
  calls at 2.1 to 2.5 times the recorded prefill's own host time.
- **The bare prices leave out the arm's Python between calls.** The null
  pass's gap held it; a call timed alone does not. The plan's c_host names
  dispatch, the library front end, the launch floor and host compute, not
  the arm's own Python, so the omission follows the definition, and the
  instrumented/bare ratio shows its size. Flagged to the researcher.
- **The chain gate now compares bare times**: a pure composite's children
  against the composite. A call that launches its own kernel beside its
  children (`addmm` beside its bias copy; the four operators of
  `cpu_boundary.LAUNCHES_BESIDE_CHILDREN`, and `_index_put_impl_`) has no
  chain that adds up to it and is priced as a leaf. Which calls are pure
  composites on the L4 is read from the recording.
- **The oracle is re-evaluated twice**: in run C's step 5 at 2048 and 16384
  (0.25 and 0.5), and in run Z at 8192. On Shakedown 4's cells it failed the
  instrumented check at 22 to 26%; that check is now recorded, not gated.
- **Compiled mean-pool under the default settings has the same
  `replace_ones_with_count` break** that made XAttention recompile per call.
  Shakedown 5 ran it for a few density prefills only. Run B's 12 prefills
  after warm-up are its first test under the recompile gate.
- **The H100 loop refuses Spot.** Under the Spot cap rule (hard delete at
  40 minutes or later, cap at most ₹350), 40 minutes at ₹629.94/h or
  ₹668.04/h cost ₹420 or ₹446. The loop runs Flex-start only until a rate of
  ₹525/h or less is on record.
- **The density figures are a rehearsal's**, from working, pre-lock values.
