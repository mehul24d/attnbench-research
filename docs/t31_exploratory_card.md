# Exploratory-session card: T3.1 live, steps 1 to 11

Prepared 7 Nov 2026 from the researcher's ruling 4 of 6 Nov 2026. Launchable
only on the researcher's yes; nothing here runs on its own.

**What its data is.** Every output says "T3.1 exploratory, code identity may
change" (`result.json` of every step, `manifest.json`,
`T31_EXPLORATORY.txt`), and the prediction goes under
`predictions/P1-exploratory/<session>`. Fixing on the Mac, redeploying and
resuming on the instance is allowed, so the cells of one step can carry
different commits (each cell manifest names its own). Repeatability, the
Gate A inputs and the floors come from the two T3.2 repeats, run afterwards
on the final code identity; if T3.2 slips past 14 Nov, it is cut to one
repeat (the cut list).

## Why

Steps 8 to 10 (the hold, the ground truth, the in-session late check) have
never run on a GPU, and steps 6 and 7 have run only in the debug session, on
staged cells. This session runs T3.1 from step 1 on the current code, and
whatever fails is fixed and resumed while the instance is up, as the debug
session did.

## What is new since the debug session (2 Nov 2026)

- **Pricing by structure** (ruling 1 of 6 Nov 2026): step 6 matches the
  graph's nodes to positions by class, stream, order and payload class, and
  labels every position (`graph`, `isolated_original`, `variant_differs`;
  dropped nodes `extra_node_dropped`). Per arm, the share of priced device
  time taken from the isolated original must be at most 5% (T4.1 row 35); an
  arm above it is not priced, is labelled "pricing not validated" in
  `06_device_components/pricing_gate.json` and the committed file's
  `not_priced`, and is left out of the hypothesis claims. Step 7 writes
  `isolated_share.json`, each arm's share of prediction time priced from the
  isolated replay. On the debug session's 2048-token cells the share is 0% to
  0.99% (dense, oracle, mean-pool, XAttention).
- **The isolated original as a compact stream** (ruling 3): segmented every
  two layers and read lazily. Offline, on session 3's banked 3.29 GB
  `oracle_d0.25_L32768` trace (the same replay of the original operators):
  compaction 42.8 s at 0.27 GB peak RSS, the lazy read and structural match
  23.2 s at 0.33 GB, every one of 2,092 device positions matched. The
  debug session's full read of the same kind of file was killed above 26 GiB.
  The GPU-side profiler peak of the segmented replay is still unmeasured.
- **`copy_from_host_pageable` and the other added mechanisms** measured in
  step 2 (ruling 2, PR #46).
- **The teardown keeps the results in the bucket** (6 Nov 2026): the
  instance syncs once more (`run_phase.sh --sync-only`), and only its logs
  and manifests come back, as one archive, after a disk check. A sync that
  fails while SSH works refuses the delete.

## What it runs

- The P1 grid from `configs/t31/p1_session.json` and the working values,
  **without `oracle_d0.5_L32768`**: the drop fallback is in force (ruling 3:
  the 5 Nov deadline was missed). `oracle_d0.25_L32768` runs.
- **An option for the researcher:** measure `oracle_d0.5_L32768` too
  (`explore --include-stuck-cell`). Its step 5 and step 6 now run segmented
  and compacted, never on a GPU so far; about 35 to 60 minutes and ₹65 to
  ₹110 more, which the ₹500 cap does not cover at the central projection.
  Recommended: not in this session.
- `--keep-going`: a step whose inputs did not fail runs after a failure; a
  failure in steps 1 to 5 blocks every step after it but the postflight
  (`t31_debug.DEPENDS`).
- **Cap:** no step starts after 230 minutes of the session
  (`explore --cap-min 230`). The in-guest halt and the hard delete hold the
  money (below).

## Time and cost (a projection, ₹108.64/h)

| | Low | Central | High | Basis |
|---|---|---|---|---|
| Setup: launch, deploy, preflight, bootstrap | 6 | 6 | 6 | Shakedown 6 |
| Steps 1 to 4 | 22 | 22 | 22 | session 3, measured |
| Step 5, 35 cells | 85 | 95 | 110 | session 3: 31 cells in 69 min (its 93-minute step less the 24 minutes in the killed cell), plus four 32768 cells and a process per cell |
| Step 6, 35 cells | 30 | 35 | 50 | the debug session's cell manifests: 34 cells in 25.3 min, plus `oracle_d0.25_L32768` |
| Steps 7 and 8 | 2.5 | 2.5 | 2.5 | the hold is 90 s |
| Step 9, ground truth | 53 | 74 | 120 | `t31_projection.py`, less `oracle_d0.5_L32768` |
| Steps 10 and 11 | 3 | 3 | 3 | |
| Teardown | 8 | 8 | 8 | |
| **Total, minutes** | **210** | **246** | **322** | |
| **₹** | **380** | **445** | **583** | |

| Limit | Minutes from boot | ₹ |
|---|---|---|
| No step starts after (`--cap-min 230`, from the session's start) | about 236 | 427 |
| In-guest halt (`GCP_HALT_MINUTES=265`): GPU billing stops | 265 | 480 |
| Hard delete (`GCP_MAX_RUN=4h35m`, DELETE), the halt + 10 min | 275 | **498, the cap** |

The cap is ₹500 (ruling 4). At the central projection the session ends at
about 246 minutes with 19 minutes to the halt; one fix and resume costs
about 10 to 20 minutes of it. At the high end the halt stops step 9 partway:
ground truth started near minute 190 and needs about 120. Each ground-truth
group syncs when its process ends, so what a finished group wrote is kept.
Spend before the session about ₹2,630 of ₹14,000; with it at most ₹3,130.

## The commands

On the Mac, at the merged `main` commit, clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=265 GCP_MAX_RUN=4h35m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-exp-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$NAME"   # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
S=results/t31_exp_$(date -u +%Y%m%dT%H%MZ)
python3 scripts/t31_debug.py explore --session $S --cap-min 230
python3 scripts/t31_session.py --debug --keep-going --session $S
```

After a fix (committed on the Mac, redeployed with `gcp_deploy_source.sh`):

```bash
python3 scripts/t31_session.py --debug --keep-going --session $S --from <step>
```

A resume from `predict` or earlier is refused once ground truth has started.

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t31_exp_s1   # logs and manifests only
bash scripts/gcp_cleanup_check.sh
```

The bulk results stay in the bucket under `<instance>/results/`.

## What it settles, and what it does not

It settles whether steps 1 to 11 run end to end on the card and the current
code, every defect one pass finds, the pricing gate's share per arm on the
whole grid, and step 6's peak at `oracle_d0.25_L32768` with the compact read.
It does not give T3.1's data of record: that is T3.2a and T3.2b on the final
code identity.

Before the launch, on the Mac: the suite, and the CPU-boundary rehearsal
(`t31_session.py --cpu-boundary`, steps 1 to 8 on the real arm classes).
**Not yet run for this card** (7 Nov 2026): the rehearsal needs the
x-attention clone at the pinned commit, and running it was held for the
researcher.
