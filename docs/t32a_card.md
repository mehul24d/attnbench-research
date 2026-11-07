# T3.2a card: the first P1 repeat, eager group only

Prepared 7 Nov 2026 from the researcher's rulings 1 to 5 of 7 Nov 2026,
after the exploratory session. Launchable only on the researcher's yes;
nothing here runs on its own.

**What its data is.** T3.2a is real data: the first of the two T3.2 repeats
that give repeatability, the Gate A inputs and the floors. It is not a debug
or exploratory session: no output carries a debug label, and the prediction
goes under `predictions/P1-T3.2a/<session>`. **Its code identity is frozen
before launch** (ruling 4): the session records the commit of its first start
(`manifest.json`, `frozen_commit`) and refuses to run at any other commit, so
no fix is made within it. A failure ends the session; its fix is the next
session's code.

## What it runs

- **Phase 0, the oracle replay rerun** (ruling 2), on the same instance and
  the same commit, before T3.2a starts. The exploratory session's step 6
  again, from its own step 5 in the bucket: its nine oracle cells rerun, the
  other 26 cells resume (they passed under the same libraries and instance
  spec). It tests the step-6 fix below on the three cells that left 16
  positions unpriced. **T3.2a starts only if the oracle passes the pricing
  gate here.** If it does not, the session stops, the instance is torn down,
  and the fallback of ruling 2 (imputation, flagged, at most 1% of the arm's
  priced device time) is built as the next code identity before T3.2a is
  launched again.
- **T3.2a**: steps 1 to 11 of T3.1 from step 1 on the P1 grid and the working
  values, `oracle_d0.5_L32768` not measured (the drop fallback stays in force,
  ruling 3 of 6 Nov 2026), and **the eager ground-truth group only**: the
  compiled groups are set to run nothing (`t31_session.py --repeat a --groups
  eager`). The compiled arms wait for their new checks to run on a GPU.

## What changed since the exploratory session

- **Step 6 matches the isolated original segment by segment** (ruling 2).
  The roll is made in each decoder layer's forward pre-hook, the hook that
  also labels the recording's layers, so a cut was already at a layer
  boundary; the replay now refuses any roll that is not
  (`_Rolled.check_cuts`). What was wrong was the match: the layers are one
  structure repeated, so a match of the whole stream reports a layer's worth
  of nodes lost or reordered anywhere as the last layers' positions. Each of
  the 16 unpriced positions is the `masked_fill_` of one layer
  (`row_blockmask`) and the three `nonzero` kernels of the next
  (`replace_ones_with_count`), at layers 25 to 27. Session 3's recordings at
  the same lengths reproduce that pattern only for a loss of 60 to 79 nodes
  straddling a layer boundary, wherever in the prefill it happened; the pulled
  manifests carry no segment bounds, so where it happened is not known. Now
  each segment's positions are compared only with the nodes of its own
  profiler session (`replay.match_segmented`), so no stretch runs across a
  cut, and `06_device_components/result.json` (pulled with the logs) names
  each segment with a loss: its layers, its unmatched positions, and its
  launches that no device activity answered (`isolated_segments`).
- **The probe stays out of compiled regions; compiled arms get two untimed
  checks** (ruling 1), not run in T3.2a: the Kineto twin check and the
  two-prompt mask dependence check (`attnbench/compiled_checks.py`).
- **The block drift bound** is T4.1's row 36, working 3% (ruling 3); P1 does
  not run in blocks.

## Time and cost (a projection from the exploratory session, ₹108.64/h)

| | Low | Central | High | Basis |
|---|---|---|---|---|
| Setup: launch, deploy, preflight, write gate, bootstrap | 6 | 6 | 8 | the exploratory session |
| Phase 0: fetch the exploratory session | 3 | 5 | 10 | not measured |
| Phase 0: step 6's nine oracle cells, 26 resumed | 31 | 33 | 40 | the exploratory session's oracle cells: 31.2 min |
| Steps 1 to 4 | 21 | 22 | 24 | measured: 21.4 min |
| Step 5, 35 cells | 74 | 77 | 89 | measured: 77.3 min |
| Step 6, 35 cells | 44 | 46 | 53 | measured: 43.8 min, plus the per-segment report's pass over each isolated stream |
| Steps 7 and 8 | 3 | 3 | 3 | measured: 2.9 min |
| Step 9, the eager group | 10 | 11 | 16 | measured: about 11 min (its last cell started 6.3 min in) |
| Steps 10 and 11 | 2 | 2 | 3 | measured: postflight 1.3 min |
| Teardown | 8 | 8 | 8 | |
| **Total, minutes** | **202** | **213** | **254** | |
| **₹** | **366** | **386** | **460** | |

| Limit | Minutes from boot | ₹ |
|---|---|---|
| In-guest halt (`GCP_HALT_MINUTES=277`): 1.3 x the central projection | 277 | 502 |
| Hard delete (`GCP_MAX_RUN=4h47m`, DELETE), the halt + 10 min | 287 | **520, the cap** |

The halt follows the Spot cards' rule of 7 Nov 2026 (1.3 x the planned
time), which leaves the high projection 23 minutes inside it. A phase 0 that
stops the session costs about 52 minutes, ₹94. Spend before the session
₹2,956 of ₹14,000 (`docs/spend_ledger.md`); with it at most ₹3,476.

## The commands

On the Mac, at the merged `main` commit, clean tree (the commit T3.2a is
frozen at):

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=277 GCP_MAX_RUN=4h47m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t32a-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$NAME"   # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`. Phase 0:

```bash
EXP=t31_exp_20261107T0953Z; EXPHOST=attnbench-l4-exp-20261107-1518
python3 scripts/t31_debug.py fetch --instance $EXPHOST --donor-session $EXP --dest results/$EXP
rm results/$EXP/06_device_components/oracle_*/cell_manifest.json      # these nine rerun
bash scripts/run_phase.sh t31_06_oracle_rerun -- python3 scripts/t31_steps.py device_components \
    --out results/$EXP/06_device_components --session results/$EXP \
    --config results/$EXP/debug_config.json
python3 - <<'EOF'
import json
d = "results/t31_exp_20261107T0953Z/06_device_components"
print(json.load(open(f"{d}/pricing_gate.json"))["oracle"])
print(json.load(open(f"{d}/result.json"))["findings"]["isolated_segments"])
EOF
```

If the oracle is validated, T3.2a:

```bash
python3 scripts/t31_session.py --repeat a --groups eager
```

A resume after an instance problem, at the same commit only:

```bash
python3 scripts/t31_session.py --repeat a --groups eager --session results/t32a_p1_<time> --from <step>
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t32a_p1_<time>   # logs and manifests only
bash scripts/gcp_cleanup_check.sh
```

The bulk results stay in the bucket under `<instance>/results/`.

## What it settles, and what it does not

Phase 0 settles whether the oracle is priced on the exploratory session's
cells with the segment match, and if not, which segments lost what. T3.2a
gives the first repeat of record for the eager group: every step on one code
identity, its prediction committed before its ground truth. It does not give
the compiled rows, the second repeat (T3.2b), or the floors, which need both
repeats.

Before the launch, on the Mac: the suite at the merged commit, and, on the
researcher's word, the CPU-boundary rehearsal (`t31_session.py
--cpu-boundary`, steps 1 to 8 on the real arm classes; 11.7 minutes for the
exploratory card). On the CPU every graph replay matches the whole sequence,
so the rehearsal does not reach the isolated original or its segment match;
phase 0 is their first run on a GPU.
