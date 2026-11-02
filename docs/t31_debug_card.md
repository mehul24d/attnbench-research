# Debug-session card: T3.1 steps 5 to 11 on staged cells

Prepared 2 Nov 2026 from the researcher's rulings after the fourth session.
**Its data is discarded.** Every output says "debug, not T3.1 data", and
nothing here is banked, cited or ledgered as T3.1. Launchable only on the
researcher's yes; nothing here runs on its own.

## Why

Two sessions stopped at `oracle_d0.5_L32768`, and each relaunch would spend
about 90 minutes in step 5 before reaching steps 6 to 11, which have never run
on a GPU. The debug session stages the fourth session's step-5 cells from the
bucket and runs the rest once with `--keep-going`, to find every failure in
steps 6 to 11 at once. The T3.1 session then runs end to end on fixed code
(ruling 3); only that run's data counts.

## What it does

- **Staged:** steps 1 to 4 and the 31 passed step-5 cells of session
  `t31_p1_20261102T1208Z` (commit `5cb47ac`), each recording checked against
  the hashes its `recording.json` pins; the donor's event logs
  (`isolated_host.events.jsonl`, 650 MB per large cell, read by no step) are
  left out. `staged_inputs.json` names the donor (session, code identity,
  instance, config and manifest hashes) and each cell's entry hash; a cell
  resumes only if its entry is the one staged, and no cell outside the list
  resumes under the donor's identity.
- **Not measured:** `oracle_d0.5_L32768`, "not measured: host memory limit of
  P1's pinned machine" (the ruling). No step prices, times or predicts it.
- **Four cells the donor never ran** (mean-pool 0.25 and 0.5, XAttention and
  XAttention without the assert, all at 32768): **a choice for the
  researcher.** The ruling says to start at step 6. Started at step 6 those
  four have no host cell, so they would be not measured too and step 6 and the
  ground truth would never see XAttention at 32768 (chunked at 16384, the
  least-tested path). Started at step 5 (`--start host_components`, the
  default here) step 5 runs just those four, about 10 minutes, and the cap
  grows by 15. Recommended: start at step 5.
- **Labelled:** `result.json` of every step carries `"debug": "debug, not T3.1
  data"`, `manifest.json` and `DEBUG_NOT_T31_DATA.txt` say so, and the
  prediction goes under `predictions/P1-debug/<session>` (a debug session
  cannot write under P1's prefix, nor P1 under a debug one).
- **`--keep-going`:** a failure stops nothing that does not need it
  (`t31_debug.DEPENDS`): predict needs step 6, hold needs predict, ground
  truth needs hold, verify needs both, the postflight needs nothing. A
  blocked step is recorded "blocked", not skipped silently.
- **Cap:** the runner starts no step after 1.5 x the projected duration of
  the steps it runs (`t31_projection.debug_cap_min`): **176 min** from step 5,
  161 from step 6 (steps 5 to 11 central 117 min; 6 to 11 107 min). The
  in-guest halt is the backstop: `GCP_HALT_MINUTES=210` (setup and staging 16
  min + the cap 176 + a step that starts just inside it).

## Time and cost (a projection, ₹108.64/h)

| | Minutes | ₹ |
|---|---|---|
| Setup: launch, deploy, preflight, bootstrap (Shakedown 6: 6) | 6 | 11 |
| Fetch the donor (6.6 GB from the bucket, a minute or two), verify, stage | 10 | 18 |
| Steps 5 to 11, central (the cap is 1.5 x this) | 117 | 212 |
| Teardown | 8 | 14 |
| **Expected, central** | **141** | **255** |
| At the cap (176 min of steps) | 200 | 362 |
| In-guest halt (`GCP_HALT_MINUTES=210`): GPU billing stops | 210 | 380 |

The halt ends GPU billing at ₹380 whatever the runner is doing; the cap
ends the session before it unless a step started inside the cap runs long.
Spend so far about ₹2,050 of ₹14,000.
Launch window and credit as the T3.1 card's.

## The commands

On the Mac, at the launch commit, clean tree (the launch is the same as the
T3.1 card's but for the halt):

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=210 GCP_MAX_RUN=3h45m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-dbg-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$NAME"   # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
D=t31_p1_20261102T1208Z; S=results/t31_dbg_$(date -u +%Y%m%dT%H%MZ)
python3 scripts/t31_debug.py fetch --instance attnbench-l4-t31-20261102-1732 --donor-session $D --dest ~/donor/$D
python3 scripts/t31_debug.py stage --donor ~/donor/$D --session $S --start host_components
python3 scripts/t31_session.py --debug --keep-going --session $S
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t31_dbg_s1   # not under t31_p1*
bash scripts/gcp_cleanup_check.sh
```

## Step 6 on the large oracle cells

The debug session's step 6 prices `oracle_d0.25_L32768` from the staged host
cell. Its manifest peak says whether step 6's own full reads (the original
operators' replay trace where the graph does not match, the recording's trace
where memsets are flagged) fit: step 6 for `oracle_d0.5_L32768` would be about
twice that. That is the only step-6 evidence for the large cell before the
T3.1 session.

**An option for the researcher:** run `oracle_d0.5_L32768` in step 5 of this
session too (leave it out of `not_measured`), which measures the segmented
replay's real peak on the card before the T3.1 session depends on it. About
35 to 60 minutes and ₹65 to ₹110 more, and the cap grows by the same. The
ruling marks the cell not measured; this runs only on the researcher's word.

## What it settles, and what it does not

It settles whether steps 6 to 11 run on the card, and every defect in them
that one pass finds: a step that runs and fails is reported with its log, a
step that could not run says which input failed. It does not settle step 5 at
`oracle_d0.5_L32768` (the memory fix, due 5 Nov, is checked offline), and a
pass here is not T3.1 data: the T3.1 session runs again from step 1.

Before the launch, on the Mac (passed 2 Nov 2026): the CPU check
`python3 scripts/t31_debug.py stage --donor <CPU-boundary rehearsal> --cpu-donor` and
`t31_session.py --debug --cpu-boundary --keep-going` (steps 5 to 8, 15 cells
staged and resumed, 15 device cells priced, prediction under
`predictions/P1-debug`), and `--debug --dry-run` (steps 6 to 11 on fakes).
