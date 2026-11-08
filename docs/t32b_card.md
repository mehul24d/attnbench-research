# T3.2b card: the second P1 repeat, eager group only

Prepared 8 Nov 2026 from the researcher's ruling 1 of 8 Nov 2026, after
T3.2a. Launchable only on the researcher's yes; nothing here runs on its own.

**What its data is.** T3.2b is real data: the second of the two T3.2
repeats. With T3.2a (`results/t32a_p1_20261107T1640Z`) it gives the
between-session spread (the noise floor, Section 5.6), the resolution floor's
spread and the Gate A inputs of record. It runs **the same commit as T3.2a,
`3f50aab`**, the merged `main` of 7 Nov 2026 (PR #50), so the two repeats
share one code identity. The session records the commit of its first start
(`manifest.json`, `frozen_commit`) and refuses to run at any other commit, so
no fix is made within it. A failure ends the session; its fix is the next
session's code. The prediction goes under `predictions/P1-T3.2b/<session>`.

## What it runs

Steps 1 to 11 of T3.1 from step 1, on the P1 grid and the working values,
exactly as T3.2a ran them:

- `oracle_d0.5_L32768` not measured (the drop fallback, ruling 3 of 6 Nov
  2026, in force for every pairing);
- **the eager ground-truth group only**; the compiled groups run nothing
  (`t31_session.py --repeat b --groups eager`);
- no phase 0: the oracle replay rerun is settled (T3.2a's phase 0 validated
  it, 0 positions unpriced).

Nothing from T3.2a is resumed: every step runs on this instance.

## What changed since T3.2a

Nothing the session runs. The branch that carries this card also changes
code that only the Mac's analysis reads: the baselines' sums are added left
to right on every Python (`baselines.plain_sum`), the order Python 3.10's
`sum` uses on the instance, and the Gate A tooling is new
(`attnbench/analysis/gate_a.py`, `scripts/gate_a.py`). The session's code is
`3f50aab` either way: it is launched from a clean clone at that commit, not
from this branch.

## Time and cost (T3.2a's measured durations x 1.3, ₹108.64/h)

T3.2a billed 227 minutes (`session_cost.txt`: boot 15:52:00Z, teardown
19:38:48Z). Its phase 0 ran from 16:03:54Z to 16:40:04Z (36.2 minutes, from
`~/t32a.log`), which T3.2b does not run.

| | Minutes, measured in T3.2a | Basis |
|---|---|---|
| Setup: launch, deploy, preflight, write gate, bootstrap | 11.9 | boot to phase 0's start |
| Steps 1 to 4 | 19.7 | step results: 0.7, 2.8, 3.2, 13.0 |
| Step 5, 35 cells | 77.2 | step result |
| Step 6, 35 cells | 45.1 | step result |
| Steps 7 and 8 | 0.4 | step results (the hold's wait is in the gap below) |
| Step 9, the eager group | 12.5 | step result |
| Steps 10 and 11 | 0.3 | step results |
| Between steps: syncs, the hold's wait, process starts | 19.0 | the session's wall time (16:40:04Z to 19:34:16Z, 174.2 min) less its steps' 155.2 min |
| Teardown, to the cost file's stamp | 4.5 | 19:34:16Z to 19:38:48Z |
| **Total, minutes** | **190.6** | the rows; 227 billed less phase 0's 36.2 is 190.8, the cost file rounding to the minute |
| **₹** | **345** | |

| Limit | Minutes from boot | ₹ |
|---|---|---|
| In-guest halt (`GCP_HALT_MINUTES=247`): 1.3 x 190.6 = 247.8, rounded down | 247 | 447 |
| Hard delete (`GCP_MAX_RUN=4h17m`, DELETE), the halt + 10 min | 257 | **466, the cap** |

This is T3.2a's card's rule (halt 1.3 x the planned time, the hard delete
10 minutes after, the cap the cost at the hard delete), with T3.2a's
measured time in place of a projection. T3.2a's own step 5 and step 6 took
77.2 and 45.1 minutes against the exploratory session's 77.3 and 43.8, so
the measured time is a close estimate; the halt leaves 56 minutes of
margin. Spend before the session ₹3,367 of ₹14,000 (`docs/spend_ledger.md`);
with it at most ₹3,833.

## The commands

On the Mac, in a fresh clone at `3f50aab` (detached, clean tree), never this
branch:

```bash
git clone <repository> t32b && cd t32b && git checkout --detach 3f50aab
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=247 GCP_MAX_RUN=4h17m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t32b-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$NAME"   # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
python3 scripts/t31_session.py --repeat b --groups eager
```

A resume after an instance problem, at the same commit only:

```bash
python3 scripts/t31_session.py --repeat b --groups eager --session results/t32b_p1_<time> --from <step>
```

Back on the Mac:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t32b_p1_<time>   # logs and manifests only
bash scripts/gcp_cleanup_check.sh
```

The bulk results stay in the bucket under `<instance>/results/`.

## Incident (8 Nov 2026)

No teardown ran after the session ended (about 23:43Z): the Mac's watcher had expired while a question was pending, so the instance idled to its in-guest halt (01:04Z) and hard delete (01:12Z), about 80 minutes and ₹142 avoidable, and `~/t32b.log` went with the disk; every step had synced, and the P2 and P4 runner now ends every session itself, synced and then halted (`scripts/session_exit.sh`).

## What it settles, and what it does not

T3.2b gives the second repeat of record for the eager group, on T3.2a's code
identity, its prediction committed before its ground truth. With T3.2a it
gives the between-session spread of every cell, from which T4.1 sets the
noise and resolution floors, and the two-repeat Gate A inputs
(`scripts/gate_a.py`, dry run on T3.2a alone: `docs/t32a_gate_a_dry_run.md`).
It does not give the compiled rows, which wait for their checks to run on a
GPU, or Gate A's delay-injection slopes: T3.1's steps include no delay
injection (T2.25), so that leg is not measured by either repeat.

Before the launch, on the Mac: the suite at `3f50aab` passed 5981 and
skipped 155 in a fresh clone before T3.2a (7 Nov 2026), and the CPU-boundary
rehearsal passed steps 1 to 8 in 9.0 minutes. The code is the same, so
neither is needed again unless the researcher asks.
