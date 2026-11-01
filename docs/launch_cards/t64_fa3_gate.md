# Launch card: the H100 FA3 correctness gate (no timing)

Prepared 31 Oct 2026. Attempt 1 (31 Oct, us-central1-a) was not granted.
From then on the gate is retried by `scripts/gcp_h100_gate_retry.sh` (below).
Not launched. The researcher launches it; nothing here
runs on its own. The gate comes before P4 (plan Section 8 calendar, 31 Oct
2026). Flex-start requests go in the gaps between L4 sessions, and every
attempt, granted or not, gets a row in `h100_flex_attempts.md`.

## What runs

`python3 scripts/t64_session.py run --gate-only --fa3-sha256 HEX` runs the
P4 runner in its gate mode, with four steps and nothing timed:

| # | Step | What it decides |
|---|---|---|
| 1 | `prereqs` | The deployed tree holds the FA3 files and `run_probe.py`. There are no plan-row requirements |
| 2 | `fa3_import` | `fa3_import_check.sh` checks the wheel's sha256 and the Python and torch tags, installs without dependencies, imports, and confirms the `fa3` backend is available on sm_90 |
| 3 | `fa3_gate` | `run_probe.py --only-backends fa3 --max-seq 4096` runs the capability probe and the correctness gate against the float64 naive reference (4096 is the longest length the oracle is allocated for). It fails on no configuration checked, on any check other than `exact`, on any failed check, or on a stamp that is not this instance's (H100, `a3-highgpu-1g`, `dws-flex-start`, clean tree) |
| 4 | `halt` | `sudo shutdown -h +1`, so `run_phase.sh` syncs this step first. The teardown deletes the stopped VM |

In this mode the step subcommand accepts only these four names, so any
session step (`preflight`, `dense_select`, `ground_truth`, and so on) is
refused. A gate session directory cannot be continued as a P4 session.

## The instance

| | |
|---|---|
| Machine | `a3-highgpu-1g`, 1 × H100 80GB, DWS Flex-start |
| Zone | `us-central1-a` (the P4 pin). `asia-southeast1-b` is a fallback only on the researcher's word |
| Image | `attnbench-env-v6-20260917`. This launcher reads **`GCP_IMAGE`** (default v5), not `ATTNBENCH_IMAGE` |
| Bootstrap | `FA3_WHEEL_SHA256=1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a bash scripts/gcp_bootstrap_instance.sh NAME ZONE --no-xattention` |
| Rate | ₹462.42/h all-in (us-central1, 20 Oct 2026). The launcher and the teardown both need `GCP_RATE_INR_HR=462.42` |

## Time and cost (estimates)

| | Minutes | ₹ at 462.42/h |
|---|---|---|
| Flex request pending | up to 115 | nothing billed (no PENDING charge listed; not yet observed) |
| Boot, deploy, preflight, write gate, bootstrap with the FA3 wheel | 20 | 154 |
| The four steps | 10 | 77 |
| Teardown | 5 | 39 |
| **Expected once granted** | **35** | **270** |
| In-guest halt (`GCP_HALT_MINUTES=60`) | 60 | 462 |
| Hard delete (`GCP_MAX_RUN=2h`) | 120 | disk only after the halt |

**Proposed cap: ₹500.** That is the halt's ₹462 plus the disk until the hard
delete. The `halt` step usually ends the VM well before then.

## The retry loop (researcher, 31 Oct 2026)

`caffeinate -i bash scripts/gcp_h100_gate_retry.sh`, from a clean checkout on
the Mac, with `ATTNBENCH_GCP_PROJECT`, `CLOUDSDK_CORE_PROJECT`,
`FA3_WHEEL_SHA256` and, for Spot, `H100_SPOT_RATE_INR_HR` set. It prints its
terms, then every 30 minutes, start to start:

- skips the round while any attnbench GPU VM (g2, a2 or a3, any status)
  exists in the project;
- walks `us-central1-a`, `asia-southeast1-b` and `asia-southeast1-c` on
  FLEX_START, each request cancelled after 300 s; if no zone has capacity,
  the same walk on SPOT. Spot is allowed for the gate alone: it times
  nothing, and `t64_session.py` refuses a Spot row in P4;
- on a grant, stops polling and runs the gate below end to end, with the
  teardown on any exit; on a launcher error, stops; after 48 rounds, stops.

**Flex: cap ₹250**, held by `--max-run-duration`, which deletes the VM
server-side at cap ÷ rate: 32 minutes at ₹463.47/h. The in-guest halt is at
40 minutes, so the delete comes first.

**Spot: cap at most ₹350, held by the in-guest halt** (researcher's rules, 1
Nov 2026, after Shakedowns 5 and 6). The halt comes at the minutes ₹350 buys
at the Spot rate, and the hard delete stays at 40 minutes, never fewer. Spot
is refused when those minutes are fewer than the gate's expected duration,
30 minutes from boot to the gate's own `halt` step (20 for setup and 10 for
the four steps, above; Shakedown 6's L4 took 5.6 minutes from launch to its
first step). At the Spot planning rates (plan Section 5.2):

| Spot rate | In-guest halt | Cap | If the halt failed: delete at 40 min |
|---|---|---|---|
| ₹668.04/h (`asia-southeast1`, the loop's) | 31 min | ₹346 | ₹446 |
| ₹629.94/h (`us-central1`) | 33 min | ₹347 | ₹420 |

Every attempt is a row in `logs/flex_attempts.tsv`; copy the rows into
`h100_flex_attempts.md`. The loop running since 1 Nov 2026 (13:38 IST) was
started from an older checkout (`45f90af`) and is Flex only; it takes this
rule when restarted from a commit that has it.

## The commands

On the Mac, from the repo root, with the launch commit checked out:

```bash
export ATTNBENCH_GCP_PROJECT=...  CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export GCP_IMAGE=attnbench-env-v6-20260917 GCP_ZONE=us-central1-a
export GCP_HALT_MINUTES=60 GCP_MAX_RUN=2h GCP_RATE_INR_HR=462.42
export FA3_WHEEL_SHA256=1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a
NAME=attnbench-h100-gate-$(date +%Y%m%d-%H%M); ZONE=us-central1-a

bash scripts/gcp_cleanup_check.sh                 # expect no compute running
bash scripts/gcp_launch_h100.sh "$NAME"           # type 'launch'; then log the attempt
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE" --no-xattention
```

On the instance: `python3 scripts/t64_session.py run --gate-only --fa3-sha256 "$FA3_WHEEL_SHA256"`.
Then, on the Mac: `gcp_teardown_session.sh "$NAME" "$ZONE" results/t64_gate` and `gcp_cleanup_check.sh`.

If the request is not granted, the launcher cancels it and confirms the
cancellation. The attempt is still logged, and the next one waits for the
next gap between L4 sessions.
