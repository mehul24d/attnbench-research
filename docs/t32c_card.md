# T3.2c card: the P1 repeat on the revised identity, eager group only

Prepared 8 Nov 2026 from the adviser's ruling D of 8 Nov 2026 (fourth paste).
Launchable only on the researcher's yes; nothing here runs on its own.

**What its data is.** T3.2c is real data: the first P1 repeat on the revised
code identity (Appendix A, 8 Nov 2026), the identity P2 and P4 run. It is the
session Gate A v2 is first read on. Gate A v2 was committed (`f5a3e83`) before
this session exists. The session runs **`0d90cfe`**, the tip of
`gate-a-floors-2026-11-08`. The session records the commit of its first
start (`manifest.json`, `frozen_commit`) and refuses any other, so no fix is
made within it. A failure ends the session, and its fix is the next
session's code. The prediction is committed (step 8, `predictions/P1-T3.2c/
<session>`), synced and read back before the hold, and ground truth starts at
least 60 s after the sync (the hold, step 9's own check).

## What it runs

Steps 1 to 11 of T3.1 from step 1, on the P1 grid and the working values:

- `oracle_d0.5_L32768` not measured (the drop fallback, in force for every
  pairing);
- **the eager ground-truth group only** (`--repeat c --groups eager`); the
  compiled groups run nothing;
- no phase 0.

## What changed since T3.2a and T3.2b (`3f50aab`)

Appendix A, 8 Nov 2026, the revised identity (1) to (5):

1. The per-layer residual is measured with the device idle (step 5).
2. Memsets are priced by their realised bytes.
3. `masked_fill_` stand-ins are priced from the isolated original.
4. The runner refuses to run on the card outside `session_exit.sh`.
5. The traced pass of Section 5.3 step 8 is added to step 9.

The orchestration-only differences are listed in the same line. Gate A's
leg 3 is not measured: T3.1's steps hold no delay injection.

## Time and cost (T3.2a's measured durations, plus the identity's additions, x 1.3, ₹108.64/h)

| | Minutes | Basis |
|---|---|---|
| T3.2a without phase 0 | 190.6 | `docs/t32b_card.md`, from T3.2a's step results and cost file |
| Step 5's idle-device residual | +15 | estimate, upper end: 7 to 21 synchronised runs per cell in place of the old span (5 to 15 min) |
| Step 9's traced pass | +25 | estimate, upper end: the eager group's 12.5 min again, doubled for profiler overhead |
| Teardown wait removed | 0 | the session halts itself; not taken off |
| **Planned, minutes** | **230.6** | |
| **₹** | **418** | |

| Limit | Minutes from boot | ₹ |
|---|---|---|
| In-guest halt (`GCP_HALT_MINUTES=299`): 1.3 x 230.6 = 299.8, rounded down | 299 | 541 |
| Hard delete (`GCP_MAX_RUN=5h9m`, DELETE), the halt + 10 min | 309 | **560, the cap** |

This uses the T3.2a and T3.2b cards' rule: halt at 1.3 x the planned time,
hard delete 10 minutes later, and the cap is the cost at the hard delete.
The two additions are estimates, flagged. With T3.2a's durations alone the
halt would be 247 min. That leaves 17 minutes of margin over the additions'
upper ends, too thin for a session that cannot be resumed past ground truth.
The session's own exit, `session_exit.sh`, syncs and halts on any exit, so
the limits only bound a hang.

Spend before the session: ₹3,816 of ₹14,000 (₹3,367 in the ledger and
T3.2b's ₹449, whose row is pending). With this session the spend is at most
₹4,376.

## The commands

On the Mac, in a fresh clone at `0d90cfe` (detached, clean tree):

```bash
git clone <repository> t32c && cd t32c && git checkout --detach 0d90cfe
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=299 GCP_MAX_RUN=5h9m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t32c-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$NAME"   # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
nohup bash scripts/session_exit.sh --log ~/t32c.log -- \
    python3 scripts/t31_session.py --repeat c --groups eager >/dev/null 2>&1 &
```

A resume after an instance problem is allowed only at the same commit and
before ground truth:

```bash
nohup bash scripts/session_exit.sh --log ~/t32c.log -- \
    python3 scripts/t31_session.py --repeat c --groups eager \
    --session results/t32c_p1_<time> --from <step> >/dev/null 2>&1 &
```

Back on the Mac, once the instance has halted itself (or at the cap):

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t32c_p1_<time>   # from the bucket if the VM is halted or gone
bash scripts/gcp_cleanup_check.sh
```

A poll loop on the Mac watches the instance for the whole session. It
outlives any question, and no blocking question is asked while the session
bills.

## Self-halt, verified

`tests/test_session_exit.py` runs four CPU dry runs under the wrapper:

- a passing dry run (11 steps);
- a simulated `RuntimeError` exit (rc 1);
- a `kill -9` of the runner (rc 137);
- a refused repeat (rc 2).

Each one syncs and then halts. The teardown's pull from the bucket is tested
in `tests/test_teardown_bucket_fallback.py`. The session never relies on a Mac watcher
to stop billing.

## What it settles, and what it does not

T3.2c gives Gate A v2's legs 1 and 2 on the identity P2 and P4 run, with the
prediction committed before the ground truth. It also gives the first
per-layer residuals and spreads under the idle-device method (T4.1 row 34)
and the first profiler-overhead rows.

It does not give:

- leg 3 (no delay injection);
- the compiled rows;
- a between-session spread on the revised identity: one repeat is one
  session.
