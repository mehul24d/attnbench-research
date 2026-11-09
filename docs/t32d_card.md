# T3.2d card: the one more cycle, P1 on the identity of 9 Nov 2026, eager group

Prepared 9 Nov 2026 from the adviser's rulings of 9 Nov 2026 (seventh paste)
and the researcher's confirmation of one more cycle (Appendix A, 9 Nov
2026). **Launchable only on the researcher's yes. Nothing here runs on its
own.**

**What its data is.** T3.2d is real data: the second P1 session whose timed
ground-truth path hashes to `4e78a962…`, beside T3.2c. It gives:
- leg 1 in a second repeat;
- the between-session spread for every cell's floor;
- the fixed oracle residual, measured;
- the clock state of the pricing steps.

The session runs **`3c44588`** (`3c44588553b5c0c07c87f188300f9fd38dc710de`, branch `t32d-identity-2026-11-09`), detached, on a clean tree. It records the commit
of its first start (`manifest.json`, `frozen_commit`) and refuses any other,
so no fix is made within it. A failure ends the session, and its fix is the
next session's code.

## What it runs

Steps 1 to 12 of T3.1 from step 1, as T3.2c ran them:
- the P1 grid and the working values;
- `oracle_d0.5_L32768` not measured;
- the eager ground-truth group only (`--repeat d --groups eager`);
- no phase 0;
- step 12, Gate A's leg 3, on the declared set, unchanged.

The prediction is committed (step 8, `predictions/P1-T3.2d/<session>`), synced
and read back before the hold. Ground truth starts at least 60 s after the
sync.

## What changed since T3.2c (`6c0326a`)

Appendix A, 9 Nov 2026:

1. **Step 9 reports an incomplete group** (exit 4); the later steps run and the session does not pass.
2. **Step 5's residual, schema 3:** the oracle's host-side mask builder runs with every Python mode suspended inside its gap. This is the primary method for every arm.
3. **T2.5 measures `event_record_synchronize`.** An event synchronise takes it by the structural rule (`event_rule`); step 12's plan prices the injector's at the round trip. P1 effect: 0 ms.
4. **NVML through steps 5 and 6**, per cell and phase (`pricing_clocks.json`). Step 6 soaks 5 s (at least 2 synchronised replays) before the profiled replay it prices from.
5. **Analysis only:**
   - the drift filter after the session;
   - Gate A's leg 1 rule: a group below 10 clean rounds is listed and excluded;
   - the floors over identical-path sessions.

**The timed ground-truth path is byte-identical** to `6c0326a`'s:
`python3 scripts/timed_path.py 6c0326a 3c44588` gives 86 items, sha256
`4e78a962f9be7c7eaa14145660ae4d86bb395ad6818fa6ae1dbe92926b88dff4`.

## Time and cost (T3.2c's measured durations, x 1.3, ₹108.64/h)

| | Minutes | Basis |
|---|---|---|
| T3.2c, boot to halt | 199 | measured (`docs/t32d_proposal.md`) |
| Step 6's soak | +3.3 | estimate: 5 s per cell, two replays at 16384 × 4 and 32768 |
| The sampler's start and stop, 70 cells | +1.2 | estimate: about 1 s per cell |
| Step 5's oracle residual, fixed | −4.6 or more | from T3.2c's records: 8 runs per cell over 40.0 s of residual, at least 7/8 of it the modes' |
| T2.5's added row | +0 | 2,000 calls of about 9 µs |
| **Planned, minutes** | **200** | T3.2c's 199, rounded up |
| **₹** | **362** | |

| Limit | Minutes from boot | ₹ |
|---|---|---|
| In-guest halt (`GCP_HALT_MINUTES=260`): 1.3 x 200 | 260 | 471 |
| Hard delete (`GCP_MAX_RUN=4h30m`, DELETE), the halt + 10 min | 270 | **489, the cap** |

The rule of the T3.2a to T3.2c cards applies: halt at 1.3 x the planned
time, hard delete 10 minutes later, and the cap is the cost at the hard
delete. The session's own exit, `session_exit.sh`, syncs and halts on any
exit, so the limits only bound a hang.

Spend before the session: ₹4,179 of ₹14,000 (`docs/spend_ledger.md`). With
this session the spend is at most ₹4,668 (headroom at least ₹9,332).

## The commands

On the Mac, in a fresh clone at `3c44588` (detached, clean tree):

```bash
git clone <repository> t32d && cd t32d && git checkout --detach 3c44588
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=260 GCP_MAX_RUN=4h30m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t32d-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a
bash scripts/gcp_cleanup_check.sh && bash scripts/gcp_launch_l4.sh "$NAME"   # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`:

```bash
nohup bash scripts/session_exit.sh --log ~/t32d.log -- \
    python3 scripts/t31_session.py --repeat d --groups eager >/dev/null 2>&1 &
```

A resume after an instance problem is allowed only at the same commit and
before ground truth:

```bash
nohup bash scripts/session_exit.sh --log ~/t32d.log -- \
    python3 scripts/t31_session.py --repeat d --groups eager \
    --session results/t32d_p1_<time> --from <step> >/dev/null 2>&1 &
```

Back on the Mac, once the instance has halted itself (or at the cap):

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t32d_p1_<time>   # from the bucket if the VM is halted or gone
bash scripts/gcp_cleanup_check.sh
```

A poll loop on the Mac watches the instance for the whole session. It
outlives any question, and no blocking question is asked while the session
bills.

## Self-halt, verified

- `tests/test_session_exit.py` runs four CPU dry runs under the wrapper at this identity, and each one syncs and then halts:
  - a passing dry run (12 steps);
  - a simulated `RuntimeError` exit (rc 1);
  - a `kill -9` of the runner (rc 137);
  - a refused repeat (rc 2).
- `tests/test_teardown_bucket_fallback.py` tests the teardown's pull from the bucket.
- A CPU-boundary rehearsal of steps 1 to 8 and 12 passed on this identity's code (9 Nov 2026). Its three new records were present:
  - the residual's schema 3, with no host-side function on the boundary, as declared;
  - each cell's `pricing_clocks.json`, phases marked and unsampled;
  - step 6's soak (99 replays in 5.0 s at 2048).

## What it settles, and what it does not

T3.2d gives Gate A v2's three legs a second session on `4e78a962…`. T3.3
then reads Gate A v2 on T3.2c and T3.2d (the one-more-cycle amendment). It
also gives:
- the floors' between-session spread;
- the fixed oracle residual;
- leg 2's declared clock test (step 5's `record` phase against step 6's `profiled_replay` phase);
- leg 3 under the event rule.

The learned predictor is re-selected after it (adviser's ruling 7 of the
sixth paste).

It does not give:
- the compiled rows;
- the other arms' residuals, diagnosed: the same instrument cost may sit in them (57 to 412 ms on T3.2c).
