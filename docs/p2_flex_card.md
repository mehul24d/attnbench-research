# P2 on Flex-start: the relaunch at `lock-2026-11-13` (13 Nov 2026)

**Not launched.** P2's relaunch needs its own yes. It is asked only after the pre-launch CPU pass
prints PASSED at the commit the tag names (`scripts/prelaunch_cpu_pass.sh p2`, with T3.2d's step 5
XAttention masks staged as P1's). The step list and minutes are in
`docs/launch_cards/t61_p2_a100.md`.

**Why a relaunch.** The relaunch of 12 Nov 2026 (`attnbench-a100-p2r-20261112-2309`, `3aa0449`)
stopped in step 7 on `xattention_dcal_L2048_b1`, with no T6.1 data (34 min, ₹169). XAttention's
Triton kernels were not nulled in the null pass. They are now null entries, and the lock is re-cut
at the fix (Appendix A, 13 Nov 2026).

## What is asked

| | |
|---|---|
| Machine | one `a2-ultragpu-1g` (A100 80GB), `asia-southeast1-c` |
| Provisioning | Flex-start; no Spot fallback (the scratchpad's `gcloud` wrapper refuses Spot, since `gcp_launch_a100.sh` falls back to Spot by itself) |
| Code | a fresh clone at the tag `lock-2026-11-13`, deployed by `gcp_deploy_source.sh` |
| Bootstrap | `gcp_bootstrap_instance.sh "$NAME" "$ZONE"`, without `--suite-record` (the researcher's ruling of 12 Nov 2026) |
| Staged by scp | P1's masks (20 files) as `results/<session>/p1_masks/`, the P1 training file (`de7f0f44…`) |
| Start | `nohup bash scripts/session_exit.sh --log ~/p2.log -- python3 scripts/t61_session.py run --session results/t61_p2_<UTC>`; no `--banked-at`, which a real run refuses |
| Cap | ₹1,934 (the halt, 6.50 h at ₹297.47/h) |
| In-guest halt | 390 min (`GCP_HALT_MINUTES=390`) |
| Hard delete | 400 min (`GCP_MAX_RUN=400m`) |
| Watcher | `t61_rehearsal_watch.sh` in the background, looping until the session exits; no blocking question while it bills |

## New in preflight (step 2)

The Triton smoke check (`t31_steps.triton_smoke`, the adviser's rulings dated 13 Nov 2026, item 3)
runs before anything long. It is untimed and runs one XAttention cell at 2048:

1. step 7's whole cell: the recording, the isolated replay, the null pass and its structure check,
   the bare pass and the residual;
2. step 8's one refusing check, the replay's live masks against the bank;
3. whether step 8 will need the isolated original, reported but not refused.

A 2048 cell took 20 to 31 s in step 7 on 12 Nov 2026, model load included. A failure stops the
session in preflight, about 5 minutes after the start.

| Step that runs XAttention's Triton path on the GPU | Covered |
|---|---|
| 5 `probe` | passed on 12 Nov 2026 |
| 6 `mask_gate` | passed on 12 Nov 2026 (verdict "fail", Triton against torch; the per-arm rule of 27 Oct 2026 reads it) |
| 7 `host_components` | by the smoke check, at 2048 |
| 8 `device_components` | its refusing check by the smoke check, at 2048 |
| 12 `ground_truth` | not covered: its timed rounds. The forward itself runs untimed in step 6 |

Above 2048 the estimator splits the query into more chunks, and each chunk launches the same two
kernels. The smoke check does not run those lengths.

## Headroom (`scripts/budget_interlock.py`)

Spent ₹5,121. P4 on Spot is ruled out (the adviser's rulings dated 13 Nov 2026, item 6).

| P2 | P4 | Spent + caps ₹ | Headroom ₹ |
|---|---|---|---|
| Flex-start ₹1,934 | Flex-start ₹3,146 | 10,201 | 3,799 |
| Flex-start ₹1,934 | Spot ₹7,037 | 14,092 | −92: ruled out |

The ruled rate is $2.40/h (₹230.38/h), the GPU alone. The cap takes the planning rate, ₹297.47/h,
host included (Appendix A, 12 Nov 2026).

## Machine

**Why 80GB:** the reasons are quota and comparability, not device memory.

- T0.17 recorded quota only for the A100 80GB line (`PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`, 1,
  `asia-southeast1`).
- Every earlier A100 measurement ran on this machine in this zone, and so did the image-family and
  host-CPU checks.
- P1 ran the same grid in the L4's 24 GB.
