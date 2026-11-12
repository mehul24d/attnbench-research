# P2 on Flex-start: cap and headroom (12 Nov 2026)

**Not launched. It cannot launch yet:** the pre-launch CPU pass at the lock commit `3fc2302`
failed (`results/prelaunch/p2_3fc2302_20261112T1231Z/prelaunch_cpu_pass.json`), and P2's
launch needs its own yes. The step list, minutes and Spot blocks are in
`docs/launch_cards/t61_p2_a100.md`. This card adds the Flex-start rate from the second
lock-wave rulings of 12 Nov 2026 (item 4).

## Machine

`a2-ultragpu-1g` (one A100 80GB) in `asia-southeast1-c`, on Flex-start.

**Why 80GB:** the reasons are quota and comparability, not device memory.

- T0.17 recorded quota only for the A100 80GB line (`PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`, 1,
  `asia-southeast1`).
- Every earlier A100 measurement ran on this machine in this zone, and so did the image-family
  and host-CPU checks.
- A 40GB `a2-highgpu-1g` would need its own quota, zone and image checks.
- P1 ran the same grid in the L4's 24 GB.

## Cost at the two rates

The ruled rate is $2.40/h, the researcher's figure of 12 Nov 2026. At ₹95.99/$ that is
₹230.38/h. The planning rate (plan Section 5.2, all-in) is ₹297.47/h. Planned hours are
5.00, the in-guest halt is at 390 min and the hard delete at 400 min.

| | ₹230.38/h ($2.40) | ₹297.47/h (planning) |
|---|---|---|
| Planned, one session | 1,152 | 1,488 |
| **Cap** (the halt, 6.50 h) | **1,498** | **1,934** |
| Hard delete (6.67 h) | 1,536 | 1,984 |

The cap rate is ₹297.47/h. Appendix A (12 Nov 2026) has the cap take the higher of the two
rates, so the cap stays ₹1,934. $2.40 is the GPU's rate alone, while ₹297.47 also covers the
host's cores, RAM and disk.

## Headroom (`scripts/budget_interlock.py`)

The budget is ₹14,000. Spent so far is ₹4,843, and the rehearsal is closed. P4's caps are
Flex-start ₹3,146 and Spot ₹7,037. At most one pairing may run on Spot.

| P2 Flex cap | P4 | Spent + caps ₹ | Headroom ₹ |
|---|---|---|---|
| ₹1,934 (planning) | Flex-start | 9,923 | 4,077 |
| ₹1,934 (planning) | Spot | 13,814 | **186** |
| ₹1,498 ($2.40) | Flex-start | 9,487 | 4,513 |
| ₹1,498 ($2.40) | Spot | 13,378 | **622** |

At either rate, P2 on Flex-start may launch. With P4 on Spot, the headroom is ₹186 at the
cap rate and ₹622 at $2.40.

## What stops the launch (pre-launch pass, `3fc2302`)

1. **T4.4 is not Done.** It has to read Done in the plan at the tagged commit. Editing the
   plan afterwards changes `plan_sha256`, and the lock then has to be re-cut.
2. **No P1 XAttention masks are staged.** At launch they come from T3.2d's step 5 bucket
   files. The pass has no copy of them.
3. **`learned.LOCKED_CONFIG` is not set** (Gate B). `attnbench/predict/learned.py` is on the
   timed path, so setting it moves the pin and the analysis commit.
