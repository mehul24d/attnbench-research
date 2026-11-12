# P2 on Flex-start: cap and headroom (12 Nov 2026)

**Not launched.** P2's launch needs its own yes. The pre-launch CPU pass passed at the lock
commit `0301fb6` (`results/prelaunch/p2_0301fb6_20261112T1420Z/prelaunch_cpu_pass.json`), with
T3.2d's step 5 XAttention masks staged as P1's, and again at the tag target `3aa0449`
(`results/prelaunch/p2_3aa0449_20261112T1607Z/prelaunch_cpu_pass.json`, sha256 `8f40ef5f83b7…`). The step list, minutes and Spot blocks are in
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

## What the pre-launch pass found (12 Nov 2026)

The three refusals at `3fc2302` are cleared (the third lock-wave rulings of 12 Nov 2026):

1. T4.4 is written Done in the plan at the lock commit.
2. T3.2d's step 5 XAttention masks are staged (`ATTNBENCH_PRELAUNCH_MASKS`; 20 files, read
   under the yes logged in `docs/permission_log.md`).
3. `learned.LOCKED_CONFIG` is config 4 (P1 cell-grouped CV).

The first re-cut's pass (`fbbbb57`) then failed at host_components: the lock fill gave the
chain-check tolerance as one fraction, and the steps read `{"aggregate"}`. The fix (`9ddb9ee`)
moved the pin to `fa28e01b…`, and the lock was cut a second time (`0301fb6`). The pass there
passed. The cap rate is ₹297.47/h, host included (ruling 4), so the cap stays ₹1,934.

P4's pass at `0301fb6` refuses only on T6.1, as ruled
(`results/prelaunch/p4_0301fb6_20261112T1422Z/`).
