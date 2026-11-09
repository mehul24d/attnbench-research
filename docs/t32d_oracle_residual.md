# The oracle's residual, the clocks, and the event rule (9 Nov 2026)

The adviser's rulings 2, 3 and 4a of 9 Nov 2026 (seventh paste), on T3.2c
(`6c0326a`). Inputs:
- the nine oracle `layer_residual.json` files from step 5 (9 files, 102,286 bytes, read only; `docs/permission_log.md`);
- the 9 Nov bucket copies already in use (prediction, step 6 `cells.json`, step 9 ground truth).

T3.2a and T3.2b were not read. The Appendix A lines of 9 Nov 2026 are the
record; this note gives the numbers behind them.

## 1. Residual against the builder's work and the layer's wall

The oracle builds each layer's mask on the host inside the prefill
(`model._mask_from_scores` → `masks.importance_block_mask`). Per query block
it runs `rand`, `topk` and `tolist()`, then one CPU assignment per kept
block. Ground truth times this inside the prefill (`mask_builds_timed` =
280 = 28 layers × 10 rounds).

| Cell | Tokens | Blocks | Assignments per layer | Residual per layer (ms) | Measured per layer (ms) | Residual / measured |
|---|---|---|---|---|---|---|
| 0.25, 2048 | 1,951 | 16 | 26 | 9.00 | 4.76 | 1.9 |
| 0.5, 2048 | 1,951 | 16 | 52 | 11.12 | 4.78 | 2.3 |
| 0.25, 4096 | 4,068 | 32 | 116 | 21.51 | 10.01 | 2.1 |
| 0.5, 4096 | 4,068 | 32 | 232 | 30.37 | 10.30 | 2.9 |
| 0.25, 8192 | 8,075 | 64 | 488 | 61.10 | 22.83 | 2.7 |
| 0.5, 8192 | 8,075 | 64 | 976 | 96.64 | 23.47 | 4.1 |
| 0.25, 16384 | 16,370 | 128 | 2,000 | 192.39 | 49.97 | 3.9 |
| 0.5, 16384 | 16,370 | 128 | 4,000 | 325.20 | 53.61 | 6.1 |
| 0.25, 32768 | 32,700 | 256 | 8,096 | 679.53 | 104.05 | 6.5 |

- **It follows the assignments, not the device.** At 16384 the two densities differ by 2,000 assignments and 132.8 ms: 66 µs each. That slope, plus the 16384 intercept scaled by rows, predicts 32768 within 4% (657 against 680 ms). Step 6's device time per layer roughly doubles per length (4.2 to 101.5 ms), while the residual grows 2.4 to 3.5 times per step.
- **It is more than the layer's whole wall** at every cell (1.9 to 6.5 times), so it cannot be host time on the timed path.
- **The gaps are repeatable.** Each cell settled at 7 runs. Layer spreads are 0.1% to 11.1%, one layer above the 10% tolerance (0.5 at 8192, labelled). The timer's overhead is 38.4 to 39.8 µs per interval, subtracted.

## 2. The hypotheses

- **H1, the instrument on: the cause.** Step 5 times the gaps with the tape's torch-function mode (`_GapReplayer`) and the gap clock's dispatch mode (`_GapMode`) active. The builder's CPU operators are not met calls, so each one runs inside a gap and pays both modes' Python dispatch. `tools/mask_modes.py` (scratchpad), the builder on this Mac (torch 2.13), native against under the two modes:

  | Blocks, density | Native (ms) | Under the modes (ms) | Times |
  |---|---|---|---|
  | 16, 0.25 | 0.24 | 1.89 | 8.0 |
  | 32, 0.25 | 0.64 | 5.97 | 9.4 |
  | 64, 0.25 | 1.89 | 19.81 | 10.5 |
  | 128, 0.25 | 6.24 | 70.73 | 11.3 |
  | 128, 0.5 | 10.09 | 131.20 | 13.0 |
  | 256, 0.25 | 23.24 | 266.85 | 11.5 |

- **H2, device waits in mask construction: rejected.**
  - The scores are on the host (`compute_importance_scores` moves them with `.cpu()`), and the builder touches no device.
  - Every gap opens after a synchronise.
  - A wait would scale with device time; the residual scales with the assignments.
- **H3, double counting: rejected.** The builder's CPU operators are in no other price. The bare pass prices device calls only. T3.1's oracle sequence has no host-compute position (18,884 kernels and 756 synchronisations over the nine cells, no `host_compute`).
- **Is the idle-device method itself wait-contaminated?** Not here: the per-assignment cost is the same at every length. Its contamination is the instrument's own dispatch, paid on any operator a gap does not meet, in every arm. The other arms' residuals (57 to 412 ms) are not diagnosed here.

## 3. The fix and its whole-prefill effect

`layer_residual` schema 3 (`idle_device_gaps_bare_host`):
- each declared host-side function (`cpu_boundary.HOST_SIDE_FUNCTIONS`) runs with every Python mode suspended;
- its own time is added to its layer's gap and recorded beside it (`host_side_s`).

It is not applied on the CPU boundary, where the host is the device.

The effect, estimated with `tools/resid_effect.py` (scratchpad):
- the committed components are re-run through the recursion;
- each oracle layer's residual is divided by the Mac's inflation at its block count, on the operation that carries it.

| Cell | Measured (ms) | Recursion now | Fixed (1 / inflation) | Residual removed |
|---|---|---|---|---|
| 0.25, 2048 | 133 | 293 (+120%) | 175 (+31%) | 175 (+31%) |
| 0.5, 2048 | 134 | 352 (+163%) | 178 (+33%) | 178 (+33%) |
| 0.25, 4096 | 280 | 647 (+131%) | 311 (+11%) | 311 (+11%) |
| 0.5, 4096 | 289 | 896 (+211%) | 319 (+11%) | 319 (+11%) |
| 0.25, 8192 | 639 | 1,770 (+177%) | 667 (+4%) | 667 (+4%) |
| 0.5, 8192 | 657 | 2,765 (+321%) | 688 (+5%) | 685 (+4%) |
| 0.25, 16384 | 1,399 | 5,473 (+291%) | 1,450 (+4%) | 1,444 (+3%) |
| 0.5, 16384 | 1,501 | 9,194 (+513%) | 1,526 (+2%) | 1,508 (0%) |
| 0.25, 32768 | 2,913 | 19,166 (+558%) | 2,933 (+1%) | 2,894 (−1%) |

- The two right-hand columns differ by at most 1.3 points of the measured prefill. The native build (about 17 ms per layer at 16384, 0.25) hides under the layer's device time, so the inflation factor's uncertainty does not matter.
- What remains at 2048 (+31%) is the size of the other arms' 2048 errors (+27% to +62%).
- This is an estimate from T3.2c's records. T3.2d measures the fixed residual directly.

**Declared before T3.2d: schema 3 is the primary residual method for every
arm.** It was resolved inside the time-box, so the schema 2 method is not
recorded beside it.

## 4. Clocks (ruling 3)

**What T3.2c recorded.** Step 9 only, 10 Hz, per block:
- the SM clock's minimum and maximum;
- the memory clock (6,251 MHz in every block);
- the throttle reasons (`sw_power_cap` in every block).

There was no power, no mean clock, and nothing during steps 5 and 6.

Peak SM clock per row, step 9 (MHz):

| Length | Dense | Oracle | Mean-pool | XAttention |
|---|---|---|---|---|
| 2048 | 1,890 | 1,980 to 2,010 | 1,980 | 1,935 to 1,995 |
| 4096 | 1,950 | 1,560 to 1,710 | 1,440 to 1,770 | 1,845 to 1,905 |
| 8192 | 1,965 | 1,920 to 1,980 | 1,680 to 1,860 | 1,920 to 1,935 |
| 16384 | 1,995 | 1,980 to 2,025 | 1,950 to 1,980 | 1,935 to 1,980 |
| 32768 | 1,935 | 2,025 | 1,980 | 1,965 to 2,010 |

**Correction.** `docs/t32c_leg1_filters.md` §4 gave dense and mean-pool at
16384 a peak of 1,215 to 1,410 MHz. That is wrong: the rows give 1,950 to
1,995 MHz. Peak clocks do not separate the oracle. So the clock-state
reading of leg 2's three failures is untested, not supported. The device-busy
shares (oracle 82 to 88%, others 93 to 104%) stand.

**The protocol for T3.2d, declared before launch:**
- Steps 5 and 6 sample the SM and memory clocks, board power and throttle bits at 10 Hz. A separate process does the sampling, per cell and per phase (`attnbench/pricing_clocks.py`; `pricing_clocks.json` in each cell's directory).
- Before the one profiled replay it prices from, step 6 runs the captured replay back to back, each replay synchronised, for 5 s and at least 2 replays (`t31_steps.soak`). The priced replay then finds the sustained, power-capped state of the timed rounds' back-to-back prefills.

**What it can and cannot explain.** The protocol prices the state the timed
rounds run in. It does not by itself explain the oracle's 0.29 to 0.38
errors. Their reference is step 5's traced prefill, where the host-bound
builder leaves the device idle between kernels. Traced, the oracle at 0.5
and 16384 runs 2,306 ms against 1,501 untraced.

The test, declared for T3.2d: for each failing family, its replay-versus-trace
error against the ratio of the SM clock medians of step 5's `record` phase
and step 6's `profiled_replay` phase.

## 5. Event synchronise (ruling 4a)

The rule, from the recorded trace's CUDA calls in the prefill region, in time
order (`attnbench/components/event_rule.py`):
- an event synchronise takes T2.5's new record-then-wait price (`event_record_synchronize`) when a `cudaEventRecord` lies after the previous synchronising call;
- otherwise it takes the completed-event price.

A synchronising call here is a stream, device or event synchronise, an
implicit one, or a blocking copy. The injector records before every wait,
so step 12's plan prices it at the round trip.

**P1 effect: 0 ms in every cell.** T3.2c's 35 priced cells have no event
synchronise. They have `copy_from_host_pageable` 1,540, `bool` 1,260 and
`copy_to_host_pinned` 812. The rule changes leg 3's `event_synchronize`
predictions only: idle about 8.6 µs per injection in place of 1.16, if T3.2d's
link row measures what step 12 measured.
