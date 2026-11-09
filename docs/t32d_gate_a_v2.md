# Gate A v2 on T3.2d, with T3.2c as a sensitivity (9 Nov 2026)

**What this is.** Gate A v2 (Appendix A, 8 Nov 2026, as amended through
9 Nov 2026) read offline on the Mac, per the adviser's rulings 2 to 4 of the
eighth 9 Nov 2026 paste:
- **primary:** T3.2d (`results/t32d_p1_20261109T0726Z`, `3c44588`, eager group);
- **sensitivity:** T3.2c (`results/t32c_p1_20261108T1555Z`, `6c0326a`), the superseded identity: its residual is schema 2.

The inputs are T3.2d's bucket artifacts (153 files, 1,938,467,888 bytes, read
only on the researcher's yes, `docs/permission_log.md`):
- the committed prediction (`prediction_P1_20261109T095833754749Z.json`, sha256 `280a3423…`, matching step 7's record);
- step 6's `cells.json` and `pricing_clocks.json`;
- step 5's recordings and `pricing_clocks.json`;
- step 9's ground truth;
- step 12's `delay_injection.json`.

T3.2c's are the 9 Nov copies already read. **It is not the Gate A decision**,
which is the researcher's (T3.3).

## 1. The declarations precede the ground-truth read

Every bound, rule and protocol applied here was committed before T3.2d's
ground truth was first read (the bucket copy of step 9, 2026-11-09T11:46:53Z).

| Declaration | Commit | Committed (UTC) |
|---|---|---|
| Gate A v2: leg 2's family bound 0.25 (provisional), leg 3's range [1/√2, √2] | `f5a3e83` | 2026-11-08T11:35:49Z |
| Resolution floor per (length, batch) | `2b08e0d` | 2026-11-08T11:39:36Z |
| Floor = max(within, between) | `6c0326a` | 2026-11-08T14:49:05Z |
| Relative drift limit (6.31%); floors over identical-path sessions only | `6583613` | 2026-11-08T22:02:38Z |
| Leg 1's minimum 10 clean rounds; the clock protocol and its declared test; the schema 3 residual as primary; the event rule; the defaults; one more cycle | `3c44588` | 2026-11-09T06:31:15Z |

- `3c44588` is T3.2d's identity, merged to `main` (PR #58) before launch.
- The session booted at 07:21Z. Its prediction synced at 09:58:44Z, and its first timed ground-truth block ran at 10:01:23Z.
- The drift limit applied is the exact `0.06311947327826187`, the 0.95 quantile over T3.2a's and T3.2b's rounds.

**Leg 1's "each repeat separately" cannot be met.** T3.2d is the only P1
session on the final identity (`3c44588`). T3.2c shares its timed path
(`4e78a962…`, `scripts/timed_path.py 6c0326a 3c44588`: 86 items,
byte-identical), but its residual is the superseded schema 2. So leg 1 is
read on T3.2d, with T3.2c beside it as a sensitivity, never as the second
repeat.

## 2. Leg 1: fails on T3.2d

Median absolute relative error of the committed prediction against the
measured prefill. Leg 1 needs the recursion below both baselines.

| Session, filter | Cells read | Excluded groups (clean of 10) | Recursion | Sum | Step form | Below both |
|---|---|---|---|---|---|---|
| T3.2d, as recorded | 35 | none | 5.05% (max 59.7%) | 51.21% | 1.96% | no |
| T3.2d, relative drift | 21 | 2048 × 1 (8), 8192 × 1 (1) | 4.29% (max 20.4%) | 15.91% | 1.83% | no |
| T3.2d, no drift filter | 35 | none | 5.05% (max 59.7%) | 51.21% | 1.96% | no |
| T3.2c (sensitivity), as recorded | 27 | 16384 × 1 (8), 16384 × 4 (2) | 6.12% (max 557.9%) | 55.79% | 7.14% | yes |
| T3.2c (sensitivity), relative drift | 28 | 2048 × 1 (8) | 3.12% (max 557.9%) | 24.73% | 2.87% | no |
| T3.2c (sensitivity), no drift filter | 35 | none | 3.33% (max 557.9%) | 50.30% | 4.81% | yes |

**The recursion beats the sum and loses to the step form** in every T3.2d
reading. No T3.2d round was throttled, so the no-filter reading is the
recorded one. T3.2c, on the superseded residual, is unchanged from
`docs/t32c_gate_a_v2.md` (the same readings, now under the rule's floors). By arm, as recorded (recursion, sum, step form):

| Arm | Recursion | Sum | Step form |
|---|---|---|---|
| dense | 2.57% | 11.91% | 2.19% |
| mean-pool | 4.80% | 26.12% | 3.04% |
| oracle | 5.80% | 69.18% | 1.81% |
| XAttention | 9.10% | 32.26% | 1.17% |
| XAttention, no assert | 9.76% | 33.46% | 1.31% |

- **The oracle fix is validated.** The recursion's oracle errors are +2.2% to +16.8% per cell, against +120% to +558% on T3.2c. The schema 3 residual removed the instrument's cost, as `docs/t32d_oracle_residual.md` estimated.
- **The recursion overpredicts in every cell.** The largest errors are at 2048 for every predictor that adds host work (mean-pool +29% recursion and +27% step form; XAttention +60% and +58%). The relative filter's exclusion of the 2048 group is what brings the maximum from 59.7% down to 20.4%.

## 3. Leg 2: fails in 6 of 35 cells

Family bound 0.25 (provisional), relative difference = replay / trace − 1,
families under the cell's floor exempt. The same six cells fail under every
filter (leg 2 reads no ground truth):

| Cell | Family: relative difference |
|---|---|
| oracle@0.25, 16384 × 1 | `block_sparse_attn_forward` 0.487; `mm` 0.457 |
| oracle@0.25, 32768 × 1 | `block_sparse_attn_forward` 0.507 |
| oracle@0.5, 16384 × 1 | `block_sparse_attn_forward` 0.505; `mm` 0.389 |
| oracle@0.5, 4096 × 1 | `addmm` 0.255; `block_sparse_attn_forward` 0.297; `mm` 0.292 |
| oracle@0.5, 8192 × 1 | `addmm` 0.331; `block_sparse_attn_forward` 0.388; `mm` 0.385 |
| XAttention@cal, 2048 × 1 | `cat` 0.271 |

The oracle's errors grew from T3.2c's 0.29 to 0.41 (3 cells under the
rule's floors: `block_sparse_attn_forward` 0.406 and `mm` 0.354 at
0.25 × 16384, 0.348 at 0.25 × 32768, 0.378 and 0.288 at 0.5 × 16384) to 0.26
to 0.51 (5 cells). Every failure is the replay slower than the trace.

## 4. The declared clock test

For each cell: the SM clock median of step 5's `record` phase against step 6's
`profiled_replay` phase (`pricing_clocks.json`, 10 Hz), beside each family's
replay-versus-trace error. Scored pairs are those at or above the cell's
floor under the rule's floors (§6).

| Record / replay SM clock ratio | Cells | Scored pairs | Failing (> 0.25) | Median abs. error |
|---|---|---|---|---|
| below 1.3 | 12 | 115 | 0 | 0.4% |
| 1.3 to 1.95 | 19 | 148 | 5 | 3.6% |
| 1.95 and above | 4 | 32 | 7 | 2.6% |

| Arm, cells at ratio ≥ 1.6 | Cells | Scored pairs | Failing | Median | Max | Record-phase power (W) |
|---|---|---|---|---|---|---|
| oracle | 9 | 66 | 11 | 6.9% | 50.7% | 32.7 to 35.0 |
| mean-pool | 6 | 48 | 0 | 1.8% | 15.2% | 33.5 to 49.8 |
| XAttention (both) | 6 | 52 | 1 | 6.5% | 27.1% | 34.1 to 55.4 |
| dense | 1 | 8 | 0 | 4.1% | 8.0% | 47.7 |

- **The replay runs at the power cap.** Step 6's soak and profiled replay hold medians of 975 to 1,230 MHz at 69 to 73 W in every cell, with `sw_power_cap` (bit 0x4) set in every cell's soak and profiled replay.
- **The trace often does not.** Step 5's `record` phase runs at a median 2,025 to 2,040 MHz in 22 cells (all nine oracle cells, mean-pool and XAttention up to 8192, dense at 2048), at 33 to 55 W. The cap's bit is seen in every record phase, so the cap is reached at some sample but is not the median state. Dense from 8192 on, and mean-pool and XAttention from 16384 on, record at the cap: 1,005 to 1,275 MHz, ratio 1.00 to 1.15.
- **Reading.** No family fails where the trace and the replay share a clock state (ratio below 1.3). All 12 failures are at a ratio of 1.72 or more. A high ratio is necessary but not sufficient: mean-pool cells at ratio 1.66 to 1.88 have no failure, on the same `mm` and `addmm` kernels. The oracle stands apart in one recorded quantity: its record-phase power stays at 33 to 35 W at every length. The host-bound builder leaves the device idle between layers, so each layer's kernels start below the power cap; mean-pool's power rises with length (33 to 72 W).
- This is consistent with leg 2's oracle failures being a clock-state mismatch between the trace and the replay that prices it, and not mispricing within a state. It is the declared test's reading, not a fix: leg 2's bound and families are unchanged.

## 5. Leg 3 under the event rule: passes

Pooled slopes over step 12's delay injection, range [1/√2, √2]:

| Mechanism | Slope |
|---|---|
| `synchronize` | 1.005 |
| `item` | 1.127 |
| `bool` | 1.090 |
| `copy_to_host_pinned` | 1.076 |
| `copy_to_host_pageable` | 1.110 |
| `event_synchronize` | **1.092** (T3.2c: 0.386) |
| `copy_from_host_pinned` | 1.043 |
| `copy_from_host_pageable` | 1.094 |

`event_synchronize` is priced at the record-then-wait round trip
(`event_rule.for_injection`), as declared. T3.2c, priced before the rule,
still fails on it (0.386); its other seven slopes are 1.035 to 1.067. The P1 effect is still 0 ms: no P1
cell on T3.2d has an event synchronise.

## 6. Floors and the scored share

The rule (Appendix A, 9 Nov 2026), q = 0.95, provisional:
- the floor per (length, batch) is the larger of the within-session spread and the between-session spread;
- the between-session spread is over T3.2c and T3.2d only (identical timed path, `4e78a962…`);
- the within-session spread is as before.

| Cell | Floor (ms) | Within | Between | Set by | Sensitivity: T3.2a–d | Sensitivity: T3.2d only |
|---|---|---|---|---|---|---|
| 2048 × 1 | 1.39 | 1.39 | 1.28 | within | 4.25 | 1.10 |
| 4096 × 1 | 7.69 | 6.83 | 7.69 | between | 8.10 | 7.67 |
| 8192 × 1 | 17.76 | 17.76 | 17.46 | within | 69.49 | 18.42 |
| 16384 × 1 | 43.69 | 14.19 | 43.69 | between | 111.05 | 13.77 |
| 16384 × 4 | 139.02 | 25.00 | 139.02 | between | 441.54 | 27.59 |
| 32768 × 1 | 101.16 | 28.33 | 101.16 | between | 321.53 | 32.11 |

**Flag.** The rule's 16384 between-session spread includes T3.2c's two short
groups (16384 × 1 at 8 and 16384 × 4 at 2 clean rounds). Without them, both
16384 floors fall to within-session only (13.77 and 27.59 ms), and the other
cells are unchanged. The rule as written includes them; no reading is chosen
here.

Leg 2's scored share on T3.2d (`(cell, family)` pairs at or above the cell
floor; device time as priced and as traced):

| Floors | Pairs scored | Priced device time | Traced device time |
|---|---|---|---|
| The rule (T3.2c + T3.2d) | 295 of 1,143 (25.8%) | 94.3% | 94.2% |
| Sensitivity (T3.2a–d) | 149 of 1,143 (13.0%) | 73.9% | 73.2% |

The same six cells fail under both.

## 7. The learned predictor, re-selected

- **Training rule (ruling 4):** targets pooled over T3.2c and T3.2d (per cell, the median over both repeats' samples); features from T3.2d; the same six pre-declared configurations; 7-fold cell-grouped CV on P1, seed 0, one thread, 1 h cap (used: about 2 s).
- **Plan default, flagged:** a repeat whose group is short of 10 clean rounds gives no target for its cells (`p1_training.short_of_rounds`, leg 1's rule). Here this leaves out T3.2c's eight 16384 cells from the pool, and they take T3.2d's median alone.
- **The training file** (`attnbench.p1_training/2`, 35 rows, repeats T3.2d and T3.2c): sha256 `de7f0f44…` (record `361c69c8…`), 1,399,192 bytes. It is built with T3.2d's and T3.2c's link figures (`02_link/recursion_inputs.json`, read on the researcher's yes). It is backed up as `p1_training_t32dc_20261109.json.gz` in both local backup folders, and uploaded to `banked/p1_training/` under the same name. The 8 Nov file is not overwritten.
- A provisional run with T3.2a's and T3.2b's link figures standing in gave the same CV: the link figures are constant within P1.

| Config | Median held-out error | Max |
|---|---|---|
| 0 (defaults, log seconds) | 82.75% | 600.9% |
| 1 (defaults, seconds) | 108.16% | 1,214.2% |
| 2 | 4.10% | 64.3% |
| 3 | 6.77% | 61.5% |
| **4** (log seconds; 500 trees, rate 0.02, 4 leaves, min child 2, λ 1.0) | **2.87%** | 64.7% |
| 5 | 3.83% | 70.9% |

Config 4 is selected (`docs/learned_cv_p1_20261109.json`), replacing the
8 Nov provisional config 2 (2.85% on the superseded features). It is not a
lock value until the lock.

## 8. What this settles, and what it does not

- **Settled on T3.2d:**
  - The oracle's residual is fixed.
  - Leg 3 passes, `event_synchronize` included.
  - Leg 2's failures follow the clock state of the trace against the replay.
- **Not passed as written:**
  - Leg 1: the step form beats the recursion in every reading.
  - Leg 2: 6 of 35 cells fail.
- **Not met:** leg 1's two repeats on the final identity.
- **The decision is the researcher's (T3.3).** No threshold was moved.

## 9. After the decision (9 Nov 2026, ninth paste)

**The decision.** The researcher confirmed the Gate A recommendation on Mon
9 Nov 2026 at 19:38 IST: "I agree with your Gate A T3.3 recommendation".
Gate A is read: legs 1 and 2 fail on P1 and leg 3 passes. The study proceeds
to the lock and to P2 and P4. The failing legs are limitations
(`docs/limitations.md`, 9 Nov 2026), and T3.2d was the last P1 revision cycle
(Appendix A, 9 Nov 2026).

**§6's flag, ruled.** A group short of 10 clean rounds leaves the floors
(adviser's ruling 1c; `scripts/cell_floors.py`, default `--min-clean-rounds 10`).
The 16384 floors are within-session only, labelled "between-session
unavailable":

| Cell | Floor (ms) | Set by | Sensitivity: T3.2a–d |
|---|---|---|---|
| 16384 × 1 | 13.77 | within; between-session unavailable | 111.05 |
| 16384 × 4 | 27.59 | within; between-session unavailable | 441.54 |

The other four cells are unchanged. Under these floors, leg 2 scores 312 of
1,143 pairs (27.3%; 95.5% of priced device time, 95.4% of traced). The same
six cells fail: `scripts/gate_a.py --v2` was rerun on T3.2d with
`--cell-floors` set to the rule's file, and its failing cells are §3's.

**Leg 1 with intervals.** `scripts/leg1_bootstrap.py` resamples cells,
paired, 10,000 draws at seed 0 (`docs/leg1_bootstrap_t32d_20261109.json`).
The recursion's error minus the step form's is +3.09 points [+1.91, +4.11]
overall, as recorded. It is above zero in every arm and inside the 6.77%
noise floor overall. The two XAttention arms' point differences are above it
(7.93 and 8.45 points), but their intervals reach down to about 2 points. The
per-arm table is in `docs/limitations.md`.
