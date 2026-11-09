# T3.2c after Gate A v2: drift filters, leg 1, legs 2 and 3 diagnosed (9 Nov 2026)

The adviser's rulings 1 and 3 to 6 of 9 Nov 2026 (sixth paste), on T3.2c
(`6c0326a`, eager group), from the bucket copies read on 9 Nov
(`docs/permission_log.md`). No new bucket read. The rules were committed
before leg 1 was computed (`6583613`, 22:02Z, Appendix A, 9 Nov 2026).
**None of this is the Gate A decision**, which is the researcher's (T3.3).

## 1. The drift filter, and leg 1 under three filters

The relative limit, from T3.2a and T3.2b alone: the 0.95 quantile
(provisional) of |dense drift| / warm-up time over their 120 rounds is
**6.31%** (median 1.42%, largest 12.63%). It is applied after the session by
`attnbench/analysis/drift_filter.py` from each file's per-round record. Throttle
flags are kept under every filter, and T3.2c had none.

The plan's minimum clean rounds per group is a lock value (T4.1 row 23,
proposed 3, not fixed). Until lock, a group needs the reps it ran with
(10). Clean rounds kept per group:

| Group | Original (absolute 52.83 ms) | Relative (6.31%) | None |
|---|---|---|---|
| 2048 × 1 | 10 | **8** | 10 |
| 4096 × 1 | 10 | 10 | 10 |
| 8192 × 1 | 10 | 10 | 10 |
| 16384 × 1 | **8** | 10 | 10 |
| 16384 × 4 | **2** | 10 | 10 |
| 32768 × 1 | 10 | 10 | 10 |

**Listed for rerun.** At the working minimum of 10:
- original filter: 16384 × 1 and 16384 × 4;
- relative filter: 2048 × 1 (its rounds 1 and 2 drifted −6.73% and −6.51%; T3.2a and T3.2b's 2048 rounds reach −12.6%).

At the proposed minimum of 3, only 16384 × 4 under the original filter.

**Leg 1.** Median absolute relative error over the cells read
(`scripts/gate_a.py --v2`, relative floor 0.0677):

| Filter, minimum | Cells | Recursion | Sum | Step form | Recursion below both? |
|---|---|---|---|---|---|
| Original, 10 | 27 | 6.12% | 55.8% | 7.14% | yes |
| Original, 3 | 34 | 4.14% | 51.7% | 5.37% | yes |
| Relative, 10 | 28 | 3.12% | 24.7% | 2.87% | **no** (step form lower) |
| Relative, 3 | 35 | 3.33% | 50.3% | 4.81% | yes |
| None, 10 or 3 | 35 | 3.33% | 50.3% | 4.81% | yes |

The ranking depends on which cells are read. Leg 1 as written needs every
cell in each repeat on the revised identity, and T3.2c is the only such
repeat.

**A finding that dominates leg 1: the oracle is overpredicted 2.2 to 6.6
times by every predictor.** Example: 32768 at 0.25, recursion 19.2 s
against 2.9 s measured. The other arms' median errors (recursion, relative
filter, all 35 cells):
- dense: 1.2%;
- mean-pool: 1.5%;
- XAttention: 4.9% (XAttention without asserts: 6.1%).

Their largest errors are at 2048 (+27% to +62%) and XAttention at 4096
(+15%).

The cause is in step 5: the idle-device per-layer residual (Appendix A,
8 Nov 2026, item B(1)) adds:

| Oracle cell | Residual added | Measured prefill |
|---|---|---|
| 0.25 at 16384 | 5,389 ms | 1.40 s |
| 0.5 at 16384 | 9,108 ms | 1.50 s |
| 0.25 at 32768 | 19,020 ms | 2.91 s |

The other arms get 57 to 412 ms. It lands on one `aten::copy_` per layer
(188 ms each at 16384 and 0.25). That is more host time than the whole
measured prefill, so it measures work that is not in the oracle's timed
path. Not diagnosed further: step 5's per-cell residual records are only in
the bucket.

## 3. Leg 3: what the slope compares, and `event_synchronize`

**What the slope is.** For each mechanism and placement (busy, idle), each
injected count c > 0 contributes one pair:
- the predicted increment over c = 0, from the plan written before timing;
- the measured increment over c = 0, the median of 10 repeats.

The pooled slope is Σ predicted × measured / Σ measured², a fit through the
origin over both placements (`delay_injection.slope_for`). Idle increments
are about 100 times larger than busy ones (up to 4,000 injections against
64), so the idle placement dominates the slope.

| `event_synchronize` | Predicted per injection | Measured per injection | Ratio |
|---|---|---|---|
| Busy | 365 µs | 428 µs | 0.854 |
| Idle | 1.16 µs | 8.59 µs | 0.135 |
| Pooled | | | **0.386** |

**Why.** The two sides time different operations:
- **The price** comes from T2.5's link microbenchmark (`components/link.py`). It times `Event.synchronize()` on an event recorded once and already complete: a host-only check.
- **The injector**, like real code, runs `ev.record(); ev.synchronize()`. Every call enqueues a marker and waits for the device to reach it: a full round trip.

On an idle device every other mechanism measures 6.4 to 10.1 µs per
injection, close to their prices (idle ratios 1.24 to 1.34). For example,
item measures 8.00 µs, against 8.59 µs for `event_synchronize`. The data
support the hypothesis: the idle price leaves out the wait for the marker
and the host's wake-up after it. What remains open:
- The direct test is one microbenchmark row on a GPU, `record()` followed by `synchronize()` per call.
- The split between device latency and host wake-up is not measured.

**Whole-prefill effect of the proposed term: 0 ms on every P1 cell.**
- No `event_synchronize` occurs in any of T3.2c's 35 priced cells. The mechanisms that occur are bool (1,260), the pinned copy to host (812) and the pageable copy from host (1,540).
- The term changes leg 3's predictions only.
- Not built.

**Also seen.** All eight mechanisms under-predict busy injections by 10 to
15% (ratios 0.854 to 0.899). Their idle ratios are 1.24 to 1.34, except
`event_synchronize`. The pooled slope through the origin averages these
opposite errors, so the seven mechanisms within the range are within it
partly by cancellation.

Idle measured increments are not linear in the count. For all eight
mechanisms, 2000 adds only 1 to 2 ms over 1000.

## 4. Leg 2: the three failing oracle cells

Per arm at the same length: the replay's relative difference against the
trace, and the median traced time per `aten::mm` call.

| 16384 × 1 | `mm` | Traced per call | `block_sparse_attn_forward` |
|---|---|---|---|
| Dense | −0.026 | 7.58 ms | |
| Mean-pool 0.25 | +0.055 | 6.75 ms | +0.041 |
| Mean-pool 0.5 | +0.018 | 7.09 ms | −0.002 |
| XAttention | +0.051 | 6.42 ms | +0.026 |
| **Oracle 0.25** | **+0.354** | **5.61 ms** | **+0.406** |
| **Oracle 0.5** | **+0.288** | **5.64 ms** | **+0.378** |

At 32768, oracle 0.25 is +0.161 (`mm`) and +0.348 (block-sparse attention).
Mean-pool and XAttention are −0.037 to +0.089.

The replay prices the same 113 `mm` calls alike in every arm (6.7 to 7.6 ms
per call at 16384). In the oracle only the compute-bound kernels trace
faster:
- `mm` −26%, `addmm` −14% and block-sparse attention −29%, against their replay;
- the memory-bound ones agree within 1%: `mul`, `silu`, `add`, `pow`.

**Correction (9 Nov 2026, `docs/t32d_oracle_residual.md` §4):** the peak
SM clocks below are wrong for dense and mean-pool. Step 9's rows give them
1,950 to 1,995 MHz at 16384, beside the oracle's 1,980 to 2,025. Peak clocks
do not separate the oracle, so the clock-state reading in this section is
untested, not supported. The device-busy shares stand.

The memory clock was fixed (6,251 MHz) and the SM clock moved under the
software power cap, which every arm hit. The oracle keeps the device busy
for only 82 to 88% of its prefill (traced device total over the measured
median), against 93 to 104% for the others. Its blocks reach a peak SM
clock of 1,965 to 2,025 MHz, against 1,215 to 1,410 MHz for dense and
mean-pool at 16384.

**Reading: mispricing by clock state, not kernel-position attribution and
not profiler distortion.**
- *Not attribution.* Moving time between positions would keep the cell total. Here every large compute-bound family in the oracle cell is overpriced in the same direction, and the cell is +20% (+18%, +11%).
- *Not profiler distortion.* It would slow traced kernels, not speed them up. The same recording setup agrees within 6% for every other arm.
- *Mispricing.* The replay runs kernels back to back at power-capped clocks. The oracle's own prefill leaves the device idle between them, and its kernels run at higher clocks.

The SM clock is sampled per block (10 Hz, minimum and maximum), not per
kernel. So this is the reading the data support, not a measurement of each
kernel's clock.

## 5. Scored share, by pairs and by priced device time

| Floors | Pairs scored | Share of priced device time | Share of traced device time |
|---|---|---|---|
| T3.2a/b/c (the 9 Nov 2026 reading; now a sensitivity) | 149 / 1,143 = 13.0% | 73.5% | 72.9% |
| T3.2c only, within-session (the rule of 9 Nov 2026) | 336 / 1,143 = 29.4% | 98.6% | 98.6% |

Under the T3.2c-only floors, leg 2 still fails in the same three cells.
The share is reported as a finding, and did not choose the floor's form.

## 6. Floors from identical-path sessions only (T3.2c alone)

By the rule of 9 Nov 2026 (quantile 0.95, provisional; `scripts/cell_floors.py
--session T3.2c ...`):

| Cell | Floor (ms) |
|---|---|
| 2048 × 1 | 1.93 |
| 4096 × 1 | 6.33 |
| 8192 × 1 | 12.86 |
| 16384 × 1 | 12.61 |
| 16384 × 4 | 5.63 |
| 32768 × 1 | 19.45 |

Every cell is within-session only.

**What is missing:**
- **The between-session spread.** It needs a second session with the timed-path hash `4e78a962…`. T3.2a and T3.2b hash to `a5ade053…`, and their floors are a sensitivity only (`docs/t32c_gate_a_v2.md`).
- **Enough rounds at 16384.** T3.2c's 16384 series are from the recorded clean rounds: 16384 × 4 rests on 2 rounds per series, and 16384 × 1 on 8.

## Reproduce

Scratchpad tools, run with `python3 -I`, on the 9 Nov bucket copies:
- `leg1_filters.py`: through `gate_a.main` with the pulled prediction and the bucket creation time step 10 recorded.
- `leg2_diag.py`, `leg2_table.py`, `leg2_more.py`.
- `scored_share.py`.
- `leg1_cells.py`.

The committed code is `drift_filter.py`, `scripts/gate_a.py` and `scripts/cell_floors.py`.
