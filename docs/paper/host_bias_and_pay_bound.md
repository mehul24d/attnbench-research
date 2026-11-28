# EXPLORATORY / POST HOC: the host-price miss, an exploratory pay screen, P3's delay injection (28 Nov 2026)

These are items 10, 11 and 12 of the extension to the 28 Nov 2026 paper-quality wave. **Everything here is EXPLORATORY / POST HOC.**
- Nothing is scored or rescored.
- No host price, threshold, estimator or matching rule is changed.
- BUNGA is not refitted, and no new measurement was taken.
- The pin `0e0fb012` is untouched. H1 stays P2 0.1183 and P4 0.2495, and both fail.

The numbers come from:
- `exploratory/null_kernel_bias.{md,json}`, written by `scripts/null_kernel_bias.py` (module `attnbench/exploratory/host_bias.py`);
- `exploratory/pay_bound.{md,json}`, written by `scripts/pay_bound.py` (module `attnbench/exploratory/pay_bound.py`).

Intervals are percentile bootstraps at the lock's settings (level 0.95, 10,000 resamples, seed 0). They resample whole cells, or whole layers for S4.

## Item 10: how far the host prices overstate host time

**The mechanism is the per-layer idle-gap residual.** The host miss is not a null-kernel bias: the committed prices do not read the null pass. Nor is it a flaw in TaxBreak's method, which this study does not use as TaxBreak wrote it (see the end of this item).

**A premise to correct first.** P2's and P4's committed predictions do not read null-kernel prices. By the ruling of 1 Nov 2026, they read:
- the **bare timer**: each call re-issued alone at its prefill inputs, at least 1,000 times;
- plus the **per-layer idle-gap residual**: the arm's Python between calls, timed with the device synchronised before each gap;
- plus a handful of positions priced by the isolated replay.

The **null pass**, which substitutes a null kernel for every launching call under the tape's and the null device's Python modes, is banked beside them as the instrumented comparison. Both are reported below.

**Direction of every bound.** The host issues every operation before the timer stops. So priced host ÷ measured prefill is a lower bound on priced host ÷ true host. A ratio above 1 proves an overstatement; a ratio below 1 says nothing.

| | P2 (A100) | P3 (L4, AWS) | P4 (H100) |
|---|---|---|---|
| Cells where the committed host price exceeds the whole measured prefill (all of them beyond the measured median's interval) | 17 of 35 | 2 of 35 | 23 of 35 |
| Committed ÷ measured in those cells: median (max) | 1.60 (2.09) | 1.09 (1.10) | 1.39 (2.14) |
| Cells where the **bare** part alone exceeds the measured prefill | 0 | 0 | 0 |
| Cells where the **residual** part alone exceeds the measured prefill | 14 | 0 | 18 |
| Residual's share of the committed host price (median cell) | 0.67 | 0.63 | 0.71 |
| Null pass ÷ committed, non-oracle cells | 1.40 [1.37, 1.43] | 1.45 [1.42, 1.48] | 1.31 [1.28, 1.34] |
| Null pass ÷ committed, oracle cells | 6.94 [4.97, 7.53] | 8.21 [5.50, 9.14] | 6.61 [4.49, 7.33] |

**Two references with a host timeline.**
- **P4 against nsys** (the banked H100 diagnostic, Q4, XAttention cells): the committed host price is 1.65x, 1.53x and 1.68x the nsys host time at 8192, 16384 and 32768. The 8192 run was flagged not comparable, because its wall time was 1.11x P4's.
- **S4** (L4, 1 Nov 2026, dense at 1951): the null pass is 1.78x [1.72, 1.86] the recording run's host span per layer over 28 layers, and 1.73x for the whole prefill. The recording is itself profiled, so this too is a lower bound.

**Per operation class** (null pass ÷ committed, same positions, oracle cells left out):
- Every class lies between 1.17 and 1.68, on all three pairings.
- Copies and fill/create are highest, at 1.38 to 1.68.
- Attention and reductions are lowest, at 1.17 to 1.29.
- The full table, with intervals, is in `exploratory/null_kernel_bias.md`.
- The residual is shared out within a layer in proportion to the null pass's gaps (the 1 Nov ruling). So class differences come from the bare part, and the class split is only as good as that allocation.

**Reading.**
1. **The committed host prices overstate host time** on the A100 and H100 pairings: by at least 1.4x to 1.6x in the host-bound cells, and by 1.5x to 1.7x against nsys on P4.
2. **The overstatement lives in the idle-gap residual, not in the bare call prices.** The bare part never exceeds a measured prefill, while the residual alone does in 14 P2 cells and 18 P4 cells.
3. **The null pass would have been worse.** It is a further 1.3x to 1.45x above the committed price. For the oracle it is 6.6x to 8.2x, because the oracle's CPU mask builder runs inside the null pass's gaps under both Python modes (the T3.2c finding of 9 Nov 2026).
4. **P3 is mostly device-bound on its L4**, so its lower bound is informative in only 2 cells.

**Does TaxBreak use the same null-kernel method?** No. TaxBreak (arXiv:2603.12465, ISPASS 2026) uses a null kernel only to measure a constant launch floor per kernel. It measures per-operation dispatch by replaying each real ATen operation in isolation, which is closer to our bare timer than to our null pass. Quoted from its Section III:
- "∆KT=T floor sys is the hardware floor from cudaLaunchKernel call to GPU kernel start, measured via null-kernel profiling."
- "T floor sys is the mean T raw launch of an empty C++ __global__ null kernel profiled under the same protocol."
- "Phase 2: Isolation replay. We first measure T floor sys with a null-kernel nsys run. Then, for each kernel-database entry, we replay the original ATen operation in isolation under nsys."

The plan's G2 row already describes our adaptation as "null kernels across the whole operation sequence rather than the launch path alone". **So nothing here is a finding against TaxBreak.** Its launch-floor use of a null kernel is not what overstates our host prices. The null pass is this study's own extension, and it is not what the committed prices read either. The overstatement sits in the per-layer idle-gap residual.

## Item 11: an exploratory screen for when sparse prefill cannot pay

**A screen, not a theorem.** It is a lower bound under stated assumptions, built and checked after the fact on the same banked cells. The assumptions are one stream, in-order issue, and the bare-timed null kernel's mean taken as the least issue cost per operation. f is itself a measured mean, so the bound can fail on a cell, and the traced reading does fail as a floor on 2 P2 cells (below).

**Construction** (`attnbench/exploratory/pay_bound.py`). Under those assumptions:
- the prefill cannot end before its device work is done (device-busy time B);
- each in-step synchronisation drains the device, and the next operation needs at least one launch floor f to be issued after the synchronisation returns;
- the host issues N operations at no less than f each.

So T_sparse ≥ LB = max(B + n_sync·f, N·f), and a sparse cell can pay only if LB < T_dense. The inputs are:
- B, either the sparse arm's traced device-busy time or the committed priced device chain (kernels measured on the pairing before the workload is timed);
- f, the pairing's bare-timed null kernel;
- N and n_sync, counted from the priced sequence;
- T_dense, the dense cell's measured median.

Nothing is fitted on P1 to P4.

| | B read from | Sparse cells | Measured wins | Wins ruled out (violations) | Losses ruled out | LB above the measured sparse time |
|---|---|---|---|---|---|---|
| P2 | traced | 29 | 4 | 0 | 17 of 25 | 2 |
| P2 | priced, before timing | 29 | 4 | 0 | 14 of 25 | 0 |
| P4 | traced | 29 | 1 | 0 | 26 of 28 | 0 |
| P4 | priced, before timing | 29 | 1 | 0 | 26 of 28 | 0 |

**Verdict: kept, as an exploratory screen.** The screen rules out no measured win, and it rules out 14 to 26 measured losses. The before-timing reading held as a floor on every banked cell (T_sparse ÷ LB ≥ 1.03). The traced reading is not a floor in 2 P2 cells, where profiling stretches the device-busy time.

**Limits, stated plainly.**
- **The device term sets LB in every cell.** The synchronisation term adds at most 476 × 6 µs, about 3 ms, and the host term N·f never binds. In effect the screen reads: "the sparse arm's device chain, plus one launch floor per synchronisation, must beat dense's measured time".
- **It needs dense's measured time.** A version built from components on both sides would need an upper bound on dense's host time, which item 10 shows is not reliable from the committed prices.
- **It is post hoc** and labelled so.
- **P3 is not checked**, because its traced ground truth is not read by these loaders.

## Item 12: P3's delay-injection output

**Complete and usable; not analysed.** In `aws_p3_g6_20261126T0906Z/session/12_delay_injection/`:
- **The step passed.** It ran from 12:24:20 to 12:28:04 UTC on 26 Nov 2026 and exited 0, with `status: passed` and no refusals.
- **The plan is complete.** It covers 64 variants: 8 mechanisms × 2 placements × 4 counts, on the dense cell at 2048 tokens, batch 1, with 10 repetitions and 3 warm-up runs each.
- **The plan was stored before timing.** It was stored at 12:25:40 and read back at 12:26:03. Its canonical content equals the stored copy (`leg3_plan.json`), sha256 `a0686da4…`.
- **Every variant has its 10 samples**, all finite, plus the unobserved control (65 sample lists). All 8 mechanisms read `measured`, with none unresolved.
- **The file is banked intact.** `delay_injection.json`'s sha256 matches the pulled block-3 manifest (`3799a172…`).
- **It is one cell** (dense, 2048 tokens, batch 1). Any analysis must be pre-stated before its slopes are read.

## Plan defaults of this extension (flagged)

- **Operation classes** are the module's `CLASSES` table. A call with no aten operator is an `entry` (a Python-called kernel).
- **Oracle cells are left out of the class table** and reported on their own.
- **P3's cells are measured against P1's committed cell keys**, as the P3 exploratory score does. P3's host prices are its own session's, not the P1 transfer's.
- **S4 has one cell.** It is the only S4 cell whose recording trace is banked. The 1 Nov 2026 meanpool and oracle figures (2.47x, 2.20x) came from scripts in a since-purged scratchpad and are not reproduced.
- **P4's nsys reference is the banked Q4 verdict**, not recomputed.
