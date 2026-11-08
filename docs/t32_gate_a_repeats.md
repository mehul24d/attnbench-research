# Gate A over T3.2a and T3.2b (T3.3 inputs, 8 Nov 2026)

**What this is.** `scripts/gate_a_repeats.py` (`attnbench/analysis/gate_a_repeats.py`)
run once, offline on the Mac, on both P1 repeats (`results/t32a_p1_20261107T1640Z`,
`results/t32b_p1_20261107T2100Z`, both at `3f50aab`, eager group), from their bucket
artifacts: each session's committed prediction read through `load_committed` against
the store (60 s skew; both loaded), its eager ground truth, step 6's `cells.json` and
step 5's recordings of all 35 cells. Each session is scored against its own ground
truth. The floors are `lock.floors` over both sessions' clean repeats at the
provisional quantile 0.95 (lock table row 17); they are not lock values. **It is not
the Gate A decision**, which is the researcher's (T3.3).

**The plan's thresholds as written** (Section 7), with nothing widened:

- Leg 1 (the recursion beats the sum and the step form on P1) is read as: the
  recursion's median absolute relative error below each rival's over every cell, in
  each session. It **fails** in both: the recursion beats the sum (13.8% against
  96.8% in T3.2a, 14.5% against 98.5% in T3.2b) and loses to the step form (3.8% in
  T3.2a, 3.6% in T3.2b). The loss is in the synchronising arms (XAttention 28 to 32%
  against 3 to 4%; oracle 14% against 11% in T3.2a, 20% against 17% in T3.2b); on dense
  the recursion is ahead in T3.2a (0.7% against 1.1%) and level in T3.2b (1.54% against
  1.53%).
- Leg 2 (replay against trace per family, within a bound fixed at lock) is **not
  applied**: no bound is set. The distribution reproduces between the repeats: priced
  device time +2.4% and +2.7% overall; `masked_fill_` +116% and +120%.
- Leg 3 (delay-injection slopes) is **not applied**: no T3.1 step runs delay injection.

So Gate A does not pass from these data as written: leg 1 fails, and legs 2 and 3
cannot be read until the lock sets the family bound and a session measures the slopes.

**Floors** (provisional quantile): the noise floor is 6.77% (between-session spread of a
cell's median, 35 cells timed in both); the resolution floor is 19.42 ms (pooled
|repeat - series median|, 70 series, 700 repeats). T3.2b ran faster than T3.2a at 16384 and
32768 tokens, by 5 to 7% in dense, the oracle and mean-pool and by 3 to 3.5% in the
XAttention arms; by 2 to 7% at 8192; within 2% at 2048 and 4096 (T3.2b slightly slower
at 2048); the per-cell table below is the between-session spread the lock reads.

**Same-session sanity, per arm** (descriptive, within one session; not a hypothesis
test): below each session's leg 1, the median absolute relative error per arm and the
sign agreement of sparse against dense at the same length and batch, a measured
difference inside the 19.42 ms resolution floor not scored.

Reproduce: `python3 scripts/gate_a_repeats.py --label "..." --repeat T3.2a <remote>
<local> <09_ground_truth> <cells.json> <recordings> --repeat T3.2b ... --floors-quantile
0.95 --floors-provisional --out <dir>`.

---

## The report: T3.2a and T3.2b (P1, eager, 3f50aab)

Sessions: T3.2a (prediction `80e4696400f6`), T3.2b (prediction `db104506753f`). Decision: the researcher's (T3.3); this is the plan's text applied to the inputs.

## The plan's thresholds as written: Gate A does not pass

| Leg | Verdict | Detail |
|---|---|---|
| 1: recursion below the sum and the step form | fails | T3.2a: sum yes, step_form no; T3.2b: sum yes, step_form no |
| 2: replay against trace per family | not applied | the per-family bound is fixed at lock (Appendix A, 27 Oct 2026); none is set |
| 3: delay-injection slopes | not applied | the slopes are not measured in T3.2a, T3.2b |

## Floors at quantile 0.95 (provisional quantile, not a lock value)

- Noise floor (between-session spread of a cell's median, (max - min) / median, over 35 cells timed in both sessions): **0.0677**.
- Resolution floor (|repeat - series median|, pooled over 70 series, 700 repeats): **19.42 ms**.
- Cells timed in one session only: 0.

| Cell | T3.2a median ms | T3.2b median ms | Between | T3.2a within | T3.2b within |
|---|---|---|---|---|---|
| dense L2048 b1 | 121.3 | 123.7 | 0.0190 | 0.1009 | 0.1039 |
| dense L4096 b1 | 282.4 | 280.6 | 0.0062 | 0.0580 | 0.0472 |
| dense L8192 b1 | 733.9 | 686.3 | 0.0671 | 0.0537 | 0.0327 |
| dense L16384 b1 | 1729.9 | 1644.8 | 0.0504 | 0.0194 | 0.0275 |
| dense L16384 b4 | 7021.4 | 6607.1 | 0.0608 | 0.0080 | 0.0030 |
| dense L32768 b1 | 4270.9 | 4023.3 | 0.0597 | 0.0121 | 0.0079 |
| meanpool@0.25 L2048 b1 | 135.7 | 138.1 | 0.0176 | 0.0148 | 0.0191 |
| meanpool@0.25 L4096 b1 | 300.0 | 298.3 | 0.0056 | 0.0780 | 0.0313 |
| meanpool@0.25 L8192 b1 | 694.0 | 679.0 | 0.0219 | 0.0616 | 0.0273 |
| meanpool@0.25 L16384 b1 | 1529.5 | 1447.0 | 0.0554 | 0.0282 | 0.0103 |
| meanpool@0.25 L32768 b1 | 3288.6 | 3084.1 | 0.0642 | 0.0140 | 0.0075 |
| meanpool@0.5 L2048 b1 | 136.8 | 139.4 | 0.0191 | 0.0105 | 0.0170 |
| meanpool@0.5 L4096 b1 | 308.7 | 304.2 | 0.0149 | 0.0668 | 0.0566 |
| meanpool@0.5 L8192 b1 | 754.3 | 704.0 | 0.0690 | 0.0796 | 0.0234 |
| meanpool@0.5 L16384 b1 | 1648.1 | 1557.0 | 0.0568 | 0.0131 | 0.0082 |
| meanpool@0.5 L32768 b1 | 3642.6 | 3442.8 | 0.0564 | 0.0096 | 0.0134 |
| oracle@0.25 L2048 b1 | 134.8 | 136.9 | 0.0156 | 0.0165 | 0.0539 |
| oracle@0.25 L4096 b1 | 282.0 | 283.7 | 0.0060 | 0.0364 | 0.0205 |
| oracle@0.25 L8192 b1 | 673.3 | 659.2 | 0.0211 | 0.0330 | 0.0414 |
| oracle@0.25 L16384 b1 | 1480.2 | 1403.9 | 0.0529 | 0.0140 | 0.0119 |
| oracle@0.25 L32768 b1 | 3195.3 | 2988.4 | 0.0669 | 0.0107 | 0.0159 |
| oracle@0.5 L2048 b1 | 135.4 | 138.2 | 0.0202 | 0.0177 | 0.0384 |
| oracle@0.5 L4096 b1 | 291.3 | 293.6 | 0.0079 | 0.0465 | 0.0258 |
| oracle@0.5 L8192 b1 | 722.6 | 673.7 | 0.0702 | 0.0485 | 0.0337 |
| oracle@0.5 L16384 b1 | 1609.2 | 1516.4 | 0.0594 | 0.0141 | 0.0224 |
| xattention L2048 b1 | 165.2 | 168.1 | 0.0171 | 0.0126 | 0.0106 |
| xattention L4096 b1 | 333.1 | 335.0 | 0.0056 | 0.0253 | 0.0228 |
| xattention L8192 b1 | 848.7 | 830.2 | 0.0220 | 0.0554 | 0.0179 |
| xattention L16384 b1 | 2104.4 | 2042.8 | 0.0297 | 0.0141 | 0.0079 |
| xattention L32768 b1 | 5180.8 | 5005.3 | 0.0345 | 0.0063 | 0.0071 |
| xattention_noassert L2048 b1 | 162.4 | 165.5 | 0.0189 | 0.0158 | 0.0104 |
| xattention_noassert L4096 b1 | 331.2 | 332.2 | 0.0031 | 0.0234 | 0.0265 |
| xattention_noassert L8192 b1 | 844.2 | 825.5 | 0.0224 | 0.0448 | 0.0107 |
| xattention_noassert L16384 b1 | 2106.5 | 2039.8 | 0.0322 | 0.0101 | 0.0065 |
| xattention_noassert L32768 b1 | 5164.6 | 4988.7 | 0.0347 | 0.0090 | 0.0037 |

### The noise floor read as n = 2 (adviser's ruling E2 of 8 Nov 2026)

The 6.77% is the 0.95 quantile over 35 cells of the gap between **two**
sessions. It is the largest systematic gap observed in those sessions, not
an interval.

- Over the 35 cells the between-session spread has median 2.24%, p90 6.58%
  and max 7.02% (oracle@0.5 L8192).
- The shift (T3.2b − T3.2a) / median is not one sign: 24 cells negative,
  11 positive, median −2.24%. It splits by length. Every cell at 2048 is
  positive (T3.2b slower, +1.6% to +2.0%). At 4096, four are positive and
  three negative. Every cell from 8192 up is negative (T3.2b faster, up to
  −7.0%).
- A between-session factor that scaled every cell alike would carry one
  sign. This one depends on length.

### The resolution floor per (length, batch) (rule committed in `2b08e0d`, Appendix A, 8 Nov 2026)

Quantile 0.95, every arm's series at the cell, both repeats
(`floors.resolution_floors_by_length`; script `scratchpad/tools/floors_e12.py`,
outside git). The scored share was computed after the rule was committed.

| Cell | Floor ms | Series | Sparse-vs-dense gaps | Unscored (inside the floor) |
|---|---|---|---|---|
| L2048 b1 | 1.60 | 14 | 12 | 0 |
| L4096 b1 | 10.02 | 14 | 12 | 3 |
| L8192 b1 | 25.49 | 14 | 12 | 5 |
| L16384 b1 | 15.95 | 14 | 12 | 0 |
| L16384 b4 | 16.76 | 2 | 0 (dense only) | 0 |
| L32768 b1 | 25.94 | 12 | 10 | 0 |
| **All** | pooled 19.42 | 70 | 58 | **8; scored 50 of 58 (86.2%)** |

- At 2048 the cell floor (1.60 ms) is below the dense series' own 0.95
  deviation (6.3 to 6.7 ms). The pool is dominated by the sparse arms'
  tighter series.
- The floor is not monotone in length (8192 is above 16384). Each cell is
  read against its own floor.

## Per session

### T3.2a leg 1

| Scope | Predictor | n | Median abs. rel. error | Signed mean | Max |
|---|---|---|---|---|---|
| all | recursion | 35 | 0.1384 | +0.1783 | 0.6406 |
| all | sum | 35 | 0.9675 | +0.7298 | 1.5668 |
| all | step_form | 35 | 0.0381 | +0.0951 | 0.6029 |
| all | recursion below | | sum: yes, step_form: no | | |
| dense | recursion | 6 | 0.0068 | -0.0013 | 0.0258 |
| dense | sum | 6 | 0.1029 | +0.1535 | 0.4643 |
| dense | step_form | 6 | 0.0105 | -0.0030 | 0.0257 |
| dense | recursion below | | sum: yes, step_form: yes | | |
| synchronisation-free | (no cells) | | | | |
| synchronising | recursion | 29 | 0.1772 | +0.2154 | 0.6406 |
| synchronising | sum | 29 | 0.9989 | +0.8490 | 1.5668 |
| synchronising | step_form | 29 | 0.0513 | +0.1155 | 0.6029 |
| synchronising | recursion below | | sum: yes, step_form: no | | |

### T3.2a per arm (descriptive, within one session: not a hypothesis test (P1 is the development pairing))

Sign agreement against the dense cell of the same length and batch; a measured difference within 19.4 ms is not scored.

| Arm | Predictor | n | Median abs. rel. error | Sign agrees | Disagrees | Inside floor |
|---|---|---|---|---|---|---|
| dense | recursion | 6 | 0.0068 |  |  |  |
| dense | sum | 6 | 0.1029 |  |  |  |
| dense | step_form | 6 | 0.0105 |  |  |  |
| meanpool | recursion | 10 | 0.0405 | 7 | 0 | 3 |
| meanpool | sum | 10 | 0.1851 | 6 | 1 | 3 |
| meanpool | step_form | 10 | 0.0423 | 6 | 1 | 3 |
| oracle | recursion | 9 | 0.1437 | 2 | 2 | 5 |
| oracle | sum | 9 | 1.1130 | 0 | 4 | 5 |
| oracle | step_form | 9 | 0.1071 | 2 | 2 | 5 |
| xattention | recursion | 5 | 0.3180 | 5 | 0 | 0 |
| xattention | sum | 5 | 0.9989 | 5 | 0 | 0 |
| xattention | step_form | 5 | 0.0336 | 5 | 0 | 0 |
| xattention_noassert | recursion | 5 | 0.3057 | 5 | 0 | 0 |
| xattention_noassert | sum | 5 | 1.0162 | 5 | 0 | 0 |
| xattention_noassert | step_form | 5 | 0.0442 | 5 | 0 | 0 |

### T3.2b leg 1

| Scope | Predictor | n | Median abs. rel. error | Signed mean | Max |
|---|---|---|---|---|---|
| all | recursion | 35 | 0.1448 | +0.1821 | 0.7353 |
| all | sum | 35 | 0.9854 | +0.7313 | 1.6746 |
| all | step_form | 35 | 0.0362 | +0.1008 | 0.6979 |
| all | recursion below | | sum: yes, step_form: no | | |
| dense | recursion | 6 | 0.0154 | +0.0132 | 0.0214 |
| dense | sum | 6 | 0.1074 | +0.1673 | 0.4592 |
| dense | step_form | 6 | 0.0153 | +0.0117 | 0.0213 |
| dense | recursion below | | sum: yes, step_form: no | | |
| synchronisation-free | (no cells) | | | | |
| synchronising | recursion | 29 | 0.2022 | +0.2171 | 0.7353 |
| synchronising | sum | 29 | 0.9918 | +0.8480 | 1.6746 |
| synchronising | step_form | 29 | 0.0409 | +0.1193 | 0.6979 |
| synchronising | recursion below | | sum: yes, step_form: no | | |

### T3.2b per arm (descriptive, within one session: not a hypothesis test (P1 is the development pairing))

Sign agreement against the dense cell of the same length and batch; a measured difference within 19.4 ms is not scored.

| Arm | Predictor | n | Median abs. rel. error | Sign agrees | Disagrees | Inside floor |
|---|---|---|---|---|---|---|
| dense | recursion | 6 | 0.0154 |  |  |  |
| dense | sum | 6 | 0.1074 |  |  |  |
| dense | step_form | 6 | 0.0153 |  |  |  |
| meanpool | recursion | 10 | 0.0215 | 5 | 0 | 5 |
| meanpool | sum | 10 | 0.1950 | 5 | 0 | 5 |
| meanpool | step_form | 10 | 0.0267 | 5 | 0 | 5 |
| oracle | recursion | 9 | 0.2022 | 2 | 2 | 5 |
| oracle | sum | 9 | 1.1221 | 0 | 4 | 5 |
| oracle | step_form | 9 | 0.1729 | 2 | 2 | 5 |
| xattention | recursion | 5 | 0.2997 | 5 | 0 | 0 |
| xattention | sum | 5 | 0.9918 | 5 | 0 | 0 |
| xattention | step_form | 5 | 0.0404 | 5 | 0 | 0 |
| xattention_noassert | recursion | 5 | 0.2764 | 5 | 0 | 0 |
| xattention_noassert | sum | 5 | 1.0055 | 5 | 0 | 0 |
| xattention_noassert | step_form | 5 | 0.0362 | 5 | 0 | 0 |
