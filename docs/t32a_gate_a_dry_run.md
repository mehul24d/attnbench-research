# Gate A dry run on T3.2a alone (8 Nov 2026)

**What this is.** The Gate A tooling (`scripts/gate_a.py`, T3.3) run once
on T3.2a (`results/t32a_p1_20261107T1640Z`, `3f50aab`), offline on the Mac,
from the session's bucket artifacts: the committed prediction read through
`load_committed` against the store (60 s skew; it loaded), the eager ground
truth, step 6's `cells.json` and step 5's recordings of all 35 cells. **It is
not the Gate A decision**: that reads both T3.2 repeats, the floors and the
family bound are lock values not yet set (run here with none), and the
delay-injection leg has no data. The per-arm table is descriptive. P1 is the
development pairing.

**What it shows, read plainly.**

- Leg 1. On this one repeat the recursion's median absolute relative error is
  13.8% over 35 cells, below the sum's (96.8%) and above the step form's
  (3.8%). It is below both only on the dense cells (0.7% against 1.0%). On the
  synchronising cells (every sparse cell: each has a synchronisation inside
  the step) it is 17.7% against the step form's 5.1%, and it over-predicts
  (signed mean +21.5%). XAttention is the largest gap: about 31% against
  3 to 4%. No cell is synchronisation-free.
- Leg 2. Over all 35 cells, the device time the prediction priced is 2.4%
  above the traced recording's (52.02 s against 50.80 s). Two families stand
  out: `masked_fill_` (in every sparse cell; for the oracle it is one of the two
  families priced from the isolated original) at +116% pooled (15.3 ms against 7.1 ms over 29
  cells) and up to 6.1 x in one cell, and the two `aten::all` memsets at
  +103% (0.18 ms against 0.09 ms). `device_copy:aten::copy_` has a q90 of
  1.40 in relative terms on small copies, +4.7% pooled.
- Leg 3: not measured. T3.1's steps include no delay injection (T2.25).
- The descriptive sign agreement (sparse against dense at the same length and
  batch, no floor): the recursion agrees on every mean-pool and XAttention
  cell and on 5 of 9 oracle cells.

Reproduce: `python3 scripts/gate_a.py --label "..." --remote <the committed
file in the bucket> --local <its copy> --ground-truth <09_ground_truth>
--cells <06_device_components/cells.json> --recordings <a directory of the
35 cells' step-5 trace.json and sequence.json> --relative-floor 0
--resolution-floor-ms 0 --out <dir>`.

The report follows as the tool wrote it.

---

## Gate A inputs: T3.2a alone, dry run: one repeat, not the Gate A decision; floors and the family bound not set (lock values)

Prediction `80e4696400f6`, P1, the median; 35 cells measured, 0 committed cells not measured. Decision: the researcher's (T3.3); this report gives the inputs only.

### Leg 1: the recursion against the sum and the step form

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

### Leg 2: replay against trace, per family (35 cells; bound none)

| Family | Cells | Positions | Pooled rel. diff. | Median abs. | q90 abs. | Max abs. |
|---|---|---|---|---|---|---|
| device_copy:aten::cat | 8 | 448 | -0.0254 | 0.0658 | 0.1374 | 0.1374 |
| device_copy:aten::copy_ | 29 | 6580 | +0.0471 | 0.1145 | 1.3974 | 2.6874 |
| kernel:aten::_flash_attention_forward | 6 | 168 | -0.0494 | 0.0616 | 0.2007 | 0.2007 |
| kernel:aten::_index_put_impl_ | 10 | 448 | -0.0788 | 0.0847 | 0.1663 | 0.1663 |
| kernel:aten::_softmax | 20 | 728 | +0.0034 | 0.0505 | 0.1708 | 0.2262 |
| kernel:aten::add | 35 | 7091 | +0.0029 | 0.0060 | 0.1181 | 0.2351 |
| kernel:aten::addmm | 35 | 2940 | +0.0055 | 0.0875 | 0.2092 | 0.2674 |
| kernel:aten::all | 5 | 308 | -0.0517 | 0.0498 | 0.1133 | 0.1133 |
| kernel:aten::arange | 35 | 4123 | -0.0842 | 0.1235 | 0.4249 | 0.5304 |
| kernel:aten::bitwise_and | 20 | 840 | -0.1090 | 0.1028 | 0.2138 | 0.2183 |
| kernel:aten::bitwise_not | 11 | 364 | -0.1148 | 0.1288 | 0.1612 | 0.1649 |
| kernel:aten::bmm | 35 | 763 | +0.0171 | 0.0805 | 0.4867 | 0.5384 |
| kernel:aten::cat | 35 | 4907 | +0.0025 | 0.0310 | 0.1598 | 0.2196 |
| kernel:aten::copy_ | 35 | 18319 | +0.0055 | 0.0194 | 0.1602 | 0.2059 |
| kernel:aten::cos | 35 | 35 | +0.0560 | 0.2959 | 0.4425 | 0.4991 |
| kernel:aten::cumsum | 29 | 2072 | +0.0007 | 0.0954 | 0.3049 | 0.6640 |
| kernel:aten::div | 10 | 1624 | +0.0022 | 0.0120 | 0.1853 | 0.1853 |
| kernel:aten::eq | 29 | 2436 | +0.0182 | 0.0888 | 0.3135 | 0.6623 |
| kernel:aten::fill_ | 29 | 12814 | +0.0098 | 0.0682 | 0.3229 | 0.4186 |
| kernel:aten::flip | 29 | 812 | -0.0141 | 0.1134 | 0.2981 | 0.6423 |
| kernel:aten::gather | 35 | 35 | -0.1307 | 0.1424 | 0.5314 | 0.5574 |
| kernel:aten::gt | 10 | 280 | -0.1979 | 0.1951 | 0.2549 | 0.2549 |
| kernel:aten::index | 29 | 812 | -0.1668 | 0.1951 | 0.3367 | 0.4683 |
| kernel:aten::lt | 20 | 1008 | -0.1054 | 0.1267 | 0.2127 | 0.2162 |
| kernel:aten::masked_fill_ | 29 | 1820 | +1.1603 | 0.5948 | 2.9219 | 6.0695 |
| kernel:aten::masked_scatter_ | 29 | 3248 | -0.0325 | 0.1148 | 0.2829 | 0.6652 |
| kernel:aten::mean | 35 | 3115 | +0.0003 | 0.0166 | 0.1690 | 0.2094 |
| kernel:aten::mm | 35 | 3955 | +0.0413 | 0.0767 | 0.2480 | 0.4943 |
| kernel:aten::mul | 35 | 11956 | +0.0009 | 0.0081 | 0.0411 | 0.0490 |
| kernel:aten::ne | 10 | 280 | -0.1151 | 0.1239 | 0.1858 | 0.1858 |
| kernel:aten::neg | 35 | 1960 | +0.0011 | 0.0293 | 0.1464 | 0.2098 |
| kernel:aten::nonzero | 29 | 2436 | -0.2238 | 0.2569 | 0.3804 | 0.3873 |
| kernel:aten::pow | 35 | 1995 | +0.0020 | 0.0032 | 0.2196 | 0.3214 |
| kernel:aten::round | 10 | 280 | -0.1082 | 0.1166 | 0.1852 | 0.1852 |
| kernel:aten::rsqrt | 35 | 1995 | -0.0270 | 0.1575 | 0.3181 | 0.4390 |
| kernel:aten::scatter_ | 10 | 280 | -0.1569 | 0.1670 | 0.2088 | 0.2088 |
| kernel:aten::silu | 35 | 980 | +0.0001 | 0.0024 | 0.0076 | 0.0104 |
| kernel:aten::sin | 35 | 35 | +0.0388 | 0.3189 | 0.4638 | 0.5257 |
| kernel:aten::sort | 29 | 1988 | +0.0330 | 0.0614 | 0.4434 | 1.0302 |
| kernel:aten::sum | 29 | 3444 | +0.0082 | 0.1845 | 0.3411 | 0.3822 |
| kernel:aten::tril | 10 | 560 | -0.1745 | 0.1884 | 0.2489 | 0.2489 |
| kernel:aten::triu | 10 | 448 | -0.0023 | 0.0085 | 0.1384 | 0.1384 |
| kernel:aten::where | 10 | 2128 | -0.0768 | 0.0820 | 0.1472 | 0.1472 |
| kernel:flash_attn::_block_sparse_attn_forward | 29 | 812 | +0.1081 | 0.0975 | 0.5213 | 0.5705 |
| memset:aten::addmm | 28 | 1764 | -0.0721 | 0.1103 | 0.1904 | 0.1961 |
| memset:aten::all | 2 | 168 | +1.0284 | 0.9975 | 1.0442 | 1.0442 |
| memset:aten::mm | 21 | 784 | +0.1680 | 0.0192 | 0.5561 | 0.6068 |
| memset:aten::sum | 4 | 112 | -0.0112 | 0.0342 | 0.1074 | 0.1074 |

### Leg 3: delay-injection slopes

not measured: the session ran no delay injection (T2.25); Gate A needs one slope per synchronisation mechanism on the L4, busy and idle

### Per arm (descriptive, within one session: not a hypothesis test (P1 is the development pairing))

Sign agreement against the dense cell of the same length and batch; a measured difference within 0.0 ms is not scored.

| Arm | Predictor | n | Median abs. rel. error | Sign agrees | Disagrees | Inside floor |
|---|---|---|---|---|---|---|
| dense | recursion | 6 | 0.0068 |  |  |  |
| dense | sum | 6 | 0.1029 |  |  |  |
| dense | step_form | 6 | 0.0105 |  |  |  |
| meanpool | recursion | 10 | 0.0405 | 10 | 0 | 0 |
| meanpool | sum | 10 | 0.1851 | 9 | 1 | 0 |
| meanpool | step_form | 10 | 0.0423 | 9 | 1 | 0 |
| oracle | recursion | 9 | 0.1437 | 5 | 4 | 0 |
| oracle | sum | 9 | 1.1130 | 3 | 6 | 0 |
| oracle | step_form | 9 | 0.1071 | 5 | 4 | 0 |
| xattention | recursion | 5 | 0.3180 | 5 | 0 | 0 |
| xattention | sum | 5 | 0.9989 | 5 | 0 | 0 |
| xattention | step_form | 5 | 0.0336 | 5 | 0 | 0 |
| xattention_noassert | recursion | 5 | 0.3057 | 5 | 0 | 0 |
| xattention_noassert | sum | 5 | 1.0162 | 5 | 0 | 0 |
| xattention_noassert | step_form | 5 | 0.0442 | 5 | 0 | 0 |
