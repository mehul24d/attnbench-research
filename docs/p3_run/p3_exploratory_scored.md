# P3 prediction against P3's ground truth — **EXPLORATORY**

Scores no hypothesis. The 22 Nov 2026 file (full transfer from P1, sha256 `72b907cf4605…`) equals P1's committed values in all 35 cells. Measured cells: 35; excluded 0, unmeasured 0.

| predictor | median abs relative error |
|---|---|
| recursion | 0.0811 |
| step_form | 0.0408 |
| sum | 0.5767 |
| sum_plus_s0 | 0.5782 |

| arm class | n | recursion | step_form | sum | sum_plus_s0 |
|---|---|---|---|---|---|
| dense | 6 | 0.0449 | 0.0424 | 0.1258 | 0.1258 |
| synchronising | 29 | 0.1088 | 0.0408 | 0.6275 | 0.6297 |

Dense backend: priced `sdpa_flash`; P3's dense selection chose sdpa_flash at 2048x1, sdpa_flash at 4096x1, fa2 at 8192x1, fa2 at 16384x1, fa2 at 16384x4, fa2 at 32768x1.

| length | batch | chosen | predicted s | measured s | relative |
|---|---|---|---|---|---|
| 1951 | 1 | sdpa_flash | 0.1245 | 0.1182 | 0.053 |
| 4068 | 1 | sdpa_flash | 0.2938 | 0.2712 | 0.084 |
| 8075 | 1 | fa2 | 0.7147 | 0.6803 | 0.051 |
| 16370 | 1 | fa2 | 1.7050 | 1.6430 | 0.038 |
| 16370 | 4 | fa2 | 6.8507 | 6.5946 | 0.039 |
| 32700 | 1 | fa2 | 4.1307 | 3.9747 | 0.039 |

Dense recursion, median abs relative: kernel as priced 0.0684 (n 2), another kernel 0.0390 (n 4).
