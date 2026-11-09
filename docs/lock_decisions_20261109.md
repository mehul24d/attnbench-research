# Lock decisions: the 11 open items, each with a default (9 Nov 2026)

Adviser's ruling 5 of the ninth 9 Nov 2026 paste. It replaces
`docs/lock_decisions_20261108.md`, whose items 12 and 13 are settled. These
are the T4.1 values whose status in `docs/lock_package_draft.json` is
"owed: decision". **User** marks an item that needs the researcher, because
it sets scientific direction or spend. The adviser rules on the rest in one
pass. Target: all 11 settled by 12 Nov 2026.

| # | Item (T4.1) | Proposed default | Needs |
|---|---|---|---|
| 1 | H0 to H3 tolerances | 10% absolute relative error for H0 (per operation family) and for H1 and H2 (median per pairing). H3 has no tolerance beyond the bootstrap widening it already names. None may be below the 6.77% noise floor. P1 recursion: 5.05% median, 59.7% max (T3.2d) | **User**: scientific |
| 2 | Ground-truth density set | P1's set: dense 1.0; block-sparse 0.25 and 0.5; XAttention calibrated. Oracle 0.5 at 32768 not measured. Compiled arms only if their ground truth is resolved by 14 Nov (Appendix A, 9 Nov 2026) | **User**: scientific, and spend if the missing cell is measured |
| 3 | H4 equivalence bound | 10 points (T1.1: −4.3 and −6.0) | **User**: scientific |
| 4 | H4 confirmatory family | m counted from the declared family (audit D4) | **User**: scientific |
| 5 | Floors quantile | 0.95 for both floors. With n = 2 sessions (T3.2c, T3.2d) it is close to the largest gap | **User**: scientific |
| 6 | HDBI boundedness threshold | Fitted on P1 by sign agreement, rows inside their cell's floor dropped, two classes (audit D3) | default |
| 7 | K2 correctness tolerance | The gate's `TOL` | default |
| 8 | Payload class edges | 8 B, 4 KiB, 1 MiB | default |
| 9 | Realised-length key rule | Key by the budget; realised tokens kept as a field | default |
| 10 | Budget cut order | As the plan: P7 first, then the chunked extension; never ground truth, H1 or H4 | default |
| 11 | Recompile gate | 0 compile events after warm-up | default |

**Changed from 8 Nov.**
- Item 1 now has a number; on 8 Nov it was "set after the floors", and the floors now exist.
- Item 6 moves from **User** to default. HDBI is a diagnostic only (Section 5.4), so its threshold sets no claim.
- Item 2 carries the compiled time-box.

**Settled outside this list.** The block drift bound is 6.77% (a flag). The
floor's form and rule are in Appendix A, 8 and 9 Nov 2026. The learned
predictor's configuration is config 4, owed only its lock entry.
