# Lock items that can be filled now: 12 for one ruling pass (11 Nov 2026)

Adviser's ruling 1 of the 11 Nov 2026 paste (T4.1 due 12 Nov 2026). Twelve of the 30 open
T4.1 items are filled here: six with evidence in hand and six that are defaults the adviser
confirmed on 9 Nov 2026 (`docs/lock_decisions_20261109.md`). **User?** marks a value that
needs the researcher's own decision before the lock commit, not only the adviser's ruling.
Nothing here is fixed: values are fixed only by the lock commit (T4.4). The item numbers are
the plan's T4.1 order (`attnbench/lock/checklist.lock_items`).

| # | Item | Proposed value | Evidence | User? |
|---|---|---|---|---|
| 9 | Context grid against each model's position limit | Budgets 2048, 4096, 8192, 16384 and 32768 (realised prompts 1951 to 32700 tokens), prefill only | `configs/t61/p2_session.json` and `configs/t64/p4_session.json` time only the primary model. Its pinned `config.json` (989aa79, sha256 `98d2ff8c…`, checked against the Mac's cached copy on 11 Nov) has `max_position_embeddings` 32768 and `use_sliding_window` false. The largest realised prompt, 32700, is below it | |
| 13 | Replay-versus-trace bound (Gate A) | 0.25 | The largest family \|relative difference\| in the two passing L4 shakedowns, rechecked 11 Nov from `06_replay/replay_vs_trace.json`: S4 0.2434 and S5 0.2363 (both `aten::sum`). Written before it was applied to T3.2 | |
| 24 | Throttle set | HW slowdown, SW thermal slowdown, HW thermal slowdown and HW power brake flag a block. The SW power cap, idle and the session's own clock lock are recorded and never flag | Researcher's decision of 27 Oct 2026; `clock_sampling.RESEARCHER_FLAG_BITS` and `NEVER_FLAGS` | |
| 30 | Probe-length rule (T2.18) | Three lengths at or below 32768 with distinct chunk counts of at least two, else two plus a full recording at 32768. XAttention: 20480, 28672 and full at 32768 | Ruling of 31 Oct 2026; `structural_probe.chunked_probe_lengths` | |
| 31 | Flagged-memset sensitivity bound (T2.3) | The rule, not a number: each memset with no device record is priced from its realised bytes (zero bytes is zero device time; otherwise the memset zoo's duration for its size). The paper reports the count and the shift | Both P1 repeats: 532 flagged memsets in 7 cells, all zero bytes, so the shift is 0 s (`docs/limitations.md`; Appendix A, 8 Nov 2026) | ✓ The plan calls it a *bound*; this fills it with a pricing rule and a reported shift. Confirm that no numeric threshold is wanted |
| 36 | Block drift bound (T6.1, T6.4) | 6.77%, a flag and not an exclusion | The 0.95 quantile of the between-session gap, T3.2a against T3.2b, n = 2 (`docs/t32_gate_a_repeats.md`). Adviser's ruling of 8 Nov 2026; `analysis/block_drift.py` | |
| 18 | HDBI boundedness threshold | Method: one threshold and two classes, fitted on P1 (T3.2d) by sign agreement, ties to the lower value, rows inside their cell's resolution floor dropped. The value is computed at the lock commit, once item 16 (the floors) is fixed | `attnbench/lock/hdbi_fit.py` (tested); default confirmed 9 Nov 2026. **Gap:** `fit_threshold` takes one floor, but the floors have been per (length, batch) since 8 Nov 2026. Proposed: drop each row by its own cell's floor | ✓ The per-cell floor in the fit (a small code change, mutation-tested) |
| 21 | K2 correctness tolerance | The gate's `TOL`: bf16 2e-2, fp16 5e-3, fp32 1e-4 (atol = rtol) | `attnbench/gates.py`; default confirmed 9 Nov 2026. The lock table's condition ("keep until Shakedown 2's float64 gate shows the real error") is not cited as met | |
| 25 | Payload class edges | 8 B, 4 KiB, 1 MiB | Default confirmed 9 Nov 2026; there is no data against it | |
| 27 | Realised-length key rule | Each prediction is keyed by the prompt's realised token count; the budget is a label | Default as corrected 9 Nov 2026. The code already does this: T3.2a to T3.2d predictions are keyed 1951, 4068, 8075, 16370 and 32700 (`t31_steps.py` step 5) | |
| 28 | Budget cut order | P7 first, then the chunked extension, then the compiled group, then the rehearsal session. It never removes ground-truth timing, H1 or H4. P4 on Spot is a cost, not a cut, with the one exception the researcher can choose at the 11 Nov decision point | Plan Section 7, Budget row (10 and 11 Nov 2026). `docs/lock_value_table.md` row 28 still carries the older wording ("full grid for the XAttention pair, …") | ✓ The decision point's "fewer blocks" is an exception to the 10 Nov "no cell dropped" rule |
| 33 | Recompile gate (T2.21) | 0 compile events after a compiled arm's warm-up, in the ground truth and the density probe; above that the cell is refused | Default confirmed 9 Nov 2026; Shakedowns 5 and 6. This item is moot if the compiled group is dropped on 14 Nov | |

**Close to ready, not in the twelve:**

- **Compiled-versus-eager density tolerance:** ±0.01. Another session's 10 Nov rulings fix this value.
- **Learned-predictor features and hyperparameters:** configuration 4, 2.87% median held-out error, on training file `de7f0f44…`. Both are computed; only their status says "owed".
- **Prompt-set hash and tokenizer version:** can be recorded from T3.2d's prompts.
