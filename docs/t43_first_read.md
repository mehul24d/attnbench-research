# T4.3 first read: plan and scoring code after the T3.2 repeats (8 Nov 2026)

**What this is.** A read of the plan and the scoring code as they stand, with
the two P1 repeats (T3.2a, T3.2b) as the test data. It was done by the same
agent that wrote much of the code. T4.3 asks for two *independent* reads
arranged by the researcher, so this is input to the first read, not the read
itself. Whether it counts is the researcher's decision.

**Scope.** The floors (`attnbench/lock/floors.py`), the decision and
boundedness scoring that consumes them (`predict/scoring.py`,
`predict/residuals.py`, `lock/hdbi_fit.py`), Gate A as written (Section 7;
`analysis/gate_a_repeats.py`), the block path (`lock/blocks.py`,
`analysis/block_drift.py`, `scripts/t61_session.py`), and the lock list
(`lock/checklist.py`, T4.1). Not read: `analysis/cross_arch.py`,
`matched.py`, `composition.py`, `canary.py` and the decode modules, which
T3.2 did not exercise.

The draft lock package is `docs/lock_package_draft.json`. It holds every T4.1
item with a proposal, its basis and its status, and leaves every value null.

## Findings that need a decision before lock

1. **Gate A cannot pass on P1 as written.** Leg 1 fails in both repeats: the
   recursion's error is 13.8% and 14.5%, against 3.8% and 3.6% for the step
   form. The loss is in the synchronising arms (XAttention 28 to 32% against
   3 to 4%). Leg 2 has no bound and leg 3 has no measurement. T3.3 has to
   rule on what a failed Gate A means for the plan. Widening the gate is not
   the remedy (`docs/t32_gate_a_repeats.md`).
2. **The resolution floor is one absolute number across a 60× range of cell
   times.**
   - The floor is 19.42 ms. That is 16% of a dense 2048 prefill (121 ms) and
     0.5% of a dense 32768 one (4271 ms).
   - At 2048, 8 of 12 sparse-against-dense gaps fall inside it; at 4096, 6 of
     12; at 8192, 4 of 12; at 16384 and 32768, none.
   - `predict/scoring.py` marks those cells "not interpreted", and
     `lock/hdbi_fit.py` drops them from its fit. Most short-context decisions
     are therefore unscored, and the HDBI threshold is fitted on long
     contexts.
   - The earlier study's decision map used banded relative floors
     (`analysis/decision.py`).
   - Proposal: a per-length or relative floor, decided before the floors are
     fixed (lock table row 16).
3. **The noise floor from two sessions is their largest systematic gap, not
   noise.**
   - With two sessions, the 0.95 quantile over 35 cells sits near the largest
     cell's gap.
   - That gap is systematic: T3.2b ran 5 to 7% faster at 16384 and 32768
     tokens in dense, the oracle and mean-pool.
   - At 6.77%, the floor is larger than the step form's whole error, so a
     tolerance (row 1) set below it cannot be met.
   - Proposal: report the session effect as one, and decide rows 16 and 17
     together.
4. **The block drift bound (3%) is below the measured between-session
   drift.**
   - The proposed anchor (dense L8192 b1) moved 6.71% between the two L4
     sessions.
   - Blocks are separate boots, so on P4's Spot blocks a 3% bound would label
     blocks for session noise.
   - The rule says never widen the bound to pass, so it has to be set now,
     before P4 (row 36).
5. **The flagged-memset sensitivity bound prices operations that do
   nothing.**
   - As ruled, a record-less memset is priced at the memset zoo's duration
     for its size.
   - Both repeats show these memsets are zero-length (`docs/limitations.md`).
     The bound would add a shift for work that never ran.
   - Proposal: price them at zero and report the count (row 31).

## Defects found and fixed in this wave

6. **Every real block would have been refused.** `tag_rows` keyed T3.1's rows
   by their label (`oracle@0.25`), their prompt's tokens (1951) and an absent
   `density`, so no real row could match a block's cells. Rows now carry
   their grid cell (`t31_row_cell`), and `lock.blocks.row_key` reads it.
7. **The block drift check would have failed on real rows.** It read
   `median_ms`, which only the fake harness wrote, so it would have raised
   KeyError, and it keyed anchors by label. It now reads `row_key`, and the
   adapter sets `median_ms` from `prefill_ms_median`. A test runs both on
   T3.1-format rows, and both fixes were break-tested.

## Open, not decisions

8. **P2 cannot commit a prediction yet.** Three things block it:
   - the learned predictor's configuration (Gate B);
   - a `ModelShape` for the primary model, which the learned features need;
   - the traced pass (Section 5.3 step 8), which this runner does not build.
     T3.1's ground truth is untraced only.
9. **The P1 training file is outside git.** Its integrity rests on the lock
   pinning its sha256 (recorded in the draft) and on uploading it. Until the
   upload, the only copy is on the Mac.
10. **Leg 1 is required in each repeat separately.** This is a plan default
    and should be flagged in Appendix A.
