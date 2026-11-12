# The drift reference, corrected (12 Nov 2026, lock wave)

The adviser's ruling 2 of the lock wave, 12 Nov 2026. It corrects ruling 1 of the fourth set
of 12 Nov 2026: a constant offset between the warm-up and the timed rounds is not drift. P2 and
P4 now compare each round's dense control with the cell's own timed rounds, not with the
warm-up time. The limit stays 6.31% (`t31_steps.RELATIVE_DRIFT_LIMIT`). P1's sessions keep the
protocol they ran under.

## The rule, declared before it is applied to any record

Written and committed before the rule was applied to T3.2a to T3.2d or to the rehearsal:

1. **Reference.** Round *i* is judged against the median of the dense control's timed time
   over the usable rounds before it, in the same cell. A usable round is one with no throttle
   flag. Drift-flagged rounds stay in the reference, and reruns count as rounds. The warm-up
   is excluded.
2. **Minimum before use: 5 usable rounds** (`t31_steps.DRIFT_REFERENCE_MIN_ROUNDS`). That is
   half of `reps` 10. A median of 5 tolerates 2 outliers. The first judgment then comes at
   round 5 of 10, which leaves the 5 reruns their full reach.
3. **The earlier rounds.** All rounds up to the one that brings the usable count to 5 are
   judged together, against the median of those 5 usable rounds. Until then they are pending:
   neither clean nor flagged. A cell that ends with rounds still pending counts them as not
   clean.
4. **Judgments are final.** A later round never re-judges an earlier one. Each judgment
   depends only on the rounds before it, or on the first 5 for the earliest rounds, so the
   rerun loop is decided online, round by round.
5. **Flag.** A round is flagged when |dense − reference| / reference > 6.31%. A throttle flag
   is kept as before. A cell needs `reps` ≥ 5.
6. **Retroactive test (before adoption).** Apply the rule to every recorded cell of T3.2a to
   T3.2d and of launch 4 A. None of these ran reruns. A cell "can reach 10 with 5 reruns"
   when it has at most 5 flagged rounds out of its 10 (and the reruns come up clean). If any
   P1 cell cannot, stop and report; the rule is not adopted.

## Retroactive test: passed, the rule is adopted

Applied to every recorded cell (`drift_filter.refilter`, mode `running`, limit
`0.06311947327826187`, `min_rounds` 5; the script is `scratchpad/retro_running.py`, not banked).
Clean rounds of 10, the running rule against the fourth set's warm-up rule:

| Session | 2048 × 1 | 4096 × 1 | 8192 × 1 | 16384 × 1 | 16384 × 4 | 32768 × 1 |
|---|---|---|---|---|---|---|
| T3.2a: running / warm-up | 9 / 6 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |
| T3.2b | 9 / 8 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |
| T3.2c | 10 / 8 | 9 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |
| T3.2d | 10 / 8 | 10 / 10 | 9 / **1** | 10 / 10 | 10 / 10 | 10 / 10 |
| Launch 4 A | 9 / 7 | — | 10 / 10 | — | — | — |

Every P1 cell keeps at least 9 of 10, so each reaches 10 with at most one of the 5 reruns. No
cell stops the rule. T3.2d's 8192 × 1, refused under the warm-up rule, keeps 9. The largest
running drifts are 7.2% (T3.2d 8192), 7.0% (T3.2b 2048) and 6.7% (T3.2a 2048, T3.2c 4096): each
is one round, just over the limit. T3.2c's 16384 groups, short as recorded under P1's absolute
floor, are complete. No round is throttled in any of these sessions, so the throttle exclusion
is exercised by the tests only.

Built in `ground_truth.running_judgments` and `_judge_rounds` (`Controls.drift_reference_min_rounds`,
set with the limit), `t31_steps.DRIFT_REFERENCE_MIN_ROUNDS` and `drift_filter`'s `running`
mode. Sixteen mutations killed.
