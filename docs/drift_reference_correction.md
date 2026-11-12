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
