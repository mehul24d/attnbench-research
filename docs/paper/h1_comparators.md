# Were the step form and the fitted sum pre-registered comparators for H1? (28 Nov 2026)

The paper-quality wave of 28 Nov 2026, item 9. This file quotes the text; it scores nothing.

## Answer: yes, both

H1's row in the plan, `docs/RESEARCH_PLAN.md` line 103, is identical at all three lock tags: `lock-2026-11-13` (`5347b73`), `lock-2026-11-13-r4` (`a16ae1a`) and `lock-2026-11-14-r3` (`7a24cba`, the scorer's tag). The sha256 of the line is the same at each. `git blame` at `7a24cba` dates the line to commit `c971d7c` (31 Oct 2026), two weeks before any P2 or P4 ground truth existed.

The row's claim column reads (quoted):

> On every confirmatory pairing (P2 to P7), the recursion over measured components predicts untraced end-to-end prefill within tolerance, and beats the fitted sum's committed prediction (the primary comparison for G1, trained only on pairings already run, Section 5.4) and also the sum, the KernelSight-style step form and the learned predictor's committed prediction (secondary).

Its test column reads (quoted):

> the recursion's median absolute relative error below that of each of the fitted sum, the sum, the KernelSight-style step form and the learned predictor on each confirmatory pairing (the constant-host-cost, nearest-pairing and + s_0 baselines are reported, not scored)

The comparators are defined in Section 5.4 at the same tag. The fitted sum's line was added at `ace5805` (18 Oct 2026) and the step form's at `c971d7c` (31 Oct 2026):

> **Fitted sum:** a·sum(h) + b·sum(l) + c·sum(d), with a, b, c ≥ 0 fitted on the same rows as the learned baseline ... so it is the fair test of G1 and the primary comparison

> **KernelSight-style step form:** our construction from KernelSight-LM's Eq. 4, without the term ℓ the authors leave undefined: max(sum h, sum l + sum d) per layer, plus every synchronisation cost as a serial tail.

The lock package (`docs/lock_package_draft.json`, `committed_predictors`, at `7a24cba`) also says: "H1 is scored as written: its primary comparison is the recursion against the fitted sum; the step form, the sum and the learned predictor are secondary comparators; the step form's own error is reported beside the recursion's on every pairing and every arm class, including where it is lower."

## What the scored result says (not rescored here)

From `docs/scoring/hypotheses_scored_20261116.json`, H1, `recursion_below`:

| Pairing | recursion | fitted sum (primary) | step form (secondary) | recursion below fitted sum | recursion below step form |
|---|---|---|---|---|---|
| P2 | 0.1183 | 0.1005 | 0.0842 | false | false |
| P4 | 0.2495 | 0.1216 | 0.2454 | false | false |

## What the paper may say

"A simpler pre-registered baseline beat the composed model" is supported, on both pairings, for the primary comparator (the fitted sum) and for one secondary comparator (the step form).

Three qualifications go with it:
- **The step form's margin on P4 is small**: 0.2454 against 0.2495.
- **The H1 row predicted something more specific**: on arms with no synchronisation inside the step, the recursion and the step form should agree, and an advantage was expected only on arms with synchronisations. On P2's synchronising class the step form was still lower (0.0988 against 0.1296).
- **The fitted sum is fitted**: its a, b and c are trained on P1 rows.
