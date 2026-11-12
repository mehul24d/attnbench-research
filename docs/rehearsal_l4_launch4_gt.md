# Launch 4 A's ground truth: which flag fired (12 Nov 2026)

The adviser's ruling 1, third set of 12 Nov 2026. Read, read-only and on the researcher's yes
(the permission log, 12 Nov 2026), from the bucket folder `attnbench-l4-reh-a-20261112-0737`:
`results/t61r_l4_20261112T0233Z_b1/12_ground_truth/eager/` `gt_L2048_b1.json`,
`gt_L8192_b1.json` and `stamps.json`, and nothing else. Commit `40b6df5`, the eager group,
`reps` 10, warm-up 3, sampler 0.1 s.

## Flag counts

| Cell | Rounds run | Clean | Throttle flags | Dense-drift flags | Rerun budget | Session floor | Drift range (relative) | Over 6.31% |
|---|---|---|---|---|---|---|---|---|
| 2048 × 1 | 10 | 1 | 0 | 9 | 0 | 5.61 ms (4.3% of the 130 ms reference) | −2.8% to −12.3%, median −6.2% | 3 (rounds 1, 4, 6) |
| 8192 × 1 | 10 | 6 | 0 | 4 | 0 | 5.61 ms (0.8% of the 688 ms reference) | −0.3% to −2.9% | 0 |

Every flag is dense drift: the dense control's round time against its warm-up reference (the
median of warm-ups 2 and 3), beyond the session floor. The ground-truth step's own message now
says this: "eager L2048_b1: 1 clean of 10 in 10 rounds, rerun budget 0 (flagged rounds: dense
drift 9); eager L8192_b1: 6 clean of 10 in 10 rounds, rerun budget 0 (flagged rounds: dense
drift 4)".

## The four checks

(a) **Throttle reasons.** None was flagged. `sw_power_cap` was seen in every block of all 20
rounds, on every arm. It is not in the throttle set: that set has four bits, by the researcher's
decision of 27 Oct 2026 (`clock_sampling.RESEARCHER_FLAG_BITS`; T4.1 item 24), and the session
recorded the same four (`controls.flag_bits`). No T3.2d round was throttled either.

(b) **Dense-control drift.** All 13 flags. Measured against the 6.31% relative limit (Appendix
A, 9 Nov 2026), 3 rounds at 2048 and none at 8192 would have been flagged. Both cells drift in
one direction: every timed round is faster than the warm-up reference. At 2048 that offset is
about 6%. T3.2c showed the same at 2048 (−6.73% and −6.51%), and T3.2a and T3.2b reach −12.6%.

(c) **The rehearsal's grid and the rerun budget.** The reps were not reduced: 10, as T3.2d.
The rerun budget is 0 in both, P1's working value until the lock (`t31_steps.rerun_budget`).
**The grid is what differs.** `dense_select` derives the session floor from the session's own
dense pre-pass, pooled over every cell of its grid. The rehearsal's grid is 2048 and 8192, so
the floor is 5.61 ms. T3.2c's floor over the full grid was 52.83 ms. T3.2d's value is not in the
local records, but it was above every round's drift: none of T3.2d's rounds was flagged as
recorded (`docs/t32d_gate_a_v2.md` §2, "as recorded": no group short). So T3.2d absorbed no
in-session flags. Under the relative limit,
T3.2d would have lost 2 rounds at 2048 × 1 and 9 at 8192 × 1.

(d) **The ground-truth path.** Between T3.2d's `3c44588` and `40b6df5`, `ground_truth.py` gained
only `tracer_layout` and `tracer_entries`, which the flagging does not call. The flag rule, the
reference, the completeness test (`clean == reps`) and the rerun loop are T3.2d's.

## What it means

The cause is configuration, not code. The rehearsal's two-length grid gave a session floor ten
times smaller than T3.2d's, against a warm-up offset of about 6% at 2048, and no reruns were
allowed. Verifying that needs no GPU: the per-round records above are the whole input to the
rule. B is not run, and the rehearsal closes. Its interruption and discard, and the drift check
across blocks, were never exercised on a GPU.

**Flag, for P2 and P4.** P2 does not derive a session floor. It flags drift against the lock's
per-cell floor (`t31_steps.session_floor_ms`, `floor_ms_at`): 1.39 ms at 2048 × 1 and 17.76 ms
at 8192 × 1. On these records that floor flags 10 of 10 rounds at 2048 and 1 of 10 at 8192. The
plan's text says the drift flag "reads the session floor `dense_select` derives" (Appendix A,
8 Nov 2026). That is P1's case only. No ruling covers P2's, so this conflicts with the plan's
text. The resolution floor is a spread of repetitions about their series median. Nothing tested
it as a limit on the dense control's offset from its warm-up reference.
