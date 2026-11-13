# The lock (12 Nov 2026, after the third lock-wave rulings; re-cut 13 Nov 2026)

The lock (T4.4) is Sat 14 Nov 2026. `docs/lock_values.json` holds all 40 of T4.1's values,
typed as `attnbench/lock/checklist.SCHEMA` requires, with the scored pairings and P2's and P4's
`pairing_values`; the checker finds it complete against the plan at this commit
(`plan_sha256`). Five values are typed "not applicable" (rows 2 to 6 below), and H4 and H5 are
stated "not scored". With the file present, P2 and P4 run in lock mode.

## The 14-value table, as ruled (12 Nov 2026)

| # | T4.1 item (row) | Lock value |
|---|---|---|
| 1 | H6 decision-error bound (2) | 0: the dispatcher makes no more decision errors than each comparator; regret reported (`dispatcher.h6_rule`) |
| 2 | Power target and n (6) | Not applicable: H4 not scored (T1.5 never ran) |
| 3 | Primary cell n, admissibility n (7) | Not applicable: H4 not scored |
| 4 | Admissibility Holm family (8) | Not applicable: H4 not scored (T1.8 never ran) |
| 5 | Cross-card dense task-score bound (12) | Not applicable: H4 and H5 not scored (audit B1 never ran) |
| 6 | Head-type criterion (22) | Not applicable: T1.3 not pursued (the researcher's delegation) |
| 7 | Null-kernel floor multiple (14) | 3.302 (15.52 µs at q 0.95 over TaxBreak's 4.7 µs; T3.1 null passes) |
| 8 | Probe bounds (15) | loop CV 0.01, loop median 0.05, null median 0.10; analysis flags only |
| 9 | Chain-check tolerance (32) | 0.10 |
| 10 | Residual-spread tolerance (34) | 0.2349: 1.5 × T3.2d's largest spread, 0.1566 (`scripts/residual_spread_tolerance.py`) |
| 11 | Pricing-mismatch share (35) | 0.05 |
| 12 | Rerun budget (23) | 5 reruns, 10 clean rounds |
| 13 | Bootstrap settings (38) | 0.95, 10,000, seed 0 |
| 14 | Prompt-set hash (26) | `cee769a2…` under 4.46.0, equal to 5.16.1's |

The researcher's items: 31 = 0; 18 = 0.8867 (re-run on this lock with
`scripts/hdbi_threshold.py`: unchanged); 28 = P7, then the chunked extension. Scored pairings:
H0 [P2], H1 [P2, P4], H2 [P2], H3 [P4], H6 [P4], H4 and H5 not scored.

The timed-path pin is `7bf6e506…` at `95c64e8`, and the analysis commit is `95c64e8`. They moved
from `fb6dca8` to `78349e0` when H6's rule changed, then to `341860c` when `learned.LOCKED_CONFIG`
was set to config 4 for Gate B, then to `9ddb9ee` when the lock fill's chain-check shape was
fixed (Appendix A, 12 Nov 2026, the second and third lock-wave rulings), then to `aa02da2` when
XAttention's Triton kernels became null entries, after P2's step 7 met them on the A100 (Appendix
A, 13 Nov 2026; the adviser's rulings dated 13 Nov 2026), then to `285f2f7` when each XAttention
arm's kernels were nulled in its own kernels module and the smoke check was widened to every arm,
after P2 at `lock-2026-11-13` met the asserts-removed arm's kernels (Appendix A, 13 Nov 2026, `-r2`),
then to `95c64e8` when the smoke's ground-truth controls became K2's correct candidates, after P2
at `lock-2026-11-13-r2` stopped in preflight (Appendix A, 13 Nov 2026, `-r3`).
T4.4 was written Done at `0301fb6`, and the tag `lock-2026-11-14` is on its child `3aa0449`; the
first re-cut's tag `lock-2026-11-13` is on `5347b73`, the second's, `lock-2026-11-13-r2`, on `886fa28`.
The `-r3` re-cut's lock commit holds this file, the plan's Appendix A line and the new pin; the tag
`lock-2026-11-13-r3` goes on its child,
which carries the suite counts and P2's relaunch card. Neither the plan nor the pin changes there.
No earlier tag is moved.

## Still to do

1. Tag the re-cut's lock commit (`scripts/tag_lock.py --lock-file docs/lock_values.json --tag
   lock-2026-11-13-r3`). Pushing the tag needs its own yes.
2. P4's Flex-start request needs its own yes.
3. P2's Flex card is built from the lock commit, with the headroom table (`docs/p2_flex_card.md`).
   Its launch needs its own yes.
4. If code on the timed path or analysis code changes after the tag, the pin and the analysis
   commit are re-checked, and the lock is re-cut.
