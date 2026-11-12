# T4.3: the readers' mutation scripts after wave 2

Rerun on 12 Nov 2026 at `4c51ea3`, wave 2's last code commit (`7ea86be`)
plus the re-pin, in a detached worktree, with `PYTHONDONTWRITEBYTECODE=1`
(ruling 3: after each wave). The re-pin came first. B's test lists include
`tests/test_t61_session.py`, whose pin test fails while the timed path differs
from the draft's pin, and it would have made every B mutation read as caught.

## Reader A (`mutate_A.py`): 59 of 59 killed

54 were applied as written and killed. All 8 of wave 1's survivors are among
them (D5, B1, R1, X2, X5, E2, E3, I3; pinned in 2c, 2d and 2e). Five did not
match, because wave 1 or 2 rewrote their lines. Each was re-expressed on the
current code with the same intent and A's own test list, and each was killed:

| Mutation | Rewritten by | Re-expressed on the current code |
|---|---|---|
| D6 dispatcher: lengths filter removed | 1a and 2b: H6's lengths are budget labels read through the floor table (`in_scope`) | `if lengths is None: return True` becomes `if True: return True` |
| B3 block_drift: relative to the last block | 2c: the reference is the median of the other blocks | the reference becomes `by_block[present[-1]]` |
| R1 check_passes: `>` bound becomes `>=` | 2d: the line also refuses a non-finite difference | `abs(d) > bound` becomes `abs(d) >= bound` |
| R2 check_passes: None passes | 2d | `d is None or not finite or abs(d) > bound` becomes `d is not None and finite and abs(d) > bound` |
| R4 check_passes: structure mismatch ignored | 2d: an unfed report is refused first | `elif unmatched or extra_nodes:` becomes `elif False:` |

## Reader B (`mutate_B.py`): 22 of 22 caught

17 were applied as written and caught. The other five (M1, M2, M14, M15,
M16), which wave 1 rewrote, were caught again in their wave 1 re-expressions
(`mutate_B_rewritten.py`).

## Wave 2's own mutations (break-tested before closing)

All killed:

- 2a: 21, over H0, H1, H2, H3, H4, H5, the marginal label, the
  within-session spread and the analysis-commit pin
  (`tests/test_hypotheses.py`). Six survived the first pass (H0's
  both-pairings rule and median, H1's strict "below", H4's Holm step and
  two-sidedness, H5(b)'s strict "smaller"); each was pinned, then killed.
- 2c: 7, among them A's B1.
- 2d: 14, among them A's R1.
- 2e: 9, A's D5, X2, X5, E2, E3 and I3, plus the bootstrap's three refusals.
- Ruling 2 of 12 Nov 2026 (the banking time from the records): 3.
- Ruling 3 (the rehearsal's skip refuses P2 and P4): 4.

**A driver fault, caught.** A first pass reported three kills that were
false. Its shell check matched "failed" inside "1 xfailed". The check now
reads `[0-9]+ failed`, and the three were re-run and killed.
