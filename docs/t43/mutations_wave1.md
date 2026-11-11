# T4.3: the readers' mutation scripts after wave 1

Rerun on 12 Nov 2026 at `c45b526`, wave 1's last code commit (ruling 3:
after each wave). Each script ran unchanged in a detached worktree at that
commit; only its root path was adapted. Both run with
`PYTHONDONTWRITEBYTECODE=1`. The first run without it reported X3 as
surviving. That was stale bytecode: X2 and X3 each shorten `crossover.py`
by the same 13 characters, and the restore and the next write landed in the
same second, so X3's tests ran X2's cached `.pyc`. Applied alone, X3 is
killed.

## Reader A (`mutate_A.py`): 50 killed, 8 survived, 1 not applicable

The 8 survivors are those A's report lists (A's run at `23f540e` killed
51 of 59). They are wave 2e's to pin, each with a test taken from the
plan's wording:

| Mutation | Wave 2 item |
|---|---|
| D5 dispatcher: missed-profitable gap `>=` floor | 2e (strict "above") |
| B1 block_drift: bound `>=` | 2c / 2e |
| R1 replay.check_passes: `>` bound becomes `>=` | 2d / 2e |
| X2 crossover: interval ignored on the positive side | 2e |
| X5 crossover_error: `contains` off by one | 2e |
| E2 exact non-inferiority: lower `>=` −margin | 2e |
| E3 Holm `_clears` `>=` | 2e |
| I3 intervals: point is the replicate mean | 2e (the full-sample point) |

D6 ("dispatcher: lengths filter removed") is not applicable. Ruling 1a
replaced the filter's text: H6's lengths are now budget labels read
through the floor table. It is re-expressed in wave 2b with H6's scoring.

## Reader B (`mutate_B.py`): 22 of 22 caught

17 were applied as written and caught. Five (M1, M2, M14, M15, M16) did
not match, because wave 1 rewrote the lines they edit:

| Mutation | Rewritten by | Re-expressed on the current code |
|---|---|---|
| M1 late check drops the skew | 1g, `start < synced + gap + skew` | `start < synced + gap` |
| M2 late check removed | 1g | the same check becomes `if False:` |
| M14 block identity check returns nothing | 1d, a missing field refused | the inequality branch becomes `elif False:` |
| M15 staged training sha256 compare removed | 1e, via `p1_training_check` | `None if True else` |
| M16 ground-truth process `ready_to_clock` removed | 1g, adds the `gap=` argument | the call is deleted |

All five were caught, each with B's own test list. The re-expressions are
in `mutate_B_rewritten.py`.

## Wave 1's own mutations (break-tested before closing)

1e (`_recompute_learned`): 4 of 4 killed. 1f: B's three mutations of the
pin, plus a line-ending edit, each move the pin
(`tests/test_timed_path.py`). 1h: 5 of 5 killed: the anchor pooled from
one block only, a short group kept, P4 ignoring P2's rows, the staged
check skipping the ledger, and a rehearsal prefix accepted.
