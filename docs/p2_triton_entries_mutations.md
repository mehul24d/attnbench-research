# Mutations: XAttention's Triton entries and the Triton smoke check (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, item 2: the fix is mutation-tested, and the structure
check is not loosened. Each mutation was applied alone in a detached worktree at `aa02da2` and
run against `tests/test_xattention_triton_entries.py` (M18 against
`tests/test_t61_session.py::test_the_real_preflight_stops_on_a_failed_triton_smoke_check`).
Driver: `pytest -x`; killed means at least one test failed. 19 of 19 killed.

| # | Mutation | File | Killed |
|---|---|---|---|
| M1 | `null_entries` never adds the Triton kernels | `scripts/t31_steps.py` | yes |
| M2 | `null_entries` adds them on any XAttention path | `scripts/t31_steps.py` | yes |
| M3 | the grid stand-in launches the real kernel | `host_profile.py` | yes |
| M4 | `install` ignores `grid` | `host_profile.py` | yes |
| M5 | the first kernel is not an entry | `host_profile.py` | yes |
| M6 | an entry's launching call is named | `host_profile.py` | yes |
| M7 | the entries are not grid entries | `host_profile.py` | yes |
| M8 | the bare price's label drops the grid | `host_profile.py` | yes |
| M9 | the bare pass skips the real launch | `bare_timer.py` | yes |
| M10 | the residual does not time an entry | `layer_residual.py` | yes |
| M11 | `align_calls` loosened: a missing entry is skipped | `host_profile.py` | yes |
| M12 | the entries are not restored | `host_profile.py` | yes |
| M13 | the smoke check accepts the torch path | `scripts/t31_steps.py` | yes |
| M14 | the smoke check ignores differing masks | `scripts/t31_steps.py` | yes |
| M15 | a failed smoke check is not a problem | `scripts/t61_session.py` | yes |
| M16 | the smoke check skips step 7's cell | `scripts/t31_steps.py` | yes |
| M17 | every pairing is a Triton pairing | `scripts/t61_session.py` | yes |
| M18 | the preflight drops the smoke check's problems | `scripts/t61_session.py` | yes |
| M19 | a smoke check that wrote nothing passes | `scripts/t61_session.py` | yes |

M18 was first killed only by `test_the_deployed_timed_path_is_the_one_decided_for_the_lock`,
which failed for the pin, not the mutation (vacuous); the preflight test was added and M18
rerun against it alone.

## The budget interlock's ruled-out provisioning (item 6)

P4 on Spot is ruled out (`budget_interlock.RULED_OUT`): neither launched nor counted against
another pairing's launch. Run against `tests/test_budget_interlock.py`; 7 of 7 killed.

| # | Mutation | Killed |
|---|---|---|
| B1 | `cases` ignores the ruled-out set | yes |
| B2 | `Case.possible` ignores it | yes |
| B3 | `launch_headroom` counts ruled-out cases | yes |
| B4 | `may_launch` ignores `possible` | yes |
| B5 | `current` drops `RULED_OUT` | yes |
| B6 | `RULED_OUT` names no real pairing | yes |
| B7 | the CLI ignores a ruled-out launch | yes |
