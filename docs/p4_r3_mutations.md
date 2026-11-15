# Mutations for P4 on Spot (the researcher's decision of 15 Nov 2026)

Each mutation was applied alone to the file named, the test file run with `-x`, and the file restored.
A mutation is detected when the run fails or errors.

| # | Mutation | File | Test file | Result |
|---|---|---|---|---|
| S1 | preflight_mode never short | `scripts/t61_session.py` | `tests/test_t61_blocks.py` | killed (1 failed) |
| S2 | a short preflight without an earlier full one with its smoke check | `scripts/t61_session.py` | `tests/test_t61_blocks.py` | killed (1 failed) |
| S3 | the short preflight runs the smoke check | `scripts/t61_session.py` | `tests/test_t61_blocks.py` | killed (1 failed) |
| S4 | the short preflight runs the GPU variants | `scripts/t61_session.py` | `tests/test_t61_blocks.py` | killed (1 failed) |
| S5 | the short preflight leaves out the kernel contract | `scripts/t61_session.py` | `tests/test_t61_blocks.py` | killed (1 failed) |
| S6 | P4 without the short preflight | `scripts/t64_session.py` | `tests/test_t61_blocks.py` | killed (1 failed) |
| F1 | library problems before the install ignored | `scripts/t64_session.py` | `tests/test_t64_session.py` | killed (1 failed) |
| F2 | a library check that cannot run passes | `scripts/t64_session.py` | `tests/test_t64_session.py` | killed (1 failed) |
| I1 | the overrun is not ₹1,500 | `scripts/budget_interlock.py` | `tests/test_budget_interlock.py` | killed (1 failed) |
| I2 | one rupee past the ceiling allowed | `scripts/budget_interlock.py` | `tests/test_budget_interlock.py` | killed (1 failed) |
| I3 | P4 on Spot still ruled out | `scripts/budget_interlock.py` | `tests/test_budget_interlock.py` | killed (1 failed) |
| B1 | the planner ignores the short preflight | `scripts/t64_blocks.py` | `tests/test_t64_card.py` | killed (1 failed) |
| B2 | the stale Spot rate | `scripts/t64_blocks.py` | `tests/test_t64_card.py` | killed (1 failed) |
| B3 | a block takes groups beyond its share | `scripts/t64_blocks.py` | `tests/test_p4_block_spec.py` | killed (1 failed) |
| B4 | the anchor also a block's own cell | `scripts/t64_blocks.py` | `tests/test_p4_block_spec.py` | killed (1 failed) |
| B5 | a grid cell left out of every block | `scripts/t64_blocks.py` | `tests/test_p4_block_spec.py` | killed (1 failed) |
| B6 | the last block drops a group | `scripts/t64_blocks.py` | `tests/test_p4_block_spec.py` | killed (1 failed) |
