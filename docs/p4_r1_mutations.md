# Mutations for the 14 Nov 2026 rulings (P2's banking, P4's image and libraries)

Each mutation was applied alone to the file named, the test file run, and the file restored.
A mutation is detected when the run fails or errors.

## P2's unblocked session (`scripts/p2_training.py --session`; `tests/test_p2_training.py`)

| # | Mutation | Result |
|---|---|---|
| U1 | a block's session accepted as unblocked | killed (1 failed) |
| U2 | a dry run's session accepted | killed (1 failed) |
| U3 | `INTERRUPTED_DISCARD.md` ignored | killed (1 failed) |
| U4 | unpassed steps ignored | killed (1 failed) |
| U5 | `late_check` left out of the required steps | killed (1 failed) |
| U6 | the block id fixed as `"session"`, not the directory's name | killed (1 failed) |

## P4's image and libraries (`scripts/gcp_launch_h100.sh`, `scripts/t61_session.py`, `scripts/t64_session.py`; `tests/test_p4_library_check.py`)

| # | Mutation | Result |
|---|---|---|
| L1 | the launcher's refusal of another image removed | killed (1 failed) |
| L2 | the launcher's default back to v5 | killed (1 failed) |
| L3 | a version difference not a problem | killed (1 failed) |
| L4 | a missing library not a problem | killed (1 failed) |
| L5 | an added library not a problem | killed (1 failed) |
| L6 | FA2 (`flash_attn`) allowed to differ beside FA3 | killed (1 failed) |
| L7 | the library problems not added to the preflight's | killed (1 failed) |
| L8 | a library check that could not run passed silently | killed (1 failed) |
| L9 | a list naming a library twice accepted | killed (1 failed) |
| L10 | P4 given no reference | detected (collection error: the reference path is empty) |
| L11 | the reference left out of P4's deployed files | killed (1 failed) |

## The interlock for a pairing that ran (`scripts/budget_interlock.py`, `RAN`; `tests/test_budget_interlock.py`)

| # | Mutation | Result |
|---|---|---|
| R1 | `ran_caps` returns the cards' caps unchanged | killed (2 failed) |
| R2 | a ran pairing keeps its card's cap on its provisioning | killed (2 failed) |
| R3 | the card's section says a ran pairing may launch | killed (1 failed) |
| R4 | the CLI launches a ran pairing | killed (1 failed) |
| R5 | `RAN` names no real pairing | killed (2 failed) |
| R6 | `current` ignores `RAN` | killed (2 failed) |
