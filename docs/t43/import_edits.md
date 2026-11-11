# T4.3: edits to the readers' reproduction tests

Both readers read commit 23f540e (11 Nov 2026). Their reports are in this
folder (`T43_read_A.md`, `T43_read_B.md`), along with their mutation scripts
(`mutate_A.py`, `mutate_B.py`, `tp_scan_B.py`). Their tests are in
`tests/test_t43_read_a.py` (A, 17 tests) and `tests/test_t43_read_b.py`
(B, 19 tests). The tests were imported in 5a0e843 unchanged except for
paths: B's tests put `scripts/` on `sys.path`, as B's conftest did, and A's
find the repo root from the file rather than from the working directory.

Each test asserts the behaviour the plan requires. A test was edited only
where an adviser ruling of 11 Nov 2026 settles that behaviour differently
from how the test was written. Every assertion of a defect still stands.

| Test | Edit | Ruling |
|---|---|---|
| B, the `complete_lock` fixture (all B tests that use a lock) | Builds a typed, complete lock (`tests/lock_fixture.py`) instead of `"x"` for every value with only H0/H2 scored. No assertion changes. | 1b: typed schema per key; SCORED_REQUIRED covers all hypotheses |
| B F3, second session predicts after the first session's ground truth | Expects session b to be refused at `predict` (non-zero exit, predict failed, no receipt). The original expected predict to pass and the late check to fail. The `late_check` assertion reads `.get`, since that step no longer runs. | 1c: a cell's prediction is refused once any ground truth for that cell has started in the pairing |
| B F4, the runner reads values the complete lock does not hold | `rerun_budget` takes its typed form `{"max_rerun_rounds", "min_clean_rounds"}`, and the comparison reads `max_rerun_rounds`. | 1b |
| B F10, P4's predict cannot train on P2 | Expects `StepFailed` (no P1 file at the given path, refused before anything is read) instead of `NotBuilt`. | 1h: P4 trains on P1 and P2 rows, so the path is built |
| A F3, the dispatcher cannot apply a per-length floor | The loop over two scalar floors becomes one `FloorTable` holding the same values (16384: 13.77 ms, 32768: 101.16 ms). | 1a: the floor is a lock table per (length, batch) at every consumer, and a scalar is refused |

A's tests that still pass scalar floors (F1, F2, F2b, F4: H6's `evaluate`)
are wave 2 (ruling 2b). They take the same `FloorTable` edit there, and any
further edit is logged here.
