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
| A F1, F2, F2b, F4 (H6) | `evaluate`'s scalar floor 5.0 becomes `F5`, a `FloorTable` at 5.0 for every (length, batch) the tests use. No assertion changes. | 1a, as F3 |
| A F2b, an empty score is not "no more decision errors" | Its one point moves from 16370 to 8190 tokens. At 16370 the point is now read at its 16384 budget and scored (F2's fix), so its first assertion (`n_scored == 0`) would assert the defect F2 removes. At 8190, outside H6's lengths, the empty score has its plan reason, and the assertion that matters (no "no more decision errors" over zero points) is unchanged. | 2b: points are matched by cell identity, realised length read at its budget |
| A F5a, F5b, a block with no anchor row, a NaN anchor | `drift` takes the bound as a required argument (`bound=BD.WORKING_BOUND`), and "not passed" is asserted as the refusal (`pytest.raises(BlockDriftError)` naming the block, or the non-finite time). | 2c: no-anchor and NaN refuse; the bound is a required argument from the lock |
| A F5c, the odd block is the reference | `bound=BD.WORKING_BOUND` added. The assertion (b1 flagged) is unchanged. | 2c |
| A F7b, a vacuous v2 slope leg | `slopes(..., mechanisms=())` is asserted to refuse (`GateAError`), not to return a non-empty `within_range`. | 2d: a vacuous input refuses |
| A F12, a degenerate bootstrap decides a sign | One repeat or one replicate is asserted to refuse (`CostModelError`), so the zero-width interval is never built. | 2e: the bootstrap refuses < 2 repeats |

A's 14 tests that wave 2 fixes carried `@WAVE2("<ruling>")`, a strict
`xfail` naming the ruling, through wave 1 (**plan default, flagged**, and
confirmed by the adviser's ruling 3(4) dated 12 Nov 2026), so wave 1 merged
with the suite green. Wave 2 removed every marker and the helper; all 36
tests pass with no marker (12 Nov 2026).
