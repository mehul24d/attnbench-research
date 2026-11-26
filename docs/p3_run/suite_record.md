# P3's suite record (26 Nov 2026, after the run)

**CPU suites only; GPU suite record not produced for P3.** The card's suite record on the
instance (`--suite-record`, `scripts/gpu_suite_record.sh`) needs a GPU, and the instance was
deleted at the run's end, so the record is made on this Mac after the run, at the commit P3
deployed, `6e9188a`, before the push (`docs/appendix_a_bundled_wave.md`, 26 Nov 2026). Each
suite ran alone, in a detached worktree at `6e9188a` with no `results/`, in a session venv
(torch 2.13.0, Python 3.12.1, `pip install -e ".[dev,eval,learned]"`), on 26 Nov 2026.

| Suite | Command | Result |
|---|---|---|
| Gating (the whole suite) | `pytest tests/ -q` | **7121 passed, 183 skipped, 1 failed** (20 min 12 s) |
| P3 driver | `pytest tests/test_aws_p3_driver.py` | **22 passed** |
| P3 halts | `pytest tests/test_p3_halts.py` | **3 passed** |
| P3 store | `pytest tests/test_aws_p3_gsutil.py` | **8 passed** |
| P3 config | `pytest tests/test_p3_session.py` | **9 passed** |
| P3 pre-launch CPU pass | `scripts/prelaunch_cpu_pass.sh p3` | **PASSED** (10 passed; all 12 steps; `prelaunch_cpu_pass_p3_6e9188a.json`) |

The one failure is `tests/test_doc_derived_numbers.py::test_the_suite_counts_match_the_collected_suite`:
`docs/development.md` said 7204 tests and `6e9188a` collects 7305; the rehearsal's and P3's
waves added 101 tests without updating the count. It is a stale document, not a code fault;
the counts are corrected at the commit after this record (7324 with the post-run wave).

**The known flaky halt-wrapper test** (`tests/test_session_exit.py::test_a_hang_up_stops_the_session_then_syncs_and_halts`,
`docs/development.md`, 17 Nov 2026) passed in the gating run, so it was not rerun. The other
known flaky test (`tests/test_shakedown3_cpu.py`'s chain check, 16 Nov 2026) passed too.
