#!/usr/bin/env bash
# Audit items S8 and S9: run the test suite ON A GPU and put the record where
# a reader can find it.
#
# S9 asks whether results/stage5/phases.parquet -- the source of every
# normalized_ms, and so the latency axis of Stage 6 and all of Stage 7 -- is
# covered by any suite run at all. Its own session log contains no suite
# record. That is not a claim the suite fails; it is that nobody can tell.
# This phase produces the missing record.
#
# S8 asks whether block_sparse and sdpa rebuild setup inside the timed region.
# tests/test_timed_region_setup.py reaches neither backend on CPU, and the two
# session logs that ran it on a GPU both record it FAILING, with no captured
# output. So the test is split out here and run on its own, ALLOWED TO FAIL,
# with -rA so the failure is in the log verbatim. A gate that aborts the
# session would destroy the evidence it exists to collect -- and the rest of
# the suite must still be able to gate, which it cannot do if one known-
# failing file can take it down.
#
# Neither phase measures anything, so both are cheap; they run before the
# measurement so a broken environment is found before the expensive part.
#
# Usage (on the instance, from the repo root):
#   bash scripts/gpu_suite_record.sh
# Exit status is the SUITE's (S9 gate). The timed-region rc is reported and
# written to /tmp/s8_timed_region.rc, and never aborts.
set -uo pipefail

TIMED=tests/test_timed_region_setup.py

bash scripts/run_phase.sh s9_suite -- \
    python3 -m pytest -q --deselect "$TIMED"
SUITE_RC=$?
echo "=== S9 suite rc=$SUITE_RC"

bash scripts/run_phase.sh s8_timed_region -- \
    python3 -m pytest "$TIMED" -v -rA
TIMED_RC=$?
echo "$TIMED_RC" > /tmp/s8_timed_region.rc
echo "=== S8 timed-region rc=$TIMED_RC (not a gate; the output IS the finding)"

exit "$SUITE_RC"
