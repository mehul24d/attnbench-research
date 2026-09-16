#!/usr/bin/env bash
# Run one measurement phase on a rented instance so that the LAST action of the
# phase -- always, whether it succeeded, failed, or died partway -- is a sync of
# everything on disk to GCS.
#
# Why this exists
# ---------------
# Two incidents, same shape. A session left 7200 rows on an unbootable disk
# because the checkpoint was written to a second file on the SAME disk. A
# 2026-09-15 H100 session (Rs 2,029) lost everything because the sync was a
# single command at teardown, and the instance's self-delete ran first.
#
# The rule that came out of both: sync each phase as it completes. Not at the
# end, not to another file on the same disk. This wrapper makes that structural
# instead of something the operator has to remember at 4am.
#
#   - the sync runs in a trap, so it happens even on a non-zero exit, an
#     unhandled signal, or a `set -e` abort inside the phase.
#   - the phase's exit code is preserved and propagated, so a caller chaining
#     phases with && still stops on failure.
#   - the sync VERIFIES by listing the objects back, and says so loudly. A
#     silent sync is what the last incident actually had.
#
# Usage (on the instance):
#   bash scripts/run_phase.sh stage0 -- python scripts/run_probe.py --out results/probe
set -euo pipefail

PHASE="${1:?usage: run_phase.sh PHASE -- CMD...}"; shift
[[ "${1:-}" == "--" ]] && shift
[[ $# -gt 0 ]] || { echo "run_phase.sh: no command given" >&2; exit 2; }

# Refuse to run a phase in a tree git calls dirty.
#
# provenance.capture() stamps git_dirty on every row a phase writes, and a
# dirty stamp disqualifies those rows from licensing anything downstream --
# load_stage1_pass_set rejects them outright. gcp_deploy_source.sh establishes
# a clean tree at deploy; this asserts nothing has broken that invariant by
# the time measurement starts, INCLUDING this script itself (its own `mkdir -p
# logs` did exactly that until logs/ was gitignored).
#
# Checked before the phase, not after, because after is a wasted GPU hour.
if command -v git >/dev/null && git rev-parse --git-dir >/dev/null 2>&1; then
  DIRT="$(git status --porcelain)"
  if [ -n "$DIRT" ]; then
    echo "REFUSING TO RUN PHASE -- working tree is dirty:" >&2
    echo "$DIRT" >&2
    echo >&2
    echo "Every row this phase writes would be stamped git_dirty=True, and a" >&2
    echo "dirty stamp means the recorded commit does not describe the code" >&2
    echo "that ran. Stage 2 rejects such passes outright, so the measurement" >&2
    echo "would be paid for and then refused. Add the path to .gitignore if" >&2
    echo "it is output, or commit it if it is source." >&2
    exit 3
  fi
fi

BUCKET="${ATTNBENCH_BUCKET:-gs://attnbench-results-research-507316}"
SESSION="${ATTNBENCH_SESSION:-$(hostname)}"
DEST="$BUCKET/$SESSION"
mkdir -p logs results
LOG="logs/${PHASE}.log"

sync_now() {
  local rc=$1 tag=$2
  echo "--- sync[$tag] $PHASE -> $DEST (rc=$rc) $(date -u +%FT%TZ)" | tee -a "$LOG"
  # results first: it is the thing that cannot be regenerated.
  gsutil -q -m rsync -r results "$DEST/results" 2>&1 | tee -a "$LOG" || {
      echo "SYNC FAILED for results -- the disk is now the only copy" >&2; return 1; }
  gsutil -q -m rsync -r logs "$DEST/logs" 2>&1 || {
      echo "SYNC FAILED for logs" >&2; return 1; }
  echo "--- synced objects under $DEST/results:"
  gsutil ls -r "$DEST/results" | tail -n 40
  return 0
}

# The trap is the point. Whatever happens to the phase, the sync runs.
RC=0
trap 'sync_now "$RC" exit || echo "POST-PHASE SYNC FAILED -- DO NOT TEAR DOWN" >&2' EXIT

echo "=== phase $PHASE start $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "=== cmd: $*" | tee -a "$LOG"
set +e
"$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
echo "=== phase $PHASE end rc=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"

exit "$RC"
