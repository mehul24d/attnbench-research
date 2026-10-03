# Sourced by gcp_launch_a100.sh, gcp_launch_h100.sh and gcp_launch_l4.sh.
#
# A launcher creates a billable instance. From 2026-10-04 it refuses unless
# one of these holds (estimator-frontier pre-registration, sec. 8.3):
#
#   FRONTIER_SESSION=<session>   and scripts/frontier_budget_gate.py prints
#                                PROCEED for it on this launcher's card. The
#                                in-guest halt and the hard cap are then set
#                                from the gate, not from the launcher's
#                                defaults or from GCP_MAX_RUN.
#   ATTNBENCH_NOT_FRONTIER=<why> the session belongs to another
#                                pre-registration (T4 calibrated, say) and is
#                                outside the frontier study's Rs 12,000 cap.
#
# Neither set is a refusal, not a default. Protection that has to be asked
# for fails by omission (tests/conftest.py, 2026-09-05).

_frontier_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_frontier_py="${ATTNBENCH_PYTHON:-$_frontier_here/../.venv/bin/python}"

frontier_gate() {
  local card="$1" out halt max
  if [[ -n "${FRONTIER_SESSION:-}" && -n "${ATTNBENCH_NOT_FRONTIER:-}" ]]; then
    echo "REFUSED: FRONTIER_SESSION and ATTNBENCH_NOT_FRONTIER are both set. Set one." >&2
    exit 3
  fi
  if [[ -n "${ATTNBENCH_NOT_FRONTIER:-}" ]]; then
    echo "Not a frontier-study session (declared: ${ATTNBENCH_NOT_FRONTIER})."
    echo "  The budget gate is skipped and this spend is outside the Rs 12,000 cap."
    return 0
  fi
  if [[ -z "${FRONTIER_SESSION:-}" ]]; then
    echo "REFUSED: no budget gate has passed for this launch." >&2
    echo "  Set FRONTIER_SESSION=<session> for a frontier-study session, or" >&2
    echo "  ATTNBENCH_NOT_FRONTIER=<reason> for a session of another pre-registration." >&2
    exit 3
  fi
  if ! out="$("$_frontier_py" "$_frontier_here/frontier_budget_gate.py" \
              --session "$FRONTIER_SESSION" --card "$card" \
              ${FRONTIER_LEDGER:+--ledger "$FRONTIER_LEDGER"} \
              ${FRONTIER_STATE:+--state "$FRONTIER_STATE"})"; then
    echo "$out"
    echo "REFUSED: the budget gate did not pass for $FRONTIER_SESSION. Nothing was created." >&2
    exit 3
  fi
  echo "$out"
  halt="$(sed -n 's/^HALT_MINUTES=//p' <<<"$out")"
  max="$(sed -n 's/^MAX_RUN_MINUTES=//p' <<<"$out")"
  if ! grep -qx 'VERDICT=PROCEED' <<<"$out" || [[ ! "$halt" =~ ^[0-9]+$ || ! "$max" =~ ^[0-9]+$ ]]; then
    echo "REFUSED: the budget gate's output could not be read. Nothing was created." >&2
    exit 3
  fi
  HALT_MINUTES="$halt"
  MAX_RUN="${max}m"
  echo "  halt and hard cap set by the gate: +${HALT_MINUTES} min, --max-run-duration=${MAX_RUN}"
}

# After a successful create. The row reserves the session to its hard cap
# until the teardown records what it cost.
frontier_record_launch() {
  local card="$1" instance="$2"
  [[ -n "${FRONTIER_SESSION:-}" ]] || return 0
  if ! "$_frontier_py" "$_frontier_here/frontier_budget_gate.py" \
        --session "$FRONTIER_SESSION" --card "$card" --record-launch "$instance" \
        ${FRONTIER_LEDGER:+--ledger "$FRONTIER_LEDGER"} \
        ${FRONTIER_STATE:+--state "$FRONTIER_STATE"} >/dev/null; then
    echo "WARNING: the instance exists but its launch was NOT written to the ledger." >&2
    echo "  Record it by hand before any other launch:" >&2
    echo "  frontier_budget_gate.py --session '$FRONTIER_SESSION' --card $card --record-launch $instance" >&2
  fi
}
