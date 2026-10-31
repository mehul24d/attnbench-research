# Submit, poll and cancel a DWS flex-start request without holding one long
# gcloud call open. Sourced by scripts/gcp_launch_h100.sh (plan T6.4), after
# scripts/_gcp_project.sh has set PROJECT:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_flex_request.sh"
#
# Why. A flex-start create given --request-valid-for-duration=2h makes the
# gcloud client wait for the whole window. One
# 2-hour client call is one laptop sleep, one dropped connection or one
# closed terminal away from an unknown state: the request may be live
# server-side, capacity may arrive, and an H100 starts billing with nobody
# watching. So the create is submitted with --async (it returns when the
# request is accepted), and this file does the waiting in short, bounded
# calls that each end in a decision.
#
# Four things hold throughout, and tests/test_flex_request.py breaks each:
#
#   1. Every gcloud call has a client-side time limit (flex_bounded). A call
#      that hangs is killed and counted as "state unknown", not waited on.
#   2. At most one request is live. A request is cancelled (deleted) and the
#      delete CONFIRMED (describe says not found) before any later attempt is
#      submitted. A cancel that cannot be confirmed stops the script loudly
#      and names the delete command; it never goes on to another zone.
#   3. The request is cancelled at a client deadline set a few minutes inside
#      the server's --request-valid-for-duration. The server window is the
#      backstop for a client that has gone away; the deadline is what
#      normally ends a wait.
#   4. Interrupting the script (Ctrl-C, closed terminal, SIGTERM) cancels a
#      live request on the way out. See flex_install_traps.
#
# The date is NOT checked here. Whoever submits calls launch_policy_check
# before each submit (launch definition, plan T0.17: a waiting request counts
# as launched when it is submitted, so a wait that runs past midnight is not
# refused, and a later attempt is).
#
# What this file assumes about Compute Engine, read from the documentation
# on 27 Oct 2026 and NOT yet observed on this project (plan T6.4 notes):
#   - a flex-start VM waits in status PENDING when requestValidForDuration
#     is 90 s or more, then moves to PROVISIONING, or Compute Engine deletes
#     it when the window ends;
#   - a PENDING VM can be deleted before the window ends;
#   - `--async` returns once the insert operation is accepted.
# The first two are what the poll depends on. The third is handled by a
# grace period in which "not found" is not yet read as "request ended".
#
# Bash 3.2 (macOS) compatible: no associative arrays, no `wait -n`.

# Seconds. Overridable for tests only in the sense that every value is read
# from the environment with a default; none changes what is allowed.
FLEX_CALL_LIMIT="${GCP_FLEX_CALL_LIMIT:-60}"      # limit for any single call
FLEX_POLL_SECONDS="${GCP_FLEX_POLL_SECONDS:-30}"  # between status checks
FLEX_GRACE_SECONDS="${GCP_FLEX_GRACE_SECONDS:-90}" # "not found" ignored this long after submit
FLEX_CONFIRM_SECONDS="${GCP_FLEX_CONFIRM_SECONDS:-300}" # to see a delete take effect
FLEX_CANCEL_STEP="${GCP_FLEX_CANCEL_STEP:-10}"   # between looks at a delete in progress
FLEX_UNKNOWN_LIMIT="${GCP_FLEX_UNKNOWN_LIMIT:-6}" # consecutive unreadable polls tolerated

# Set while a request may exist, so the exit trap knows to cancel it.
FLEX_LIVE=0
FLEX_NAME=""
FLEX_ZONE=""
# Result of flex_wait: running | expired | cancelled | failed | unreadable
FLEX_OUTCOME=""
FLEX_WAITED=0
FLEX_LAST_STATUS=""

flex_now() { command date +%s; }

# flex_bounded SECONDS CMD [ARGS...]
# Runs CMD in the background and kills it after SECONDS. Prints CMD's stdout
# and stderr (redacted of the project ID) to the same two streams, and
# returns CMD's status, or 124 on a timeout. macOS has no `timeout`.
flex_bounded() {
  local limit="$1" out err pid ticks=0 rc=0
  shift
  out="$(mktemp)"; err="$(mktemp)"
  command "$@" >"$out" 2>"$err" &
  pid=$!
  # Five checks a second, so a call that answers at once costs ~0.2 s, not 1.
  while kill -0 "$pid" 2>/dev/null; do
    if (( ticks >= limit * 5 )); then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      rc=124
      break
    fi
    sleep 0.2
    ticks=$((ticks + 1))
  done
  if (( rc == 0 )); then
    wait "$pid" 2>/dev/null || rc=$?
  fi
  gcp_redact <"$out"
  gcp_redact <"$err" >&2
  rm -f "$out" "$err"
  return "$rc"
}

# flex_status NAME ZONE -> prints one word:
#   the instance status (PENDING, PROVISIONING, STAGING, RUNNING, ...),
#   NOT_FOUND when the API says the instance does not exist, or
#   UNKNOWN when the call failed, timed out or answered with anything else.
flex_status() {
  local name="$1" zone="$2" out err rc=0 status
  out="$(mktemp)"; err="$(mktemp)"
  flex_bounded "$FLEX_CALL_LIMIT" gcloud compute instances describe "$name" \
    --project="$PROJECT" --zone="$zone" --format="value(status)" \
    >"$out" 2>"$err" || rc=$?
  status="$(tr -d '[:space:]' <"$out")"
  if (( rc == 0 )) && [[ "$status" =~ ^[A-Z_]+$ ]]; then
    echo "$status"
  elif (( rc != 0 && rc != 124 )) && grep -qiE 'was not found|notFound|NOT_FOUND' "$err"; then
    echo "NOT_FOUND"
  else
    echo "UNKNOWN"
  fi
  rm -f "$out" "$err"
}

# flex_cancel NAME ZONE
# Deletes the request and waits until the API says it is gone. Returns 0 only
# on that confirmation; 1 otherwise, after printing the manual command.
flex_cancel() {
  local name="$1" zone="$2" start now st
  echo ">> cancelling request $name in $zone" >&2
  flex_bounded "$FLEX_CALL_LIMIT" gcloud compute instances delete "$name" \
    --project="$PROJECT" --zone="$zone" --quiet >/dev/null 2>&1 || true
  start="$(flex_now)"
  while :; do
    st="$(flex_status "$name" "$zone")"
    if [[ "$st" == "NOT_FOUND" ]]; then
      FLEX_LIVE=0
      echo ">> $name is gone from $zone (confirmed)" >&2
      return 0
    fi
    now="$(flex_now)"
    if (( now - start >= FLEX_CONFIRM_SECONDS )); then
      echo "!! COULD NOT CONFIRM THAT $name IS DELETED (last status: $st)." >&2
      echo "!! A request or instance may still exist and may bill. Run:" >&2
      echo "!!   gcloud compute instances delete $name --zone=$zone --project=\$ATTNBENCH_GCP_PROJECT --quiet" >&2
      echo "!!   bash scripts/gcp_cleanup_check.sh" >&2
      return 1
    fi
    # Not gone yet: delete again (idempotent) and look again.
    flex_bounded "$FLEX_CALL_LIMIT" gcloud compute instances delete "$name" \
      --project="$PROJECT" --zone="$zone" --quiet >/dev/null 2>&1 || true
    sleep "$FLEX_CANCEL_STEP"
  done
}

# flex_wait NAME ZONE CANCEL_AFTER_SECONDS
# Polls a submitted request. Sets FLEX_OUTCOME and FLEX_WAITED (seconds).
#   running     the VM is RUNNING (or on its way: PROVISIONING, STAGING)
#   expired     the request disappeared: the server window ended or the
#               request was refused; nothing exists, nothing bills
#   cancelled   the client deadline passed; the request was deleted and the
#               delete confirmed
#   no_capacity the VM went to STOPPING, STOPPED or TERMINATED before it was
#               granted: Compute Engine ending the request for capacity (attempt
#               1, 31 Oct 2026: STOPPING while the insert ended
#               ZONE_RESOURCE_POOL_EXHAUSTED); deleted, the delete confirmed
#   failed      the VM is in another state this script does not expect
#               (suspended, ...); it was deleted and the delete confirmed
#   unreadable  too many polls in a row could not be read; a cancel was tried
# Returns 0 only for running. If a cancel could not be confirmed it exits the
# script with status 3 (FLEX_LIVE stays 1, the exit trap tries once more).
flex_wait() {
  local name="$1" zone="$2" cancel_after="$3" start now st unknown=0 seen_live=0
  FLEX_NAME="$name"; FLEX_ZONE="$zone"
  FLEX_OUTCOME=""
  start="$(flex_now)"
  while :; do
    now="$(flex_now)"
    FLEX_WAITED=$((now - start))
    st="$(flex_status "$name" "$zone")"
    FLEX_LAST_STATUS="$st"
    case "$st" in
      RUNNING|PROVISIONING|STAGING)
        # Past PENDING, the request has been granted: capacity is held and
        # the max-run-duration clock starts at RUNNING.
        FLEX_OUTCOME="running"; FLEX_LIVE=0; return 0 ;;
      PENDING)
        seen_live=1; unknown=0 ;;
      NOT_FOUND)
        if (( seen_live == 1 || FLEX_WAITED >= FLEX_GRACE_SECONDS )); then
          FLEX_OUTCOME="expired"; FLEX_LIVE=0; return 1
        fi
        unknown=0 ;;  # just submitted: not visible yet
      STOPPING|STOPPED|TERMINATED)
        echo ">> $name is $st before it was granted: Compute Engine ended the request (no capacity)." >&2
        flex_cancel "$name" "$zone" || exit 3
        FLEX_OUTCOME="no_capacity"; return 1 ;;
      UNKNOWN)
        unknown=$((unknown + 1))
        if (( unknown >= FLEX_UNKNOWN_LIMIT )); then
          echo "!! $unknown status checks in a row could not be read." >&2
          flex_cancel "$name" "$zone" || exit 3
          FLEX_OUTCOME="unreadable"; return 1
        fi ;;
      *)
        echo "!! $name is $st, which a waiting flex-start request should not be." >&2
        flex_cancel "$name" "$zone" || exit 3
        FLEX_OUTCOME="failed"; return 1 ;;
    esac
    if (( FLEX_WAITED >= cancel_after )); then
      echo ">> no capacity after ${FLEX_WAITED}s (client deadline ${cancel_after}s)" >&2
      flex_cancel "$name" "$zone" || exit 3
      FLEX_OUTCOME="cancelled"; return 1
    fi
    sleep "$FLEX_POLL_SECONDS"
  done
}

# Best-effort reason a request ended, from the insert operation. Prints
# nothing when it cannot be read. For the log only: the caller does not
# decide anything from it except through flex_is_capacity_error.
flex_end_reason() {
  local name="$1" zone="$2"
  flex_bounded "$FLEX_CALL_LIMIT" gcloud compute operations list \
    --project="$PROJECT" --zones="$zone" \
    --filter="targetLink~/instances/${name}\$ AND operationType=insert" \
    --format="value(status,error.errors[0].code,error.errors[0].message)" \
    2>/dev/null | head -3 || true
}

# Same patterns the compile launcher treats as "try again later", so a
# refusal for capacity or quota continues and anything else is a bad request.
flex_is_capacity_error() {
  grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|resource pool exhausted|currently unavailable|no available capacity|QUOTA_EXCEEDED|Quota .* exceeded' "$1"
}

# Cancel a live request when the script is interrupted or dies. Install after
# the script's own EXIT cleanup is defined; this chains it.
#   flex_install_traps cleanup_fn
flex_install_traps() {
  FLEX_USER_CLEANUP="${1:-}"
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  trap flex_on_exit EXIT
}

flex_on_exit() {
  local rc=$?
  trap - EXIT
  if [[ "$FLEX_LIVE" == "1" && -n "$FLEX_NAME" ]]; then
    echo "!! leaving with a request live: cancelling $FLEX_NAME" >&2
    flex_cancel "$FLEX_NAME" "$FLEX_ZONE" || rc=3
  fi
  if [[ -n "${FLEX_USER_CLEANUP:-}" ]]; then "$FLEX_USER_CLEANUP"; fi
  exit "$rc"
}
