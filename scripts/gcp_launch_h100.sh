#!/usr/bin/env bash
# Launch the H100 (sm90) measurement instance on DWS Flex Start (plan T6.4).
#
# HOW THE REQUEST IS MADE (changed 27 Oct 2026). The create is submitted with
# --async and then POLLED, in bounded calls, by scripts/_flex_request.sh. The
# old form was one create call that waited out the whole
# --request-valid-for-duration (up to 2 h) in a single client call. Now:
#   - every gcloud call has a client time limit and none runs for the wait;
#   - the date is re-checked (launch_policy_check) before EVERY submit, so an
#     attempt after midnight on the last launch date is refused; a request
#     already waiting is not (it counted as launched when it was submitted);
#   - the request is CANCELLED (deleted, and the delete confirmed) at the
#     client deadline GCP_CANCEL_AFTER_SECONDS (default 6900 s), five minutes
#     inside the server window of 2 h, which stays as the backstop for a
#     client that has gone away;
#   - at most one request is live: a zone is tried only after the previous
#     request is confirmed gone. Zones come from GCP_ZONE_FALLBACKS, in order
#     (default: the pinned zone only). a3-highgpu-1g is offered in
#     asia-southeast1-b and -c only, and the two regions are priced alike to
#     within Rs 1/h (below), so a fallback is a researcher's choice, not a
#     default: GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b".
#   - Ctrl-C, SIGTERM or a closed terminal cancels a live request on the way
#     out. A laptop that sleeps cannot: then the server window ends it, and a
#     VM that starts meanwhile is bounded by --max-run-duration below.
# The script ends when the VM is RUNNING, when every zone has been tried, or
# on an error. It does not wait for the VM to boot.
#
# Creates a BILLABLE resource. Planning rates, all-in, from the Cloud Billing
# catalog on 2026-10-20 (plan Section 5.2): GPU + 26 cores + 234 GiB RAM + the
# 200 GB balanced boot disk + the two bundled local-SSD partitions (375 GiB
# each assumed):
#
#   asia-southeast1  -> $4.8284/h -> Rs 463.47/h
#   us-central1      -> $4.8174/h -> Rs 462.42/h
#
# (Superseded: $4.7380/h, Rs 417/h, asia-southeast1, 2026-09-13; $4.8237/h,
# Rs 425/h, us-central1, 2026-09-15.)
#
# The DWS Flex Start H100 GPU SKU is globally FLAT at $4.200761/h -- Americas,
# Netherlands and Singapore are identical to six decimal places -- while the
# on-demand H100 SKU varies $9.80-$12.74 across the same regions. Only the
# vCPU/RAM/local-SSD lines move regionally. On 2026-10-20 the core and RAM
# lines were also equal in these two regions, and only the disk lines differ,
# which is the whole Rs ~1/h spread above. Flex Start is also 41% CHEAPER than Spot for H100.
#
# Approved bracket (us-central1): expected 3h30m = Rs 1,488 ; cap 5h = Rs 2,125.
# Those were approved at Rs 425/h. At the 2026-10-20 planning rate, Rs 462.42/h,
# the same durations are Rs 1,618 and Rs 2,312.
#
# Non-negotiables baked in:
#   - --provisioning-model=FLEX_START: a bounded window with no preemption
#     inside it, so no segment splitting and no partial-band branch.
#   - --max-run-duration=5h with --instance-termination-action=DELETE: the cap
#     is enforced by GCP and DELETES the instance (not stops it), so a
#     forgotten session cannot bill past the approved ceiling and cannot leave
#     a disk behind either.
#   - --maintenance-policy=TERMINATE (required for GPU; stated so it cannot
#     silently drift).
#   - the in-guest halt is set WELL SHORT of the DELETE cap, not just slightly
#     short of it. Halt at 3h30m (210 min), DELETE at 5h, leaving a ~1h30m
#     window in which the instance is TERMINATED (GPU/vCPU billing stopped) but
#     the boot disk still EXISTS and can be mounted from a cheap CPU instance.
#     On 2026-09-15 the two were 13 minutes apart, which left no recovery path
#     at all. A bound on spend is not a bound on loss; this gap is the loss
#     bound.
#   - --scopes=cloud-platform. THIS IS NOT OPTIONAL. The default GCE scope set
#     grants `devstorage.read_only`, so every `gsutil cp` to a results bucket
#     fails 403. On 2026-09-15 a full 4h47m H100 session (Rs 2,029) was run
#     without it: the instance self-deleted on its run-duration cap, taking the
#     boot disk and every result row with it, and the bucket was empty because
#     the sync could never have worked. The cap and the in-guest shutdown both
#     behaved exactly as designed -- they bounded the spend and destroyed the
#     data, because the only copy was on the disk they deleted.
set -euo pipefail

ZONE="${GCP_ZONE:-us-central1-a}"   # plan Section 5.2: P4 pinned zone
ZONE_FALLBACKS="${GCP_ZONE_FALLBACKS:-$ZONE}"   # zones tried in order, one request at a time
source "$(dirname "${BASH_SOURCE[0]}")/_launch_policy.sh"
launch_policy_check "$ZONE_FALLBACKS"
# A declared rate, required, as for the A100: the teardown's old default (80,
# an L4 rate) costs an H100 session at about a sixth of its rate.
RATE_INR_HR="${GCP_RATE_INR_HR:-}"
if [[ ! "$RATE_INR_HR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "REFUSING TO LAUNCH -- declare the planning rate as GCP_RATE_INR_HR (Rs/h, all-in):" >&2
  echo "  462.42 in us-central1, 463.47 in asia-southeast1 (plan Section 5.2)." >&2
  echo "Pass the same rate to gcp_teardown_session.sh." >&2
  exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/_gcp_project.sh"
gcp_project_pin
source "$(dirname "${BASH_SOURCE[0]}")/_flex_request.sh"
MACHINE_TYPE="a3-highgpu-1g"
ACCELERATOR="type=nvidia-h100-80gb,count=1"
IMAGE="${GCP_IMAGE:-attnbench-env-v5-20260905}"
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
# How long a FLEX_START request may wait for capacity before Compute Engine
# deletes it (--request-valid-for-duration). Compute Engine accepts 90 s to
# 2 h for a request pinned to one zone (0 s takes only capacity that is free
# at once). A waiting request counts as launched when it is submitted (plan
# T0.17). This is the server-side backstop; the client cancels earlier.
FLEX_REQUEST_VALID_FOR="2h"
FLEX_REQUEST_VALID_FOR_SECONDS=7200   # the same duration; a test holds the two equal
CANCEL_AFTER_SECONDS="${GCP_CANCEL_AFTER_SECONDS:-6900}"
CREATE_LIMIT="${GCP_FLEX_CREATE_LIMIT:-120}"   # seconds for the one --async submit call
if [[ ! "$CANCEL_AFTER_SECONDS" =~ ^[0-9]+$ ]] \
   || (( CANCEL_AFTER_SECONDS < 120 || CANCEL_AFTER_SECONDS >= FLEX_REQUEST_VALID_FOR_SECONDS )); then
  echo "REFUSING TO LAUNCH -- GCP_CANCEL_AFTER_SECONDS must be a whole number of seconds" >&2
  echo "from 120 to $((FLEX_REQUEST_VALID_FOR_SECONDS - 1)) (inside the server window of $FLEX_REQUEST_VALID_FOR)." >&2
  exit 1
fi
# Overridable, like the A100 and L4 launchers, so a short job is not launched
# under a ceiling sized for a full session (silent_failure_patterns #40).
MAX_RUN="${GCP_MAX_RUN:-5h}"
HALT_MINUTES="${GCP_HALT_MINUTES:-210}"   # DELETE at MAX_RUN; the gap is the recovery window.
INSTANCE_NAME="${1:-attnbench-h100-$(date +%Y%m%d-%H%M)}"


STARTUP_SCRIPT="$(mktemp)"
CREATE_ERR="$(mktemp)"
cleanup() { rm -f "$STARTUP_SCRIPT" "$CREATE_ERR"; }
flex_install_traps cleanup
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
logger "attnbench-h100: scheduling in-guest halt in ${HALT_MINUTES} minutes"
shutdown -h +${HALT_MINUTES}
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : ATTNBENCH_GCP_PROJECT (pinned, not printed)"
echo "  name         : $INSTANCE_NAME"
echo "  zone(s)      : $ZONE_FALLBACKS (in order, one request at a time)"
echo "  machine-type : $MACHINE_TYPE"
echo "  accelerator  : $ACCELERATOR"
echo "  image        : $IMAGE"
echo "  boot disk    : $BOOT_DISK_SIZE ($BOOT_DISK_TYPE)"
# Priced per zone: asia-southeast1 Rs ~463/h, us-central1 Rs ~462/h (header,
# planning rates of 2026-10-20; Rs 417/h and Rs 425/h until then).
# The rate is declared, and a fixed figure here would go stale. Printing each
# zone's own rate keeps the prompt and the cap consistent; an unrecognised
# zone says so rather than guessing.
for Z in $ZONE_FALLBACKS; do
  case "$Z" in
    asia-southeast1-*) RATE_LINE="Rs ~463/h all-in (asia-southeast1, planning rate 2026-10-20)" ;;
    us-central1-*)     RATE_LINE="Rs ~462/h all-in (us-central1, planning rate 2026-10-20)" ;;
    *)                 RATE_LINE="rate NOT PRICED for $Z -- confirm before proceeding" ;;
  esac
  echo "  $Z : FLEX_START (DWS)  -- $RATE_LINE"
done
echo "  declared rate: Rs ${RATE_INR_HR}/h (GCP_RATE_INR_HR)"
echo "  waiting      : submitted --async, polled every ${FLEX_POLL_SECONDS}s, cancelled after ${CANCEL_AFTER_SECONDS}s"
echo "                 (server window $FLEX_REQUEST_VALID_FOR); Google's pricing page lists no PENDING charge (not yet observed here)"
echo "  in-guest halt: +${HALT_MINUTES} min -- GPU billing stops, disk survives"
echo "  hard cap     : --max-run-duration=$MAX_RUN, action=DELETE, counted from RUNNING"
echo

# Every attempt, whatever its end, is one line here (plan Section 8: "every
# attempt logged"): submit time (UTC), instance, zone, outcome, seconds
# waited, the insert operation's reason when there is one. logs/ is
# git-ignored, so a launch never dirties the tree; the reviewed record is
# docs/launch_cards/h100_flex_attempts.md.
ATTEMPT_LOG="${GCP_FLEX_ATTEMPT_LOG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs/flex_attempts.tsv}"
mkdir -p "$(dirname "$ATTEMPT_LOG")"
log_attempt() {   # outcome waited reason
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$SUBMITTED_AT" "$INSTANCE_NAME" "$Z" "$1" "${2:-0}" \
    "$(printf '%s' "${3:-}" | tr '\t\n' '  ' | gcp_redact)" >> "$ATTEMPT_LOG"
}

CREATED_ZONE=""
for Z in $ZONE_FALLBACKS; do
  echo
  echo ">> attempting $Z"
  SUBMITTED_AT="$(command date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Launch definition (plan T0.17): every create request, each zone included,
  # must be submitted by the end of the last GCP launch date, India time, so
  # the date is checked again before each submit. Nothing is live here: the
  # previous request, if any, was confirmed gone before this iteration.
  launch_policy_check "$Z"

  # Marked live BEFORE the call: a submit that times out on the client may
  # still have been accepted by the server.
  FLEX_LIVE=1; FLEX_NAME="$INSTANCE_NAME"; FLEX_ZONE="$Z"
  SUBMIT_RC=0
  flex_bounded "$CREATE_LIMIT" gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$Z" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="$ACCELERATOR" \
    --image="$IMAGE" \
    --image-project="$PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --maintenance-policy=TERMINATE \
    --provisioning-model=FLEX_START \
    --metadata=attnbench-provisioning-model=FLEX_START \
    --request-valid-for-duration="$FLEX_REQUEST_VALID_FOR" \
    --instance-termination-action=DELETE \
    --max-run-duration="$MAX_RUN" \
    --reservation-affinity=none \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
    --async 2>"$CREATE_ERR" || SUBMIT_RC=$?

  if (( SUBMIT_RC != 0 )); then
    cat "$CREATE_ERR" >&2
    # Whatever the error said, look: only "not found" proves nothing exists.
    # A timed-out or unreadable submit is cancelled and the cancel confirmed.
    if [[ "$(flex_status "$INSTANCE_NAME" "$Z")" == "NOT_FOUND" ]]; then
      FLEX_LIVE=0
    else
      flex_cancel "$INSTANCE_NAME" "$Z" || exit 3
    fi
    if (( SUBMIT_RC == 124 )); then
      log_attempt submit_timeout 0
      echo "STOPPING -- the submit call did not answer within ${CREATE_LIMIT}s. Any request it" >&2
      echo "made was cancelled and confirmed gone. Check the network, then run again." >&2
      exit 1
    fi
    if flex_is_capacity_error "$CREATE_ERR"; then
      log_attempt capacity_refused 0 "$(head -c 300 "$CREATE_ERR")"
      echo ">> $Z unavailable (nothing created, nothing billed)"
      continue
    fi
    log_attempt rejected 0 "$(head -c 300 "$CREATE_ERR")"
    echo >&2
    echo "CONFIGURATION ERROR -- the request was rejected, and not for capacity." >&2
    echo "It will fail the same way in every zone. Nothing is waiting and nothing" >&2
    echo "is billing. Fix the request; do not run it again unchanged." >&2
    exit 1
  fi

  echo ">> request accepted in $Z; waiting (poll ${FLEX_POLL_SECONDS}s, cancel after ${CANCEL_AFTER_SECONDS}s)"
  if flex_wait "$INSTANCE_NAME" "$Z" "$CANCEL_AFTER_SECONDS"; then
    CREATED_ZONE="$Z"
    log_attempt granted "$FLEX_WAITED"
    break                     # <-- do not remove: a success must end the loop
  fi

  REASON=""
  case "$FLEX_OUTCOME" in
    expired|no_capacity) REASON="$(flex_end_reason "$INSTANCE_NAME" "$Z")" ;;
  esac
  log_attempt "$FLEX_OUTCOME" "$FLEX_WAITED" "$REASON"
  case "$FLEX_OUTCOME" in
    cancelled)
      echo ">> $Z: no capacity within ${FLEX_WAITED}s; request cancelled (nothing billed)" ;;
    no_capacity)
      echo ">> $Z: no capacity: the request was stopped after ${FLEX_WAITED}s; cancelled and confirmed gone (nothing billed)"
      [[ -n "$REASON" ]] && echo "   operation: $REASON" ;;
    expired)
      echo ">> $Z: the request ended after ${FLEX_WAITED}s without a VM (nothing billed)"
      if [[ -n "$REASON" ]]; then
        echo "   operation: $REASON"
        if printf '%s\n' "$REASON" | awk -F'\t' '$2 != "" {f = 1} END {exit !f}' \
           && ! printf '%s\n' "$REASON" | grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|QUOTA_EXCEEDED|RESOURCE_POOL|enough resources|Quota .* exceeded'; then
          echo "The operation reports an error that is not about capacity or quota." >&2
          echo "Fix the request; trying another zone would fail the same way." >&2
          exit 1
        fi
      fi ;;
    *)
      echo "STOPPING -- the request could not be followed ($FLEX_OUTCOME). No other zone was tried." >&2
      echo "Check: bash scripts/gcp_cleanup_check.sh" >&2
      exit 1 ;;
  esac
done

if [[ -z "$CREATED_ZONE" ]]; then
  echo >&2
  echo "No zone in '$ZONE_FALLBACKS' gave a VM. Nothing is waiting and nothing is billing." >&2
  echo "A request can be made again later (the launch date is checked again then)." >&2
  exit 1
fi

echo
echo "Granted in $CREATED_ZONE after ${FLEX_WAITED}s of waiting (VM status $FLEX_LAST_STATUS). Billing now."
echo "  halts   at +${HALT_MINUTES}m (disk still recoverable after this)"
echo "  DELETES at $MAX_RUN (disk goes with it)"
echo "GATE: verify the GCS write path before starting any phase:"
echo "  bash scripts/gcp_verify_gcs_writable.sh $INSTANCE_NAME $CREATED_ZONE"
echo "  teardown early: gcloud compute instances delete $INSTANCE_NAME --zone=$CREATED_ZONE --project=\$ATTNBENCH_GCP_PROJECT --quiet"
echo "  confirm clean : bash scripts/gcp_cleanup_check.sh"
echo "FLEX_WAIT_SECONDS=$FLEX_WAITED"
echo "PROVISIONING_MODEL=FLEX_START"
