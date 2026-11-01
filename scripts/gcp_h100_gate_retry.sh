#!/usr/bin/env bash
# The H100 FA3 correctness gate, retried until an H100 is granted (T6.4;
# researcher, 31 Oct 2026). Run it in a Mac terminal from a clean checkout of
# the launch commit, and keep the Mac awake:
#
#     caffeinate -i bash scripts/gcp_h100_gate_retry.sh
#
# Every H100_POLL_MINUTES (30), start to start, one round:
#   1. If any attnbench GPU VM (g2, a2 or a3, any status) exists in the
#      project, the round is skipped: one GPU session at a time, and the L4
#      sessions come first.
#   2. gcp_launch_h100.sh over H100_ZONES, FLEX_START, each zone's request
#      cancelled after H100_FLEX_WAIT_S (300 s).
#   3. If no zone had capacity and H100_SPOT=1, the same walk on SPOT.
# A round with no capacity anywhere (the launcher's exit 4) waits for the
# next one. Any other launcher failure stops the loop. Every attempt is a
# row in logs/flex_attempts.tsv (the launcher's) and every round a line in
# logs/h100_gate_retry.log.
#
# On a grant the loop stops and the gate runs on that VM, gate only, with
# nothing timed: the GCS write gate, deploy, preflight, bootstrap with the FA3
# wheel and no XAttention, then `t64_session.py run --gate-only`. Teardown
# runs on any exit after the grant, the gate's failure included.
#
# The cap. H100_CAP_INR (250) is held by the VM's --max-run-duration, which
# deletes it server-side at cap / rate minutes after RUNNING, whether or not
# this Mac is still awake. At the Flex rate (Rs 463.47/h, the higher of the
# two regions) that is 32 minutes, so the in-guest halt (GCP_HALT_MINUTES,
# 40) is a second line that the delete comes before. H100_SPOT=1 needs
# H100_SPOT_RATE_INR_HR (plan Section 5.2: Rs 668.04/h, the higher region).
#
# The Spot cap (researcher's rule, 1 Nov 2026, after Shakedown 5): Spot's
# hard delete is H100_SPOT_DELETE_MINUTES (40) after RUNNING, never fewer
# than 40, so it never comes before the in-guest halt, and its cap is what
# those minutes cost at the Spot rate, at most H100_SPOT_CAP_MAX_INR (350).
# Spot is refused when 40 minutes cost more than that: at the planning
# rates (Rs 629.94/h and Rs 668.04/h) they cost Rs 420 and Rs 446, so the
# rule allows Spot only at Rs 525/h or less.
#
# Stops: on a grant (after the gate and teardown), on a launcher error, after
# H100_MAX_ROUNDS rounds (48, a day), or on Ctrl-C (the launcher cancels a
# live request on the way out).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
source "$HERE/_gcp_project.sh"
gcp_project_pin

ZONES="${H100_ZONES:-us-central1-a asia-southeast1-b asia-southeast1-c}"
POLL_MINUTES="${H100_POLL_MINUTES:-30}"
FLEX_WAIT_S="${H100_FLEX_WAIT_S:-300}"
SPOT="${H100_SPOT:-1}"
CAP_INR="${H100_CAP_INR:-250}"
HALT_MINUTES="${GCP_HALT_MINUTES:-40}"
MAX_ROUNDS="${H100_MAX_ROUNDS:-48}"
GATE_MINUTES="${H100_GATE_MINUTES:-30}"
SPOT_DELETE_MINUTES="${H100_SPOT_DELETE_MINUTES:-40}"
SPOT_CAP_MAX="${H100_SPOT_CAP_MAX_INR:-350}"
FLEX_RATE="463.47"            # plan Section 5.2, 20 Oct 2026: the higher of us-central1 and asia-southeast1
SPOT_RATE="${H100_SPOT_RATE_INR_HR:-}"
export GCP_IMAGE="${GCP_IMAGE:-attnbench-env-v6-20260917}"
SHA="${FA3_WHEEL_SHA256:-}"
LOG="${H100_RETRY_LOG:-$ROOT/logs/h100_gate_retry.log}"
mkdir -p "$(dirname "$LOG")"

say() { printf '%s %s\n' "$(command date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | gcp_redact | tee -a "$LOG"; }
die() { say "STOP: $*"; exit 1; }

# Minutes of VM the cap buys at a rate, rounded down.
cap_minutes() { awk -v c="$CAP_INR" -v r="$1" 'BEGIN { printf "%d", c / r * 60 }'; }

[[ "$SHA" =~ ^[0-9a-f]{64}$ ]] || die "FA3_WHEEL_SHA256 is not set to the wheel's sha256"
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] \
  || die "the tree is dirty: the gate needs git_dirty = False, and a grant would be wasted"
FLEX_MAX=$(cap_minutes "$FLEX_RATE")
(( FLEX_MAX >= GATE_MINUTES )) \
  || die "Rs $CAP_INR buys $FLEX_MAX min at the Flex rate; the gate needs $GATE_MINUTES"
SPOT_MAX=0
if [[ "$SPOT" == 1 ]]; then
  if [[ ! "$SPOT_RATE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    die "H100_SPOT=1 needs H100_SPOT_RATE_INR_HR (no Spot planning rate is on record), or set H100_SPOT=0"
  fi
  (( SPOT_DELETE_MINUTES >= 40 )) \
    || die "H100_SPOT_DELETE_MINUTES=$SPOT_DELETE_MINUTES: Spot's hard delete is never under 40 min"
  (( SPOT_CAP_MAX <= 350 )) \
    || die "H100_SPOT_CAP_MAX_INR=$SPOT_CAP_MAX: the Spot cap is at most Rs 350"
  SPOT_MAX="$SPOT_DELETE_MINUTES"
  SPOT_CAP=$(awk -v r="$SPOT_RATE" -v m="$SPOT_MAX" \
    'BEGIN { c = r * m / 60; printf "%d", (c == int(c)) ? c : int(c) + 1 }')
  if (( SPOT_CAP > SPOT_CAP_MAX )); then
    say "Spot refused: $SPOT_MAX min at Rs ${SPOT_RATE}/h cost Rs $SPOT_CAP, above the Rs $SPOT_CAP_MAX Spot maximum"
    SPOT=0
  fi
fi

echo "H100 FA3 gate retry loop: TERMS"
echo "  commit       : $(git -C "$ROOT" rev-parse --short HEAD) (clean tree)"
echo "  zones        : $ZONES (in order)"
echo "  each round   : FLEX_START, ${FLEX_WAIT_S}s per zone$([[ "$SPOT" == 1 ]] && echo ", then SPOT")"
echo "  polling      : a round every ${POLL_MINUTES} min, start to start, at most $MAX_ROUNDS rounds"
echo "  before each  : skipped while any attnbench GPU VM exists in the project"
echo "  cap          : Rs $CAP_INR = hard delete after $FLEX_MAX min on Flex (Rs ${FLEX_RATE}/h)$([[ "$SPOT" == 1 ]] && echo "; Spot Rs $SPOT_CAP = hard delete after $SPOT_MAX min (Rs ${SPOT_RATE}/h, at most Rs $SPOT_CAP_MAX)")"
echo "  in-guest halt: +${HALT_MINUTES} min"
echo "  on a grant   : the loop stops; the gate runs (gate only, nothing timed); teardown on any exit"
echo "  image        : $GCP_IMAGE"
echo

# Every attnbench VM on a GPU machine family, any status.
gpu_vms() {
  gcloud compute instances list --project="$PROJECT" --filter="name~^attnbench-" \
    --format="value(name,zone.basename(),machineType.basename(),status)" \
    | awk '$3 ~ /^(g2|a2|a3)-/'
}

# One walk over the zones on one model. Sets GRANT_ZONE on a grant.
# Returns 0 granted, 4 no capacity anywhere, anything else an error.
walk() {   # model max_minutes rate
  local out rc=0
  out="$(mktemp)"
  # pipefail: the pipeline's status is the launcher's (tee succeeds).
  GCP_H100_PROVISIONING="$1" GCP_ZONE_FALLBACKS="$ZONES" GCP_ZONE="${ZONES%% *}" \
  GCP_CANCEL_AFTER_SECONDS="$FLEX_WAIT_S" GCP_MAX_RUN="${2}m" GCP_HALT_MINUTES="$HALT_MINUTES" \
  GCP_RATE_INR_HR="$3" bash "$HERE/gcp_launch_h100.sh" "$NAME" 2>&1 \
    | gcp_redact | tee -a "$LOG" | tee "$out" || rc=$?
  GRANT_ZONE="$(sed -n 's/^Granted in \([a-z0-9-]*\) .*/\1/p' "$out" | tail -1)"
  rm -f "$out"
  return "$rc"
}

ROUND=0
GRANT_ZONE=""
while :; do
  ROUND=$((ROUND + 1))
  (( ROUND <= MAX_ROUNDS )) || die "$MAX_ROUNDS rounds without a grant"
  START=$(command date +%s)
  NAME="attnbench-h100-gate-$(command date -u +%Y%m%d-%H%M)"
  if ! others="$(gpu_vms)"; then
    say "round $ROUND: skipped; the instance list could not be read"
  elif [[ -n "$others" ]]; then
    say "round $ROUND: skipped; an attnbench GPU VM exists: $(echo "$others" | tr '\n' ';')"
  else
    say "round $ROUND: $NAME, FLEX_START over $ZONES"
    rc=0; walk FLEX_START "$FLEX_MAX" "$FLEX_RATE" || rc=$?
    MODEL=FLEX_START; RATE="$FLEX_RATE"
    if (( rc == 4 )) && [[ "$SPOT" == 1 ]]; then
      say "round $ROUND: no Flex capacity; SPOT over $ZONES"
      rc=0; walk SPOT "$SPOT_MAX" "$SPOT_RATE" || rc=$?
      MODEL=SPOT; RATE="$SPOT_RATE"
    fi
    case "$rc" in
      0) [[ -n "$GRANT_ZONE" ]] || die "the launcher exited 0 but named no zone; check gcp_cleanup_check.sh"
         say "round $ROUND: GRANTED $NAME in $GRANT_ZONE on $MODEL; the loop stops"
         break ;;
      4) say "round $ROUND: no capacity in any zone (nothing live, nothing billed)" ;;
      *) die "the launcher exited $rc in round $ROUND; see $LOG and gcp_cleanup_check.sh" ;;
    esac
  fi
  WAKE=$((START + POLL_MINUTES * 60))
  NOW=$(command date +%s)
  if (( WAKE > NOW )); then
    say "next round at $(command date -u -r "$WAKE" +%H:%MZ 2>/dev/null || echo "+$(( (WAKE - NOW) / 60 )) min")"
    sleep $((WAKE - NOW))
  fi
done

# --- the gate, on the granted VM ---------------------------------------------------
ZONE="$GRANT_ZONE"
teardown() {
  local rc=$?
  trap - EXIT
  say "teardown of $NAME ($ZONE), the gate's exit $rc"
  bash "$HERE/gcp_session_elapsed.sh" "$NAME" "$ZONE" "$RATE" 2>&1 | gcp_redact | tee -a "$LOG" || true
  bash "$HERE/gcp_teardown_session.sh" "$NAME" "$ZONE" "results/t64_gate" 2>&1 | gcp_redact | tee -a "$LOG" || true
  bash "$HERE/gcp_cleanup_check.sh" 2>&1 | gcp_redact | tee -a "$LOG" || true
  say "done; log the attempt in docs/launch_cards/h100_flex_attempts.md"
  exit "$rc"
}
trap teardown EXIT
export GCP_RATE_INR_HR="$RATE" FA3_WHEEL_SHA256="$SHA"

say "waiting for SSH on $NAME"
for _ in $(seq 1 20); do
  gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet --command=true \
    >/dev/null 2>&1 && break
  sleep 15
done
step() { say "gate: $*"; "$@" 2>&1 | gcp_redact | tee -a "$LOG"; }   # a failure ends the script: teardown
step bash "$HERE/gcp_verify_gcs_writable.sh" "$NAME" "$ZONE"
step bash "$HERE/gcp_deploy_source.sh" "$NAME" "$ZONE"
step bash "$HERE/gcp_preflight_instance.sh" "$NAME" "$ZONE" --quarantine
step bash "$HERE/gcp_bootstrap_instance.sh" "$NAME" "$ZONE" --no-xattention
step gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
  --command="cd ~/attnbench_scaffold && python3 scripts/t64_session.py run --gate-only --fa3-sha256 $SHA"
