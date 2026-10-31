#!/usr/bin/env bash
# Tear down an AWS session instance in the one order that cannot lose
# evidence (task T5.1; the counterpart of gcp_teardown_session.sh, whose
# header gives the history behind each step):
#   1. console log FIRST: it goes with the instance, and it is the only
#      diagnostic channel that survives losing SSH
#   2. results: they need SSH, which may already be gone; a failure here
#      never prevents step 4
#   3. age manifest and session_cost.txt: the manifest needs SSH; the cost
#      falls back to the instance's LaunchTime from the API when SSH is gone
#   4. terminate (never stop: a stopped instance still bills for its volume),
#      wait until terminated, then remove the hard-delete schedule
#   5. verify nothing is left billing, in every region
#
# Steps 1 to 3 may fail without aborting: an instance whose SSH is dead is
# exactly when deletion matters most.
#
# Refused before anything is copied or deleted:
#   - no declared rate (AWS_RATE_INR_HR, Rs/h all-in);
#   - the account or profile not pinned (scripts/_aws_account.sh);
#   - the instance is in another region than the one given (the GCP
#     teardown's 2 Oct 2026 wrong-zone failure); the right region is named;
#   - the instance has no attnbench-session tag: this script deletes only
#     instances aws_launch_session.sh created.
#
#   AWS_RATE_INR_HR=... bash scripts/aws_teardown_session.sh INSTANCE_ID REGION [RESULTS_DIR]
#
# SSH: AWS_SSH_KEY (private key path) and AWS_SSH_USER (default ubuntu).
set -uo pipefail

ID="${1:?usage: aws_teardown_session.sh INSTANCE_ID REGION [RESULTS_DIR]}"
REGION="${2:?region required}"
S="${3:-results/gpu_session_$(date +%Y%m%d)}"

RATE_INR_HR="${AWS_RATE_INR_HR:-}"
if [[ ! "$RATE_INR_HR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "REFUSING: set AWS_RATE_INR_HR to the planning rate of the instance used (Rs/h," >&2
  echo "all-in, plan Section 5.2). Nothing was copied or deleted." >&2
  exit 2
fi
if [[ ! "$ID" =~ ^i-[0-9a-f]{8,17}$ ]]; then
  echo "REFUSING: '$ID' is not an EC2 instance id. Nothing was copied or deleted." >&2
  exit 2
fi
if [[ ! "$REGION" =~ ^[a-z]{2}(-[a-z]+)+-[0-9]$ ]]; then
  echo "REFUSING: '$REGION' is not an AWS region. Nothing was copied or deleted." >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/_aws_account.sh"
aws_account_pin
P=(--profile "$AWS_PIN_PROFILE" --region "$REGION")

describe() {   # state, session tag, public DNS, launch time; tab-separated
  aws ec2 describe-instances --profile "$AWS_PIN_PROFILE" --region "$1" --instance-ids "$ID" \
    --query 'Reservations[0].Instances[0].[State.Name,Tags[?Key==`attnbench-session`]|[0].Value,PublicDnsName,LaunchTime]' \
    --output text 2>/dev/null
}

FOUND=1
if ! INFO="$(describe "$REGION")" || [[ -z "$INFO" ]]; then
  ACTUAL=""
  for R in $(aws ec2 describe-regions --profile "$AWS_PIN_PROFILE" --region "$AWS_STS_REGION" \
               --query 'Regions[].RegionName' --output text 2>/dev/null); do
    [[ "$R" == "$REGION" ]] && continue
    if describe "$R" >/dev/null; then ACTUAL="$R"; break; fi
  done
  if [[ -n "$ACTUAL" ]]; then
    echo "REFUSING: $ID is not in $REGION; it is in $ACTUAL. Nothing was copied or deleted." >&2
    echo "Re-run: bash $0 $ID $ACTUAL $S" >&2
    exit 2
  fi
  echo "   NOTE: $ID was not found in any region -- nothing to copy or delete"
  FOUND=0
fi

STATE="" SESSION="" DNS="" LAUNCH=""
if [[ "$FOUND" -eq 1 ]]; then
  IFS=$'\t' read -r STATE SESSION DNS LAUNCH <<<"$INFO"
  if [[ -z "$SESSION" || "$SESSION" == "None" ]]; then
    echo "REFUSING: $ID has no attnbench-session tag, so aws_launch_session.sh did not" >&2
    echo "create it. This script deletes only the study's own instances. Nothing was" >&2
    echo "copied or deleted." >&2
    exit 2
  fi
  [[ "$DNS" == "None" ]] && DNS=""
  echo "   $ID: session $SESSION, state $STATE"
fi
mkdir -p "$S"

SSH_USER="${AWS_SSH_USER:-ubuntu}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
[[ -n "${AWS_SSH_KEY:-}" ]] && SSH_OPTS+=(-i "$AWS_SSH_KEY")
CAN_SSH=0
[[ "$FOUND" -eq 1 && -n "$DNS" && "$STATE" == "running" ]] && CAN_SSH=1

echo "== 1/5 console log -> $S/console.log (before anything can destroy it)"
# Fetched to a temporary file and moved into place only on success, so a
# failed fetch on a re-run never empties the log a first run saved.
if [[ "$FOUND" -eq 1 ]] && aws ec2 get-console-output "${P[@]}" --instance-id "$ID" --latest \
       --query Output --output text > "$S/console.log.partial" 2>/dev/null \
     && [[ -s "$S/console.log.partial" ]] && [[ "$(head -c 4 "$S/console.log.partial")" != "None" ]]; then
  mv "$S/console.log.partial" "$S/console.log"
  echo "   saved $(wc -l < "$S/console.log") lines"
  echo -n "   OOM kills recorded: "
  grep -ciE "oom-kill|Killed process" "$S/console.log" || true
else
  rm -f "$S/console.log.partial"
  echo "   WARNING: could not fetch the console log -- continuing to deletion anyway"
fi

echo "== 2/5 results"
if [[ "$CAN_SSH" -eq 1 ]]; then
  scp "${SSH_OPTS[@]}" -r "$SSH_USER@$DNS:~/attnbench_scaffold/results" "$S/" 2>/dev/null \
    && echo "   copied" \
    || echo "   WARNING: scp failed (SSH may be gone) -- continuing to deletion"
  for LOG in fa_build.log bsa_build.log batch_probe.log anchor16k.log; do
    scp "${SSH_OPTS[@]}" "$SSH_USER@$DNS:~/$LOG" "$S/$LOG" 2>/dev/null \
      && echo "   copied $LOG" || true
  done
else
  echo "   WARNING: no SSH (state '$STATE', public DNS '${DNS:-none}') -- continuing to deletion"
fi

echo "== 3/5 age manifest and session cost"
# Both boot fields come FROM THE INSTANCE, as in the GCP teardown: the
# instance converts its own clock, and only the epoch crosses the wire.
BOOT_LINE=""
if [[ "$CAN_SSH" -eq 1 ]]; then
  BOOT_LINE="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$DNS" \
    'printf "%s|%s\n" "$(uptime -s)" "$(date -d "$(uptime -s)" +%s)"' 2>/dev/null | tr -d '\r')"
fi
BOOT_UTC="${BOOT_LINE%%|*}"
BOOT_EPOCH="${BOOT_LINE##*|}"
BASIS="instance boot clock (uptime -s, over SSH)"
if [[ -n "$BOOT_UTC" ]]; then
  echo "   instance booted: $BOOT_UTC"
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$DNS" "
    echo 'boot_time: $BOOT_UTC'
    echo '--- PREDATES BOOT (image-carried, NOT this session) ---'
    find ~/attnbench_scaffold/results ~/*.log -type f ! -newermt '$BOOT_UTC' 2>/dev/null | sort
    echo '--- PRODUCED THIS SESSION ---'
    find ~/attnbench_scaffold/results ~/*.log -type f -newermt '$BOOT_UTC' 2>/dev/null | sort
  " > "$S/file_age_manifest.txt" 2>/dev/null \
    && echo "   wrote $S/file_age_manifest.txt" \
    || echo "   WARNING: could not build the manifest -- copied files are UNATTRIBUTED"
  PREDATE=$(sed -n '/PREDATES BOOT/,/PRODUCED THIS/p' "$S/file_age_manifest.txt" 2>/dev/null \
            | grep -c '^/' || true)
  if [[ "${PREDATE:-0}" -gt 0 ]]; then
    echo "   !! ${PREDATE} file(s) predate this boot: IMAGE-CARRIED, not this"
    echo "      session's output. Do not read them as measurements from today."
  fi
else
  echo "   WARNING: could not read boot time over SSH -- copied files are UNATTRIBUTED"
  # The API's LaunchTime is when the instance last entered running, so it
  # still bounds the bill when SSH is gone. It resets on a stop and start,
  # which then under-counts; the basis line says which clock was used.
  if [[ "$LAUNCH" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2} ]]; then
    T="${LAUNCH:0:19}"
    BOOT_EPOCH="$(command date -u -d "${T/T/ }" +%s 2>/dev/null \
                  || command date -u -j -f '%Y-%m-%dT%H:%M:%S' "$T" +%s 2>/dev/null || true)"
    BOOT_UTC="${T}Z"
    BASIS="EC2 LaunchTime (API; SSH unavailable; resets on stop and start)"
  fi
fi
if [[ "$BOOT_EPOCH" =~ ^[0-9]+$ ]]; then
  MINS=$(( ( $(command date +%s) - BOOT_EPOCH + 59 ) / 60 ))
  COST=$(awk -v m="$MINS" -v r="$RATE_INR_HR" 'BEGIN{printf "%.0f", m*r/60}')
  printf 'provider: aws\ninstance: %s\nsession: %s\nregion: %s\nboot: %s\nbasis: %s\nteardown: %s\nminutes: %s\nrate_inr_hr: %s\nest_inr: %s\n' \
    "$ID" "$SESSION" "$REGION" "$BOOT_UTC" "$BASIS" "$(command date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$MINS" "$RATE_INR_HR" "$COST" > "$S/session_cost.txt"
  echo "   billable: ${MINS} min  ~= INR ${COST} at ${RATE_INR_HR}/hr -> $S/session_cost.txt"
else
  echo "   WARNING: no boot or launch time -- session_cost.txt NOT written"
fi

echo "== 4/5 terminate (not stop -- a stopped instance still bills for its volume)"
if [[ "$FOUND" -eq 1 ]]; then
  if aws ec2 terminate-instances "${P[@]}" --instance-ids "$ID" >/dev/null; then
    aws ec2 wait instance-terminated "${P[@]}" --instance-ids "$ID" \
      && echo "   terminated" \
      || echo "   WARNING: terminate requested but not yet confirmed -- see step 5"
  else
    echo "   !! TERMINATE FAILED -- terminate $ID in $REGION by hand, now"
  fi
  # The schedule deletes itself after it fires; once the instance is gone it
  # has nothing to do, so it is removed. Absent is fine.
  aws scheduler delete-schedule "${P[@]}" --name "attnbench-hard-delete-$ID" >/dev/null 2>&1 \
    && echo "   removed its hard-delete schedule" || true
fi

echo "== 5/5 verify nothing is left billing"
bash "$(dirname "$0")/aws_cleanup_check.sh"
