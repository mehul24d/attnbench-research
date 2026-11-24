# Launch policy shared by every AWS launch script (task T5.1; the AWS
# counterpart of scripts/_launch_policy.sh). Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_aws_launch_policy.sh"
#     aws_launch_policy_check "$REGION" "$AZ"
#
# It refuses, before anything is created or billed:
#   - every launch while LAST_AWS_LAUNCH_DATE is UNDECIDED;
#   - any launch after that date (India time, the researcher's time zone;
#     the date itself is inclusive);
#   - any Australia region but Sydney: ap-southeast-2 and its availability
#     zones (ap-southeast-2a, -2b, ...) are allowed (rule 12 lifted for AWS
#     Sydney only, the researcher's decision of 21 Nov 2026, plan Appendix
#     A); ap-southeast-4 (Melbourne) and Sydney's Local Zones (such as
#     ap-southeast-2-per-1a, Perth) stay refused.
#
# A launch is a submitted `ec2 run-instances` request, as a GCP launch is a
# submitted create (plan T0.17). A script that tries more than once calls
# aws_launch_policy_check again before each attempt.
#
# Today's date comes only from `date`, called through `command`, exactly as
# in _launch_policy.sh: no environment variable changes it, and the tests pin
# it with a fake `date` first on PATH.

# PROVISIONAL (28 Oct 2026): 20 Dec 2026 is proposed and NOT yet confirmed
# by the researcher. This line is the only place the date is set; when it is
# confirmed or changed, edit it here and in the plan in the same commit, as
# for LAST_GCP_LAUNCH_DATE. The plan (Section 6, rule 13; Gate C) still
# allows no AWS session before T5.1 is done, quotas are granted and each
# pairing is costed: this date is a last day, not a go-ahead.
LAST_AWS_LAUNCH_DATE="2026-12-20"

aws_launch_policy_check() {
  local places="$*"
  if [[ "$LAST_AWS_LAUNCH_DATE" == "UNDECIDED" ]]; then
    echo "REFUSING TO LAUNCH -- no last AWS launch date has been decided. AWS sessions" >&2
    echo "wait for T5.1, Gate C and the researcher's date (docs/RESEARCH_PLAN.md," >&2
    echo "Section 6, rule 13). Set LAST_AWS_LAUNCH_DATE in scripts/_aws_launch_policy.sh" >&2
    echo "and the plan in the same commit." >&2
    exit 1
  fi
  if [[ ! "$LAST_AWS_LAUNCH_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "REFUSING TO LAUNCH -- LAST_AWS_LAUNCH_DATE ('$LAST_AWS_LAUNCH_DATE') is not a date." >&2
    exit 1
  fi
  local today
  today="$(TZ=Asia/Kolkata command date +%F)"
  if [[ ! "$today" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "REFUSING TO LAUNCH -- cannot read today's date ('$today')." >&2
    exit 1
  fi
  # ISO dates compare correctly as strings.
  if [[ "$today" > "$LAST_AWS_LAUNCH_DATE" ]]; then
    echo "REFUSING TO LAUNCH -- today ($today, India time) is after the last AWS" >&2
    echo "launch date ($LAST_AWS_LAUNCH_DATE)." >&2
    exit 1
  fi

  local p
  for p in $places; do
    case "$p" in
      ap-southeast-2|ap-southeast-2[a-z])
        ;;  # Sydney and its availability zones: allowed (rule 12, 21 Nov 2026)
      ap-southeast-2*|ap-southeast-4*)
        echo "REFUSING TO LAUNCH -- '$p' is an Australia region other than Sydney, or a" >&2
        echo "Sydney Local Zone, which the plan forbids (docs/RESEARCH_PLAN.md, Section 6," >&2
        echo "rule 12; only ap-southeast-2 and its availability zones are allowed)." >&2
        exit 1 ;;
    esac
  done
}

# --- zone fallback (the researcher's launch yes of 23 Nov 2026, third rehearsal
# attempt: "fallback across 2b, 2c and 2a on a no-capacity error only") ------
# Twice that day run-instances was refused in ap-southeast-2a with
# InsufficientInstanceCapacity while AWS named 2b and 2c as having capacity.
# A launch script may therefore take an ordered list of zones and try them in
# one pass, no loop:
#
#     aws_zone_list "$REGION" "$AWS_SESSION_AZS"       # validates; sets AWS_ZONES
#     aws_launch_across_zones ATTEMPT_FN CONFIRM_FN "${AWS_ZONES[@]}"
#
# ATTEMPT_FN ZONE submits one create request in ZONE. It sets AWS_ATTEMPT_ID
# to the instance id ("" when none came back) and AWS_ATTEMPT_ERR to AWS's
# error text, and returns 0 only when an instance was created.
# CONFIRM_FN ZONE confirms, after a refusal, that the session has no instance
# and no volume anywhere in the region; it returns 0 only when it is sure.
#
# The next zone is tried only when the refusal is InsufficientInstanceCapacity,
# no instance id came back, and CONFIRM_FN confirms. Any other error, a stray
# instance or volume, or a confirmation that cannot be read stops at once.
# The launch policy is checked again before every attempt.
#
# Returns 0 with AWS_LAUNCHED_ZONE set; 3 when every zone refused for
# capacity; 1 otherwise. AWS_ZONES_REFUSED lists the zones refused for
# capacity, in order.
AWS_CAPACITY_ERROR="InsufficientInstanceCapacity"

aws_is_capacity_refusal() {     # aws_is_capacity_refusal "AWS error text"
  [[ "$1" == *"($AWS_CAPACITY_ERROR)"* ]]
}

aws_zone_list() {               # aws_zone_list REGION "ZONE ZONE ..."
  local region="$1" z seen=" "
  AWS_ZONES=()
  for z in $2; do
    if [[ ! "$z" =~ ^${region}[a-z]$ ]]; then
      echo "REFUSING TO LAUNCH -- zone '$z' in the zone list is not an availability zone of $region." >&2
      exit 1
    fi
    if [[ "$seen" == *" $z "* ]]; then
      echo "REFUSING TO LAUNCH -- zone '$z' appears twice in the zone list (one pass, no loop)." >&2
      exit 1
    fi
    seen+="$z "
    AWS_ZONES+=("$z")
  done
  if [[ ${#AWS_ZONES[@]} -eq 0 ]]; then
    echo "REFUSING TO LAUNCH -- the zone list is empty." >&2
    exit 1
  fi
}

aws_launch_across_zones() {     # aws_launch_across_zones ATTEMPT_FN CONFIRM_FN ZONE...
  # One pass, no wait, no precheck: the third attempt's form (23 Nov 2026).
  local attempt="$1" confirm="$2"; shift 2
  aws_launch_window "$attempt" "$confirm" true 0 "$@"
}

# --- the retry window (the researcher's launch yes of 24 Nov 2026, fourth
# rehearsal attempt: "retrying zones 2b, 2c and 2a on a no-capacity error
# only for at most 3 hours") ----------------------------------------------------
# On 23 Nov 2026 the third attempt was refused for capacity in all three
# Sydney zones within 17 s. The window repeats the one-pass fallback:
#
#     aws_launch_window ATTEMPT_FN CONFIRM_FN PRECHECK_FN WINDOW_MINUTES ZONE...
#
# A pass tries every zone in order, exactly as aws_launch_across_zones does.
# The next pass starts AWS_RETRY_INTERVAL_MINUTES (10) after the last one
# started (at once, if that pass took longer), and only while the window (at
# most AWS_RETRY_WINDOW_MAX_MINUTES, 180, from its start) has not ended and
# fewer than WINDOW_MINUTES / 10 passes have begun; WINDOW_MINUTES 0 is one
# pass. Before every attempt: the launch policy,
# then PRECHECK_FN ZONE (the launcher's: the security group still admits this
# Mac's current address on port 22). A failed precheck stops the window at once
# and nothing is changed to make it pass: the security group is never
# re-pointed here.
#
# Returns 0 with AWS_LAUNCHED_ZONE set; 3 when the window ended with every
# attempt refused for capacity; 4 when the precheck failed; 1 otherwise.
# AWS_ZONES_REFUSED lists the zones refused in the last pass, in order;
# AWS_WINDOW_PASSES counts the passes begun and AWS_CAPACITY_REFUSALS every
# confirmed capacity refusal of the window.
AWS_RETRY_INTERVAL_MINUTES=10
AWS_RETRY_WINDOW_MAX_MINUTES=180

aws_window_minutes_ok() {       # aws_window_minutes_ok MINUTES
  [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 <= AWS_RETRY_WINDOW_MAX_MINUTES ))
}

# One attempt in one zone. 0: created; 2: refused for capacity, nothing left
# (the caller may go on); 4: the precheck failed; 1: stop.
_aws_try_zone() {               # _aws_try_zone ATTEMPT_FN CONFIRM_FN PRECHECK_FN ZONE LABEL
  local attempt="$1" confirm="$2" precheck="$3" z="$4" label="$5"
  aws_launch_policy_check "$z"
  if ! "$precheck" "$z"; then
    echo "STOPPING -- before the create request in ${z:-the region}, the security group no longer" >&2
    echo "admits this Mac's current address on port 22 (above). It was not re-pointed; nothing" >&2
    echo "more is tried. Re-point it (sg_repoint.sh) and ask for a new launch." >&2
    return 4
  fi
  AWS_ATTEMPT_ID="" AWS_ATTEMPT_ERR=""
  echo "Zone ${z:-any} ($label): submitting the create request."
  if "$attempt" "$z"; then
    AWS_LAUNCHED_ZONE="$z"
    return 0
  fi
  if ! aws_is_capacity_refusal "$AWS_ATTEMPT_ERR"; then
    echo "STOPPING -- the create request in ${z:-the region} failed with an error other than" >&2
    echo "$AWS_CAPACITY_ERROR (above); no other zone is tried." >&2
    return 1
  fi
  if [[ -n "$AWS_ATTEMPT_ID" ]]; then
    echo "STOPPING -- ${z:-the region} refused for capacity but an instance id came back" >&2
    echo "('$AWS_ATTEMPT_ID'); no other zone is tried." >&2
    return 1
  fi
  echo "Zone ${z:-any} refused for capacity, no instance id came back; confirming nothing was created."
  if ! "$confirm" "$z"; then
    echo "STOPPING -- after the refusal in ${z:-the region}, it could not be confirmed that the" >&2
    echo "session has no instance and no volume (above); no other zone is tried." >&2
    return 1
  fi
  return 2
}

aws_launch_window() {           # aws_launch_window ATTEMPT_FN CONFIRM_FN PRECHECK_FN WINDOW_MINUTES ZONE...
  local attempt="$1" confirm="$2" precheck="$3" window="$4"; shift 4
  if ! aws_window_minutes_ok "$window"; then
    echo "REFUSING TO LAUNCH -- the retry window ('$window' minutes) must be whole minutes from 0" >&2
    echo "to $AWS_RETRY_WINDOW_MAX_MINUTES." >&2
    exit 1
  fi
  window=$(( 10#$window ))
  local interval_s=$(( AWS_RETRY_INTERVAL_MINUTES * 60 )) window_s=$(( window * 60 ))
  local passes=$(( (window_s + interval_s - 1) / interval_s ))
  (( passes < 1 )) && passes=1
  local z n total=$# rc start now next pass_start
  AWS_LAUNCHED_ZONE="" AWS_ZONES_REFUSED="" AWS_WINDOW_PASSES=0 AWS_CAPACITY_REFUSALS=0
  AWS_ATTEMPT_ID="" AWS_ATTEMPT_ERR=""
  start="$(command date -u +%s)"
  pass_start="$start"
  while :; do
    AWS_WINDOW_PASSES=$(( AWS_WINDOW_PASSES + 1 ))
    AWS_ZONES_REFUSED="" n=0
    (( passes > 1 )) && echo "Pass $AWS_WINDOW_PASSES of at most $passes (window $window min, a pass every $AWS_RETRY_INTERVAL_MINUTES min)."
    for z in "$@"; do
      n=$(( n + 1 ))
      rc=0
      _aws_try_zone "$attempt" "$confirm" "$precheck" "$z" "$n of $total" || rc=$?
      [[ $rc -eq 2 ]] || return $rc
      AWS_CAPACITY_REFUSALS=$(( AWS_CAPACITY_REFUSALS + 1 ))
      AWS_ZONES_REFUSED+="${AWS_ZONES_REFUSED:+ }${z:-any}"
    done
    (( AWS_WINDOW_PASSES >= passes )) && break
    next=$(( pass_start + interval_s ))
    now="$(command date -u +%s)"
    (( now >= start + window_s )) && break
    if (( next > now )); then
      echo "Pass $AWS_WINDOW_PASSES: every zone refused for capacity; the next pass starts in $(( next - now )) s (the window ends at +$window min)."
      command sleep $(( next - now ))
    fi
    pass_start="$(command date -u +%s)"
    (( pass_start >= start + window_s )) && break
  done
  if (( passes > 1 )); then
    echo "STOPPING -- the retry window ($window min) ended: every zone refused for capacity in each of" >&2
    echo "$AWS_WINDOW_PASSES passes ($AWS_CAPACITY_REFUSALS refusals)." >&2
  else
    echo "STOPPING -- every zone refused for capacity: $AWS_ZONES_REFUSED." >&2
  fi
  return 3
}
