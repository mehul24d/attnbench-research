# Launch policy shared by every AWS launch script (task T5.1; the AWS
# counterpart of scripts/_launch_policy.sh). Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_aws_launch_policy.sh"
#     aws_launch_policy_check "$REGION" "$AZ"
#
# It refuses, before anything is created or billed:
#   - every launch while the last AWS launch date is UNDECIDED. The plan
#     (Section 6, rule 13; Gate C) allows no AWS session until T5.1 is done,
#     quotas are granted and each pairing is costed, and it sets no AWS
#     launch date yet. The researcher decides it; it is then written here and
#     in the plan in the same commit, as LAST_GCP_LAUNCH_DATE was;
#   - any launch after that date (India time, the researcher's time zone;
#     the date itself is inclusive);
#   - any region or zone in Australia: ap-southeast-2 (Sydney, with its
#     Local Zones such as ap-southeast-2-per-1) and ap-southeast-4
#     (Melbourne).
#
# A launch is a submitted `ec2 run-instances` request, as a GCP launch is a
# submitted create (plan T0.17). A script that tries more than once calls
# aws_launch_policy_check again before each attempt.
#
# Today's date comes only from `date`, called through `command`, exactly as
# in _launch_policy.sh: no environment variable changes it, and the tests pin
# it with a fake `date` first on PATH.

LAST_AWS_LAUNCH_DATE="UNDECIDED"

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
      ap-southeast-2*|ap-southeast-4*)
        echo "REFUSING TO LAUNCH -- '$p' is in an Australia region, which the plan" >&2
        echo "forbids (docs/RESEARCH_PLAN.md, Section 6)." >&2
        exit 1 ;;
    esac
  done
}
