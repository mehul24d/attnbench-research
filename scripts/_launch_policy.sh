# Launch policy shared by every GCP launch script (docs/RESEARCH_PLAN.md,
# Section 6, rules 12 and 14). Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_launch_policy.sh"
#     launch_policy_check "$ZONE"            # or "$ZONE_FALLBACKS"
#
# It refuses, before anything is created or billed:
#   - any session after the last GCP launch date (India time, the
#     researcher's time zone; the date itself is inclusive);
#   - any zone in an Australia region.
#
# A launch is a submitted instance-create request (`compute instances
# create`; plan T0.17). Every request, a retry or a fallback included, must
# be submitted by the end of the last date, so a script that tries more than
# once calls launch_policy_check again before each attempt. A request that
# waits for capacity counts as launched when it is submitted.
#
# Today's date comes only from `date`, called through `command` so that a
# shell function named `date` (exported, or defined by BASH_ENV) cannot stand
# in for it. No environment variable changes it. This guards against an
# accidental override, such as a test setting left in a shell. It does not
# stop deliberate circumvention: a fake `date` placed first on PATH still
# changes the result, and that is how the tests pin the date, the same way
# they replace gcloud. Changing the last date is a plan decision, made by
# editing LAST_GCP_LAUNCH_DATE here and the plan in the same commit.

LAST_GCP_LAUNCH_DATE="2026-12-09"
# The GCP credit expires on 1 Dec 2026 (plan Section 5.2, Funding). A launch
# after this date, India time, is approved by the researcher personally and
# billed outside the credit; the check says so and goes on.
LAST_CREDIT_DATE="2026-11-30"

launch_policy_check() {
  local zones="$*"
  local today
  today="$(TZ=Asia/Kolkata command date +%F)"

  if [[ ! "$today" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "REFUSING TO LAUNCH -- cannot read today's date ('$today')." >&2
    exit 1
  fi
  # ISO dates compare correctly as strings.
  if [[ "$today" > "$LAST_GCP_LAUNCH_DATE" ]]; then
    echo "REFUSING TO LAUNCH -- today ($today, India time) is after the last GCP" >&2
    echo "launch date ($LAST_GCP_LAUNCH_DATE) set in docs/RESEARCH_PLAN.md." >&2
    exit 1
  fi
  if [[ "$today" > "$LAST_CREDIT_DATE" ]]; then
    echo "This launch is outside the GCP credit (expires 1 Dec 2026); billed to the researcher." >&2
  fi

  local z
  for z in $zones; do
    if [[ "$z" == australia-* ]]; then
      echo "REFUSING TO LAUNCH -- zone '$z' is in an Australia region, which the" >&2
      echo "plan forbids (docs/RESEARCH_PLAN.md, Section 6)." >&2
      exit 1
    fi
  done
}
