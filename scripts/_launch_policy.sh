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
# ATTNBENCH_TODAY (YYYY-MM-DD) overrides today's date. It exists so tests can
# exercise both sides of the cutoff without depending on when they run. It is
# not a way to launch after the date: changing the date is a plan decision,
# made by editing LAST_GCP_LAUNCH_DATE here and the plan in the same commit.

LAST_GCP_LAUNCH_DATE="2026-11-22"

launch_policy_check() {
  local zones="$*"
  local today="${ATTNBENCH_TODAY:-$(TZ=Asia/Kolkata date +%F)}"

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

  local z
  for z in $zones; do
    if [[ "$z" == australia-* ]]; then
      echo "REFUSING TO LAUNCH -- zone '$z' is in an Australia region, which the" >&2
      echo "plan forbids (docs/RESEARCH_PLAN.md, Section 6)." >&2
      exit 1
    fi
  done
}
