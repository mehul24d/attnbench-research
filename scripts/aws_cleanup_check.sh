#!/usr/bin/env bash
# One-command end-of-session cost check for the study's AWS account (task
# T5.1; the counterpart of gcp_cleanup_check.sh). Read-only: it lists, it
# never deletes.
#
# AWS resources are regional, and a forgotten instance in a region nobody
# looks at bills as surely as one in the session's region. So every region
# enabled for the account is checked, not only the one a session used.
#
# Exit 1 when anything that bills by the hour exists: instances in any state
# but terminated (a stopped instance still bills for its volumes), volumes
# not attached to an instance, Elastic IPs. Images (AMIs) and snapshots are
# listed and named in the summary but are not errors: an image is meant to
# outlive its instance. The same lesson as the GCP check's: a summary that
# says "Clean" while storage bills is the failure, so it says which.
# Hard-delete schedules left behind are listed too: they bill nothing, but
# one that remains means a session's hard delete has not fired yet.
# Every listing is region-wide, so it covers every availability zone; the
# zones are named per region, and the study's instances terminated within
# the last hour or so (EC2 keeps them visible that long) are listed with
# their zone, so a launch that fell back to another zone shows where it ran
# (the researcher's yes of 23 Nov 2026). Neither is an error.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_aws_account.sh"
aws_account_pin

echo "AWS Cleanup Check, account: ATTNBENCH_AWS_ACCOUNT (pinned, not printed)"
echo "=================================================================="

REGIONS="$(aws ec2 describe-regions --profile "$AWS_PIN_PROFILE" --region "$AWS_STS_REGION" \
             --query 'Regions[].RegionName' --output text)"
if [[ -z "$REGIONS" ]]; then
  echo "ERROR: could not list the account's regions. Nothing was checked." >&2
  exit 2
fi

FOUND_SOMETHING=0
QUERY_FAILED=0
IMAGES_SEEN=0
SNAPSHOTS_SEEN=0

# One listing: prints it under its heading when non-empty and returns 0 when
# it found something. A failed query is not "nothing": it is counted and
# makes the check fail, since a listing that could not run proves nothing.
#
# A listing's error text is never printed as it comes: the organization's
# policy denies EC2 outside the study's Region, and that denial names the
# policy's ARN, the role and its session id (26 Nov 2026). A denial by the
# organization's policy is named as such, per region, and counted apart; any
# other error is printed with ARNs and 12-digit numbers masked. Either way the
# listing did not run, so it is not "nothing" (the exit is unchanged).
mask_err() {
  sed -E 's/arn:aws[A-Za-z0-9-]*:[^[:space:]"'"'"',)]*/arn:<masked>/g;
          s/assumed-role\/[^[:space:]"'"'"',)]+/assumed-role\/<masked>/g;
          s/(^|[^0-9])[0-9]{12}([^0-9]|$)/\1<id>\2/g' | sed 's/^/     /' >&2
}
ORG_DENIED_REGIONS=""
listing() {
  local heading="$1"; shift
  local out rc=0 errf err
  errf="$(mktemp)"
  out="$(aws "$@" 2>"$errf")" || rc=$?
  err="$(cat "$errf")"; rm -f "$errf"
  if [[ "$rc" -ne 0 ]]; then
    if printf '%s' "$err" | grep -qi 'service control policy'; then
      echo "   (denied by the organization's policy, not listed: $heading)"
      case " $ORG_DENIED_REGIONS " in *" $R "*) ;; *) ORG_DENIED_REGIONS="${ORG_DENIED_REGIONS:+$ORG_DENIED_REGIONS }$R" ;; esac
      [[ "${SOFT:-0}" -eq 1 ]] || QUERY_FAILED=1
      return 1
    fi
    [[ -n "$err" ]] && printf '%s\n' "$err" | mask_err
    if [[ "${SOFT:-0}" -eq 1 ]]; then
      echo "   (could not list: $heading)"
    else
      echo "   !! query failed: $heading"
      QUERY_FAILED=1
    fi
    return 1
  fi
  out="$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | grep -v '^None$' || true)"
  if [[ -n "$out" ]]; then
    echo "   $heading:"
    printf '%s\n' "$out" | sed 's/^/     /'
    return 0
  fi
  return 1
}

for R in $REGIONS; do
  echo
  echo "## $R"
  P=(--profile "$AWS_PIN_PROFILE" --region "$R")
  listing "instances not terminated (a stopped instance still bills for its volumes)" \
    ec2 describe-instances "${P[@]}" \
    --filters "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,Tags[?Key==`attnbench-session`]|[0].Value]' \
    --output text && FOUND_SOMETHING=1
  listing "unattached volumes (billed with no instance)" \
    ec2 describe-volumes "${P[@]}" --filters "Name=status,Values=available" \
    --query 'Volumes[].[VolumeId,Size,VolumeType,AvailabilityZone]' --output text \
    && FOUND_SOMETHING=1
  listing "Elastic IPs (billed by the hour)" \
    ec2 describe-addresses "${P[@]}" \
    --query 'Addresses[].[PublicIp,AllocationId,InstanceId]' --output text \
    && FOUND_SOMETHING=1
  listing "images owned by this account (storage-billed, not an error)" \
    ec2 describe-images "${P[@]}" --owners self \
    --query 'Images[].[ImageId,Name,CreationDate]' --output text \
    && IMAGES_SEEN=1
  listing "snapshots owned by this account (storage-billed, not an error)" \
    ec2 describe-snapshots "${P[@]}" --owner-ids self \
    --query 'Snapshots[].[SnapshotId,VolumeSize,StartTime]' --output text \
    && SNAPSHOTS_SEEN=1
  SOFT=1 listing "availability zones these listings cover" \
    ec2 describe-availability-zones "${P[@]}" --filters Name=zone-type,Values=availability-zone \
    --query 'join(`", "`, AvailabilityZones[].ZoneName)' --output text || true
  SOFT=1 listing "the study's instances terminated recently, with their zone (not billed)" \
    ec2 describe-instances "${P[@]}" \
    --filters "Name=instance-state-name,Values=terminated" "Name=tag-key,Values=attnbench-session" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,Tags[?Key==`attnbench-session`]|[0].Value]' \
    --output text || true
  # Soft: Scheduler is not offered in every region, and a schedule bills nothing.
  SOFT=1 listing "hard-delete schedules still pending (not billed)" \
    scheduler list-schedules "${P[@]}" --name-prefix attnbench-hard-delete- \
    --query 'Schedules[].[Name,State]' --output text || true
done

echo
echo "=================================================================="
STORAGE_KINDS=()
[[ "$IMAGES_SEEN" -eq 1 ]]    && STORAGE_KINDS+=("images")
[[ "$SNAPSHOTS_SEEN" -eq 1 ]] && STORAGE_KINDS+=("snapshots")

if [[ -n "$ORG_DENIED_REGIONS" ]]; then
  echo "Denied by the organization's policy (not listed): $(echo $ORG_DENIED_REGIONS | wc -w | tr -d ' ') region(s): $ORG_DENIED_REGIONS"
fi
if [[ "$QUERY_FAILED" -eq 1 ]]; then
  echo "INCOMPLETE: at least one listing failed (marked !! or denied above). Not clean until it runs."
  exit 2
elif [[ "$FOUND_SOMETHING" -eq 1 ]]; then
  echo "ATTENTION: billable resources exist above. Review before walking away."
  exit 1
elif [[ ${#STORAGE_KINDS[@]} -gt 0 ]]; then
  # bash 3.2 (macOS): "${ARR[@]}" on an empty array aborts under set -u,
  # hence the length test first.
  echo "No compute running -- but storage still bills: $(IFS=,; echo "${STORAGE_KINDS[*]}" | sed 's/,/, /g')."
  echo "Listed above. Delete any that are superseded."
  exit 0
else
  echo "Clean -- no instances, unattached volumes, Elastic IPs, images or snapshots in any region."
  exit 0
fi
