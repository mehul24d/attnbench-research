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
listing() {
  local heading="$1"; shift
  local out rc=0
  out="$(aws "$@")" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
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

if [[ "$QUERY_FAILED" -eq 1 ]]; then
  echo "INCOMPLETE: at least one listing failed (marked !! above). Not clean until it runs."
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
