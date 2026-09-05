#!/usr/bin/env bash
# One-command end-of-session cost check.
# This account is paid with no auto-suspend: a forgotten instance or disk
# keeps billing indefinitely. Run this before closing your terminal.
# Read-only — lists resources, never deletes anything.
set -uo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

echo "GCP Cleanup Check — project: $PROJECT"
echo "=================================================================="

FOUND_SOMETHING=0

echo
echo "## Running / stopped instances (any state still bills for disks)"
INSTANCES=$(gcloud compute instances list --project="$PROJECT" \
  --format="table(name,zone,machineType.basename(),status)" 2>/dev/null)
if [[ -n "$INSTANCES" ]]; then
  echo "$INSTANCES"
  FOUND_SOMETHING=1
else
  echo "(none)"
fi

echo
echo "## Unattached persistent disks (billed even with no instance)"
DISKS=$(gcloud compute disks list --project="$PROJECT" \
  --filter="-users:*" \
  --format="table(name,zone,sizeGb,status)" 2>/dev/null)
if [[ -n "$DISKS" ]]; then
  echo "$DISKS"
  FOUND_SOMETHING=1
else
  echo "(none)"
fi

echo
echo "## Machine images (billed for storage indefinitely, independent of any instance)"
# These outlive every instance by design -- that is what they are for -- so
# unlike the sections above, finding one here is not automatically a problem.
# They are listed because they were previously invisible to this check
# entirely, which meant an ongoing storage charge nothing was watching.
IMAGES=$(gcloud compute machine-images list --project="$PROJECT" \
  --format="table(name,status,totalStorageBytes.size(units_out=G,precision=1):label=SIZE_GB,creationTimestamp.date('%Y-%m-%d'):label=CREATED)" 2>/dev/null)
if [[ -n "$IMAGES" ]]; then
  echo "$IMAGES"
  echo "(storage-billed, not an error -- delete any that are superseded)"
else
  echo "(none)"
fi

echo
echo "## Disk images (billed for storage indefinitely, independent of any instance)"
# A SECOND place storage lives, and the one this check missed until
# 2026-09-05. The machine-images section above was added because those were
# invisible here; the identical argument applies to plain disk images and was
# not carried across, so `attnbench-env-v5-20260905` (200 GB) billed while a
# script whose entire job is finding forgotten storage reported "Clean".
# --no-standard-images is load-bearing: without it this lists every public
# Debian/Ubuntu/COS image and the project's own rows are lost in the noise.
DISK_IMAGES=$(gcloud compute images list --project="$PROJECT" \
  --no-standard-images \
  --format="table(name,status,diskSizeGb,creationTimestamp.date('%Y-%m-%d'):label=CREATED)" 2>/dev/null)
if [[ -n "$DISK_IMAGES" ]]; then
  echo "$DISK_IMAGES"
  echo "(storage-billed, not an error -- delete any that are superseded)"
else
  echo "(none)"
fi

echo
echo "## Snapshots (billed for storage indefinitely, independent of any disk)"
# The third storage place. Listed for the same reason as the two above and
# because this project creates them transiently: the 2026-09-05 seed-and-
# restore path made one to move a boot disk between zones, and a snapshot
# left behind after that kind of one-off bills silently forever.
SNAPSHOTS=$(gcloud compute snapshots list --project="$PROJECT" \
  --format="table(name,diskSizeGb,status,creationTimestamp.date('%Y-%m-%d'):label=CREATED)" 2>/dev/null)
if [[ -n "$SNAPSHOTS" ]]; then
  echo "$SNAPSHOTS"
  echo "(storage-billed, not an error -- delete any left over from a transfer)"
else
  echo "(none)"
fi

echo
echo "## Reserved static IPs (billed while unattached, and sometimes while attached-but-unused)"
ADDRS=$(gcloud compute addresses list --project="$PROJECT" \
  --format="table(name,region,address,status)" 2>/dev/null)
if [[ -n "$ADDRS" ]]; then
  echo "$ADDRS"
  FOUND_SOMETHING=1
else
  echo "(none)"
fi

echo
echo "=================================================================="
# Storage artefacts are counted separately from FOUND_SOMETHING because they
# are not errors -- an image is supposed to outlive its instance. But they are
# NOT nothing, and the one-word summary is what a tired reader actually reads:
# on 2026-09-05 this script printed "Clean" over a 200 GB image it had not even
# looked for. A summary that says "clean" while storage bills is the same
# failure as not listing the storage at all, so it now says which.
STORAGE_KINDS=()
[[ -n "$IMAGES" ]]      && STORAGE_KINDS+=("machine images")
[[ -n "$DISK_IMAGES" ]] && STORAGE_KINDS+=("disk images")
[[ -n "$SNAPSHOTS" ]]   && STORAGE_KINDS+=("snapshots")

if [[ "$FOUND_SOMETHING" -eq 1 ]]; then
  echo "ATTENTION: billable resources exist above. Review before walking away."
  exit 1
elif [[ ${#STORAGE_KINDS[@]} -gt 0 ]]; then
  # bash 3.2 (macOS): "${ARR[@]}" on an empty array aborts under set -u, hence
  # the length test above rather than expanding unconditionally.
  # IFS=, so ${ARR[*]} joins with ", " instead of the default single space --
  # "machine images disk images" reads as one item, which defeats the point.
  echo "No compute running — but storage still bills: $(IFS=,; echo "${STORAGE_KINDS[*]}" | sed 's/,/, /g')."
  echo "Listed above with sizes. Delete any that are superseded."
  exit 0
else
  echo "Clean — no instances, unattached disks, reserved IPs, images or snapshots."
  exit 0
fi
