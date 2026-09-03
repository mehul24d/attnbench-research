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
if [[ "$FOUND_SOMETHING" -eq 1 ]]; then
  echo "ATTENTION: billable resources exist above. Review before walking away."
  exit 1
else
  echo "Clean — no instances, unattached disks, or reserved IPs found."
  exit 0
fi
