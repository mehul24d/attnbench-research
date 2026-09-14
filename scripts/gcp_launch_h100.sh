#!/usr/bin/env bash
# Launch the H100 (sm90) measurement instance on DWS Flex Start.
#
# Creates a BILLABLE resource. Rate pinned from the Cloud Billing catalog on
# 2026-09-13 for asia-southeast1, at Rs 88/$:
#
#   DWS Flex Start   GPU $4.2008  vCPU $0.2843  RAM $0.2228  disk $0.0301
#                    -> $4.7380/h -> Rs 417/h all-in
#   (Spot is $8.05/h = Rs 709/h. For H100 in this region Flex Start is
#    41% CHEAPER than Spot -- the opposite of the usual assumption.)
#
# Approved bracket: expected 3h30m = Rs 1,460 ; hard cap 5h = Rs 2,085.
#
# Non-negotiables baked in:
#   - --provisioning-model=FLEX_START: a bounded window with no preemption
#     inside it, so no segment splitting and no partial-band branch.
#   - --max-run-duration=5h with --instance-termination-action=DELETE: the cap
#     is enforced by GCP and DELETES the instance (not stops it), so a
#     forgotten session cannot bill past the approved ceiling and cannot leave
#     a disk behind either.
#   - --maintenance-policy=TERMINATE (required for GPU; stated so it cannot
#     silently drift).
#   - the belt-and-braces in-guest `shutdown -h +285` as well, so the guest
#     also stops itself slightly before the GCP-side deletion.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-asia-southeast1-b}"
MACHINE_TYPE="a3-highgpu-1g"
ACCELERATOR="type=nvidia-h100-80gb,count=1"
IMAGE="attnbench-env-v5-20260905"
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
MAX_RUN="5h"
INSTANCE_NAME="${1:-attnbench-h100-$(date +%Y%m%d-%H%M)}"

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<'EOF'
#!/bin/bash
logger "attnbench-h100: scheduling in-guest shutdown in 285 minutes"
shutdown -h +285
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : $PROJECT"
echo "  name         : $INSTANCE_NAME"
echo "  zone         : $ZONE"
echo "  machine-type : $MACHINE_TYPE"
echo "  accelerator  : $ACCELERATOR"
echo "  image        : $IMAGE"
echo "  boot disk    : $BOOT_DISK_SIZE ($BOOT_DISK_TYPE)"
echo "  provisioning : FLEX_START (DWS)  -- Rs 417/h all-in"
echo "  hard cap     : --max-run-duration=$MAX_RUN, action=DELETE (Rs 2,085 ceiling)"
echo

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --image="$IMAGE" \
  --image-project="$PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="$BOOT_DISK_TYPE" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=FLEX_START \
  --instance-termination-action=DELETE \
  --max-run-duration="$MAX_RUN" \
  --reservation-affinity=none \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo
echo "Created. Billing now. It self-DELETES at $MAX_RUN from start."
echo "  teardown early: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT --quiet"
echo "  confirm clean : bash scripts/gcp_cleanup_check.sh"
