#!/usr/bin/env bash
# Launch the H100 (sm90) measurement instance on DWS Flex Start.
#
# Creates a BILLABLE resource. Rates pinned from the Cloud Billing catalog:
#
#   asia-southeast1 (2026-09-13)  -> $4.7380/h -> Rs 417/h all-in
#   us-central1     (2026-09-15)  -> $4.8237/h -> Rs 425/h all-in
#
# The DWS Flex Start H100 GPU SKU is globally FLAT at $4.200761/h -- Americas,
# Netherlands and Singapore are identical to six decimal places -- while the
# on-demand H100 SKU varies $9.80-$12.74 across the same regions. Only the
# vCPU/RAM/local-SSD lines move regionally, which is the whole Rs 8/h spread
# above. Flex Start is also 41% CHEAPER than Spot for H100.
#
# Approved bracket (us-central1): expected 3h30m = Rs 1,488 ; cap 5h = Rs 2,125.
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
#   - the in-guest halt is set WELL SHORT of the DELETE cap, not just slightly
#     short of it. Halt at 3h30m (210 min), DELETE at 5h, leaving a ~1h30m
#     window in which the instance is TERMINATED (GPU/vCPU billing stopped) but
#     the boot disk still EXISTS and can be mounted from a cheap CPU instance.
#     On 2026-09-15 the two were 13 minutes apart, which left no recovery path
#     at all. A bound on spend is not a bound on loss; this gap is the loss
#     bound.
#   - --scopes=cloud-platform. THIS IS NOT OPTIONAL. The default GCE scope set
#     grants `devstorage.read_only`, so every `gsutil cp` to a results bucket
#     fails 403. On 2026-09-15 a full 4h47m H100 session (Rs 2,029) was run
#     without it: the instance self-deleted on its run-duration cap, taking the
#     boot disk and every result row with it, and the bucket was empty because
#     the sync could never have worked. The cap and the in-guest shutdown both
#     behaved exactly as designed -- they bounded the spend and destroyed the
#     data, because the only copy was on the disk they deleted.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="a3-highgpu-1g"
ACCELERATOR="type=nvidia-h100-80gb,count=1"
IMAGE="attnbench-env-v5-20260905"
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
MAX_RUN="5h"
HALT_MINUTES="210"   # 3h30m. DELETE is at 5h; the gap is the recovery window.
INSTANCE_NAME="${1:-attnbench-h100-$(date +%Y%m%d-%H%M)}"

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
logger "attnbench-h100: scheduling in-guest halt in ${HALT_MINUTES} minutes"
shutdown -h +${HALT_MINUTES}
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
echo "  in-guest halt: +${HALT_MINUTES} min (3h30m) -- GPU billing stops, disk survives"
echo "  hard cap     : --max-run-duration=$MAX_RUN, action=DELETE (Rs 2,125 ceiling)"
echo "  recovery win : ~1h30m between halt and DELETE"
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
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo
echo "Created. Billing now."
echo "  halts   at +${HALT_MINUTES}m (disk still recoverable after this)"
echo "  DELETES at $MAX_RUN (disk goes with it)"
echo "GATE: verify the GCS write path before starting any phase:"
echo "  bash scripts/gcp_verify_gcs_writable.sh $INSTANCE_NAME $ZONE"
echo "  teardown early: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT --quiet"
echo "  confirm clean : bash scripts/gcp_cleanup_check.sh"
