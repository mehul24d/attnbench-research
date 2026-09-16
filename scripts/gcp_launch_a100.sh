#!/usr/bin/env bash
# Launch the A100-80GB (sm80) Stage 5 instance: DWS Flex Start first, Spot as
# fallback.
#
# Creates a BILLABLE resource. Rates pinned from the Cloud Billing catalog,
# asia-southeast1, verified 2026-09-16:
#
#   DWS Defined Duration A100 80GB   $2.277654/h   <- attempted first
#   Spot A100 80GB                   $2.648100/h   <- fallback
#   on-demand A100 80GB              $4.846072/h   <- quota is 0, unusable
#
# All-in on Spot = GPU + 12 x $0.023340 core + 170 x $0.003128 RAM
#                = $3.4599/h ~= Rs 321/h.
#
# DWS is both CHEAPER than Spot and non-preemptible. That inversion was first
# recorded for H100 Flex Start (41% under Spot) and holds on a second
# accelerator here, so it is a pattern rather than an H100 quirk. It does NOT
# hold for L4, where Flex Start and on-demand are priced identically -- so the
# rule is "check the SKU per accelerator", not "DWS is always cheaper".
#
# Whether DWS draws on PREEMPTIBLE_NVIDIA_A100_80GB_GPUS (limit 1, the only
# non-zero A100 quota in the project) is NOT answerable from the quota API.
# The only test is an attempt, which is why this script attempts and falls
# back rather than deciding in advance.
#
# ONE ZONE. a2-ultragpu-1g is offered in asia-southeast1-c and in neither -a
# nor -b (re-verified 2026-09-16). There is nothing to walk, so a zone loop
# would turn one refusal into a slow one.
#
# Non-negotiables baked in, same as the H100 launcher:
#   --scopes=cloud-platform      (its absence cost Rs 2,029 on 2026-09-15)
#   --max-run-duration + DELETE  (GCP-enforced ceiling, leaves no disk behind)
#   --maintenance-policy=TERMINATE
#   in-guest halt WELL short of the DELETE, so there is a recovery window in
#   which billing has stopped but the disk still exists.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-asia-southeast1-c}"
MACHINE_TYPE="a2-ultragpu-1g"
ACCELERATOR="type=nvidia-a100-80gb,count=1"
IMAGE="attnbench-env-v5-20260905"
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
MAX_RUN="2h"
HALT_MINUTES="90"    # DELETE is at 2h; the 30m gap is the recovery window.
INSTANCE_NAME="${1:-attnbench-a100-$(date +%Y%m%d-%H%M)}"

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
logger "attnbench-a100: scheduling in-guest halt in ${HALT_MINUTES} minutes"
shutdown -h +${HALT_MINUTES}
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : $PROJECT"
echo "  name         : $INSTANCE_NAME"
echo "  zone         : $ZONE  (only zone offering $MACHINE_TYPE)"
echo "  machine-type : $MACHINE_TYPE  (12 vCPU, 170GB, 1x A100 80GB)"
echo "  image        : $IMAGE  (plain DISK image -- no accelerator binding)"
echo "  boot disk    : $BOOT_DISK_SIZE ($BOOT_DISK_TYPE)"
echo "  provisioning : FLEX_START attempted first, SPOT on refusal"
echo "  rate         : Rs ~284/h on DWS, Rs 321/h on Spot (pinned 2026-09-16)"
echo "  in-guest halt: +${HALT_MINUTES} min"
echo "  hard cap     : --max-run-duration=$MAX_RUN, action=DELETE (Rs 642 ceiling)"
echo "  recovery win : ~30m between halt and DELETE"
echo

create_with() {
  local model="$1"; shift
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
    --provisioning-model="$model" \
    --instance-termination-action=DELETE \
    --max-run-duration="$MAX_RUN" \
    --reservation-affinity=none \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" "$@"
}

echo "=== attempt 1: FLEX_START (DWS, \$2.277654/h GPU, non-preemptible) ==="
if create_with FLEX_START; then
  MODEL_USED="FLEX_START"
else
  echo
  echo "FLEX_START refused. Falling back to SPOT (\$2.648100/h GPU, preemptible)."
  echo "Per-band sync means a preemption costs at most one band."
  echo
  echo "=== attempt 2: SPOT ==="
  create_with SPOT
  MODEL_USED="SPOT"
fi

echo
echo "Created on $MODEL_USED. Billing now."
echo "  halts   at +${HALT_MINUTES}m (disk still recoverable after this)"
echo "  DELETES at $MAX_RUN (disk goes with it)"
echo "GATE: verify the GCS write path before starting any phase:"
echo "  bash scripts/gcp_verify_gcs_writable.sh $INSTANCE_NAME $ZONE"
echo "  teardown early: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT --quiet"
echo "  confirm clean : bash scripts/gcp_cleanup_check.sh"
echo "PROVISIONING_MODEL=$MODEL_USED"
