#!/usr/bin/env bash
# Launch the A100-80GB (sm80) Stage 5 instance: DWS Flex Start first, Spot as
# fallback.
#
# Creates a BILLABLE resource. GPU rates from the Cloud Billing catalog,
# asia-southeast1, checked 2026-10-20 (plan Section 5.2):
#
#   DWS Defined Duration A100 80GB   $2.277654/h   <- attempted first
#   Spot A100 80GB                   $2.780500/h   <- fallback
#   on-demand A100 80GB              $4.846072/h   <- quota is 0, unusable
#
# Planning rate, all-in = GPU + 12 cores + 170 GiB RAM + the 200 GB balanced
# boot disk + the one bundled local-SSD partition (375 GiB assumed):
#   DWS  = $2.277654 + 12 x $0.021449 + 170 x $0.002874 + disks
#        = $3.0990/h ~= Rs 297.47/h
#   Spot = $2.780500 + 12 x $0.023340 + 170 x $0.003128 + disks
#        = $3.6477/h ~= Rs 350.14/h
# (Superseded: Spot GPU $2.648100/h and Spot all-in $3.4599/h, Rs 321/h,
# GPU + cores + RAM only, pinned 2026-09-16.)
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

ZONE="${GCP_ZONE:-asia-southeast1-c}"   # plan Section 5.2: P2 pinned zone
source "$(dirname "${BASH_SOURCE[0]}")/_launch_policy.sh"
launch_policy_check "$ZONE"
# A declared rate, required: the planning rate this session is costed at,
# Rs/h all-in (plan Section 5.2). It is not defaulted, because the two ways
# this script can create differ by 18% and the teardown's old default (80, an
# L4 rate) under-costed an A100 session by a factor of 3.7.
RATE_INR_HR="${GCP_RATE_INR_HR:-}"
if [[ ! "$RATE_INR_HR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "REFUSING TO LAUNCH -- declare the planning rate as GCP_RATE_INR_HR (Rs/h, all-in):" >&2
  echo "  297.47 for FLEX_START (DWS), 350.14 for the Spot fallback (plan Section 5.2)." >&2
  echo "Pass the rate of the model actually used to gcp_teardown_session.sh as well." >&2
  exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/_gcp_project.sh"
gcp_project_pin
MACHINE_TYPE="a2-ultragpu-1g"
ACCELERATOR="type=nvidia-a100-80gb,count=1"
# Overridable so a freshly captured image can be put to work instead of
# sitting in storage. v6 (2026-09-17) is boot-tested -- see
# gcp_boot_test_image.sh -- and carries the repo at ae736b4 plus the built
# flash-attn, which is the expensive part of a cold setup.
IMAGE="${GCP_IMAGE:-attnbench-env-v5-20260905}"
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
# How long a FLEX_START request may wait for capacity before it fails
# (--request-valid-for-duration). Compute Engine accepts 90 s to 2 h for a
# request pinned to one zone (0 s takes only capacity that is free at once).
# A waiting request counts as launched when it is submitted (plan T0.17).
FLEX_REQUEST_VALID_FOR="15m"
# Overridable, and they SHOULD be overridden for short work. These defaults
# size a full multi-band session; a 35-minute job launched under them has a
# 4h30m ceiling, and on 2026-09-17 a session that finished its useful work in
# 98 minutes idled for another 90 (Rs 429) inside exactly that headroom
# (docs/silent_failure_patterns.md #40). The cap is not a safety net if it is
# set so far out that it never binds -- size it to the work.
MAX_RUN="${GCP_MAX_RUN:-4h30m}"
HALT_MINUTES="${GCP_HALT_MINUTES:-210}"   # DELETE at MAX_RUN; the gap is the recovery window.
INSTANCE_NAME="${1:-attnbench-a100-$(date +%Y%m%d-%H%M)}"


STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
logger "attnbench-a100: scheduling in-guest halt in ${HALT_MINUTES} minutes"
shutdown -h +${HALT_MINUTES}
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : ATTNBENCH_GCP_PROJECT (pinned, not printed)"
echo "  name         : $INSTANCE_NAME"
echo "  zone         : $ZONE  (only zone offering $MACHINE_TYPE)"
echo "  machine-type : $MACHINE_TYPE  (12 vCPU, 170GB, 1x A100 80GB)"
echo "  image        : $IMAGE  (plain DISK image -- no accelerator binding)"
echo "  boot disk    : $BOOT_DISK_SIZE ($BOOT_DISK_TYPE)"
echo "  provisioning : FLEX_START attempted first, SPOT on refusal"
echo "  rate         : Rs ~297/h on DWS, Rs ~350/h on Spot (planning rate, all-in, 2026-10-20)"
echo "  declared rate: Rs ${RATE_INR_HR}/h (GCP_RATE_INR_HR)"
echo "  in-guest halt: +${HALT_MINUTES} min"
echo "  hard cap     : --max-run-duration=$MAX_RUN, action=DELETE (Rs 1,339 on DWS, Rs 1,576 on Spot at the default 4h30m)"
echo "  recovery win : ~1h between halt and DELETE"
echo

create_with() {
  local model="$1"; shift
  # Launch definition (plan T0.17): every create request, the Spot fallback
  # included, must be submitted by the end of the last GCP launch date,
  # India time, so the date is checked again before each attempt.
  launch_policy_check "$ZONE"
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
    --metadata=attnbench-provisioning-model="$model" \
    --instance-termination-action=DELETE \
    --max-run-duration="$MAX_RUN" \
    --reservation-affinity=none \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" "$@"
}

echo "=== attempt 1: FLEX_START (DWS, Rs ~297/h all-in, non-preemptible) ==="
if create_with FLEX_START --request-valid-for-duration="$FLEX_REQUEST_VALID_FOR"; then
  MODEL_USED="FLEX_START"
else
  echo
  echo "FLEX_START refused. Falling back to SPOT (Rs ~350/h all-in, preemptible)."
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
echo "  teardown early: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=\$ATTNBENCH_GCP_PROJECT --quiet"
echo "  confirm clean : bash scripts/gcp_cleanup_check.sh"
if [[ "$MODEL_USED" == "SPOT" && "$RATE_INR_HR" != "350.14" ]]; then
  echo "NOTE: created on SPOT, but the declared rate is Rs ${RATE_INR_HR}/h. Tear down with"
  echo "  GCP_RATE_INR_HR=350.14, the Spot planning rate."
fi
echo "PROVISIONING_MODEL=$MODEL_USED"
