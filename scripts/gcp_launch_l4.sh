#!/usr/bin/env bash
# Launch the Stage 3/timing-probe validation L4 instance. Creates a BILLABLE
# resource -- do not run this without having already confirmed the on-demand
# hourly rate and session estimate with whoever owns the billing account.
#
# Non-negotiables baked in, per the validation-session spending rules:
#   - startup script hard-caps the instance to 4 hours via `shutdown -h +240`,
#     so a forgotten instance self-terminates instead of billing indefinitely.
#   - --maintenance-policy=TERMINATE (required for GPU instances anyway;
#     stated explicitly so it can never silently drift to MIGRATE).
#   - --scopes=cloud-platform. NOT OPTIONAL: default GCE scopes grant only
#     devstorage.read_only, so every gsutil cp to the results bucket 403s.
#     A 4h47m H100 session (Rs 2,029) was lost to its absence on 2026-09-15.
#   - --max-run-duration with DELETE, so a forgotten session cannot bill past
#     the ceiling; the in-guest halt is set WELL short of it (90 min vs 3h) so
#     there is a window in which the disk still exists and can be recovered.
#   - --provisioning-model is left at its default (STANDARD/on-demand) --
#     deliberately NOT spot for this session: preemption mid-probe would
#     waste more than spot pricing saves.
#   - A Deep Learning VM base image (CUDA + PyTorch preinstalled), so the
#     kernel-package install phase isn't also fighting a bare-OS CUDA setup.
#   - Boot disk capped at 200GB.
#
# This script only creates the instance. It does not install anything, run
# any probe, or delete anything -- see the Phase 1-6 runbook.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="g2-standard-8"
ACCELERATOR="type=nvidia-l4,count=1"
IMAGE="${ATTNBENCH_IMAGE:-attnbench-env-v5-20260905}"   # the project env, not a stock ML image
# Overridable, like gcp_launch_a100.sh, because the defaults are sized for a
# full Stage 3 band and a short session inherits a ceiling it cannot reach.
# The 2026-09-08 ledger line -- 162 idle minutes at Rs 0 measured, on an
# instance that had self-halted but was not deleted -- is the cheap version of
# this; the expensive version is a session that hangs and bills to the cap.
# Set both for a short run: GCP_HALT_MINUTES well under GCP_MAX_RUN, so the
# GPU stops billing while the disk survives for recovery.
MAX_RUN="${GCP_MAX_RUN:-7h}"
HALT_MINUTES="${GCP_HALT_MINUTES:-300}"   # DELETE at MAX_RUN; the gap is the
                                          # disk-recovery window.
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
INSTANCE_NAME="${1:-attnbench-l4-validation-$(date +%Y%m%d-%H%M)}"

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
# Hard 4-hour cap: this instance self-terminates even if every teardown
# step in the session runbook is skipped or forgotten. Scheduled from
# instance boot time, not from launch-script invocation time.
logger "attnbench-l4: scheduling in-guest halt in ${HALT_MINUTES} minutes"
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
echo "  provisioning : STANDARD (on-demand, not spot)"
echo "  in-guest halt: +${HALT_MINUTES} min -- GPU billing stops, disk survives"
echo "  hard cap     : --max-run-duration=$MAX_RUN action=DELETE (at Rs 78/h)"
echo "  rate         : Rs 78/h (g2-standard-8 + 1x L4, us-central1, pinned 2026-09-16)"
echo
read -r -p "Type 'launch' to proceed, anything else to abort: " CONFIRM
if [[ "$CONFIRM" != "launch" ]]; then
  echo "Aborted -- no instance created."
  exit 1
fi

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
  --provisioning-model=STANDARD \
  --max-run-duration="$MAX_RUN" \
  --instance-termination-action=DELETE \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo
echo "Created. This instance is now billing. Remember:"
echo "  - it self-terminates at 4 hours from boot regardless of what you do"
echo "  - delete (not stop) it the moment the session is done:"
echo "      gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
echo "  - run scripts/gcp_cleanup_check.sh afterward to confirm nothing is left billing"
