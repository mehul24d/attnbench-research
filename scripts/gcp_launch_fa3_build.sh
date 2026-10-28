#!/usr/bin/env bash
# Launch the CPU VM that builds the FA3 wheel (T6.4, docs/t64_fa3_launch_card.md).
# Creates a BILLABLE resource. Run only after the researcher's yes to the
# launch card; the yes is typed at the prompt below.
#
# Researcher's choices (28 Oct 2026): n2-standard-16, on-demand, us-central1,
# a hard cap sized from the planning rate, no third-party wheel.
#   - n2-standard-16, 16 vCPU / 64 GB, no accelerator. N2_CPUS is 200; the
#     all-regions CPU line is 32, so this VM and any GPU VM never run at once
#     (16 + 26 for the H100 is 42).
#   - STANDARD (on-demand): a preemption would restart a build the H100
#     session waits on.
#   - Planning rate Rs 77.21/h all-in, us-central1, 20 Oct 2026 catalogue at
#     Rs 95.99/$: N2 core $0.031611 x 16 + N2 RAM $0.004237 x 64 GiB
#     (Rs 74.58) + 200 GB pd-balanced at $0.10/GiB-month over 730 h (Rs 2.63).
#   - Hard cap: --max-run-duration=3h, action DELETE, so at most Rs 232 at the
#     planning rate (researcher's settings for the restricted build, 28 Oct
#     2026). The in-guest halt at +150 min stops compute billing first;
#     build_fa3_wheel.sh stops its own build at 120 min, stores the logs (and
#     a wheel, if any) and halts the VM itself as soon as the build ends.
#   - The same disk image as the H100 (a plain disk image boots any family,
#     docs/machine_image_family_lock.md), so the wheel's Python and torch are
#     the H100's.
#   - --scopes=cloud-platform, or the upload to the results bucket 403s.
#
# This script only creates the instance. Deploy, build and teardown are the
# launch card's steps.
set -euo pipefail

ZONE="${GCP_ZONE:-us-central1-a}"   # us-central1, the H100's pinned zone
source "$(dirname "${BASH_SOURCE[0]}")/_launch_policy.sh"
launch_policy_check "$ZONE"
RATE_INR_HR="${GCP_RATE_INR_HR:-}"
if [[ ! "$RATE_INR_HR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "REFUSING TO LAUNCH -- declare the planning rate as GCP_RATE_INR_HR (Rs/h, all-in):" >&2
  echo "  77.21 for n2-standard-16 in us-central1 (docs/t64_fa3_launch_card.md)." >&2
  echo "Pass the same rate to gcp_teardown_session.sh." >&2
  exit 1
fi
source "$(dirname "${BASH_SOURCE[0]}")/_gcp_project.sh"
gcp_project_pin
MACHINE_TYPE="n2-standard-16"
IMAGE="${GCP_IMAGE:-attnbench-env-v5-20260905}"
MAX_RUN_HOURS=3
MAX_RUN="${MAX_RUN_HOURS}h"
HALT_MINUTES=150
BOOT_DISK_SIZE="200GB"
BOOT_DISK_TYPE="pd-balanced"
INSTANCE_NAME="${1:-attnbench-fa3-build-$(date +%Y%m%d-%H%M)}"
CEILING_INR="$(awk -v r="$RATE_INR_HR" -v h="$MAX_RUN_HOURS" 'BEGIN { printf "%.0f", r * h }')"

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
logger "attnbench-fa3-build: scheduling in-guest halt in ${HALT_MINUTES} minutes"
shutdown -h +${HALT_MINUTES}
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : ATTNBENCH_GCP_PROJECT (pinned, not printed)"
echo "  name         : $INSTANCE_NAME"
echo "  zone         : $ZONE"
echo "  machine-type : $MACHINE_TYPE (no accelerator)"
echo "  image        : $IMAGE"
echo "  boot disk    : $BOOT_DISK_SIZE ($BOOT_DISK_TYPE)"
echo "  provisioning : STANDARD (on-demand, not spot)"
echo "  declared rate: Rs ${RATE_INR_HR}/h (GCP_RATE_INR_HR; planning rate Rs 77.21/h)"
echo "  in-guest halt: +${HALT_MINUTES} min -- compute billing stops, disk survives"
echo "  hard cap     : --max-run-duration=$MAX_RUN, action=DELETE -- at most Rs ${CEILING_INR}"
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
  --image="$IMAGE" \
  --image-project="$PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="$BOOT_DISK_TYPE" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --metadata=attnbench-provisioning-model=STANDARD \
  --max-run-duration="$MAX_RUN" \
  --instance-termination-action=DELETE \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo
echo "Created. This instance is now billing. Remember:"
echo "  - it halts at +${HALT_MINUTES} min from boot regardless of what you do"
echo "  - it DELETES at $MAX_RUN from boot, disk included"
echo "  - delete it through scripts/gcp_teardown_session.sh $INSTANCE_NAME $ZONE"
echo "    with GCP_RATE_INR_HR=$RATE_INR_HR, then run scripts/gcp_cleanup_check.sh"
