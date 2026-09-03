#!/usr/bin/env bash
# Launch the dedicated COMPILE session instance. Creates a BILLABLE resource --
# do not run this without having already confirmed the on-demand hourly rate
# and session estimate with whoever owns the billing account.
#
# This is not the validation launcher (scripts/gcp_launch_l4.sh). Two
# deliberate differences, both of which cost money if they drift:
#
#   1. It boots from the v2 machine image captured at the end of the
#      2026-09-02 validation session, NOT from a bare Deep Learning VM
#      family image. The v2 image already carries the working environment
#      (torch 2.9.1+cu129, transformers pinned to 4.46.0 for the torchaudio
#      ABI break, flash-linear-attention, the repo, and the model weights),
#      so this session spends its hours compiling rather than re-doing an
#      install that was already paid for once.
#
#   2. The hard cap is 6 hours, not 4. flash-attn and Block-Sparse-Attention
#      both build their own multi-architecture gencode lists internally and
#      ignore TORCH_CUDA_ARCH_LIST (confirmed by inspection AND by watching
#      `ptxas -arch sm_90` run during a supposedly sm_89-scoped build), so
#      neither can be scoped down to this card. That is exactly why they were
#      deferred out of the 4-hour validation session and given a session of
#      their own.
#
# Everything else is unchanged and non-negotiable: on-demand (not spot --
# preemption 70 minutes into a compile wastes far more than spot saves),
# TERMINATE maintenance policy, self-terminating startup script, and an
# interactive confirmation before anything bills.
#
# This script only creates the instance. It installs nothing, builds nothing,
# and deletes nothing -- see the compile-session runbook.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${GCP_ZONE:-asia-south1-b}"
MACHINE_TYPE="g2-standard-8"
SOURCE_MACHINE_IMAGE="${GCP_SOURCE_IMAGE:-attnbench-l4-image-v3-20260903}"
CAP_MINUTES="${GCP_CAP_MINUTES:-360}"
INSTANCE_NAME="${1:-attnbench-l4-compile-$(date +%Y%m%d-%H%M)}"

# Zones tried in order, stopping at the FIRST success. asia-south1 stocked
# out in all three zones on 2026-09-03 and cleared on a retry minutes later,
# so a fallback list is worth having -- but a fallback list is exactly where
# a duplicate-instance bug lives, hence the break-on-success below and the
# preflight after it. Both are required; neither alone is sufficient.
ZONE_FALLBACKS="${GCP_ZONE_FALLBACKS:-$ZONE}"

if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

# PREFLIGHT: refuse to launch if this project already has an instance.
#
# On 2026-09-03 an ad-hoc zone-retry loop did not break after a successful
# create and went on to attempt the remaining two zones. Both happened to be
# stocked out, so exactly one instance existed -- luck, not design. Had they
# had capacity, three g2-standard-8 + L4 instances would have been created
# and billed simultaneously, which is the worst failure mode this project
# has (see the standing rule about a forgotten instance).
#
# The loop is fixed below, but a loop-shaped bug should not be the only thing
# standing between a typo and triple billing. This check is structural: it
# does not care why a second create was attempted.
EXISTING="$(gcloud compute instances list --project="$PROJECT" \
              --format="value(name,zone,status)" 2>/dev/null)"
if [[ -n "$EXISTING" ]] && [[ "${GCP_ALLOW_SECOND_INSTANCE:-0}" != "1" ]]; then
  echo "REFUSING TO LAUNCH -- this project already has instance(s):" >&2
  echo "$EXISTING" >&2
  echo >&2
  echo "Every session in this study uses exactly one instance. If the one" >&2
  echo "above is a leftover, delete it (do not stop it -- a stopped instance" >&2
  echo "still bills for its disk):" >&2
  echo "    gcloud compute instances delete NAME --zone=ZONE --quiet" >&2
  echo "If you genuinely need a second, set GCP_ALLOW_SECOND_INSTANCE=1." >&2
  exit 1
fi

if ! gcloud compute machine-images describe "$SOURCE_MACHINE_IMAGE" \
      --project="$PROJECT" --format="value(status)" 2>/dev/null | grep -q READY; then
  echo "Source machine image '$SOURCE_MACHINE_IMAGE' is missing or not READY." >&2
  echo "Booting this session from a bare DLVM image instead would spend most" >&2
  echo "of the 6-hour cap redoing an install, so this fails closed." >&2
  exit 1
fi

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
# Hard cap: this instance self-terminates even if every teardown step in the
# runbook is skipped or forgotten. Scheduled from instance boot time, not
# from launch-script invocation time. A compile session is the likeliest
# place to walk away from a terminal, so this matters more here, not less.
#
# STANDING HAZARD -- this runs on EVERY boot, not just the first. A stop/start
# (capturing a machine image is the usual reason) re-arms the full cap from
# the new boot time, so an instance intended to die at T+6h can quietly live
# to T+11h. The cap is a backstop against forgetting, and this is the one way
# it fails open. After any restart, re-arm the original wall-clock deadline
# by hand: `sudo shutdown -h HH:MM`.
logger "attnbench-l4-compile: scheduling hard shutdown in $CAP_MINUTES minutes"
shutdown -h +$CAP_MINUTES
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : $PROJECT"
echo "  name         : $INSTANCE_NAME"
echo "  zone(s)      : $ZONE_FALLBACKS (tried in order, STOPPING at first success)"
echo "  machine-type : $MACHINE_TYPE (1x nvidia-l4, inherited from the image)"
echo "  source image : $SOURCE_MACHINE_IMAGE (machine image)"
echo "  provisioning : STANDARD (on-demand, not spot)"
echo "  hard cap     : shutdown -h +$CAP_MINUTES ($((CAP_MINUTES / 60))h from boot)"
echo
read -r -p "Type 'launch' to proceed, anything else to abort: " CONFIRM
if [[ "$CONFIRM" != "launch" ]]; then
  echo "Aborted -- no instance created."
  exit 1
fi

# Try each zone in turn and STOP at the first success.
#
# The `break` is the whole point of this loop, and it is the line that was
# missing from the ad-hoc version on 2026-09-03. A stockout creates nothing
# and costs nothing, so continuing after a FAILURE is free; continuing after
# a SUCCESS bills a second GPU instance. Those two cases look nearly
# identical in a hurried one-liner, which is why this lives in a script.
CREATED_ZONE=""
for Z in $ZONE_FALLBACKS; do
  echo
  echo ">> attempting $Z"
  if gcloud compute instances create "$INSTANCE_NAME" \
      --project="$PROJECT" \
      --zone="$Z" \
      --machine-type="$MACHINE_TYPE" \
      --source-machine-image="$SOURCE_MACHINE_IMAGE" \
      --maintenance-policy=TERMINATE \
      --provisioning-model=STANDARD \
      --metadata-from-file=startup-script="$STARTUP_SCRIPT"; then
    CREATED_ZONE="$Z"
    break                     # <-- do not remove: see comment above
  fi
  echo ">> $Z unavailable (nothing created, nothing billed)"
done

if [[ -z "$CREATED_ZONE" ]]; then
  echo >&2
  echo "No zone in '$ZONE_FALLBACKS' had capacity. Nothing was created and" >&2
  echo "nothing is billing. Stockouts have cleared within minutes before --" >&2
  echo "retrying the same list shortly is reasonable." >&2
  exit 1
fi

echo
echo "Created in $CREATED_ZONE."
echo "Created. This instance is now billing. Remember:"
echo "  - it self-terminates at $((CAP_MINUTES / 60))h from boot regardless of what you do"
echo
echo "  !! THE CAP RESETS ON EVERY BOOT. The shutdown is scheduled by the"
echo "     startup script, which re-runs when the instance starts. Stopping"
echo "     this instance (e.g. to capture a machine image) and starting it"
echo "     again silently re-arms a FRESH $((CAP_MINUTES / 60))h from the new boot,"
echo "     extending the session past the deadline you agreed to."
echo "     After any stop/start, immediately re-arm the ORIGINAL deadline:"
echo "         sudo shutdown -h HH:MM   # original cap wall-clock time"
echo "     Or avoid the stop entirely: machine images can be captured from a"
echo "     running instance (sync first)."
echo "  - delete (not stop) it the moment the session is done:"
echo "      bash scripts/gcp_teardown_session.sh $INSTANCE_NAME $CREATED_ZONE"
echo "    (that saves the serial log BEFORE deleting -- the 2026-09-02"
echo "     session lost its only diagnostic evidence by deleting first)"
echo "  - run scripts/gcp_cleanup_check.sh afterward to confirm nothing is left billing"
echo "  - machine images are billed separately from instances and are NOT"
echo "    covered by gcp_cleanup_check.sh -- check them with:"
echo "      gcloud compute machine-images list --project=$PROJECT"
