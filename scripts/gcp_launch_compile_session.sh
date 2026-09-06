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
MACHINE_TYPE="${GCP_MACHINE_TYPE:-g2-standard-8}"
# v4, not v3. A stale default is not a harmless one: on 2026-09-05 a launch
# that set GCP_ZONE and the instance name but left this alone booted the v3
# image -- an image predating the deploy-provenance fix. Bump this when a new
# image is captured, in the same commit.
SOURCE_MACHINE_IMAGE="${GCP_SOURCE_IMAGE:-attnbench-l4-image-v4-20260903}"
CAP_MINUTES="${GCP_CAP_MINUTES:-360}"
INSTANCE_NAME="${1:-attnbench-l4-compile-$(date +%Y%m%d-%H%M)}"

# Zones tried in order, stopping at the FIRST success. asia-south1 stocked
# out in all three zones on 2026-09-03 and cleared on a retry minutes later,
# so a fallback list is worth having -- but a fallback list is exactly where
# a duplicate-instance bug lives, hence the break-on-success below and the
# preflight after it. Both are required; neither alone is sufficient.
ZONE_FALLBACKS="${GCP_ZONE_FALLBACKS:-$ZONE}"

# STANDARD (on-demand) by default -- see the header: preemption 70 minutes
# into a compile wastes far more than spot saves.
#
# Overridden to SPOT for the A100 sessions, and not as a cost decision: the
# project's A100 grant is PREEMPTIBLE_NVIDIA_A100_80GB_GPUS=1 while the
# on-demand NVIDIA_A100_80GB_GPUS quota is 0, so Spot is the only shape that
# can be created at all. Band-major checkpointing is what makes that
# survivable, and check_host_continuity refuses to resume a preempted run
# into the same results file -- the recovery path is a new segment, joined at
# analysis time.
PROVISIONING_MODEL="${GCP_PROVISIONING_MODEL:-STANDARD}"

# Accelerator override. Empty means "inherit whatever the machine image
# recorded", which is right for every G2/L4 session and WRONG for A2.
#
# attnbench-l4-image-v4 records guestAccelerators: nvidia-l4, because that is
# the card it was captured from. Creating from it with --machine-type=
# a2-ultragpu-1g would carry an L4 request onto a machine type whose A100 is
# part of the machine type itself. Set GCP_ACCELERATOR to replace the
# inherited list, e.g.:
#
#   GCP_ACCELERATOR="type=nvidia-a100-80gb,count=1"
#
# A create that is rejected for a bad accelerator shape costs nothing -- the
# instance never exists -- so this is safe to get wrong once. It is NOT safe
# to leave implicit, which is why it is a named variable and not a comment.
ACCELERATOR="${GCP_ACCELERATOR:-}"

# Boot disk interface. Empty means "inherit from the machine image", which is
# right for G2 and WRONG for A2 -- the same shape as GCP_ACCELERATOR above.
#
# attnbench-l4-image-v4 was captured from a g2-standard-8, and g2-vm supports
# NVME, so the image records interface: NVME. a2-ultragpu-1g is not in the
# NVME-capable family list and the create is rejected outright:
#
#   Invalid value for field 'resource.disks[0].interface': 'NVME'.
#
# Set GCP_BOOT_DISK_INTERFACE=SCSI for A2. This is the second image property
# that must not be inherited across families; if a third appears, this file is
# where it goes.
BOOT_DISK_INTERFACE="${GCP_BOOT_DISK_INTERFACE:-}"

# Boot disk shape, used ONLY on the plain-disk-image path. A machine image
# carries its own disk record and these are ignored there; a disk image
# carries only contents, so gcloud creates the boot disk and needs to be told
# how big and of what type.
BOOT_DISK_SIZE="${GCP_BOOT_DISK_SIZE:-200GB}"
BOOT_DISK_TYPE="${GCP_BOOT_DISK_TYPE:-pd-balanced}"

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

# PREFLIGHT: a machine image is locked to the family it was captured from.
#
# Learned the hard way on 2026-09-05, over four failed creates and no successes.
# attnbench-l4-image-v4 was captured from a g2-standard-8 and records THREE
# properties that a different family rejects:
#
#   machineType         g2-standard-8   -- overridable (--machine-type)
#   guestAccelerators   nvidia-l4       -- replaceable, but NEVER CLEARABLE
#   disks[0].interface  NVME            -- not overridable at all
#
# The last two are the trap. `--boot-disk-interface` shapes a boot disk gcloud
# CREATES; it is ignored when the disk comes from a machine image, so an A2
# rejects the inherited NVME outright. And gcloud requires `type=` whenever
# --accelerator is given, so there is no way to say "no accelerator" -- which
# means the image can only boot on a family that accepts an L4, i.e. g2 alone.
# Those two constraints together are mutually exclusive with escaping the
# contended GPU pool, so no CPU family can be used as a workaround.
#
# The durable fix is a plain DISK image rather than a machine image -- see
# docs/machine_image_family_lock.md. This check exists so that until that is
# done, the failure arrives here in one second with an explanation, rather
# than as a sequence of unrelated-looking API rejections.
# WHICH KIND OF IMAGE IS THIS?
#
# A machine image and a plain disk image are different resources with
# different create flags, and asking the wrong API returns "not found" rather
# than "wrong kind". Until 2026-09-06 this script assumed machine image
# unconditionally: pointed at attnbench-env-v5 -- the plain disk image that
# is the DURABLE FIX for the family lock this preflight exists to catch --
# it failed closed at the READY check below with "missing or not READY",
# which is true of no machine image by that name and false of the resource.
# Failing closed cost nothing; it also could not be fixed by any flag.
#
# Detected rather than declared, because the answer is a property of the
# resource and a GCP_SOURCE_IMAGE_KIND variable would just be one more thing
# to leave stale next to GCP_SOURCE_IMAGE.
IMAGE_KIND=""
if gcloud compute machine-images describe "$SOURCE_MACHINE_IMAGE" \
     --project="$PROJECT" --format="value(status)" 2>/dev/null | grep -q READY; then
  IMAGE_KIND="machine-image"
elif gcloud compute images describe "$SOURCE_MACHINE_IMAGE" \
     --project="$PROJECT" --format="value(status)" 2>/dev/null | grep -q READY; then
  IMAGE_KIND="disk-image"
else
  echo "Source image '$SOURCE_MACHINE_IMAGE' is not a READY machine image or" >&2
  echo "a READY disk image in project '$PROJECT'." >&2
  echo "Booting this session from a bare DLVM family image instead would" >&2
  echo "spend most of the cap redoing an install, so this fails closed." >&2
  echo >&2
  echo "  machine images: gcloud compute machine-images list --project=$PROJECT" >&2
  echo "  disk images   : gcloud compute images list --project=$PROJECT --no-standard-images" >&2
  exit 1
fi

IMAGE_PROPS="$(gcloud compute machine-images describe "$SOURCE_MACHINE_IMAGE" \
    --project="$PROJECT" \
    --format="value(sourceInstanceProperties.machineType,sourceInstanceProperties.guestAccelerators[0].acceleratorType)" \
    2>/dev/null || true)"
IMAGE_MACHINE_TYPE="$(printf '%s' "$IMAGE_PROPS" | awk '{print $1}')"
IMAGE_ACCELERATOR="$(printf '%s' "$IMAGE_PROPS" | awk '{print $2}')"
IMAGE_FAMILY="${IMAGE_MACHINE_TYPE%%-*}"
TARGET_FAMILY="${MACHINE_TYPE%%-*}"

# Scoped to machine images explicitly. It would also be skipped implicitly --
# a disk image has no sourceInstanceProperties, so IMAGE_PROPS comes back
# empty -- but "this check happens not to fire" is not the same statement as
# "this check does not apply", and only the second one survives an edit.
if [[ "$IMAGE_KIND" == "machine-image" ]] \
   && [[ -n "$IMAGE_MACHINE_TYPE" ]] && [[ "$IMAGE_FAMILY" != "$TARGET_FAMILY" ]] \
   && [[ "${GCP_ALLOW_CROSS_FAMILY_IMAGE:-0}" != "1" ]]; then
  echo "REFUSING TO LAUNCH -- cross-family machine image." >&2
  echo >&2
  echo "  machine image : $SOURCE_MACHINE_IMAGE" >&2
  echo "  captured from : $IMAGE_MACHINE_TYPE (family '$IMAGE_FAMILY')" >&2
  echo "  requested     : $MACHINE_TYPE (family '$TARGET_FAMILY')" >&2
  if [[ -n "$IMAGE_ACCELERATOR" ]]; then
    echo "  image carries : guestAccelerators=$IMAGE_ACCELERATOR" >&2
  fi
  echo >&2
  echo "A machine image records the machine type, the accelerators AND the" >&2
  echo "boot disk interface of the instance it was captured from. The machine" >&2
  echo "type can be overridden; the accelerator can only be REPLACED, never" >&2
  echo "cleared; the disk interface cannot be changed at all. So this image" >&2
  echo "can only boot the family it came from." >&2
  echo >&2
  echo "Fix: capture a plain DISK image instead --" >&2
  echo "    gcloud compute images create attnbench-env-vN \\" >&2
  echo "        --source-disk=DISK --source-disk-zone=ZONE --force" >&2
  echo "A disk image carries no machine type, no accelerators and no disk" >&2
  echo "interface, so it boots any family. See docs/machine_image_family_lock.md" >&2
  echo >&2
  echo "To attempt anyway: GCP_ALLOW_CROSS_FAMILY_IMAGE=1" >&2
  exit 1
fi

# On the disk-image path, VERIFY the absence of the three locking properties
# rather than assuming it. "A disk image carries no machineType" is true of
# the resource type, but this project's rule is that the guard observes the
# artefact in front of it -- the machine-image assumption above was also true
# of every image this script had ever been pointed at, right up until it
# wasn't. If any of the three ever appears, that is a resource this script
# does not understand and it should stop, not improvise.
if [[ "$IMAGE_KIND" == "disk-image" ]]; then
  DISK_IMAGE_LOCKS="$(gcloud compute images describe "$SOURCE_MACHINE_IMAGE" \
      --project="$PROJECT" --format="json" 2>/dev/null \
      | grep -Eo '"(machineType|guestAccelerators|sourceInstanceProperties)"' \
      | sort -u || true)"
  # `|| true` is load-bearing under `set -euo pipefail`: grep exits 1 when it
  # finds nothing, which is the GOOD case here, and a bare command
  # substitution propagates that as the assignment's status. Without it this
  # script exits silently -- no message, no create -- exactly when the image
  # is clean. Caught by running the disk-image path against a fake gcloud
  # before spending an instance on it.
  if [[ -n "$DISK_IMAGE_LOCKS" ]]; then
    echo "REFUSING TO LAUNCH -- '$SOURCE_MACHINE_IMAGE' describes itself as a" >&2
    echo "disk image but carries instance-shaped properties:" >&2
    echo "$DISK_IMAGE_LOCKS" >&2
    echo "See docs/machine_image_family_lock.md." >&2
    exit 1
  fi
  # A disk image carries no accelerator, so one has to be asked for. G2
  # bundles its L4 with the machine type and also accepts the explicit form
  # (scripts/gcp_launch_l4.sh has created G2s this way against a DLVM disk
  # image); other families need it stated. Defaulted only for g2 -- an
  # unrecognised family must say what card it wants rather than inherit a
  # guess.
  if [[ -z "$ACCELERATOR" ]] && [[ "${MACHINE_TYPE%%-*}" == "g2" ]]; then
    ACCELERATOR="type=nvidia-l4,count=1"
  fi
  if [[ -z "$ACCELERATOR" ]]; then
    echo "REFUSING TO LAUNCH -- a plain disk image carries no accelerator and" >&2
    echo "none was given for machine type '$MACHINE_TYPE'." >&2
    echo "Set GCP_ACCELERATOR, e.g. type=nvidia-a100-80gb,count=1" >&2
    exit 1
  fi
fi

STARTUP_SCRIPT="$(mktemp)"
CREATE_ERR="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT" "$CREATE_ERR"' EXIT
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
# by hand: 'sudo shutdown -h HH:MM'.
#
# Those quotes are single, and that is load-bearing. This heredoc is unquoted
# so that CAP_MINUTES expands at launch time, and the same expansion performs
# command substitution anywhere in the body -- comment lines included, since a
# heredoc body is not shell source and a leading hash protects nothing.
#
# On 2026-09-05 this line quoted the recovery command in backticks instead.
# The launching Mac ran it and substituted the empty output, so the instance
# shipped with the instruction deleted. Harmless only by luck: HH:MM is not a
# valid time and sudo had no tty. Keep this body substitution-free; a test
# enforces it.
logger "attnbench-l4-compile: scheduling hard shutdown in $CAP_MINUTES minutes"
shutdown -h +$CAP_MINUTES

# ---------------------------------------------------------------------------
# Clear derived caches on EVERY boot.
#
# A machine image preserves whatever the instance it was captured from had on
# disk, including caches whose whole purpose is to make expensive work free.
# On 2026-09-03 the v3 image carried results/accuracy/score_cache from the
# session that captured it, so the next session's timed scoring pass loaded a
# 7 MB file and reported "0.007 s, 9041 TFLOPS" for a phase that genuinely
# takes 12 s. The Stage 3 estimate built on it would have been hours low.
#
# It was caught only because 9041 TFLOPS is absurd on its face. At a plausible
# magnitude -- a partially warm cache, a shorter phase -- the same failure is
# invisible. That is why this is a scripted boot step and not a line in a
# runbook: anything that can be forgotten will be, and this one does not
# announce itself when it fires.
#
# Deliberately narrow. It removes ONLY caches of derived intermediate values
# that a later run can recompute. Measurement outputs (results/*.parquet, the
# logs) are never touched -- those are the session's product, and an image
# that quietly deleted them would be a far worse failure than the one this
# prevents.
for CACHE in /home/*/attnbench_scaffold/results/accuracy/score_cache; do
  if [ -d "\$CACHE" ]; then
    logger "attnbench: clearing stale derived cache \$CACHE"
    rm -rf "\$CACHE"
  fi
done
EOF

echo "About to create a BILLABLE instance:"
echo "  project      : $PROJECT"
echo "  name         : $INSTANCE_NAME"
echo "  machine-type : $MACHINE_TYPE"
echo "  provisioning : $PROVISIONING_MODEL"
echo "  zone(s)      : $ZONE_FALLBACKS (tried in order, STOPPING at first success)"
echo "  machine-type : $MACHINE_TYPE"
echo "  accelerator  : ${ACCELERATOR:-inherited from the machine image}"
echo "  boot disk    : ${BOOT_DISK_INTERFACE:-interface inherited from the machine image}"
echo "  source image : $SOURCE_MACHINE_IMAGE ($IMAGE_KIND)"
echo "  provisioning : $PROVISIONING_MODEL"
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
  # macOS ships bash 3.2, where "${ARR[@]}" on an EMPTY array is an unbound
  # variable under `set -u` -- so the plain form aborts every launch from this
  # laptop before gcloud is ever called. The ${ARR[@]+...} guard is the 3.2-safe
  # idiom. It failed closed rather than open, but it failed on every G2 launch
  # too, not just the A2 one this flag exists for.
  ACCEL_FLAG=()
  [[ -n "$ACCELERATOR" ]] && ACCEL_FLAG=(--accelerator="$ACCELERATOR")
  [[ -n "$BOOT_DISK_INTERFACE" ]] && \
    ACCEL_FLAG+=(--boot-disk-interface="$BOOT_DISK_INTERFACE")
  # The source flags differ by kind and are not interchangeable:
  # --source-machine-image restores a whole instance shape; --image restores
  # disk contents only, so the boot disk has to be sized here.
  if [[ "$IMAGE_KIND" == "disk-image" ]]; then
    SOURCE_FLAG=(--image="$SOURCE_MACHINE_IMAGE" --image-project="$PROJECT"
                 --boot-disk-size="$BOOT_DISK_SIZE"
                 --boot-disk-type="$BOOT_DISK_TYPE")
  else
    SOURCE_FLAG=(--source-machine-image="$SOURCE_MACHINE_IMAGE")
  fi
  if gcloud compute instances create "$INSTANCE_NAME" \
      --project="$PROJECT" \
      --zone="$Z" \
      --machine-type="$MACHINE_TYPE" \
      "${SOURCE_FLAG[@]}" \
      --maintenance-policy=TERMINATE \
      --provisioning-model="$PROVISIONING_MODEL" \
      ${ACCEL_FLAG[@]+"${ACCEL_FLAG[@]}"} \
      --metadata-from-file=startup-script="$STARTUP_SCRIPT" 2>"$CREATE_ERR"; then
    CREATED_ZONE="$Z"
    break                     # <-- do not remove: see comment above
  fi
  cat "$CREATE_ERR" >&2

  # Classify the failure. A stockout and a bad request are both "create
  # returned non-zero", and treating them alike is how this script told a
  # human to "wait and attempt again later" after an NVME/A2 disk-interface
  # rejection on 2026-09-05 -- advice that would have been followed patiently,
  # forever, for an error that is deterministic and would never clear.
  #
  # A capacity failure is worth retrying and worth trying another zone.
  # A validation failure will fail identically in every zone, so iterating the
  # fallback list just prints the same rejection N times and buries it.
  if grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|resource pool exhausted|currently unavailable|no available capacity|QUOTA_EXCEEDED|Quota .* exceeded' "$CREATE_ERR"; then
    echo ">> $Z unavailable (nothing created, nothing billed)"
    continue
  fi

  echo >&2
  echo "CONFIGURATION ERROR -- this is NOT a stockout." >&2
  echo "The request was rejected as invalid, so it will fail the same way in" >&2
  echo "every zone and on every retry. Nothing was created and nothing is" >&2
  echo "billing. Fix the request, do not wait and try again." >&2
  echo >&2
  echo "Overrides that commonly need setting when the machine image was" >&2
  echo "captured on a DIFFERENT machine family than the target:" >&2
  echo "    GCP_ACCELERATOR           e.g. type=nvidia-a100-80gb,count=1" >&2
  echo "    GCP_BOOT_DISK_INTERFACE   e.g. SCSI  (A2 rejects the image's NVME)" >&2
  exit 1
done

if [[ -z "$CREATED_ZONE" ]]; then
  echo >&2
  N_ZONES="$(printf '%s\n' $ZONE_FALLBACKS | wc -w | tr -d ' ')"
  echo "No zone in '$ZONE_FALLBACKS' had capacity. Nothing was created and" >&2
  echo "nothing is billing." >&2
  if [[ "$N_ZONES" == "1" ]]; then
    # There is nothing to retry INTO. Saying "retry the list" here would be
    # advice to re-run the identical single attempt, which reads as progress
    # and is not. a2-ultragpu-1g exists in asia-southeast1-c alone, so a
    # stockout there is a hard stop, not a routing problem.
    echo >&2
    echo "This is a SINGLE-ZONE target -- there is no fallback to try." >&2
    echo "Capacity cannot be checked in advance (a successful check IS the" >&2
    echo "booking), so the only options are to wait and attempt again later," >&2
    echo "or to choose a different machine type or region." >&2
    echo "Do not sit in a retry loop against one zone." >&2
  else
    echo "Stockouts have cleared within minutes before -- retrying the same" >&2
    echo "list shortly is reasonable." >&2
  fi
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
