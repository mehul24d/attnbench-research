#!/usr/bin/env bash
# Boot a captured image on a CHEAP CPU-only instance and prove it is usable.
#
# gcp_capture_image.sh ends by saying "boot it once before trusting it. A READY
# image is not a booting image." This is that step, and it exists as a script
# because the alternative is discovering the problem at the start of a rented
# GPU session -- which is exactly how the 2026-09-16 v6 attempt was lost: a
# dirty tree was baked into an artifact, run_phase.sh would have refused to
# start on it, and the cause was frozen inside a disk image and invisible to
# git log.
#
# CPU-only on purpose. Everything worth checking here -- does it boot, does
# sshd answer, is the repo clean, does the venv import, did the expensive
# builds survive -- is answerable without an accelerator, at roughly Rs 6/h
# against Rs 284/h. `import torch` works fine with no GPU present; only
# torch.cuda.is_available() is expected to be False, and that is asserted
# rather than treated as a failure.
#
# Hard-capped and self-deleting: a verification instance that outlives its
# verification is the same failure it is meant to prevent.
set -euo pipefail

IMAGE="${1:?usage: gcp_boot_test_image.sh IMAGE [ZONE] [MACHINE_TYPE]}"
ZONE="${2:-asia-southeast1-c}"
MACHINE="${3:-e2-medium}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
NAME="boottest-${IMAGE}"
NAME="${NAME:0:62}"

cleanup() {
  echo "== deleting $NAME =="
  gcloud compute instances delete "$NAME" --zone="$ZONE" --project="$PROJECT" \
    --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== booting $IMAGE on $MACHINE (CPU-only) in $ZONE =="
gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --image="$IMAGE" --image-project="$PROJECT" \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --max-run-duration=20m --instance-termination-action=DELETE \
  --no-restart-on-failure \
  --format='value(name,status)'
# No --maintenance-policy=TERMINATE: e2 rejects it unless the instance is
# preemptible, and it buys nothing here. --max-run-duration=20m with
# action=DELETE is the guardrail, and the EXIT trap deletes on every path
# including a failed check.

echo "== waiting for sshd =="
READY=0
for i in $(seq 1 30); do
  if gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" \
       --tunnel-through-iap --command='true' >/dev/null 2>&1; then
    READY=1; echo "  sshd answered after ~$((i*10))s"; break
  fi
  sleep 10
done
if [[ "$READY" -ne 1 ]]; then
  echo "BOOT TEST FAILED: sshd never answered. The image is NOT usable." >&2
  exit 2
fi

echo "== checks =="
# Shipped as a file and executed, NOT nested as a heredoc inside
# `ssh --command='...'`. The nested form produced EMPTY output on the first
# attempt and the caller reported success anyway -- empty output read as
# success is this project's most repeated failure, and it appeared here in
# the script written to prevent a different instance of it.
gcloud compute scp "$(dirname "$0")/_image_boot_checks.py" \
  "$NAME:/tmp/_image_boot_checks.py" --zone="$ZONE" --project="$PROJECT" \
  --tunnel-through-iap >/dev/null 2>&1 \
  || { echo "BOOT TEST FAILED: could not copy checks to the guest" >&2; exit 4; }

OUT="$(gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" \
        --tunnel-through-iap --command='python3 /tmp/_image_boot_checks.py' 2>&1 || true)"
echo "$OUT" | grep '^CHECK' | sed 's/^/  /'

N_CHECK="$(echo "$OUT" | grep -c '^CHECK' || true)"
if [[ "${N_CHECK:-0}" -lt 7 ]]; then
  echo "BOOT TEST FAILED: expected 7 CHECK lines, got ${N_CHECK:-0}." >&2
  echo "  Silence is not a pass. Raw output follows:" >&2
  echo "$OUT" | sed 's/^/    /' >&2
  exit 5
fi
if echo "$OUT" | grep -q '^CHECK .* FAIL'; then
  echo; echo "BOOT TEST FAILED -- do not use $IMAGE" >&2; exit 3
fi
echo; echo "BOOT TEST PASSED -- $IMAGE boots and is usable"
