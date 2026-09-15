#!/usr/bin/env bash
# GATE: prove the instance can WRITE to and READ BACK from the results bucket,
# before any phase depends on that path.
#
# This exists because on 2026-09-15 a 4h47m H100 session (Rs 2,029) produced
# zero recoverable results: the instance was created without
# --scopes=cloud-platform, so it held only `devstorage.read_only` and every
# `gsutil cp` to the bucket would have returned 403. The sync was never
# exercised before the session depended on it, and the instance's self-delete
# took the only copy. An assumed write path is not a write path.
#
# Exits non-zero on any failure. A non-zero exit here means TEAR DOWN -- do not
# start a phase, do not "try it again later in the run".
set -euo pipefail

INSTANCE="${1:?usage: gcp_verify_gcs_writable.sh INSTANCE ZONE [BUCKET]}"
ZONE="${2:?usage: gcp_verify_gcs_writable.sh INSTANCE ZONE [BUCKET]}"
BUCKET="${3:-gs://attnbench-results-research-507316}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

TOKEN="writecheck-$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
echo "GCS write gate: $BUCKET  (token $TOKEN)"

REMOTE=$(cat <<EOF
set -euo pipefail
echo "\$(hostname) \$(date -u +%FT%TZ) $TOKEN" > /tmp/$TOKEN.txt
gsutil -q cp /tmp/$TOKEN.txt $BUCKET/_writecheck/$TOKEN.txt
gsutil -q cp $BUCKET/_writecheck/$TOKEN.txt /tmp/$TOKEN.readback.txt
cmp /tmp/$TOKEN.txt /tmp/$TOKEN.readback.txt
echo "SCOPES:"; curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes
echo "WRITE_GATE_PASS \$(cat /tmp/$TOKEN.readback.txt)"
EOF
)

if ! OUT=$(gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
      --command="$REMOTE" 2>&1); then
  echo "$OUT"
  echo "WRITE GATE FAILED (ssh/remote non-zero). TEAR DOWN." >&2
  exit 1
fi
echo "$OUT"

if ! grep -q "WRITE_GATE_PASS" <<<"$OUT"; then
  echo "WRITE GATE FAILED (no pass marker). TEAR DOWN." >&2
  exit 1
fi
if ! grep -q "devstorage.full_control\|cloud-platform" <<<"$OUT"; then
  echo "WRITE GATE FAILED (scopes lack write access). TEAR DOWN." >&2
  exit 1
fi

# Independently confirm from the client side that the object is really there.
gcloud storage ls "$BUCKET/_writecheck/$TOKEN.txt" --project="$PROJECT"
echo "WRITE GATE PASSED -- round trip verified from the guest AND listed from the client."
