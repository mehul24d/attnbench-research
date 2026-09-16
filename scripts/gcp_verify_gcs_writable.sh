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
# The gate found a SECOND, independent cause on 2026-09-16, on an instance
# whose scopes were correct:
#
#   AccessDeniedException: 403 425125335855-compute@developer.gserviceaccount.com
#   does not have storage.objects.list access to the bucket
#
# Writing to a bucket from a GCE instance needs BOTH layers, and they fail
# identically with a 403:
#   1. the instance's OAuth scope set must include write access
#      (--scopes=cloud-platform), and
#   2. the instance's SERVICE ACCOUNT must hold an IAM role on the bucket
#      (roles/storage.objectAdmin).
# Fixing (1) after the 2026-09-15 loss did not fix (2), and nothing in the
# launch configuration reveals (2) -- `instances describe` happily shows
# cloud-platform scopes on an instance that cannot write a single object.
# Only an actual round trip distinguishes them, which is why this gate does a
# round trip rather than an inspection.
#
# Exits non-zero on any failure. A non-zero exit here means TEAR DOWN -- do not
# start a phase, do not "try it again later in the run".
#
# WITH ONE EXCEPTION, added 2026-09-16 after this gate reported
# "WRITE GATE FAILED (ssh/remote non-zero). TEAR DOWN." for an ssh exit 255 on
# an instance that had reached RUNNING ten seconds earlier and was completely
# healthy. sshd simply was not accepting connections yet.
#
# The gate exists to separate exactly one condition from everything else --
# "this instance cannot write to GCS" -- and it was collapsing three outcomes
# into one verdict:
#
#   transport not ready yet  -> retryable, costs seconds
#   transport broken         -> investigate, do not assume either way
#   GCS write refused        -> TEAR DOWN, this is the Rs 2,029 condition
#
# Both directions of that conflation are expensive. Reading a not-ready
# transport as a refused write destroys a healthy instance and re-pays the
# boot; reading a refused write as a not-ready transport retries past the
# exact condition the gate was built for. So the gate now WAITS for the
# transport to answer before testing the write path, and distinguishes the two
# in its exit code: 2 means the transport never came up (the caller decides),
# any other non-zero still means TEAR DOWN.
#
# The guard was not broken. It was imprecise about which failure it had seen,
# and its verdict was right for one cause and wrong for another -- which is a
# different defect from the ones in silent_failure_patterns.md #36, where the
# checks did not run at all.
set -euo pipefail

INSTANCE="${1:?usage: gcp_verify_gcs_writable.sh INSTANCE ZONE [BUCKET]}"
ZONE="${2:?usage: gcp_verify_gcs_writable.sh INSTANCE ZONE [BUCKET]}"
BUCKET="${3:-gs://attnbench-results-research-507316}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

SSH_READY_TIMEOUT="${SSH_READY_TIMEOUT:-300}"

# Precondition: the transport must answer before the write path is meaningful.
# Without this the gate's verdict conflates "not up yet" with "cannot write".
echo "waiting for sshd on $INSTANCE (up to ${SSH_READY_TIMEOUT}s)..."
_t0=$(date +%s)
until gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
        --command=true >/dev/null 2>&1; do
  if [ $(( $(date +%s) - _t0 )) -gt "$SSH_READY_TIMEOUT" ]; then
    echo "TRANSPORT NEVER CAME UP in ${SSH_READY_TIMEOUT}s -- this is NOT a" >&2
    echo "verdict on the write path, which was never tested. Exit 2." >&2
    exit 2
  fi
  sleep 10
done
echo "sshd answered after $(( $(date +%s) - _t0 ))s"

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
