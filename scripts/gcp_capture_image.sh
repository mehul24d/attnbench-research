#!/usr/bin/env bash
# Capture a boot-disk image from a RUNNING instance, with the preconditions
# that the 2026-09-16 v6 attempt lacked.
#
# What went wrong that day, in order:
#
#  1. The instance was "cleaned" with `rm -rf results`. results/ is gitignored
#     but TRACKED FILES live under it -- two on the day this happened
#     (results/stage3_s1/INVALID_ROWS.md, results/stage3_s1b/README.md), nine
#     from 2026-09-17 once the S7/S8 evidence artifacts were force-added, and
#     `git ls-files results/` whenever you are reading this -- so the clean
#     silently deleted them.
#     Second occurrence: the preflight's `mv results <stamp>` did the same
#     thing earlier and stamped git_dirty=True on 492 Stage 1 passes.
#     gitignored and untracked are NOT the same set, and this repo is a
#     standing counterexample.
#  2. The image was captured in that state, so a dirty tree was baked into an
#     artifact. run_phase.sh refuses to run on a dirty tree, so every future
#     session from that image would refuse to start -- with the cause frozen
#     inside a disk image and invisible to git log. A defect that reaches an
#     artifact outlives the session that made it.
#  3. The re-capture failed: `gcloud compute images create --force` means
#     "capture from a RUNNING disk", NOT "overwrite an existing image". The
#     bad image stood, and its source disk went with the instance.
#
# So: assert clean BEFORE capture, refuse an existing name, and verify the
# image is READY before the caller is told it may tear down.
set -euo pipefail

INSTANCE="${1:?usage: gcp_capture_image.sh INSTANCE ZONE IMAGE_NAME [REPO_DIR]}"
ZONE="${2:?usage: gcp_capture_image.sh INSTANCE ZONE IMAGE_NAME [REPO_DIR]}"
IMAGE="${3:?usage: gcp_capture_image.sh INSTANCE ZONE IMAGE_NAME [REPO_DIR]}"
REPO="${4:-attnbench_scaffold}"
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

echo "== precondition 1: image name must not already exist =="
if gcloud compute images describe "$IMAGE" --project="$PROJECT" >/dev/null 2>&1; then
  echo "REFUSING: image '$IMAGE' already exists." >&2
  echo "  --force does NOT overwrite; it means 'capture from a running disk'." >&2
  echo "  Pick a new name, or delete the existing image deliberately." >&2
  exit 3
fi
echo "  ok, '$IMAGE' is free"

echo "== precondition 2: the guest repo must be CLEAN =="
# Single-quoted heredoc: this must evaluate on the guest, not here.
DIRT="$(gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
  --command="cd \$HOME/$REPO && git status --porcelain" 2>/dev/null || echo "__SSH_FAILED__")"
if [ "$DIRT" = "__SSH_FAILED__" ]; then
  echo "REFUSING: could not read git state on the guest." >&2
  exit 4
fi
if [ -n "$DIRT" ]; then
  echo "REFUSING: guest repo is dirty -- this would bake it into the image:" >&2
  echo "$DIRT" | sed 's/^/    /' >&2
  echo "  If a clean step deleted tracked files under a gitignored path," >&2
  echo "  restore them with: git checkout -- <path>" >&2
  exit 5
fi
echo "  ok, git status --porcelain is empty"

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command=sync >/dev/null 2>&1 || true

echo "== capturing $IMAGE =="
gcloud compute images create "$IMAGE" \
  --source-disk="$INSTANCE" --source-disk-zone="$ZONE" --project="$PROJECT" \
  --storage-location=asia --force \
  --description="${IMAGE_DESCRIPTION:-attnbench env, captured with a verified-clean repo}"

echo "== postcondition: image must be READY =="
ST="$(gcloud compute images describe "$IMAGE" --project="$PROJECT" --format='value(status)')"
echo "  status: $ST"
[ "$ST" = "READY" ] || { echo "IMAGE NOT READY -- do NOT tear down the source." >&2; exit 6; }
echo "IMAGE_CAPTURED=$IMAGE"
echo "NOTE: boot it once before trusting it. A READY image is not a booting image."
