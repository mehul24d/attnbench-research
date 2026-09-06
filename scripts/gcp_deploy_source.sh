#!/usr/bin/env bash
# Deploy this repo's HEAD to a rented instance, and REFUSE unless the instance
# ends up describing exactly that commit.
#
# Why this exists
# ---------------
# On 2026-09-04 a 4576-row result set -- a complete Stage 0 and the first
# complete Stage 1 in the project -- was stamped `git_commit=b6ed63b`, a
# commit 18 behind the code that produced it. The deploy had been a tarball
# untarred over the instance's existing checkout: it replaced `attnbench/`,
# `scripts/`, `tests/` and `docs/`, and left `.git` describing the machine
# image. `provenance.capture()` then faithfully reported the image's commit.
#
# The stamp was wrong in the worst available way: plausible. b6ed63b is a real
# commit in this repo's history, so nothing looked malformed. It was
# self-contradicting only to a reader who noticed the table contained 84
# `illegal_memory_access` rows and that the code writing that status did not
# exist at b6ed63b.
#
# Copying source WITHOUT copying the history that identifies it is the root
# cause, so this script does not copy source. There is no git remote for this
# project, so `git pull` is not available; a bundle is the same thing offline
# -- real objects, real refs, a real commit on the other side.
#
# Usage:
#   bash scripts/gcp_deploy_source.sh INSTANCE ZONE [REMOTE_DIR]
set -euo pipefail

NAME="${1:?usage: gcp_deploy_source.sh INSTANCE ZONE [REMOTE_DIR]}"
ZONE="${2:?zone required}"
REMOTE_DIR="${3:-~/attnbench_scaffold}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# 1. Refuse to deploy something that is not a commit.
#
# Shipping a dirty tree would reproduce the original bug from the other end:
# the instance would hold code that no commit describes, and would stamp rows
# with a commit that does not match them. `results/` and `*.parquet` are
# gitignored, so measurement output never trips this.
DIRT="$(git status --porcelain)"
if [[ -n "$DIRT" ]]; then
  echo "REFUSING TO DEPLOY -- working tree is dirty:" >&2
  echo "$DIRT" >&2
  echo >&2
  echo "Every result row records the commit it was produced at. Deploying" >&2
  echo "uncommitted changes makes that field a lie in a way no downstream" >&2
  echo "check can detect. Commit (or stash) first." >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
echo "deploying $COMMIT to $NAME:$REMOTE_DIR"

BUNDLE="$(mktemp -t attnbench-deploy).bundle"
trap 'rm -f "$BUNDLE"' EXIT
# Full history reachable from HEAD, so the bundle is self-contained and does
# not depend on what the instance's repo happens to already have.
git bundle create "$BUNDLE" HEAD >/dev/null 2>&1

gcloud compute scp "$BUNDLE" "$NAME:/tmp/deploy.bundle" --zone="$ZONE" >/dev/null

# 2. Check the bundle out on the instance. `-f` and `clean -fd` because the
#    instance may be carrying a previous tarball deploy's edits; `clean` is
#    deliberately WITHOUT -x, so gitignored measurement output survives.
#
#    The detach is required on the SECOND deploy of a session and every one
#    after it. git refuses to fetch into the branch a non-bare repo currently
#    has checked out -- `--force` does not override it:
#
#      fatal: Refusing to fetch into current branch refs/heads/deployed
#             of non-bare repository
#
#    The first deploy always works (HEAD is on the image's own branch), so
#    this was invisible until a session deployed twice. It would have fired
#    at Stage 3 segment 2 regardless: the segmentation plan pins a commit per
#    segment and re-deploys each time, and a mid-session fix is exactly the
#    hazard `check_code_continuity` exists to catch.
gcloud compute ssh "$NAME" --zone="$ZONE" --command="
set -euo pipefail
cd $REMOTE_DIR
git checkout --quiet --detach
git fetch --quiet /tmp/deploy.bundle HEAD:refs/heads/deployed --force
git checkout --quiet --force deployed
git clean -qfd
rm -f /tmp/deploy.bundle
"

# 3. Verify from the instance's own git, not from what we believe we sent.
#
# This is the check that would have caught the original failure, and it is
# deliberately a SEPARATE ssh round trip: asking the machine that will write
# the provenance what it thinks its commit is, after the deploy claims to have
# finished, is the only question whose answer matters.
REMOTE_STATE="$(gcloud compute ssh "$NAME" --zone="$ZONE" --command="
cd $REMOTE_DIR
echo \"commit=\$(git rev-parse HEAD)\"
echo \"dirty=\$(git status --porcelain | wc -l | tr -d ' ')\"
" 2>/dev/null)"

REMOTE_COMMIT="$(sed -n 's/^commit=//p' <<< "$REMOTE_STATE")"
REMOTE_DIRTY="$(sed -n 's/^dirty=//p' <<< "$REMOTE_STATE")"

if [[ "$REMOTE_COMMIT" != "$COMMIT" ]]; then
  echo "DEPLOY FAILED -- instance is at ${REMOTE_COMMIT:-<unknown>}," >&2
  echo "expected $COMMIT. Do not run anything: every row it writes would" >&2
  echo "carry a commit that does not describe the code." >&2
  exit 1
fi
if [[ "$REMOTE_DIRTY" != "0" ]]; then
  echo "DEPLOY FAILED -- instance tree is dirty ($REMOTE_DIRTY file(s))." >&2
  echo "The commit above would not describe what actually runs." >&2
  exit 1
fi

echo "verified: $NAME is at $COMMIT with a clean tree"
