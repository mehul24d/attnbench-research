#!/usr/bin/env bash
# Refuse to start a session on an instance whose results/ tree is not empty.
#
# Why this exists
# ---------------
# The machine image attnbench-env-v5-20260905 ships a populated results/ tree
# left over from whichever session baked it. On 2026-09-16 that tree was found
# to contain `results/probe/versions.json` stamped:
#
#     gpu_name: NVIDIA L4
#
# Stage 0 resumes on (backend, config_key). Run it on an H100 against that
# tree and every config the L4 already probed is skipped, and the H100's
# probe.parquet silently inherits L4 capability rows under an H100 provenance
# stamp. That is the b6ed63b failure again: a stamp that is wrong in the
# plausible direction, self-contradicting only to a reader who already knows
# what to look for.
#
# The specific shape is contamination that survives teardown by living inside
# an artifact nobody thinks of as holding state. Deleting it from THIS image
# fixes this image. Refusing to start on ANY image that has the problem is
# what catches the next one, so that is what this does.
#
# Usage:
#   bash scripts/gcp_preflight_instance.sh INSTANCE ZONE            # refuse
#   bash scripts/gcp_preflight_instance.sh INSTANCE ZONE --quarantine
set -euo pipefail

NAME="${1:?usage: gcp_preflight_instance.sh INSTANCE ZONE [--quarantine]}"
ZONE="${2:?zone required}"
MODE="${3:-refuse}"
REMOTE_DIR="${REMOTE_DIR:-~/attnbench_scaffold}"

OUT="$(gcloud compute ssh "$NAME" --zone="$ZONE" --command="
set -euo pipefail
cd $REMOTE_DIR
if [ ! -d results ] || [ -z \"\$(ls -A results 2>/dev/null)\" ]; then
  echo PREFLIGHT_RESULTS_EMPTY
  exit 0
fi
echo PREFLIGHT_RESULTS_DIRTY
echo '--- what is in it ---'
find results -maxdepth 2 -type f | head -20
for v in \$(find results -name versions.json | head -5); do
  echo \"--- \$v\"
  python3 -c \"import json,sys;d=json.load(open('\$v'));print('    gpu_name:',d.get('gpu_name'));print('    git_commit:',d.get('git_commit'))\"
done
" 2>&1)"

echo "$OUT"

if grep -q PREFLIGHT_RESULTS_EMPTY <<<"$OUT"; then
  echo "PREFLIGHT PASSED -- results/ is empty; nothing to inherit."
  exit 0
fi

if [[ "$MODE" == "--quarantine" ]]; then
  STAMP="results_preexisting_$(date -u +%Y%m%dT%H%M%SZ)"
  gcloud compute ssh "$NAME" --zone="$ZONE" --command="
    cd $REMOTE_DIR && mv results '$STAMP' && mkdir -p results && echo QUARANTINED_TO=$STAMP"
  echo "PREFLIGHT: pre-existing results/ quarantined as $STAMP. Safe to start."
  exit 0
fi

cat >&2 <<'MSG'

PREFLIGHT FAILED -- this instance has a non-empty results/ before any phase ran.

Stage 0 and the sweep both resume on (backend, config_key). Starting here
would skip every config the PREVIOUS machine already probed and stamp its rows
with THIS machine's provenance. The resulting table is not malformed and no
downstream check detects it.

Re-run with --quarantine to move it aside, or fix the image.
MSG
exit 1
