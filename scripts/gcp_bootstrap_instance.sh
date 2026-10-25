#!/usr/bin/env bash
# Bootstrap a fresh instance for accuracy work: the pinned RULER data, the
# pinned wonderwords and nltk, and the XAttention install. Run from the Mac,
# from the repository root, after gcp_deploy_source.sh, the preflight and the
# GCS write gate:
#
#   bash scripts/gcp_bootstrap_instance.sh INSTANCE ZONE [--no-xattention]
#
# Why a script. T1.1's two sessions (22 and 25 Oct 2026) did these steps by
# hand, because image v6 carries neither the RULER data nor wonderwords and
# nltk. Each step was the same, and each was a place to skip the pin check.
#
# In order, stopping at the first failure:
#   1. pin the project (scripts/_gcp_project.sh);
#   2. verify the LOCAL RULER data against ruler_data.PINNED_SHA256 before
#      copying anything, so a bad local copy is never shipped;
#   3. copy it to ~/.cache/attnbench/ruler_data on the instance, unless the
#      copy there already verifies; a copy there that fails is moved aside,
#      not deleted;
#   4. install wonderwords and nltk at ruler_data.PINNED_VERSIONS, the one
#      place the versions are written, into the user site;
#   5. verify on the instance: every SHA-256 pin, the word pool (wonderwords
#      at its pinned version) and the sentence tokenizer (nltk and punkt_tab);
#   6. scripts/install_xattention.sh through run_phase.sh, so its log is
#      synced (skipped with --no-xattention);
#   7. check the guest tree is still clean.
#
# Every remote step prints a marker, and the step passes only if the marker
# comes back: a silent ssh is a failure, not a success.
set -euo pipefail

NAME="${1:?usage: gcp_bootstrap_instance.sh INSTANCE ZONE [--no-xattention]}"
ZONE="${2:?usage: gcp_bootstrap_instance.sh INSTANCE ZONE [--no-xattention]}"
XATTN=1
[[ "${3:-}" == "--no-xattention" ]] && XATTN=0
REMOTE_DIR="${REMOTE_DIR:-~/attnbench_scaffold}"
PY="${ATTNBENCH_PYTHON:-python3}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

source "$(dirname "${BASH_SOURCE[0]}")/_gcp_project.sh"
gcp_project_pin

remote() {   # remote MARKER COMMAND: run on the instance, require MARKER back
  local marker="$1" cmd="$2" out
  out="$(gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
           --command="$cmd" 2>&1)" || true
  printf '%s\n' "$out" | sed 's/^/   | /'
  grep -q "^$marker\$" <<<"$out"
}

echo "== 1/6 local RULER data against the SHA-256 pins"
LOCAL_DIR="$("$PY" - <<'PY'
from attnbench.accuracy import ruler_data as r
for name in r.PINNED_SHA256:
    r._verified(name)
print(r.data_dir())
PY
)" || { echo "REFUSING: the local RULER data does not verify against the pins;" >&2
         echo "nothing was copied. Fetch it with scripts/fetch_ruler_data.py." >&2
         exit 1; }
N_PINS="$("$PY" -c 'from attnbench.accuracy import ruler_data as r; print(len(r.PINNED_SHA256))')"
PINS="$("$PY" -c 'from attnbench.accuracy import ruler_data as r; print(" ".join(f"{k}=={v}" for k, v in sorted(r.PINNED_VERSIONS.items())))')"
echo "   $N_PINS pinned files verified locally; packages: $PINS"

CHECK_PINS="cd $REMOTE_DIR && python3 -c 'from attnbench.accuracy import ruler_data as r; [r._verified(n) for n in r.PINNED_SHA256]' && echo PINS_OK"

echo "== 2/6 RULER data on the instance"
if remote PINS_OK "$CHECK_PINS"; then
  echo "   already present and verified; not copied"
else
  remote AREA_READY 'mkdir -p ~/.cache/attnbench && D=~/.cache/attnbench/ruler_data && { [ ! -e "$D" ] || mv "$D" "$D.failed.$(date -u +%Y%m%dT%H%M%SZ)"; } && echo AREA_READY' \
    || { echo "FAILED: could not prepare ~/.cache/attnbench on the instance" >&2; exit 1; }
  gcloud compute scp --recurse "$LOCAL_DIR" "$NAME:~/.cache/attnbench/ruler_data" \
    --zone="$ZONE" --project="$PROJECT" >/dev/null \
    || { echo "FAILED: copying the RULER data" >&2; exit 1; }
  remote PINS_OK "$CHECK_PINS" \
    || { echo "FAILED: the copied RULER data does not verify on the instance" >&2; exit 1; }
  echo "   copied and verified"
fi

echo "== 3/6 pinned packages: $PINS"
remote PKGS_OK "python3 -m pip install --user -q $PINS && echo PKGS_OK" \
  || { echo "FAILED: installing $PINS" >&2; exit 1; }

echo "== 4/6 word pool and sentence tokenizer on the instance"
remote RULER_OK "cd $REMOTE_DIR && python3 -c 'from attnbench.accuracy import ruler_data as r; r.word_pool(); r.sent_tokenizer()' && echo RULER_OK" \
  || { echo "FAILED: the word pool or the sentence tokenizer does not load" >&2; exit 1; }

if [[ "$XATTN" == 1 ]]; then
  echo "== 5/6 XAttention install, through run_phase.sh"
  remote XATTN_OK "cd $REMOTE_DIR && bash scripts/run_phase.sh bootstrap_xattention -- bash scripts/install_xattention.sh > ~/bootstrap_xattention.log 2>&1 && echo XATTN_OK; tail -5 ~/bootstrap_xattention.log" \
    || { echo "FAILED: install_xattention.sh (log: ~/bootstrap_xattention.log on the instance)" >&2; exit 1; }
else
  echo "== 5/6 XAttention install skipped (--no-xattention)"
fi

echo "== 6/6 guest tree still clean"
remote TREE_CLEAN "cd $REMOTE_DIR && [ -z \"\$(git status --porcelain)\" ] && echo TREE_CLEAN" \
  || { echo "FAILED: the guest tree is dirty after the bootstrap" >&2; exit 1; }

echo "BOOTSTRAP_OK $NAME"
