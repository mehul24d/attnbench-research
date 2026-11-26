#!/usr/bin/env bash
# Bootstrap an AWS instance booted from the environment image for accuracy
# work (adviser's rulings of 23 Nov 2026: a repository script with a test,
# before the P3 launch). The AWS counterpart of gcp_bootstrap_instance.sh,
# steps 1 to 4 and 8, over plain ssh/scp: the image, like v6, carries neither
# the RULER data nor wonderwords and nltk, nor lightgbm. Run from the Mac, from
# the repository root, after the deploy:
#
#   AWS_SSH_KEY=~/.ssh/key bash scripts/aws_bootstrap_instance.sh USER@HOST
#
# In order, stopping at the first failure:
#   1. verify the LOCAL RULER data against ruler_data.PINNED_SHA256 before
#      copying anything, so a bad local copy is never shipped;
#   2. copy it to ~/.cache/attnbench/ruler_data on the instance, unless the
#      copy there already verifies; a copy there that fails is moved aside,
#      not deleted;
#   3. install wonderwords and nltk at ruler_data.PINNED_VERSIONS and lightgbm
#      at predict.learned.LIGHTGBM_VERSION into the user site of the image's
#      /usr/bin/python3; lightgbm must then import at that version;
#   4. on the instance: every SHA-256 pin, the word pool and the sentence
#      tokenizer;
#   5. check the guest tree is still clean.
# Every remote step prints a marker, and the step passes only if the marker
# comes back: a silent ssh is a failure, not a success.
set -euo pipefail

HOST="${1:?usage: aws_bootstrap_instance.sh USER@HOST}"
KEY="${AWS_SSH_KEY:?set AWS_SSH_KEY to the private key of the session}"
REMOTE_DIR="${REMOTE_DIR:-~/attnbench_scaffold}"
RPY="${REMOTE_PYTHON:-/usr/bin/python3}"
PY="${ATTNBENCH_PYTHON:-python3}"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30
          -o ServerAliveCountMax=4)
[[ -n "${AWS_KNOWN_HOSTS:-}" ]] && SSH_OPTS+=(-o "UserKnownHostsFile=$AWS_KNOWN_HOSTS")

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

remote() {   # remote MARKER COMMAND: run on the instance, require MARKER back
  local marker="$1" cmd="$2" out
  out="$(ssh "${SSH_OPTS[@]}" "$HOST" "$cmd" 2>&1)" || true
  printf '%s\n' "$out" | sed 's/^/   | /'
  grep -q "^$marker\$" <<<"$out"
}

echo "== 1/5 local RULER data against the SHA-256 pins"
LOCAL_DIR="$("$PY" - <<'PY'
from attnbench.accuracy import ruler_data as r
for name in r.PINNED_SHA256:
    r._verified(name)
print(r.data_dir())
PY
)" || { echo "REFUSING: the local RULER data does not verify against the pins;" >&2
         echo "nothing was copied. Fetch it with scripts/fetch_ruler_data.py." >&2
         exit 1; }
PINS="$("$PY" -c 'from attnbench.accuracy import ruler_data as r; print(" ".join(f"{k}=={v}" for k, v in sorted(r.PINNED_VERSIONS.items())))')"
LGBM="$("$PY" -c 'from attnbench.predict.learned import LIGHTGBM_VERSION as v; print(v)')"
echo "   verified locally; packages: $PINS lightgbm==$LGBM"

CHECK_PINS="cd $REMOTE_DIR && $RPY -c 'from attnbench.accuracy import ruler_data as r; [r._verified(n) for n in r.PINNED_SHA256]' && echo PINS_OK"

echo "== 2/5 RULER data on the instance"
if remote PINS_OK "$CHECK_PINS"; then
  echo "   already present and verified; not copied"
else
  remote AREA_READY 'mkdir -p ~/.cache/attnbench && D=~/.cache/attnbench/ruler_data && { [ ! -e "$D" ] || mv "$D" "$D.failed.$(date -u +%Y%m%dT%H%M%SZ)"; } && echo AREA_READY' \
    || { echo "FAILED: could not prepare ~/.cache/attnbench on the instance" >&2; exit 1; }
  scp "${SSH_OPTS[@]}" -r -q "$LOCAL_DIR" "$HOST:~/.cache/attnbench/ruler_data" \
    || { echo "FAILED: copying the RULER data" >&2; exit 1; }
  remote PINS_OK "$CHECK_PINS" \
    || { echo "FAILED: the copied RULER data does not verify on the instance" >&2; exit 1; }
  echo "   copied and verified"
fi

echo "== 3/5 pinned packages: $PINS lightgbm==$LGBM"
remote PKGS_OK "$RPY -m pip install --user -q $PINS lightgbm==$LGBM && $RPY -c 'import lightgbm; assert lightgbm.__version__ == \"$LGBM\", lightgbm.__version__' && echo PKGS_OK" \
  || { echo "FAILED: installing $PINS lightgbm==$LGBM, or lightgbm does not import at that version" >&2; exit 1; }

echo "== 4/5 word pool and sentence tokenizer on the instance"
remote RULER_OK "cd $REMOTE_DIR && $RPY -c 'from attnbench.accuracy import ruler_data as r; r.word_pool(); r.sent_tokenizer()' && echo RULER_OK" \
  || { echo "FAILED: the word pool or the sentence tokenizer does not load" >&2; exit 1; }

echo "== 5/5 guest tree still clean"
remote TREE_CLEAN "cd $REMOTE_DIR && [ -z \"\$(git status --porcelain)\" ] && echo TREE_CLEAN" \
  || { echo "FAILED: the guest tree is dirty after the bootstrap" >&2; exit 1; }

echo "BOOTSTRAP_OK"
