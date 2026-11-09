#!/usr/bin/env bash
# Bootstrap a fresh instance for accuracy work: the pinned RULER data, the
# pinned wonderwords and nltk, and the XAttention install. Run from the Mac,
# from the repository root, after gcp_deploy_source.sh, the preflight and the
# GCS write gate:
#
#   bash scripts/gcp_bootstrap_instance.sh INSTANCE ZONE [--no-xattention] [--suite-record]
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
#      place the versions are written, and lightgbm at
#      predict.learned.LIGHTGBM_VERSION (the learned predictor's step needs it
#      and image v6 has none), into the user site; lightgbm must then import
#      at that version;
#   5. verify on the instance: every SHA-256 pin, the word pool (wonderwords
#      at its pinned version) and the sentence tokenizer (nltk and punkt_tab);
#   6. scripts/install_xattention.sh through run_phase.sh, so its log is
#      synced (skipped with --no-xattention);
#   7. the FA3 wheel, when FA3_WHEEL_SHA256 is set to the hash recorded from
#      its build: scripts/fa3_import_check.sh fetches it, checks hash, stack
#      and tag, installs it and imports it (an H100 only). Unset, the step is
#      skipped and says so: the H100's dense set is then incomplete;
#   8. with --suite-record, T2.26's record before anything is measured:
#      scripts/gpu_suite_record.sh on the instance (the suite on its GPU,
#      synced by run_phase.sh), bounded by SUITE_RECORD_MIN (default 25)
#      minutes; a suite that does not pass, or reaches the bound, stops here;
#   9. check the guest tree is still clean.
#
# Every remote step prints a marker, and the step passes only if the marker
# comes back: a silent ssh is a failure, not a success.
set -euo pipefail

USAGE="usage: gcp_bootstrap_instance.sh INSTANCE ZONE [--no-xattention] [--suite-record]"
NAME="${1:?$USAGE}"
ZONE="${2:?$USAGE}"
XATTN=1
SUITE=0
for opt in "${@:3}"; do
  case "$opt" in
    --no-xattention) XATTN=0 ;;
    --suite-record) SUITE=1 ;;
    *) echo "REFUSING: unknown option $opt ($USAGE). Nothing was run." >&2; exit 2 ;;
  esac
done
SUITE_MIN="${SUITE_RECORD_MIN:-25}"
if [[ ! "$SUITE_MIN" =~ ^[1-9][0-9]*$ ]]; then
  echo "REFUSING: SUITE_RECORD_MIN is not a whole number of minutes. Nothing was run." >&2
  exit 1
fi
REMOTE_DIR="${REMOTE_DIR:-~/attnbench_scaffold}"
PY="${ATTNBENCH_PYTHON:-python3}"

FA3_SHA="${FA3_WHEEL_SHA256:-}"
if [[ -n "$FA3_SHA" && ! "$FA3_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "REFUSING: FA3_WHEEL_SHA256 is not a sha256 (64 lowercase hex). Nothing was run." >&2
  exit 1
fi

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

echo "== 1/8 local RULER data against the SHA-256 pins"
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

echo "== 2/8 RULER data on the instance"
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

LGBM="$("$PY" -c 'from attnbench.predict.learned import LIGHTGBM_VERSION as v; print(v)')"
echo "== 3/8 pinned packages: $PINS lightgbm==$LGBM"
remote PKGS_OK "python3 -m pip install --user -q $PINS lightgbm==$LGBM && python3 -c 'import lightgbm; assert lightgbm.__version__ == \"$LGBM\", lightgbm.__version__' && echo PKGS_OK" \
  || { echo "FAILED: installing $PINS lightgbm==$LGBM, or lightgbm does not import at that version" >&2; exit 1; }

echo "== 4/8 word pool and sentence tokenizer on the instance"
remote RULER_OK "cd $REMOTE_DIR && python3 -c 'from attnbench.accuracy import ruler_data as r; r.word_pool(); r.sent_tokenizer()' && echo RULER_OK" \
  || { echo "FAILED: the word pool or the sentence tokenizer does not load" >&2; exit 1; }

if [[ "$XATTN" == 1 ]]; then
  echo "== 5/8 XAttention install, through run_phase.sh"
  remote XATTN_OK "cd $REMOTE_DIR && bash scripts/run_phase.sh bootstrap_xattention -- bash scripts/install_xattention.sh > ~/bootstrap_xattention.log 2>&1 && echo XATTN_OK; tail -5 ~/bootstrap_xattention.log" \
    || { echo "FAILED: install_xattention.sh (log: ~/bootstrap_xattention.log on the instance)" >&2; exit 1; }
else
  echo "== 5/8 XAttention install skipped (--no-xattention)"
fi

if [[ -n "$FA3_SHA" ]]; then
  echo "== 6/8 FA3 wheel: fetch, check, install, import (scripts/fa3_import_check.sh)"
  remote FA3_OK "cd $REMOTE_DIR && bash scripts/fa3_import_check.sh $FA3_SHA > ~/bootstrap_fa3.log 2>&1 && echo FA3_OK; tail -5 ~/bootstrap_fa3.log" \
    || { echo "FAILED: the FA3 import check (log: ~/bootstrap_fa3.log on the instance)" >&2; exit 1; }
else
  echo "== 6/8 FA3 skipped: FA3_WHEEL_SHA256 is not set. On an H100 the dense set is then"
  echo "   incomplete (dense_set_incomplete=\"fa3:no wheel\"); elsewhere FA3 does not apply."
fi

if [[ "$SUITE" == 1 ]]; then
  echo "== 7/8 T2.26: the suite record on this GPU (scripts/gpu_suite_record.sh, at most ${SUITE_MIN} min)"
  remote SUITE_RC=0 "cd $REMOTE_DIR && timeout ${SUITE_MIN}m bash scripts/gpu_suite_record.sh > ~/bootstrap_suite.log 2>&1; echo SUITE_RC=\$?; tail -5 ~/bootstrap_suite.log" \
    || { echo "FAILED: the suite record did not pass (rc 124 is the ${SUITE_MIN} min bound; log: ~/bootstrap_suite.log on the instance, synced as phase s9_suite)" >&2; exit 1; }
else
  echo "== 7/8 suite record skipped (no --suite-record)"
fi

echo "== 8/8 guest tree still clean"
remote TREE_CLEAN "cd $REMOTE_DIR && [ -z \"\$(git status --porcelain)\" ] && echo TREE_CLEAN" \
  || { echo "FAILED: the guest tree is dirty after the bootstrap" >&2; exit 1; }

echo "BOOTSTRAP_OK $NAME"
