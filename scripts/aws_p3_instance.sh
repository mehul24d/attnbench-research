#!/usr/bin/env bash
# One block of P3 on the instance (AWS g6.2xlarge; docs/launch_cards/aws_p3_g6.md).
# The Mac's driver (scripts/aws_p3_driver.sh) starts each block detached, pulls
# and checks its steps when it ends, and only then starts the next:
#
#   bash scripts/aws_p3_instance.sh b1|b2|b3 SESSION
#
#   b1  preflight, link, dense_select, probe      (--until probe)
#   b2  host_components: the P6 gate's input      (--from host_components --until host_components)
#   b3  device_components to delay_injection      (--from device_components)
#
# T3.1's runner (scripts/t31_session.py) on configs/t31/p3_session.json, with no
# bucket sync (--no-sync): results leave the instance only by the Mac's pulls.
# The prediction (step 7) and the leg-3 plan (step 12) go to the Mac through
# scripts/aws_p3_gsutil.py, put first on PATH as `gsutil` (ATTNBENCH_BUCKET
# gs://p3-mac), so the pinned step code makes its usual calls.
#
# The exit path (the counterpart of scripts/session_exit.sh, which syncs to GCS
# and halts at once, so it cannot end a block that the Mac still has to pull):
# on any exit of the block -- passed, failed, killed or hung up on -- write
# ~/p3/<block>.exit with the code and the time, then schedule a halt of the
# instance ATTNBENCH_P3_IDLE_HALT_MIN (30) minutes later. The Mac's next block
# cancels it; the Mac's teardown makes it moot. So an instance whose driver has
# gone stops billing for the GPU within 30 minutes of its block's end, and keeps
# its disk until the hard delete. The launcher's in-guest halt and hard delete
# stay the outer bounds.
set -uo pipefail
BLOCK="${1:-}"; SESSION="${2:-}"
case "$BLOCK" in
  b1) ARGS=(--until probe) ;;
  b2) ARGS=(--from host_components --until host_components) ;;
  b3) ARGS=(--from device_components) ;;
  *) echo "usage: aws_p3_instance.sh b1|b2|b3 SESSION" >&2; exit 2 ;;
esac
[[ -n "$SESSION" ]] || { echo "usage: aws_p3_instance.sh b1|b2|b3 SESSION" >&2; exit 2; }
PY="${ATTNBENCH_PYTHON:-/usr/bin/python3}"
CONFIG="configs/t31/p3_session.json"
STATE="${ATTNBENCH_P3_STATE:-$HOME/p3}"
IDLE_MIN="${ATTNBENCH_P3_IDLE_HALT_MIN:-30}"
IDLE_S="${ATTNBENCH_P3_IDLE_HALT_S:-$(( IDLE_MIN * 60 ))}"   # seconds: the fake remote's tests only
REPO="$(pwd)"
mkdir -p "$STATE"
LOG="$STATE/$BLOCK.log"
RC=1
DONE=0

finish() {
  (( DONE )) && return
  DONE=1
  trap '' TERM INT HUP
  printf '%s\t%s\n' "$RC" "$(date -u +%FT%TZ)" > "$STATE/$BLOCK.exit.part"
  mv "$STATE/$BLOCK.exit.part" "$STATE/$BLOCK.exit"
  echo "P3_BLOCK_EXIT $BLOCK rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
  # The idle halt: cancelled by the next block, moot after the teardown.
  nohup bash -c "sleep $IDLE_S; logger 'attnbench P3: $BLOCK ended rc=$RC, no next block in $IDLE_S s; halting' 2>/dev/null; sudo -n shutdown -h now" \
    >/dev/null 2>&1 < /dev/null &
  echo $! > "$STATE/idle_halt.pid"
}
refuse() { echo "REFUSING: $*" | tee -a "$LOG" >&2; RC=2; exit 2; }
CHILD=""
on_signal() { RC=$1; [[ -n "$CHILD" ]] && kill -TERM "$CHILD" 2>/dev/null; exit "$1"; }
trap finish EXIT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP
trap 'on_signal 130' INT

# A pending idle halt is cancelled first: this block is the next one.
if [[ -f "$STATE/idle_halt.pid" ]]; then
  kill "$(cat "$STATE/idle_halt.pid")" 2>/dev/null || true
  rm -f "$STATE/idle_halt.pid"
fi
[[ -e "$STATE/$BLOCK.started" ]] && { DONE=1; echo "REFUSING: $BLOCK has already started ($STATE/$BLOCK.started)" >&2; exit 2; }
date -u +%FT%TZ > "$STATE/$BLOCK.started"
echo "P3_BLOCK_START $BLOCK $(date -u +%FT%TZ) commit $(git rev-parse HEAD)" >> "$LOG"

[[ -z "$(git status --porcelain)" ]] || refuse "the tree is not clean"
[[ "$("$PY" -c 'import platform; print(platform.python_version())')" == "3.10.12" ]] \
  || refuse "$PY is not Python 3.10.12"
case "$BLOCK" in
  b1) [[ -e "$SESSION" ]] && refuse "$SESSION exists; block 1 starts a new session" ;;
  b2) [[ "$(cut -f1 "$STATE/b1.exit" 2>/dev/null)" == 0 ]] || refuse "block 1 did not end with 0" ;;
  b3) [[ "$(cut -f1 "$STATE/b2.exit" 2>/dev/null)" == 0 ]] || refuse "block 2 did not end with 0" ;;
esac

# The prediction store: the Mac, through aws_p3_gsutil.py.
mkdir -p "$STATE/bin" "$HOME/p3_store"
printf '#!/usr/bin/env bash\nexec "%s" "%s/scripts/aws_p3_gsutil.py" "$@"\n' "$PY" "$REPO" > "$STATE/bin/gsutil"
chmod +x "$STATE/bin/gsutil"
export PATH="$STATE/bin:$PATH" ATTNBENCH_BUCKET="gs://p3-mac" ATTNBENCH_P3_STORE="$HOME/p3_store"
export ATTNBENCH_EXPECTED_HOST="aws:g6.2xlarge" ATTNBENCH_PYTHON="$PY" PYTHONDONTWRITEBYTECODE=1
# The exit path above stands in for session_exit.sh's (see the header).
export ATTNBENCH_SESSION_EXIT=armed

"$PY" scripts/t31_session.py --config "$CONFIG" --session "$SESSION" --no-sync "${ARGS[@]}" >> "$LOG" 2>&1 &
CHILD=$!
wait "$CHILD"
RC=$?
exit "$RC"
