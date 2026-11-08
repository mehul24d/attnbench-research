#!/usr/bin/env bash
# Run a whole session on a rented instance so that its LAST act, on any exit
# -- passed, failed, refused, killed, or hung up on -- is a sync of its logs
# and manifests to the bucket and then a halt of the VM. No watcher on the Mac
# is needed for either (researcher's item 4, 8 Nov 2026).
#
# Why this exists
# ---------------
# T3.2b (8 Nov 2026) finished at 23:43Z and then billed about 80 idle minutes
# (about Rs 142): the teardown waited on a watcher on the Mac, the watcher had
# expired, and the instance ran on to its in-guest halt at +247 min and its
# hard delete. The delete took ~/t32b.log with the disk. Every step had
# synced (run_phase.sh), the session's own log had not.
#
# What it does
# ------------
#   - runs the command in the background, its output appended to --log (a
#     ~/*.log, so the teardown's step 2b also finds it), and waits for it;
#   - exports ATTNBENCH_SESSION_EXIT=armed, which a real P2 or P4 session
#     (scripts/t61_session.py) requires before its first step;
#   - on EXIT, and on TERM, INT and HUP (each first stopped in the command),
#     once: writes "SESSION_EXIT=<rc> <UTC>" to the log, copies every ~/*.log
#     into logs/home/ and an exit marker into logs/session_exit.json (logs/ is
#     gitignored), and syncs results/ and logs/ with `run_phase.sh
#     --sync-only` (its verified rsync, the one sync path), up to
#     ATTNBENCH_EXIT_SYNC_TRIES times; after a sync that worked it rewrites the
#     marker with "synced": true and syncs once more, so the bucket's marker
#     says so;
#   - then halts: `sudo -n shutdown -h now`, whether the sync worked or not.
#     A halted VM stops billing for the GPU and keeps its disk until the hard
#     delete, so a failed sync leaves the disk to recover; a running one would
#     only bill. The outcome also goes to the serial console (`logger`), which
#     the teardown saves first.
#
# The teardown (gcp_teardown_session.sh) pulls the logs and manifests from
# the bucket when the instance is halted or gone, and reads the marker.
#
# Usage (on the instance, from ~/attnbench_scaffold, under nohup):
#   nohup bash scripts/session_exit.sh --log ~/p2.log -- \
#       python3 scripts/t61_session.py run --banked-at P1=<time> >/dev/null 2>&1 &
set -uo pipefail

LOG=""
if [[ "${1:-}" == "--log" ]]; then LOG="${2:?--log needs a file}"; shift 2; fi
[[ -n "$LOG" ]] || { echo "usage: session_exit.sh --log FILE -- CMD..." >&2; exit 2; }
[[ "${1:-}" == "--" ]] && shift
[[ $# -gt 0 ]] || { echo "session_exit.sh: no command given" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIES="${ATTNBENCH_EXIT_SYNC_TRIES:-3}"
WAIT_S="${ATTNBENCH_EXIT_RETRY_WAIT_S:-20}"
STOP_GRACE_S="${ATTNBENCH_EXIT_STOP_GRACE_S:-30}"
RC=0
CHILD=""
FINISHED=0

stamp() { date -u +%FT%TZ; }

marker() {   # marker SYNCED TRY
  mkdir -p logs
  printf '{"rc": %s, "exited_at": "%s", "synced": %s, "sync_try": %s, "tries": %s, "log": "%s", "host": "%s"}\n' \
    "$RC" "$EXITED_AT" "$1" "$2" "$TRIES" "$(basename "$LOG")" "$(hostname)" > logs/session_exit.json
}

home_logs() {
  mkdir -p logs/home
  for f in "$HOME"/*.log; do [[ -f "$f" ]] && cp -p "$f" logs/home/; done
  return 0
}

stop_child() {
  [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null || return 0
  kill -TERM "$CHILD" 2>/dev/null
  local i=0
  while kill -0 "$CHILD" 2>/dev/null && (( i < STOP_GRACE_S )); do sleep 1; i=$((i + 1)); done
  kill -KILL "$CHILD" 2>/dev/null || true
}

finish() {
  (( FINISHED )) && return
  FINISHED=1
  trap '' TERM INT HUP                 # the exit path itself is not interrupted
  stop_child
  EXITED_AT="$(stamp)"
  echo "SESSION_EXIT=$RC $EXITED_AT" >> "$LOG"
  local ok=0 try
  for (( try = 1; try <= TRIES; try++ )); do
    home_logs
    marker false "$try"
    if bash "$HERE/run_phase.sh" --sync-only session_exit >> "$LOG" 2>&1; then
      ok=1
      echo "--- session_exit: synced on try $try $(stamp)" >> "$LOG"
      marker true "$try"
      home_logs
      bash "$HERE/run_phase.sh" --sync-only session_exit >/dev/null 2>&1 || true
      break
    fi
    echo "--- session_exit: SYNC FAILED on try $try of $TRIES $(stamp)" >> "$LOG"
    (( try < TRIES )) && sleep "$WAIT_S"
  done
  logger "attnbench session_exit: rc=$RC synced=$ok; halting" 2>/dev/null || true
  echo "--- session_exit: halting (rc=$RC, synced=$ok) $(stamp)" >> "$LOG"
  sudo -n shutdown -h now
}

on_signal() { RC=$1; exit "$1"; }
trap 'finish' EXIT
trap 'on_signal 143' TERM
trap 'on_signal 130' INT
trap 'on_signal 129' HUP

export ATTNBENCH_SESSION_EXIT=armed
echo "SESSION_START $(stamp) cmd: $*" >> "$LOG"
"$@" >> "$LOG" 2>&1 &
CHILD=$!
wait "$CHILD"
RC=$?
CHILD=""
exit "$RC"
