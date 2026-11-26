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
# on any exit of the block -- passed, failed, killed or hung up on -- schedule
# the idle halt, `shutdown -P +ATTNBENCH_P3_IDLE_HALT_MIN` (30), unless the
# planned halt comes first, and then write ~/p3/<block>.exit with the code and
# the time. So an instance whose driver has gone stops billing for the GPU within
# 30 minutes of its block's end, and keeps its disk until the hard delete.
#
# The halts (rebuilt 26 Nov 2026, after the aborted session of 25 Nov 2026). The
# old idle halt was a `sleep` started after `trap '' TERM`; it inherited the
# ignored TERM, so the next block's `kill` did nothing and it halted the
# instance in block 2. Now no halt is a process of ours: systemd keeps one
# pending shutdown, and a new `shutdown` replaces it. The planned halt is the
# launcher's in-guest halt (`shutdown -h +473` from boot; ATTNBENCH_P3_HALT_MIN,
# which the driver passes). `shutdown -c` is never used: it would cancel the
# planned halt too. Instead a block cancels the idle halt by re-scheduling the
# planned halt in its place, reads the pending shutdown back
# (/run/systemd/shutdown/scheduled), and while it runs the cancel file
# ~/p3/<block>.running keeps a watchdog checking that the pending shutdown is the
# planned one, re-arming it if not. So no idle halt can power off while a
# block runs, and nothing here cancels or delays the planned halt. The watchdog
# ends by itself when the cancel file goes; it is never killed.
#
#   bash scripts/aws_p3_instance.sh push-halt [DELETE_EPOCH]
#
# The driver runs this as each pull starts (26 Nov 2026, after P3's run): the
# pending halt goes back to now + ATTNBENCH_P3_IDLE_HALT_MIN, never earlier than
# it was, never past the planned halt and never within
# ATTNBENCH_P3_DELETE_MARGIN_MIN (10) of the hard delete. While a block runs
# (a cancel file exists) it changes nothing: that block's watchdog keeps the
# planned halt. systemd writes /run/nologin 5 min before a scheduled halt, and
# from then every new SSH login is refused, so a halt pushed back more than
# 5 min also removes a /run/nologin left by the halt it replaced. In P3's run
# the final re-pull's scp opened its connection in that window and was refused.
set -uo pipefail
BLOCK="${1:-}"; SESSION="${2:-}"
case "$BLOCK" in
  push-halt) ARGS=(); SESSION="${SESSION:--}" ;;   # SESSION is the delete time here
  b1) ARGS=(--until probe) ;;
  b2) ARGS=(--from host_components --until host_components) ;;
  b3) ARGS=(--from device_components) ;;
  *) echo "usage: aws_p3_instance.sh b1|b2|b3 SESSION | push-halt [DELETE_EPOCH]" >&2; exit 2 ;;
esac
[[ -n "$SESSION" ]] || { echo "usage: aws_p3_instance.sh b1|b2|b3 SESSION" >&2; exit 2; }
PY="${ATTNBENCH_PYTHON:-/usr/bin/python3}"
CONFIG="configs/t31/p3_session.json"
STATE="${ATTNBENCH_P3_STATE:-$HOME/p3}"
IDLE_MIN="${ATTNBENCH_P3_IDLE_HALT_MIN:-30}"
HALT_MIN="${ATTNBENCH_P3_HALT_MIN:-473}"       # the launcher's in-guest halt, from boot
# Test settings only (the halt tests and the fake remote): a minute's length in
# seconds, the boot time, the watchdog's period and where systemd's pending
# shutdown is read.
MIN_S="${ATTNBENCH_P3_MINUTE_S:-60}"
BOOT="${ATTNBENCH_P3_BOOT_EPOCH:-$(( $(date +%s) - $(cut -d. -f1 /proc/uptime) ))}"
WATCH_S="${ATTNBENCH_P3_WATCH_S:-15}"
SCHED="${ATTNBENCH_P3_SHUTDOWN_SCHEDULED:-/run/systemd/shutdown/scheduled}"
PLANNED_AT=$(( BOOT + HALT_MIN * MIN_S ))
REPO="$(pwd)"
mkdir -p "$STATE"
LOG="$STATE/$BLOCK.log"
RC=1
DONE=0

RUNNING="$STATE/$BLOCK.running"   # the cancel file: this block runs
WD=""
iso() { date -u -d "@$1" +%FT%TZ 2>/dev/null || date -u -r "$1" +%FT%TZ; }
pending_s() {   # the pending shutdown's time (epoch s), empty if none or unreadable
  local u
  u="$( { cat "$SCHED" 2>/dev/null || sudo -n cat "$SCHED" 2>/dev/null; } | sed -n 's/^USEC=//p' | head -1)"
  [[ "$u" =~ ^[0-9]+$ ]] && echo $(( u / 1000000 ))
}
planned_ok() {  # the pending shutdown is the planned halt (re-armed in whole minutes, so up to a minute early)
  local p; p="$(pending_s)"
  [[ -n "$p" ]] && (( p >= PLANNED_AT - MIN_S - 2 && p <= PLANNED_AT + 2 ))
}
arm_planned() { # schedule the planned halt; it replaces any pending shutdown
  local m=$(( (PLANNED_AT - $(date +%s)) / MIN_S ))
  (( m < 0 )) && m=0
  sudo -n shutdown -P --no-wall "+$m" >/dev/null 2>&1
}
watchdog() {    # while the cancel file exists: the pending shutdown is the planned halt
  local warned=0
  while [[ -e "$RUNNING" ]]; do
    sleep "$WATCH_S"
    [[ -e "$RUNNING" ]] || break
    planned_ok && continue
    arm_planned
    if planned_ok; then
      echo "P3_HALT_WATCHDOG $BLOCK re-armed the planned halt $(iso $PLANNED_AT) $(date -u +%FT%TZ)" >> "$LOG"
    elif (( ! warned )); then
      warned=1; echo "P3_HALT_WARN $BLOCK the pending shutdown cannot be read back as the planned halt $(date -u +%FT%TZ)" >> "$LOG"
    fi
  done
}
idle_halt() {
  local now; now="$(date +%s)"
  if (( now + IDLE_MIN * MIN_S >= PLANNED_AT )); then
    echo "P3_IDLE_HALT not set: the planned halt $(iso $PLANNED_AT) comes first" >> "$LOG"; return
  fi
  sudo -n shutdown -P --no-wall "+$IDLE_MIN" >/dev/null 2>&1
  echo "P3_IDLE_HALT +$IDLE_MIN min, pending $(iso "$(pending_s || echo 0)") $(date -u +%FT%TZ)" >> "$LOG"
}

NOLOGIN="${ATTNBENCH_P3_NOLOGIN:-/run/nologin}"
DELETE_MARGIN_MIN="${ATTNBENCH_P3_DELETE_MARGIN_MIN:-10}"
push_halt() {   # push-halt [DELETE_EPOCH]: a pull starts; 0 when the halt reads back as set
  local now bound target m p f del="${1:-}"
  now="$(date +%s)"
  for f in "$STATE"/b?.running; do
    [[ -e "$f" ]] || continue
    echo "P3_HALT_PUSH unchanged: $(basename "$f" .running) runs (its watchdog keeps the planned halt) $(iso "$now")" >> "$LOG"
    return 0
  done
  bound=$PLANNED_AT
  if [[ "$del" =~ ^[0-9]+$ ]] && (( del - DELETE_MARGIN_MIN * MIN_S < bound )); then
    bound=$(( del - DELETE_MARGIN_MIN * MIN_S ))
  fi
  target=$(( now + IDLE_MIN * MIN_S ))
  (( target > bound )) && target=$bound
  p="$(pending_s)"
  if [[ -n "$p" ]] && (( p >= target )); then
    echo "P3_HALT_PUSH unchanged: pending $(iso "$p") is not before $(iso "$target") $(iso "$now")" >> "$LOG"
    return 0
  fi
  m=$(( (target - now) / MIN_S )); (( m < 0 )) && m=0
  sudo -n shutdown -P --no-wall "+$m" >/dev/null 2>&1
  p="$(pending_s)"
  if [[ -z "$p" ]] || (( p < target - MIN_S - 2 || p > target + 2 )); then
    echo "P3_HALT_WARN push-halt: the pending shutdown does not read back as $(iso "$target") $(iso "$now")" >> "$LOG"
    return 1
  fi
  if (( p - now > 5 * MIN_S )) && [[ -e "$NOLOGIN" ]]; then
    sudo -n rm -f "$NOLOGIN" 2>/dev/null || rm -f "$NOLOGIN" 2>/dev/null
  fi
  echo "P3_HALT_PUSH pending $(iso "$p") (+$m min; bound $(iso "$bound")) $(iso "$now")" >> "$LOG"
  echo "pending $(iso "$p") (+$m min; bound $(iso "$bound"))"
}
if [[ "$BLOCK" == push-halt ]]; then
  LOG="$STATE/halt.log"
  push_halt "$([[ "$SESSION" == - ]] || echo "$SESSION")"
  exit $?
fi

finish() {
  (( DONE )) && return
  DONE=1
  trap '' TERM INT HUP
  rm -f "$RUNNING"
  [[ -n "$WD" ]] && wait "$WD" 2>/dev/null   # the watchdog ends on its own within WATCH_S
  idle_halt
  # The exit file last: the driver starts the next block only once this one's
  # idle halt is set, so a late idle halt cannot replace the next block's halt.
  printf '%s\t%s\n' "$RC" "$(date -u +%FT%TZ)" > "$STATE/$BLOCK.exit.part"
  mv "$STATE/$BLOCK.exit.part" "$STATE/$BLOCK.exit"
  echo "P3_BLOCK_EXIT $BLOCK rc=$RC $(date -u +%FT%TZ)" >> "$LOG"
}
refuse() { echo "REFUSING: $*" | tee -a "$LOG" >&2; RC=2; exit 2; }
CHILD=""
on_signal() { RC=$1; [[ -n "$CHILD" ]] && kill -TERM "$CHILD" 2>/dev/null; exit "$1"; }
trap finish EXIT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP
trap 'on_signal 130' INT

[[ -e "$STATE/$BLOCK.started" ]] && { DONE=1; echo "REFUSING: $BLOCK has already started ($STATE/$BLOCK.started)" >&2; exit 2; }
date -u +%FT%TZ > "$STATE/$BLOCK.started"
echo "P3_BLOCK_START $BLOCK $(date -u +%FT%TZ) commit $(git rev-parse HEAD)" >> "$LOG"

# The halts: this block cancels a pending idle halt by putting the planned halt
# back in its place, reads it back, and keeps it there while it runs.
touch "$RUNNING"
planned_ok || arm_planned
if planned_ok; then
  echo "P3_PLANNED_HALT pending $(iso "$(pending_s)") (boot + $HALT_MIN min: $(iso $PLANNED_AT))" >> "$LOG"
else
  echo "P3_HALT_WARN $BLOCK the pending shutdown cannot be read back as the planned halt ($(iso $PLANNED_AT)); the hard delete stays the bound" >> "$LOG"
fi
watchdog >/dev/null 2>&1 < /dev/null &
WD=$!

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
