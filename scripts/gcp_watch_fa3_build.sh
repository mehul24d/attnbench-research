#!/usr/bin/env bash
# Watch the FA3 build VM from the Mac until the build has ended, then say
# what the bucket holds. Read-only: it describes the instance, reads the build
# output over ssh and lists the bucket; it never creates, stops or deletes.
#
# Usage:  bash scripts/gcp_watch_fa3_build.sh INSTANCE ZONE
#
# Why it exists (28 Oct 2026): the first build failed 12 minutes in, and the
# watcher then in use called `timeout`, which macOS does not have. Every check
# failed, the failure was not reported, and the VM sat idle for two hours. So:
#   - no `timeout`: each cloud call runs in the background and is killed
#     (with its children) after FA3_WATCH_CALL_LIMIT_S, in plain bash 3.2;
#   - a check that fails is reported at once, and FA3_WATCH_MAX_FAILURES in a
#     row stop the watch with exit 3: a watcher that cannot see is an alarm,
#     not a quiet loop;
#   - the end is read from the instance (it halts itself when the build ends,
#     scripts/build_fa3_wheel.sh) and the verdict from the bucket.
#
# Exit codes: 0 the wheel and its manifest are in the bucket; 2 the VM
# stopped and a new failed-<UTC>/ log directory is there; 3 too many failed
# checks in a row (look by hand now); 4 the watch ran out of polls with the VM
# still running; 5 the VM stopped and the bucket shows neither outcome.
set -euo pipefail

NAME="${1:?usage: gcp_watch_fa3_build.sh INSTANCE ZONE}"
ZONE="${2:?usage: gcp_watch_fa3_build.sh INSTANCE ZONE}"
INTERVAL_S="${FA3_WATCH_INTERVAL_S:-120}"
CALL_LIMIT_S="${FA3_WATCH_CALL_LIMIT_S:-90}"
MAX_FAILURES="${FA3_WATCH_MAX_FAILURES:-3}"
MAX_POLLS="${FA3_WATCH_MAX_POLLS:-100}"   # 100 x 2 min covers the 3 h cap

source "$(dirname "${BASH_SOURCE[0]}")/_gcp_project.sh"
gcp_project_pin
source "$(dirname "${BASH_SOURCE[0]}")/_fa3_pin.sh"
PREFIX="gs://attnbench-results-${PROJECT}/fa3/${FA3_COMMIT}/"

# limited CMD...: run CMD, killing it and its children after CALL_LIMIT_S.
# Prints CMD's output (both streams); returns its status, or 124 when killed.
limited() {
  local out pid waited=0 rc=0
  out="$(mktemp)"
  "$@" >"$out" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= CALL_LIMIT_S )); then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      gcp_redact <"$out"; rm -f "$out"
      return 124
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "$pid" || rc=$?
  gcp_redact <"$out"; rm -f "$out"
  return "$rc"
}

now() { date -u +%H:%M:%SZ; }
failures=0
failed_check() {   # $1: what failed, $2: its status, $3: what it printed (redacted)
  failures=$((failures + 1))
  echo "$(now) CHECK FAILED ($failures/$MAX_FAILURES): $1 (status $2)" >&2
  [[ -n "${3:-}" ]] && tail -n 3 <<<"$3" | sed 's/^/   | /' >&2
  if (( failures >= MAX_FAILURES )); then
    echo "STOPPING: $MAX_FAILURES checks in a row failed, so this watch cannot see the VM." >&2
    echo "Look by hand now: gcloud compute instances describe $NAME --zone=$ZONE" >&2
    exit 3
  fi
}

# The verdict belongs to this build: the build writes started.json (its UTC
# start) before compiling, a failure's logs go to failed-<UTC>/ and a
# verified wheel's manifest.json carries the same start. So an earlier
# build's failed-<UTC>/ (older than the start) or manifest (another start)
# is not this build's outcome, whenever this watch was begun.
verdict() {
  local rc=0 listing started manifest stamp new
  listing="$(limited command gcloud storage ls "$PREFIX" --project="$PROJECT")" || rc=$?
  (( rc == 0 )) || { failed_check "storage ls" "$rc" "$listing"; return 1; }
  started="$(limited command gcloud storage cat "${PREFIX}started.json" --project="$PROJECT" \
             | sed -n 's/.*"started": *"\([0-9TZ:-]*\)".*/\1/p')" || rc=$?
  if (( rc != 0 )) || [[ -z "$started" ]]; then
    echo "VERDICT: unknown; this build's started.json could not be read, so it never reached its build step." >&2
    exit 5
  fi
  if grep -q '/manifest\.json$' <<<"$listing"; then
    manifest="$(limited command gcloud storage cat "${PREFIX}manifest.json" --project="$PROJECT" || true)"
    if grep -q "\"started_utc\": \"$started\"" <<<"$manifest"; then
      echo "VERDICT: wheel stored and read back (manifest.json for the build started $started)."
      exit 0
    fi
  fi
  stamp="$(tr -d ':-' <<<"$started")"          # 2026-10-28T13:54:23Z -> 20261028T135423Z
  new="$(grep -o 'failed-[0-9]\{8\}T[0-9]\{6\}Z/$' <<<"$listing" \
         | awk -v s="failed-$stamp" '$0 >= s' || true)"
  if [[ -n "$new" ]]; then
    echo "VERDICT: the build started $started failed; its logs: ${PREFIX}$(tail -n 1 <<<"$new")"
    exit 2
  fi
  echo "VERDICT: unknown; the VM stopped and the bucket shows neither a wheel nor logs from the build started $started." >&2
  exit 5
}

poll=0
while (( poll < MAX_POLLS )); do
  poll=$((poll + 1))
  rc=0
  STATUS="$(limited command gcloud compute instances describe "$NAME" --zone="$ZONE" --project="$PROJECT" --format='value(status)')" || rc=$?
  if (( rc != 0 )); then
    failed_check "instances describe" "$rc" "$STATUS"
    sleep "$INTERVAL_S"; continue
  fi
  STATUS="$(tail -n 1 <<<"$STATUS" | tr -d '[:space:]')"   # a warning may precede it
  case "$STATUS" in
    RUNNING)
      rc=0
      OUT="$(limited command gcloud compute ssh "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet \
               --command='grep -o "\[[0-9]*/[0-9]*\]" ~/fa3_build/out/build.log 2>/dev/null | tail -1; tail -n 2 ~/fa3_build.out 2>/dev/null; echo WATCH_OK')" || rc=$?
      if (( rc != 0 )) || ! grep -q '^WATCH_OK$' <<<"$OUT"; then
        failed_check "ssh to read the build output (no WATCH_OK back)" "$rc" "$OUT"
      else
        failures=0
        echo "$(now) RUNNING: $(grep -v '^WATCH_OK$' <<<"$OUT" | tr '\n' ' ')"
      fi ;;
    TERMINATED|STOPPED|STOPPING|SUSPENDED|SUSPENDING)
      echo "$(now) the VM is $STATUS: the build has ended."
      verdict || { sleep "$INTERVAL_S"; continue; } ;;
    *)
      failures=0
      echo "$(now) $STATUS" ;;
  esac
  sleep "$INTERVAL_S"
done
echo "STOPPING: $MAX_POLLS polls and the VM is still running; look by hand." >&2
exit 4
