#!/usr/bin/env bash
# The rehearsal's watcher, on the Mac (docs/launch_cards/t61_rehearsal_l4.md):
# a poll loop that runs until the instance has halted itself (TERMINATED) or
# is gone (the hard delete), printing each status change and the minutes
# since the watch began. It changes nothing on the instance: the self-halt
# (scripts/session_exit.sh, then the in-guest halt, then the hard delete) is
# what ends the session; this only says when it has.
#
# Usage:
#   bash scripts/t61_rehearsal_watch.sh NAME ZONE [--every 60] > ~/reh_watch_NAME.log 2>&1 &
set -uo pipefail

NAME="${1:?}"; ZONE="${2:?}"
EVERY=60
[[ "${3:-}" == "--every" ]] && EVERY="${4:?}"
: "${ATTNBENCH_GCP_PROJECT:?set the project of this study}"
say() { echo "=== $(date -u +%FT%TZ) $*"; }

t0=$(date +%s); last=""
while true; do
  st="$(gcloud compute instances describe "$NAME" --zone="$ZONE" \
          --project="$ATTNBENCH_GCP_PROJECT" --format='value(status)' 2>/dev/null)" || st=""
  [[ -n "$st" ]] || st=GONE
  if [[ "$st" != "$last" ]]; then
    say "$NAME $st at +$(( ($(date +%s) - t0) / 60 )) min"
    last="$st"
  fi
  case "$st" in
    TERMINATED|GONE) say "WATCH_DONE: $NAME is $st"; exit 0 ;;
  esac
  sleep "$EVERY"
done
