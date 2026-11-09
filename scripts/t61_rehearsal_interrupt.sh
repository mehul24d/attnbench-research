#!/usr/bin/env bash
# The rehearsal's simulated interruption, from the Mac
# (docs/launch_cards/t61_rehearsal_l4.md, instance B): wait until the b2
# session to be interrupted has opened its ground truth, reset the instance
# (a hard restart, as a preemption gives: no clean exit), wait for SSH, then
# start the boot2 command under the exit wrapper. Gives up, resetting
# nothing, if ground truth has not begun within --within minutes of the
# start: the interruption is then reported as not rehearsed, and the boot1
# session runs on and halts by itself.
#
# Usage:
#   bash scripts/t61_rehearsal_interrupt.sh NAME ZONE A_INSTANCE B1 X Y [--within 20]
set -uo pipefail

NAME="${1:?}"; ZONE="${2:?}"; A="${3:?}"; B1="${4:?}"; X="${5:?}"; Y="${6:?}"
WITHIN=20
[[ "${7:-}" == "--within" ]] && WITHIN="${8:?}"
: "${ATTNBENCH_GCP_PROJECT:?set the project of this study}"
GC=(--zone="$ZONE" --project="$ATTNBENCH_GCP_PROJECT")
say() { echo "=== $(date -u +%FT%TZ) $*"; }
on() { gcloud compute ssh "$NAME" "${GC[@]}" --command "$1" 2>/dev/null; }

deadline=$(( $(date +%s) + WITHIN * 60 ))
until on "test -e ~/attnbench_scaffold/$X/12_ground_truth/step.log"; do
  if (( $(date +%s) > deadline )); then
    say "INTERRUPT_NOT_REHEARSED: no ground truth in $X within $WITHIN min; nothing reset"
    exit 3
  fi
  sleep 20
done
say "ground truth opened in $X; resetting $NAME"
gcloud compute instances reset "$NAME" "${GC[@]}" || { say "reset failed"; exit 1; }
for _ in $(seq 1 30); do on true && break; sleep 10; done
on true || { say "no SSH after the reset"; exit 1; }
say "boot 2: $(on 'cat /proc/sys/kernel/random/boot_id')"
on "cd ~/attnbench_scaffold && nohup bash scripts/session_exit.sh --log ~/reh_b2.log -- \
    bash scripts/t61_rehearsal_b.sh boot2 $A $B1 $X $Y >/dev/null 2>&1 &"
say "INTERRUPT_DONE: boot2 started"
