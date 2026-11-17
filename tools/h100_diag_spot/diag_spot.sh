#!/bin/bash
# EXPLORATORY: the H100 diagnostic session on Spot (adviser's rulings, 17 Nov 2026, item 2c;
# the user's launch yes the same day). From $REPO, a clean clone at the committed diag branch
# (the deploy bundles its HEAD), on the Mac:
#
#   refuse   unless $APPROVED names the cell list's sha256 and $REPO's cells file and card
#            carry the same one; unless $REPO is clean; unless no session has been granted
#            before ($W/granted: one session, ever)
#   money    spent so far ($DIAG_SPENT) + the cap (Rs 1,300) <= the ceiling (Rs 15,500), and
#            the hard delete at the rate fits inside the cap, else STOP (exit 5)
#   window   requests only inside $DIAG_WINDOW_UTC (06-14); outside it the driver waits for
#            its start; a request about to go out outside it, or in its last $DIAG_MARGIN s,
#            is refused; no request possible before $DIAG_UNTIL (the end of the first window
#            the driver sees) -> NOT GRANTED (exit 4)
#   acquire  Spot a3-highgpu-1g in us-central1-a only, image v6 (the launcher's pin), halt
#            110 billed min, delete at 120; no capacity -> again after $DIAG_SPACING s, inside
#            the window; a grant elsewhere or not RUNNING is deleted and the driver stops
#   chain    P4's spot_chain.sh, unchanged: ssh, deploy, preflight --quarantine, GCS write
#            gate, bootstrap (FA3 from the wheel's sha256)
#   start    scripts/h100_diag.py run under session_exit.sh (it syncs and halts the VM on any
#            exit)
#   watch    every 60 s until the VM is halted or gone; past the hard delete + 5 min still
#            RUNNING -> deleted from here
#   teardown scripts/gcp_teardown_session.sh; the spend booked on billed minutes: the
#            teardown's session_cost.txt, or Compute Engine's create and preemption times on
#            a reclaim, else the delete bound (and the log says which)
#   reclaim  the first reclaim stops the driver (exit 7); nothing reruns
#   single   one lock for every copy of this driver ($DIAG_LOCK), taken atomically before the
#            first request; any attnbench-h100-diag-* VM listed refuses (17 Nov 2026, item 2)
#   override --window-override (the researcher's yes of 17 Nov 2026): attempt 1 goes out now,
#            outside the window; refused unless $APPROVED carries OVERRIDE_LINE and POLICY_LINE
#            below. A reclaim of attempt 1 is booked, then attempt 2 goes out inside the window
#            only if the diag spend so far + Rs $PROJ fits the cap; its delete is cut so that the
#            two together stay inside the cap; a second reclaim stops (exit 7). Without the
#            option the first reclaim stops, as before
#   rerun    --rerun (the researcher's yes of 17 Nov 2026, the rerun wave): the rerun's cell list
#            and card, cap Rs 525, halt 40 billed min, delete at 50, one attempt, the first reclaim
#            stops; $APPROVED must also carry RERUN_TERMS below; its own $DIAG_DIR (live_rerun);
#            refused with --window-override
#
# Every external command is overridable (CMD_*), so the whole path runs with a mocked gcloud
# and clock on the Mac (test/run_mock.sh). Exit: 0 session ran and torn down; 2 refused;
# 4 not granted; 5 money; 7 reclaimed; 1 any other failure (look before anything else).
set -uo pipefail
SP=$(cd "$(dirname "$0")" && pwd)
OVERRIDE=0 RERUN=0
for a in "$@"; do case "$a" in --window-override) OVERRIDE=1 ;; --rerun) RERUN=1 ;; *) echo "diag_spot.sh: unknown option $a" >&2; exit 2 ;; esac; done
(( OVERRIDE && RERUN )) && { echo "diag_spot.sh: --rerun takes no --window-override" >&2; exit 2; }
: "${REPO:?REPO: the clean clone at the diag commit}" "${APPROVED:?APPROVED: the approval file}"
: "${DIAG_SPENT:?DIAG_SPENT: Rs spent so far}" "${FA3_WHEEL_SHA256:?}"
W=${DIAG_DIR:-$SP/$( (( RERUN )) && echo live_rerun || echo live)}; mkdir -p "$W"
LOG=$W/diag.log
G=${CMD_GCLOUD:-gcloud}
SLEEP=${CMD_SLEEP:-sleep}
NOW=${CMD_NOW:-date +%s}
LAUNCH=${CMD_LAUNCH:-bash scripts/gcp_launch_h100.sh}
CHAIN=${CMD_CHAIN:-bash $SP/spot_chain.sh}
START=${CMD_START:-}
TEARDOWN=${CMD_TEARDOWN:-bash scripts/gcp_teardown_session.sh}
P=${ATTNBENCH_GCP_PROJECT:?}
ZONE=us-central1-a RATE=629.94 HALT=110 DEL=120 CAP=1300 CEILING=15500 PROJ=504
WINDOW=${DIAG_WINDOW_UTC:-6-14} MARGIN=${DIAG_MARGIN:-300} SPACING=${DIAG_SPACING:-600}
CELLS=configs/exploratory/h100_diag_cells.json CARD=docs/exploratory/h100_diag_card.md
RERUN_TERMS="terms: Spot us-central1-a, image v6, cap Rs 525, halt 40 billed min, delete at 50, one attempt, requests only inside 06:00-14:00Z, stop on the first reclaim"
if (( RERUN )); then
  HALT=40 DEL=50 CAP=525
  CELLS=configs/exploratory/h100_diag_rerun_cells.json CARD=docs/exploratory/h100_diag_rerun_card.md
fi
OVERRIDE_LINE="window_override: attempt 1 now, outside 06:00-14:00Z; attempt 2 only inside it"
POLICY_LINE="attempt_policy: attempt 2 only after a reclaim of attempt 1, inside the window, if the diag spend + Rs 504 fits the Rs 1300 cap; stop on the second reclaim"
say() { echo "$(date -u -r "$($NOW)" +%FT%TZ) $*" | sed "s/$P/<project>/g" >> "$LOG"; }
stop() { say "$2"; exit "$1"; }
# seconds until the window opens; 0 inside it with at least M s left (P4's spot_blocks3.sh)
wait_s() { python3 -I -c 'import sys; t, ws, we, m = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); h = t % 86400 / 3600; print(0 if ws <= h and (t + m) % 86400 / 3600 < we and h < we else int((ws * 3600 - t % 86400) % 86400) or 1)' \
  "$($NOW)" "${WINDOW%-*}" "${WINDOW#*-}" "${1:-0}"; }
status() { $G compute instances describe "$1" --zone "$2" --project "$P" --format='value(status)' 2>/dev/null; }
# attempt 1 under the override goes out whatever the clock; anything else waits for the window
free() { (( OVERRIDE && GRANTS == 0 )); }
no_diag_vm() {
  local vms; vms=$($G compute instances list --project "$P" --filter="name~^attnbench-h100-diag-" --format='value(name)' 2>/dev/null) \
    || stop 2 "REFUSED: could not list instances to look for a diag VM"
  [[ -z "$vms" ]] || stop "${1:-2}" "REFUSED: a diag VM exists ($(echo $vms)); one session only"
}
cd "$REPO" || exit 2

# --- refuse
SHA=$(shasum -a 256 "$CELLS" | cut -d' ' -f1)
grep -qx "cells_sha256: $SHA" "$APPROVED" || stop 2 "REFUSED: $APPROVED does not approve $CELLS ($SHA)"
grep -q "sha256 \`$SHA\`" "$CARD" || stop 2 "REFUSED: $CARD does not name $SHA"
[[ -z "$(git status --porcelain)" ]] || stop 2 "REFUSED: $REPO is not clean; the deploy bundles HEAD"
[[ -e "$W/granted" ]] && stop 2 "REFUSED: a session was granted before ($(cat "$W/granted")); one session only"
if (( RERUN )); then
  grep -qxF "$RERUN_TERMS" "$APPROVED" || stop 2 "REFUSED: --rerun, but $APPROVED does not name the rerun's terms (\"$RERUN_TERMS\")"
fi
if (( OVERRIDE )); then
  grep -qxF "$OVERRIDE_LINE" "$APPROVED" || stop 2 "REFUSED: --window-override, but $APPROVED does not name the override (\"$OVERRIDE_LINE\")"
  grep -qxF "$POLICY_LINE" "$APPROVED" || stop 2 "REFUSED: --window-override, but $APPROVED does not name the attempt policy (\"$POLICY_LINE\")"
fi
# --- single instance
LOCK=${DIAG_LOCK:-$HOME/.attnbench/h100_diag.lock}; mkdir -p "$(dirname "$LOCK")"
held() {
  local pid; pid=$(sed -n 's/^pid=\([0-9]*\).*/\1/p' "$LOCK" 2>/dev/null | head -1)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "REFUSED: driver pid $pid holds $LOCK and is live ($(head -1 "$LOCK")); one session only"
  elif grep -q '^finished' "$LOCK" 2>/dev/null; then
    echo "REFUSED: stale lock from a finished run ($(head -1 "$LOCK"); $(grep '^finished' "$LOCK" | tail -1)); one session only, nothing reruns; remove $LOCK only under a new launch yes"
  else
    echo "REFUSED: stale lock: pid ${pid:-?} is not running and recorded no finish ($(head -1 "$LOCK" 2>/dev/null)); it may have been killed after a request; check that no attnbench-h100-diag-* VM is listed, then remove $LOCK only under a new launch yes"
  fi
}
[[ -e "$LOCK" ]] && stop 2 "$(held)"
no_diag_vm 2
( set -o noclobber; echo "pid=$$ started=$(date -u +%FT%TZ) driver=$SP/diag_spot.sh" > "$LOCK" ) 2>/dev/null \
  || stop 2 "$(held)"
# kept once a request went out (the stale lock of a finished run); removed if none did
REQ=0
trap 'rc=$?; if (( REQ )); then echo "finished rc=$rc $(date -u +%FT%TZ)" >> "$LOCK"; else rm -f "$LOCK"; fi' EXIT
say "LOCK $LOCK taken (pid $$); no diag VM listed"
# --- money
MONEY=$(python3 -I -c 'import math, sys; s, cap, ceil, rate, d = map(float, sys.argv[1:]); b = math.ceil(d * rate / 60)
print("ok" if s + cap <= ceil and b <= cap else "no", b)' "$DIAG_SPENT" $CAP $CEILING $RATE $DEL)
say "GATE: spent Rs $DIAG_SPENT + cap Rs $CAP vs ceiling Rs $CEILING; delete bound Rs ${MONEY#* } at Rs $RATE/h ($DEL min): ${MONEY% *}"
[[ "${MONEY% *}" == ok ]] || stop 5 "STOP: the session could pass the ceiling or its cap; nothing requested"
# the end of the first window this driver sees
UNTIL=${DIAG_UNTIL:-$(python3 -I -c 'import sys; t, we = int(sys.argv[1]), int(sys.argv[2]); d = t - t % 86400 + we * 3600; print(d if t < d else d + 86400)' "$($NOW)" "${WINDOW#*-}")}
say "TERMS: Spot $ZONE only, image v6, halt $HALT min, delete $DEL min, Rs $RATE/h, one session; requests $WINDOW UTC until $(date -u -r "$UNTIL" +%FT%TZ)"
(( OVERRIDE )) && say "OVERRIDE: attempt 1 now, outside the window; attempt 2 only after a reclaim, inside it, if the diag spend + Rs $PROJ fits Rs $CAP; stop on the second reclaim"
[[ -n "${DIAG_DRY:-}" ]] && stop 0 "DRY: the checks passed; nothing requested"

GRANTS=0 DSPENT=0 K=0 HALT_K=$HALT DEL_K=$DEL BOUND_K=${MONEY#* }
while :; do
  # --- acquire
  NAME=""
  while :; do
    if free; then WS=0; else WS=$(wait_s "$MARGIN"); fi
    (( $($NOW) + WS < UNTIL )) || stop 4 "NOT GRANTED: no request can go out before $(date -u -r "$UNTIL" +%FT%TZ); stopped"
    if (( WS > 0 )); then say "WINDOW: waits $WS s for ${WINDOW%-*}:00Z"; $SLEEP "$WS"; continue; fi
    K=$((K+1)); N=attnbench-h100-diag-$(TZ=Asia/Kolkata date -r "$($NOW)" +%Y%m%d-%H%M%S)-a$K
    # the last look before the request goes out
    if free; then say "OVERRIDE: $N goes out under --window-override (attempt 1)"
    else (( $(wait_s "$MARGIN") == 0 )) || { say "REFUSED: $N would go out outside $WINDOW UTC or in its last $MARGIN s; not sent"; continue; }; fi
    REQ=1; say "REQUEST $N (session $((GRANTS+1)): halt $HALT_K min, delete $DEL_K min)"
    GCP_H100_PROVISIONING=SPOT GCP_ZONE=$ZONE GCP_ZONE_FALLBACKS=$ZONE GCP_RATE_INR_HR=$RATE \
      GCP_HALT_MINUTES=$HALT_K GCP_MAX_RUN=${DEL_K}m GCP_CANCEL_AFTER_SECONDS=${DIAG_CANCEL_AFTER:-300} \
      $LAUNCH "$N" > "$W/launch_a$K.log" 2>&1
    RC=$?
    sed -i '' "s/$P/<project>/g" "$W/launch_a$K.log"
    if (( RC == 4 )); then say "ATTEMPT $N: no Spot capacity"; $SLEEP "$SPACING"; continue; fi
    if (( RC != 0 )); then
      $G compute instances delete "$N" --zone "$ZONE" --project "$P" --quiet >/dev/null 2>&1
      (( RC == 3 )) && { say "ATTEMPT $N: launcher rc=3 (cancel unconfirmed); deleted by name"; $SLEEP "$SPACING"; continue; }
      stop 1 "STOPPED: the launcher failed (rc=$RC); anything of that name deleted"
    fi
    Z=$($G compute instances list --project "$P" --filter="name=$N" --format='value(zone.basename())' 2>/dev/null | head -1)
    if [[ "$Z" != "$ZONE" ]]; then
      [[ -n "$Z" ]] && $G compute instances delete "$N" --zone "$Z" --project "$P" --quiet >/dev/null 2>&1
      stop 1 "STOPPED: $N granted in '${Z:-nowhere}', not $ZONE; deleted"
    fi
    ST=""; for i in $(seq 1 20); do ST=$(status "$N" "$Z"); [[ "$ST" == RUNNING || -z "$ST" || "$ST" == TERMINATED ]] && break; $SLEEP 15; done
    NAME=$N; T0=$($NOW); GRANTS=$((GRANTS+1)); echo "$NAME $(date -u -r "$T0" +%FT%TZ)" >> "$W/granted"
    [[ "$ST" == RUNNING ]] && break
    say "GRANT $NAME $Z ${ST:-gone}, not RUNNING; torn down"; OK=0; STARTED=0; OVERRUN=0; break
  done
  SESSION=results/h100_diag_$(date -u -r "$T0" +%Y%m%dT%H%M%SZ)
  # --- chain, start, watch
  if [[ "$ST" == RUNNING ]]; then
    say "GRANT $NAME $Z RUNNING (Spot, Rs $RATE/h; halt $HALT_K min, delete $DEL_K min)"
    OK=1; STARTED=0; OVERRUN=0
    NAME=$NAME ZONE=$Z $CHAIN > "$W/chain_s$GRANTS.out" 2>&1 || OK=0
    if (( OK )); then
      CMD="cd ~/attnbench_scaffold && { nohup bash scripts/session_exit.sh --log ~/h100_diag.log -- \
python3 scripts/h100_diag.py run --cells $CELLS --out $SESSION >/dev/null 2>&1 < /dev/null & }"
      if ${START:-$G compute ssh "$NAME" --zone "$Z" --project "$P" --command} "$CMD" < /dev/null > "$W/start_s$GRANTS.out" 2>&1; then
        STARTED=1; say "STARTED $SESSION on $NAME"
      else OK=0; fi
    fi
    if (( OK )); then
      last=""; while :; do
        ST=$(status "$NAME" "$Z"); ST=${ST:-GONE}
        [[ "$ST" != "$last" ]] && { say "WATCH $NAME $ST"; last=$ST; }
        [[ "$ST" == TERMINATED || "$ST" == GONE ]] && break
        if (( $($NOW) - T0 > (DEL_K + 5) * 60 )); then
          say "OVERRUN: $NAME still $ST past the delete at $DEL_K min; deleting from here"
          $G compute instances delete "$NAME" --zone "$Z" --project "$P" --quiet >/dev/null 2>&1; OVERRUN=1; break
        fi
        $SLEEP 60
      done
    else
      say "SETUP FAILED on $NAME (chain or start); see chain_s$GRANTS.out, start_s$GRANTS.out"
    fi
  fi
  # --- teardown, book
  PRE=0
  $G compute operations list --project "$P" --filter="operationType=compute.instances.preempted AND targetLink~$NAME" \
     --format='value(name)' 2>/dev/null | grep -q . && PRE=1
  OVR=""; (( PRE )) && OVR="diag session preempted: the instance is gone or halted by Compute Engine"
  NOSYNC=""; (( ! STARTED )) && { OVR="diag session setup failed before the run: no session on the disk"; NOSYNC=$OVR; }
  GCP_RATE_INR_HR=$RATE TEARDOWN_DELETE_WITHOUT_HOME_LOGS="$OVR" TEARDOWN_DELETE_WITHOUT_BUCKET_SYNC="$NOSYNC" \
    $TEARDOWN "$NAME" "$Z" "$SESSION" > "$W/teardown_s$GRANTS.out" 2>&1
  TRC=$?; sed -i '' "s/$P/<project>/g" "$W/teardown_s$GRANTS.out"; say "TEARDOWN $NAME rc=$TRC"
  LEFT=$($G compute instances list --project "$P" --filter="name=$NAME" --format='value(status)' 2>/dev/null)
  [[ -n "$LEFT" ]] && say "LEFT: $NAME is still there ($LEFT) after the teardown; look before anything else"
  COST=$(sed -n 's/^est_inr: //p' "$SESSION/session_cost.txt" 2>/dev/null)
  MINS=$(sed -n 's/^minutes: //p' "$SESSION/session_cost.txt" 2>/dev/null); BASIS="the teardown's billed minutes ($MINS min)"
  if [[ -z "$COST" ]] && (( PRE )); then
    read -r COST MINS PS PE < <($G compute operations list --project "$P" --filter="targetLink~$NAME" \
      --format='value(operationType,insertTime,endTime)' 2>/dev/null | python3 -I "$SP/preempt_cost.py" "$RATE")
    BASIS="Compute Engine: create ${PS:-?} to preemption end ${PE:-?}, ${MINS:-?} min"
  fi
  [[ -z "${COST:-}" ]] && { COST=$BOUND_K; BASIS="no billed minutes read: the delete bound ($DEL_K min)"; }
  DSPENT=$(python3 -I -c 'import sys; print(round(float(sys.argv[1]) + float(sys.argv[2])))' "$DSPENT" "$COST")
  TOTAL=$(python3 -I -c 'import sys; print(round(float(sys.argv[1]) + float(sys.argv[2])))' "$DIAG_SPENT" "$DSPENT")
  echo "$NAME $COST $TOTAL" >> "$W/booked"
  say "BOOKED Rs $COST on $BASIS; diag spend Rs $DSPENT of Rs $CAP; spent now Rs $TOTAL of Rs $CEILING"
  if (( PRE )); then
    (( OVERRIDE )) || stop 7 "RECLAIMED: $NAME was preempted; stopped at the first reclaim, nothing reruns"
    (( GRANTS < 2 )) || stop 7 "RECLAIMED AGAIN: $NAME was preempted; stopped at the second reclaim, nothing reruns"
    [[ -n "$LEFT" ]] && stop 1 "STOPPED: $NAME is still listed after the first reclaim; no second attempt"
    # attempt 2: the diag spend + the projection inside the cap, and a delete cut so that the two
    # sessions together stay inside it (5 min kept for the watch's overrun grace)
    G2=$(python3 -I -c 'import math, sys; s, d, cap, ceil, rate, proj, halt, dl = map(float, sys.argv[1:])
dk = min(dl, math.floor((cap - d) * 60 / rate) - 5); hk = min(halt, dk - 10); b = math.ceil(dk * rate / 60)
print("ok" if d + proj <= cap and s + d + b <= ceil and hk > 0 else "no", int(dk), int(hk), b)' \
      "$DIAG_SPENT" "$DSPENT" $CAP $CEILING $RATE $PROJ $HALT $DEL)
    read -r G2OK DEL_K HALT_K BOUND_K <<< "$G2"
    say "GATE 2: diag spend Rs $DSPENT + Rs $PROJ vs cap Rs $CAP: $G2OK; session 2 halt $HALT_K min, delete $DEL_K min (bound Rs $BOUND_K)"
    [[ "$G2OK" == ok ]] || stop 5 "STOP: attempt 2 does not fit the cap after the reclaim; nothing more requested"
    no_diag_vm 1
    say "RECLAIMED: $NAME (session 1); attempt 2 waits for the window"
    continue
  fi
  (( OK && ! OVERRUN && TRC == 0 && ! ${#LEFT} )) || stop 1 "STOPPED: setup, teardown or the delete did not finish clean; look"
  stop 0 "DONE: $SESSION ran and $NAME is gone"
done
