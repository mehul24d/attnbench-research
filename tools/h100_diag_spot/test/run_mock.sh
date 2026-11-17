#!/bin/bash
# diag_spot.sh end to end with gcloud, the launcher, the clock and the instance mocked.
#   run_mock.sh SCENARIO
set -uo pipefail
SP=$(cd "$(dirname "$0")/.." && pwd); T=$SP/test; WT=${WT:-$(cd "$SP/../.." && pwd)}
SC="$1"; export M=$(mktemp -d "$T/m_${SC}_XXXX"); R=$M/repo
mkdir -p $R/configs/exploratory $R/docs/exploratory
cp $WT/configs/exploratory/h100_diag_cells.json $WT/configs/exploratory/h100_diag_rerun_cells.json $R/configs/exploratory/
cp $WT/docs/exploratory/h100_diag_card.md $WT/docs/exploratory/h100_diag_rerun_card.md $R/docs/exploratory/
git -C $R init -q && git -C $R add -A && git -C $R -c user.email=m@m -c user.name=m commit -qm mock
SHA=$(shasum -a 256 $R/configs/exploratory/h100_diag_cells.json | cut -d' ' -f1)
echo "cells_sha256: $SHA" > $M/approved
at() { date -u -j -f %FT%TZ "2026-11-17T$1:00Z" +%s; }
echo $(at 08:00) > $M/clock; echo us-central1-a > $M/grant_zone; printf 'RUNNING\nRUNNING\nTERMINATED\n' > $M/statuses
printf '0\n' > $M/launch_rcs
export REPO=$R APPROVED=$M/approved DIAG_DIR=$M/w DIAG_SPENT=9580 FA3_WHEEL_SHA256=x ATTNBENCH_GCP_PROJECT=mockproj
export CMD_GCLOUD=$T/bin/gcloud CMD_LAUNCH=$T/bin/launch CMD_SLEEP=$T/bin/sleep CMD_NOW="cat $M/clock"
export CMD_CHAIN=true CMD_TEARDOWN=$T/bin/teardown
export DIAG_LOCK=${DIAG_LOCK:-$M/lock}
OVL="window_override: attempt 1 now, outside 06:00-14:00Z; attempt 2 only inside it"
POL="attempt_policy: attempt 2 only after a reclaim of attempt 1, inside the window, if the diag spend + Rs 504 fits the Rs 1300 cap; stop on the second reclaim"
TERMS="terms: Spot us-central1-a, image v6, cap Rs 525, halt 40 billed min, delete at 50, one attempt, requests only inside 06:00-14:00Z, stop on the first reclaim"
RSHA=$(shasum -a 256 $R/configs/exploratory/h100_diag_rerun_cells.json | cut -d' ' -f1)
rerun() { ARGS=--rerun; echo "cells_sha256: $RSHA" > $M/approved; echo "$TERMS" >> $M/approved; }
over() { ARGS=--window-override; echo "$OVL" >> $M/approved; echo "$POL" >> $M/approved; echo $(at 01:50) > $M/clock; }
case "$SC" in
  clean) ;;
  before) echo $(at 05:00) > $M/clock ;;
  after) echo $(at 15:00) > $M/clock ;;
  late) echo $(at 13:56) > $M/clock ;;
  nocap) printf '4\n4\n0\n' > $M/launch_rcs ;;
  nocap_all) echo $(at 13:00) > $M/clock; printf '4\n%.0s' $(seq 1 20) > $M/launch_rcs ;;
  cancel3) printf '3\n0\n' > $M/launch_rcs ;;
  launchfail) printf '1\n' > $M/launch_rcs ;;
  reclaim) printf 'RUNNING\nRUNNING\nTERMINATED\n' > $M/statuses; touch $M/preempted $M/no_cost ;;
  money) export DIAG_SPENT=14300 ;;
  unapproved) echo "cells_sha256: 0000" > $M/approved ;;
  dirty) touch $R/stray ;;
  twice) mkdir -p $M/w; echo "earlier-vm 2026-11-17T07:00:00Z" > $M/w/granted ;;
  wrongzone) echo us-central1-b > $M/grant_zone ;;
  notrunning) printf 'STOPPING\nTERMINATED\n' > $M/statuses ;;
  overrun) printf 'RUNNING\n' > $M/statuses ;;
  setupfail) export CMD_CHAIN=false ;;
  nocost) touch $M/no_cost ;;
  dry) export DIAG_DRY=1 ;;
  vm_exists) echo attnbench-h100-diag-20261117-113000-a1 > $M/existing_vms ;;
  list_fails) touch $M/list_fails ;;
  live_lock) sleep 30 & echo "pid=$! started=2026-11-17T05:00:00Z driver=/x/diag_spot.sh" > $DIAG_LOCK ;;
  stale_finished) printf 'pid=999999 started=2026-11-17T06:00:00Z driver=/x/diag_spot.sh\nfinished rc=0 2026-11-17T08:05:00Z\n' > $DIAG_LOCK ;;
  now_no_override) echo $(at 01:50) > $M/clock ;;
  override_flag_only) ARGS=--window-override; echo $(at 01:50) > $M/clock ;;
  override_no_policy) ARGS=--window-override; echo "$OVL" >> $M/approved; echo $(at 01:50) > $M/clock ;;
  override_now) over ;;
  override_nocap) over; printf '4\n4\n0\n' > $M/launch_rcs ;;
  override_reclaim1) over; echo 1 > $M/preempted; printf 'RUNNING\nRUNNING\nTERMINATED\nRUNNING\nRUNNING\nTERMINATED\n' > $M/statuses; printf '0\n0\n' > $M/launch_rcs ;;
  override_reclaim2) over; : > $M/preempted; touch $M/no_cost; printf 'RUNNING\nRUNNING\nTERMINATED\nRUNNING\nRUNNING\nTERMINATED\n' > $M/statuses; printf '0\n0\n' > $M/launch_rcs ;;
  override_money2) over; echo 1 > $M/preempted; touch $M/no_cost; echo 2026-11-17T09:20:10+00:00 > $M/preempt_end; printf '0\n0\n' > $M/launch_rcs ;;
  override_live_lock) over; sleep 30 & echo "pid=$! started=2026-11-17T01:00:00Z driver=/x/diag_spot.sh" > $DIAG_LOCK ;;
  override_vm_exists) over; echo attnbench-h100-diag-20261117-070000-a1 > $M/existing_vms ;;
  rerun_clean) rerun ;;
  rerun_before) rerun; echo $(at 05:00) > $M/clock ;;
  rerun_after) rerun; echo $(at 15:00) > $M/clock ;;
  rerun_money) rerun; export DIAG_SPENT=14976 ;;
  rerun_money_edge) rerun; export DIAG_SPENT=14975 ;;
  rerun_no_terms) rerun; echo "cells_sha256: $RSHA" > $M/approved ;;
  rerun_first_approval) ARGS=--rerun; echo "$TERMS" >> $M/approved ;;
  rerun_override) rerun; ARGS="--rerun --window-override" ;;
  rerun_reclaim) rerun; printf 'RUNNING\nRUNNING\nTERMINATED\nRUNNING\nRUNNING\nTERMINATED\n' > $M/statuses; printf '0\n0\n' > $M/launch_rcs; : > $M/preempted; touch $M/no_cost ;;
  rerun_nocap) rerun; printf '4\n4\n0\n' > $M/launch_rcs ;;
  rerun_stale_finished) rerun; printf 'pid=999999 started=2026-11-17T01:54:24Z driver=/x/diag_spot.sh\nfinished rc=0 2026-11-17T02:16:39Z\n' > $DIAG_LOCK ;;
  rerun_vm_exists) rerun; echo attnbench-h100-diag-20261117-113000-a1 > $M/existing_vms ;;
  rerun_twice) rerun; mkdir -p $M/w; echo "earlier-vm 2026-11-17T07:00:00Z" > $M/w/granted ;;
  stale_killed) echo "pid=999999 started=2026-11-17T06:00:00Z driver=/x/diag_spot.sh" > $DIAG_LOCK ;;
esac
bash $SP/diag_spot.sh ${ARGS:-}; rc=$?
echo "SCENARIO $SC rc=$rc"; sed 's/^/  /' $M/w/diag.log
for f in launches started teardowns deleted; do [[ -e $M/$f ]] && { echo "  [$f]"; sed "s|$M|<M>|g; s/^/    /" $M/$f; }; done
[[ -e $M/w/booked ]] && echo "  [booked] $(cat $M/w/booked)"
[[ -e $DIAG_LOCK ]] && { echo "  [lock]"; sed "s|$M|<M>|g; s/^/    /" $DIAG_LOCK; } || echo "  [lock] none"
exit 0
