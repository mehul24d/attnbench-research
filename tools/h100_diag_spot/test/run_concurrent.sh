#!/bin/bash
# Two diag_spot.sh starts released together on one lock and one mocked project: only one may request.
#   run_concurrent.sh [DRIVER] [TRIALS]   (OVR=1: under --window-override; RERUN=1: under --rerun)
set -uo pipefail
SP=$(cd "$(dirname "$0")/.." && pwd); T=$SP/test; DRV=${1:-$SP/diag_spot.sh}; N=${2:-20}
WT=${WT:-$(cd "$SP/../.." && pwd)}
ok=0; bad=0
for i in $(seq 1 $N); do
  export M=$(mktemp -d "$T/c_XXXX"); R=$M/repo
  mkdir -p $R/configs/exploratory $R/docs/exploratory
  cp $WT/configs/exploratory/h100_diag_cells.json $WT/configs/exploratory/h100_diag_rerun_cells.json $R/configs/exploratory/
  cp $WT/docs/exploratory/h100_diag_card.md $WT/docs/exploratory/h100_diag_rerun_card.md $R/docs/exploratory/
  git -C $R init -q && git -C $R add -A && git -C $R -c user.email=m@m -c user.name=m commit -qm mock
  CF=h100_diag_cells.json; [[ -n "${RERUN:-}" ]] && CF=h100_diag_rerun_cells.json
  echo "cells_sha256: $(shasum -a 256 $R/configs/exploratory/$CF | cut -d' ' -f1)" > $M/approved
  [[ -n "${RERUN:-}" ]] && echo "terms: Spot us-central1-a, image v6, cap Rs 525, halt 40 billed min, delete at 50, one attempt, requests only inside 06:00-14:00Z, stop on the first reclaim" >> $M/approved
  if [[ -n "${OVR:-}" ]]; then echo "window_override: attempt 1 now, outside 06:00-14:00Z; attempt 2 only inside it" >> $M/approved
    echo "attempt_policy: attempt 2 only after a reclaim of attempt 1, inside the window, if the diag spend + Rs 504 fits the Rs 1300 cap; stop on the second reclaim" >> $M/approved; fi
  date -u -j -f %FT%TZ $( [[ -n "${OVR:-}" ]] && echo 2026-11-17T01:50:00Z || echo 2026-11-17T08:00:00Z ) +%s > $M/clock; echo us-central1-a > $M/grant_zone
  printf 'RUNNING\nRUNNING\nTERMINATED\n' > $M/statuses; printf '0\n0\n' > $M/launch_rcs
  export REPO=$R APPROVED=$M/approved DIAG_SPENT=9580 FA3_WHEEL_SHA256=x ATTNBENCH_GCP_PROJECT=mockproj
  export CMD_GCLOUD=$T/bin/gcloud CMD_LAUNCH=$T/bin/launch CMD_SLEEP=$T/bin/sleep CMD_NOW="cat $M/clock"
  export CMD_CHAIN=true CMD_TEARDOWN=$T/bin/teardown DIAG_LOCK=$M/lock
  for k in 1 2; do ( while [[ ! -e $M/go ]]; do :; done; DIAG_DIR=$M/w$k bash $DRV ${OVR:+--window-override} ${RERUN:+--rerun}; echo $? > $M/rc$k ) & done
  touch $M/go; wait
  L=$(wc -l < $M/launches 2>/dev/null || echo 0); L=${L// /}
  why=$(grep -h REFUSED $M/w1/diag.log $M/w2/diag.log 2>/dev/null | sed -E 's/.*REFUSED: (driver pid|stale lock from a finished|stale lock)[^(]*.*/\1/' | head -1)
  echo "trial $i: requests=$L rc=$(cat $M/rc1),$(cat $M/rc2) refused_by=[${why:-none}]"
  [[ $L == 1 ]] && ok=$((ok+1)) || bad=$((bad+1))
done
echo "RESULT $DRV: $ok of $N trials had exactly one request, $bad did not"
