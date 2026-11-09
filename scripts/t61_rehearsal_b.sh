#!/usr/bin/env bash
# The L4 rehearsal, instance B (docs/launch_cards/t61_rehearsal_l4.md): block
# b2 of the P2 runner on a new instance, with a simulated interruption.
# Run under scripts/session_exit.sh. EXPLORATORY: no row enters any claim.
#
#   boot1  fetch b1's session from the bucket (A's instance), then start b2
#          in its own directory; the Mac resets the instance once b2's ground
#          truth has begun (scripts/t61_rehearsal_interrupt.sh).
#   boot2  (after the reset) the resume of that b2 must be refused and leave
#          INTERRUPTED_DISCARD.md; then b2 again in a new directory; then the
#          between-block drift check over b1, the discarded b2 and the new b2.
#
# Usage (on the instance, from ~/attnbench_scaffold):
#   nohup bash scripts/session_exit.sh --log ~/reh_b1.log -- \
#       bash scripts/t61_rehearsal_b.sh boot1 <A instance> <b1 session> <b2 interrupted> >/dev/null 2>&1 &
#   nohup bash scripts/session_exit.sh --log ~/reh_b2.log -- \
#       bash scripts/t61_rehearsal_b.sh boot2 <A instance> <b1 session> <b2 interrupted> <b2 session> >/dev/null 2>&1 &
set -uo pipefail

MODE="${1:?boot1 or boot2}"; A="${2:?the A instance}"; B1="${3:?b1 session}"; X="${4:?interrupted b2}"
Y="${5:-}"
BANKED="${ATTNBENCH_REH_BANKED_AT:-P1=2026-11-09T11:12:31+00:00}"
LOCK=configs/t61/rehearsal_lock.json
say() { echo "=== $(date -u +%FT%TZ) $*"; }
b2() { python3 scripts/t61_rehearsal.py run --lock "$LOCK" --block b2 --import-block "$B1" \
         --banked-at "$BANKED" "$@"; }

case "$MODE" in
  boot1)
    BK="$(python3 -c 'import sys; sys.path.insert(0, "scripts"); import t61_session as S; print(S.results_bucket())')" \
      || { say "no results bucket"; exit 1; }
    mkdir -p "$B1"
    gsutil -q -m rsync -r "$BK/$A/$B1" "$B1" || { say "fetching b1 failed"; exit 1; }
    say "b1 fetched"
    b2 --session "$X"
    say "b2 (to be interrupted) rc=$? -- reaching here means the reset came too late"
    exit 1
    ;;
  boot2)
    [[ -n "$Y" ]] || { say "boot2 needs the new b2 session"; exit 2; }
    b2 --session "$X" --from ground_truth
    rc=$?
    if [[ $rc -eq 2 && -f "$X/INTERRUPTED_DISCARD.md" ]]; then
      say "the interrupted b2 was refused and discarded (rc=2)"; DISCARD_OK=1
    else
      say "the interrupted b2 was NOT discarded (rc=$rc)"; DISCARD_OK=0
    fi
    b2 --session "$Y"
    RC_B2=$?
    say "b2 rc=$RC_B2"
    python3 scripts/t61_rehearsal.py trace-sizes --session "$Y"
    python3 - "$B1" "$X" "$Y" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
import t61_rehearsal as R
import t61_session as S
from pathlib import Path
from attnbench.analysis import block_drift as BD
dirs = sys.argv[1:]
got = BD.collect(dirs, ground_truth_dir=S.step_dir(Path(), "ground_truth", R.REHEARSAL).name)
res = BD.drift(got["rows"], order=["b1", "b2"])
out = {"blocks": got["blocks"], "discarded": got["discarded"], "drift": res,
       "bound": BD.WORKING_BOUND, "exploratory": R.EXPLORATORY_NOTE}
S.write_json(Path(dirs[-1]) / "block_drift.json", out)
print(json.dumps(out, default=str, indent=1))
EOF
    say "drift rc=$?"
    [[ $DISCARD_OK -eq 1 ]] || exit 1
    exit "$RC_B2"
    ;;
  *) echo "usage: t61_rehearsal_b.sh boot1|boot2 ..." >&2; exit 2 ;;
esac
