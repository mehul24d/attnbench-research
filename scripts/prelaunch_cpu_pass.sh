#!/usr/bin/env bash
# The pre-launch CPU pass (adviser's ruling 4 of the 11 Nov 2026 paste): every
# launch card runs it on the card's commit before the researcher is asked for
# a yes, and the card's yes cites its record.
#
#   scripts/prelaunch_cpu_pass.sh <rehearsal|p2|p4> [training file] [out dir]
#
# The session's real config and name, the real start checks and every step's
# real body a CPU can run; only the GPU steps, the device and the bucket are
# mocked (tests/test_prelaunch_cpu_pass.py says which). Refuses a dirty tree,
# since the pass is of the commit that deploys. P2 and P4 read the lock at
# docs/lock_values.json; ATTNBENCH_PRELAUNCH_LOCK names a block spec and
# ATTNBENCH_PRELAUNCH_BANKED ("P1=ISO P2=ISO") the banking times.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in rehearsal|p2|p4) ;; *) echo "usage: $0 <rehearsal|p2|p4> [training] [out]" >&2; exit 2;; esac
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING: the tree is dirty; the pass is of the commit that deploys" >&2; exit 2
fi
out=${3:-results/prelaunch/${1}_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%MZ)}
mkdir -p "$out"
set +e
ATTNBENCH_PRELAUNCH=$1 ATTNBENCH_PRELAUNCH_TRAINING=${2:-} ATTNBENCH_PRELAUNCH_OUT=$out \
  python3 -m pytest -q -s -p no:cacheprovider tests/test_prelaunch_cpu_pass.py 2>&1 | tee "$out/pass.log"
rc=${PIPESTATUS[0]}
set -e
python3 - "$out/prelaunch_cpu_pass.json" "$rc" <<'PY'
import json, sys
p, rc = sys.argv[1], int(sys.argv[2])
try:
    r = json.load(open(p))
except FileNotFoundError:
    r = {"passed": False}
ok = rc == 0 and r.get("passed")
print(f"{'PASSED' if ok else 'FAILED'}: pre-launch CPU pass of {r.get('config')} "
      f"({r.get('pairing')}) at {str(r.get('commit'))[:7]}; record {p}")
if r.get("start_refusal"):
    print(f"  start refused: {r['start_refusal']}")
for name, s in (r.get("sessions") or {}).items():
    bad = [a for a in s["attempts"] if a["status"] not in ("passed", "skipped")]
    print(f"  {name}: rc {s['rc']}" + (f"; {bad[-1]['step']}: {bad[-1]['reason'][:300]}" if bad else ""))
PY
exit "$rc"
