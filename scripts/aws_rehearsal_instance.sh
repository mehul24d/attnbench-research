#!/usr/bin/env bash
# The L4 rehearsal's work (g6.2xlarge from the fourth attempt), on the instance (docs/launch_cards/aws_rehearsal_g6.md;
# adviser's task of 23 Nov 2026, second, item 2). Runs from the deployed
# repository's root after the deploy; the Mac's driver starts it detached and
# pulls OUT when it ends. EXPLORATORY: nothing it writes enters training, the
# floors, the prediction or any claim. One attempt: the first failed check
# stops it, and nothing is retried.
#
#   bash scripts/aws_rehearsal_instance.sh OUT
#
#   0. guards: a clean tree, /usr/bin/python3 at 3.10.12, a new OUT
#   1. gpu             GPU and driver, the libraries, flash-attn's cubins
#                      (scripts/aws_rehearsal_checks.py gpu)
#   2. fa2             the dense selection's correctness screen at 2048 and
#                      8192, fa2 correct at both (aws_rehearsal_checks.py fa2)
#   3. dense_select    T3.1's step 3 on the one cell of configs/aws/rehearsal_g6.json
#                      (dense, 8192, batch 1): the session floor's dense pre-pass
#                      and K2's timed rounds, every L4 candidate
#   4. host_components T3.1's step 5 on the same cell, timed
#
# Each check's wall time goes to OUT/timings.tsv (check, status, seconds);
# the last line printed is REHEARSAL_OK or REHEARSAL_FAILED <check>.
set -uo pipefail
OUT="${1:?usage: aws_rehearsal_instance.sh OUT}"
PY="${ATTNBENCH_PYTHON:-/usr/bin/python3}"
CONFIG="configs/aws/rehearsal_g6.json"
export ATTNBENCH_PYTHON="$PY" PYTHONDONTWRITEBYTECODE=1

fail() { echo "REHEARSAL_FAILED $1"; exit 1; }
[[ -e "$OUT" ]] && { echo "REFUSING: $OUT exists"; fail guards; }
[[ -z "$(git status --porcelain)" ]] || { echo "REFUSING: the tree is not clean"; fail guards; }
[[ "$("$PY" -c 'import platform; print(platform.python_version())')" == "3.10.12" ]] \
  || { echo "REFUSING: $PY is not Python 3.10.12"; fail guards; }
mkdir -p "$OUT/session"
printf 'check\tstatus\tseconds\n' > "$OUT/timings.tsv"
echo "commit $(git rev-parse HEAD)" > "$OUT/commit.txt"

run() {   # run CHECK CMD...: timed, logged, stop on failure
  local name="$1"; shift
  echo "== $name: $*"
  local t0 t1 rc
  t0="$(date +%s.%N)"
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  t1="$(date +%s.%N)"
  local s; s="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b - a}')"
  local st=passed; [[ $rc -eq 0 ]] || st=failed
  printf '%s\t%s\t%s\n' "$name" "$st" "$s" >> "$OUT/timings.tsv"
  echo "   $name $st in ${s} s"
  [[ $rc -eq 0 ]] || { tail -25 "$OUT/$name.log"; fail "$name"; }
}

S="$OUT/session"
run gpu "$PY" scripts/aws_rehearsal_checks.py gpu "$OUT/checks"
run fa2 "$PY" scripts/aws_rehearsal_checks.py fa2 "$OUT/checks" --config "$CONFIG"
run dense_select "$PY" scripts/t31_steps.py dense_select --out "$S/03_dense_select" \
  --session "$S" --config "$CONFIG"
run host_components "$PY" scripts/t31_steps.py host_components --out "$S/05_host_components" \
  --session "$S" --config "$CONFIG"
echo "REHEARSAL_OK"
