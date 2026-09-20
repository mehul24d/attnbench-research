#!/usr/bin/env bash
# The arbitrary-free-block control for the attention-sink rule, one band.
#
# S1a showed that forcing the sink lifts 2048/0.9 hard -- niah_single 63->97,
# niah_multikey 0->44, vt 25.8->70.4 -- but the fix grants the sink AND one
# more block per row (+53.8% of active blocks at that cell), and nothing so
# far separates them. This arm grants one ARBITRARY off-diagonal block per
# row instead, at the identical per-row block count
# (mask_source=importance_randfree), so:
#
#   control ~= forced sink  -> the sink is not special; the gain was density
#   forced sink >> control  -> the sink is doing real work
#
# Everything else is held to S1a: same example ids, decode pinned to
# sdpa_math, cold score cache, dense arm gated against the banked run.
#
# Usage (on the instance, from the repo root):
#   bash scripts/sink_control_run.sh 2048 0.9; sudo shutdown -h +5
set -uo pipefail

BAND="${1:?usage: sink_control_run.sh BAND SPARSITY}"
SPARSITY="${2:?usage: sink_control_run.sh BAND SPARSITY}"
OUT=results/sink_control
REF=results/canary_ref
RCFILE="/tmp/sink_control_${BAND}.rc"

finish() { echo "$1" > "$RCFILE"; echo "=== sink control band $BAND rc=$1 $(date -u +%FT%TZ)"; exit "$1"; }

for f in "$REF/stage3_s1.parquet" "$REF/stage3_s1b.parquet"; do
  [ -s "$f" ] || { echo "canary reference missing: $f" >&2; finish 5; }
done

if [ ! -e "$OUT/accuracy.parquet" ]; then
  N=$(find results/accuracy/score_cache -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$N" -ne 0 ]; then
    echo "SCORE CACHE NOT COLD ($N files) -- refusing, per-arm precision would" >&2
    echo "differ from S1a and the comparison would not be like-for-like." >&2
    finish 4
  fi
fi

COMMON=(--seq-lens "$BAND" --tasks niah_single,niah_multikey,vt
        --n-per-length 100 --lock-clocks --no-gla
        --pin-fallback-decode-backend sdpa_math --out "$OUT")

bash scripts/run_phase.sh "ctrl_dense_${BAND}" -- \
    python3 -u scripts/run_accuracy.py "${COMMON[@]}" --only-backends sdpa_flash \
&& bash scripts/run_phase.sh "ctrl_canary_${BAND}" -- \
    python3 scripts/check_dense_canary.py --new "$OUT/accuracy.parquet" \
      --banked "$REF/stage3_s1.parquet" "$REF/stage3_s1b.parquet" \
      --seq-len "$BAND" --min-checked 300 --record "$OUT/canary_${BAND}.json" \
&& bash scripts/run_phase.sh "ctrl_randfree_${BAND}" -- \
    python3 -u scripts/run_accuracy.py "${COMMON[@]}" \
      --sparsities "$SPARSITY" --mask-source importance_randfree
finish $?
