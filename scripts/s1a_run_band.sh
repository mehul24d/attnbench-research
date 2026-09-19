#!/usr/bin/env bash
# Audit item S1a, one band, on the instance:
#   dense arm alone -> dense canary (GATE) -> sparse arms.
#
# The three are chained with &&, so a canary that fails -- any dense example
# whose prediction, expected answer or tokenized length differs from the
# banked run, or fewer than 300 examples compared -- measures no sparse row.
# Each step is a run_phase.sh phase, so each syncs to GCS on exit whatever
# happened.
#
# What is held fixed against the banked stage3_s1 / stage3_s1b rows:
#   - example ids: n=100 is the seeded prefix of the banked n=300;
#   - sparse decode kernel: pinned to sdpa_math, as those runs decoded
#     (--pin-fallback-decode-backend; rows stamped decode_pinned=True);
#   - per-arm score precision: the score cache must be COLD at session start,
#     so the 0.5 arm ranks from fp32 and 0.75/0.9 from the fp16 cache copy,
#     exactly as the originals did (limitations.md, S5).
# What changes: the attention sink is forced (37675a0). Nothing else.
#
# Usage (on the instance, from the repo root):
#   bash scripts/s1a_run_band.sh 2048; sudo shutdown -h +5
# Exit status is also written to /tmp/s1a_<band>.rc.
set -uo pipefail

BAND="${1:?usage: s1a_run_band.sh BAND}"
OUT=results/s1a
REF=results/canary_ref
RCFILE="/tmp/s1a_${BAND}.rc"

finish() { echo "$1" > "$RCFILE"; echo "=== s1a band $BAND rc=$1 $(date -u +%FT%TZ)"; exit "$1"; }

for f in "$REF/stage3_s1.parquet" "$REF/stage3_s1b.parquet"; do
  [ -s "$f" ] || { echo "canary reference missing: $f" >&2; finish 5; }
done

# Cold cache, checked at the first band of the session (no S1a output yet).
# Later bands have their own keys (seq_len is in the key), so their first
# touch is a miss regardless of what earlier bands wrote.
if [ ! -e "$OUT/accuracy.parquet" ]; then
  N=$(find results/accuracy/score_cache -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$N" -ne 0 ]; then
    echo "SCORE CACHE NOT COLD ($N files) -- the 0.5 arm would rank from fp16" >&2
    echo "hits instead of fp32 misses, unlike the banked runs. Refusing." >&2
    finish 4
  fi
fi

COMMON=(--seq-lens "$BAND" --tasks niah_single,niah_multikey,vt
        --n-per-length 100 --lock-clocks --no-gla
        --pin-fallback-decode-backend sdpa_math --out "$OUT")

bash scripts/run_phase.sh "s1a_dense_${BAND}" -- \
    python3 -u scripts/run_accuracy.py "${COMMON[@]}" --only-backends sdpa_flash \
&& bash scripts/run_phase.sh "s1a_canary_${BAND}" -- \
    python3 scripts/check_dense_canary.py --new "$OUT/accuracy.parquet" \
      --banked "$REF/stage3_s1.parquet" "$REF/stage3_s1b.parquet" \
      --seq-len "$BAND" --min-checked 300 --record "$OUT/canary_${BAND}.json" \
&& bash scripts/run_phase.sh "s1a_sparse_${BAND}" -- \
    python3 -u scripts/run_accuracy.py "${COMMON[@]}"
finish $?
