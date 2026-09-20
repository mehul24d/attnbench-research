#!/usr/bin/env bash
# Audit item S7, the 16384 band: did instance 45 move a published number?
#
# Instance 45 (5cc3a40) is the tie-break jitter stream. The draw was taken
# AFTER the `budget <= 0` continue, so a row skipped at one sparsity did not
# advance the generator, and every later row at that sparsity drew different
# jitter than the same row at another sparsity. Where scores tie -- which is
# where the jitter decides -- the arms then broke the tie differently, so the
# 0.9 mask was not guaranteed to be a subset of 0.75. The fix draws first.
#
# Ties are an fp16 artefact and they get commoner with length, so the blast
# radius is length-dependent: 1.7% of layer-masks at 2048, 73.8% at 16384.
# 16384 is the headline band, which is why it is the one re-measured.
#
# Bounded locally first, from the banked run's own fp16 score tensors (they
# survive in GCS and its 0.75/0.9 arms ranked from them): over 200 real
# examples, 0 have an unchanged mask. There is no subset that can be skipped,
# so this runs the whole cell.
#
# What is held fixed against results/accuracy_forced_sink/accuracy.parquet
# (host attnbench-l4-20260916-1501, commit 166b2df, n=100, forced sink):
#   - example ids: same grid, same --n-per-length 100, same tasks;
#   - sparse decode kernel: PINNED to sdpa_flash. The banked rows stamp
#     decode_backend=sdpa_flash and that is also today's DENSE_DECODE_BACKEND,
#     so the pin is a no-op today and a guard against it drifting. NOTE this
#     is NOT the sdpa_math that s1a_run_band.sh pins -- the 2048-8192 bands
#     decoded through sdpa_math and this one did not;
#   - per-arm score precision: the cache must be COLD at session start, so
#     the 0.5 arm ranks from fp32 and 0.75/0.9 from the fp16 cache copy,
#     exactly as the banked run did (limitations.md, S5). A warm cache would
#     make all three arms fp16 and confound the mask change with removing
#     that asymmetry, which is why the surviving GCS cache is NOT reused.
# What changes: the jitter draw. Plus 93d821c, which pins allow_tf32=False
# for the scoring pass -- torch's default on this card already, so expected
# to be a no-op, and checked rather than assumed: the new run's fp16 score
# tensors are compared byte-for-byte against the banked ones afterwards
# (scripts/check_score_canary.py). The dense canary cannot cover that, since
# a dense cell never computes importance scores.
#
# vt is deliberately excluded: its stopping behaviour is a known confound
# (limitations.md), so a flip there would not attribute cleanly.
#
# Usage (on the instance, from the repo root):
#   bash scripts/s7_run_jitter_band.sh; sudo shutdown -h +5
# Exit status is also written to /tmp/s7_jitter.rc.
set -uo pipefail

BAND=16384
TASKS=niah_single,niah_multikey
N=100
MIN_CHECKED=200          # N x tasks, the dense examples the canary must compare
OUT=results/s7_jitter
REF=results/canary_ref
BANKED="$REF/accuracy_forced_sink.parquet"
BANKED_MD5=4101d787f7951b72e289a583fdeb5c71
BANKED_CACHE=/tmp/banked_score_cache   # outside results/, so run_phase.sh does
                                       # not rsync 370MB of banked tensors back
BUCKET="${ATTNBENCH_BUCKET:-gs://attnbench-results-research-507316}"
BANKED_URI="$BUCKET/l4-forced-sink/results/accuracy_forced_sink/accuracy.parquet"
BANKED_CACHE_URI="$BUCKET/l4-forced-sink/results/accuracy/score_cache"
RCFILE=/tmp/s7_jitter.rc

finish() { echo "$1" > "$RCFILE"; echo "=== s7 jitter band $BAND rc=$1 $(date -u +%FT%TZ)"; exit "$1"; }

# 1. The canary reference, fetched from the bucket rather than shipped: it is
#    gitignored measurement output, and the bucket copy IS the banked file.
#    Verified by digest, because "a file arrived" is not "the right file".
mkdir -p "$REF"
if [ ! -s "$BANKED" ]; then
  gsutil cp "$BANKED_URI" "$BANKED" || { echo "could not fetch $BANKED_URI" >&2; finish 5; }
fi
GOT=$(md5sum "$BANKED" | cut -d' ' -f1)
if [ "$GOT" != "$BANKED_MD5" ]; then
  echo "canary reference digest $GOT != expected $BANKED_MD5 -- refusing" >&2
  finish 5
fi
echo "canary reference: $BANKED ($GOT) ok"

# 2. Cold cache. Checked before the first arm; later phases resume into the
#    same output and their cache entries are this session's own.
if [ ! -e "$OUT/accuracy.parquet" ]; then
  NCACHE=$(find results/accuracy/score_cache -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NCACHE" -ne 0 ]; then
    echo "SCORE CACHE NOT COLD ($NCACHE files) -- every arm would rank from" >&2
    echo "fp16, unlike the banked run, and the comparison would not isolate" >&2
    echo "the mask rule. Refusing." >&2
    finish 4
  fi
  echo "score cache: cold"
fi

COMMON=(--seq-lens "$BAND" --tasks "$TASKS"
        --n-per-length "$N" --lock-clocks --no-gla
        --pin-fallback-decode-backend sdpa_flash --out "$OUT")

# The dense arm gates the sparse one: a canary failure measures no sparse row.
bash scripts/run_phase.sh "s7_dense_${BAND}" -- \
    python3 -u scripts/run_accuracy.py "${COMMON[@]}" --only-backends sdpa_flash \
&& bash scripts/run_phase.sh "s7_canary_${BAND}" -- \
    python3 scripts/check_dense_canary.py --new "$OUT/accuracy.parquet" \
      --banked "$BANKED" \
      --seq-len "$BAND" --min-checked "$MIN_CHECKED" --record "$OUT/canary_${BAND}.json" \
&& bash scripts/run_phase.sh "s7_sparse_${BAND}" -- \
    python3 -u scripts/run_accuracy.py "${COMMON[@]}"
MEASURE_RC=$?
[ "$MEASURE_RC" -eq 0 ] || finish "$MEASURE_RC"

# Both of the next two run, and NEITHER gates the other. The score canary says
# whether the scoring pass reproduced; the comparison says what moved. Reading
# the comparison without the canary is how a scoring change gets attributed to
# the mask rule, and skipping the comparison when the canary fails would throw
# away a measurement that has already been paid for -- it would just have to
# be read as "something changed" rather than "the mask rule changed".
bash scripts/run_phase.sh "s7_scorecanary_${BAND}" -- \
    python3 scripts/check_score_canary.py --rows "$OUT/accuracy.parquet" \
      --new-cache results/accuracy/score_cache --banked-cache "$BANKED_CACHE" \
      --fetch-from "$BANKED_CACHE_URI" \
      --seq-len "$BAND" --min-checked "$MIN_CHECKED" \
      --record "$OUT/score_canary_${BAND}.json"
SCORE_RC=$?

bash scripts/run_phase.sh "s7_compare_${BAND}" -- \
    python3 scripts/run_s1a_comparison.py --new "$OUT/accuracy.parquet" \
      --banked "$BANKED" --seq-len "$BAND" --n "$N" \
      --label "S7 band $BAND: post-jitter-fix vs banked forced-sink" \
      --out "$OUT/comparison_${BAND}.parquet"
CMP_RC=$?

echo "=== s7 score canary rc=$SCORE_RC, comparison rc=$CMP_RC"
[ "$SCORE_RC" -eq 0 ] || echo "READ THE COMPARISON AS CONFOUNDED: the scoring" \
    "pass did not reproduce, so flips are not attributable to the mask rule." >&2
[ "$SCORE_RC" -eq 0 ] && [ "$CMP_RC" -eq 0 ]
finish $?
