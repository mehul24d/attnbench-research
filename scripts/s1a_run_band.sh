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
#   bash scripts/s1a_run_band.sh 2048
# (the teardown arms itself on completion -- see arm_teardown below)
# Exit status is also written to /tmp/s1a_<band>.rc.
set -uo pipefail

BAND="${1:?usage: s1a_run_band.sh BAND}"
OUT=results/s1a
REF=results/canary_ref
RCFILE="/tmp/s1a_${BAND}.rc"


# Arm the teardown on completion, whatever the outcome.
#
# The rule is older than this script: "a long run must arm its own teardown on
# completion, so the idle window is bounded by the machine rather than by
# whether anyone is awake" (spend_ledger.md, after the 2026-09-06 ~7h idle),
# demonstrated on 2026-09-07 when the chained halt fired with nobody watching.
# It lived in this file's own usage line as `; sudo shutdown -h +5` -- and on
# 2026-09-20 the S7 session was launched WITHOUT it, finished on estimate at
# 17:51Z, and sat until its 130-minute cap collected it at 18:40Z. 49 idle
# minutes, ~Rs 64. A rule that must be remembered at every call site will be
# forgotten at one of them (instance 47), so it is armed HERE, not documented.
#
# A halt is not a delete: GPU billing stops, the disk survives, and the
# `--max-run-duration` DELETE remains the outer backstop. `sudo shutdown -c`
# cancels it, which is why a FAILED run gets a longer grace -- long enough to
# ssh in and look at the box, short enough to stay bounded.
#
# CHAINING: set ATTNBENCH_NO_HALT=1 for every band but the last when running
# several in one session, or the first band's halt collects the machine out
# from under the second. Opt-OUT rather than opt-in on purpose: forgetting the
# flag kills a band loudly, in a way the log shows immediately, while
# forgetting to arm bills silently until a cap notices. Loud and cheap beats
# quiet and expensive.
#
# UNVERIFIED, to check on the next session: whether `shutdown -h` REPLACES the
# boot-time halt that gcp_launch_l4.sh schedules, or is refused because one is
# already pending. Both are plausible on systemd and this has not been run on
# an instance. The `shutdown -c` below makes the replace case explicit rather
# than relying on it; if the refusal case is real, the cancel is what makes
# this work at all. Recorded as a prediction to test, not as a settled fact --
# and note that between the cancel and the re-arm the in-guest backstop is
# momentarily gone, which the GCE-level DELETE still covers.
arm_teardown() {
  local rc="$1" mins
  [ "${ATTNBENCH_NO_HALT:-0}" = "1" ] && { echo "teardown NOT armed (ATTNBENCH_NO_HALT=1)"; return; }
  if [ "$rc" -eq 0 ]; then mins=5; else mins=20; fi
  echo "arming teardown: sudo shutdown -h +$mins (rc=$rc; 'sudo shutdown -c' cancels)"
  sudo shutdown -c >/dev/null 2>&1 || true
  sudo shutdown -h "+$mins" \
    || echo "COULD NOT ARM SHUTDOWN -- delete this instance by hand" >&2
}

finish() {
  echo "$1" > "$RCFILE"
  echo "=== s1a band $BAND rc=$1 $(date -u +%FT%TZ)"
  arm_teardown "$1"
  exit "$1"
}

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
