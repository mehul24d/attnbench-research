#!/usr/bin/env bash
# The L4 rehearsal, instance A (docs/launch_cards/t61_rehearsal_l4.md): block
# b1 of the P2 runner, then T2.17's gate, the compiled ground truth (only with
# ATTNBENCH_REH_COMPILED=1: scope v, the first cut), each after the one before
# whatever it returned. T2.26's suite record is not here, and the relaunch does
# not run it (`gcp_bootstrap_instance.sh --suite-record` is off: 27 tests assume
# a CPU-only host, docs/gpu_suite_failures_20261110.md). Run under
# scripts/session_exit.sh, whose last act syncs and halts. EXPLORATORY: no
# row here enters training or any claim.
#
# Usage (on the instance, from ~/attnbench_scaffold):
#   nohup bash scripts/session_exit.sh --log ~/reh_a.log -- \
#       bash scripts/t61_rehearsal_a.sh results/t61r_l4_<UTC>_b1 >/dev/null 2>&1 &
set -uo pipefail

SESSION="${1:?usage: t61_rehearsal_a.sh results/<b1 session>}"
BANKED="${ATTNBENCH_REH_BANKED_AT:-P1=2026-11-09T11:12:31+00:00}"   # T3.2d's teardown
T32D="attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components"
TRAINING="banked/p1_training/p1_training_t32dc_20261109.json.gz"
TRAINING_SHA=de7f0f4497e87d7993681f4a6ba6544dcb9196853fb4e25d36ef1770df8a8c5f
COMPILED="${ATTNBENCH_REH_COMPILED:-0}"   # 1 only when the launch yes names the cap with (v)

say() { echo "=== $(date -u +%FT%TZ) $*"; }
bucket() { python3 -c 'import sys; sys.path.insert(0, "scripts"); import t61_session as S; print(S.results_bucket())'; }

B="$(bucket)" || { say "no results bucket"; exit 1; }
mkdir -p "$SESSION/p1_masks" results/p1_training

# Staging (the card's named bucket reads): T3.2d's XAttention masks at 2048
# and 8192, and the P1 training file, its hash checked.
for arm in xattention xattention_noassert; do
  for L in 2048 8192; do
    cell="${arm}_dcal_L${L}_b1"
    mkdir -p "$SESSION/p1_masks/$cell/recording"
    gsutil -q cp "$B/$T32D/$cell/recording/masks.npz" "$B/$T32D/$cell/recording/masks.json" \
      "$SESSION/p1_masks/$cell/recording/" || { say "staging $cell failed"; exit 1; }
  done
done
gsutil -q cp "$B/$TRAINING" results/p1_training/p1_training.json.gz || { say "no training file"; exit 1; }
got="$(sha256sum results/p1_training/p1_training.json.gz | cut -d' ' -f1)"
[[ "$got" == "$TRAINING_SHA" ]] || { say "training file sha256 $got, expected $TRAINING_SHA"; exit 1; }
du -ab "$SESSION/p1_masks" > "$SESSION/p1_masks_sizes.txt"
say "staged"

# (i)-(iii): block b1 of the P2 runner
python3 scripts/t61_rehearsal.py run --lock configs/t61/rehearsal_lock.json --block b1 \
  --session "$SESSION" --banked-at "$BANKED"
RC_B1=$?
say "b1 rc=$RC_B1"
python3 scripts/t61_rehearsal.py trace-sizes --session "$SESSION"

# (iv) T2.17: the per-session checks and the bitwise gate at 2048, three prompts
bash scripts/run_phase.sh t61r_assert_gate -- \
  python3 scripts/t227_steps.py assert_gate --out "$SESSION/assert_gate" --length 2048 --gate-prompts 3
say "assert_gate rc=$?"

# (v) the compiled groups at 2048 against b1's prediction (refused unless b1's
# predict and ground truth passed)
if [[ "$COMPILED" == 1 ]]; then
  bash scripts/run_phase.sh t61r_compiled -- \
    python3 scripts/t61_rehearsal.py compiled --session "$SESSION" --out "$SESSION/compiled"
  say "compiled rc=$?"
  python3 scripts/t61_rehearsal.py trace-sizes --session "$SESSION"
else
  say "compiled not run (scope v cut from this card's cap)"
fi
exit "$RC_B1"
