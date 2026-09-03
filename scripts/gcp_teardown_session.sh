#!/usr/bin/env bash
# Tear down a session instance in the one order that cannot lose evidence.
#
# Written during the 2026-09-03 session's build wait, deliberately BEFORE the
# deadline, because teardown is executed at the moment time pressure is
# highest and that is exactly when a step gets skipped. On 2026-09-02 the
# serial console log was read only through transient `grep`/`tail` and never
# saved; the instance was then deleted, and the question it would have
# answered -- whether sshd was among the eight OOM kills -- is now permanently
# unanswerable. That is the failure this script exists to make impossible.
#
# Order is load-bearing:
#   1. serial console log FIRST -- it is destroyed with the instance and is
#      the only diagnostic channel that survives losing SSH
#   2. results second -- they need SSH, which may already be gone; a failure
#      here must not prevent step 3
#   3. delete (never stop -- a stopped instance still bills for its disk)
#   4. verify nothing is left billing, including machine images, which
#      gcp_cleanup_check.sh does not cover
#
# Steps 1 and 2 are allowed to fail without aborting the run: an instance
# whose SSH is dead is precisely when deletion matters most.
#
#   bash scripts/gcp_teardown_session.sh INSTANCE_NAME ZONE [RESULTS_DIR]
set -uo pipefail

NAME="${1:?usage: gcp_teardown_session.sh INSTANCE_NAME ZONE [RESULTS_DIR]}"
ZONE="${2:?zone required}"
S="${3:-results/gpu_session_$(date +%Y%m%d)}"
mkdir -p "$S"

echo "== 1/4 serial console log -> $S/serial_console.log (before anything can destroy it)"
if gcloud compute instances get-serial-port-output "$NAME" --zone="$ZONE" \
     > "$S/serial_console.log" 2>/dev/null; then
  echo "   saved $(wc -l < "$S/serial_console.log") lines"
  echo -n "   OOM kills recorded: "
  grep -ciE "oom-kill|Killed process" "$S/serial_console.log" || true
else
  echo "   WARNING: could not fetch serial output -- continuing to deletion anyway"
fi

echo "== 2/4 results"
gcloud compute scp --recurse "$NAME:~/attnbench_scaffold/results" "$S/" --zone="$ZONE" 2>/dev/null \
  && echo "   copied" \
  || echo "   WARNING: scp failed (SSH may be gone) -- continuing to deletion"
for LOG in fa_build.log bsa_build.log batch_probe.log anchor16k.log; do
  gcloud compute scp "$NAME:~/$LOG" "$S/$LOG" --zone="$ZONE" 2>/dev/null \
    && echo "   copied $LOG" || true
done

echo "== 3/4 delete (not stop -- a stopped instance still bills for its disk)"
gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet

echo "== 4/4 verify nothing is left billing"
bash "$(dirname "$0")/gcp_cleanup_check.sh"
echo
echo "machine images (billed for storage, NOT covered by the check above):"
gcloud compute machine-images list
