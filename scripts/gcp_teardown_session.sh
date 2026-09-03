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
#      here must not prevent step 4
#   3. age manifest third -- it also needs SSH, but unlike the results it can
#      be reconstructed from nothing, so it never goes ahead of them. Copying
#      does not alter mtimes on the instance, so it is just as accurate here
#   4. delete (never stop -- a stopped instance still bills for its disk)
#   5. verify nothing is left billing, including machine images, which
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

echo "== 1/5 serial console log -> $S/serial_console.log (before anything can destroy it)"
if gcloud compute instances get-serial-port-output "$NAME" --zone="$ZONE" \
     > "$S/serial_console.log" 2>/dev/null; then
  echo "   saved $(wc -l < "$S/serial_console.log") lines"
  echo -n "   OOM kills recorded: "
  grep -ciE "oom-kill|Killed process" "$S/serial_console.log" || true
else
  echo "   WARNING: could not fetch serial output -- continuing to deletion anyway"
fi

echo "== 2/5 results"
gcloud compute scp --recurse "$NAME:~/attnbench_scaffold/results" "$S/" --zone="$ZONE" 2>/dev/null \
  && echo "   copied" \
  || echo "   WARNING: scp failed (SSH may be gone) -- continuing to deletion"
for LOG in fa_build.log bsa_build.log batch_probe.log anchor16k.log; do
  gcloud compute scp "$NAME:~/$LOG" "$S/$LOG" --zone="$ZONE" 2>/dev/null \
    && echo "   copied $LOG" || true
done

echo "== 3/5 age manifest: which files predate this boot"
#
# A machine image is a disk snapshot, so files created by the session that
# CAPTURED it ride along inside it -- and this script cannot tell one of those
# from a file the current session produced. On 2026-09-04 it copied
# anchor16k.log, bsa_build.log and fa_build.log off a 6-minute diagnostic
# instance and filed them under that day's date. They were session-3/4
# artifacts that had been sitting in the v4 image.
#
# That is instance 7's hazard arriving through a new door: the boot-time cache
# clear stops stale DERIVED state from being read, but says nothing about
# stale OUTPUTS being copied out and misattributed. It is mechanically
# detectable, so it is detected here rather than left to whoever reads the
# directory later.
#
# mtimes are read ON THE INSTANCE: `gcloud compute scp` does not preserve
# them, so every local copy carries the copy time and the same comparison run
# here would be vacuous -- it would classify everything as "this session".
# Copying does not alter the source, so running after step 2 costs nothing.
#
# Nothing is deleted or renamed. A file older than boot is evidence; this only
# labels it. `-printf` is avoided deliberately so the classification is plain
# POSIX find and tests/test_teardown_age_manifest.py can execute the identical
# command against a fake tree rather than a paraphrase of it.
# Both fields come FROM THE INSTANCE. `uptime -s` prints instance-local time,
# and this script runs on a workstation that is not in the instance's timezone
# (IST vs UTC = 5h30m), so converting that string locally would be wrong by
# the offset -- and wrong in a way that still produces a plausible number.
# The instance converts its own clock; only the epoch crosses the wire.
BOOT_LINE="$(gcloud compute ssh "$NAME" --zone="$ZONE" --quiet \
              --command='printf "%s|%s\n" "$(uptime -s)" "$(date -d "$(uptime -s)" +%s)"' \
              2>/dev/null | tr -d '\r')"
BOOT_UTC="${BOOT_LINE%%|*}"
BOOT_EPOCH="${BOOT_LINE##*|}"
if [[ -n "$BOOT_UTC" ]]; then
  echo "   instance booted: $BOOT_UTC"
  gcloud compute ssh "$NAME" --zone="$ZONE" --quiet --command="
    echo 'boot_time: $BOOT_UTC'
    echo '--- PREDATES BOOT (image-carried, NOT this session) ---'
    find ~/attnbench_scaffold/results ~/*.log -type f ! -newermt '$BOOT_UTC' 2>/dev/null | sort
    echo '--- PRODUCED THIS SESSION ---'
    find ~/attnbench_scaffold/results ~/*.log -type f -newermt '$BOOT_UTC' 2>/dev/null | sort
  " > "$S/file_age_manifest.txt" 2>/dev/null \
    && echo "   wrote $S/file_age_manifest.txt" \
    || echo "   WARNING: could not build the manifest -- copied files are UNATTRIBUTED"
  # Billable duration, from the instance's own boot clock rather than from a
  # number carried along in prose. The standing rule is to report elapsed time
  # and spend on every report, and a total accumulated by hand across sessions
  # drifts -- it had reached a ~4% spread by 2026-09-04. This writes the
  # session's own row; docs/spend_ledger.md is assembled from these files.
  RATE_INR_HR="${GCP_RATE_INR_HR:-80}"       # g2-standard-8 + 1x L4, on-demand
  if [[ "$BOOT_EPOCH" =~ ^[0-9]+$ ]]; then
    MINS=$(( ( $(date +%s) - BOOT_EPOCH + 59 ) / 60 ))
    COST=$(awk -v m="$MINS" -v r="$RATE_INR_HR" 'BEGIN{printf "%.0f", m*r/60}')
    printf 'instance: %s\nzone: %s\nboot: %s\nteardown: %s\nminutes: %s\nrate_inr_hr: %s\nest_inr: %s\n' \
      "$NAME" "$ZONE" "$BOOT_UTC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$MINS" "$RATE_INR_HR" "$COST" > "$S/session_cost.txt"
    echo "   billable: ${MINS} min  ~= INR ${COST} at ${RATE_INR_HR}/hr -> $S/session_cost.txt"
  else
    echo "   WARNING: could not compute elapsed time from boot '$BOOT_UTC'"
  fi

  PREDATE=$(sed -n '/PREDATES BOOT/,/PRODUCED THIS/p' "$S/file_age_manifest.txt" 2>/dev/null \
            | grep -c '^/' || true)
  if [[ "${PREDATE:-0}" -gt 0 ]]; then
    echo "   !! ${PREDATE} file(s) predate this boot: IMAGE-CARRIED, not this"
    echo "      session's output. Do not read them as measurements from today."
  else
    echo "   every copied file postdates boot"
  fi
else
  echo "   WARNING: could not read boot time -- copied files will be UNATTRIBUTED"
fi

echo "== 4/5 delete (not stop -- a stopped instance still bills for its disk)"
gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet

echo "== 5/5 verify nothing is left billing"
bash "$(dirname "$0")/gcp_cleanup_check.sh"
echo
echo "machine images (billed for storage, NOT covered by the check above):"
gcloud compute machine-images list
