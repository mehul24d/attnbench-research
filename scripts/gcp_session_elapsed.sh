#!/usr/bin/env bash
# Report elapsed time and spend for a RUNNING session, from a source, at any
# point -- not only at teardown.
#
# Why this exists
# ---------------
# The standing rule is to report elapsed time and spend at every phase
# boundary. gcp_teardown_session.sh already computes both from the instance's
# own boot clock, but only once, at the end. So mid-session reports had no
# source to read and were produced by arithmetic on remembered timestamps
# instead. On 2026-09-17 that drifted from a reported 2.0 h to an actual
# 3.5 h, and the gap was Rs 429 of idle GPU nobody was counting
# (docs/silent_failure_patterns.md #40).
#
# This is the third time the same shape has cost something: a running spend
# total maintained by restatement drifted through Rs 902/910/918/942 until it
# was made to come from `session_cost.txt` (docs/spend_ledger.md), and
# provenance once recorded the literal string "HEAD" as a git commit because
# nothing compared it to a source (#3). Read it, don't compute it.
#
# Boot time comes from the GCE operations log rather than the guest, because
# it is authoritative, survives the instance, and needs no SSH -- so this
# still works when sshd is wedged, which is exactly when a session is most
# likely to be quietly burning.
set -euo pipefail

NAME="${1:?usage: gcp_session_elapsed.sh INSTANCE [ZONE] [RATE_INR_HR]}"
ZONE="${2:-asia-southeast1-c}"
RATE="${3:-${GCP_RATE_INR_HR:-284}}"
PROJECT="${GCP_PROJECT:-research-507316}"

STATUS=$(gcloud compute instances describe "$NAME" --project="$PROJECT" \
           --zone="$ZONE" --format='value(status)' 2>/dev/null || echo "GONE")

# creationTimestamp from the instance if it still exists; otherwise the
# insert operation, so a deleted session can still be accounted for.
if [[ "$STATUS" != "GONE" ]]; then
  BOOT=$(gcloud compute instances describe "$NAME" --project="$PROJECT" \
           --zone="$ZONE" --format='value(creationTimestamp)')
else
  # A name can be reused (gcp_boot_test_image.sh did, until 2026-09-19), and
  # pairing the FIRST insert with the EARLIEST end then yields an interval
  # belonging to no session at all. Take the most recent insert, say so when
  # there was more than one, and below take the first end event AFTER it.
  INSERTS=$(gcloud compute operations list --project="$PROJECT" \
           --filter="targetLink~${NAME}$ AND operationType=insert" \
           --format='value(startTime)' 2>/dev/null | sort)
  N_INS=$(printf '%s\n' "$INSERTS" | grep -c . || true)
  BOOT=$(printf '%s\n' "$INSERTS" | tail -1)
  if [[ "${N_INS:-0}" -gt 1 ]]; then
    echo "WARNING: $N_INS instances have used the name $NAME; reporting the most recent." >&2
  fi
fi
[[ -n "$BOOT" ]] || { echo "could not determine boot time for $NAME" >&2; exit 2; }

# End of the billed window: now if running, else the guest-shutdown or
# delete event, whichever the log has.
END=""
if [[ "$STATUS" == "GONE" || "$STATUS" == "TERMINATED" ]]; then
  END=$(gcloud compute operations list --project="$PROJECT" \
          --filter="targetLink~${NAME}$ AND (operationType=compute.instances.guestTerminate OR operationType=compute.instances.deferredDelete OR operationType=delete)" \
          --format='value(startTime)' 2>/dev/null | sort \
        | awk -v b="$BOOT" '$0 > b' | head -1)
fi

python3 - "$BOOT" "${END:-}" "$RATE" "$NAME" "$STATUS" <<'PY'
import sys
from datetime import datetime, timezone

boot_s, end_s, rate_s, name, status = sys.argv[1:6]
rate = float(rate_s)

def parse(t):
    # GCE emits RFC3339 with a numeric offset; normalise Z and let fromisoformat do it.
    return datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)

boot = parse(boot_s)
end = parse(end_s) if end_s else datetime.now(timezone.utc)
hours = (end - boot).total_seconds() / 3600.0
print(f"instance : {name}  [{status}]")
print(f"boot     : {boot:%Y-%m-%dT%H:%M:%SZ}")
print(f"{'now' if not end_s else 'ended':9}: {end:%Y-%m-%dT%H:%M:%SZ}")
print(f"elapsed  : {hours:.2f} h ({hours*60:.0f} min)")
print(f"spend    : Rs {hours*rate:,.0f} at Rs {rate:,.0f}/h")
if not end_s:
    print(f"           Rs {rate/60:.1f}/min while this stays up")
PY
