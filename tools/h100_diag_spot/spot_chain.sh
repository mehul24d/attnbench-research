#!/bin/bash
# One Spot block's setup, from $REPO (the fresh clone at the tag): deploy, preflight
# --quarantine, the GCS write gate, bootstrap without the suite record. NAME, ZONE in env.
set -uo pipefail
: "${NAME:?}" "${ZONE:?}" "${ATTNBENCH_GCP_PROJECT:?}"
red() { sed "s/$ATTNBENCH_GCP_PROJECT/<project>/g"; }
# a setup step that fails on the ssh transport (the network, not the step) runs once more after ssh
# answers again (16 Nov 2026, rehearsal 3); rc 255 is ssh's own and counts even when the step printed
# nothing (16 Nov, b2-a5's preflight); any other failure stops at once
SSH_FLAKE='ssh_dispatch_run_fatal|Operation timed out|Connection timed out|Connection refused|Connection closed|Connection reset|Broken pipe|exited with return code \\[255\\]'
step() { local n="$1"; shift; local t; t=$(mktemp); echo "=== $(date -u +%FT%TZ) $n"; "$@" 2>&1 | red | tee "$t"; local rc=${PIPESTATUS[0]}
         if (( rc != 0 )) && { (( rc == 255 )) || grep -qE "$SSH_FLAKE" "$t"; } && [[ "$n" != ssh_ready ]]; then
           echo "=== $(date -u +%FT%TZ) $n rc=$rc on the ssh transport: once more after ssh answers"
           ssh_ready && { "$@" 2>&1 | red; rc=${PIPESTATUS[0]}; }
         fi; rm -f "$t"
         echo "=== $(date -u +%FT%TZ) $n rc=$rc"; (( rc == 0 )) || { echo "SETUP STOPPED at $n"; exit "$rc"; }; }
# the launcher returns at RUNNING, before sshd is up (16 Nov 2026: deploy refused 2 s after RUNNING)
ssh_ready() { local i; for i in $(seq 1 ${SSH_TRIES:-40}); do
                ${CMD_GCLOUD:-gcloud} compute ssh "$NAME" --zone "$ZONE" --command true < /dev/null >/dev/null 2>&1 && { echo "ssh up after $i tries"; return 0; }
                ${CMD_SLEEP:-sleep} 15; done; echo "ssh not up after ${SSH_TRIES:-40} tries"; return 1; }
step ssh_ready ssh_ready
step deploy bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
step preflight bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
step gcs bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
# REHEARSAL_BOOTSTRAP_NO_FA3 (rehearsal 3 only, on an e2: the FA3 import needs a GPU) skips FA3 at
# bootstrap alone; the session's own fa3_import step still runs with the wheel's sha256
step bootstrap ${REHEARSAL_BOOTSTRAP_NO_FA3:+env -u FA3_WHEEL_SHA256} bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
