#!/usr/bin/env bash
# Match processes by command line WITHOUT matching the command doing the
# matching, or any of its ancestors.
#
# This pattern has now cost two sessions:
#
#   2026-09-03  `pkill -f 'pip install'` killed the invoking SSH command,
#               because the SSH wrapper's own command line contains the
#               pattern. The build died with it.
#   2026-09-04  `pgrep -f run_probe.py` matched the polling command itself,
#               so a probe that had already died to an Xid 31 MMU fault
#               reported RUNNING for six more minutes. I watched a corpse.
#
# Both are the same bug: a process-matching command whose own command line
# contains the string it matches. Both were "documented" in the runbook -- the
# `[n]vcc` bracket trick is written down twice -- and the rule was still missed
# twice, because it has to be remembered at every call site. So it lives here
# instead, where forgetting is not an option.
#
# The bracket trick alone is insufficient: it defeats *self*-match but not an
# ancestor, e.g. the `bash -c` that gcloud spawns to run the whole poll. This
# walks the full ancestor chain and excludes every pid in it.
#
#   bash scripts/procmatch.sh status  PATTERN   # RUNNING/NOT_RUNNING + pids
#   bash scripts/procmatch.sh pids    PATTERN   # matching pids, one per line
#   bash scripts/procmatch.sh kill    PATTERN [SIGNAL]
#
# Exit status for `status`: 0 if at least one genuine match, 1 if none.
set -uo pipefail

ACTION="${1:?usage: procmatch.sh status|pids|kill PATTERN [SIGNAL]}"
PATTERN="${2:?pattern required}"
SIGNAL="${3:-TERM}"

# Every pid from here up to init. `pgrep` output containing any of these is
# this command seeing itself, never the thing being looked for.
ancestors() {
  local p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    printf '%s\n' "$p"
    p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
  done
}

genuine_pids() {
  local anc; anc="$(ancestors)"
  # Everything spawned as part of THIS invocation shares our process group:
  # ancestors, and equally the forked subshells they spawn for pipelines and
  # command substitution. Those siblings inherit the parent's command line, so
  # they carry the pattern while being nobody's ancestor -- ancestor-walking
  # alone reported `RUNNING pids=1338 5634` where 5634 was a subshell of the
  # poller itself. When the real pid died, that phantom kept the answer at
  # RUNNING for three minutes.
  #
  # The process group is the correct primitive: it is exactly "the job this
  # command is part of". The trade-off is deliberate -- a target started by
  # this same shell and NOT detached shares our pgid and will be missed. For a
  # tool whose whole purpose is checking on `nohup`ed background work over ssh
  # (pgid of its own, ppid 1), that is the right way to be wrong: it errs
  # toward reporting NOT_RUNNING, and a false "not running" is investigated
  # while a false "running" is believed.
  local mypgid; mypgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
  local found; found="$(pgrep -f -- "$PATTERN" 2>/dev/null || true)"
  local pid
  for pid in $found; do
    # skip self and every ancestor
    if printf '%s\n' "$anc" | grep -qx -- "$pid"; then continue; fi
    # skip anything in our own process group (siblings, subshells, the job)
    if [ -n "$mypgid" ]; then
      pidpgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
      [ "$pidpgid" = "$mypgid" ] && continue
    fi
    # Skip anything whose command line is this script (belt and braces: a
    # sibling invocation of procmatch.sh is not the target either).
    # `-o args=`, NOT `-o cmd=`: the latter is GNU-only, and BSD/macOS ps
    # answers it by printing its list of format keywords, which silently
    # turned this filter into a no-op on the workstation where tests run.
    case "$(ps -o args= -p "$pid" 2>/dev/null)" in
      *procmatch.sh*) continue ;;
    esac
    printf '%s\n' "$pid"
  done
}

case "$ACTION" in
  pids)
    genuine_pids
    ;;
  status)
    PIDS="$(genuine_pids | tr '\n' ' ' | sed 's/ *$//')"
    if [ -n "$PIDS" ]; then
      echo "RUNNING pids=$PIDS"
      exit 0
    fi
    echo "NOT_RUNNING"
    exit 1
    ;;
  kill)
    PIDS="$(genuine_pids | tr '\n' ' ' | sed 's/ *$//')"
    if [ -z "$PIDS" ]; then
      echo "NOT_RUNNING (nothing killed)"
      exit 1
    fi
    echo "killing -$SIGNAL: $PIDS"
    # shellcheck disable=SC2086
    kill -"$SIGNAL" $PIDS
    ;;
  *)
    echo "unknown action '$ACTION' (want status|pids|kill)" >&2
    exit 2
    ;;
esac
