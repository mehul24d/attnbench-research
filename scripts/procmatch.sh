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

ACTION="${1:?usage: procmatch.sh status|pids|kill (PATTERN|--file PATH) [SIGNAL]}"

# `--file PATH` reads the pattern from a file instead of argv. This is the
# only construction that removes the problem at the root rather than filtering
# its symptoms: if the pattern never appears in ANY command line, no ancestor,
# sibling, subshell or leftover from a previous invocation can carry it, and
# there is nothing for pgrep to falsely match.
#
# Ancestor and process-group exclusion still run and are still worth having --
# they cover a caller who passes the pattern directly. But they are filters,
# and a filter has edges: on 2026-09-04 a subshell left over from the PREVIOUS
# ssh invocation was in neither our ancestor chain nor our process group, and
# was reported as a live match for three minutes after the real process died.
if [ "${2:-}" = "--file" ]; then
  PATTERN_FILE="${3:?--file needs a path}"
  [ -r "$PATTERN_FILE" ] || { echo "cannot read $PATTERN_FILE" >&2; exit 2; }
  PATTERN="$(head -n1 "$PATTERN_FILE")"
  [ -n "$PATTERN" ] || { echo "$PATTERN_FILE is empty" >&2; exit 2; }
  SIGNAL="${4:-TERM}"
else
  PATTERN="${2:?pattern required}"
  SIGNAL="${3:-TERM}"
fi

# Our own pgid, read ONCE and at top level. The sibling filter below is built
# on it, and an inert filter here does not degrade gracefully -- it reports
# this command's own subshells as live matches, which is the failure this
# whole script exists to prevent. Refuse rather than answer badly. At top
# level rather than inside genuine_pids, because `status` and `kill` call that
# in a command substitution, where an `exit` would end only the subshell and
# the refusal would be reported as NOT_RUNNING.
MYPGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
if [ -z "$MYPGID" ]; then
  echo "procmatch: cannot read own pgid; refusing to answer" >&2
  exit 2
fi

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
  local mypgid="$MYPGID"
  local found; found="$(pgrep -f -- "$PATTERN" 2>/dev/null || true)"
  local pid
  for pid in $found; do
    # skip self and every ancestor
    if printf '%s\n' "$anc" | grep -qx -- "$pid"; then continue; fi

    # ONE ps call for both remaining filters, and an EMPTY result excludes.
    #
    # This is the hole that made all three of this script's own tests fail on
    # Linux on 2026-09-20, while passing on the macOS workstation. The
    # matching pid is typically a subshell this very command forked for a
    # pipeline or a command substitution: it carries our argv, so pgrep
    # returns it, and it exits within milliseconds. Both filters below then
    # ask `ps` about a pid that is already gone, `ps` prints nothing, and
    # nothing is what BOTH tests were written against -- an empty pgid is not
    # equal to ours, and an empty command line does not contain
    # "procmatch.sh". So the phantom fell through both and was reported as a
    # genuine match. `status` answered RUNNING for a pattern matching nothing,
    # and `kill` signalled the phantom ("No such process").
    #
    # Both filters failed OPEN, which is the wrong direction and the script's
    # own docstring says so: it must err toward NOT_RUNNING, because a false
    # "not running" is investigated and a false "running" is believed. A pid
    # `ps` cannot read has exited or is not ours to judge; either way it is
    # not the live process anyone is asking about.
    #
    # `-o args=`, NOT `-o cmd=`: the latter is GNU-only, and BSD/macOS ps
    # answers it by printing its list of format keywords, which silently
    # turned the second filter into a no-op on the workstation where tests run.
    local info; info="$(ps -o pgid=,args= -p "$pid" 2>/dev/null | sed 's/^ *//')"
    [ -n "$info" ] || continue

    # skip anything in our own process group (siblings, subshells, the job)
    local pidpgid="${info%% *}"
    [ "$pidpgid" = "$mypgid" ] && continue

    # Skip anything whose command line is this script (belt and braces: a
    # sibling invocation of procmatch.sh is not the target either).
    case "${info#* }" in
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
