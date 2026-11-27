#!/usr/bin/env bash
# The Mac's driver for P3: one g6.2xlarge in Sydney, all three blocks in one
# session (docs/launch_cards/aws_p3_g6.md). Run at the Mac, from the repository
# root of the commit the launch yes names, under caffeinate:
#
#   caffeinate -i bash scripts/aws_p3_driver.sh
#
# The rehearsal's launch path, unchanged (scripts/aws_launch_session.sh P3): the
# launch policy, the zones 2b, 2c, 2a again every 5 minutes on a capacity
# refusal only for at most 3 hours (scripts/_aws_launch_policy.sh), the security
# group checked against this Mac's address before every create request and never
# re-pointed, the credit gate, the hard-delete schedule. P3's numbers:
#   rate Rs 143.21/h, cap Rs 1,353 for P3 in all (6.3 h x 1.5; this session's
#   cap is that less the aborted session's Rs 143), in-guest halt +473 min (the
#   planned 6.3 h plus 25%), hard delete +503 min (30 min later); worst case
#   Rs 1,201. Expected about 4.7 h (an observation, below), planned 6.3 h.
# The launch deadline is 18:30 UTC on 2 Dec 2026: a window is shortened to end
# by then, and none starts after it (no L4 by then: P3 is dropped). Windows are
# recorded in ~/.attnbench/p3_windows.tsv: a new one only after every earlier one
# ended having created nothing, and never after a launch (one P3 session).
#
# After the launch: SSH, the deploy (a git bundle of HEAD), the bootstrap
# (scripts/aws_bootstrap_instance.sh: the RULER data against its pins, the
# pinned wonderwords and nltk, lightgbm), then the blocks, each started detached
# on the instance (scripts/aws_p3_instance.sh) and polled every minute:
#   b1  preflight .. probe; b2 host_components; b3 device_components .. delay_injection.
# When a block ends, its step directories are pulled as one tar with a sha256 of
# the tar and of every file, both checked here before the next block starts.
# After b2 the P6 gate is read here (scripts/p6_host_effect.py against P1's T3.2d
# host components): d and the verdict are recorded and gate P6 only; block 3 runs
# whatever the verdict (the researcher's ruling of 26 Nov 2026). A block that
# fails, a pull that does not check, or a b2 without all 35 cells stops the
# session: pull what there is, tear down, report.
# While a block runs, the driver is also the prediction store: it witnesses each
# object scripts/aws_p3_gsutil.py spools (pull, sha256, keep, send its copy back,
# write the witness with this Mac's time).
# SSH lost: the driver keeps polling. Each failed poll re-reads whether the
# security group admits this Mac's current address and says so; it never
# re-points the group (the researcher runs sg_repoint.sh). At the pull deadline
# (10 min before the in-guest halt) it pulls what it can and tears down; an
# instance whose driver has gone halts itself 30 min after its block ends
# (aws_p3_instance.sh) and is deleted at +503 min.
# The teardown is the rehearsal's: scripts/aws_teardown_session.sh, retried until
# EC2 reads the instance terminated, then scripts/aws_cleanup_check.sh.
#
# P6 (the researcher's task of 26 Nov 2026 after P3's push, item 2e;
# docs/launch_cards/aws_p6_p5.md): the same driver, with its fixes, for one
# p5.4xlarge (H100) in Sydney:
#
#   caffeinate -i bash scripts/aws_p3_driver.sh P6
#
# P6's numbers: rate Rs 990.2/h, cap Rs 5,940 (4.0 h x 1.5), in-guest halt
# +312 min, hard delete +359 min (worst case Rs 5,925), the zones 2b, 2c, 2a
# (read 26 Nov 2026), its own windows (~/.attnbench/p6_windows.tsv), the same
# deadline. Its blocks are P3's on configs/t31/p6_session.json, with block 1's
# image and dense-backend check first (scripts/p6_backend_check.py; a failure
# stops the session) and block 3 ending at postflight; no gate after block 2
# (P3's gate was P6's). The record is attnbench_backups/aws_p6_p5_<UTC>/.
#
# P6 on Spot (the researcher's task of 27 Nov 2026; docs/launch_cards/aws_p6_p5_spot.md):
#
#   caffeinate -i bash scripts/aws_p3_driver.sh P6 spot
#
# The launcher's spot-first mode: each pass of the window asks for a one-time
# Spot p5.4xlarge (bid at most the on-demand $8.944/h) in 2b, 2c and 2a, then
# on-demand in 2b, 2c and 2a; no Spot request when the Spot quota reads under
# 16 vCPU. On a Spot instance a block's poll also reads the instance's
# interruption notice (IMDSv2 spot/instance-action), and a lost poll reads the
# instance's state reason. An interruption (the notice, or the instance
# terminated with Server.SpotInstanceTermination, or a pull cut by either)
# discards the block it hit in full, as P4's Spot blocks were (plan Appendix A,
# 15 Nov 2026: "An interrupted block is still discarded in full and rerun on a
# new instance"). The driver then: tears the instance down; counts what it
# cost at the rate, from launch to the teardown, rounded up; relaunches at
# most twice, in block 1's zone (P4's rule: later blocks run in b1's zone),
# Spot then on-demand, for a window of up to 60 min, before the deadline; sets
# the new instance's hard delete to what is left of P6's Rs 5,940 in all
# (never more than +359 min, its halt 47 min earlier, never more than +312),
# and stops if that halt cannot fit the blocks left (the card's minutes);
# boots, deploys and bootstraps it as the first; requires it to match the
# first instance (type, zone, GPU and driver, CPU model, vCPUs); sends the
# finished blocks back (one tar, every file's sha256 checked on the instance);
# runs the image and dense-backend check (block 1's abort gate) again before a
# later block; and reruns the interrupted block from its start. The Mac's
# prediction store of an interrupted attempt is set aside with its logs
# (attempt_N/), so a rerun block 3 commits its own prediction. A third
# interruption stops the session with what it has.
#
# Test mode (tests/test_aws_p3_driver.py, the fake remote): P3_DRY=1 lets P3_*
# variables point the AWS side, the paths and the clock at fakes. Without
# P3_DRY=1 any P3_* variable is refused.
set -uo pipefail

# Never edit a running bash script (it reads itself as it goes): run a copy.
if [[ -z "${P3_DRIVER_COPY:-}" ]]; then
  WT0="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  COPY="$(mktemp -t p3_driver)" || exit 1
  cp "${BASH_SOURCE[0]}" "$COPY" || exit 1
  exec env P3_DRIVER_COPY="$COPY" P3_DRIVER_WT="$WT0" bash "$COPY" "$@"
fi
WT="$P3_DRIVER_WT"
PAIRING="${1:-P3}"
case "$PAIRING" in P3|P6) ;; *) echo "REFUSING: the pairing is '$PAIRING', not P3 or P6; nothing launched" >&2; exit 1 ;; esac
SPOT=0
case "${2:-}" in
  "") ;;
  spot) [[ "$PAIRING" == P6 ]] || { echo "REFUSING: spot is for P6 only; nothing launched" >&2; exit 1; }; SPOT=1 ;;
  *) echo "REFUSING: the second argument is '${2}', not spot; nothing launched" >&2; exit 1 ;;
esac

REC_BASE="$HOME/Desktop/research/attnbench_backups"
AWS_ENV="$HOME/.attnbench/aws_env.sh"
WINDOWS="$HOME/.attnbench/$(tr 'A-Z' 'a-z' <<<"$PAIRING")_windows.tsv"
P1_HOST="$HOME/Desktop/research/attnbench_backups/bucket_20261117/attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components"
VPY="$HOME/Desktop/research/attnbench_scaffold/.venv/bin/python"
POLL_S=60; SSH_TRIES=40; SSH_WAIT_S=15; TEARDOWN_WAIT_S=30
LAUNCH=(bash scripts/aws_launch_session.sh "$PAIRING"); TEARDOWN=scripts/aws_teardown_session.sh
CLEANUP=scripts/aws_cleanup_check.sh; BOOTSTRAP=scripts/aws_bootstrap_instance.sh
END_EPOCH=1796236200                 # 2026-12-02T18:30:00Z, the launch deadline
NOW_EPOCH="$(date -u +%s)"
HALT_MIN=473; DELETE_MIN=503; RATE=143.21
# The cap: Rs 1,353 for P3 in all (the researcher's relaunch yes of 26 Nov 2026),
# less what the aborted session of 25 Nov 2026 cost (about Rs 143: 59 min
# running, then its volume while stopped). The worst case, 503 min at the rate,
# is Rs 1,201, inside it.
P3_TOTAL_CAP=1353; ABORTED_INR=143; CAP=$(( P3_TOTAL_CAP - ABORTED_INR ))
ITYPE_WANT=g6.2xlarge; REC_NAME=aws_p3_g6; AZS="ap-southeast-2b ap-southeast-2c ap-southeast-2a"
BLOCKS_LAST="12_delay_injection"
CRED_NEED_S=41400                    # 11.5 h: the window and the run, 3 h + 503 min = 11.4 h
if [[ "$PAIRING" == P6 ]]; then
  # P6 (docs/launch_cards/aws_p6_p5.md): the card rate of Appendix A, 21 Nov 2026
  # (p5.4xlarge at $8.944/h, Sydney's public price file of 20 Nov 2026, and 200 GB
  # gp3, x 1.15 x Rs 95.99); planned 4.0 h, cap 4.0 h x rate x 1.5. The worst
  # case, 359 min at the rate, is Rs 5,925, inside it. Zones offering it in
  # Sydney, read 26 Nov 2026: 2b, 2c, 2a.
  HALT_MIN=312; DELETE_MIN=359; RATE=990.2; CAP=5940
  ITYPE_WANT=p5.4xlarge; REC_NAME=aws_p6_p5
  BLOCKS_LAST=""                     # block 3 ends at postflight (no delay injection)
  CRED_NEED_S=34200                  # 9.5 h: 3 h + 359 min = 9.0 h, and a margin
fi
# Spot (P6 only): the bid is the on-demand price (p5.4xlarge in Sydney, the price
# file of 20 Nov 2026), so the rate above bounds a Spot hour too.
SPOT_MAX_USD=8.944
MAX_RELAUNCHES=2; RELAUNCH_WINDOW_MAX=60
# A relaunched instance's halt must fit the blocks left, at the card's minutes:
# boot, deploy, bootstrap and the backend check 20; then P3's blocks on the same
# AMD host (26 Nov 2026): b1 35, b2 81, b3 99.
SETUP_MIN=20; BLOCK_MIN_b1=35; BLOCK_MIN_b2=81; BLOCK_MIN_b3=99
PULL_MARGIN_MIN=10                   # the last pull starts 10 min before the halt
# The final re-pull of every step directory, after all three blocks were pulled
# and checked, is off unless P3_DRIVER_FINAL_REPULL=1 (26 Nov 2026: it re-sends
# every block, about 8 GB, and in P3's run its scp was refused). A failed block
# still pulls whatever there is.
FINAL_REPULL="${P3_DRIVER_FINAL_REPULL:-0}"
PUSH_EVERY_S=600                     # while a copy runs, the halt is pushed back every 10 min
if [[ "${P3_DRY:-}" == 1 ]]; then    # the fake remote, never AWS
  REC_BASE="$P3_REC_BASE"; AWS_ENV="$P3_AWS_ENV"; WINDOWS="$P3_WINDOWS"; P1_HOST="$P3_P1_HOST"
  VPY="$P3_VPY"; POLL_S="$P3_POLL_S"; SSH_TRIES=3; SSH_WAIT_S=1; TEARDOWN_WAIT_S=1
  LAUNCH=(bash "$P3_LAUNCH" "$PAIRING"); TEARDOWN="$P3_TEARDOWN"; CLEANUP="$P3_CLEANUP"; BOOTSTRAP="$P3_BOOTSTRAP"
  NOW_EPOCH="${P3_NOW_EPOCH:-$NOW_EPOCH}"
  [[ -n "${P3_PULL_DEADLINE_S:-}" ]] && PULL_DEADLINE_S="$P3_PULL_DEADLINE_S"
  PUSH_EVERY_S="${P3_PUSH_EVERY_S:-$PUSH_EVERY_S}"
  CAP="${P3_CAP:-$CAP}"
elif env | grep '^P3_' | grep -qv '^P3_DRIVER_'; then
  echo "REFUSING: P3_* variables are set but P3_DRY is not 1; nothing launched" >&2; exit 1
fi

UTC="$(date -u +%Y%m%dT%H%MZ)"
REC="$REC_BASE/${REC_NAME}_$UTC"
[[ -e "$REC" ]] && { echo "REFUSING: $REC exists" >&2; exit 1; }
mkdir -p "$REC"
cp "$P3_DRIVER_COPY" "$REC/driver.sh"
source "$AWS_ENV"
# The log is filtered: the account, any 12-digit number, ARNs and role sessions
# (the organization's denials name its policy, the role and the session; 26 Nov 2026).
exec > >(sed -u -E "s/${ATTNBENCH_AWS_ACCOUNT}/<account>/g; s/(^|[^0-9a-f])[0-9]{12}([^0-9]|$)/\1<id>\2/g; s#arn:aws[A-Za-z0-9-]*:[^[:space:]\"',)]*#arn:<masked>#g; s#assumed-role/[^[:space:]\"',)]+#assumed-role/<masked>#g" >> "$REC/driver.log") 2>&1
status() { echo "$(date -u +%FT%TZ) $*"; echo "$(date -u +%FT%TZ) $*" > "$REC/status.txt"; }
refuse() { status "REFUSED $*; nothing launched"; exit 1; }

# --- the launch windows ---------------------------------------------------------
(( NOW_EPOCH < END_EPOCH )) || refuse "18:30 UTC on 2 Dec 2026 has passed: no $([[ $PAIRING == P6 ]] && echo H100 || echo L4) by then, so $PAIRING is dropped"
LEFT_MIN=$(( (END_EPOCH - NOW_EPOCH) / 60 ))
WINDOW_MIN=$(( LEFT_MIN < 180 ? LEFT_MIN : 180 ))
(( WINDOW_MIN >= 5 )) || refuse "less than 5 minutes are left before 18:30 UTC on 2 Dec 2026"
mkdir -p "$(dirname "$WINDOWS")"; touch "$WINDOWS"
# The ledger: "start<TAB>UTC<TAB>REC" when a window begins, "end<TAB>UTC<TAB>REC<TAB>HOW" when it ends.
# One P3 session: a launched window refuses the next, unless a dated line
# "aborted<TAB>UTC<TAB>REC<TAB>why" names its record (written on the researcher's
# yes to relaunch; the aborted session of 25 Nov 2026), and then only one more.
LAUNCHED_LIVE="$(awk -F'\t' '$1 == "end" && $4 ~ /^LAUNCHED/ {l[$3] = 1} $1 == "aborted" {a[$3] = 1}
  END {for (r in l) if (!(r in a)) n++; print n + 0}' "$WINDOWS")"
(( LAUNCHED_LIVE == 0 )) || refuse "a $PAIRING instance has already been launched ($WINDOWS): one $PAIRING session"
NABORTED="$(awk -F'\t' '$1 == "aborted" {n++} END {print n + 0}' "$WINDOWS")"
(( NABORTED <= 1 )) || refuse "$NABORTED $PAIRING sessions are recorded aborted ($WINDOWS): a further relaunch needs its own yes and code"
OPEN="$(awk -F'\t' '$1 == "start" {s[$3] = $2} $1 == "end" && $4 == "NOTHING_CREATED" {delete s[$3]}
  $1 == "end" && $4 ~ /^LAUNCHED/ {l[$3] = 1} $1 == "aborted" && ($3 in l) {delete s[$3]} END {for (r in s) print s[r]}' "$WINDOWS")"
[[ -z "$OPEN" ]] || refuse "a window started at $OPEN has no end line saying it created nothing ($WINDOWS)"
win_end() { printf 'end\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$REC" "$1" >> "$WINDOWS"; }

unset AWS_SESSION_AZ AWS_SUBNET_ID
export AWS_SESSION_REGION=ap-southeast-2 AWS_SESSION_AZS="$AZS"
export AWS_RATE_INR_HR=$RATE AWS_COST_CAP_INR=$CAP AWS_HALT_MINUTES=$HALT_MIN AWS_MAX_RUN_MINUTES=$DELETE_MIN
export AWS_LAUNCH_WINDOW_MINUTES=$WINDOW_MIN
export AWS_AMI_ID=ami-096c5dc30364c9041
(( SPOT )) && export AWS_PROVISIONING=spot-first AWS_SPOT_MAX_PRICE_USD=$SPOT_MAX_USD
HALT_MIN_FIRST=$HALT_MIN; DELETE_MIN_FIRST=$DELETE_MIN
R=(--profile attnbench --region ap-southeast-2)
KH="$REC/known_hosts"
SSH_OPTS=(-i "$AWS_SSH_KEY" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile="$KH" -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

cd "$WT" || exit 1
[[ -z "$(git status --porcelain)" ]] || refuse "the tree is not clean"
COMMIT="$(git rev-parse HEAD)"
[[ "$PAIRING" == P6 ]] || { [[ -d "$P1_HOST" ]] && (( $(ls "$P1_HOST"/*/bare_prices.json 2>/dev/null | wc -l) == 35 )); } \
  || refuse "P1's T3.2d host components (35 cells) are not at $P1_HOST: the gate could not be read"

# The SSH source: does the security group admit this Mac's current address?
# 0 yes; 1 the address could not be read; 2 the group could not be read; 3 no.
sg_admits() {
  local ip cidrs
  ip="$(curl -s -m 10 https://checkip.amazonaws.com | tr -d '[:space:]')"
  [[ -n "$ip" ]] || return 1
  cidrs="$(command aws ec2 describe-security-groups "${R[@]}" --group-ids "$AWS_SECURITY_GROUP_ID" \
    --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)" || return 2
  [[ " $(tr -s '[:space:]' ' ' <<<"$cidrs") " == *" $ip/32 "* ]] || return 3
}
sg_admits; SG=$?
case $SG in
  0) ;;
  1) refuse "SSH preflight: could not read this Mac's public address (network?)" ;;
  2) refuse "SSH preflight: could not read the security group (run aws login --profile attnbench, then start again)" ;;
  *) refuse "SSH preflight: the security group does not admit this Mac's address (re-point it with sg_repoint.sh)" ;;
esac
status "SSH preflight passed (the security group admits this Mac's current address)"

# The credentials (the researcher's task of 26 Nov 2026, item 4). `aws login`
# credentials rotate every 15 min inside a session that ends at most 12 h after
# the login (the AWS CLI user guide, "Login for AWS local development using
# console credentials"), so the window and the run (3 h + 503 min = 11.4 h) need
# a login under 30 min old. The launch command stamps the login time
# (ATTNBENCH_LOGIN_AT); a session expiry in the login cache more than 1 h ahead,
# if it holds one, is used when earlier. Only expiry timestamps are read from the
# cache; nothing else in it is printed. Logged at the start of every window; a
# warning under 11.5 h, not a refusal (the hard delete still bounds the cost).
LOGIN_CACHE="${AWS_LOGIN_CACHE_DIRECTORY:-$HOME/.aws/login/cache}"
[[ "${P3_DRY:-}" == 1 ]] && LOGIN_CACHE="$P3_LOGIN_CACHE"
CRED="$("$VPY" -I - "$LOGIN_CACHE" "${ATTNBENCH_LOGIN_AT:-}" "$NOW_EPOCH" <<'PY' 2>/dev/null
import glob, json, os, sys
from datetime import datetime, timezone
cache, login_at, now = sys.argv[1], sys.argv[2], int(sys.argv[3])

def stamps(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "expir" in k.lower() and isinstance(v, (str, int, float)):
                yield v
            else:
                yield from stamps(v)
    elif isinstance(o, list):
        for v in o:
            yield from stamps(v)

def epoch(v):
    if isinstance(v, (int, float)):
        return v / 1000 if v > 1e11 else float(v)
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()

found = []
for f in glob.glob(os.path.join(cache, "*.json")):
    try:
        found += [e for e in map(epoch, stamps(json.load(open(f)))) if e]
    except (OSError, ValueError):
        pass
cands = []
if login_at.isdigit():
    cands.append((int(login_at) + 12 * 3600, "aws login at %s + 12 h" % datetime.fromtimestamp(
        int(login_at), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
longer = [e for e in found if e > now + 3600]
if longer:
    cands.append((int(max(longer)), "the login cache's session expiry"))
if not cands:
    print("unknown\tno login time (ATTNBENCH_LOGIN_AT) and no session expiry in the login cache")
else:
    end, src = min(cands)
    print("%d\t%s" % (end, src))
PY
)"
CRED_END="${CRED%%$'\t'*}"; CRED_SRC="${CRED#*$'\t'}"
if [[ ! "$CRED_END" =~ ^[0-9]+$ ]]; then
  printf 'window_start\t%s\tunknown\t%s\n' "$(date -u +%FT%TZ)" "${CRED_SRC:-not read}" >> "$REC/credentials.tsv"
  status "WARN CREDENTIALS the session's end is not known (${CRED_SRC:-not read}): start the window with the launch command, which stamps the login time"
else
  CRED_LEFT_S=$(( CRED_END - NOW_EPOCH ))
  CRED_H="$(awk -v s="$CRED_LEFT_S" 'BEGIN {printf "%.1f", s / 3600}')"
  CRED_UTC="$(date -u -r "$CRED_END" +%FT%TZ)"
  printf 'window_start\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$CRED_UTC" "$CRED_SRC" >> "$REC/credentials.tsv"
  status "CREDENTIALS the aws login session ends by $CRED_UTC, $CRED_H h from now ($CRED_SRC)"
  CRED_NEED_H="$(awk -v s="$CRED_NEED_S" 'BEGIN {printf "%.1f", s / 3600}')"
  (( CRED_LEFT_S >= CRED_NEED_S )) || status "WARN CREDENTIALS end in $CRED_H h, under $CRED_NEED_H h: the window and the run need that, so the teardown could fail on expired credentials (the instance then halts at +$HALT_MIN and is deleted at +$DELETE_MIN). Stop with Ctrl-C now, run aws login --profile attnbench, and start the window again"
fi

printf 'start\t%s\t%s\n' "$(date -u +%FT%TZ)" "$REC" >> "$WINDOWS"
status "LAUNCHING $PAIRING commit ${COMMIT:0:7} ($ITYPE_WANT$( (( SPOT )) && echo ", Spot then on-demand in each pass, bid at most \$$SPOT_MAX_USD/h"); window ${WINDOW_MIN} min, a pass every 5 min; halt +$HALT_MIN, delete +$DELETE_MIN, cap Rs $CAP; progress in launcher.log)"
# launch LOG: one launcher run, its output in LOG; sets LRC ID LZONE LRESULT PASSES NREFUSED NEW_MARKET.
launch() {
  local lout
  echo launch | "${LAUNCH[@]}" 2>&1 | sed -u -E "s/${ATTNBENCH_AWS_ACCOUNT}/<account>/g" | tee "$1"
  LRC=${PIPESTATUS[1]}
  lout="$(cat "$1")"
  ID="$(sed -n 's/^INSTANCE_ID=//p' <<<"$lout")"
  LZONE="$(sed -n 's/^LAUNCH_ZONE=//p' <<<"$lout" | tail -1)"
  LRESULT="$(sed -n 's/^LAUNCH_RESULT=//p' <<<"$lout" | tail -1)"
  PASSES="$(sed -n 's/^WINDOW_PASSES=//p' <<<"$lout" | tail -1)"
  NREFUSED="$(sed -n 's/^CAPACITY_REFUSALS=//p' <<<"$lout" | tail -1)"
  NEW_MARKET="$(sed -n 's/^LAUNCH_MARKET=//p' <<<"$lout" | tail -1)"
  [[ "$NEW_MARKET" == spot ]] || NEW_MARKET=on-demand
}
launch "$REC/launcher.log"
if [[ $LRC -ne 0 || ! "$ID" =~ ^i- ]]; then
  if [[ "$LRESULT" == SSH_SOURCE_NOT_ADMITTED ]]; then
    win_end NOTHING_CREATED
    status "STOPPED the security group no longer admits this Mac's address (not re-pointed); passes ${PASSES:-?}, capacity refusals ${NREFUSED:-?}; nothing created"
  elif [[ "$LRESULT" == EVERY_ZONE_REFUSED_FOR_CAPACITY && -z "$ID" ]]; then
    win_end NOTHING_CREATED
    status "NOT LAUNCHED the window ended: passes ${PASSES:-?}, capacity refusals ${NREFUSED:-?}, each confirmed; nothing created"
  else
    win_end "UNCLEAR rc=$LRC"
    status "FAIL launcher rc=$LRC id='${ID}' passes ${PASSES:-?}, capacity refusals ${NREFUSED:-?}${LRESULT:+ ($LRESULT)}"
  fi
  bash "$CLEANUP"; exit 1
fi
win_end "LAUNCHED $ID"
MARKET="$NEW_MARKET"; FIRST_ZONE="$LZONE"
ATTEMPT=1; RELAUNCHES=0; SPENT_INR=0; NOTICE_SEEN=0
started() {    # a new instance: its times and its record
  LAUNCH_EPOCH="$(date -u +%s)"
  DELETE_AT=$(( LAUNCH_EPOCH + DELETE_MIN * 60 ))   # the hard delete, as the launcher schedules it
  PULL_DEADLINE=$(( LAUNCH_EPOCH + ${PULL_DEADLINE_S:-$(( (HALT_MIN - PULL_MARGIN_MIN) * 60 ))} ))
  echo "$ID" > "$REC/instance_id.txt"
  status "LAUNCHED $ID ($MARKET) in zone ${LZONE:-unknown} on pass ${PASSES:-?} (capacity refusals before it: ${NREFUSED:-0}); halt $(date -u -r $(( LAUNCH_EPOCH + HALT_MIN * 60 )) +%FT%TZ 2>/dev/null), delete $(date -u -r $(( LAUNCH_EPOCH + DELETE_MIN * 60 )) +%FT%TZ 2>/dev/null) (about)"
}
started

TORN=0
teardown() {   # retried until EC2 reads the instance back terminated
  (( TORN )) && return; TORN=1
  status "TEARDOWN ($1)"
  local n st
  for n in 1 2 3 4 5; do
    AWS_RATE_INR_HR=$RATE bash "$TEARDOWN" "$ID" ap-southeast-2 "$REC/teardown_$n"
    st="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)"
    [[ "$st" == terminated || "$st" == shutting-down ]] && break
    status "TEARDOWN attempt $n left the instance '$st'; retrying in ${TEARDOWN_WAIT_S} s"; sleep "$TEARDOWN_WAIT_S"
  done
  [[ "$st" == terminated || "$st" == shutting-down ]] || status "TEARDOWN FAILED after 5 attempts: terminate $ID by hand NOW"
  bash "$CLEANUP"; echo "cleanup_check rc=$?"
}
stop() { teardown "$1"; status "$2"; exit 1; }
trap 'teardown "driver interrupted"; status "STOPPED driver interrupted; torn down"; exit 1' INT TERM HUP

rssh() { ssh "${SSH_OPTS[@]}" "$H" "$@"; }
boot() {       # running, SSH, the deploy, the bootstrap, the host's fingerprint
  local ok i
  status "WAITING for running + SSH"
  command aws ec2 wait instance-running "${R[@]}" --instance-ids "$ID"
  DNS="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
  IZONE="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)"
  ITYPE="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].InstanceType' --output text)"
  printf 'launcher_zone\t%s\ninstance_zone\t%s\ninstance_type\t%s\nwindow_passes\t%s\ncapacity_refusals\t%s\nmarket\t%s\n' \
    "$LZONE" "$IZONE" "$ITYPE" "${PASSES:-}" "${NREFUSED:-}" "$MARKET" > "$REC/placement.txt"
  [[ "$ITYPE" == "$ITYPE_WANT" ]] || stop "instance type '$ITYPE'" "FAIL the instance is not a $ITYPE_WANT"
  H="ubuntu@$DNS"
  ok=0
  for i in $(seq 1 "$SSH_TRIES"); do rssh true 2>/dev/null && { ok=1; break; }; sleep "$SSH_WAIT_S"; done
  (( ok )) || stop "no SSH after the boot wait" "FAIL no SSH"
  status "SSH up at +$(( ($(date -u +%s) - LAUNCH_EPOCH) / 60 )) min"

  status "DEPLOYING bundle ${COMMIT:0:7}"
  BUNDLE="${TMPDIR:-/tmp}/p3_$UTC.bundle"
  [[ -e "$BUNDLE" ]] || git bundle create "$BUNDLE" HEAD >/dev/null 2>&1
  scp "${SSH_OPTS[@]}" "$BUNDLE" "$H:~/" || stop "scp failed" "FAIL deploy"
  rssh "set -e; cd ~/attnbench_scaffold; git fetch -q ~/p3_$UTC.bundle HEAD; git checkout -q --detach $COMMIT; test -z \"\$(git status --porcelain)\"; git rev-parse HEAD; nvidia-smi -L; nproc; free -g | head -2" \
    > "$REC/deploy.txt" 2>&1 || { cat "$REC/deploy.txt"; stop "checkout failed" "FAIL deploy"; }
  cat "$REC/deploy.txt"
  status "BOOTSTRAP: $BOOTSTRAP"
  AWS_SSH_KEY="$AWS_SSH_KEY" AWS_KNOWN_HOSTS="$KH" ATTNBENCH_PYTHON="$VPY" PYTHONDONTWRITEBYTECODE=1 \
    bash "$BOOTSTRAP" "$H" > "$REC/bootstrap.txt" 2>&1; BRC=$?
  cat "$REC/bootstrap.txt"
  [[ $BRC -eq 0 && "$(tail -1 "$REC/bootstrap.txt")" == BOOTSTRAP_OK ]] || stop "bootstrap failed rc=$BRC" "FAIL bootstrap"
  status "BOOTSTRAP ok at +$(( ($(date -u +%s) - LAUNCH_EPOCH) / 60 )) min"
  # The host every later instance must match (P4's rule, 15 Nov 2026): type, zone,
  # GPU and driver, CPU model, vCPUs.
  { printf 'instance_type\t%s\nzone\t%s\n' "$ITYPE" "$IZONE"
    rssh "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1; grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//'; nproc" 2>/dev/null \
      | sed 's/^/host\t/'; } > "$REC/host.txt"
}
boot
cp "$REC/host.txt" "$REC/host_first.txt"

# --- the store: witness what the instance spooled ---------------------------------
mkdir -p "$REC/store"
witness_store() {
  local list o rel dest want got at
  list="$(rssh 'cd ~/p3_store 2>/dev/null || exit 0; find . -name "*.request" | sort | while read -r r; do o="${r%.request}"; [ -e "$o.witness" ] || echo "${o#./}"; done' 2>/dev/null)" || return 1
  for rel in $list; do
    if [[ ! "$rel" =~ ^p3-mac/[A-Za-z0-9._/-]+$ || "$rel" == *..* ]]; then
      status "STORE refused the path '$rel'; not witnessed"; continue
    fi
    dest="$REC/store/$rel"
    want="$(rssh "cat ~/p3_store/$rel.request" 2>/dev/null | sed -n 's/.*"sha256": *"\([0-9a-f]\{64\}\)".*/\1/p')"
    [[ -n "$want" ]] || { status "STORE could not read the request for $rel; retried next poll"; continue; }
    if [[ -e "$dest" ]]; then
      got="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
      [[ "$got" == "$want" ]] || { status "STORE $rel is already held here with other bytes: not overwritten, not witnessed"; continue; }
    else
      mkdir -p "$(dirname "$dest")"
      scp "${SSH_OPTS[@]}" "$H:~/p3_store/$rel" "$dest.part" >/dev/null 2>&1 || { rm -f "$dest.part"; status "STORE pull of $rel failed; retried next poll"; continue; }
      got="$(shasum -a 256 "$dest.part" | cut -d' ' -f1)"
      [[ "$got" == "$want" ]] || { rm -f "$dest.part"; status "STORE $rel came with sha256 $got, not the request's; not witnessed"; continue; }
      mv "$dest.part" "$dest"
    fi
    at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"         # the Mac holds the whole object at this time
    scp "${SSH_OPTS[@]}" "$dest" "$H:~/p3_store/$rel.mac" >/dev/null 2>&1 || { status "STORE return of $rel failed; retried next poll"; continue; }
    rssh "printf '{\"sha256\": \"%s\", \"received_utc\": \"%s\"}\n' $got $at > ~/p3_store/$rel.witness.part && mv ~/p3_store/$rel.witness.part ~/p3_store/$rel.witness" \
      || { status "STORE witness of $rel not written; retried next poll"; continue; }
    printf '%s\t%s\t%s\n' "$at" "$rel" "$got" >> "$REC/store_witness.tsv"
    status "STORE witnessed $rel at $at (sha256 ${got:0:12})"
  done
}

# --- pulls: one tar per block, its sha256 and every file's, checked here ------------
mkdir -p "$REC/session"
# Error text from ssh and scp, kept for the record without addresses or ids.
mask() { sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g; s/ec2-[0-9-]+[^ :]*/<host>/g; s/ip-[0-9-]+/<host>/g;
                 s/arn:[^ ]*/arn:<masked>/g; s/i-[0-9a-f]{8,17}/i-<id>/g'; }
# push_halt WHY: as a pull starts, the instance's pending halt goes back to now +
# 30 min, never past the planned halt or the delete (aws_p3_instance.sh
# push-halt; 26 Nov 2026: block 3's pull ended 6 min before its idle halt, and
# systemd refuses new logins in a halt's last 5 min). A failure is a warning.
# While the copy runs it is pushed back again every PUSH_EVERY_S, so a copy of
# any length keeps the halt over 20 min away (until the planned halt).
push_halt() {
  local out
  if out="$(rssh "cd ~/attnbench_scaffold && ATTNBENCH_P3_HALT_MIN=$HALT_MIN bash scripts/aws_p3_instance.sh push-halt $DELETE_AT" 2>&1)"; then
    status "HALT pushed back ($1): ${out:-unchanged}"
  else
    status "WARN HALT not pushed back ($1): $(tail -1 <<<"$out" | mask | cut -c1-160)"
  fi
}
pull() {   # pull NAME DIR...: 0 when every file checks
  local name="$1"; shift
  local d="$REC/pull/$name" sum
  mkdir -p "$d"
  push_halt "pull $name"
  sum="$(rssh "set -e; cd ~/p3_session; for x in $*; do test -e \"\$x\"; done; find $* manifest.json -type f | LC_ALL=C sort > ~/p3/$name.files; tr '\n' '\0' < ~/p3/$name.files | xargs -0 sha256sum > ~/p3/$name.sha256; tar cf ~/p3/$name.tar -T ~/p3/$name.files; sha256sum ~/p3/$name.tar | cut -d' ' -f1" 2>"$d/remote.err")" \
    || { status "PULL $name: the instance could not pack $* ($(tail -1 "$d/remote.err"))"; return 1; }
  local kp rc
  ( s=""; trap '[[ -n "$s" ]] && kill "$s" 2>/dev/null; exit 0' TERM
    while :; do sleep "$PUSH_EVERY_S" & s=$!; wait "$s"; push_halt "pull $name, copying"; done ) </dev/null &
  kp=$!
  scp "${SSH_OPTS[@]}" "$H:~/p3/$name.tar" "$H:~/p3/$name.sha256" "$d/" 2>&1 >/dev/null | mask > "$d/scp.err"
  rc=${PIPESTATUS[0]}
  kill "$kp" 2>/dev/null; wait "$kp" 2>/dev/null
  [[ "$rc" == 0 ]] \
    || { status "PULL $name: scp failed ($(tail -1 "$d/scp.err" | cut -c1-160))"; return 1; }
  [[ "$(shasum -a 256 "$d/$name.tar" | cut -d' ' -f1)" == "$sum" ]] \
    || { status "PULL $name: the tar's sha256 differs from the instance's"; return 1; }
  tar xf "$d/$name.tar" -C "$REC/session" || { status "PULL $name: tar failed"; return 1; }
  (cd "$REC/session" && shasum -a 256 -c "$d/$name.sha256" > "$d/check.txt" 2>&1) \
    || { status "PULL $name: $(grep -vc ': OK$' "$d/check.txt") file(s) do not check"; return 1; }
  status "PULLED $name: $(wc -l < "$d/$name.sha256" | tr -d ' ') files, every sha256 checked"
}
pull_logs() {
  scp "${SSH_OPTS[@]}" "$H:~/p3/*.log" "$H:~/p3/*.exit" "$REC/" >/dev/null 2>&1 || true
  [[ "$PAIRING" == P6 ]] && { scp "${SSH_OPTS[@]}" "$H:~/p3/b1_backends.json" "$REC/" >/dev/null 2>&1 || true; }
  return 0
}

BLOCK_DIRS_b1="01_preflight 02_link 03_dense_select 04_probe"
BLOCK_DIRS_b2="05_host_components"
BLOCK_DIRS_b3="06_device_components 07_predict 08_hold 09_ground_truth 10_verify 11_postflight${BLOCKS_LAST:+ $BLOCKS_LAST}"
block_dirs() { eval "echo \$BLOCK_DIRS_$1"; }
whatever_there_is() {   # pull every step directory that exists, as a last act
  local have
  have="$(rssh 'cd ~/p3_session 2>/dev/null && ls -d [0-9][0-9]_* 2>/dev/null' 2>/dev/null | tr '\n' ' ')"
  [[ -n "$have" ]] && pull final $have
  pull_logs
}

# A Spot instance's interruption: the notice seen at a poll, or the instance
# shutting down or terminated by EC2 with Server.SpotInstanceTermination.
spot_interrupted() {
  [[ "$MARKET" == spot ]] || return 1
  (( NOTICE_SEEN )) && return 0
  local st why
  st="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)" || return 1
  why="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" \
    --query 'Reservations[0].Instances[0].StateReason.Code' --output text 2>/dev/null)" || return 1
  [[ ( "$st" == shutting-down || "$st" == terminated ) && "$why" == Server.SpotInstanceTermination ]]
}
# The poll's IMDSv2 read of the interruption notice (Spot only): its body, if any,
# follows the marker line.
SPOT_POLL='; echo ---SPOT---; t=$(curl -s -f -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60") && curl -s -f -m 2 -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/spot/instance-action; true'

run_block() {   # run_block B: 0 passed and pulled; 1 failed; 2 SSH lost to the deadline; 3 Spot interrupted
  local b="$1" out ex fails=0 now mins st said_stopped=0 extra="" notice
  [[ "$MARKET" == spot ]] && extra="$SPOT_POLL"
  status "BLOCK $b START ($(block_dirs "$b"))"
  rssh "nohup bash -c 'cd ~/attnbench_scaffold && ATTNBENCH_P3_MARKET=$MARKET ATTNBENCH_P3_PAIRING=$PAIRING ATTNBENCH_P3_HALT_MIN=$HALT_MIN bash scripts/aws_p3_instance.sh $b ~/p3_session' >/dev/null 2>&1 < /dev/null &" \
    || { status "BLOCK $b could not be started over SSH"; spot_interrupted && return 3; return 1; }
  while :; do
    sleep "$POLL_S"
    now="$(date -u +%s)"; mins=$(( (now - LAUNCH_EPOCH) / 60 ))
    if out="$(rssh "cat ~/p3/$b.exit 2>/dev/null; echo ---; grep -E '^=====|^-----|BLOCK ENDED|STOPPED|REFUSING|P3_HALT_WARN' ~/p3/$b.log 2>/dev/null | tail -2$extra" 2>&1)"; then
      (( fails )) && status "SSH back after $fails failed poll(s)"
      fails=0
      notice="$(sed -n '/^---SPOT---$/,$p' <<<"$out" | grep -o '"action"[^,}]*' | head -1 || true)"
      [[ -n "$notice" ]] && { NOTICE_SEEN=1; status "SPOT INTERRUPTION NOTICE in $b at +${mins}min ($notice)"; }
      witness_store || true
      ex="$(head -1 <<<"$out" | cut -f1 | grep -E '^[0-9]+$' || true)"
      status "POLL $b +${mins}min exit='${ex}' :: $(sed '/^---SPOT---$/,$d' <<<"$out" | tr '\n' ' ' | cut -c1-260)"
      if [[ -n "$ex" ]]; then
        pull_logs
        [[ "$ex" == 0 ]] || { status "BLOCK $b FAILED exit $ex"; return 1; }
        pull "$b" $(block_dirs "$b") || { spot_interrupted && { status "SPOT INTERRUPTED during the pull of $b"; return 3; }; return 1; }
        return 0
      fi
      (( NOTICE_SEEN )) && return 3
    else
      fails=$(( fails + 1 ))
      sg_admits; SG=$?
      case $SG in
        0) why="the group still admits this Mac's address (network?)" ;;
        1) why="this Mac's public address could not be read (network down?)" ;;
        2) why="the security group could not be read (aws login --profile attnbench?)" ;;
        *) why="the group no longer admits this Mac's address: re-point it with sg_repoint.sh (the driver never does)" ;;
      esac
      # The instance's own state (26 Nov 2026): a stopped instance is said at once.
      st="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)" || st="unreadable"
      if spot_interrupted; then
        status "SPOT INTERRUPTED in $b at +${mins}min: the instance is '$st' (Server.SpotInstanceTermination)"
        return 3
      fi
      if [[ "$st" == stopped || "$st" == stopping ]]; then
        (( said_stopped )) || status "INSTANCE STOPPED ($st) in $b at +${mins}min: it halted itself; SSH cannot come back. The card's recovery (one start, only to pull data, before the delete) or Ctrl-C to tear down"
        said_stopped=1
      else
        status "SSH LOST $b poll $fails at +${mins}min: $why; the instance is '${st:-unreadable}'"
      fi
    fi
    if (( now > PULL_DEADLINE )); then
      status "DEADLINE $b at +${mins}min (10 min before the halt): pulling what there is"
      return 2
    fi
  done
}

# --- a Spot interruption: discard the block, relaunch, restore, resume -------------
ceil_inr() { awk -v r="$RATE" -v m="$1" 'BEGIN{c=r*m/60; i=int(c); if (c>i) i++; print i}'; }
archive_attempt() {   # the interrupted instance's files, set aside as attempt_N/
  local d="$REC/attempt_$ATTEMPT" f
  mkdir -p "$d"
  for f in "$REC"/b?.log "$REC"/b?.exit "$REC"/b1_backends.json "$REC"/relaunch_backends.json "$REC/store" \
           "$REC/store_witness.tsv" "$REC/placement.txt" "$REC/deploy.txt" "$REC/bootstrap.txt" \
           "$REC/instance_id.txt" "$REC/host.txt"; do
    [[ -e "$f" ]] && mv "$f" "$d/"
  done
  mkdir -p "$REC/store"
}
restore_session() {   # restore_session B: the blocks before B, checked file by file on the instance
  local b="$1" d="$REC/restore_$ATTEMPT" n got x
  mkdir -p "$d"
  ( cd "$REC/session" && find . -type f | LC_ALL=C sort > "$d/files" && tr '\n' '\0' < "$d/files" \
      | xargs -0 shasum -a 256 > "$d/restore.sha256" && tar cf "$d/restore.tar" -T "$d/files" ) \
    || stop "restore not packed" "DONE FAILED the finished blocks could not be packed for the new instance"
  n="$(wc -l < "$d/restore.sha256" | tr -d ' ')"
  rssh "mkdir -p ~/p3 && test ! -e ~/p3_session && mkdir ~/p3_session" \
    || stop "restore" "DONE FAILED the new instance already has a session, or SSH failed"
  scp "${SSH_OPTS[@]}" "$d/restore.tar" "$d/restore.sha256" "$H:~/p3/" >/dev/null 2>&1 \
    || stop "restore scp" "DONE FAILED the finished blocks could not be sent to the new instance"
  got="$(rssh "cd ~/p3_session && tar xf ~/p3/restore.tar && sha256sum -c ~/p3/restore.sha256 2>/dev/null | grep -c ': OK\$'" 2>/dev/null)"
  [[ "$got" == "$n" ]] || stop "restore check" "DONE FAILED the restored blocks do not check on the new instance (${got:-0} of $n files)"
  for x in b1 b2 b3; do
    [[ "$x" == "$b" ]] && break
    rssh "printf '0\t%s\trestored, attempt %s\n' $(date -u +%FT%TZ) $ATTEMPT > ~/p3/$x.exit && date -u +%FT%TZ > ~/p3/$x.started" \
      || stop "restore exit files" "DONE FAILED the finished blocks' exit files could not be written"
  done
  status "RESTORED the blocks before $b on the new instance: $n files, every sha256 checked there"
}
relaunch() {   # relaunch B: B is discarded; a new instance, ready to run B from its start
  local b="$1" now mins used remaining del halt need x win
  NOTICE_SEEN=0
  teardown "Spot interruption in block $b"
  now="$(date -u +%s)"; mins=$(( (now - LAUNCH_EPOCH + 59) / 60 ))
  used="$(ceil_inr "$mins")"; SPENT_INR=$(( SPENT_INR + used ))
  printf 'attempt\t%s\t%s\t%s\t%s\tmin %s\tRs %s\tinterrupted in %s\n' "$ATTEMPT" "$ID" "$MARKET" "$FIRST_ZONE" "$mins" "$used" "$b" >> "$REC/cost.tsv"
  status "INTERRUPTED attempt $ATTEMPT in block $b after $mins min (Rs $used at the rate; Rs $SPENT_INR of the Rs $CAP cap counted); block $b is discarded in full"
  (( RELAUNCHES < MAX_RELAUNCHES )) || { status "DONE FAILED Spot interrupted in block $b with all $MAX_RELAUNCHES relaunches used; blocks before $b in $REC/session"; exit 1; }
  remaining=$(( CAP - SPENT_INR ))
  del="$(awk -v c="$remaining" -v r="$RATE" 'BEGIN{print int(c * 60 / r)}')"
  (( del > DELETE_MIN_FIRST )) && del=$DELETE_MIN_FIRST
  halt=$(( del - (DELETE_MIN_FIRST - HALT_MIN_FIRST) )); (( halt > HALT_MIN_FIRST )) && halt=$HALT_MIN_FIRST
  need=$SETUP_MIN; x="$b"
  for x in b1 b2 b3; do [[ "$x" < "$b" ]] || need=$(( need + $(eval "echo \$BLOCK_MIN_$x") )); done
  (( halt - PULL_MARGIN_MIN >= need )) || { status "DONE FAILED Spot interrupted in block $b: what is left of the cap, Rs $remaining, allows a halt at +$halt min, short of the $need min the blocks from $b need (+$PULL_MARGIN_MIN for the last pull); no relaunch"; exit 1; }
  now="$(date -u +%s)"
  (( now < END_EPOCH )) || { status "DONE FAILED Spot interrupted in block $b after 18:30 UTC on 2 Dec 2026: no relaunch"; exit 1; }
  win=$(( (END_EPOCH - now) / 60 )); (( win > RELAUNCH_WINDOW_MAX )) && win=$RELAUNCH_WINDOW_MAX
  (( win >= 5 )) || { status "DONE FAILED Spot interrupted in block $b: under 5 min left before the deadline; no relaunch"; exit 1; }
  archive_attempt
  RELAUNCHES=$(( RELAUNCHES + 1 )); ATTEMPT=$(( ATTEMPT + 1 ))
  HALT_MIN=$halt; DELETE_MIN=$del
  export AWS_SESSION_AZS="$FIRST_ZONE" AWS_COST_CAP_INR=$remaining AWS_HALT_MINUTES=$halt AWS_MAX_RUN_MINUTES=$del AWS_LAUNCH_WINDOW_MINUTES=$win
  status "RELAUNCH $RELAUNCHES of $MAX_RELAUNCHES: $ITYPE_WANT in $FIRST_ZONE (block 1's zone), Spot then on-demand, window $win min; halt +$halt, delete +$del, cap Rs $remaining; then block $b from its start"
  launch "$REC/launcher_relaunch$RELAUNCHES.log"
  if [[ $LRC -ne 0 || ! "$ID" =~ ^i- ]]; then
    printf 'relaunch\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$REC" "NOTHING_CREATED rc=$LRC ${LRESULT:-}" >> "$WINDOWS"
    status "DONE FAILED Spot interrupted in block $b; relaunch $RELAUNCHES created nothing (rc $LRC${LRESULT:+, $LRESULT}); blocks before $b in $REC/session"
    bash "$CLEANUP"; exit 1
  fi
  printf 'relaunch\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$REC" "LAUNCHED $ID" >> "$WINDOWS"
  MARKET="$NEW_MARKET"; TORN=0
  started
  boot
  if ! diff -q "$REC/host_first.txt" "$REC/host.txt" >/dev/null; then
    stop "host differs" "DONE FAILED the relaunched instance does not match the first ($(diff "$REC/host_first.txt" "$REC/host.txt" | grep '^>' | tr '\n' ' ' | cut -c1-160))"
  fi
  [[ "$b" == b1 ]] || restore_session "$b"
  if [[ "$PAIRING" == P6 && "$b" != b1 ]]; then   # block 1's abort gate, again on the new instance
    if ! rssh "cd ~/attnbench_scaffold && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/p6_backend_check.py --out ~/p3/relaunch_backends.json" > "$REC/relaunch_backends.txt" 2>&1; then
      scp "${SSH_OPTS[@]}" "$H:~/p3/relaunch_backends.json" "$REC/" >/dev/null 2>&1 || true
      stop "backend check" "DONE FAILED the image and dense-backend check failed on the relaunched instance ($REC/relaunch_backends.txt)"
    fi
    scp "${SSH_OPTS[@]}" "$H:~/p3/relaunch_backends.json" "$REC/" >/dev/null 2>&1 || true
    status "BACKEND CHECK passed on the relaunched instance"
  fi
}

GATE=""
BLOCKS=(b1 b2 b3); k=0
while (( k < ${#BLOCKS[@]} )); do
  B="${BLOCKS[k]}"
  run_block "$B"; brc=$?
  if (( brc == 3 )); then
    relaunch "$B"
    continue
  fi
  if (( brc != 0 )); then
    whatever_there_is
    stop "block $B rc=$brc" "DONE FAILED at block $B (rc $brc); results so far in $REC/session"
  fi
  if [[ "$B" == b2 && "$PAIRING" == P3 ]]; then
    N3="$(ls "$REC/session/05_host_components"/*/bare_prices.json 2>/dev/null | wc -l | tr -d ' ')"
    "$VPY" scripts/p6_host_effect.py "$REC/session/05_host_components" "$P1_HOST" > "$REC/gate.json" 2> "$REC/gate.err"
    GRC=$?
    GCELLS="$("$VPY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"])' "$REC/gate.json" 2>/dev/null)"
    if [[ $GRC -ne 0 || "$N3" != 35 || "$GCELLS" != 35 ]]; then
      whatever_there_is
      stop "gate not readable" "DONE FAILED no-go after block 2: P3 has $N3 cells with host components, the gate read ${GCELLS:-none} of 35 (rc $GRC)"
    fi
    GATE="$("$VPY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("d=%.4f p95=%.4f total_ratio-1=%+.4f verdict=%s" % (r["d"], r["p95"], r["total_ratio_minus_1"], r["verdict"]))' "$REC/gate.json")"
    [[ -n "$GATE" ]] || { whatever_there_is; stop "gate not readable" "DONE FAILED the gate's record could not be read"; }
    echo "$GATE" > "$REC/gate.txt"
    status "GATE after block 2 (35 cells; P6 only, block 3 runs whatever it reads): $GATE"
  fi
  k=$(( k + 1 ))
done
if [[ "$FINAL_REPULL" == 1 ]]; then
  whatever_there_is
else
  pull_logs
  status "FINAL RE-PULL skipped: every block was pulled and checked (P3_DRIVER_FINAL_REPULL=1 turns it on)"
fi
push_halt teardown
rssh 'ls ~/.cache/huggingface/hub/*/snapshots/ 2>&1; du -sh ~/.cache/huggingface 2>&1' > "$REC/hf_snapshots.txt" 2>&1 || true
teardown "all three blocks passed"
(( RELAUNCHES )) && printf 'attempt\t%s\t%s\t%s\t%s\tmin %s\tRs %s\tfinished\n' "$ATTEMPT" "$ID" "$MARKET" "$FIRST_ZONE" \
  "$(( ($(date -u +%s) - LAUNCH_EPOCH + 59) / 60 ))" "$(ceil_inr $(( ($(date -u +%s) - LAUNCH_EPOCH + 59) / 60 )))" >> "$REC/cost.tsv"
status "DONE OK all three blocks passed and pulled${GATE:+; $GATE}$( (( RELAUNCHES )) && echo "; relaunches $RELAUNCHES")"
