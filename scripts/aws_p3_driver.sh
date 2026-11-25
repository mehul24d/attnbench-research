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
#   rate Rs 143.21/h, cap Rs 1,353 (6.3 h x 1.5), in-guest halt +473 min (the
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

REC_BASE="$HOME/Desktop/research/attnbench_backups"
AWS_ENV="$HOME/.attnbench/aws_env.sh"
WINDOWS="$HOME/.attnbench/p3_windows.tsv"
P1_HOST="$HOME/Desktop/research/attnbench_backups/bucket_20261117/attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components"
VPY="$HOME/Desktop/research/attnbench_scaffold/.venv/bin/python"
POLL_S=60; SSH_TRIES=40; SSH_WAIT_S=15; TEARDOWN_WAIT_S=30
LAUNCH=(bash scripts/aws_launch_session.sh P3); TEARDOWN=scripts/aws_teardown_session.sh
CLEANUP=scripts/aws_cleanup_check.sh; BOOTSTRAP=scripts/aws_bootstrap_instance.sh
END_EPOCH=1796236200                 # 2026-12-02T18:30:00Z, the launch deadline
NOW_EPOCH="$(date -u +%s)"
HALT_MIN=473; DELETE_MIN=503; CAP=1353; RATE=143.21
PULL_MARGIN_MIN=10                   # the last pull starts 10 min before the halt
if [[ "${P3_DRY:-}" == 1 ]]; then    # the fake remote, never AWS
  REC_BASE="$P3_REC_BASE"; AWS_ENV="$P3_AWS_ENV"; WINDOWS="$P3_WINDOWS"; P1_HOST="$P3_P1_HOST"
  VPY="$P3_VPY"; POLL_S="$P3_POLL_S"; SSH_TRIES=3; SSH_WAIT_S=1; TEARDOWN_WAIT_S=1
  LAUNCH=(bash "$P3_LAUNCH"); TEARDOWN="$P3_TEARDOWN"; CLEANUP="$P3_CLEANUP"; BOOTSTRAP="$P3_BOOTSTRAP"
  NOW_EPOCH="${P3_NOW_EPOCH:-$NOW_EPOCH}"
  [[ -n "${P3_PULL_DEADLINE_S:-}" ]] && PULL_DEADLINE_S="$P3_PULL_DEADLINE_S"
elif env | grep '^P3_' | grep -qv '^P3_DRIVER_'; then
  echo "REFUSING: P3_* variables are set but P3_DRY is not 1; nothing launched" >&2; exit 1
fi

UTC="$(date -u +%Y%m%dT%H%MZ)"
REC="$REC_BASE/aws_p3_g6_$UTC"
[[ -e "$REC" ]] && { echo "REFUSING: $REC exists" >&2; exit 1; }
mkdir -p "$REC"
cp "$P3_DRIVER_COPY" "$REC/driver.sh"
source "$AWS_ENV"
exec > >(sed -u -E "s/${ATTNBENCH_AWS_ACCOUNT}/<account>/g; s/(^|[^0-9a-f])[0-9]{12}([^0-9]|$)/\1<id>\2/g" >> "$REC/driver.log") 2>&1
status() { echo "$(date -u +%FT%TZ) $*"; echo "$(date -u +%FT%TZ) $*" > "$REC/status.txt"; }
refuse() { status "REFUSED $*; nothing launched"; exit 1; }

# --- the launch windows ---------------------------------------------------------
(( NOW_EPOCH < END_EPOCH )) || refuse "18:30 UTC on 2 Dec 2026 has passed: no L4 by then, so P3 is dropped"
LEFT_MIN=$(( (END_EPOCH - NOW_EPOCH) / 60 ))
WINDOW_MIN=$(( LEFT_MIN < 180 ? LEFT_MIN : 180 ))
(( WINDOW_MIN >= 5 )) || refuse "less than 5 minutes are left before 18:30 UTC on 2 Dec 2026"
mkdir -p "$(dirname "$WINDOWS")"; touch "$WINDOWS"
# The ledger: "start<TAB>UTC<TAB>REC" when a window begins, "end<TAB>UTC<TAB>REC<TAB>HOW" when it ends.
awk -F'\t' '$1 == "end" && $4 ~ /^LAUNCHED/ {f = 1} END {exit !f}' "$WINDOWS" \
  && refuse "a P3 instance has already been launched ($WINDOWS): one P3 session"
OPEN="$(awk -F'\t' '$1 == "start" {s[$3] = $2} $1 == "end" && $4 == "NOTHING_CREATED" {delete s[$3]} END {for (r in s) print s[r]}' "$WINDOWS")"
[[ -z "$OPEN" ]] || refuse "a window started at $OPEN has no end line saying it created nothing ($WINDOWS)"
win_end() { printf 'end\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$REC" "$1" >> "$WINDOWS"; }

unset AWS_SESSION_AZ AWS_SUBNET_ID
export AWS_SESSION_REGION=ap-southeast-2 AWS_SESSION_AZS="ap-southeast-2b ap-southeast-2c ap-southeast-2a"
export AWS_RATE_INR_HR=$RATE AWS_COST_CAP_INR=$CAP AWS_HALT_MINUTES=$HALT_MIN AWS_MAX_RUN_MINUTES=$DELETE_MIN
export AWS_LAUNCH_WINDOW_MINUTES=$WINDOW_MIN
export AWS_AMI_ID=ami-096c5dc30364c9041
R=(--profile attnbench --region ap-southeast-2)
KH="$REC/known_hosts"
SSH_OPTS=(-i "$AWS_SSH_KEY" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new
          -o UserKnownHostsFile="$KH" -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

cd "$WT" || exit 1
[[ -z "$(git status --porcelain)" ]] || refuse "the tree is not clean"
COMMIT="$(git rev-parse HEAD)"
[[ -d "$P1_HOST" ]] && (( $(ls "$P1_HOST"/*/bare_prices.json 2>/dev/null | wc -l) == 35 )) \
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

printf 'start\t%s\t%s\n' "$(date -u +%FT%TZ)" "$REC" >> "$WINDOWS"
status "LAUNCHING P3 commit ${COMMIT:0:7} (g6.2xlarge; window ${WINDOW_MIN} min, a pass every 5 min; halt +$HALT_MIN, delete +$DELETE_MIN, cap Rs $CAP; progress in launcher.log)"
echo launch | "${LAUNCH[@]}" 2>&1 | sed -u -E "s/${ATTNBENCH_AWS_ACCOUNT}/<account>/g" | tee "$REC/launcher.log"
LRC=${PIPESTATUS[1]}
LOUT="$(cat "$REC/launcher.log")"
ID="$(sed -n 's/^INSTANCE_ID=//p' <<<"$LOUT")"
LZONE="$(sed -n 's/^LAUNCH_ZONE=//p' <<<"$LOUT" | tail -1)"
LRESULT="$(sed -n 's/^LAUNCH_RESULT=//p' <<<"$LOUT" | tail -1)"
PASSES="$(sed -n 's/^WINDOW_PASSES=//p' <<<"$LOUT" | tail -1)"
NREFUSED="$(sed -n 's/^CAPACITY_REFUSALS=//p' <<<"$LOUT" | tail -1)"
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
LAUNCH_EPOCH="$(date -u +%s)"
PULL_DEADLINE=$(( LAUNCH_EPOCH + ${PULL_DEADLINE_S:-$(( (HALT_MIN - PULL_MARGIN_MIN) * 60 ))} ))
echo "$ID" > "$REC/instance_id.txt"
status "LAUNCHED $ID in zone ${LZONE:-unknown} on pass ${PASSES:-?} (capacity refusals before it: ${NREFUSED:-0}); halt $(date -u -r $(( LAUNCH_EPOCH + HALT_MIN * 60 )) +%FT%TZ 2>/dev/null), delete $(date -u -r $(( LAUNCH_EPOCH + DELETE_MIN * 60 )) +%FT%TZ 2>/dev/null) (about)"

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

status "WAITING for running + SSH"
command aws ec2 wait instance-running "${R[@]}" --instance-ids "$ID"
DNS="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
IZONE="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)"
ITYPE="$(command aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" --query 'Reservations[0].Instances[0].InstanceType' --output text)"
printf 'launcher_zone\t%s\ninstance_zone\t%s\ninstance_type\t%s\nwindow_passes\t%s\ncapacity_refusals\t%s\n' \
  "$LZONE" "$IZONE" "$ITYPE" "${PASSES:-}" "${NREFUSED:-}" > "$REC/placement.txt"
[[ "$ITYPE" == g6.2xlarge ]] || stop "instance type '$ITYPE'" "FAIL the instance is not a g6.2xlarge"
H="ubuntu@$DNS"
rssh() { ssh "${SSH_OPTS[@]}" "$H" "$@"; }
ok=0
for i in $(seq 1 "$SSH_TRIES"); do rssh true 2>/dev/null && { ok=1; break; }; sleep "$SSH_WAIT_S"; done
(( ok )) || stop "no SSH after the boot wait" "FAIL no SSH"
status "SSH up at +$(( ($(date -u +%s) - LAUNCH_EPOCH) / 60 )) min"

status "DEPLOYING bundle ${COMMIT:0:7}"
BUNDLE="${TMPDIR:-/tmp}/p3_$UTC.bundle"
git bundle create "$BUNDLE" HEAD >/dev/null 2>&1
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
pull() {   # pull NAME DIR...: 0 when every file checks
  local name="$1"; shift
  local d="$REC/pull/$name" sum
  mkdir -p "$d"
  sum="$(rssh "set -e; cd ~/p3_session; for x in $*; do test -e \"\$x\"; done; find $* manifest.json -type f | LC_ALL=C sort > ~/p3/$name.files; tr '\n' '\0' < ~/p3/$name.files | xargs -0 sha256sum > ~/p3/$name.sha256; tar cf ~/p3/$name.tar -T ~/p3/$name.files; sha256sum ~/p3/$name.tar | cut -d' ' -f1" 2>"$d/remote.err")" \
    || { status "PULL $name: the instance could not pack $* ($(tail -1 "$d/remote.err"))"; return 1; }
  scp "${SSH_OPTS[@]}" "$H:~/p3/$name.tar" "$H:~/p3/$name.sha256" "$d/" >/dev/null 2>&1 \
    || { status "PULL $name: scp failed"; return 1; }
  [[ "$(shasum -a 256 "$d/$name.tar" | cut -d' ' -f1)" == "$sum" ]] \
    || { status "PULL $name: the tar's sha256 differs from the instance's"; return 1; }
  tar xf "$d/$name.tar" -C "$REC/session" || { status "PULL $name: tar failed"; return 1; }
  (cd "$REC/session" && shasum -a 256 -c "$d/$name.sha256" > "$d/check.txt" 2>&1) \
    || { status "PULL $name: $(grep -vc ': OK$' "$d/check.txt") file(s) do not check"; return 1; }
  status "PULLED $name: $(wc -l < "$d/$name.sha256" | tr -d ' ') files, every sha256 checked"
}
pull_logs() { scp "${SSH_OPTS[@]}" "$H:~/p3/*.log" "$H:~/p3/*.exit" "$REC/" >/dev/null 2>&1 || true; }

BLOCK_DIRS_b1="01_preflight 02_link 03_dense_select 04_probe"
BLOCK_DIRS_b2="05_host_components"
BLOCK_DIRS_b3="06_device_components 07_predict 08_hold 09_ground_truth 10_verify 11_postflight 12_delay_injection"
block_dirs() { eval "echo \$BLOCK_DIRS_$1"; }
whatever_there_is() {   # pull every step directory that exists, as a last act
  local have
  have="$(rssh 'cd ~/p3_session 2>/dev/null && ls -d [0-9][0-9]_* 2>/dev/null' 2>/dev/null | tr '\n' ' ')"
  [[ -n "$have" ]] && pull final $have
  pull_logs
}

run_block() {   # run_block B: 0 passed and pulled; 1 failed; 2 SSH lost to the deadline
  local b="$1" out ex fails=0 now mins
  status "BLOCK $b START ($(block_dirs "$b"))"
  rssh "nohup bash -c 'cd ~/attnbench_scaffold && bash scripts/aws_p3_instance.sh $b ~/p3_session' >/dev/null 2>&1 < /dev/null &" \
    || { status "BLOCK $b could not be started over SSH"; return 1; }
  while :; do
    sleep "$POLL_S"
    now="$(date -u +%s)"; mins=$(( (now - LAUNCH_EPOCH) / 60 ))
    if out="$(rssh "cat ~/p3/$b.exit 2>/dev/null; echo ---; grep -E '^=====|^-----|BLOCK ENDED|STOPPED|REFUSING' ~/p3/$b.log 2>/dev/null | tail -2" 2>&1)"; then
      (( fails )) && status "SSH back after $fails failed poll(s)"
      fails=0
      witness_store || true
      ex="$(head -1 <<<"$out" | cut -f1 | grep -E '^[0-9]+$' || true)"
      status "POLL $b +${mins}min exit='${ex}' :: $(tr '\n' ' ' <<<"$out" | cut -c1-260)"
      if [[ -n "$ex" ]]; then
        pull_logs
        [[ "$ex" == 0 ]] || { status "BLOCK $b FAILED exit $ex"; return 1; }
        pull "$b" $(block_dirs "$b") || return 1
        return 0
      fi
    else
      fails=$(( fails + 1 ))
      sg_admits; SG=$?
      case $SG in
        0) why="the group still admits this Mac's address (network?)" ;;
        1) why="this Mac's public address could not be read (network down?)" ;;
        2) why="the security group could not be read (aws login --profile attnbench?)" ;;
        *) why="the group no longer admits this Mac's address: re-point it with sg_repoint.sh (the driver never does)" ;;
      esac
      status "SSH LOST $b poll $fails at +${mins}min: $why; the instance runs on"
    fi
    if (( now > PULL_DEADLINE )); then
      status "DEADLINE $b at +${mins}min (10 min before the halt): pulling what there is"
      return 2
    fi
  done
}

GATE=""
for B in b1 b2 b3; do
  run_block "$B"; brc=$?
  if (( brc != 0 )); then
    whatever_there_is
    stop "block $B rc=$brc" "DONE FAILED at block $B (rc $brc); results so far in $REC/session"
  fi
  if [[ "$B" == b2 ]]; then
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
done
whatever_there_is
rssh 'ls ~/.cache/huggingface/hub/*/snapshots/ 2>&1; du -sh ~/.cache/huggingface 2>&1' > "$REC/hf_snapshots.txt" 2>&1 || true
teardown "all three blocks passed"
status "DONE OK all three blocks passed and pulled; $GATE"
