"""Fakes for the AWS session scripts (task T5.1): `aws`, `ssh`, `scp`,
`date`, `curl` and `sleep`, first on PATH, every call logged.

The real cloud CLIs are dropped from PATH, not just shadowed, the same way
the GCP tests do it, so no path through a script can reach AWS. The fakes
answer only the calls the scripts make; anything else is logged as
UNHANDLED and fails, so a new call site cannot pass by accident.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

ACCOUNT = "111122223333"
OTHER_ACCOUNT = "444455556666"
PROFILE = "pinned-profile"
INSTANCE = "i-0123456789abcdef0"
REGION = "us-east-1"
REGIONS = "us-east-1\tus-west-2\tap-south-1"
NOW = 1793170800            # 2026-10-28T07:00:00Z
BOOT = 1793167200           # 2026-10-28T06:00:00Z, one hour earlier
MY_IP = "203.0.113.7"       # this Mac's public address, as the fake curl reads it

FAKE_AWS = r"""#!/usr/bin/env bash
args="$*"; printf '%s\n' "${args//$'\n'/ }" >> "$AWS_LOG"
region="" profile="" prev=""
for a in "$@"; do
  [[ "$prev" == "--region" ]] && region="$a"
  [[ "$prev" == "--profile" ]] && profile="$a"
  prev="$a"
done
case "$1 $2" in
  "sts get-caller-identity")
    if [[ -n "$profile" ]]; then echo "${FAKE_PROFILE_ACCOUNT:-$FAKE_ACCOUNT}"
    else echo "${FAKE_ACTIVE_ACCOUNT:-$FAKE_ACCOUNT}"; fi
    exit 0 ;;
  "freetier get-account-plan-state")
    [[ -n "${FAKE_PLAN_FAIL:-}" ]] && { echo "AccessDenied" >&2; exit 254; }
    printf '%s\t%s\t%s\t%s\n' "${FAKE_PLAN_TYPE:-PAID}" "${FAKE_PLAN_STATUS:-ACTIVE}" \
      "${FAKE_CREDITS:-140.0}" "${FAKE_CREDIT_UNIT:-USD}"
    exit 0 ;;
  "ec2 describe-regions") printf '%b\n' "$FAKE_REGIONS"; exit 0 ;;
  "ec2 describe-instance-type-offerings")
    [[ -n "${FAKE_NOT_OFFERED:-}" ]] && { echo None; exit 0; }
    for a in "$@"; do case "$a" in Name=instance-type,Values=*) echo "${a#*Values=}";; esac; done
    exit 0 ;;
  "ec2 describe-images")
    if [[ "$*" == *"--owners self"* ]]; then
      [[ "$region" == "${FAKE_IMAGE_REGION:-}" ]] && printf '%b\n' "$FAKE_IMAGES"
      exit 0
    fi
    echo /dev/sda1; exit 0 ;;
  "ec2 run-instances")
    [[ -n "${FAKE_RUN_FAIL:-}" ]] && { echo "An error occurred (InsufficientInstanceCapacity)" >&2; exit 254; }
    itype=""; prev2=""
    for a in "$@"; do [[ "$prev2" == "--instance-type" ]] && itype="$a"; prev2="$a"; done
    # FAKE_RUN_REFUSE_FIRST=N: the first N requests, in any zone, refused for
    # capacity (a count kept in FAKE_RUN_COUNT_FILE), then the zone rules below.
    if [[ -n "${FAKE_RUN_REFUSE_FIRST:-}" ]]; then
      k=$(( $(cat "$FAKE_RUN_COUNT_FILE" 2>/dev/null || echo 0) + 1 )); echo "$k" > "$FAKE_RUN_COUNT_FILE"
      if (( k <= FAKE_RUN_REFUSE_FIRST )); then
        echo "aws: [ERROR]: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 2): We currently do not have sufficient $itype capacity in the Availability Zone you requested (any)." >&2
        exit 254
      fi
      # FAKE_RUN_ERROR_AT=N: request N fails with FAKE_RUN_ERROR instead
      if [[ "$k" == "${FAKE_RUN_ERROR_AT:-}" ]]; then
        echo "aws: [ERROR]: An error occurred (${FAKE_RUN_ERROR:-Unsupported}) when calling the RunInstances operation: refused." >&2
        exit 254
      fi
    fi
    # Per zone: FAKE_RUN_REFUSE_ZONES refuse for capacity, FAKE_RUN_ERROR_ZONES
    # fail with FAKE_RUN_ERROR (another error), as the real CLI words them.
    zone=""
    for a in "$@"; do case "$a" in AvailabilityZone=*) zone="${a#AvailabilityZone=}";; esac; done
    if [[ -n "$zone" && " ${FAKE_RUN_REFUSE_ZONES:-} " == *" $zone "* ]]; then
      echo "" >&2
      echo "aws: [ERROR]: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 2): We currently do not have sufficient $itype capacity in the Availability Zone you requested ($zone)." >&2
      exit 254
    fi
    if [[ -n "$zone" && " ${FAKE_RUN_ERROR_ZONES:-} " == *" $zone "* ]]; then
      echo "aws: [ERROR]: An error occurred (${FAKE_RUN_ERROR:-Unsupported}) when calling the RunInstances operation: refused in $zone." >&2
      exit 254
    fi
    echo "arn:aws:ec2:$region:$FAKE_ACCOUNT:instance/$FAKE_INSTANCE" >&2
    echo "$FAKE_INSTANCE"; exit 0 ;;
  "iam get-role")
    [[ -n "${FAKE_ROLE_FAIL:-}" ]] && { echo "NoSuchEntity" >&2; exit 254; }
    if [[ -n "${FAKE_TRUST:-}" ]]; then printf '%b\n' "$FAKE_TRUST"; exit 0; fi
    printf '1\tAllow\tscheduler.amazonaws.com\tsts:AssumeRole\tStringEquals\taws:SourceAccount,aws:SourceArn\t%s\t%s\n' \
      "${FAKE_TRUST_ARN:-arn:aws:scheduler:$region:$FAKE_ACCOUNT:schedule-group/default}" "$FAKE_ACCOUNT"
    exit 0 ;;
  "scheduler create-schedule")
    if [[ "$*" == *"--name attnbench-dryreg-"* ]]; then
      [[ -n "${FAKE_DRYREG_FAIL:-}" ]] && { echo "ValidationException: The execution role you provide must allow AWS EventBridge Scheduler to assume the role." >&2; exit 254; }
    else
      [[ -n "${FAKE_SCHED_FAIL:-}" ]] && { echo "AccessDenied for arn:aws:iam::$FAKE_ACCOUNT:role/x" >&2; exit 254; }
    fi
    echo "{\"ScheduleArn\": \"arn:aws:scheduler:$region:$FAKE_ACCOUNT:schedule/default/x\"}"; exit 0 ;;
  "scheduler get-schedule")
    if [[ "$*" == *"--name attnbench-dryreg-"* ]]; then echo "${FAKE_DRYREG_STATE:-DISABLED}"
    else echo "${FAKE_SCHED_STATE:-ENABLED}"; fi
    exit 0 ;;
  "scheduler list-schedules") exit 0 ;;
  "scheduler delete-schedule")
    [[ "$*" == *"--name attnbench-dryreg-"* && -n "${FAKE_DRYREG_DELETE_FAIL:-}" ]] && exit 254
    exit 0 ;;
  "ec2 terminate-instances") exit "${FAKE_TERMINATE_RC:-0}" ;;
  "ec2 wait") exit 0 ;;
  "ec2 get-console-output")
    [[ -n "${FAKE_CONSOLE_FAIL:-}" ]] && exit 254
    printf 'boot line\nKilled process 42 (python)\n'; exit 0 ;;
  "ec2 describe-instances")
    if [[ "$*" == *"--instance-ids"* ]]; then
      if [[ "$region" == "${FAKE_INSTANCE_REGION:-}" ]]; then
        # Text output as the real CLI gives it: an empty string prints as
        # nothing (a stopped instance's PublicDnsName), None as "None".
        # FAKE_DNS="" is a stopped instance's empty public DNS.
        st="${FAKE_STATE:-running}" tag="${FAKE_SESSION_TAG:-None}"
        dns="${FAKE_DNS-ec2-1-2-3-4.compute-1.amazonaws.com}"
        lt="${FAKE_LAUNCH:-2026-10-28T06:00:00+00:00}"
        case "$*" in
          *"PublicDnsName,LaunchTime]"*) printf '%s\t%s\t%s\t%s\n' "$st" "$tag" "$dns" "$lt" ;;
          *"attnbench-session"*) printf '%s\n' "$tag" ;;
          *PublicDnsName*) printf '%s\n' "$dns" ;;
          *LaunchTime*) printf '%s\n' "$lt" ;;
          *Placement.AvailabilityZone*) printf '%s\n' "${FAKE_ZONE:-${region}a}" ;;
          *State.Name*) printf '%s\n' "$st" ;;
          *) echo "UNHANDLED query $*" >> "$AWS_LOG"; exit 1 ;;
        esac
        exit 0
      fi
      echo "An error occurred (InvalidInstanceID.NotFound)" >&2; exit 254
    fi
    if [[ "$*" == *"Name=tag:attnbench-session,"* ]]; then      # the launcher's confirmation
      [[ -n "${FAKE_LIST_FAIL:-}" ]] && { echo "RequestLimitExceeded" >&2; exit 254; }
      [[ -n "${FAKE_STRAY_INSTANCE:-}" ]] && printf '%s\t%s\n' "$FAKE_STRAY_INSTANCE" "${region}b"
      exit 0
    fi
    if [[ "$*" == *"Values=terminated"* ]]; then                  # the cleanup's recent list
      [[ "$region" == "${FAKE_TERMINATED_REGION:-}" ]] && printf '%s\tg6.xlarge\t%s\tsess-1\n' "$FAKE_INSTANCE" "${FAKE_ZONE:-${region}b}"
      exit 0
    fi
    [[ "$region" == "${FAKE_LIVE_REGION:-}" ]] && printf '%s\trunning\n' "$FAKE_INSTANCE"
    exit 0 ;;
  "ec2 describe-availability-zones")
    # As the CLI's JMESPath reads the join separator: `" "` is the string " ";
    # a bare ` ` is an old-style literal whose JSON is empty, so the names run
    # together (seen on 23 Nov 2026: "ap-southeast-2aap-southeast-2b...").
    sep=""
    q="$*"; [[ "$q" =~ join\(\`\"([^\"]*)\"\` ]] && sep="${BASH_REMATCH[1]}"
    printf '%s\n' "${region}a${sep}${region}b${sep}${region}c"; exit 0 ;;
  "ec2 describe-volumes")
    if [[ "$*" == *"Name=tag:attnbench-session,"* ]]; then
      [[ -n "${FAKE_STRAY_VOLUME:-}" ]] && printf '%s\t%s\tavailable\n' "$FAKE_STRAY_VOLUME" "${region}b"
      exit 0
    fi
    [[ "$region" == "${FAKE_VOLUME_REGION:-}" ]] && echo "vol-0abc 200 gp3"
    exit 0 ;;
  "ec2 describe-addresses") exit 0 ;;
  "ec2 describe-security-groups")
    [[ -n "${FAKE_SG_FAIL:-}" ]] && { echo "RequestLimitExceeded" >&2; exit 254; }
    printf '%s\n' "${FAKE_SG_CIDRS-$FAKE_MY_IP/32}"; exit 0 ;;
  "ec2 describe-snapshots")
    [[ -n "${FAKE_FAIL_SNAPSHOTS:-}" ]] && exit 254
    exit 0 ;;
esac
echo "UNHANDLED $*" >> "$AWS_LOG"
exit 1
"""

FAKE_SSH = r"""#!/usr/bin/env bash
args="$*"; printf 'ssh %s\n' "${args//$'\n'/ }" >> "$AWS_LOG"
[[ -n "${FAKE_SSH_FAIL:-}" ]] && exit 255
case "$*" in
  *"uptime -s"*) echo "2026-10-28 06:00:00|$FAKE_BOOT"; exit 0 ;;
  *) echo "boot_time: x"; echo "--- PREDATES BOOT (image-carried, NOT this session) ---";
     echo "--- PRODUCED THIS SESSION ---"; echo "/home/u/attnbench_scaffold/results/a.parquet"; exit 0 ;;
esac
"""

FAKE_SCP = r"""#!/usr/bin/env bash
printf 'scp %s\n' "$*" >> "$AWS_LOG"
[[ -n "${FAKE_SSH_FAIL:-}" ]] && exit 1
exit 0
"""

# This Mac's public address. FAKE_MY_IP_SEQ: a file of addresses, one consumed
# per read (the last repeats), as for FAKE_TODAY_SEQ.
FAKE_CURL = r"""#!/usr/bin/env bash
printf 'curl %s\n' "$*" >> "$AWS_LOG"
[[ -n "${FAKE_CURL_FAIL:-}" ]] && exit 28
if [[ -n "${FAKE_MY_IP_SEQ:-}" ]]; then
  head -1 "$FAKE_MY_IP_SEQ"
  [[ $(wc -l < "$FAKE_MY_IP_SEQ") -gt 1 ]] && sed -i.bak 1d "$FAKE_MY_IP_SEQ"
  exit 0
fi
echo "$FAKE_MY_IP"
"""

# A sleep that sleeps for no time: it moves the fake clock (FAKE_CLOCK, a file
# holding epoch seconds) by its argument plus FAKE_SLEEP_EXTRA.
FAKE_SLEEP = r"""#!/usr/bin/env bash
printf 'sleep %s\n' "$*" >> "$AWS_LOG"
if [[ -n "${FAKE_CLOCK:-}" ]]; then
  echo $(( $(cat "$FAKE_CLOCK") + $1 + ${FAKE_SLEEP_EXTRA:-0} )) > "$FAKE_CLOCK"
fi
exit 0
"""

# Today (India time) for the launch policy, and a fixed "now" for epochs
# (FAKE_CLOCK, when set, wins over FAKE_NOW); everything else goes to the real
# date.
FAKE_DATE = r"""#!/usr/bin/env bash
if [[ "$*" == "+%F" ]]; then
  # FAKE_TODAY_SEQ: a file of dates, one consumed per read (the last repeats).
  if [[ -n "${FAKE_TODAY_SEQ:-}" ]]; then
    head -1 "$FAKE_TODAY_SEQ"
    [[ $(wc -l < "$FAKE_TODAY_SEQ") -gt 1 ]] && sed -i.bak 1d "$FAKE_TODAY_SEQ"
    exit 0
  fi
  echo "$FAKE_TODAY"; exit 0
fi
last="${*: -1}"
if [[ "$last" == "+%s" && "$*" != *"-d"* && "$*" != *"-j"* ]]; then
  [[ -n "${FAKE_CLOCK:-}" ]] && { cat "$FAKE_CLOCK"; exit 0; }
  [[ -n "${FAKE_NOW:-}" ]] && { echo "$FAKE_NOW"; exit 0; }
fi
exec /bin/date "$@"
"""

AWS_FILES = ("_aws_account.sh", "_aws_launch_policy.sh", "aws_launch_session.sh",
             "aws_cleanup_check.sh", "aws_teardown_session.sh")


LAST_DATE_RE = re.compile(r'^LAST_AWS_LAUNCH_DATE="([^"]*)"$', re.M)


def committed_last_date() -> str:
    """The value set in scripts/_aws_launch_policy.sh, whatever it is."""
    m = LAST_DATE_RE.findall((SCRIPTS / "_aws_launch_policy.sh").read_text())
    assert len(m) == 1, "the last AWS launch date must be set in exactly one line"
    return m[0]


def scripts_copy(tmp_path: Path, last_date: str | None) -> Path:
    """The AWS scripts in a scratch directory, with the last AWS launch date
    set to `last_date` (None keeps the committed value)."""
    d = tmp_path / "scripts"
    d.mkdir(exist_ok=True)
    for name in AWS_FILES:
        shutil.copy(SCRIPTS / name, d / name)
    if last_date is not None:
        pol = d / "_aws_launch_policy.sh"
        text, n = LAST_DATE_RE.subn(f'LAST_AWS_LAUNCH_DATE="{last_date}"', pol.read_text())
        assert n == 1
        pol.write_text(text)
    return d


def launch_env() -> dict[str, str]:
    """A launch that every guard accepts: Rs 500/h for 120 minutes is
    Rs 1,000 worst case, under a Rs 1,200 cap."""
    return {
        "AWS_SESSION_REGION": REGION,
        "AWS_AMI_ID": "ami-0123456789abcdef0",
        "AWS_KEY_NAME": "attnbench-key",
        "AWS_SECURITY_GROUP_ID": "sg-0123456789abcdef0",
        "AWS_RATE_INR_HR": "500",
        "AWS_COST_CAP_INR": "1200",
        "AWS_MAX_RUN_MINUTES": "120",
    }


def run(script: Path, args, tmp_path: Path, *, extra=None, pin=True,
        stdin="launch\n", today="2026-10-28"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("aws", FAKE_AWS), ("ssh", FAKE_SSH), ("scp", FAKE_SCP),
                       ("date", FAKE_DATE), ("curl", FAKE_CURL), ("sleep", FAKE_SLEEP)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "aws.log"
    log.write_text("")
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("aws", "gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AWS_", "ATTNBENCH_", "GCP_", "FAKE_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), HOME=str(tmp_path),
               AWS_LOG=str(log), FAKE_ACCOUNT=ACCOUNT, FAKE_INSTANCE=INSTANCE,
               FAKE_REGIONS=REGIONS, FAKE_TODAY=today, FAKE_NOW=str(NOW),
               FAKE_BOOT=str(BOOT), FAKE_MY_IP=MY_IP,
               FAKE_RUN_COUNT_FILE=str(tmp_path / "run_count"))
    if pin:
        env.update(ATTNBENCH_AWS_ACCOUNT=ACCOUNT, ATTNBENCH_AWS_PROFILE=PROFILE)
    env.update(extra or {})
    p = subprocess.run(["bash", str(script), *args], input=stdin, capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=120)
    lines = log.read_text().splitlines()
    return p, lines


def aws_calls(lines):
    return [c for c in lines if not c.startswith(("ssh ", "scp ", "curl ", "sleep ", "UNHANDLED "))]
