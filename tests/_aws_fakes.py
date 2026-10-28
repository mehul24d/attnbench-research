"""Fakes for the AWS session scripts (task T5.1): `aws`, `ssh`, `scp` and
`date`, first on PATH, every call logged.

The real cloud CLIs are dropped from PATH, not just shadowed, the same way
the GCP tests do it, so no path through a script can reach AWS. The fakes
answer only the calls the scripts make; anything else is logged as
UNHANDLED and fails, so a new call site cannot pass by accident.
"""

from __future__ import annotations

import os
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
    echo "arn:aws:ec2:$region:$FAKE_ACCOUNT:instance/$FAKE_INSTANCE" >&2
    echo "$FAKE_INSTANCE"; exit 0 ;;
  "scheduler create-schedule")
    [[ -n "${FAKE_SCHED_FAIL:-}" ]] && { echo "AccessDenied for arn:aws:iam::$FAKE_ACCOUNT:role/x" >&2; exit 254; }
    echo "{\"ScheduleArn\": \"arn:aws:scheduler:$region:$FAKE_ACCOUNT:schedule/default/x\"}"; exit 0 ;;
  "scheduler get-schedule") echo "${FAKE_SCHED_STATE:-ENABLED}"; exit 0 ;;
  "scheduler list-schedules") exit 0 ;;
  "scheduler delete-schedule") exit 0 ;;
  "ec2 terminate-instances") exit "${FAKE_TERMINATE_RC:-0}" ;;
  "ec2 wait") exit 0 ;;
  "ec2 get-console-output")
    [[ -n "${FAKE_CONSOLE_FAIL:-}" ]] && exit 254
    printf 'boot line\nKilled process 42 (python)\n'; exit 0 ;;
  "ec2 describe-instances")
    if [[ "$*" == *"--instance-ids"* ]]; then
      if [[ "$region" == "${FAKE_INSTANCE_REGION:-}" ]]; then
        printf '%s\t%s\t%s\t%s\n' "${FAKE_STATE:-running}" "${FAKE_SESSION_TAG:-None}" \
          "ec2-1-2-3-4.compute-1.amazonaws.com" "${FAKE_LAUNCH:-2026-10-28T06:00:00+00:00}"
        exit 0
      fi
      echo "An error occurred (InvalidInstanceID.NotFound)" >&2; exit 254
    fi
    [[ "$region" == "${FAKE_LIVE_REGION:-}" ]] && printf '%s\trunning\n' "$FAKE_INSTANCE"
    exit 0 ;;
  "ec2 describe-volumes")
    [[ "$region" == "${FAKE_VOLUME_REGION:-}" ]] && echo "vol-0abc 200 gp3"
    exit 0 ;;
  "ec2 describe-addresses") exit 0 ;;
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

# Today (India time) for the launch policy, and a fixed "now" for epochs;
# everything else goes to the real date.
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
if [[ "$last" == "+%s" && "$*" != *"-d"* && "$*" != *"-j"* && -n "${FAKE_NOW:-}" ]]; then
  echo "$FAKE_NOW"; exit 0
fi
exec /bin/date "$@"
"""

AWS_FILES = ("_aws_account.sh", "_aws_launch_policy.sh", "aws_launch_session.sh",
             "aws_cleanup_check.sh", "aws_teardown_session.sh")


def scripts_copy(tmp_path: Path, last_date: str | None) -> Path:
    """The AWS scripts in a scratch directory, with the last AWS launch date
    set to `last_date` (None keeps the committed value)."""
    d = tmp_path / "scripts"
    d.mkdir(exist_ok=True)
    for name in AWS_FILES:
        shutil.copy(SCRIPTS / name, d / name)
    if last_date is not None:
        pol = d / "_aws_launch_policy.sh"
        text = pol.read_text()
        assert 'LAST_AWS_LAUNCH_DATE="UNDECIDED"' in text
        pol.write_text(text.replace('LAST_AWS_LAUNCH_DATE="UNDECIDED"',
                                    f'LAST_AWS_LAUNCH_DATE="{last_date}"'))
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
                       ("date", FAKE_DATE)):
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
               FAKE_BOOT=str(BOOT))
    if pin:
        env.update(ATTNBENCH_AWS_ACCOUNT=ACCOUNT, ATTNBENCH_AWS_PROFILE=PROFILE)
    env.update(extra or {})
    p = subprocess.run(["bash", str(script), *args], input=stdin, capture_output=True,
                       text=True, env=env, cwd=tmp_path, timeout=120)
    lines = log.read_text().splitlines()
    return p, lines


def aws_calls(lines):
    return [c for c in lines if not c.startswith(("ssh ", "scp ", "UNHANDLED "))]
