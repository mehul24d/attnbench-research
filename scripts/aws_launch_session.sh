#!/usr/bin/env bash
# Launch one AWS pairing session (P3, P5, P6 or the optional P7; plan
# Section 5.2, task T5.1). Creates a BILLABLE resource.
#
#   bash scripts/aws_launch_session.sh PAIRING [NAME]
#
# The AWS counterpart of the gcp_launch_*.sh scripts, with the same
# guarantees, each checked before anything is created:
#   - launch policy (scripts/_aws_launch_policy.sh): no launch while the last
#     AWS launch date is undecided or past, never an Australia region;
#     checked again immediately before the create request;
#   - account and profile pinned (scripts/_aws_account.sh), never printed;
#   - a declared planning rate (AWS_RATE_INR_HR, Rs/h all-in), required, not
#     defaulted: the GCP teardown's old default under-costed an A100 session
#     by a factor of 3.7;
#   - a hard-delete limit (AWS_MAX_RUN_MINUTES), required: the GCP defaults
#     were set so far out that they never bound
#     (docs/silent_failure_patterns.md #40);
#   - a cost cap (AWS_COST_CAP_INR), required: the launch is refused unless
#     rate x hard-delete limit fits under it, so the worst case is bounded
#     before anything bills;
#   - an in-guest halt (AWS_HALT_MINUTES, default 60 minutes before the hard
#     delete) with shutdown behaviour "stop", so GPU billing stops while the
#     root volume survives for recovery. On Spot the behaviour must be
#     "terminate" (one-time Spot instances cannot stop), so there the halt is
#     the delete and there is no recovery window; the summary says so;
#   - the hard delete is enforced by AWS, not by the guest: a one-time
#     EventBridge Scheduler schedule calls ec2:TerminateInstances at launch +
#     AWS_MAX_RUN_MINUTES and then deletes itself. GCP's
#     --max-run-duration=DELETE has no EC2 equivalent. If the schedule
#     cannot be created, or does not read back ENABLED, the instance is
#     terminated at once: no instance is left without its hard delete;
#   - root volume 200 GB gp3, encrypted, DeleteOnTermination=true, so a
#     terminated instance leaves no disk billing;
#   - IMDSv2 only, with instance tags readable from instance metadata so a
#     later provenance reader can see the provisioning-model tag;
#   - tags on the instance and its volume: attnbench-session, -pairing,
#     -provisioning-model (STANDARD or SPOT, the vocabulary of
#     attnbench/provenance.py), -hard-delete-at. The teardown refuses an
#     instance without attnbench-session;
#   - typed confirmation ("launch") after the summary.
#
# The machine types below are PROPOSED (plan Section 5.2 names the
# families; the sizes are not yet decided), and nothing has a default
# region, AMI, key pair or security group: each is a researcher decision.
#
# Required environment:
#   AWS_SESSION_REGION     e.g. us-east-1 (no default)
#   AWS_AMI_ID             ami-...
#   AWS_KEY_NAME           EC2 key pair name
#   AWS_SECURITY_GROUP_ID  sg-...
#   AWS_RATE_INR_HR        planning rate, Rs/h all-in
#   AWS_COST_CAP_INR       session cost cap, Rs
#   AWS_MAX_RUN_MINUTES    hard-delete limit, minutes from launch
# Optional:
#   AWS_SESSION_AZ         availability zone inside the region
#   AWS_SUBNET_ID          subnet-... (fixes the zone; then AZ is not passed)
#   AWS_HALT_MINUTES       in-guest halt (default AWS_MAX_RUN_MINUTES - 60)
#   AWS_PROVISIONING       on-demand (default) or spot
#   AWS_HARD_DELETE_ROLE   IAM role the scheduler assumes (default
#                          attnbench-hard-delete); it needs only
#                          ec2:TerminateInstances
set -euo pipefail

HERE="$(dirname "${BASH_SOURCE[0]}")"
PAIRING="${1:?usage: aws_launch_session.sh PAIRING [NAME]   (PAIRING: P3, P5, P6 or P7)}"

# PROPOSED sizes: one GPU each, 8 vCPUs where the family allows, to match
# P1's g2-standard-8; p5.4xlarge is the plan's own pin.
case "$PAIRING" in
  P3) INSTANCE_TYPE="g6.2xlarge";  GPU="1x L4 24GB" ;;
  P5) INSTANCE_TYPE="g6e.2xlarge"; GPU="1x L40S 48GB" ;;
  P6) INSTANCE_TYPE="p5.4xlarge";  GPU="1x H100 80GB" ;;
  P7) INSTANCE_TYPE="g5.2xlarge";  GPU="1x A10G 24GB" ;;
  *) echo "REFUSING -- unknown pairing '$PAIRING'; AWS pairings are P3, P5, P6 and P7." >&2
     exit 1 ;;
esac
PAIR_LC="$(echo "$PAIRING" | tr 'A-Z' 'a-z')"
NAME="${2:-attnbench-${PAIR_LC}-$(date -u +%Y%m%d-%H%M)}"
if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,62}$ ]]; then
  echo "REFUSING -- session name must be letters, digits and '-' (at most 63)." >&2
  exit 1
fi

REGION="${AWS_SESSION_REGION:-}"
AZ="${AWS_SESSION_AZ:-}"
if [[ ! "$REGION" =~ ^[a-z]{2}(-[a-z]+)+-[0-9]$ ]]; then
  echo "REFUSING TO LAUNCH -- set AWS_SESSION_REGION to the pairing's region (no default:" >&2
  echo "the AWS regions are a researcher decision, plan Section 5.2)." >&2
  exit 1
fi
if [[ -n "$AZ" && "$AZ" != "$REGION"* ]]; then
  echo "REFUSING TO LAUNCH -- AWS_SESSION_AZ ($AZ) is not in $REGION." >&2
  exit 1
fi
source "$HERE/_aws_launch_policy.sh"
aws_launch_policy_check "$REGION" $AZ

# --- money: rate, hard-delete limit, cost cap --------------------------------
NUM='^[0-9]+([.][0-9]+)?$'
RATE_INR_HR="${AWS_RATE_INR_HR:-}"
COST_CAP_INR="${AWS_COST_CAP_INR:-}"
MAX_RUN_MINUTES="${AWS_MAX_RUN_MINUTES:-}"
if [[ ! "$RATE_INR_HR" =~ $NUM ]]; then
  echo "REFUSING TO LAUNCH -- declare the planning rate as AWS_RATE_INR_HR (Rs/h, all-in," >&2
  echo "plan Section 5.2). Pass the same rate to aws_teardown_session.sh." >&2
  exit 1
fi
if [[ ! "$COST_CAP_INR" =~ $NUM ]]; then
  echo "REFUSING TO LAUNCH -- declare the session cost cap as AWS_COST_CAP_INR (Rs)." >&2
  exit 1
fi
if [[ ! "$MAX_RUN_MINUTES" =~ ^[0-9]+$ ]] || (( MAX_RUN_MINUTES < 30 || MAX_RUN_MINUTES > 1440 )); then
  echo "REFUSING TO LAUNCH -- declare the hard-delete limit as AWS_MAX_RUN_MINUTES, whole" >&2
  echo "minutes from 30 to 1440, sized to the work (no default)." >&2
  exit 1
fi
HALT_MINUTES="${AWS_HALT_MINUTES:-$(( MAX_RUN_MINUTES - 60 ))}"
if [[ ! "$HALT_MINUTES" =~ ^[0-9]+$ ]] || (( HALT_MINUTES < 20 || HALT_MINUTES >= MAX_RUN_MINUTES )); then
  echo "REFUSING TO LAUNCH -- the in-guest halt (AWS_HALT_MINUTES=$HALT_MINUTES) must be at" >&2
  echo "least 20 minutes and earlier than the hard delete ($MAX_RUN_MINUTES minutes)." >&2
  exit 1
fi
# Worst case: the full rate until the hard delete. Rs, rounded up.
WORST_INR="$(awk -v r="$RATE_INR_HR" -v m="$MAX_RUN_MINUTES" \
  'BEGIN{c=r*m/60; i=int(c); if (c>i) i++; print i}')"
if ! awk -v w="$WORST_INR" -v c="$COST_CAP_INR" 'BEGIN{exit !(w<=c)}'; then
  echo "REFUSING TO LAUNCH -- the worst case, Rs ${WORST_INR} (Rs ${RATE_INR_HR}/h for" >&2
  echo "${MAX_RUN_MINUTES} min to the hard delete), exceeds the cost cap of Rs ${COST_CAP_INR}." >&2
  echo "Shorten AWS_MAX_RUN_MINUTES or have the cap raised." >&2
  exit 1
fi

case "${AWS_PROVISIONING:-on-demand}" in
  on-demand) MODEL="STANDARD"; SHUTDOWN_BEHAVIOR="stop" ;;
  spot)      MODEL="SPOT";     SHUTDOWN_BEHAVIOR="terminate" ;;
  *) echo "REFUSING TO LAUNCH -- AWS_PROVISIONING must be on-demand or spot." >&2; exit 1 ;;
esac

AMI="${AWS_AMI_ID:-}"
KEY_NAME="${AWS_KEY_NAME:-}"
SG="${AWS_SECURITY_GROUP_ID:-}"
SUBNET="${AWS_SUBNET_ID:-}"
ROLE_NAME="${AWS_HARD_DELETE_ROLE:-attnbench-hard-delete}"
[[ "$AMI" =~ ^ami-[0-9a-f]{8,17}$ ]] || { echo "REFUSING TO LAUNCH -- set AWS_AMI_ID (ami-...)." >&2; exit 1; }
[[ "$KEY_NAME" =~ ^[A-Za-z0-9._-]{1,255}$ ]] || { echo "REFUSING TO LAUNCH -- set AWS_KEY_NAME." >&2; exit 1; }
[[ "$SG" =~ ^sg-[0-9a-f]{8,17}$ ]] || { echo "REFUSING TO LAUNCH -- set AWS_SECURITY_GROUP_ID (sg-...)." >&2; exit 1; }
[[ -z "$SUBNET" || "$SUBNET" =~ ^subnet-[0-9a-f]{8,17}$ ]] || { echo "REFUSING TO LAUNCH -- AWS_SUBNET_ID is not subnet-...." >&2; exit 1; }
[[ "$ROLE_NAME" =~ ^[A-Za-z0-9+=,.@_-]{1,64}$ ]] || { echo "REFUSING TO LAUNCH -- AWS_HARD_DELETE_ROLE is not a role name." >&2; exit 1; }

source "$HERE/_aws_account.sh"
aws_account_pin
P=(--profile "$AWS_PIN_PROFILE" --region "$REGION")

# --- read-only checks against the account ------------------------------------
OFFERED="$(aws ec2 describe-instance-type-offerings "${P[@]}" \
  --location-type "$([[ -n "$AZ" ]] && echo availability-zone || echo region)" \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" "Name=location,Values=${AZ:-$REGION}" \
  --query 'InstanceTypeOfferings[0].InstanceType' --output text)"
if [[ "$OFFERED" != "$INSTANCE_TYPE" ]]; then
  echo "REFUSING TO LAUNCH -- $INSTANCE_TYPE is not offered in ${AZ:-$REGION}. Nothing was created." >&2
  exit 1
fi
ROOT_DEVICE="$(aws ec2 describe-images "${P[@]}" --image-ids "$AMI" \
  --query 'Images[0].RootDeviceName' --output text)"
if [[ ! "$ROOT_DEVICE" =~ ^/dev/[a-z0-9]+$ ]]; then
  echo "REFUSING TO LAUNCH -- cannot read the root device of $AMI in $REGION. Nothing was created." >&2
  exit 1
fi

# Epoch seconds to an ISO UTC time, on GNU (the instance, CI) and BSD (the
# researcher's Mac) date alike.
iso_utc() {
  command date -u -d "@$1" +%Y-%m-%dT%H:%M:%S 2>/dev/null \
    || command date -u -r "$1" +%Y-%m-%dT%H:%M:%S
}

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
# Hard cap inside the guest, scheduled from boot. The shutdown behaviour is
# "$SHUTDOWN_BEHAVIOR", so this ${SHUTDOWN_BEHAVIOR}s the instance.
logger "attnbench-aws: scheduling in-guest halt in ${HALT_MINUTES} minutes"
shutdown -h +${HALT_MINUTES}
EOF

echo "About to create a BILLABLE instance:"
echo "  account      : ATTNBENCH_AWS_ACCOUNT via ATTNBENCH_AWS_PROFILE (pinned, not printed)"
echo "  pairing      : $PAIRING"
echo "  name         : $NAME"
echo "  region / AZ  : $REGION / ${AZ:-${SUBNET:-any}}"
echo "  type         : $INSTANCE_TYPE ($GPU)"
echo "  image        : $AMI (root $ROOT_DEVICE)"
echo "  root volume  : 200 GB gp3, encrypted, deleted with the instance"
echo "  provisioning : $MODEL"
echo "  rate         : Rs ${RATE_INR_HR}/h (AWS_RATE_INR_HR, planning rate, all-in)"
if [[ "$SHUTDOWN_BEHAVIOR" == "stop" ]]; then
  echo "  in-guest halt: +${HALT_MINUTES} min, STOPS the instance (GPU billing stops, disk survives)"
else
  echo "  in-guest halt: +${HALT_MINUTES} min, TERMINATES the instance (Spot: no recovery window)"
fi
echo "  hard delete  : +${MAX_RUN_MINUTES} min, ec2:TerminateInstances by EventBridge Scheduler"
echo "  worst case   : Rs ${WORST_INR}, cost cap Rs ${COST_CAP_INR}"
echo
read -r -p "Type 'launch' to proceed, anything else to abort: " CONFIRM
if [[ "$CONFIRM" != "launch" ]]; then
  echo "Aborted -- no instance created."
  exit 1
fi

# The prompt may have waited past midnight: the launch is the request.
aws_launch_policy_check "$REGION" $AZ

PLACEMENT=()
if [[ -n "$SUBNET" ]]; then
  PLACEMENT=(--subnet-id "$SUBNET")
elif [[ -n "$AZ" ]]; then
  PLACEMENT=(--placement "AvailabilityZone=$AZ")
fi
MARKET=()
if [[ "$MODEL" == "SPOT" ]]; then
  MARKET=(--instance-market-options
          'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi
# One hard-delete time, fixed before the request: the tag and the schedule
# carry the same value, and the instance runs at most AWS_MAX_RUN_MINUTES.
DELETE_AT="$(iso_utc $(( $(command date -u +%s) + MAX_RUN_MINUTES * 60 )))"
TAGS="{Key=Name,Value=$NAME},{Key=attnbench-session,Value=$NAME},{Key=attnbench-pairing,Value=$PAIRING},{Key=attnbench-provisioning-model,Value=$MODEL},{Key=attnbench-hard-delete-at,Value=${DELETE_AT}Z}"

# bash 3.2 (macOS) aborts on "${EMPTY[@]}" under set -u, hence the guards.
INSTANCE_ID="$(aws ec2 run-instances "${P[@]}" \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --count 1 \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG" \
  ${PLACEMENT[@]+"${PLACEMENT[@]}"} \
  ${MARKET[@]+"${MARKET[@]}"} \
  --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled,InstanceMetadataTags=enabled' \
  --block-device-mappings "DeviceName=$ROOT_DEVICE,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}" \
  --tag-specifications "ResourceType=instance,Tags=[$TAGS]" "ResourceType=volume,Tags=[$TAGS]" \
  --user-data "file://$STARTUP_SCRIPT" \
  --query 'Instances[0].InstanceId' --output text)" || INSTANCE_ID=""
if [[ ! "$INSTANCE_ID" =~ ^i-[0-9a-f]{8,17}$ ]]; then
  echo "ERROR -- run-instances returned no instance id ('$INSTANCE_ID'). An instance may" >&2
  echo "still exist: run scripts/aws_cleanup_check.sh now." >&2
  exit 1
fi
echo "Created $INSTANCE_ID. Billing now. Setting the hard delete."

terminate_now() {
  echo "!! $1 -- terminating $INSTANCE_ID now, so it is never left without a hard delete." >&2
  if aws ec2 terminate-instances "${P[@]}" --instance-ids "$INSTANCE_ID" >/dev/null; then
    echo "   terminated. Run scripts/aws_cleanup_check.sh to confirm." >&2
  else
    echo "   TERMINATE FAILED. Terminate $INSTANCE_ID by hand in $REGION NOW, then run" >&2
    echo "   scripts/aws_cleanup_check.sh." >&2
  fi
  exit 1
}

SCHEDULE="attnbench-hard-delete-$INSTANCE_ID"
TARGET="{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:terminateInstances\",\"RoleArn\":\"arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"${INSTANCE_ID}\\\"]}\"}"
if ! aws scheduler create-schedule "${P[@]}" \
     --name "$SCHEDULE" \
     --schedule-expression "at(${DELETE_AT})" \
     --schedule-expression-timezone UTC \
     --flexible-time-window Mode=OFF \
     --action-after-completion DELETE \
     --target "$TARGET" >/dev/null; then
  terminate_now "the hard-delete schedule could not be created"
fi
STATE="$(aws scheduler get-schedule "${P[@]}" --name "$SCHEDULE" --query State --output text || true)"
if [[ "$STATE" != "ENABLED" ]]; then
  terminate_now "the hard-delete schedule reads back as '$STATE', not ENABLED"
fi

echo
echo "Created on $MODEL. Billing now."
echo "  halts   at +${HALT_MINUTES} min from boot ($SHUTDOWN_BEHAVIOR)"
echo "  DELETES at ${DELETE_AT}Z (schedule $SCHEDULE), root volume with it"
echo "  teardown: AWS_RATE_INR_HR=$RATE_INR_HR bash scripts/aws_teardown_session.sh $INSTANCE_ID $REGION"
echo "  confirm clean: bash scripts/aws_cleanup_check.sh"
echo "INSTANCE_ID=$INSTANCE_ID"
echo "HARD_DELETE_AT=${DELETE_AT}Z"
echo "PROVISIONING_MODEL=$MODEL"
