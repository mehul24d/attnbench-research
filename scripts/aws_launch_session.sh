#!/usr/bin/env bash
# Launch one AWS pairing session (P3, P5, P6 or the optional P7; plan
# Section 5.2, task T5.1). Creates a BILLABLE resource.
#
#   bash scripts/aws_launch_session.sh PAIRING [NAME]
#
# PAIRING is P3, P5, P6 or P7, or ENV: the CPU-only environment build on
# m7a.xlarge that makes the AMI the GPU sessions boot from (adviser's
# rulings dated 21 Nov 2026; card docs/launch_cards/aws_env_build_m7a.md).
# ENV takes every guarantee below; it times nothing. PROBE is a short
# inspection of the same image on the same type, with no build (adviser's
# rulings dated 22 Nov 2026, probe): every guarantee below, but its hard
# delete may be 5 to 15 minutes and its halt from 3 minutes (the ruling's
# 8 and 6), where every other pairing needs 30 and 20.
#
# The AWS counterpart of the gcp_launch_*.sh scripts, with the same
# guarantees, each checked before anything is created:
#   - launch policy (scripts/_aws_launch_policy.sh): no launch while the last
#     AWS launch date is undecided or past, no Australia region but Sydney;
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
#   - on-demand only. Spot is refused (researcher's decision, 28 Oct 2026):
#     an interruption would ruin a timed run;
#   - an in-guest halt (AWS_HALT_MINUTES, default 60 minutes before the hard
#     delete) with shutdown behaviour "stop", so GPU billing stops while the
#     root volume survives for recovery;
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
#     -provisioning-model (STANDARD, the vocabulary of
#     attnbench/provenance.py), -hard-delete-at. The teardown refuses an
#     instance without attnbench-session;
#   - a credit gate (adviser's rulings dated 22 Nov 2026, credits): no launch
#     unless the account is on AWS's paid plan and its remaining credits
#     exceed the planned AWS spend so far plus this block's cap (see the
#     credits section below). The Free plan admits only Free Tier instance
#     types: the environment build's first attempt (22 Nov 2026) was refused
#     at run-instances for that reason;
#   - the hard-delete role checked twice (adviser's rulings dated 22 Nov 2026,
#     role fix): its trust policy is read back and must be the documented
#     form, one statement letting scheduler.amazonaws.com assume the role on
#     the conditions aws:SourceAccount = the pinned account and
#     aws:SourceArn = this region's schedule group
#     (arn:aws:scheduler:<region>:<account>:schedule-group/default), not a
#     schedule; and, after the typed confirmation and before run-instances, a
#     dry registration: a throwaway DISABLED schedule thirty days out, using
#     the role, for a non-existent instance, which AWS must accept and which
#     is deleted at once. The environment build's attempt of 21 Nov 2026
#     (21:15 UTC) created its instance and then had the hard-delete schedule
#     refused, because the trust condition named the schedule; the launcher
#     terminated it about 3 s later. AWS validates the role at create time,
#     a disabled schedule included (checked 21 Nov 2026 against that role);
#   - typed confirmation ("launch") after the summary.
#
# AWS services allowed (adviser's rulings dated 22 Nov 2026, credits): EC2,
# EBS volumes, EBS snapshots and AMIs, networking (the VPC, security group,
# key pair and public address the instance uses), IAM (the hard-delete role),
# EventBridge Scheduler (the hard delete) and Service Quotas (quota reads).
# Nothing else is used for the work: no S3, Lambda, Bedrock, RDS, DynamoDB,
# SageMaker or Marketplace image. The account's own administration is not
# work: the Free Tier API read of the plan and credits, and AWS Budgets
# alerts (about Rs 6,000 gross; about $1 net of credits). Billing data lags
# by up to a day, so a budget alert can arrive a day after the spend.
# After the first AWS usage, check the next day that the credits were
# applied to the EC2 line of the bill; if they were not, stop before P3.
#
# The machine types below were accepted by the researcher on 28 Oct 2026.
# Nothing has a default region, AMI, key pair, security group, rate or cap:
# each is the researcher's to set, and the launch is refused without it.
#
# PROPOSED planning rates, NOT confirmed (the researcher confirms them; until
# then AWS_RATE_INR_HR has no default). Source: the AWS Price List bulk API,
# AmazonEC2 offer for us-east-1, publication 2026-10-28T15:03:53Z, terms
# effective 2026-10-01, on-demand Linux shared tenancy. us-east-1 is only a
# reference: the regions are not decided, and prices differ by region.
#   instance       USD/h    SKU               host
#   g6.2xlarge     0.9776   XCA3J9EEM739PU6M  AMD EPYC 7R13   (P3)
#   g6e.2xlarge    2.24208  HK3A8PU2TSC6EKP6  AMD EPYC 7R13   (P5)
#   p5.4xlarge     6.88     K9VKFEBARNNMVRXV  AMD EPYC 7R13   (P6)
#   g5.2xlarge     1.212    FX9TUYTJKEENVBSR  AMD EPYC 7R32   (P7)
#   gp3 storage    0.08 per GB-month, SKU JG3KUJMBRGHV3N8G: 200 GB is
#                  0.021918/h at 730 h per month
# Planning rate, all-in = (instance + 200 GB gp3) x 1.15, at Rs 95.99 per
# USD (the rate the plan's GCP table used; an assumption here, as are the
# 730 hours). Taxes and data transfer are not included.
#   P3 Rs 110.34/h   P5 Rs 249.92/h   P6 Rs 761.89/h   P7 Rs 136.21/h
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
#   AWS_PROVISIONING       on-demand (the default and the only value accepted)
#   AWS_HARD_DELETE_ROLE   IAM role the scheduler assumes (default
#                          attnbench-hard-delete); it needs only
#                          ec2:TerminateInstances
#   AWS_LAUNCH_RECORD      the launch record the credit gate sums (default
#                          ~/.attnbench/aws_launches.tsv, off-repo)
set -euo pipefail

HERE="$(dirname "${BASH_SOURCE[0]}")"
PAIRING="${1:?usage: aws_launch_session.sh PAIRING [NAME]   (PAIRING: P3, P5, P6, P7 or ENV)}"

# One GPU each, 8 vCPUs where the family allows, to match P1's
# g2-standard-8; p5.4xlarge is the plan's own pin. Accepted 28 Oct 2026.
case "$PAIRING" in
  P3) INSTANCE_TYPE="g6.2xlarge";  GPU="1x L4 24GB" ;;
  P5) INSTANCE_TYPE="g6e.2xlarge"; GPU="1x L40S 48GB" ;;
  P6) INSTANCE_TYPE="p5.4xlarge";  GPU="1x H100 80GB" ;;
  P7) INSTANCE_TYPE="g5.2xlarge";  GPU="1x A10G 24GB" ;;
  ENV) INSTANCE_TYPE="m7a.xlarge"; GPU="no GPU: environment build only" ;;
  PROBE) INSTANCE_TYPE="m7a.xlarge"; GPU="no GPU: image inspection only, no build" ;;
  *) echo "REFUSING -- unknown pairing '$PAIRING'; AWS pairings are P3, P5, P6 and P7, ENV is the environment build and PROBE its image inspection." >&2
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
if [[ "$PAIRING" == PROBE ]]; then MIN_RUN=5; MAX_RUN=15; MIN_HALT=3; else MIN_RUN=30; MAX_RUN=1440; MIN_HALT=20; fi
if [[ ! "$MAX_RUN_MINUTES" =~ ^[0-9]+$ ]] || (( MAX_RUN_MINUTES < MIN_RUN || MAX_RUN_MINUTES > MAX_RUN )); then
  echo "REFUSING TO LAUNCH -- declare the hard-delete limit as AWS_MAX_RUN_MINUTES, whole" >&2
  echo "minutes from $MIN_RUN to $MAX_RUN for $PAIRING, sized to the work (no default)." >&2
  exit 1
fi
if [[ "$PAIRING" == PROBE ]]; then HALT_DEFAULT=$(( MAX_RUN_MINUTES - 2 )); else HALT_DEFAULT=$(( MAX_RUN_MINUTES - 60 )); fi
HALT_MINUTES="${AWS_HALT_MINUTES:-$HALT_DEFAULT}"
if [[ ! "$HALT_MINUTES" =~ ^[0-9]+$ ]] || (( HALT_MINUTES < MIN_HALT || HALT_MINUTES >= MAX_RUN_MINUTES )); then
  echo "REFUSING TO LAUNCH -- the in-guest halt (AWS_HALT_MINUTES=$HALT_MINUTES) must be at" >&2
  echo "least $MIN_HALT minutes and earlier than the hard delete ($MAX_RUN_MINUTES minutes)." >&2
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

# On-demand only: a Spot interruption would ruin a timed run (researcher's
# decision, 28 Oct 2026).
if [[ "${AWS_PROVISIONING-on-demand}" != "on-demand" ]]; then
  echo "REFUSING TO LAUNCH -- AWS_PROVISIONING='${AWS_PROVISIONING}': measurement sessions" >&2
  echo "are on-demand only (a Spot interruption would ruin a timed run)." >&2
  exit 1
fi
MODEL="STANDARD"
SHUTDOWN_BEHAVIOR="stop"

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

# --- credits: the remaining credits must cover everything planned -----------
# Refused unless the account is on the paid plan (the Free plan admits only
# Free Tier instance types) and its remaining credits, read from the Free
# Tier API, exceed the planned AWS spend so far plus this block's cap. The
# planned spend so far is the sum of the caps in the launch record: one line
# per create request this launcher has submitted, written at the block's cap
# whatever it later cost, because the credit balance lags the bill by up to
# a day. A line is struck by hand, with a leading '#', only once the bill
# shows what the block cost (or that it created nothing). Credits are in
# USD; caps are converted at Rs 95.99 per USD, the rate every AWS planning
# rate in this repository was built at (plan Section 5.2).
INR_PER_USD="95.99"
LAUNCH_RECORD="${AWS_LAUNCH_RECORD:-$HOME/.attnbench/aws_launches.tsv}"
PLAN_STATE="$(aws freetier get-account-plan-state --profile "$AWS_PIN_PROFILE" --region us-east-1 \
  --query '[accountPlanType,accountPlanStatus,accountPlanRemainingCredits.amount,accountPlanRemainingCredits.unit]' \
  --output text 2>/dev/null)" || PLAN_STATE=""
read -r PLAN_TYPE PLAN_STATUS CREDITS_USD CREDIT_UNIT _ <<<"$PLAN_STATE" || true
if [[ "${PLAN_TYPE:-}" != "PAID" || "${PLAN_STATUS:-}" != "ACTIVE" ]]; then
  echo "REFUSING TO LAUNCH -- the account's plan reads '${PLAN_TYPE:-unreadable} ${PLAN_STATUS:-}', not" >&2
  echo "'PAID ACTIVE'. AWS's Free plan admits only Free Tier instance types. Nothing was created." >&2
  exit 1
fi
if [[ "${CREDIT_UNIT:-}" != "USD" || ! "${CREDITS_USD:-}" =~ $NUM ]]; then
  echo "REFUSING TO LAUNCH -- cannot read the remaining credits in USD ('${CREDITS_USD:-} ${CREDIT_UNIT:-}')." >&2
  echo "Nothing was created." >&2
  exit 1
fi
PLANNED_INR=0
if [[ -e "$LAUNCH_RECORD" ]]; then
  PLANNED_INR="$(awk -F'\t' '/^[[:space:]]*(#|$)/ {next}
    NF != 4 || $4 !~ /^[0-9]+([.][0-9]+)?$/ {bad = 1; exit}
    {s += $4} END {if (bad) print "BAD"; else printf "%.2f\n", s + 0}' "$LAUNCH_RECORD")"
  if [[ ! "$PLANNED_INR" =~ $NUM ]]; then
    echo "REFUSING TO LAUNCH -- the launch record $LAUNCH_RECORD has a line that is not" >&2
    echo "'UTC<tab>PAIRING<tab>INSTANCE<tab>CAP_INR' (strike lines with a leading '#'). Nothing was created." >&2
    exit 1
  fi
fi
CREDITS_INR="$(awk -v c="$CREDITS_USD" -v r="$INR_PER_USD" 'BEGIN{printf "%.2f\n", c * r}')"
NEED_INR="$(awk -v p="$PLANNED_INR" -v k="$COST_CAP_INR" 'BEGIN{printf "%.2f\n", p + k}')"
if ! awk -v c="$CREDITS_INR" -v n="$NEED_INR" 'BEGIN{exit !(c > n)}'; then
  echo "REFUSING TO LAUNCH -- the remaining credits, \$${CREDITS_USD} (Rs ${CREDITS_INR} at Rs ${INR_PER_USD}/\$)," >&2
  echo "do not exceed the planned AWS spend so far, Rs ${PLANNED_INR}, plus this block's cap," >&2
  echo "Rs ${COST_CAP_INR} (Rs ${NEED_INR} in all). Nothing was created." >&2
  exit 1
fi

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
# The hard-delete role's trust policy, read back and compared with the
# documented form field by field. The aws wrapper redacts the pinned account
# ID in everything it returns, so the read is compared with the wanted form
# redacted the same way; any other account ID would still show as digits.
GROUP_ARN="arn:aws:scheduler:${REGION}:${ACCOUNT}:schedule-group/default"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"
TRUST="$(aws iam get-role "${P[@]}" --role-name "$ROLE_NAME" \
  --query 'Role.AssumeRolePolicyDocument.[length(Statement), Statement[0].Effect, Statement[0].Principal.Service, Statement[0].Action, join(`,`, sort(keys(Statement[0].Condition))), join(`,`, sort(keys(Statement[0].Condition.StringEquals))), Statement[0].Condition.StringEquals."aws:SourceArn", Statement[0].Condition.StringEquals."aws:SourceAccount"]' \
  --output text 2>/dev/null)" || TRUST=""
TRUST_WANT="$(printf '1\tAllow\tscheduler.amazonaws.com\tsts:AssumeRole\tStringEquals\taws:SourceAccount,aws:SourceArn\t%s\t%s' "$GROUP_ARN" "$ACCOUNT")"
TRUST_WANT="${TRUST_WANT//$ACCOUNT/<account>}"
if [[ -z "$TRUST" || "$TRUST" != "$TRUST_WANT" ]]; then
  echo "REFUSING TO LAUNCH -- the trust policy of role $ROLE_NAME is not the documented form." >&2
  echo "  read: '${TRUST//$ACCOUNT/<account>}'" >&2
  echo "  want: '${TRUST_WANT}'" >&2
  echo "  (aws:SourceArn must name the schedule group, not a schedule.) Nothing was created." >&2
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
# "$SHUTDOWN_BEHAVIOR", so this stops the instance; the disk survives.
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
echo "  in-guest halt: +${HALT_MINUTES} min, STOPS the instance (GPU billing stops, disk survives)"
echo "  hard delete  : +${MAX_RUN_MINUTES} min, ec2:TerminateInstances by EventBridge Scheduler"
echo "  role trust   : ${ROLE_NAME}, aws:SourceArn = schedule-group/default (read back)"
echo "  worst case   : Rs ${WORST_INR}, cost cap Rs ${COST_CAP_INR}"
echo "  credits      : \$${CREDITS_USD} left (Rs ${CREDITS_INR}) > Rs ${PLANNED_INR} planned so far + Rs ${COST_CAP_INR} cap"
echo
read -r -p "Type 'launch' to proceed, anything else to abort: " CONFIRM
if [[ "$CONFIRM" != "launch" ]]; then
  echo "Aborted -- no instance created."
  exit 1
fi

# The prompt may have waited past midnight: the launch is the request.
aws_launch_policy_check "$REGION" $AZ

# Dry registration: AWS must accept a schedule using the role before any
# instance exists. A throwaway, DISABLED, thirty days out, for an instance
# id that cannot exist; read back, then deleted.
DRY="attnbench-dryreg-$(command date -u +%s)-$$"
DRY_AT="$(iso_utc $(( $(command date -u +%s) + 30 * 86400 )))"
DRY_TARGET="{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:terminateInstances\",\"RoleArn\":\"${ROLE_ARN}\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"i-00000000000000000\\\"]}\"}"
if ! aws scheduler create-schedule "${P[@]}" \
     --name "$DRY" \
     --state DISABLED \
     --schedule-expression "at(${DRY_AT})" \
     --schedule-expression-timezone UTC \
     --flexible-time-window Mode=OFF \
     --action-after-completion DELETE \
     --target "$DRY_TARGET" >/dev/null; then
  echo "REFUSING TO LAUNCH -- AWS refused the dry registration of a schedule using role" >&2
  echo "$ROLE_NAME (above). Nothing was created." >&2
  exit 1
fi
DRY_STATE="$(aws scheduler get-schedule "${P[@]}" --name "$DRY" --query State --output text || true)"
if ! aws scheduler delete-schedule "${P[@]}" --name "$DRY" >/dev/null; then
  echo "REFUSING TO LAUNCH -- the throwaway schedule $DRY could not be deleted; delete it" >&2
  echo "by hand (it is disabled and targets no real instance). Nothing was created." >&2
  exit 1
fi
if [[ "$DRY_STATE" != "DISABLED" ]]; then
  echo "REFUSING TO LAUNCH -- the throwaway schedule read back as '$DRY_STATE', not DISABLED" >&2
  echo "(it has been deleted). Nothing was created." >&2
  exit 1
fi
echo "Dry registration accepted (schedule $DRY, deleted)."

PLACEMENT=()
if [[ -n "$SUBNET" ]]; then
  PLACEMENT=(--subnet-id "$SUBNET")
elif [[ -n "$AZ" ]]; then
  PLACEMENT=(--placement "AvailabilityZone=$AZ")
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
  --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled,InstanceMetadataTags=enabled' \
  --block-device-mappings "DeviceName=$ROOT_DEVICE,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}" \
  --tag-specifications "ResourceType=instance,Tags=[$TAGS]" "ResourceType=volume,Tags=[$TAGS]" \
  --user-data "file://$STARTUP_SCRIPT" \
  --query 'Instances[0].InstanceId' --output text)" || INSTANCE_ID=""
# Every submitted create request goes into the launch record at its cap,
# created or not: an instance may exist even when no id came back.
mkdir -p "$(dirname "$LAUNCH_RECORD")"
printf '%s\t%s\t%s\t%s\n' "$(command date -u +%Y-%m-%dT%H:%M:%SZ)" "$PAIRING" \
  "${INSTANCE_ID:-no-id}" "$COST_CAP_INR" >> "$LAUNCH_RECORD"
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
TARGET="{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:terminateInstances\",\"RoleArn\":\"${ROLE_ARN}\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"${INSTANCE_ID}\\\"]}\"}"
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
