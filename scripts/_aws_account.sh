# AWS account and profile pinning shared by the AWS launcher, the cleanup
# check and the teardown (task T5.1). The AWS counterpart of
# scripts/_gcp_project.sh. Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_aws_account.sh"
#     aws_account_pin              # sets ACCOUNT and AWS_PIN_PROFILE, or refuses
#
# Why. The GCP scripts once fell back to the workstation's default project,
# which had been switched to another study's project (25 Oct 2026). The AWS
# CLI has the same hazard twice over: the default profile, AWS_PROFILE and
# credentials in the environment each decide which account a call reaches.
#
# So two variables are kept OUTSIDE the repository (set them in the shell
# profile):
#   ATTNBENCH_AWS_ACCOUNT  the study's 12-digit account ID
#   ATTNBENCH_AWS_PROFILE  the named CLI profile for that account
# Every aws call passes --profile "$AWS_PIN_PROFILE"; the wrapper below
# refuses a call that does not. Before anything else the scripts check, with
# the read-only `aws sts get-caller-identity`, that
#   1. the pinned profile resolves to the pinned account, and
#   2. the identity a hand-typed command would use (no --profile: the
#      environment's credentials, AWS_PROFILE or the default profile) is the
#      same account,
# and refuse otherwise, because then any command typed by hand during the
# session would go to another account. To satisfy (2), run
#     export AWS_PROFILE="$ATTNBENCH_AWS_PROFILE"
# in the session's shell, with no AWS_ACCESS_KEY_ID set.
#
# Neither value is ever printed. Refusals name neither, and the CLI's own
# output (ARNs carry the account ID) passes through aws_redact on both
# streams, which stay separate.

AWS_STS_REGION="us-east-1"   # STS is global; a region is passed so the call
                             # never depends on the shell's default region.

aws_account_pin() {
  local pinned="${ATTNBENCH_AWS_ACCOUNT:-}" profile="${ATTNBENCH_AWS_PROFILE:-}"
  if [[ -z "$pinned" || -z "$profile" ]]; then
    echo "REFUSING -- ATTNBENCH_AWS_ACCOUNT and ATTNBENCH_AWS_PROFILE must both be set" >&2
    echo "to this study's AWS account and CLI profile in your shell profile (they are" >&2
    echo "kept out of the repository). Nothing was queried, created or deleted." >&2
    exit 1
  fi
  if [[ ! "$pinned" =~ ^[0-9]{12}$ ]]; then
    echo "REFUSING -- ATTNBENCH_AWS_ACCOUNT is not a 12-digit AWS account ID (value" >&2
    echo "not printed). Nothing was queried, created or deleted." >&2
    exit 1
  fi
  # Letters, digits, hyphen and underscore only, so the value can be compared
  # and passed as one word.
  if [[ ! "$profile" =~ ^[A-Za-z0-9_-]{1,64}$ ]]; then
    echo "REFUSING -- ATTNBENCH_AWS_PROFILE is not a plain profile name (letters," >&2
    echo "digits, '-' and '_'; value not printed). Nothing was queried, created or deleted." >&2
    exit 1
  fi
  local via_profile active
  via_profile="$(command aws sts get-caller-identity --profile "$profile" \
                   --region "$AWS_STS_REGION" --query Account --output text 2>/dev/null || true)"
  if [[ "$via_profile" != "$pinned" ]]; then
    echo "REFUSING -- the profile ATTNBENCH_AWS_PROFILE does not resolve to" >&2
    echo "ATTNBENCH_AWS_ACCOUNT (neither is printed), or its credentials did not answer." >&2
    echo "Nothing was queried, created or deleted." >&2
    exit 1
  fi
  active="$(command aws sts get-caller-identity \
              --region "$AWS_STS_REGION" --query Account --output text 2>/dev/null || true)"
  if [[ "$active" != "$pinned" ]]; then
    echo "REFUSING -- the active AWS identity is not ATTNBENCH_AWS_ACCOUNT (neither is" >&2
    echo "printed). Unset AWS_ACCESS_KEY_ID and friends, then run" >&2
    echo '    export AWS_PROFILE="$ATTNBENCH_AWS_PROFILE"' >&2
    echo "in this shell, and re-run. Nothing was queried, created or deleted." >&2
    exit 1
  fi
  ACCOUNT="$pinned"
  AWS_PIN_PROFILE="$profile"
}

# Only the account ID is redacted. API responses never carry the profile
# name, and redacting a short name such as "attnbench" would garble the tag
# values the scripts parse; the profile is simply never echoed.
aws_redact() {
  if [[ -n "${ACCOUNT:-}" ]]; then
    sed -e "s/${ACCOUNT}/<account>/g"
  else
    cat
  fi
}

# Every aws call in a script that sources this file goes through here, so the
# redaction cannot be forgotten at a new call site, and a call that does not
# carry the pinned profile is refused before it runs. Output is buffered
# until the call returns, and the exit status is the CLI's own.
aws() {
  if [[ -z "${AWS_PIN_PROFILE:-}" ]]; then
    echo "REFUSING aws call -- aws_account_pin has not run." >&2
    return 97
  fi
  local a prev="" has_profile=0
  for a in "$@"; do
    if [[ "$prev" == "--profile" && "$a" == "$AWS_PIN_PROFILE" ]] \
       || [[ "$a" == "--profile=$AWS_PIN_PROFILE" ]]; then
      has_profile=1
    fi
    prev="$a"
  done
  if [[ "$has_profile" -ne 1 ]]; then
    echo "REFUSING aws call without the pinned --profile ($1 $2)." >&2
    return 97
  fi
  local out err rc=0
  out="$(mktemp)"; err="$(mktemp)"
  command aws "$@" >"$out" 2>"$err" || rc=$?
  aws_redact <"$out"
  aws_redact <"$err" >&2
  rm -f "$out" "$err"
  return "$rc"
}
