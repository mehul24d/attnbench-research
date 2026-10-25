# GCP project pinning shared by every launcher, the cleanup check and the
# teardown. Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_gcp_project.sh"
#     gcp_project_pin                      # sets PROJECT, or refuses
#
# Why. On 23 Oct 2026 the workstation's gcloud default project was switched
# to a project of unrelated research, which holds an instance that is not
# this study's. Every script here used to fall back to that default. On
# 25 Oct 2026 the cleanup check listed the wrong project and reported this
# project's images as gone, and the teardown deletes whatever instance name
# it is given in whatever project gcloud points at.
#
# So the project comes from one variable kept OUTSIDE the repository,
# ATTNBENCH_GCP_PROJECT (set it in the shell profile), and is passed to every
# gcloud call as --project. The scripts also refuse while the active gcloud
# project (what `gcloud config get-value project` returns: the active
# configuration, or CLOUDSDK_CORE_PROJECT if set) is anything else, because
# then any command typed by hand during the session would go to the other
# project. Use a named configuration for this study, or export
# CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT" in the session's shell.
#
# The project ID is never printed. Refusals do not name either project, and
# gcloud's own output (resource URLs carry the project) passes through
# gcp_redact on both streams, which stay separate: the compile launcher reads
# create errors from stderr to tell a stockout from a bad request.

gcp_project_pin() {
  local pinned="${ATTNBENCH_GCP_PROJECT:-}"
  if [[ -z "$pinned" ]]; then
    echo "REFUSING -- ATTNBENCH_GCP_PROJECT is not set. Set it to this study's GCP" >&2
    echo "project in your shell profile (it is kept out of the repository)." >&2
    echo "Nothing was queried, created or deleted." >&2
    exit 1
  fi
  # A GCP project ID: 6 to 30 characters, lowercase letters, digits and
  # hyphens, starting with a letter and not ending with a hyphen. Checked so
  # that the value is also safe inside gcp_redact's sed expression.
  if [[ ! "$pinned" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
    echo "REFUSING -- ATTNBENCH_GCP_PROJECT is not a valid GCP project ID" >&2
    echo "(value not printed). Nothing was queried, created or deleted." >&2
    exit 1
  fi
  local active
  active="$(command gcloud config get-value project 2>/dev/null || true)"
  if [[ "$active" != "$pinned" ]]; then
    echo "REFUSING -- the active gcloud project is not ATTNBENCH_GCP_PROJECT" >&2
    echo "(neither is printed). Activate this study's gcloud configuration, or run" >&2
    echo '    export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"' >&2
    echo "in this shell, then re-run. Nothing was queried, created or deleted." >&2
    exit 1
  fi
  PROJECT="$pinned"
}

gcp_redact() {
  if [[ -n "${PROJECT:-}" ]]; then sed -e "s/${PROJECT}/<project>/g"; else cat; fi
}

# Every gcloud call in a script that sources this file goes through here, so
# the redaction cannot be forgotten at a new call site. Output is buffered
# until the call returns, and the exit status is gcloud's own.
gcloud() {
  local out err rc=0
  out="$(mktemp)"; err="$(mktemp)"
  command gcloud "$@" >"$out" 2>"$err" || rc=$?
  gcp_redact <"$out"
  gcp_redact <"$err" >&2
  rm -f "$out" "$err"
  return "$rc"
}
