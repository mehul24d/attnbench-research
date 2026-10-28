# Sourced by build_fa3_wheel.sh (CPU build VM) and fa3_import_check.sh (H100).
#
# The FA3 pin (researcher's rule, 28 Oct 2026): the archived e9515d5d only if
# it had a documented successful build on this stack, otherwise the latest
# tagged release, never main. No build of e9515d5d is documented anywhere in
# the repository or its banked results (it is only named, as "main today", in
# docs/archive/estimator_frontier_preregistration.md). The latest release on
# 28 Oct 2026 (GitHub's releases/latest; the fa4-* tags are pre-releases of
# another package) is v2.8.3.post2; its tag object ec83c683 points at the
# commit below, which carries hopper/ at package version 3.0.0.b1.
FA3_REPO_URL="https://github.com/Dao-AILab/flash-attention.git"
FA3_TAG="v2.8.3.post2"
FA3_COMMIT="49774b31c8d1172a5b28f72e9aed0f322bf4ccba"

# The results bucket, as in run_phase.sh: ATTNBENCH_BUCKET, or the one named
# after the instance's own project, read from the metadata server. Sets
# BUCKET and FA3_PREFIX, or refuses. The project ID is never printed.
fa3_bucket() {
  FA3_PROJECT_ID=""
  if [[ -z "${ATTNBENCH_BUCKET:-}" ]]; then
    FA3_PROJECT_ID="$(curl -fs --max-time 5 -H 'Metadata-Flavor: Google' \
      "${ATTNBENCH_METADATA_URL:-http://metadata.google.internal/computeMetadata/v1/project/project-id}" \
      2>/dev/null || true)"
    if [[ ! "$FA3_PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
      echo "REFUSING -- no results bucket: ATTNBENCH_BUCKET is unset and the" >&2
      echo "instance's project could not be read from the metadata server." >&2
      exit 4
    fi
  fi
  BUCKET="${ATTNBENCH_BUCKET:-gs://attnbench-results-${FA3_PROJECT_ID}}"
  FA3_PREFIX="$BUCKET/fa3/$FA3_COMMIT"
}

fa3_redact() {
  if [[ -n "${FA3_PROJECT_ID:-}" ]]; then sed -e "s/${FA3_PROJECT_ID}/<project>/g"; else cat; fi
}
