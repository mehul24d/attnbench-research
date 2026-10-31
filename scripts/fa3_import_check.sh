#!/usr/bin/env bash
# On the H100: fetch the FA3 wheel stored by build_fa3_wheel.sh, check it is
# the recorded one and was built for this Python and torch, install it, and
# check that it imports and the fa3 backend is available.
#
# Usage:  bash scripts/fa3_import_check.sh <sha256 recorded from the build>
#
# The hash is an argument, not read from the bucket, so a replaced wheel and
# a replaced manifest cannot agree with each other and pass. Exit 0 means the
# backend is available; anything else means the H100 record says
# dense_set_incomplete="fa3:<reason>" (docs/t64_fa3_plan.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/_fa3_pin.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPECTED="${1:-}"
if [[ ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: bash scripts/fa3_import_check.sh <sha256 of the wheel, 64 hex>" >&2
  exit 1
fi
fa3_bucket
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

gcloud storage cp "$FA3_PREFIX/manifest.json" "$WORK/manifest.json" 2>&1 | fa3_redact
WHEEL_NAME="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["wheel"])' "$WORK/manifest.json")"
if [[ ! "$WHEEL_NAME" =~ ^[A-Za-z0-9_.+-]+\.whl$ ]]; then
  echo "FATAL: the manifest names no plausible wheel file" >&2; exit 1
fi
gcloud storage cp "$FA3_PREFIX/$WHEEL_NAME" "$WORK/$WHEEL_NAME" 2>&1 | fa3_redact

fa3_module fa3_wheel check --wheel "$WORK/$WHEEL_NAME" \
  --manifest "$WORK/manifest.json" --sha256 "$EXPECTED"
# --no-deps: the wheel lists torch, and pip must not touch the pinned torch.
python3 -m pip install -q --no-deps --force-reinstall "$WORK/$WHEEL_NAME"
fa3_module fa3_wheel import
echo "fa3 $FA3_TAG ($FA3_COMMIT) wheel $EXPECTED: installed and available."
