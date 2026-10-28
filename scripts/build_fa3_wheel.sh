#!/usr/bin/env bash
# Build the FlashAttention-3 wheel (Dao-AILab/flash-attention, hopper/) from
# the pinned source, store it in the results bucket with its SHA-256, and
# read it back. Runs on the CPU build VM (scripts/gcp_launch_fa3_build.sh,
# docs/t64_fa3_launch_card.md), booted from the same disk image as the H100
# so the wheel matches the H100's Python and torch. Never on the Mac.
#
# Usage:  bash scripts/build_fa3_wheel.sh
#
# Order: everything cheap and refusable first (bucket, memory guard, ninja,
# toolchain and stack, a bucket write), then the source, then the build. A
# refusal costs minutes; a build that fails at the upload costs the session.
#
# Exit codes: 0 wheel stored and read back; 1 a check refused before the
# build; 2 the build failed or hit its deadline (logs are stored); 3 the
# stored wheel did not read back with the same hash; 4 no bucket.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/_fa3_pin.sh"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# MAX_JOBS x NVCC_THREADS x 5 GB must fit in free memory (attnbench/build_guards.py).
# n2-standard-16 has 64 GB: 8 x 1 leaves room if an FA3 frontend is bigger
# than FA2's measured 5 GB. The sm_90a-only build has one gencode target, so
# NVCC_THREADS above 1 buys nothing.
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
# Stop the build here, not at the VM's in-guest halt, so the logs (and a
# wheel, if there is one) are stored before the machine stops.
BUILD_TIMEOUT_MIN="${FA3_BUILD_TIMEOUT_MIN:-225}"   # the VM halts at 270
# The H100 image's stack (docs/limitations.md, docs/retry_session_runbook.md:
# Python 3.10.12, torch 2.9.1+cu129). Overridable only so the tests can run on
# a workstation; the manifest records what was found either way.
EXPECT_PYTHON="${FA3_EXPECT_PYTHON:-3.10}"
EXPECT_TORCH="${FA3_EXPECT_TORCH:-2.9.1+cu129}"
MIN_NVCC="12.3"   # hopper/setup.py refuses below this
WORK="${FA3_WORK_DIR:-$HOME/fa3_build}"
SRC="$WORK/src"; OUT="$WORK/out"; DIST="$OUT/dist"

# 1. The bucket. Refuses (exit 4) before any work.
fa3_bucket
mkdir -p "$OUT"
LOG="$OUT/build.log"
echo "fa3 build: tag $FA3_TAG commit $FA3_COMMIT -> $FA3_PREFIX" | fa3_redact

# 2. Memory guard, as in build_flash_attn.sh: it refuses, it does not warn.
python3 -m attnbench.build_guards --max-jobs "$MAX_JOBS" --nvcc-threads "$NVCC_THREADS" \
  ${FA_FREE_GB_OVERRIDE:+--free-gb "$FA_FREE_GB_OVERRIDE"} || exit 1

# 3. ninja on PATH, or torch compiles serially with no warning.
export PATH="$HOME/.local/bin:$PATH"
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH (pip install --user ninja)" >&2; exit 1; }
python3 -c 'from torch.utils.cpp_extension import is_ninja_available
assert is_ninja_available(), "torch cannot see ninja"' || exit 1

# 4. Toolchain and stack against the H100 image.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null; then CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else CUDA_HOME=/usr/local/cuda; fi
fi
export CUDA_HOME
NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p' || true)"
python3 -m attnbench.fa3_wheel stack --nvcc "$NVCC_VERSION" --expect-python "$EXPECT_PYTHON" \
  --expect-torch "$EXPECT_TORCH" --min-nvcc "$MIN_NVCC" --out "$OUT/stack.json" || exit 1

# 5. A write to the bucket now, not after the build (the 2026-09-15 loss was
# a 403 found at the end).
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"started": "%s", "commit": "%s", "tag": "%s"}\n' "$STARTED" "$FA3_COMMIT" "$FA3_TAG" > "$OUT/started.json"
gcloud storage cp "$OUT/started.json" "$FA3_PREFIX/started.json" 2>&1 | fa3_redact || {
  echo "FATAL: cannot write to the results bucket; nothing was built" >&2; exit 1; }

# 6. The pinned source: fetched by commit, checked against the pin and against
# the tag upstream, with the CUTLASS submodule at the commit the pin records.
rm -rf "$SRC"; mkdir -p "$SRC"
git -C "$SRC" init -q
git -C "$SRC" remote add origin "$FA3_REPO_URL"
git -C "$SRC" fetch -q --depth 1 origin "$FA3_COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD
HEAD_COMMIT="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$HEAD_COMMIT" != "$FA3_COMMIT" ]]; then
  echo "FATAL: checked out $HEAD_COMMIT, the pin is $FA3_COMMIT" >&2; exit 1
fi
TAG_COMMIT="$(git -C "$SRC" ls-remote origin "refs/tags/${FA3_TAG}^{}" | cut -f1)"
if [[ "$TAG_COMMIT" != "$FA3_COMMIT" ]]; then
  echo "FATAL: upstream $FA3_TAG now points at '${TAG_COMMIT}', not the pin" >&2; exit 1
fi
git -C "$SRC" submodule update -q --init --depth 1 csrc/cutlass
CUTLASS_COMMIT="$(git -C "$SRC/csrc/cutlass" rev-parse HEAD)"

# 7. The build. FLASH_ATTENTION_FORCE_BUILD=TRUE is not optional: without it
# hopper/setup.py's bdist_wheel first tries to download a prebuilt wheel from
# GitHub and installs that if it exists -- a third-party binary in place of
# this build. The three DISABLE flags drop what the backend never calls
# (backward, FP8, the sm_80 kernels) and declares unsupported
# (attnbench/backends/fa3.py); no head dim is disabled. Upstream fetches
# nvcc 12.6.85 / ptxas 12.8.93 from NVIDIA unless the local CUDA is 12.8; the
# manifest records that.
BUILD_FLAGS=(FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
             FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE)
MEMLOG="$OUT/memory.tsv"
( while :; do
    if [[ -r /proc/meminfo ]]; then
      avail="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
      cicc="$(ps -C cicc -o rss= 2>/dev/null | sort -n | tail -1 | tr -d ' ')"
      n="$(ps -C cicc -o pid= 2>/dev/null | wc -l | tr -d ' ')"
      printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$avail" "${cicc:-0}" "$n" >> "$MEMLOG"
    fi
    sleep 15
  done ) >/dev/null 2>&1 &   # not on stdout: a held pipe outlives the script
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
T0="$(date +%s)"
echo "START $(date -u +%H:%M:%SZ) jobs=$MAX_JOBS nvcc_threads=$NVCC_THREADS deadline=${BUILD_TIMEOUT_MIN}m"
set +e
( cd "$SRC/hopper" && env "${BUILD_FLAGS[@]}" \
    timeout "${BUILD_TIMEOUT_MIN}m" python3 setup.py bdist_wheel --dist-dir "$DIST" ) > "$LOG" 2>&1
RC=$?
set -e
WALL_S=$(( $(date +%s) - T0 ))
kill "$SAMPLER" 2>/dev/null || true
echo "END $(date -u +%H:%M:%SZ) rc=$RC wall=${WALL_S}s"
NINJA_LOG="$(find "$SRC/hopper/build" -name .ninja_log 2>/dev/null | head -1 || true)"
[[ -n "$NINJA_LOG" ]] && cp "$NINJA_LOG" "$OUT/ninja_log.txt"

store_logs() {   # $1: destination prefix
  local f
  for f in build.log memory.tsv ninja_log.txt stack.json; do
    [[ -f "$OUT/$f" ]] && gcloud storage cp "$OUT/$f" "$1/$f" 2>&1 | fa3_redact
  done
  return 0
}

WHEELS=()
if [[ -d "$DIST" ]]; then while IFS= read -r w; do WHEELS+=("$w"); done < <(find "$DIST" -name '*.whl'); fi
if (( RC != 0 )) || (( ${#WHEELS[@]} != 1 )) || grep -q "Guessing wheel URL" "$LOG"; then
  FAILED="$FA3_PREFIX/failed-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "FATAL: no usable wheel (rc=$RC, wheels=${#WHEELS[@]}; rc 124 is the ${BUILD_TIMEOUT_MIN}m deadline)." >&2
  grep -q "Guessing wheel URL" "$LOG" && echo "FATAL: setup.py tried to download a prebuilt wheel." >&2
  store_logs "$FAILED"
  echo "Logs stored under $FAILED" | fa3_redact
  exit 2
fi
WHEEL="${WHEELS[0]}"

# 8. Manifest: the hash, the pin, the stack, the flags, the measured cost.
TOOLCHAIN="local CUDA $NVCC_VERSION"
[[ -x "$SRC/third_party/nvidia/backend/bin/nvcc" ]] && TOOLCHAIN="upstream download (nvcc 12.6.85, ptxas 12.8.93); local CUDA $NVCC_VERSION"
FIELDS="$(python3 - "$OUT/stack.json" "$MEMLOG" <<EOF
import json, sys
mem = [l.split("\t") for l in open(sys.argv[2]).read().split("\n") if l] if __import__("os").path.exists(sys.argv[2]) else []
print(json.dumps({
    "repo": "$FA3_REPO_URL", "tag": "$FA3_TAG", "commit": "$FA3_COMMIT",
    "cutlass_commit": "$CUTLASS_COMMIT", "build_flags": "${BUILD_FLAGS[*]}".split(),
    "max_jobs": $MAX_JOBS, "nvcc_threads": $NVCC_THREADS, "toolchain": "$TOOLCHAIN",
    "stack": json.load(open(sys.argv[1])), "started_utc": "$STARTED", "wall_s": $WALL_S,
    "min_mem_available_kb": min((int(m[1]) for m in mem), default=None),
    "max_cicc_rss_kb": max((int(m[2]) for m in mem), default=None),
    "max_cicc_processes": max((int(m[3]) for m in mem), default=None),
}))
EOF
)"
SHA="$(python3 -m attnbench.fa3_wheel manifest --wheel "$WHEEL" --out "$OUT/manifest.json" --fields "$FIELDS")"
printf '%s  %s\n' "$SHA" "$(basename "$WHEEL")" > "$OUT/SHA256SUMS"

# 9. Store, then read the wheel back and hash it again.
DEST="$FA3_PREFIX"
gcloud storage cp "$WHEEL" "$DEST/$(basename "$WHEEL")" 2>&1 | fa3_redact
gcloud storage cp "$OUT/manifest.json" "$DEST/manifest.json" 2>&1 | fa3_redact
gcloud storage cp "$OUT/SHA256SUMS" "$DEST/SHA256SUMS" 2>&1 | fa3_redact
store_logs "$DEST"
mkdir -p "$OUT/readback"
gcloud storage cp "$DEST/$(basename "$WHEEL")" "$OUT/readback/" 2>&1 | fa3_redact
BACK="$(python3 -c 'import sys; from attnbench.fa3_wheel import sha256; print(sha256(sys.argv[1]))' \
        "$OUT/readback/$(basename "$WHEEL")" 2>/dev/null || true)"
if [[ "$BACK" != "$SHA" ]]; then
  echo "FATAL: the stored wheel reads back as '${BACK}', not $SHA" >&2; exit 3
fi
echo
echo "FA3 wheel stored and read back."
echo "  wheel  : $(basename "$WHEEL")"
echo "  sha256 : $SHA"
echo "  commit : $FA3_COMMIT ($FA3_TAG), cutlass $CUTLASS_COMMIT"
echo "  where  : $DEST/" | fa3_redact
echo "  wall   : ${WALL_S}s"
echo "Record the sha256; the H100 import check takes it as its argument."
