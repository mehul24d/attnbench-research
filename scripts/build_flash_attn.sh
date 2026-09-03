#!/usr/bin/env bash
# Build flash-attn from source, scoped to the architectures this study can
# actually run on and parallelised properly.
#
# Every setting here exists because its absence cost real billed time on
# 2026-09-02. None of them are tuning preferences; each one is a fix for a
# specific silent failure.
#
# Usage:
#   bash scripts/build_flash_attn.sh [version]
set -euo pipefail

VERSION="${1:-2.8.3.post1}"
SRC_DIR="${FA_SRC_DIR:-$HOME/fa}"

# Concurrency settings are set here, at the top, so the memory guard in
# section 1 can reject an impossible configuration before ANY work happens.
# The reasoning behind each value is in section 3 -- read that before
# changing either of them, and note that they MULTIPLY.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

# ---------------------------------------------------------------------------
# 1. Memory guard. FIRST, because it is pure arithmetic and needs nothing.
# ---------------------------------------------------------------------------
# It runs ahead of the ninja and toolchain checks deliberately: it is the
# cheapest check, it depends on no environment, and it guards the failure
# that is by far the most expensive to discover late (the OOM killer took a
# build 90 minutes in on 2026-09-02, after SSH had already died).
#
# This REFUSES; it does not warn. A warning in a build log on a machine whose
# SSH is about to die is not a control.
#
# The logic lives in attnbench/build_guards.py rather than inline so it is
# unit-testable -- including a test that runs THIS script with an impossible
# configuration and asserts it aborts before downloading anything
# (tests/test_build_guards.py). A guard nobody has watched fire is a guess.
#
# FA_FREE_GB_OVERRIDE lets that test force the memory figure without needing
# a machine of a particular size. It is never set in real use.
GUARD_ARGS=(--max-jobs "$MAX_JOBS" --nvcc-threads "$NVCC_THREADS")
if [[ -n "${FA_FREE_GB_OVERRIDE:-}" ]]; then
  GUARD_ARGS+=(--free-gb "$FA_FREE_GB_OVERRIDE")
fi
python3 -m attnbench.build_guards "${GUARD_ARGS[@]}" || exit 1

# ---------------------------------------------------------------------------
# 2. ninja must be on PATH, or torch silently compiles serially.
# ---------------------------------------------------------------------------
# pip installs ninja to ~/.local/bin, which is not on the default PATH on a
# GCP Deep Learning VM. torch.utils.cpp_extension.is_ninja_available() then
# returns False and BuildExtension falls back to serial distutils -- one nvcc
# at a time, with MAX_JOBS inert because MAX_JOBS only controls ninja. There
# is no warning; the build just takes 5x longer and looks like "compiles are
# slow". Measured: 0.33 files/min serial vs ~1.6 parallel over 72 targets.
export PATH="$HOME/.local/bin:$PATH"
command -v ninja >/dev/null || {
  echo "FATAL: ninja not on PATH -- the build would silently go serial." >&2
  echo "  pip install --user ninja" >&2
  exit 1
}
python3 -c 'from torch.utils.cpp_extension import is_ninja_available
assert is_ninja_available(), "torch cannot see ninja despite it being on PATH"' || exit 1

# ---------------------------------------------------------------------------
# 3. Architecture scoping. PERMANENT -- every rebuild should inherit this.
# ---------------------------------------------------------------------------
# setup.py's default is "80;90;100;120". sm_100/sm_120 are Blackwell, which
# CLAUDE.md's hardware ceiling (24GB L4/4090, or 80GB H100 if the pending
# quota lands) never reaches -- they were half the compile work and none of
# the value.
#
#   sm_80  what an L4 (sm_89) actually executes, via Ampere minor-version
#          binary compatibility. flash-attn exposes no sm_89 target, so this
#          is the only option for the card this study runs on.
#   sm_90  kept so a built image stays usable if the H100 quota lands.
#
# NOTE: this is FLASH_ATTN_CUDA_ARCHS, not TORCH_CUDA_ARCH_LIST. flash-attn's
# setup.py genuinely does ignore TORCH_CUDA_ARCH_LIST -- checking only that
# one and concluding "this build cannot be scoped" is how the 4-architecture
# build happened. Each package gets checked for its own hook.
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;90}"

# ---------------------------------------------------------------------------
# 4. Job count. MEMORY is the binding constraint here, not cores.
# ---------------------------------------------------------------------------
# setup.py picks min(cpu_count//2, free_gb/9) = 2 on a g2-standard-8, and
# that /9 is not conservative padding -- it encodes real knowledge about how
# much memory nvcc's frontend needs. Overriding it to 5 produced 8 OOM-kill
# events on 2026-09-02 and cost the session its build.
#
# What the OOM log actually measured: the memory hog is `cicc` (nvcc's CUDA
# frontend), consistently 4.5-5.0 GB resident EACH. And the count of
# concurrent cicc processes is not MAX_JOBS -- it is:
#
#     concurrent cicc  ~=  MAX_JOBS x NVCC_THREADS
#
# because --threads spawns a separate frontend per gencode target. So
# MAX_JOBS=5 x NVCC_THREADS=2 = ~10 cicc x ~5 GB = ~50 GB demanded on a
# 31 GB box. The machine had no chance.
#
# On ~28 GB usable, the ceiling is about 5 concurrent cicc. Hence 4 x 1.
# NOTE: MAX_JOBS=4 with NVCC_THREADS=2 (an earlier "fix" here) is 8 cicc and
# would still have OOM'd -- raising job count while leaving NVCC_THREADS at
# 2 is the trap, because the two multiply.
#
# NVCC_THREADS is 1, not the setup.py default of 2. Each extra nvcc thread is
# another ~5 GB cicc, and --threads only parallelises across gencode targets
# (of which there are 2) -- it buys a little latency per file at
# multiplicative memory cost. Parallelism belongs in MAX_JOBS, where ninja
# schedules it across the whole 72-target list instead.
#
# Both values are exported at the top of this script so the memory guard can
# see them; this section is where they are explained, not where they are set.

mkdir -p "$SRC_DIR" && cd "$SRC_DIR"
if [[ ! -d "flash_attn-${VERSION}" ]]; then
  # `pip download --no-binary :all:` runs the build-requirements hook, which
  # fails here for the same reason the build needs --no-build-isolation.
  # Fetch the sdist directly instead.
  URL=$(python3 -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/flash-attn/${VERSION}/json'))
print([u['url'] for u in d['urls'] if u['packagetype'] == 'sdist'][0])")
  echo "fetching ${URL}"
  curl -sL -o "fa-${VERSION}.tar.gz" "$URL"
  tar xzf "fa-${VERSION}.tar.gz"
fi

cd "flash_attn-${VERSION}"
echo "START $(date -u +%H:%M:%SZ)  archs=${FLASH_ATTN_CUDA_ARCHS}  jobs=${MAX_JOBS}"
# --no-build-isolation: pip's isolated build env cannot see the installed
# torch, and the build dies in get_requires_for_build_wheel without it. That
# in turn means build deps are NOT provided for you -- psutil in particular
# is imported by setup.py itself and must already be present.
python3 -c 'import psutil' 2>/dev/null || pip install --user -q psutil
pip install --user --no-build-isolation .
echo "END $(date -u +%H:%M:%SZ) rc=$?"
