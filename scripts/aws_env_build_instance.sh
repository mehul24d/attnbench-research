#!/usr/bin/env bash
# Build the study's environment on the AWS environment-build instance, steps
# 2 to 6 of docs/launch_cards/aws_env_build_m7a.md (adviser's rulings dated
# 21 Nov 2026, fourth wave, item 3). Runs ON the instance, from the deployed
# repository's root, after the deploy (card step 1); the Mac then stops the
# instance and makes the AMI (card step 7). It times nothing.
#
#   FA3_WHEEL_SHA256=<64 hex> bash scripts/aws_env_build_instance.sh FA3_WHEEL
#
# The environment is image v6's (docs/image_libraries/attnbench-env-v6-20260917/),
# the one P1, P2 and P4 ran: the system Python 3.10.12, its packages as root
# and the session user's site, no venv. In order, stopping at the first
# failure, every step logged under ENV_BUILD_OUT:
#   0. guards: a clean tree, Python 3.10.12, a new output directory, the FA3
#      wheel and its recorded sha256, the FA3 build's manifest (committed,
#      configs/t64/fa3_manifest_49774b31.json, the bucket's copy read 21 Nov
#      2026) at its recorded sha256, nvcc 12.x, and flash-attn's memory
#      guard at MAX_JOBS, so a build the instance cannot hold is refused in
#      seconds, not after the installs; then cuobjdump, found in this order
#      (adviser's rulings dated 22 Nov 2026, probe): the toolkit nvcc came
#      from, then any toolkit under /usr/local at or above nvcc's version
#      (lowest first: plan default, flagged), then `apt-get install
#      cuda-cuobjdump-<major>-<minor>` for nvcc's version (once more after an
#      `apt-get update` if the shipped cache fails), else a refusal. The
#      probe of 21 Nov 2026 22:16 UTC (docs/launch_cards/
#      aws_env_build_probe_20261121.txt) found nvcc in all four toolkits of
#      the base image and cuobjdump in none, and the NVIDIA repository's
#      cuda-cuobjdump-12-9 (12.9.82) in apt's cache. Any refusal or failure
#      prints the full tool inventory;
#   2. v6's packages from an index, exact versions, --no-deps: the system list
#      as root, less the 69 packages apt installed on v6 (they come with the
#      base image), --ignore-installed so v6's version shadows an apt copy at
#      another (plan default, flagged), then the user site. Local builds
#      are left to step 4; xformers'
#      local build (0.0.33+5d4b92a5.d20251029) has no index copy, and PyPI's
#      0.0.33 stands in (xformers is no arm and is not in the library check:
#      plan default, flagged). torch and triton must then read 2.9.1+cu129
#      and 3.5.1;
#   3. the FA3 wheel: attnbench.fa3_wheel.check_wheel in full against the
#      manifest (sha256, name, the build's Python and torch, tags), installed,
#      NOT imported (no GPU here; P6's fa3_import step imports it);
#   4. from source: flash-attn 2.8.3.post1 (scripts/build_flash_attn.sh, its
#      memory guard), block-sparse-attention at bsa_contract.BSA_COMMIT, and
#      x-attention at xattention.XATTN_COMMIT (editable, outside the tree, as
#      scripts/install_xattention.sh does; its import is left to the GPU
#      rehearsal). No official flash-attn 2.8.3.post1 wheel is built for
#      torch 2.9 (its GitHub release has torch 2.4 to 2.8 only, read 21 Nov
#      2026), so it is compiled. Architectures: TORCH_CUDA_ARCH_LIST covers
#      the L4 (8.9) and the H100 (9.0), for any extension that reads it.
#      flash-attn and block-sparse-attention ignore it: their setup.py read
#      FLASH_ATTN_CUDA_ARCHS and BLOCK_SPARSE_ATTN_CUDA_ARCHS and accept 80,
#      90, 100, 110, 120, never 89 (an 89 is dropped without a word). The
#      L4 runs their sm_80 code by Ampere minor-version binary compatibility,
#      so both are 80;90, v6's archs; block-sparse-attention's default would
#      add three Blackwell archs. Each built library's cubins are then read
#      with cuobjdump and must be sm_80 and sm_90 exactly;
#   5. CPU smoke: attnbench, flash_attn and block_sparse_attn import (no GPU,
#      no driver: the import error is logged), and this environment's libraries equal
#      P2's (configs/t64/p2_libraries.json, FA3 the one allowed difference);
#      then a CPU test subset;
#   6. the record: pip freeze (user, user site, root), dpkg -l, uname, each
#      diffed against v6's list, SHA256SUMS; the tree still clean.
# ENV_BUILD_OK is printed last, only when every step passed.
#
# ENV_BUILD_HELPER_PYTHON and ENV_BUILD_FA_SCRIPT exist for the fake-command
# test (tests/test_aws_env_build_instance.py); they are never set in real use.
set -euo pipefail

USAGE="usage: FA3_WHEEL_SHA256=<sha256> bash scripts/aws_env_build_instance.sh FA3_WHEEL"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WHEEL="${1:-}"
MANIFEST="configs/t64/fa3_manifest_49774b31.json"
MANIFEST_SHA256="3c2f965784d8ae9e3df81048c5ac579d309554b8ce9472783d894837e7bade87"
FA3_SHA="${FA3_WHEEL_SHA256:-}"
PY="${ATTNBENCH_PYTHON:-/usr/bin/python3}"
HELPER_PY="${ENV_BUILD_HELPER_PYTHON:-$PY}"
FA_BUILD="${ENV_BUILD_FA_SCRIPT:-scripts/build_flash_attn.sh}"
OUT="${ENV_BUILD_OUT:-$HOME/env_build_$(date -u +%Y%m%dT%H%MZ)}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
# The architectures (header, step 4). TORCH_CUDA_ARCH_LIST for the ruling's
# L4 and H100; the two packages' own variables are what they read.
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
export FLASH_ATTN_CUDA_ARCHS="80;90"
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90"
WANT_SM="sm_80 sm_90"
# v6's toolkit was CUDA 12.9 (the FA3 manifest's stack); the newest 12.x
# otherwise. CUDA_HOME is what torch's extension builder reads.
# ENV_BUILD_CUDA_ROOT exists for the fake-command test; it is /usr/local.
CUDA_ROOT="${ENV_BUILD_CUDA_ROOT:-/usr/local}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -d "$CUDA_ROOT/cuda-12.9" ]]; then CUDA_HOME="$CUDA_ROOT/cuda-12.9"; else CUDA_HOME="$CUDA_ROOT/cuda"; fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
LIBS="docs/image_libraries/attnbench-env-v6-20260917"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
XFORMERS_STANDIN="xformers==0.0.33"
FA2_VERSION="2.8.3.post1"
BSA_URL="https://github.com/mit-han-lab/Block-Sparse-Attention.git"
XATTN_DIR="${XATTN_DIR:-$HOME/x-attention}"
HELPER=(scripts/env_build_lists.py)

NVCC_REL=""
CUOBJDUMP=""
tool_inventory() {   # every CUDA tool the build could use, to stderr
  {
    echo "-- tool inventory --"
    echo "CUDA_HOME=$CUDA_HOME nvcc=${NVCC_REL:-none} cuobjdump=${CUOBJDUMP:-none}"
    echo "on PATH: nvcc=$(command -v nvcc || echo none) cuobjdump=$(command -v cuobjdump || echo none)"
    local d
    for d in "$CUDA_ROOT"/cuda*; do
      [[ -e "$d" ]] || continue
      if [[ -L "$d" ]]; then echo "$d -> $(readlink "$d")"; continue; fi
      echo "$d: nvcc $("$d/bin/nvcc" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' || true)," \
        "cuobjdump $([[ -x "$d/bin/cuobjdump" ]] && echo present || echo absent)"
      echo "  bin: $(ls "$d/bin" 2>/dev/null | tr '\n' ' ')"
    done
    echo "cuobjdump files: $(find "$CUDA_ROOT" /usr/bin /opt -maxdepth 4 -name cuobjdump 2>/dev/null | tr '\n' ' ')"
    if [[ "$NVCC_REL" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "apt: $(apt-cache policy "cuda-cuobjdump-${NVCC_REL/./-}" 2>/dev/null | tr -s ' \n' ' ' || true)"
    fi
    echo "-- end of tool inventory --"
  } >&2
}
refuse() { echo "REFUSING: $*. Nothing was installed." >&2; tool_inventory; exit 1; }
fail() { echo "FAILED at step $STEP: $*" >&2; tool_inventory; exit 1; }
find_cuobjdump() {   # the search order (header, step 0); prints the path
  local maj="${NVCC_REL%%.*}" min="${NVCC_REL#*.}" d v
  [[ -x "$CUDA_HOME/bin/cuobjdump" ]] && { echo "$CUDA_HOME/bin/cuobjdump"; return 0; }
  for d in "$CUDA_ROOT"/cuda-*; do
    [[ -d "$d" && ! -L "$d" ]] || continue
    v="${d##*/cuda-}"
    [[ "$v" =~ ^[0-9]+\.[0-9]+$ ]] && echo "${v%%.*} ${v#*.} $d"
  done | sort -n -k1,1 -k2,2 | while read -r a b d; do
    if (( a > maj || (a == maj && b >= min) )) && [[ -x "$d/bin/cuobjdump" ]]; then
      echo "$d/bin/cuobjdump"; break
    fi
  done
}
pin_of() {   # pin_of FILE NAME: the 40-hex constant NAME = "..." in FILE
  local v
  v="$(sed -nE "s/^$2 = \"([0-9a-f]{40})\".*/\1/p" "$1")"
  [[ "$v" =~ ^[0-9a-f]{40}$ ]] || refuse "no 40-hex $2 in $1"
  printf '%s\n' "$v"
}
logged() {   # logged NAME CMD...: run CMD, output to $OUT/logs/NAME.log
  local log="$OUT/logs/$1.log"; shift
  echo "   \$ $* (log: logs/$(basename "$log"))"
  "$@" >"$log" 2>&1
}

STEP=0
echo "== 0/6 guards"
[[ -n "$WHEEL" ]] || refuse "$USAGE"
[[ "$FA3_SHA" =~ ^[0-9a-f]{64}$ ]] || refuse "FA3_WHEEL_SHA256 is not a sha256 (64 lowercase hex)"
[[ -f "$WHEEL" ]] || refuse "no FA3 wheel at $WHEEL"
[[ -f "$MANIFEST" ]] || refuse "no FA3 manifest at $MANIFEST"
[[ "$("$HELPER_PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MANIFEST")" == "$MANIFEST_SHA256" ]] \
  || refuse "the FA3 manifest $MANIFEST is not the recorded one (sha256 ${MANIFEST_SHA256:0:12}...)"
NVCC_REL="$(nvcc --version 2>/dev/null | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')"
[[ "$NVCC_REL" =~ ^12\.[0-9]+$ ]] || refuse "no nvcc 12.x under CUDA_HOME=$CUDA_HOME (found '${NVCC_REL:-none}')"
[[ -z "$(git status --porcelain)" ]] || refuse "the tree is not clean"
[[ ! -e "$OUT" ]] || refuse "$OUT exists; an attempt's record is never overwritten"
case "$(cd "$(dirname "$XATTN_DIR")" 2>/dev/null && pwd)/" in
  "$ROOT/"*) refuse "$XATTN_DIR is inside the repository; it would dirty the tree" ;;
esac
PYVER="$("$PY" -c 'import platform; print(platform.python_version())')" || refuse "$PY does not run"
[[ "$PYVER" == "3.10.12" ]] || refuse "$PY is Python $PYVER; image v6's is 3.10.12"
PYTHONPATH="$ROOT" "$HELPER_PY" -m attnbench.build_guards --max-jobs "$MAX_JOBS" \
  --nvcc-threads "$NVCC_THREADS" || refuse "flash-attn's memory guard: MAX_JOBS=$MAX_JOBS does not fit here"
BSA_COMMIT="$(pin_of attnbench/bsa_contract.py BSA_COMMIT)"
XATTN_COMMIT="$(pin_of attnbench/backends/xattention.py XATTN_COMMIT)"
XATTN_REPO="$(sed -nE 's/^XATTN_REPO = "(https:[^"]+)".*/\1/p' attnbench/backends/xattention.py)"
[[ -n "$XATTN_REPO" ]] || refuse "no XATTN_REPO in attnbench/backends/xattention.py"
mkdir -p "$OUT/logs"
# cuobjdump last among the guards: the apt install is the one thing step 0
# may install, so every free check has passed before it.
CUOBJDUMP="$(find_cuobjdump)"
if [[ -z "$CUOBJDUMP" ]]; then
  CUOBJ_PKG="cuda-cuobjdump-${NVCC_REL/./-}"
  echo "   no cuobjdump in $CUDA_HOME or a toolkit at or above $NVCC_REL; apt-get install $CUOBJ_PKG"
  logged apt_cuobjdump sudo apt-get install -y --no-install-recommends "$CUOBJ_PKG" \
    || { logged apt_update sudo apt-get update -q && logged apt_cuobjdump_retry sudo apt-get install -y --no-install-recommends "$CUOBJ_PKG"; } \
    || { echo "REFUSING: apt-get could not install $CUOBJ_PKG (logs/apt_*.log). Nothing else was installed." >&2; tool_inventory; exit 1; }
  CUOBJDUMP="$(find_cuobjdump)"
  [[ -n "$CUOBJDUMP" ]] || { echo "REFUSING: no cuobjdump in $CUDA_HOME or a toolkit at or above $NVCC_REL after installing $CUOBJ_PKG. Nothing else was installed." >&2; tool_inventory; exit 1; }
fi
"$CUOBJDUMP" --version >/dev/null 2>&1 || refuse "$CUOBJDUMP does not run"
echo "   commit $(git rev-parse HEAD), Python $PYVER, nvcc $NVCC_REL ($CUDA_HOME), cuobjdump $CUOBJDUMP, MAX_JOBS=$MAX_JOBS, out $OUT"

STEP=2
echo "== 2/6 image v6's packages from an index (exact, --no-deps)"
"$HELPER_PY" "${HELPER[@]}" requirements "$LIBS/pip_freeze_root.txt" --dpkg "$LIBS/dpkg_l.txt" >"$OUT/req_system.txt" \
  || fail "the system list does not read as pins"
"$HELPER_PY" "${HELPER[@]}" requirements "$LIBS/pip_freeze_usersite_mehuldahiya.txt" \
  --replace "$XFORMERS_STANDIN" >"$OUT/req_usersite.txt" || fail "the user-site list does not read as pins"
"$PY" -m pip --version >/dev/null 2>&1 || logged apt_pip sudo apt-get install -y python3-pip \
  || fail "no pip for $PY"
logged pip_system sudo -H "$PY" -m pip install --no-deps --ignore-installed --extra-index-url "$TORCH_INDEX" \
  -r "$OUT/req_system.txt" || fail "installing the system list"
logged pip_usersite "$PY" -m pip install --user --no-deps --extra-index-url "$TORCH_INDEX" \
  -r "$OUT/req_usersite.txt" || fail "installing the user-site list"
GOT="$("$PY" -c 'import torch, triton; print(torch.__version__, triton.__version__)')" \
  || fail "torch or triton does not import"
[[ "$GOT" == "2.9.1+cu129 3.5.1" ]] || fail "torch and triton are $GOT, not 2.9.1+cu129 3.5.1"
echo "   torch and triton: $GOT"

STEP=3
echo "== 3/6 FA3 wheel: checked against its manifest, installed, not imported"
"$HELPER_PY" "${HELPER[@]}" fa3-wheel "$WHEEL" --sha256 "$FA3_SHA" \
  --manifest "$MANIFEST" || fail "the FA3 wheel does not check"
logged fa3_install "$PY" -m pip install --user --no-deps "$WHEEL" || fail "installing the FA3 wheel"

STEP=4
echo "== 4/6 from source: flash-attn $FA2_VERSION, block-sparse-attention, x-attention"
logged fa_build bash "$FA_BUILD" "$FA2_VERSION" || fail "building flash-attn $FA2_VERSION"
logged bsa_build "$PY" -m pip install --user --no-build-isolation \
  "block-sparse-attn @ git+$BSA_URL@$BSA_COMMIT" || fail "building block-sparse-attention $BSA_COMMIT"
[[ -d "$XATTN_DIR/.git" ]] || logged xattn_clone git clone -q "$XATTN_REPO" "$XATTN_DIR" \
  || fail "cloning x-attention"
git -C "$XATTN_DIR" checkout -q --detach "$XATTN_COMMIT" || fail "checking out x-attention $XATTN_COMMIT"
logged xattn_install "$PY" -m pip install --user --no-deps -e "$XATTN_DIR" || fail "installing x-attention"
# The built cubins: sm_80 (what the L4 runs) and sm_90, nothing else.
USERSITE="$("$PY" -c 'import site; print(site.getusersitepackages())')"
for lib in flash_attn_2_cuda block_sparse_attn_cuda; do
  so="$(find "$USERSITE" -maxdepth 1 -name "${lib}*.so" | head -1)"
  [[ -n "$so" ]] || fail "no ${lib}*.so in the user site"
  got_sm="$("$CUOBJDUMP" --list-elf "$so" | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ' | sed 's/ $//')"
  echo "$lib $got_sm" >>"$OUT/cuda_archs.txt"
  [[ "$got_sm" == "$WANT_SM" ]] || fail "$lib's cubins are '$got_sm', not '$WANT_SM'"
done
echo "   cubins: $(tr '\n' ';' <"$OUT/cuda_archs.txt")"

STEP=5
echo "== 5/6 CPU smoke: imports, P2's libraries, a CPU test subset"
"$PY" -c 'import attnbench, attnbench.components.shakedown_arms' || fail "attnbench does not import"
logged imports "$PY" -c 'import flash_attn, block_sparse_attn; print(flash_attn.__version__)' \
  || { tail -5 "$OUT/logs/imports.log" >&2; fail "flash_attn or block_sparse_attn does not import without a GPU"; }
"$HELPER_PY" "${HELPER[@]}" libraries >"$OUT/libraries.json" \
  || { cat "$OUT/libraries.json" >&2; fail "the libraries differ from P2's"; }
PYTHONDONTWRITEBYTECODE=1 logged cpu_tests "$PY" -m pytest -q -p no:cacheprovider \
  tests/test_build_guards.py tests/test_env_build_lists.py || fail "the CPU test subset"

STEP=6
echo "== 6/6 the record"
"$PY" -m pip freeze --all >"$OUT/pip_freeze_user.txt"
"$PY" -m pip freeze --user >"$OUT/pip_freeze_usersite.txt"
sudo -H "$PY" -m pip freeze --all >"$OUT/pip_freeze_root.txt"
dpkg -l >"$OUT/dpkg_l.txt"
uname -a >"$OUT/meta.txt"
for pair in user:pip_freeze_user_mehuldahiya usersite:pip_freeze_usersite_mehuldahiya \
            root:pip_freeze_root; do
  "$HELPER_PY" "${HELPER[@]}" diff-freeze "$OUT/pip_freeze_${pair%%:*}.txt" "$LIBS/${pair#*:}.txt" \
    >"$OUT/diff_vs_v6_${pair%%:*}.json" || fail "diffing the ${pair%%:*} list"
done
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum >SHA256SUMS) \
  || fail "writing SHA256SUMS"
[[ -z "$(git status --porcelain)" ]] || fail "the build left the tree dirty"
echo "   record in $OUT"
echo "ENV_BUILD_OK"
