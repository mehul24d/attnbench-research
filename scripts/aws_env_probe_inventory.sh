#!/usr/bin/env bash
# Inventory of the environment-build base image, run ON the probe instance
# (`scripts/aws_launch_session.sh PROBE`; adviser's rulings dated 22 Nov 2026,
# probe). Read-only: it installs, builds and writes nothing. Piped over SSH
# (`ssh HOST bash -s < scripts/aws_env_probe_inventory.sh`), so it needs no
# deploy. It prints no hostname, address or instance id; the Mac redacts its
# output again before it is committed.
#
# Sections: each CUDA toolkit under /usr/local (its bin listing, nvcc's
# release, whether cuobjdump is there and runs), every cuobjdump on the
# system, every pip and python3* on PATH with torch in each, CPU/memory/disk, apt sources and the CUDA
# packages apt knows of.
set -u
sec() { printf '\n## %s\n' "$*"; }

sec "os"
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -r
sec "cpu memory disk"
echo "nproc $(nproc)"
free -m | sed -n '1,2p'
df -h --output=target,size,used,avail / /tmp "$HOME" 2>/dev/null | awk '!seen[$0]++'

sec "cuda toolkits"
ls -ld /usr/local/cuda* 2>/dev/null | awk '{print $NF, $(NF-1), $(NF-2)}'
echo "/usr/local/cuda -> $(readlink -f /usr/local/cuda 2>/dev/null || echo none)"
for d in /usr/local/cuda-*; do
  [[ -d "$d" && ! -L "$d" ]] || continue
  echo "### $d"
  echo "nvcc: $("$d/bin/nvcc" --version 2>/dev/null | sed -nE 's/.*release ([0-9.]+), V([0-9.]+).*/release \1 (V\2)/p' || true)"
  if [[ -x "$d/bin/cuobjdump" ]]; then
    echo "cuobjdump: present, $("$d/bin/cuobjdump" --version 2>/dev/null | sed -nE 's/.*release ([0-9.]+).*/release \1/p')"
  else
    echo "cuobjdump: absent"
  fi
  echo "version.json cuda: $(sed -nE '/"cuda" *:/{n;s/.*"version" *: *"([^"]+)".*/\1/p;}' "$d/version.json" 2>/dev/null | head -1)"
  echo "bin: $(ls "$d/bin" 2>/dev/null | tr '\n' ' ')"
done

sec "every cuobjdump and nvcc on the system"
find /usr /opt -xdev \( -name cuobjdump -o -name nvcc \) -type f 2>/dev/null | sort
echo "on PATH: cuobjdump=$(command -v cuobjdump || echo none) nvcc=$(command -v nvcc || echo none)"

sec "python and torch"
# Every pip and python3* on PATH, then the fixed candidates (adviser's task of
# 22 Nov 2026, the interpreter wave: the image's bare `pip` was a second
# Python's, 3.11, which the first probe never listed). No bare pip or python
# (tests/test_build_interpreter.py).
{
  IFS=: read -ra dirs <<<"$PATH"
  for d in "${dirs[@]}"; do
    for f in "$d"/pip "$d"/pip3 "$d"/pip3.[0-9]*; do
      [[ -x "$f" && ! -d "$f" ]] && echo "PIP $f"
    done
    for f in "$d"/python3 "$d"/python3.[0-9]*; do
      [[ -x "$f" && ! -d "$f" && "$f" != *-config ]] && echo "PY $f"
    done
  done
  for f in /usr/bin/python3 /opt/conda/bin/python /opt/pytorch/bin/python; do
    [[ -x "$f" ]] && echo "PY $f"
  done
} | awk '!seen[$0]++' | while read -r kind f; do
  if [[ "$kind" == PIP ]]; then
    echo "$f -> $(readlink -f "$f"): $("$f" --version 2>&1 | head -1)"
    continue
  fi
  echo "$f -> $(readlink -f "$f"): $("$f" -c 'import platform; print(platform.python_version())' 2>&1)"
  echo "  torch: $("$f" -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1 | cut -c1-200)"
  echo "  pip: $("$f" -m pip --version 2>&1 | cut -d' ' -f1,2)"
done
ls -d /opt/*/ 2>/dev/null | tr '\n' ' '; echo

sec "apt sources"
grep -hvE '^\s*(#|$)' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null
for f in /etc/apt/sources.list.d/*.sources; do [[ -f "$f" ]] && { echo "# $f"; grep -E '^(URIs|Suites|Components):' "$f"; }; done
sec "cuda packages installed"
dpkg -l 2>/dev/null | awk '$1=="ii" && $2 ~ /^(cuda-(cuobjdump|nvcc|toolkit|compiler)|cuda-toolkit|nvidia-driver|libcudnn)/ {print $2, $3}'
sec "cuda cuobjdump packages apt offers (cache as shipped, not updated)"
apt-cache policy 'cuda-cuobjdump-12-*' 'cuda-cuobjdump-13-*' 2>/dev/null | grep -E '^cuda|Installed|Candidate'
echo "PROBE_INVENTORY_DONE"
