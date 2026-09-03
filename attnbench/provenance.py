"""Environment capture and clock control.

Built before anything else on purpose. No measurement is ever recorded without a
provenance stamp attached, because the moment results from two machines end up
in the same dataframe without one, the dataset is unusable.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch


def _sh(cmd: list[str]) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git_state() -> tuple[Optional[str], Optional[bool]]:
    """(commit_sha, dirty), or (None, None) when there is no usable commit.

    Two failure modes this guards, both discovered 2026-09-03 with results
    already on disk:

    **`git rev-parse HEAD` can "succeed" with a non-commit.** When HEAD does
    not resolve -- an empty repository, a fresh branch with no commits -- git
    writes its error to stderr but echoes the unresolved argument, `HEAD`, to
    *stdout*. A naive capture stores the literal string "HEAD" as the commit.
    That is worse than storing nothing: the field looks populated, every row
    agrees with every other row, and any check comparing commits across
    result files passes while verifying nothing. The SHA format is therefore
    validated, not assumed.

    **The repository must be this project's, not whatever encloses the
    process's CWD.** On the 2026-09-03 machine the enclosing repo was the
    user's *home directory*, so `git status --porcelain` reported thousands of
    unrelated untracked files and `git_dirty` was True regardless of the
    project's actual state. Git is anchored to the directory containing this
    package, and `--show-toplevel` is recorded implicitly by that anchoring.

    Returning None on failure is deliberate: provenance capture must not
    raise, or an unrelated environment problem takes down a measurement run.
    The refusal to *use* commit-less results belongs at the join, where there
    is something to refuse -- see analysis/cross_arch.py.
    """
    root = str(Path(__file__).resolve().parent.parent)
    commit = _sh(["git", "-C", root, "rev-parse", "HEAD"])
    if not commit or not _SHA_RE.match(commit):
        return None, None
    status = _sh(["git", "-C", root, "status", "--porcelain"])
    return commit, bool(status)


def _pkg(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


@dataclass
class Provenance:
    """Everything that could plausibly move a number."""

    timestamp: float
    host: str
    python: str
    platform: str

    torch: Optional[str]
    torch_cuda: Optional[str]
    cudnn: Optional[int]
    triton: Optional[str]
    flash_attn: Optional[str]
    flashinfer: Optional[str]
    xformers: Optional[str]
    fla: Optional[str]

    driver: Optional[str]
    gpu_name: Optional[str]
    compute_capability: Optional[str]
    gpu_memory_gb: Optional[float]
    gpu_count: int

    clocks_locked: bool
    sm_clock_mhz: Optional[int]
    mem_clock_mhz: Optional[int]
    persistence_mode: Optional[str]

    git_commit: Optional[str]
    git_dirty: Optional[bool]

    def to_dict(self) -> dict:
        return asdict(self)


def capture(clocks_locked: bool = False) -> Provenance:
    cc = mem = name = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        cc = f"{major}.{minor}"
        name = torch.cuda.get_device_name()
        mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    smi = _sh(["nvidia-smi",
               "--query-gpu=driver_version,clocks.sm,clocks.mem,persistence_mode",
               "--format=csv,noheader,nounits"])
    driver = sm_clk = mem_clk = persist = None
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        if len(parts) == 4:
            driver, sm_clk, mem_clk, persist = parts
            sm_clk = int(sm_clk) if sm_clk.isdigit() else None
            mem_clk = int(mem_clk) if mem_clk.isdigit() else None

    commit, dirty = _git_state()

    return Provenance(
        timestamp=time.time(),
        host=platform.node(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        triton=_pkg("triton"),
        flash_attn=_pkg("flash-attn"),
        flashinfer=_pkg("flashinfer-python") or _pkg("flashinfer"),
        xformers=_pkg("xformers"),
        fla=_pkg("flash-linear-attention"),
        driver=driver,
        gpu_name=name,
        compute_capability=cc,
        gpu_memory_gb=mem,
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        clocks_locked=clocks_locked,
        sm_clock_mhz=sm_clk,
        mem_clock_mhz=mem_clk,
        persistence_mode=persist,
        git_commit=commit,
        git_dirty=dirty,
    )


def write(path: str | Path = "versions.json", **kw) -> Provenance:
    p = capture(**kw)
    Path(path).write_text(json.dumps(p.to_dict(), indent=2))
    return p


# ---------------------------------------------------------------------------
# Clock control
# ---------------------------------------------------------------------------

def lock_clocks(sm_mhz: Optional[int] = None) -> bool:
    """Pin SM clocks and enable persistence mode.

    Needs root or an admin-granted permission, so it will often fail on shared
    university clusters and on some rental hosts. Failure is returned, not
    raised, but an unlocked run must be flagged in the results: unlocked clocks
    on a shared host produce variance that is easy to mistake for a real effect.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    _sh(["nvidia-smi", "-pm", "1"])
    if sm_mhz is None:
        q = _sh(["nvidia-smi", "--query-gpu=clocks.max.sm",
                 "--format=csv,noheader,nounits"])
        if not q:
            return False
        sm_mhz = int(int(q.splitlines()[0].strip()) * 0.85)  # headroom, avoids throttle
    res = _sh(["nvidia-smi", "-lgc", f"{sm_mhz},{sm_mhz}"])
    return res is not None


def unlock_clocks() -> None:
    _sh(["nvidia-smi", "-rgc"])


def assert_exclusive() -> tuple[str, str]:
    """Check no other process holds the GPU.

    Returns (status, detail) with status one of "exclusive", "contaminated",
    or "unverifiable". Contaminated timing data is worse than less data, so
    Stage 2 and Stage 5 (the only stages that time anything) must treat both
    "contaminated" and "unverifiable" as blocking -- an unconfirmed check is
    not the same as a confirmed-clean one. Other stages don't time anything
    and may proceed regardless, recording the status as a flag rather than
    gating on it.
    """
    out = _sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
               "--format=csv,noheader"])
    if out is None:
        return "unverifiable", "nvidia-smi unavailable; cannot verify"
    others = [l for l in out.splitlines()
              if l.strip() and not l.startswith(str(os.getpid()))]
    if others:
        return "contaminated", f"other processes on device: {others}"
    return "exclusive", "exclusive"
