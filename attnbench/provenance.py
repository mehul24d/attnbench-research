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
        if out.returncode != 0:
            return None
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


def _sh_result(cmd: list[str]) -> tuple[bool, str]:
    """(succeeded, stdout) -- distinguishes EMPTY OUTPUT from FAILURE.

    `_sh` returns `stdout.strip() or None`, which conflates the two. That is
    fine where empty output is meaningless, and wrong wherever empty output is
    the answer.

    Concretely: `nvidia-smi --query-compute-apps` prints NOTHING when no other
    process holds the GPU -- the exact condition Stage 2 requires. Through
    `_sh` that became None, which `assert_exclusive` reported as "nvidia-smi
    unavailable; cannot verify", and every timing cell was blocked on a
    perfectly clean GPU.

    This is the mirror of the `git rev-parse HEAD` bug (see
    docs/silent_failure_patterns.md instance 3): there, non-empty stdout was
    read as success; here, empty stdout was read as failure. Both come from
    inferring status from output content instead of the exit code.
    """
    if shutil.which(cmd[0]) is None:
        return False, ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.returncode == 0, out.stdout.strip()
    except Exception:
        return False, ""


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
# Which of these fields anything actually reads
# ---------------------------------------------------------------------------
#
# On 2026-09-04 every row of a 4576-row result set was stamped with a commit 18
# behind the code that produced it. The stamp was not silently wrong:
# `git_dirty=True` was recorded correctly on every one of those rows. No gate
# read it, so the table would have cleared the Stage 2 commit check carrying a
# stamp that provably could not be right -- it contained 84
# `illegal_memory_access` rows, and the code that writes that status did not
# exist at the commit named.
#
# A field that is set correctly and read by nobody fails exactly as a missing
# field does. So the split is declared here rather than left to be discovered:
# every field of Provenance must appear in one of these two sets, and
# tests/test_provenance_consulted.py fails if a new field appears in neither,
# or if a GATED field is not actually read by any gate module.

GATED_FIELDS = frozenset({
    "git_commit",   # sweep.load_stage1_passes, analysis.cross_arch join
    "git_dirty",    # sweep.load_stage1_passes -- added because of the above
    "host",         # sweep.check_host_continuity, cross_arch grouping
    "gpu_name",     # analysis.canary, cross_arch grouping
    "clocks_locked",  # cross_arch.Speedup, canary.CanaryDrift -- see below
})

# Kept for the record and for post-hoc analysis, consulted by no gate. This is
# a deliberate classification, not a backlog: a version string is evidence when
# reading a result months later, and gating on it would block runs over
# differences that usually do not matter.
#
# `clocks_locked` was in this set until 2026-09-04, when the audit above found
# it: the module docstring says "an unlocked run must be flagged in the
# results", it was flagged, and nothing read it. It is now GATED, but at
# ANALYSIS time rather than measurement time -- `nvidia-smi -lgc` needs root
# and fails on most rental hosts, so refusing to measure would delete the
# second architecture and with it the hardware-conditional finding, which
# costs far more than the extra variance. So the flag travels onto every ratio
# (cross_arch.Speedup) and every drift line (canary.CanaryDrift), where the
# harm actually lands. The raw clock readings stay decorative: they are useful
# for reading a result later, and nothing can act on 1710 vs 1695 MHz.
RECORDED_FIELDS = frozenset({
    "timestamp", "python", "platform",
    "torch", "torch_cuda", "cudnn", "triton",
    "flash_attn", "flashinfer", "xformers", "fla",
    "driver", "compute_capability", "gpu_memory_gb", "gpu_count",
    "sm_clock_mhz", "mem_clock_mhz", "persistence_mode",
})


def stamp_onto(df, stamp: dict) -> None:
    """Merge a provenance stamp onto a frame of measured rows, in place.

    A stamp fills in what the harness did NOT measure. Where the rows already
    carry a field, the rows own it, and a stamp that disagrees is an error --
    raised, not resolved by preferring one side.

    Why this is a function and not three lines at a call site
    --------------------------------------------------------
    On 2026-09-07 Stage 5 measured `clocks_locked` correctly, threaded it
    through `measure_band` onto every PhaseMeasurement, and then destroyed it
    on the last write:

        prov = provenance.capture().to_dict()     # clocks_locked defaults False
        for k, v in prov.items():
            df[k] = v                             # measured True -> stamped False

    `capture()` defaults `clocks_locked` to False, so the loop wrote a
    plausible falsehood over a measurement, in a GATED field --
    `cross_arch.Speedup` and `canary.CanaryDrift` both read it. Three other
    columns of the same file contradicted it (persistence_mode Enabled,
    sm_clock_mhz pinned at 1740 against a requested 1734, and the run log's
    success branch), and nothing looked at them.

    `scripts/run_accuracy.py` had already fixed exactly this in its own body,
    with a comment naming the failure mode -- "a run whose clocks ARE pinned
    would be recorded as unpinned, and the fact would be lost". Stage 5 was
    written afterwards and did not inherit it, because the fix lived in a
    call site rather than in the thing every call site uses. That is the
    single-source-of-truth asymmetry this project keeps meeting: the fix was
    correct and local, so the next harness reproduced the bug from scratch.

    A caller that genuinely measured a field passes it to `capture()` too, at
    which point the two agree and this never fires. It can only fire on a
    real inconsistency, so raising costs a correct run nothing -- and callers
    write their per-band output before this, so a raise here cannot destroy a
    completed measurement.
    """
    conflicts = []
    for k, v in stamp.items():
        if k in df.columns:
            existing = df[k].unique()
            if len(existing) != 1 or existing[0] != v:
                conflicts.append(f"  {k}: rows={list(existing)} stamp={v!r}")
            continue
        df[k] = v
    if conflicts:
        raise ValueError(
            "provenance stamp disagrees with measured rows:\n"
            + "\n".join(conflicts)
            + "\n\nThe rows own any field they measured. Pass the measured "
              "value to provenance.capture() so the stamp agrees, rather "
              "than letting the stamp's default overwrite it.")


ANALYSIS_STAMP_COLUMNS = ("analysis_tool", "analysis_git_commit",
                          "analysis_git_dirty", "analysis_host",
                          "analysis_timestamp")


def stamp_analysis(df, tool: str) -> None:
    """Stamp a DERIVED frame with what produced it, in place.

    Separate from `stamp_onto`, and deliberately under `analysis_`-prefixed
    names, because the two describe different machines. A `phases.parquet`
    stamp says which GPU, driver and clock state produced a *measurement*.
    An `pareto.parquet` stamp says which checkout of which script reduced
    already-measured rows -- run on a laptop, hours or days later, at a
    different commit. Writing the second under the first's column names
    would put a laptop's `gpu_name` beside a measurement it never touched,
    which is `clocks_locked=False` over a locked run wearing a new hat
    (see `stamp_onto`).

    Prefixed names also cannot collide with the measurement columns the
    source rows already carry, so this can never overwrite a measured field.

    Why it exists at all: README line 89 says "No result row is written
    without a provenance stamp", and on 2026-09-12 fourteen files across
    Stages 4, 5, 6, 7 and cross_arch had none -- every derived artifact in
    the project. They were not wrong, they were unverifiable, which is the
    whole reason the rule is there.
    """
    commit, dirty = _git_state()
    stamp = {
        "analysis_tool": tool,
        "analysis_git_commit": commit,
        "analysis_git_dirty": dirty,
        "analysis_host": platform.node(),
        "analysis_timestamp": time.time(),
    }
    for k, v in stamp.items():
        df[k] = v


def stamp_integrity_problems(stamp) -> list[str]:
    """Reasons the provenance on a row cannot be trusted to describe the code.

    Takes anything with `.get` or `[]` access -- a dict, a pandas row. Returns
    a list of human-readable problems; empty means usable.

    This checks the stamp's INTERNAL consistency, not whether it matches any
    particular commit. "Is this stamp meaningful at all" and "is it the commit
    I expect" are different questions, and conflating them is how a dirty tree
    passed a commit check: the commit matched, and the commit was meaningless.
    """
    def get(key):
        try:
            return stamp[key]
        except (KeyError, IndexError, TypeError):
            return getattr(stamp, key, None)

    def is_true(v) -> bool:
        """True for Python True, numpy.bool_(True), 1 -- and NOT for NaN.

        `v is True` fails on numpy.bool_, which is what a pandas column hands
        back, so the dirty check silently passed every parquet row it was
        written to catch. And plain `bool(v)` is worse: a missing value reads
        as NaN, `bool(nan)` is True, and every row with no git_dirty at all
        would be reported dirty -- a check that fires on absence is as useless
        as one that never fires.
        """
        if v is None or isinstance(v, str):
            return False
        if isinstance(v, float) and v != v:      # NaN
            return False
        try:
            return bool(v)
        except Exception:
            return False

    problems: list[str] = []

    commit = get("git_commit")
    if commit is None or (isinstance(commit, float) and commit != commit):
        problems.append("no git_commit: the code that produced this row is "
                        "unidentifiable")
    elif not _SHA_RE.match(str(commit)):
        problems.append(f"git_commit {str(commit)[:20]!r} is not a 40-hex SHA")

    if is_true(get("git_dirty")):
        problems.append(
            "git_dirty: the working tree did not match the commit named, so "
            "the commit does not describe the code that ran. This is how a "
            "4576-row table came to be stamped 18 commits behind itself on "
            "2026-09-04 -- source deployed over an image's checkout, leaving "
            "`.git` describing the image")

    return problems


# ---------------------------------------------------------------------------
# Clock control
# ---------------------------------------------------------------------------

def lock_clocks(sm_mhz: Optional[int] = None) -> bool:
    """Pin SM clocks and enable persistence mode. Returns whether it worked.

    Needs root or an admin-granted permission, so it will often fail on shared
    university clusters and on some rental hosts. Failure is returned, not
    raised, but an unlocked run must be flagged in the results: unlocked clocks
    on a shared host produce variance that is easy to mistake for a real effect.

    **This returned True on a lock that did not happen, until 2026-09-06.** It
    ended with `res = _sh([...]); return res is not None`, and `_sh` returns
    `stdout.strip() or None` -- it never looks at the exit code. Run without
    root, `nvidia-smi -lgc` exits 4 and prints

        The current user does not have permission to change clocks for GPU ...

    to STDOUT. Non-empty stdout, so `_sh` returned a string, so this returned
    True. Measured on the 2026-09-06 L4: returned True, clock unchanged at
    2040 MHz (the unlocked maximum) rather than the 1734 requested.

    Third instance of stdout being read as an outcome (see
    docs/silent_failure_patterns.md): `git rev-parse HEAD` echoing "HEAD" on
    failure, `--query-compute-apps` printing nothing on success, and now this.
    `_sh_result` exists precisely because `_sh` cannot answer "did it work" --
    the fix is to use it, and the reason this one is worse than the other two
    is its direction: it reports a control as ESTABLISHED when it is absent,
    so every row it stamps overstates how well the run was controlled.
    """
    if shutil.which("nvidia-smi") is None:
        return False

    def _smi(args: list[str]) -> tuple[bool, str]:
        """nvidia-smi, escalating to passwordless sudo for the calls that
        need it.

        Clock control needs root; the measurement itself must NOT run as
        root (it would write results/ owned by root and leave the next
        segment unable to append). So the two split, and the escalation
        lives here rather than in the caller -- otherwise every caller
        either runs the whole run as root or ASSERTS the lock state on the
        command line, and an asserted control is the thing this function
        was just fixed for.

        `sudo -n` never prompts: no sudo, no rights, or a password required
        all fail immediately and return False, which is the truthful answer.
        """
        ok, out = _sh_result(["nvidia-smi", *args])
        if ok:
            return True, out
        if shutil.which("sudo") is None:
            return False, out
        return _sh_result(["sudo", "-n", "nvidia-smi", *args])

    _smi(["-pm", "1"])
    if sm_mhz is None:
        ok, q = _sh_result(["nvidia-smi", "--query-gpu=clocks.max.sm",
                            "--format=csv,noheader,nounits"])
        if not ok or not q.strip():
            return False
        sm_mhz = int(int(q.splitlines()[0].strip()) * 0.85)  # headroom, avoids throttle
    ok, _ = _smi(["-lgc", f"{sm_mhz},{sm_mhz}"])
    return ok


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
    ok, out = _sh_result(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"])
    if not ok:
        return "unverifiable", "nvidia-smi unavailable; cannot verify"
    others = [l for l in out.splitlines()
              if l.strip() and not l.startswith(str(os.getpid()))]
    if others:
        return "contaminated", f"other processes on device: {others}"
    return "exclusive", "exclusive"
