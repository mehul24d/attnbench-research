"""A child process run under a host-memory ceiling (researcher's ruling, 2
Nov 2026, after T3.1's third session).

That session's step 5 ran every cell in one process, which the kernel's
out-of-memory killer took at 31.6 GB on the 31 GB host, and the step with
it. Step 5 now runs each cell in its own process under `run_guarded`: the
parent samples the child's resident set and, at the ceiling, kills it and
says so, so a cell that grows too large fails alone, with its peak and the
reason recorded, and the killer never chooses.

The watch is from outside because the growth is in native code (the
profiler's buffers, its export, `json` decoding), which a thread in the
child cannot interrupt. A sample is taken every `poll_s`; a spike shorter
than that can pass between two samples, so the ceiling leaves margin below
the host's memory. The child reports its own high-water mark at exit
(`self_peak_mb`), which no sample can miss.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional, Sequence

POLL_S = 0.25


def rss_mb(pid: int) -> Optional[float]:
    """The process's current resident set in MiB, or None once it is gone.
    Linux reads `/proc/<pid>/status`; elsewhere `ps`."""
    status = Path(f"/proc/{pid}/status")
    if Path("/proc").is_dir():
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
        except (FileNotFoundError, ProcessLookupError):
            return None
        return None
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    text = out.stdout.strip()
    return int(text) / 1024.0 if out.returncode == 0 and text else None


def self_peak_mb() -> float:
    """This process's high-water resident set in MiB (`ru_maxrss`: KiB on
    Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024.0


@dataclass(frozen=True)
class Guarded:
    returncode: int
    seconds: float
    sampled_peak_mb: float
    ceiling_mb: float
    over_ceiling: bool          # killed by the guard at the ceiling

    def record(self) -> dict:
        return {"returncode": self.returncode, "seconds": round(self.seconds, 1),
                "sampled_peak_rss_mb": round(self.sampled_peak_mb, 1),
                "rss_ceiling_mb": round(self.ceiling_mb, 1), "over_ceiling": self.over_ceiling}


def run_guarded(cmd: Sequence[str], *, ceiling_mb: float, poll_s: float = POLL_S,
                log: Optional[IO] = None, cwd=None, env=None, clock=time.monotonic) -> Guarded:
    """Run `cmd`, sampling its resident set every `poll_s`; kill it (SIGKILL)
    at `ceiling_mb`. Output goes to `log` (default: this process's)."""
    if not ceiling_mb or ceiling_mb <= 0:
        raise ValueError(f"the ceiling must be positive, got {ceiling_mb!r}")
    t0 = clock()
    proc = subprocess.Popen(list(cmd), stdout=log, stderr=subprocess.STDOUT if log else None,
                            cwd=cwd, env=env)
    peak, over = 0.0, False
    while proc.poll() is None:
        rss = rss_mb(proc.pid)
        if rss is not None:
            peak = max(peak, rss)
            if rss >= ceiling_mb:
                over = True
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
        time.sleep(poll_s)
    return Guarded(proc.returncode, clock() - t0, peak, float(ceiling_mb), over)
