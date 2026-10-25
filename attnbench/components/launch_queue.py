"""Launch-queue depth Q (T2.6; plan Sections 5.3 step 3, 5.4 and 5.6).

The recursion's queue line, `if Q launches are pending: H_i = max(H_i,
D_(i-Q))`, says the host cannot issue operation i until operation i - Q has
finished: at most Q launches are in flight, the one being issued included.

**The probe.** With the device idle, launch one long blocker (a
`torch.cuda._sleep` spin of `blocker_s` seconds), then `launches` short
kernels back to back, timing each launch on the host. While the blocker
runs nothing completes, so launches return at host speed until the queue is
full; the next one waits until the blocker finishes. If the first launch
that waits is short kernel number j (counting from 1), then the blocker and
the j - 1 kernels before it were in flight together, and

    Q = (j - 1) + 1 = j.

A launch "waits" when its host time exceeds `blocked_fraction` of the
blocker. If no launch waits, the queue held everything issued: the trial
gives only the lower bound Q > `launches` and records no Q. A blocker too
short to outlast the fast launches looks the same, which is why the bound
is reported, not a guess.

Two kernels are probed: the null kernel of the host-speed probe
(`torch.cuda._sleep(0)`) and a one-element in-place add, which carries
PyTorch's ordinary elementwise arguments. A launch's queue footprint can
depend on its arguments, so both are recorded; `queue_depth` takes the kind
explicitly.

Device operations are injected (`QueueOps`) so the analysis is tested on
CPU against a simulated queue; `cuda_queue_ops` builds the real ones.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

KINDS = {"sleep0": "torch.cuda._sleep(0)",
         "add": "x.add_(1) on a one-element tensor"}
DEFAULT_KIND = "sleep0"


@dataclass
class QueueOps:
    """sync(); blocker(seconds) launches the long kernel; launcher(kind)
    returns a callable that launches one short kernel of that kind."""
    sync: Callable[[], None]
    blocker: Callable[[float], None]
    launcher: Callable[[str], Callable[[], None]]


def cuda_queue_ops(calibration_cycles: int = 10_000_000) -> QueueOps:
    """Real operations. `_sleep` takes cycles, so the cycle rate is
    calibrated once with CUDA events."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("the launch-queue probe needs a CUDA device")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(calibration_cycles)
    end.record()
    end.synchronize()
    cycles_per_s = calibration_cycles / (start.elapsed_time(end) * 1e-3)
    x = torch.zeros(1, device="cuda")

    def launcher(kind: str):
        if kind == "sleep0":
            return lambda: torch.cuda._sleep(0)
        if kind == "add":
            return lambda: x.add_(1)
        raise KeyError(kind)

    ops = QueueOps(sync=torch.cuda.synchronize,
                   blocker=lambda s: torch.cuda._sleep(int(s * cycles_per_s)),
                   launcher=launcher)
    ops.cycles_per_s = cycles_per_s          # recorded on the row
    return ops


def trial(ops: QueueOps, kind: str, *, launches: int, blocker_s: float,
          clock: Callable[[], int] = time.perf_counter_ns) -> list[float]:
    """Host seconds per short-kernel launch behind one blocker."""
    launch = ops.launcher(kind)
    ops.sync()
    ops.blocker(blocker_s)
    out = []
    try:
        for _ in range(launches):
            t0 = clock()
            launch()
            out.append((clock() - t0) * 1e-9)
    finally:
        ops.sync()
    return out


def depth_from_trial(times: Sequence[float], blocker_s: float,
                     blocked_fraction: float = 0.25) -> dict:
    """Q from one trial's launch times (module docstring). `q` is None when
    no launch waited, and `q_lower_bound` is then len(times) + 1."""
    if not 0 < blocked_fraction < 1:
        raise ValueError("blocked_fraction must be in (0, 1)")
    limit = blocked_fraction * blocker_s
    first = next((j for j, t in enumerate(times, start=1) if t > limit), None)
    fast = times if first is None else times[:first - 1]
    return {"q": first,
            "q_lower_bound": len(times) + 1 if first is None else first,
            "blocked_wait_s": None if first is None else times[first - 1],
            "fast_launch_median_s": statistics.median(fast) if fast else None}


def measure(ops: QueueOps, *, kinds: Sequence[str] = tuple(KINDS),
            trials: int = 5, launches: int = 16384, blocker_s: float = 2.0,
            blocked_fraction: float = 0.25, dry_run: bool = False,
            clock=time.perf_counter_ns) -> dict:
    """One flat row: per kind, every trial's Q and the agreed Q if all
    trials agree."""
    row = {"dry_run": dry_run, "trials": trials, "launches": launches,
           "blocker_s": blocker_s, "blocked_fraction": blocked_fraction,
           "cycles_per_s": getattr(ops, "cycles_per_s", None)}
    for kind in kinds:
        if kind not in KINDS:
            raise KeyError(f"unknown kernel kind {kind!r}; known: {sorted(KINDS)}")
        got = [depth_from_trial(trial(ops, kind, launches=launches,
                                      blocker_s=blocker_s, clock=clock),
                                blocker_s, blocked_fraction)
               for _ in range(trials)]
        qs = [g["q"] for g in got]
        row[f"{kind}_kernel"] = KINDS[kind]
        row[f"{kind}_q_trials"] = ",".join("none" if q is None else str(q) for q in qs)
        row[f"{kind}_q"] = qs[0] if None not in qs and len(set(qs)) == 1 else None
        row[f"{kind}_q_lower_bound"] = min(g["q_lower_bound"] for g in got)
        waits = [g["blocked_wait_s"] for g in got if g["blocked_wait_s"] is not None]
        row[f"{kind}_blocked_wait_s_median"] = statistics.median(waits) if waits else None
        fasts = [g["fast_launch_median_s"] for g in got if g["fast_launch_median_s"]]
        row[f"{kind}_fast_launch_s_median"] = statistics.median(fasts) if fasts else None
    return row


def queue_depth(row, kind: str) -> int:
    """Q for the recursion. Refuses a dry-run row, a kind not measured, and
    trials that disagree or never filled the queue: each would put a number
    in the model that the probe did not establish."""
    if bool(row["dry_run"]):
        raise ValueError("this is a dry-run row: its Q comes from a simulation")
    key = f"{kind}_q"
    if key not in row:
        raise KeyError(f"kind {kind!r} was not measured")
    q = row[key]
    if q is None or q != q:
        raise ValueError(
            f"no agreed Q for {kind}: trials gave {row[f'{kind}_q_trials']} "
            f"(lower bound {row[f'{kind}_q_lower_bound']}); rerun with more "
            f"launches or a longer blocker, or report the spread")
    return int(q)


class SimulatedQueue:
    """A device with a true queue depth, for tests and `--dry-run`.

    Time is one clock in ns. A launch costs `host_ns`; the blocker runs
    `blocker_s`, each short kernel `kernel_ns` after it. A launch that finds
    `depth` operations in flight waits until the oldest one finishes.
    """

    def __init__(self, depth: int, host_ns: int = 5_000, kernel_ns: int = 2_000):
        self.depth, self.host_ns, self.kernel_ns = depth, host_ns, kernel_ns
        self.t = 0
        self.done_at: list[int] = []     # completion time of each in-flight op

    def clock(self) -> int:
        return self.t

    def _issue(self, run_ns: int) -> None:
        # Host work first, then the wait for a free slot, as the recursion
        # orders them: H_i = max(H_(i-1) + h_i, D_(i-Q)).
        self.t += self.host_ns
        self.done_at = [d for d in self.done_at if d > self.t]
        if len(self.done_at) >= self.depth:
            self.t = self.done_at[len(self.done_at) - self.depth]
            self.done_at = [d for d in self.done_at if d > self.t]
        start = max(self.t, self.done_at[-1] if self.done_at else self.t)
        self.done_at.append(start + run_ns)

    def ops(self) -> QueueOps:
        def sync():
            if self.done_at:
                self.t = max(self.t, self.done_at[-1])
            self.done_at = []

        return QueueOps(sync=sync,
                        blocker=lambda s: self._issue(int(s * 1e9)),
                        launcher=lambda kind: (lambda: self._issue(self.kernel_ns)))
