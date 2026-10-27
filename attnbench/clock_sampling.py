"""GPU clock and throttle sampling around a timed block (plan Sections 5.3
step 1 and 5.6; T2.13).

Section 5.6: "Clocks and throttle reasons sampled in every timed block;
throttled blocks flagged and rerun." This module reads the clocks and the
throttle state and says, per block, whether it was throttled. It decides
nothing about reruns; `ground_truth.time_arms` does.

**Two traps in reading throttle state.**

1. An instantaneous throttle-reason read taken after the closing
   synchronise sees an idle GPU: the "GPU idle" bit is set and everything
   else is clear, whatever happened during the block. So a boundary read
   cannot certify a block unthrottled. What can: the cumulative violation
   counters NVML keeps (nanoseconds spent below the requested clocks because
   of the power or thermal policy), read before and after the block. Any
   throttle inside the block moves them. A reader that has no such counters
   must be paired with an in-block sampling thread (`interval_s`), and
   `BlockSampler` refuses to be built with neither (`ClockSamplingError`):
   with neither, every block would pass whatever the card did.
2. A sampling thread runs on the host. On a host-bound arm, which is the
   case this study is about, a thread that wakes every few milliseconds is
   itself a perturbation. So the thread is opt-in, and the harness's default
   path needs none when the counters exist. Its cost is for the L4 dry run to
   measure (T2.27); it is not assumed to be zero.

**Which reasons flag a block is a decision, not a default.** The bits are
NVML's. Some are normal under load (a power-limited card sits at its
software power cap for the whole run); some are the card being forced down.
`BlockSampler` takes `flag_bits` with no default, so the choice is written
where the session is configured and recorded on every row. Every reason seen
is recorded whether or not it flags.

**The session's detector is a separate process** (Appendix A, 27 Oct 2026
and the T2.13 follow-up of 28 Oct 2026). `ProcessSampler` runs the reader in a
process of its own at about 10 Hz and the harness only reads what that
process has written. It sees what the counters cannot: hardware slowdown has
no cumulative counter, and the power counter is the integral of the software
cap and the hardware brake together, so it cannot tell a capped block from a
braked one. The decision rule is on the bits alone: the researcher's set
(`RESEARCHER_FLAG_BITS`) flags, the software power cap is recorded on every
block and never flags. Each block is judged on the samples taken inside it
plus the one just before and the one just after, so a block shorter than the
sampling interval is still bracketed; the bracketing samples can only add a
flag (a rerun), never hide one. The process is not a thread in the harness's
interpreter, so it takes no GIL from a host-bound arm; it does take a core's
time, and what it costs a host-bound arm is for the L4 dry run to measure.
It is started and stopped outside the timed blocks, and a dead or stalled
sampler stops the session instead of passing every block.

The device is behind a `ClockReader`, so the logic runs on CPU against a
scripted reader. `NvmlReader` is the real one; it has not been run against a
GPU in this repository yet (Shakedown 2 does that).
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol

# NVML's throttle-reason bits (nvml.h, nvmlClocksThrottleReason*).
GPU_IDLE = 0x1
APPLICATIONS_CLOCKS_SETTING = 0x2
SW_POWER_CAP = 0x4
HW_SLOWDOWN = 0x8
SYNC_BOOST = 0x10
SW_THERMAL_SLOWDOWN = 0x20
HW_THERMAL_SLOWDOWN = 0x40
HW_POWER_BRAKE_SLOWDOWN = 0x80
DISPLAY_CLOCK_SETTING = 0x100

REASON_NAMES = {
    GPU_IDLE: "gpu_idle",
    APPLICATIONS_CLOCKS_SETTING: "applications_clocks_setting",
    SW_POWER_CAP: "sw_power_cap",
    HW_SLOWDOWN: "hw_slowdown",
    SYNC_BOOST: "sync_boost",
    SW_THERMAL_SLOWDOWN: "sw_thermal_slowdown",
    HW_THERMAL_SLOWDOWN: "hw_thermal_slowdown",
    HW_POWER_BRAKE_SLOWDOWN: "hw_power_brake_slowdown",
    DISPLAY_CLOCK_SETTING: "display_clock_setting",
}

# Bits the card sets for reasons that are not a slowdown: idle, and the clock
# lock the session itself asked for. They are recorded and never flag.
NEVER_FLAGS = GPU_IDLE | APPLICATIONS_CLOCKS_SETTING | DISPLAY_CLOCK_SETTING

# The researcher's set (Appendix A, 27 Oct 2026): hardware slowdown, both
# thermal slowdowns and the hardware power brake flag a block. The software
# power cap, normal on an L4 under load, is recorded and never flags. A session
# passes this to its sampler; the samplers themselves have no default.
RESEARCHER_FLAG_BITS = (HW_SLOWDOWN | SW_THERMAL_SLOWDOWN | HW_THERMAL_SLOWDOWN
                        | HW_POWER_BRAKE_SLOWDOWN)
SAMPLER_HZ = 10.0

# NVML's cumulative violation counters, and the throttle bits each one is the
# time-integral of.
VIOLATION_POLICIES = {
    "power": SW_POWER_CAP | HW_POWER_BRAKE_SLOWDOWN,
    "thermal": SW_THERMAL_SLOWDOWN | HW_THERMAL_SLOWDOWN,
}


class ClockSamplingError(RuntimeError):
    """Throttle state that cannot be observed, or a reader that cannot be built."""


def reason_names(bits: int) -> list:
    return [name for bit, name in REASON_NAMES.items() if bits & bit]


@dataclass(frozen=True)
class ClockSample:
    sm_mhz: int
    mem_mhz: int
    throttle_bits: int
    # Cumulative nanoseconds below the requested clocks, per policy; None
    # when the device cannot report them.
    violation_ns: Optional[Mapping[str, int]] = None


class ClockReader(Protocol):
    observes_transients: bool   # True when read() carries cumulative counters

    def read(self) -> ClockSample: ...


class NvmlReader:
    """`ClockReader` over NVML (`nvidia-ml-py`, imported as `pynvml`).

    `nvml` is injectable so the call sequence is tested without a GPU. The
    two function names NVML has used for the throttle read are both tried;
    a library with neither is refused at construction rather than guessed at.
    """

    _THROTTLE_FUNCS = ("nvmlDeviceGetCurrentClocksEventReasons",
                       "nvmlDeviceGetCurrentClocksThrottleReasons")

    def __init__(self, index: int = 0, *, nvml=None):
        if nvml is None:
            try:
                import pynvml as nvml
            except ImportError as e:
                raise ClockSamplingError(
                    "pynvml (nvidia-ml-py) is not installed, so GPU clocks and "
                    "throttle state cannot be sampled") from e
        self._nvml = nvml
        nvml.nvmlInit()
        self._handle = nvml.nvmlDeviceGetHandleByIndex(index)
        for name in self._THROTTLE_FUNCS:
            if hasattr(nvml, name):
                self._throttle = getattr(nvml, name)
                break
        else:
            raise ClockSamplingError(
                f"NVML has none of {self._THROTTLE_FUNCS}; throttle reasons "
                f"cannot be read")
        self.observes_transients = self._probe_violation_counters()

    def _policy_constant(self, policy: str):
        return getattr(self._nvml, f"NVML_PERF_POLICY_{policy.upper()}", None)

    def _violation(self) -> Optional[dict]:
        out = {}
        for policy in VIOLATION_POLICIES:
            const = self._policy_constant(policy)
            if const is None or not hasattr(self._nvml, "nvmlDeviceGetViolationStatus"):
                return None
            try:
                out[policy] = int(self._nvml.nvmlDeviceGetViolationStatus(
                    self._handle, const).violationTime)
            except Exception:   # NVMLError_NotSupported and its relatives
                return None
        return out

    def _probe_violation_counters(self) -> bool:
        return self._violation() is not None

    def read(self) -> ClockSample:
        n = self._nvml
        return ClockSample(
            sm_mhz=int(n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_SM)),
            mem_mhz=int(n.nvmlDeviceGetClockInfo(self._handle, n.NVML_CLOCK_MEM)),
            throttle_bits=int(self._throttle(self._handle)),
            violation_ns=self._violation())


@dataclass
class BlockClock:
    """What the sampler saw during one timed block."""
    samples: int
    sm_mhz_min: int
    sm_mhz_max: int
    mem_mhz_min: int
    mem_mhz_max: int
    throttle_bits_seen: int          # union over the in-block reads, all bits
    violation_delta_ns: Optional[dict]
    flagged: bool
    flag_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "sm_mhz_min": self.sm_mhz_min, "sm_mhz_max": self.sm_mhz_max,
            "mem_mhz_min": self.mem_mhz_min, "mem_mhz_max": self.mem_mhz_max,
            "throttle_reasons_seen": reason_names(self.throttle_bits_seen),
            "violation_delta_ns": self.violation_delta_ns,
            "flagged": self.flagged, "flag_reasons": list(self.flag_reasons),
        }


class BlockSampler:
    """Brackets a timed block with clock reads, and optionally samples inside it.

    `begin()` is called after the opening synchronise and before the clock
    starts, `end()` after the clock has stopped, so the reads are outside the
    timed region. With `interval_s` a thread also reads during the block.
    `end()` returns the block's `BlockClock`.
    """

    def __init__(self, reader: ClockReader, *, flag_bits: int,
                 interval_s: Optional[float] = None):
        if interval_s is not None and not interval_s > 0:
            raise ValueError("interval_s must be positive or None")
        if not reader.observes_transients and interval_s is None:
            raise ClockSamplingError(
                "this reader has no cumulative violation counters and no "
                "in-block sampling interval was given, so a throttle inside a "
                "block could not be seen: a boundary read after the closing "
                "synchronise sees an idle GPU. Give interval_s, or use a "
                "device that reports violation time")
        if flag_bits & NEVER_FLAGS:
            raise ValueError(
                f"{reason_names(flag_bits & NEVER_FLAGS)} are not slowdowns and "
                f"cannot flag a block")
        if not flag_bits:
            raise ValueError("flag_bits is empty, so no block could ever be flagged")
        self.reader, self.flag_bits, self.interval_s = reader, flag_bits, interval_s
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._samples: list = []
        self._before: Optional[ClockSample] = None
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def _take(self) -> ClockSample:
        s = self.reader.read()
        with self._lock:
            self._samples.append(s)
        return s

    def _loop(self, stop: threading.Event) -> None:
        try:
            while not stop.wait(self.interval_s):
                self._take()
        except BaseException as e:   # surfaced by end(), never swallowed
            self._error = e

    def begin(self) -> None:
        if self._before is not None:
            raise ClockSamplingError("begin() called twice without end()")
        self._before = self._take()
        if self.interval_s is not None:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, args=(self._stop,),
                                            daemon=True)
            self._thread.start()

    def abort(self) -> None:
        """Stop a block that raised. Records nothing."""
        if self._stop is not None:
            self._stop.set()
            self._thread.join()
        self._reset()

    def end(self) -> BlockClock:
        if self._before is None:
            raise ClockSamplingError("end() without begin()")
        if self._stop is not None:
            self._stop.set()
            self._thread.join()
        error = self._error
        after = self._take()
        samples, before = list(self._samples), self._before
        self._reset()
        if error is not None:
            raise ClockSamplingError("the sampling thread failed") from error

        bits = 0
        for s in samples:
            bits |= s.throttle_bits
        delta = None
        if before.violation_ns is not None and after.violation_ns is not None:
            delta = {p: after.violation_ns[p] - before.violation_ns[p]
                     for p in before.violation_ns}
            if any(d < 0 for d in delta.values()):
                raise ClockSamplingError(
                    f"a cumulative violation counter went backwards {delta}; "
                    f"the device was reset or the reader is wrong")
        reasons = reason_names(bits & self.flag_bits)
        if delta:
            for policy, d in delta.items():
                # A counter is the integral of all its bits, so it can flag
                # only if every one of them is in the flag set: the power
                # counter holds the software cap, which never flags.
                if d > 0 and not VIOLATION_POLICIES[policy] & ~self.flag_bits:
                    reasons.append(f"{policy}_violation_{d}ns")
        sm = [s.sm_mhz for s in samples]
        mem = [s.mem_mhz for s in samples]
        return BlockClock(
            samples=len(samples), sm_mhz_min=min(sm), sm_mhz_max=max(sm),
            mem_mhz_min=min(mem), mem_mhz_max=max(mem),
            throttle_bits_seen=bits, violation_delta_ns=delta,
            flagged=bool(reasons), flag_reasons=reasons)


# --------------------------------------------------------------------------
# The sampler process
# --------------------------------------------------------------------------

def nvml_reader_factory(index: int = 0) -> "NvmlReader":
    """What the sampler process calls to build the real reader (a top-level
    function, so it can cross a spawn)."""
    return NvmlReader(index)


_FIELDS = 4          # monotonic seconds, sm MHz, mem MHz, throttle bits
_ERR_BYTES = 512
_LOCK_TIMEOUT_S = 2.0


def _sampler_main(reader_factory, factory_args, interval_s, ring, count, lock,
                  stop, error) -> None:
    """The sampler process. Reads, appends to the ring, sleeps to the next
    tick. A failure to build or to read is written to `error` for the parent,
    not swallowed. Nothing here waits on a primitive another process could
    die holding, except the ring's lock for the few microseconds of a write
    (the parent takes it with a timeout)."""
    try:
        reader = reader_factory(*factory_args)
        capacity = len(ring) // _FIELDS
        next_tick = time.monotonic()
        while not stop.value:
            s = reader.read()
            now = time.monotonic()
            with lock:
                i = count.value % capacity
                ring[i * _FIELDS:(i + 1) * _FIELDS] = [now, s.sm_mhz, s.mem_mhz,
                                                       float(s.throttle_bits)]
                count.value += 1
            next_tick += interval_s
            while not stop.value and time.monotonic() < next_tick:
                time.sleep(min(0.02, max(0.0, next_tick - time.monotonic())))
    except BaseException as e:     # surfaced to the parent by start() or end()
        error.value = f"{type(e).__name__}: {e}".encode("utf-8", "replace")[:_ERR_BYTES - 1]


class ProcessSampler:
    """Throttle detection from a sampler process (module docstring).

    Same interface as `BlockSampler` (`begin`, `end`, `abort`, `flag_bits`,
    `interval_s`), so it is a `Controls.sampler`. `reader_factory(*factory_args)`
    is called in the child and must be picklable (a top-level function):
    `nvml_reader_factory` for the real one. Use as a context manager, or call
    `start()` and `stop()`; both are outside any timed block.
    """

    def __init__(self, reader_factory: Callable, factory_args: tuple = (), *,
                 flag_bits: int, interval_s: float = 1.0 / SAMPLER_HZ,
                 capacity: int = 8192, start_timeout_s: float = 60.0,
                 stall_s: Optional[float] = None):
        if not interval_s > 0:
            raise ValueError("interval_s must be positive")
        if flag_bits & NEVER_FLAGS:
            raise ValueError(
                f"{reason_names(flag_bits & NEVER_FLAGS)} are not slowdowns and "
                f"cannot flag a block")
        if not flag_bits:
            raise ValueError("flag_bits is empty, so no block could ever be flagged")
        if capacity < 8:
            raise ValueError("capacity must hold at least a few samples")
        self.reader_factory, self.factory_args = reader_factory, tuple(factory_args)
        self.flag_bits, self.interval_s, self.capacity = flag_bits, interval_s, capacity
        self.start_timeout_s = start_timeout_s
        self.stall_s = stall_s if stall_s is not None else max(2.0, 10 * interval_s)
        self._proc = None
        self._n0: Optional[int] = None

    # -- life cycle ---------------------------------------------------------
    def start(self) -> None:
        if self._proc is not None:
            raise ClockSamplingError("the sampler process is already running")
        ctx = multiprocessing.get_context("spawn")   # never fork a CUDA process
        self._ring = ctx.Array("d", self.capacity * _FIELDS, lock=False)
        self._count = ctx.Value("q", 0, lock=False)
        self._lock = ctx.Lock()
        self._stop = ctx.Value("b", 0, lock=False)
        self._error = ctx.Array("c", _ERR_BYTES, lock=False)
        self._proc = ctx.Process(
            target=_sampler_main, daemon=True,
            args=(self.reader_factory, self.factory_args, self.interval_s,
                  self._ring, self._count, self._lock, self._stop, self._error))
        self._proc.start()
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            self._raise_if_failed()
            if self._count.value > 0:
                return
            time.sleep(0.01)
        self.stop()
        raise ClockSamplingError(
            f"the sampler process took no first sample within "
            f"{self.start_timeout_s}s")

    def stop(self) -> None:
        proc, self._proc, self._n0 = self._proc, None, None
        if proc is None:
            return
        self._stop.value = 1
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- the ring -----------------------------------------------------------
    def _acquire(self) -> None:
        if not self._lock.acquire(timeout=_LOCK_TIMEOUT_S):
            raise ClockSamplingError(
                "the sampler's ring lock is held and not released (the sampler "
                "process died mid-write?); its samples cannot be trusted")

    def _n(self) -> int:
        return self._count.value        # an aligned 8-byte read; the ring is read under the lock

    def _raise_if_failed(self) -> None:
        msg = self._error.value
        if msg:
            raise ClockSamplingError(
                f"the sampler process failed: {msg.decode('utf-8', 'replace')}")
        if self._proc is not None and not self._proc.is_alive():
            raise ClockSamplingError(
                "the sampler process is no longer running, so a throttle could "
                "not be seen; no block may be accepted")

    def _slice(self, lo: int, hi: int) -> list:
        """Samples with ring indices lo..hi-1, refused if any was overwritten."""
        self._acquire()
        try:
            n = self._count.value
            if hi > n:
                raise ClockSamplingError("asked for samples not yet taken")
            if n - lo > self.capacity:
                raise ClockSamplingError(
                    f"the block outlasted the sampler's ring ({self.capacity} "
                    f"samples); raise capacity")
            rows = [self._ring[(i % self.capacity) * _FIELDS:
                               (i % self.capacity + 1) * _FIELDS]
                    for i in range(lo, hi)]
        finally:
            self._lock.release()
        return [ClockSample(sm_mhz=int(r[1]), mem_mhz=int(r[2]), throttle_bits=int(r[3]))
                for r in rows]

    # -- the BlockSampler interface -----------------------------------------
    def begin(self) -> None:
        if self._proc is None:
            raise ClockSamplingError("begin() before start()")
        if self._n0 is not None:
            raise ClockSamplingError("begin() called twice without end()")
        self._raise_if_failed()
        self._n0 = self._n()

    def abort(self) -> None:
        """Stop a block that raised. Records nothing; the process keeps running."""
        self._n0 = None

    def end(self) -> BlockClock:
        if self._n0 is None:
            raise ClockSamplingError("end() without begin()")
        n0, self._n0 = self._n0, None
        n1 = self._n()
        deadline = time.monotonic() + self.stall_s
        while self._n() <= n1:                      # wait for one sample after the block
            self._raise_if_failed()
            if time.monotonic() > deadline:
                raise ClockSamplingError(
                    f"the sampler took no sample in {self.stall_s}s (interval "
                    f"{self.interval_s}s): it is stalled, so this block's "
                    f"throttle state is unknown")
            time.sleep(min(0.005, self.interval_s / 4))
        samples = self._slice(max(0, n0 - 1), n1 + 1)
        bits = 0
        for s in samples:
            bits |= s.throttle_bits
        sm, mem = [s.sm_mhz for s in samples], [s.mem_mhz for s in samples]
        reasons = reason_names(bits & self.flag_bits)
        return BlockClock(
            samples=len(samples), sm_mhz_min=min(sm), sm_mhz_max=max(sm),
            mem_mhz_min=min(mem), mem_mhz_max=max(mem),
            throttle_bits_seen=bits, violation_delta_ns=None,
            flagged=bool(reasons), flag_reasons=reasons)
