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

The device is behind a `ClockReader`, so the logic runs on CPU against a
scripted reader. `NvmlReader` is the real one; it has not been run against a
GPU in this repository yet (the L4 dry run does that).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol

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
                if d > 0 and VIOLATION_POLICIES[policy] & self.flag_bits:
                    reasons.append(f"{policy}_violation_{d}ns")
        sm = [s.sm_mhz for s in samples]
        mem = [s.mem_mhz for s in samples]
        return BlockClock(
            samples=len(samples), sm_mhz_min=min(sm), sm_mhz_max=max(sm),
            mem_mhz_min=min(mem), mem_mhz_max=max(mem),
            throttle_bits_seen=bits, violation_delta_ns=delta,
            flagged=bool(reasons), flag_reasons=reasons)
