"""Link microbenchmarks (T2.5; plan Sections 5.1, 5.3 step 3 and 5.6).

One provenance-stamped row per pairing with every link figure the model
reads:

  - **s_fixed(m)** for each mechanism in `predict.recursion.SYNC_MECHANISMS`:
    the host time to issue and complete mechanism m at the smallest payload
    (one byte), with the device idle. The device is synchronised before every
    repetition, outside the timer, and only the mechanism is timed, so the
    figure holds no transfer time and no second synchronise (unlike the
    double-synchronise harness of `scripts/measure_mask_h2d_tax.py`, whose
    29 us floor was a property of that harness).
  - **Copy bandwidth** host to device and device to host, pinned and
    pageable, from CUDA events around one copy: the device-side time the
    copy occupies the stream, which is what l_i is (Section 5.4).
  - **Blocking device-to-host bandwidth**, pinned and pageable, timed on the
    host: the payload term of s_0 for the two copy mechanisms, which the
    host waits for. B = payload / (t - s_fixed) at the largest payload, the
    plan's s_0 = s_fixed + payload / B read at one point, not fitted.
  - **Loaded bandwidth**: the same six tables while a CPU load runs in a
    separate process, one per context length. A separate process, so the
    load contends for memory and never for this interpreter's lock: Section
    5.4 charges contention to transfer terms only, never to host costs.
  - **PCIe** generation and width, current (read while a copy loop runs,
    because an idle link downshifts) and maximum; and **NUMA** placement:
    the GPU's node, that node's CPUs, and this process's affinity.

`mechanism_costs` turns the row into the recursion's `MechanismCost` table
and `hardware_features` into the learned baseline's link features. Both
refuse a dry-run row.

Every device operation is injected (`DeviceOps`), so the orchestration is
tested on CPU with fakes; `cuda_ops` builds the real ones.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..predict.recursion import SYNC_MECHANISMS, MechanismCost
from ..predict.training import HardwareFeatures

SMALLEST_PAYLOAD_BYTES = 1
DEFAULT_SIZES = (1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28)   # 1 MiB to 256 MiB
DEFAULT_CONTEXT_LENGTHS = (2048, 4096, 8192, 16384, 32768)       # Section 5.3 grid
DIRECTIONS = ("h2d", "d2h")
MEMORY_KINDS = ("pinned", "pageable")
# The two synchronising copies, and the host memory each lands in.
COPY_MECHANISMS = {"copy_to_host_pinned": "pinned",
                   "copy_to_host_pageable": "pageable"}


@dataclass
class DeviceOps:
    """What the measurement needs from a device; `cuda_ops` builds the real
    one, tests pass fakes.

      sync()                        block until the device is idle
      mechanism(name)               a callable that performs mechanism `name`
                                    at the smallest payload
      copy(direction, memory, n)    a callable that issues one n-byte copy,
                                    asynchronously where CUDA allows
      blocking_d2h(memory, n)       a callable that copies n bytes to the
                                    host and returns when they are there
      device_time(fn)               device seconds `fn` occupies the stream
    """
    sync: Callable[[], None]
    mechanism: Callable[[str], Callable[[], object]]
    copy: Callable[[str, str, int], Callable[[], object]]
    blocking_d2h: Callable[[str, int], Callable[[], object]]
    device_time: Callable[[Callable[[], object]], float]


def cuda_ops(device: str = "cuda") -> DeviceOps:
    """The real operations. Buffers are allocated once per (kind, size) and
    reused, so no timed repetition allocates."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("the link microbenchmarks need a CUDA device")
    dev = torch.device(device)
    cache: dict = {}

    def buf(where: str, n: int):
        key = (where, n)
        if key not in cache:
            if where == "device":
                cache[key] = torch.zeros(n, dtype=torch.uint8, device=dev)
            else:
                cache[key] = torch.zeros(n, dtype=torch.uint8,
                                         pin_memory=(where == "pinned"))
        return cache[key]

    one_u8 = torch.zeros(1, dtype=torch.uint8, device=dev)
    one_bool = torch.zeros(1, dtype=torch.bool, device=dev)

    def mechanism(name: str):
        if name == "synchronize":
            return torch.cuda.synchronize
        if name == "item":
            return one_u8.item
        if name == "bool":
            return lambda: bool(one_bool)
        if name in COPY_MECHANISMS:
            host = buf(COPY_MECHANISMS[name], SMALLEST_PAYLOAD_BYTES)
            return lambda: host.copy_(one_u8, non_blocking=False)
        raise KeyError(name)

    def copy(direction: str, memory: str, n: int):
        host, gpu = buf(memory, n), buf("device", n)
        if direction == "h2d":
            return lambda: gpu.copy_(host, non_blocking=True)
        return lambda: host.copy_(gpu, non_blocking=True)

    def blocking_d2h(memory: str, n: int):
        host, gpu = buf(memory, n), buf("device", n)
        return lambda: host.copy_(gpu, non_blocking=False)

    def device_time(fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e-3

    return DeviceOps(sync=torch.cuda.synchronize, mechanism=mechanism, copy=copy,
                     blocking_d2h=blocking_d2h, device_time=device_time)


def dry_run_ops(bytes_per_s: float = 10e9) -> DeviceOps:
    """A device that does nothing and reports every copy at `bytes_per_s`.
    For `scripts/measure_link.py --dry-run` only: it exercises the whole
    measurement on CPU, and the row it produces says `dry_run=True`, which
    `mechanism_costs` and `hardware_features` refuse."""
    def nothing():
        return None

    return DeviceOps(
        sync=nothing,
        mechanism=lambda name: nothing,
        copy=lambda direction, memory, n: (lambda: n),
        blocking_d2h=lambda memory, n: nothing,
        device_time=lambda fn: (fn() or 0) / bytes_per_s)


# --- timing ------------------------------------------------------------------------

def time_host(op: Callable[[], object], reps: int, *, sync: Callable[[], None],
              clock: Callable[[], int] = time.perf_counter_ns) -> list[float]:
    """Host seconds for `op`, the device synchronised before each repetition
    outside the timer, so every repetition starts from an idle device."""
    out = []
    for _ in range(reps):
        sync()
        t0 = clock()
        op()
        out.append((clock() - t0) * 1e-9)
    sync()
    return out


def time_device(op: Callable[[], object], reps: int, ops: DeviceOps) -> list[float]:
    out = []
    for _ in range(reps):
        ops.sync()
        out.append(ops.device_time(op))
    return out


def summary(xs: Sequence[float]) -> dict:
    s = sorted(xs)
    if not s:
        raise ValueError("nothing to summarise")
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if len(s) > 1 else 0.0
    return {"n": len(s), "median": statistics.median(s), "min": s[0],
            "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
            "cv": sd / mean if mean > 0 else None}


def s_fixed_table(ops: DeviceOps, reps: int, *, clock=time.perf_counter_ns) -> dict:
    """{mechanism: summary} for every mechanism the recursion costs."""
    return {m: summary(time_host(ops.mechanism(m), reps, sync=ops.sync, clock=clock))
            for m in SYNC_MECHANISMS}


def bandwidth_tables(ops: DeviceOps, sizes: Sequence[int], reps: int,
                     s_fixed: dict, *, clock=time.perf_counter_ns) -> dict:
    """Bytes per second per size for the four event-timed copies and the two
    blocking device-to-host copies. Keys: "<direction>_<memory>" and
    "blocking_d2h_<memory>"; each value is a list of
    {bytes, median_s, bytes_per_s}, smallest size first."""
    out: dict = {}
    for direction in DIRECTIONS:
        for memory in MEMORY_KINDS:
            rows = []
            for n in sizes:
                med = summary(time_device(ops.copy(direction, memory, n), reps, ops))["median"]
                rows.append({"bytes": n, "median_s": med, "bytes_per_s": n / med})
            out[f"{direction}_{memory}"] = rows
    for mech, memory in COPY_MECHANISMS.items():
        rows = []
        fixed = s_fixed[mech]["median"]
        for n in sizes:
            med = summary(time_host(ops.blocking_d2h(memory, n), reps,
                                    sync=ops.sync, clock=clock))["median"]
            net = med - fixed
            rows.append({"bytes": n, "median_s": med,
                         "bytes_per_s": n / net if net > 0 else None})
        out[f"blocking_d2h_{memory}"] = rows
    return out


def headline(table: list[dict]) -> Optional[float]:
    """The bandwidth the model uses: at the largest payload."""
    return table[-1]["bytes_per_s"] if table else None


# --- the CPU load --------------------------------------------------------------------

def _reference_builder_load(context_length: int, density: float) -> Callable[[], object]:
    import torch
    from .. import masks
    n = masks._n_blocks(context_length, 128)
    scores = torch.rand(n, n, generator=torch.Generator().manual_seed(context_length))

    def build():
        return masks.importance_block_mask(
            context_length, 128, 1.0 - density, scores, causal=True,
            identity_seed=f"link-load-{context_length}")
    return build


def _python_loop_load(context_length: int, density: float) -> Callable[[], object]:
    from .host_speed import single_thread_loop
    return lambda: single_thread_loop(10_000)


LOADS = {"reference_builder": _reference_builder_load,
         "python_loop": _python_loop_load}


def _load_worker(name, context_length, density, ready, stop, count) -> None:
    work = LOADS[name](context_length, density)
    work()
    ready.set()
    while not stop.is_set():
        work()
        with count.get_lock():
            count.value += 1


class CpuLoad:
    """A load process running `LOADS[name]` until the block ends.

    Entering waits until the worker has finished one build, so no
    measurement starts before the load does; `iterations` is how many builds
    completed during the block, recorded so a row can show the load ran.
    """

    def __init__(self, name: str, context_length: int, density: float,
                 start_timeout: float = 120.0):
        if name not in LOADS:
            raise KeyError(f"unknown load {name!r}; known: {sorted(LOADS)}")
        self.name, self.context_length, self.density = name, context_length, density
        self.start_timeout = start_timeout
        self.iterations: Optional[int] = None

    def __enter__(self) -> "CpuLoad":
        ctx = mp.get_context("spawn")
        self._ready, self._stop = ctx.Event(), ctx.Event()
        self._count = ctx.Value("q", 0)
        self._proc = ctx.Process(target=_load_worker, daemon=True, args=(
            self.name, self.context_length, self.density,
            self._ready, self._stop, self._count))
        self._proc.start()
        if not self._ready.wait(self.start_timeout):
            self._halt()
            raise RuntimeError(f"the {self.name} load did not start within "
                               f"{self.start_timeout:.0f} s")
        with self._count.get_lock():
            self._count.value = 0
        return self

    def _halt(self) -> None:
        self._stop.set()
        self._proc.join(30)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5)

    def __exit__(self, *exc) -> None:
        with self._count.get_lock():
            self.iterations = int(self._count.value)
        self._halt()


# --- PCIe and NUMA ---------------------------------------------------------------------

PCIE_QUERY = ("pcie.link.gen.current,pcie.link.gen.max,"
              "pcie.link.width.current,pcie.link.width.max")


def query_pcie() -> Optional[dict]:
    from ..provenance import _sh_result
    ok, out = _sh_result(["nvidia-smi", f"--query-gpu={PCIE_QUERY}",
                          "--format=csv,noheader,nounits"])
    if not ok or not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) != 4:
        return None

    def num(v):
        return int(v) if v.isdigit() else None
    return dict(zip(("gen_current", "gen_max", "width_current", "width_max"),
                    map(num, parts)))


def pcie_during(copy: Callable[[], object], query: Callable[[], Optional[dict]],
                *, settle_s: float = 0.2) -> Optional[dict]:
    """Query the link while `copy` runs in a loop on another thread (a copy
    call releases the interpreter lock), so `gen_current` is the speed the
    link trains to under traffic rather than its idle downshift."""
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            copy()

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        time.sleep(settle_s)
        return query()
    finally:
        stop.set()
        t.join(30)


def _sysfs_bus_id(nvidia_bus_id: str) -> Optional[str]:
    """`00000000:00:04.0` (nvidia-smi) to `0000:00:04.0` (sysfs)."""
    try:
        domain, bus, devfn = nvidia_bus_id.strip().split(":")
        return f"{int(domain, 16):04x}:{bus.lower()}:{devfn.lower()}"
    except ValueError:
        return None


def _cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def numa_placement(nvidia_bus_id: Optional[str], *, sysfs: str = "/sys",
                   affinity: Optional[set[int]] = None) -> dict:
    """The GPU's NUMA node, its CPUs, this process's affinity, and whether
    the affinity lies within that node (Section 5.6: threads pinned to the
    GPU's node). None wherever the host does not say; a node of -1 is the
    kernel's "no NUMA information" and is recorded as it is."""
    import os
    if affinity is None and hasattr(os, "sched_getaffinity"):
        affinity = set(os.sched_getaffinity(0))
    out = {"gpu_pci_bus_id": nvidia_bus_id, "gpu_numa_node": None,
           "numa_node_cpus": None,
           "process_cpu_affinity": (",".join(map(str, sorted(affinity)))
                                    if affinity else None),
           "affinity_on_gpu_node": None}
    sys_id = _sysfs_bus_id(nvidia_bus_id) if nvidia_bus_id else None
    if sys_id is None:
        return out
    try:
        node = int(Path(sysfs, "bus/pci/devices", sys_id, "numa_node").read_text())
    except (OSError, ValueError):
        return out
    out["gpu_numa_node"] = node
    if node < 0:
        return out
    try:
        cpus = _cpulist(Path(sysfs, f"devices/system/node/node{node}/cpulist").read_text())
    except OSError:
        return out
    out["numa_node_cpus"] = ",".join(map(str, sorted(cpus)))
    if affinity:
        out["affinity_on_gpu_node"] = affinity <= cpus
    return out


def gpu_bus_id() -> Optional[str]:
    from ..provenance import _sh_result
    ok, out = _sh_result(["nvidia-smi", "--query-gpu=pci.bus_id",
                          "--format=csv,noheader"])
    return out.splitlines()[0].strip() if ok and out else None


# --- the row ---------------------------------------------------------------------------

def measure(ops: DeviceOps, *, sync_reps: int = 2000, copy_reps: int = 50,
            sizes: Sequence[int] = DEFAULT_SIZES,
            context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
            load: str = "reference_builder", load_density: float = 0.5,
            pcie: Optional[Callable[[], Optional[dict]]] = None,
            numa: Optional[dict] = None, dry_run: bool = False,
            clock=time.perf_counter_ns) -> dict:
    """Every T2.5 field in one flat row. Per-size tables travel as JSON in
    `bandwidth_detail` and `loaded_bandwidth_detail`; the headline figures
    (largest payload) are columns of their own."""
    sizes = sorted(sizes)
    ops.sync()
    fixed = s_fixed_table(ops, sync_reps, clock=clock)
    row: dict = {"dry_run": dry_run, "sync_reps": sync_reps, "copy_reps": copy_reps,
                 "sizes_bytes": ",".join(map(str, sizes)),
                 "smallest_payload_bytes": SMALLEST_PAYLOAD_BYTES}
    for m, s in fixed.items():
        for k in ("median", "min", "p90", "cv"):
            row[f"s_fixed_{m}_{k}"] = s[k]
    unloaded = bandwidth_tables(ops, sizes, copy_reps, fixed, clock=clock)
    for key, table in unloaded.items():
        row[f"bw_{key}_Bps"] = headline(table)
    row["bandwidth_detail"] = json.dumps(unloaded)

    loaded_detail = {}
    row["load"], row["load_density"] = load, load_density
    for L in context_lengths:
        with CpuLoad(load, L, load_density) as cpu:
            tables = bandwidth_tables(ops, sizes, copy_reps, fixed, clock=clock)
        loaded_detail[str(L)] = tables
        row[f"load_iterations_{L}"] = cpu.iterations
        for key, table in tables.items():
            row[f"loaded_bw_{key}_{L}_Bps"] = headline(table)
    row["context_lengths"] = ",".join(map(str, context_lengths))
    row["loaded_bandwidth_detail"] = json.dumps(loaded_detail)

    idle = busy = None
    if pcie is not None:
        idle = pcie()
        busy = pcie_during(ops.copy("h2d", "pinned", sizes[-1]), pcie)
    for k in ("gen_current", "gen_max", "width_current", "width_max"):
        row[f"pcie_{k}_idle"] = idle.get(k) if idle else None
        row[f"pcie_{k}_under_copy"] = busy.get(k) if busy else None
    row.update(numa or numa_placement(None))
    return row


# --- readers -----------------------------------------------------------------------------

def _refuse_dry_run(row) -> None:
    """A dry-run row's figures come from fakes. Read only behind the stubbed
    GPU boundary (`attnbench.cpu_boundary`), whose rehearsal never prices
    anything banked."""
    from .. import cpu_boundary
    if bool(row["dry_run"]) and not cpu_boundary.active():
        raise ValueError("this is a dry-run row: its figures come from fakes")


def mechanism_costs(row, context_length: Optional[int] = None) -> dict[str, MechanismCost]:
    """The recursion's s_fixed table (seconds, bytes per second).

    The copy mechanisms take the blocking device-to-host bandwidth; with a
    `context_length`, also the loaded figure measured under that length's
    load. Mechanisms that move no payload carry no bandwidth."""
    _refuse_dry_run(row)
    out = {}
    for m in SYNC_MECHANISMS:
        bw = loaded = None
        if m in COPY_MECHANISMS:
            mem = COPY_MECHANISMS[m]
            bw = row[f"bw_blocking_d2h_{mem}_Bps"]
            if context_length is not None:
                loaded = row[f"loaded_bw_blocking_d2h_{mem}_{context_length}_Bps"]
        out[m] = MechanismCost(s_fixed=float(row[f"s_fixed_{m}_median"]),
                               bandwidth=bw, loaded_bandwidth=loaded)
    return out


def hardware_features(row, direction: str = "h2d") -> HardwareFeatures:
    """The learned baseline's pinned and pageable bandwidth. Host to device
    by default: the direction the arms' non-synchronising copies (masks)
    move."""
    _refuse_dry_run(row)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    return HardwareFeatures(
        pinned_bandwidth=float(row[f"bw_{direction}_pinned_Bps"]),
        pageable_bandwidth=float(row[f"bw_{direction}_pageable_Bps"]))
