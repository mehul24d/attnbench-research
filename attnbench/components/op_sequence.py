"""Operation sequence and synchronisation map per code path (T2.3).

Plan Section 5.1 defines the artifact: "the ordered operations of each arm,
the positions and mechanisms of its GPU-to-CPU synchronisations", recorded
per code path (GPU architecture, library versions, which implementation
runs) with every duration discarded. This module reads it out of a PyTorch
profiler trace (the Kineto chrome-trace JSON that
`torch.profiler.profile(...).export_chrome_trace` writes) and does four
things with it:

  - `parse_trace` turns one recorded prefill into an `OpSequence`: kernels,
    memsets, device-to-device copies, CUDA-graph launches, host-device
    copies and synchronisations, in the order the host issued them, each
    with its layer and GPU stream. Host-side compute that the recording
    marks (`HOST_COMPUTE`) is kept as a marker, because Section 5.1 charges
    it to the next operation's host cost.
  - Each synchronisation is classified by mechanism, from the aten operators
    that enclose the CUDA call, and composites are split (below).
  - `compare` classifies two sequences of one arm as "same sequence" or
    "different sequence", and `stream_check` flags an arm that uses more
    than one GPU stream (the recursion's scope, Section 5.4).
  - `sync_map` and `sync_column` give the positions and mechanisms of the
    synchronisations and the text of the arms table's "Synchronisations per
    layer" column (Section 5.5).

`to_operations` hands the sequence to the recursion (T2.8) once the
components are measured, and `to_record` writes the artifact with no
durations in it.

**The trace contract.** A trace is usable only if the recording followed
`attnbench.components.trace_recording`:

  - one user annotation `REGION` around exactly one prefill;
  - one annotation `attnbench::layer.<i>` around each decoder layer;
  - `HOST_COMPUTE` around host-side estimator work (CPU mask building);
  - CPU and CUDA activities both recorded, and nothing issued to the GPU
    from another host thread during the region.

**Mechanisms.** A CUDA call that waits for the device becomes one
synchronisation. It takes the mechanism of the aten operator that made it:

  - under `aten::is_nonzero` (Python `bool()` of a GPU tensor, which an
    `assert` on a GPU tensor also calls): "bool";
  - under `aten::item` or `aten::_local_scalar_dense`: "item";
  - a device-to-host copy and the stream synchronise that completes it,
    both inside one aten operator (`memcpy_and_sync`), fold into one
    synchronisation whose payload is the copy's bytes: "copy_to_host_pinned"
    or "copy_to_host_pageable", from the copy's destination memory;
  - a synchronise under no aten operator (`torch.cuda.synchronize()`, or a
    stream's `synchronize()`): "synchronize".

Three more kinds of wait are recognised and named, though Section 5.1
prices none of them: a host-to-device copy completed by a stream synchronise
("copy_from_host_pinned" or "copy_from_host_pageable"), an event
synchronise ("event_synchronize"), and an implicit device synchronisation
inside an allocator call such as `cudaFree` ("implicit"). They are kept in
the sequence and the sync map, so they are visible, and `to_operations`
refuses them until a researcher decides how they are costed.

**Composites.** Section 5.1 splits an `assert` on a GPU tensor and sizing
from `nonzero()` into a kernel followed by a synchronisation, with no
operation carrying both. In a trace the split is already there: the
kernels the composite launches are their own operations, and only the read
back waits. What the trace adds is attribution. A synchronisation whose
outermost enclosing aten operator is not itself a mechanism (`aten::nonzero`,
or boolean indexing through `aten::index`, which calls it) is marked
`composite` with that operator's name, and so is every device operation that
operator launched, before and after the read back. An `assert` cannot be
told from a plain `bool()` in the trace, since both call `aten::is_nonzero`
and Python records no line for it; `attribute_asserts` finds them by
aligning the shipped arm with the asserts-removed arm (T2.17), where each
removed run must be device work followed by one "bool" synchronisation.

**What "same sequence" means.** Section 5.1 does not define it. Here two
sequences are the same when every operation has the same `op_key`: its
kind, layer, issuing aten operator (innermost and outermost), mechanism,
composite, copy direction and memory kinds, and payload bytes. Kernel names
are left out by default, because they name the GPU architecture's library
kernels (two GPUs running one code path issue differently named GEMMs), and
the plan records the architecture in the code path, not in the sequence. A
kernel launched with no aten operator around it (a Triton kernel called from
Python) is identified by its kernel name, which is its Python function's
name and does not change with the GPU. A cuBLAS heuristic that issues an
extra kernel on one GPU makes the sequences different, which is the point:
the host then issues a different number of operations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from ..predict import recursion

# --- the trace contract ------------------------------------------------------

REGION = "attnbench::prefill"
HOST_COMPUTE = "attnbench::host_compute"
LAYER_PREFIX = "attnbench::layer."
LAYER_PATTERN = re.compile(r"^attnbench::layer\.(\d+)$")


def layer_annotation(i: int) -> str:
    return f"{LAYER_PREFIX}{int(i)}"


SCHEMA = "attnbench.op_sequence/1"


class SequenceError(ValueError):
    """A trace or sequence this module cannot turn into a usable artifact."""


class SeqKind(str, Enum):
    KERNEL = "kernel"
    MEMSET = "memset"
    DEVICE_COPY = "device_copy"   # device to device: device work, no link cost
    GRAPH = "graph"               # one CUDA-graph launch, many device nodes
    COPY = "copy"                 # host-device copy the host does not wait for
    SYNC = "sync"
    HOST_COMPUTE = "host_compute"


DEVICE_KINDS = frozenset({SeqKind.KERNEL, SeqKind.MEMSET, SeqKind.DEVICE_COPY,
                          SeqKind.GRAPH})

# Waits that Section 5.1 names but does not price. See the module docstring.
UNPRICED_MECHANISMS = ("copy_from_host_pinned", "copy_from_host_pageable",
                       "event_synchronize", "implicit")

# Aten operators that are the mechanism itself, so a synchronisation under
# them alone is not a composite.
MECHANISM_OPS = frozenset({
    "aten::item", "aten::_local_scalar_dense", "aten::is_nonzero",
    "aten::to", "aten::_to_copy", "aten::copy_", "aten::cpu",
})

# --- CUDA API names ----------------------------------------------------------

_LAUNCH_PREFIXES = ("cudaLaunchKernel", "cuLaunchKernel",
                    "cudaLaunchCooperativeKernel", "cuLaunchCooperativeKernel")
_GRAPH_LAUNCHES = ("cudaGraphLaunch", "cuGraphLaunch")
_STREAM_SYNCS = frozenset({"cudaStreamSynchronize", "cudaDeviceSynchronize",
                           "cuStreamSynchronize", "cuCtxSynchronize"})
_EVENT_SYNCS = frozenset({"cudaEventSynchronize", "cuEventSynchronize"})
_IMPLICIT_SYNCS = frozenset({"cudaFree", "cudaFreeHost", "cuMemFree",
                             "cuMemFree_v2", "cuMemFreeHost", "cudaDeviceReset"})
_CROSS_STREAM = frozenset({"cudaStreamWaitEvent", "cuStreamWaitEvent"})

_HOST_CATS = frozenset({"cuda_runtime", "cuda_driver"})
_DEVICE_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

_MEMCPY_NAME = re.compile(r"^Memcpy (\w+)(?: \((\w+) -> (\w+)\))?$")


def api_class(name: str) -> Optional[str]:
    """What a CUDA runtime or driver call does to the sequence, or None if
    it does nothing the sequence records (cudaGetDevice, cudaEventRecord)."""
    if name.startswith(_LAUNCH_PREFIXES):
        return "launch"
    if name.startswith(_GRAPH_LAUNCHES):
        return "graph"
    if name.startswith(("cudaMemcpy", "cuMemcpy")):
        return "memcpy" if "Async" in name else "memcpy_blocking"
    if name.startswith(("cudaMemset", "cuMemset")):
        return "memset"
    if name in _STREAM_SYNCS:
        return "sync"
    if name in _EVENT_SYNCS:
        return "event_sync"
    if name in _IMPLICIT_SYNCS:
        return "implicit_sync"
    if name in _CROSS_STREAM:
        return "cross_stream"
    return None


def parse_memcpy_name(name: str) -> tuple[str, Optional[str], Optional[str]]:
    """'Memcpy DtoH (Device -> Pinned)' -> ('DtoH', 'Device', 'Pinned')."""
    m = _MEMCPY_NAME.match(name)
    if not m:
        raise SequenceError(f"unrecognised copy activity name {name!r}")
    direction, src, dst = m.groups()
    if direction not in ("HtoD", "DtoH", "DtoD"):
        raise SequenceError(
            f"copy {name!r}: only HtoD, DtoH and DtoD copies are in scope "
            f"(one GPU, no peer copies)")
    for mem in (src, dst):
        if mem == "Managed":
            raise SequenceError(
                f"copy {name!r} touches managed memory; unified memory is not "
                f"used in this study (Section 5.6), so the row is flagged")
    return direction, src, dst


# --- the artifact ------------------------------------------------------------

@dataclass(frozen=True)
class CodePath:
    """What Section 5.1 records a sequence against.

    `gpu` must equal the device name the trace itself reports.
    `implementation` names which implementation runs (for example
    "xattn:torch" or "xattn:triton", or "compile:reduce-overhead").
    `python_optimize` is `sys.flags.optimize`, which K8 makes part of the
    code path. `libraries` holds (name, version) pairs.
    """

    arm: str
    gpu: str
    implementation: str
    libraries: tuple[tuple[str, str], ...] = ()
    python_optimize: int = 0

    def __post_init__(self):
        object.__setattr__(self, "libraries", tuple(sorted(
            (str(k), str(v)) for k, v in self.libraries)))


@dataclass(frozen=True)
class SeqOp:
    """One operation, as the host issued it. No durations.

    `aten_op` is the innermost aten operator around the CUDA call and
    `origin` the outermost; `kernels` names the device activities (one for a
    launch, every node for a graph). A synchronisation has a `mechanism`,
    and `payload_bytes` for a folded copy. `composite` names the composite
    this operation belongs to. `loaded` follows `_mark_loaded`.
    `correlation` ties the operation back to its trace and is not part of
    the artifact.
    """

    kind: SeqKind
    layer: Optional[int] = None
    api: str = ""
    aten_op: Optional[str] = None
    origin: Optional[str] = None
    kernels: tuple[str, ...] = ()
    direction: Optional[str] = None
    src_mem: Optional[str] = None
    dst_mem: Optional[str] = None
    payload_bytes: int = 0
    mechanism: Optional[str] = None
    composite: Optional[str] = None
    loaded: bool = False
    stream: Optional[int] = None
    frame: Optional[str] = None
    correlation: Optional[int] = field(default=None, compare=False)

    @property
    def is_device(self) -> bool:
        return self.kind in DEVICE_KINDS

    @property
    def priced(self) -> bool:
        """Whether the recursion has a cost for this synchronisation."""
        return self.kind is not SeqKind.SYNC or \
            self.mechanism in recursion.SYNC_MECHANISMS


@dataclass(frozen=True)
class OpSequence:
    code_path: CodePath
    context_length: int
    batch: int
    ops: tuple[SeqOp, ...]
    streams: tuple[int, ...]
    cross_stream_waits: int = 0
    ignored_calls: tuple[tuple[str, int], ...] = field(default=(), compare=False)

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({op.layer for op in self.ops if op.layer is not None}))

    def device_ops(self) -> tuple[tuple[int, SeqOp], ...]:
        """(position, op) for every operation with device work, in order:
        the kernel sequence T2.4 replays."""
        return tuple((i, op) for i, op in enumerate(self.ops) if op.is_device)


def op_key(op: SeqOp, *, payload: bool = True, kernel_names: bool = False) -> tuple:
    """The identity `compare` uses (module docstring, "same sequence")."""
    if kernel_names:
        kernels = op.kernels
    elif op.kind is SeqKind.GRAPH:
        kernels = (len(op.kernels),)
    elif op.is_device and op.aten_op is None:
        kernels = op.kernels
    else:
        kernels = ()
    return (op.kind.value, op.layer, op.aten_op, op.origin, op.mechanism,
            op.composite, op.direction, op.src_mem, op.dst_mem,
            op.payload_bytes if payload else None, kernels)


def operation_family(op: SeqOp) -> str:
    """The grouping for T2.4's replay-versus-trace check: the issuing aten
    operator, or the kernel name for a launch with none."""
    if op.aten_op is not None:
        return f"{op.kind.value}:{op.aten_op}"
    return f"{op.kind.value}:{op.kernels[0] if op.kernels else op.api}"


# --- reading a trace ---------------------------------------------------------

def load_trace(source: Union[str, Path, Mapping]) -> Mapping:
    """A trace dict, from a dict or a .json / .json.gz path."""
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class _Ev:
    idx: int
    cat: str
    name: str
    pid: object
    tid: object
    ts: float
    end: float
    args: dict


_EPS = 1e-3  # microseconds; Kineto writes nanosecond-resolution timestamps


def _events(trace: Mapping) -> list[_Ev]:
    out = []
    for idx, e in enumerate(trace.get("traceEvents", ())):
        if e.get("ph") != "X":
            continue
        ts, dur = e.get("ts"), e.get("dur", 0)
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        out.append(_Ev(idx, e.get("cat", ""), e.get("name", ""), e.get("pid"),
                       e.get("tid"), float(ts), float(ts) + float(dur),
                       dict(e.get("args") or {})))
    return out


def _containers(intervals: Sequence[_Ev], probes: Sequence[_Ev]) -> dict[int, list[_Ev]]:
    """For each probe, the intervals that contain it, outermost first.

    A sweep in start order with a stack. Intervals of one category nest on
    one thread; the explicit containment test keeps the answer right if two
    ever cross.
    """
    marks = [(iv.ts, 0, -iv.end, n, iv) for n, iv in enumerate(intervals)]
    marks += [(p.ts, 1, 0.0, n, p) for n, p in enumerate(probes)]
    marks.sort(key=lambda m: m[:4])
    stack: list[_Ev] = []
    found: dict[int, list[_Ev]] = {}
    for ts, is_probe, _, _, ev in marks:
        while stack and stack[-1].end < ts - _EPS:
            stack.pop()
        if is_probe:
            found[ev.idx] = [iv for iv in stack
                             if iv.ts <= ev.ts + _EPS and iv.end >= ev.end - _EPS]
        else:
            stack.append(ev)
    return found


def _int_arg(args: Mapping, key: str) -> Optional[int]:
    v = args.get(key)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


_FRAME = re.compile(r"^(.*)\((\d+)\): (.+)$")


def _frame_name(name: str) -> Optional[str]:
    """'/a/b/site-packages/xattn/src/X.py(12): f' -> 'X.py(12): f'.

    Only Python-defined functions; built-in calls carry an object address
    that changes per run. The directory is dropped so no host path reaches
    the artifact. The line is the function's first line, not the call's.
    """
    m = _FRAME.match(name)
    if not m:
        return None
    return f"{Path(m.group(1)).name}({m.group(2)}): {m.group(3)}"


def trace_gpus(trace: Mapping) -> dict[int, str]:
    """Device id -> name, from the trace's deviceProperties."""
    out = {}
    for d in trace.get("deviceProperties") or ():
        if "id" in d and "name" in d:
            out[int(d["id"])] = str(d["name"])
    return out


def _region(events: Sequence[_Ev], region: str, occurrence: Optional[int]) -> _Ev:
    found = sorted((e for e in events if e.cat == "user_annotation" and e.name == region),
                   key=lambda e: e.ts)
    if not found:
        raise SequenceError(
            f"no {region!r} annotation in the trace; record with "
            f"trace_recording.record_prefill_trace")
    if occurrence is None:
        if len(found) != 1:
            raise SequenceError(
                f"{len(found)} {region!r} regions in the trace; a trace holds "
                f"one prefill, or name the occurrence to read")
        return found[0]
    if not 0 <= occurrence < len(found):
        raise SequenceError(f"occurrence {occurrence} of {len(found)} regions")
    return found[occurrence]


def _mechanism(names: Sequence[str], folded: Optional[SeqOp], cls: str) -> str:
    if cls == "event_sync":
        return "event_synchronize"
    if cls == "implicit_sync":
        return "implicit"
    if "aten::is_nonzero" in names:
        return "bool"
    if "aten::item" in names or "aten::_local_scalar_dense" in names:
        return "item"
    if folded is not None:
        mem = folded.dst_mem if folded.direction == "DtoH" else folded.src_mem
        if mem not in ("Pinned", "Pageable"):
            raise SequenceError(
                f"a blocking {folded.direction} copy with host memory "
                f"{mem!r}: pinned or pageable is needed to choose its mechanism")
        side = "to_host" if folded.direction == "DtoH" else "from_host"
        return f"copy_{side}_{mem.lower()}"
    return "synchronize"


def parse_trace(trace: Union[str, Path, Mapping], code_path: CodePath, *,
                context_length: int, batch: int, region: str = REGION,
                occurrence: Optional[int] = None) -> OpSequence:
    """One recorded prefill -> its operation sequence (module docstring)."""
    trace = load_trace(trace)
    events = _events(trace)
    reg = _region(events, region, occurrence)
    thread = (reg.pid, reg.tid)
    inside = lambda e: e.ts >= reg.ts - _EPS and e.end <= reg.end + _EPS

    gpus = trace_gpus(trace)
    if not gpus:
        raise SequenceError(
            "the trace lists no CUDA device; record it with "
            "ProfilerActivity.CUDA as well as CPU")

    # Host side: every call on the region's thread, and none on any other.
    host_calls = [e for e in events if e.cat in _HOST_CATS]
    calls = [e for e in host_calls if (e.pid, e.tid) == thread and inside(e)]
    for e in host_calls:
        if (e.pid, e.tid) != thread and e.pid == reg.pid and \
                api_class(e.name) is not None and e.ts < reg.end and e.end > reg.ts:
            raise SequenceError(
                f"{e.name} issued from thread {e.tid} during the region; the "
                f"sequence is one host thread's issue order")
    marks = [e for e in events if e.cat == "user_annotation" and e.name == HOST_COMPUTE
             and (e.pid, e.tid) == thread and inside(e)]
    probes = calls + marks

    on_thread = lambda cat: [e for e in events if e.cat == cat and (e.pid, e.tid) == thread]
    aten = _containers(on_thread("cpu_op"), probes)
    notes = _containers(on_thread("user_annotation"), probes)
    frames = _containers(on_thread("python_function"), probes)

    # Device side, by correlation id.
    by_corr: dict[int, list[_Ev]] = {}
    for e in events:
        if e.cat in _DEVICE_CATS:
            c = _int_arg(e.args, "correlation")
            if c is not None:
                by_corr.setdefault(c, []).append(e)
    all_corr = {_int_arg(e.args, "correlation") for e in host_calls}

    def layer_of(p: _Ev) -> Optional[int]:
        found = [int(m.group(1)) for a in notes[p.idx]
                 if (m := LAYER_PATTERN.match(a.name))]
        if len(found) > 1:
            raise SequenceError(
                f"layer annotations nested {found}; each operation belongs to "
                f"one decoder layer")
        return found[0] if found else None

    def context(p: _Ev) -> dict:
        ops = aten[p.idx]
        fr = frames[p.idx]
        named = [f for f in (_frame_name(x.name) for x in fr) if f]
        return dict(layer=layer_of(p), aten_op=ops[-1].name if ops else None,
                    origin=ops[0].name if ops else None,
                    frame=named[-1] if named else None)

    drafts: list[tuple[SeqOp, dict]] = []   # (op, bookkeeping)
    ignored: Counter = Counter()
    cross_waits = 0
    region_corr = set()
    for p in sorted(probes, key=lambda e: (e.ts, e.idx)):
        if p.cat == "user_annotation":
            if any(a.name == HOST_COMPUTE for a in notes[p.idx] if a is not p):
                raise SequenceError("a host-compute annotation inside another one")
            drafts.append((SeqOp(SeqKind.HOST_COMPUTE, layer=layer_of(p)), {}))
            continue
        cls = api_class(p.name)
        if cls is None:
            ignored[p.name] += 1
            continue
        if cls == "cross_stream":
            cross_waits += 1
            continue
        corr = _int_arg(p.args, "correlation")
        region_corr.add(corr)
        acts = sorted(by_corr.get(corr, ()), key=lambda e: e.ts) if corr is not None else []
        ctx = context(p)
        book = {"aten_id": aten[p.idx][-1].idx if aten[p.idx] else None,
                "origin_id": aten[p.idx][0].idx if aten[p.idx] else None,
                "names": [a.name for a in aten[p.idx]]}
        streams = {_int_arg(a.args, "stream") for a in acts}
        stream = next(iter(streams)) if len(streams) == 1 else None

        def one(cat: str) -> _Ev:
            if len(acts) != 1 or acts[0].cat != cat:
                raise SequenceError(
                    f"{p.name} (correlation {corr}) has {len(acts)} device "
                    f"activities, expected one {cat!r}; the trace is incomplete")
            return acts[0]

        if cls == "launch":
            k = one("kernel")
            op = SeqOp(SeqKind.KERNEL, api=p.name, kernels=(k.name,), stream=stream,
                       correlation=corr, **ctx)
        elif cls == "graph":
            if not acts:
                raise SequenceError(f"{p.name} (correlation {corr}) launched no "
                                    f"recorded device work")
            op = SeqOp(SeqKind.GRAPH, api=p.name, kernels=tuple(a.name for a in acts),
                       stream=stream, correlation=corr, **ctx)
        elif cls == "memset":
            one("gpu_memset")
            op = SeqOp(SeqKind.MEMSET, api=p.name, kernels=(acts[0].name,),
                       stream=stream, correlation=corr, **ctx)
        elif cls in ("memcpy", "memcpy_blocking"):
            a = one("gpu_memcpy")
            direction, src, dst = parse_memcpy_name(a.name)
            nbytes = _int_arg(a.args, "bytes")
            if nbytes is None:
                raise SequenceError(f"copy {a.name!r} records no byte count")
            if direction == "DtoD":
                op = SeqOp(SeqKind.DEVICE_COPY, api=p.name, kernels=(a.name,),
                           direction=direction, src_mem=src, dst_mem=dst,
                           stream=stream, correlation=corr, **ctx)
            else:
                op = SeqOp(SeqKind.COPY, api=p.name, kernels=(a.name,),
                           direction=direction, src_mem=src, dst_mem=dst,
                           payload_bytes=nbytes, stream=stream, correlation=corr, **ctx)
                if cls == "memcpy_blocking":
                    op = replace(op, kind=SeqKind.SYNC,
                                 mechanism=_mechanism(book["names"], op, "sync"))
                else:
                    book["foldable"] = True
        else:   # a wait: sync, event_sync, implicit_sync
            prev, prev_book = drafts[-1] if drafts else (None, {})
            # drafts[-1] is the call just before this one: a copy, folded
            # only if it is the same aten operator's (memcpy_and_sync).
            fold = (cls == "sync" and prev_book.get("foldable")
                    and book["aten_id"] is not None
                    and prev_book["aten_id"] == book["aten_id"])
            if fold:
                drafts.pop()
                op = replace(prev, kind=SeqKind.SYNC, api=f"{prev.api}+{p.name}",
                             mechanism=_mechanism(book["names"], prev, cls))
            else:
                op = SeqOp(SeqKind.SYNC, api=p.name, correlation=corr,
                           mechanism=_mechanism(book["names"], None, cls), **ctx)
        drafts.append((op, book))

    _check_device_side(events, by_corr, all_corr, region_corr, reg, drafts)
    ops = _mark_composites(drafts)
    ops = _mark_loaded(ops)
    streams = sorted({_int_arg(a.args, "stream") for c in region_corr
                      for a in by_corr.get(c, ())} - {None})
    _check_gpu(trace, code_path, by_corr, region_corr, gpus)
    return OpSequence(code_path=code_path, context_length=int(context_length),
                      batch=int(batch), ops=tuple(ops), streams=tuple(streams),
                      cross_stream_waits=cross_waits,
                      ignored_calls=tuple(sorted(ignored.items())))


def _check_device_side(events, by_corr, all_corr, region_corr, reg, drafts) -> None:
    """No device work the host side cannot account for, and in order.

    An activity whose correlation matches no host call (a launch the
    profiler did not see, such as a driver-API launch with driver tracing
    off) that ran during the region would be missing from the sequence.
    """
    region_acts = [a for c in region_corr for a in by_corr.get(c, ())]
    last = max([reg.end] + [a.end for a in region_acts])
    for e in events:
        if e.cat in _DEVICE_CATS and _int_arg(e.args, "correlation") not in all_corr \
                and reg.ts <= e.ts <= last:
            raise SequenceError(
                f"device activity {e.name!r} ran during the region with no host "
                f"call in the trace; the sequence would miss it")
    streams = {_int_arg(a.args, "stream") for a in region_acts}
    if len(streams) == 1:
        starts = [min(a.ts for a in by_corr[op.correlation])
                  for op, _ in drafts if op.correlation in by_corr]
        for a, b in zip(starts, starts[1:]):
            if b < a - _EPS:
                raise SequenceError(
                    "device work on one stream ran out of host issue order; the "
                    "trace is inconsistent")


def _check_gpu(trace, code_path: CodePath, by_corr, region_corr, gpus) -> None:
    devices = {_int_arg(a.args, "device") for c in region_corr
               for a in by_corr.get(c, ())} - {None}
    if not devices:
        raise SequenceError("no device work in the region")
    names = {gpus.get(d) for d in devices}
    if len(names) != 1 or None in names:
        raise SequenceError(f"the region ran on devices {sorted(devices)}, named "
                            f"{sorted(map(str, names))}; one GPU is in scope")
    (name,) = names
    if name != code_path.gpu:
        raise SequenceError(
            f"the trace ran on {name!r}, and the code path says {code_path.gpu!r}")


def _mark_composites(drafts) -> list[SeqOp]:
    composite_of: dict[int, str] = {}
    for op, book in drafts:
        if op.kind is SeqKind.SYNC and book.get("origin_id") is not None \
                and book["names"][0] not in MECHANISM_OPS:
            composite_of[book["origin_id"]] = book["names"][0]
    out = []
    for op, book in drafts:
        name = composite_of.get(book.get("origin_id"))
        if name is not None and (op.is_device or op.kind is SeqKind.SYNC):
            op = replace(op, composite=name)
        out.append(op)
    return out


def _mark_loaded(ops: Sequence[SeqOp]) -> list[SeqOp]:
    """A copy is loaded where the sequence puts it alongside host-side
    estimator work: a host-compute marker after it and before the next
    synchronisation (while it may still be in flight). A synchronisation is
    never loaded by this rule, since the host waits through its copy."""
    out = list(ops)
    for i, op in enumerate(out):
        if op.kind is not SeqKind.COPY:
            continue
        loaded = False
        for later in out[i + 1:]:
            if later.kind is SeqKind.SYNC:
                break
            if later.kind is SeqKind.HOST_COMPUTE:
                loaded = True
                break
        out[i] = replace(op, loaded=loaded)
    return out


# --- checks over sequences ---------------------------------------------------

@dataclass(frozen=True)
class StreamCheck:
    single_stream: bool
    streams: tuple[int, ...]
    cross_stream_waits: int
    message: str


def stream_check(seq: OpSequence) -> StreamCheck:
    """Flags an arm that uses more than one GPU stream (Section 5.4 scope)."""
    single = len(seq.streams) <= 1 and seq.cross_stream_waits == 0
    if single:
        msg = f"one stream ({seq.streams[0]})" if seq.streams else "no device work"
    else:
        msg = (f"{seq.code_path.arm}: {len(seq.streams)} streams "
               f"{list(seq.streams)}, {seq.cross_stream_waits} cross-stream "
               f"waits; outside the recursion's one-stream scope")
    return StreamCheck(single, seq.streams, seq.cross_stream_waits, msg)


SAME, DIFFERENT = "same sequence", "different sequence"


@dataclass(frozen=True)
class Comparison:
    verdict: str
    first_difference: Optional[int]
    reason: str
    lengths: tuple[int, int]

    @property
    def same(self) -> bool:
        return self.verdict == SAME


def _describe(op: Optional[SeqOp]) -> str:
    if op is None:
        return "nothing"
    bits = [op.kind.value]
    if op.mechanism:
        bits.append(op.mechanism)
    if op.composite:
        bits.append(f"composite {op.composite}")
    if op.aten_op:
        bits.append(op.aten_op)
    elif op.kernels:
        bits.append(op.kernels[0])
    if op.direction:
        bits.append(op.direction)
    if op.payload_bytes:
        bits.append(f"{op.payload_bytes} B")
    return f"{' '.join(bits)} (layer {op.layer})"


def compare(a: OpSequence, b: OpSequence, *, payload: bool = True,
            kernel_names: bool = False, same_cell: bool = True) -> Comparison:
    """"same sequence" or "different sequence" for one arm (Section 5.1)."""
    if a.code_path.arm != b.code_path.arm:
        raise SequenceError(f"compare one arm with itself, got "
                            f"{a.code_path.arm!r} and {b.code_path.arm!r}")
    if same_cell and (a.context_length, a.batch) != (b.context_length, b.batch):
        raise SequenceError(
            f"different cells: ({a.context_length}, {a.batch}) and "
            f"({b.context_length}, {b.batch}); pass same_cell=False to compare "
            f"across lengths")
    ka = [op_key(o, payload=payload, kernel_names=kernel_names) for o in a.ops]
    kb = [op_key(o, payload=payload, kernel_names=kernel_names) for o in b.ops]
    lengths = (len(ka), len(kb))
    for i, (x, y) in enumerate(zip(ka, kb)):
        if x != y:
            return Comparison(DIFFERENT, i, f"operation {i}: {_describe(a.ops[i])} "
                              f"against {_describe(b.ops[i])}", lengths)
    if len(ka) != len(kb):
        i = min(len(ka), len(kb))
        longer = a if len(ka) > len(kb) else b
        return Comparison(DIFFERENT, i, f"one sequence ends at operation {i}; the "
                          f"other continues with {_describe(longer.ops[i])}", lengths)
    return Comparison(SAME, None, f"{len(ka)} operations agree", lengths)


@dataclass(frozen=True)
class SyncEntry:
    position: int
    mechanism: str
    composite: Optional[str]
    payload_bytes: int
    layer: Optional[int]
    aten_op: Optional[str]
    priced: bool


def sync_map(seq: OpSequence) -> tuple[SyncEntry, ...]:
    """Positions and mechanisms of the synchronisations (Section 5.1)."""
    return tuple(SyncEntry(i, op.mechanism, op.composite, op.payload_bytes,
                           op.layer, op.aten_op, op.priced)
                 for i, op in enumerate(seq.ops) if op.kind is SeqKind.SYNC)


def _sync_label(e: SyncEntry) -> str:
    return f"{e.mechanism} ({e.composite})" if e.composite else e.mechanism


def _counts_text(c: Counter) -> str:
    return ", ".join(f"{k} x{n}" for k, n in sorted(c.items()))


def sync_column(seq: OpSequence) -> str:
    """The arms table's "Synchronisations per layer" entry for one sequence.

    Counted per layer annotation; synchronisations outside every layer are
    stated separately. Layers with none count as zero.
    """
    entries = sync_map(seq)
    if not entries:
        return "None (checked in the trace)"
    layers = seq.layers
    per_layer = {l: Counter() for l in layers}
    outside = Counter()
    for e in entries:
        (per_layer[e.layer] if e.layer is not None else outside)[_sync_label(e)] += 1
    parts = []
    if layers:
        totals = [sum(per_layer[l].values()) for l in layers]
        if len({tuple(sorted(per_layer[l].items())) for l in layers}) == 1:
            n = totals[0]
            parts.append(f"{n} per layer in all {len(layers)} layers"
                         + (f": {_counts_text(per_layer[layers[0]])}" if n else ""))
        else:
            total = Counter()
            for l in layers:
                total.update(per_layer[l])
            parts.append(f"{min(totals)} to {max(totals)} per layer over "
                         f"{len(layers)} layers ({sum(totals)} in all: "
                         f"{_counts_text(total)})")
    if outside:
        parts.append(f"{sum(outside.values())} outside the layers: {_counts_text(outside)}")
    unpriced = sorted({e.mechanism for e in entries if not e.priced})
    if unpriced:
        parts.append(f"unpriced: {', '.join(unpriced)}")
    return "; ".join(parts)


# --- asserts, from the paired arms -------------------------------------------

def _segments(ops: Sequence[SeqOp]) -> list[tuple[Optional[int], list[int]]]:
    segs: list[tuple[Optional[int], list[int]]] = []
    for i, op in enumerate(ops):
        if segs and segs[-1][0] == op.layer:
            segs[-1][1].append(i)
        else:
            segs.append((op.layer, [i]))
    return segs


def _align_removal(a: Sequence[tuple], a_ops: Sequence[SeqOp],
                   b: Sequence[tuple]) -> Optional[list[int]]:
    """Positions of `a` to remove so that `b` is left, where every removed
    run is device work followed by one "bool" synchronisation; None if no
    such removal exists. Dynamic programme over (i, j)."""
    n, m = len(a), len(b)
    # run_end[i]: k if a[i..k-1] is device work and a[k] a "bool" sync.
    run_end = [None] * n
    nxt = None
    for i in range(n - 1, -1, -1):
        op = a_ops[i]
        if op.kind is SeqKind.SYNC:
            nxt = i if op.mechanism == "bool" else None
        elif not op.is_device:
            nxt = None
        run_end[i] = nxt
    ok = [[False] * (m + 1) for _ in range(n + 1)]
    ok[n][m] = True
    for i in range(n - 1, -1, -1):
        for j in range(m, -1, -1):
            v = j < m and a[i] == b[j] and ok[i + 1][j + 1]
            k = run_end[i]
            if not v and k is not None:
                v = ok[k + 1][j]
            ok[i][j] = v
    if not ok[0][0]:
        return None
    removed, i, j = [], 0, 0
    while i < n:
        if j < m and a[i] == b[j] and ok[i + 1][j + 1]:
            i, j = i + 1, j + 1
        else:
            k = run_end[i]
            removed.extend(range(i, k + 1))
            i = k + 1
    return removed


def attribute_asserts(with_asserts: OpSequence, stripped: OpSequence) -> OpSequence:
    """Mark the asserts in the shipped arm, from the asserts-removed arm.

    The import hook of T2.17 removes Python `assert` statements, each with
    the predicate it evaluates, and nothing else. So the stripped sequence
    must be the shipped one with runs removed, each run being the
    predicate's device work and one "bool" synchronisation. Those runs are
    the composites Section 5.1 splits: they come back marked `composite =
    "assert"`. Anything else between the two sequences is refused.
    """
    if (with_asserts.code_path.gpu, with_asserts.context_length, with_asserts.batch) != \
            (stripped.code_path.gpu, stripped.context_length, stripped.batch):
        raise SequenceError("the paired arms must be recorded on one GPU at one cell")
    sa, sb = _segments(with_asserts.ops), _segments(stripped.ops)
    if [l for l, _ in sa] != [l for l, _ in sb]:
        raise SequenceError("the paired arms differ in their layer structure")
    ops = list(with_asserts.ops)
    for (layer, ia), (_, ib) in zip(sa, sb):
        ka = [op_key(ops[i], kernel_names=True) for i in ia]
        kb = [op_key(stripped.ops[i], kernel_names=True) for i in ib]
        removed = _align_removal(ka, [ops[i] for i in ia], kb)
        if removed is None:
            raise SequenceError(
                f"layer {layer}: the stripped arm is not the shipped arm with "
                f"assert statements removed (device work then one 'bool' "
                f"synchronisation per assert)")
        for r in removed:
            ops[ia[r]] = replace(ops[ia[r]], composite="assert")
    return replace(with_asserts, ops=tuple(ops))


# --- to the recursion --------------------------------------------------------

def to_operations(seq: OpSequence, *, host: Mapping[int, float],
                  device: Mapping[int, float],
                  link: Mapping[int, float]) -> list[recursion.Operation]:
    """The sequence with its measured components, as recursion operations.

    Costs are keyed by sequence position: `host` for every operation except
    a synchronisation (T2.2's null-kernel profile, and host-side compute at
    target size for a host-compute marker), `device` for every operation
    with device work (T2.4's replay), `link` for every host-device copy
    (T2.5). A synchronisation is priced by the recursion from its mechanism.
    Missing or extra keys are refused.

    A host-compute marker is not an operation: its cost is added to the host
    cost of the next operation that is not a synchronisation (the recursion
    refuses host cost at one). Kernels, memsets, device-to-device copies and
    graph launches all become recursion kernels. The closing synchronise is
    not added here; `recursion.close_prefill` does that.
    """
    want_host = {i for i, op in enumerate(seq.ops) if op.kind is not SeqKind.SYNC}
    want_dev = {i for i, op in enumerate(seq.ops) if op.is_device}
    want_link = {i for i, op in enumerate(seq.ops) if op.kind is SeqKind.COPY}
    for name, got, want in (("host", host, want_host), ("device", device, want_dev),
                            ("link", link, want_link)):
        if set(got) != want:
            missing, extra = sorted(want - set(got)), sorted(set(got) - want)
            raise SequenceError(f"{name} costs: missing positions {missing}, "
                                f"unexpected positions {extra}")
    out: list[recursion.Operation] = []
    carried = 0.0
    for i, op in enumerate(seq.ops):
        common = dict(layer=op.layer, stream=op.stream,
                      label=op.composite or op.aten_op or (op.kernels[0] if op.kernels else op.api))
        if op.kind is SeqKind.HOST_COMPUTE:
            carried += host[i]
        elif op.kind is SeqKind.SYNC:
            if not op.priced:
                raise SequenceError(
                    f"operation {i} is a {op.mechanism!r} wait, which Section "
                    f"5.1 does not price; a researcher decides its cost")
            out.append(recursion.sync_op(op.mechanism, payload_bytes=op.payload_bytes,
                                         loaded=op.loaded, **common))
        elif op.kind is SeqKind.COPY:
            out.append(recursion.copy_op(host[i] + carried, link[i],
                                         payload_bytes=op.payload_bytes, **common))
            carried = 0.0
        else:
            out.append(recursion.kernel_op(host[i] + carried, device[i], **common))
            carried = 0.0
    if carried:
        raise SequenceError("host compute after the last operation has no "
                            "operation to be charged to")
    return out


def device_durations(trace: Union[str, Path, Mapping], seq: OpSequence) -> dict[int, float]:
    """Position -> device time in seconds, read from the trace `seq` came from.

    For T2.4's replay-versus-trace check only. A sequence artifact carries
    no durations, and no prediction input may come from a trace of the
    composed workload (Section 5.1).
    """
    trace = load_trace(trace)
    by_corr: dict[int, float] = {}
    for e in _events(trace):
        if e.cat in _DEVICE_CATS:
            c = _int_arg(e.args, "correlation")
            if c is not None:
                by_corr[c] = by_corr.get(c, 0.0) + (e.end - e.ts) * 1e-6
    out = {}
    for i, op in seq.device_ops():
        if op.correlation not in by_corr:
            raise SequenceError(f"operation {i} is not in this trace")
        out[i] = by_corr[op.correlation]
    return out


# --- the artifact on disk ----------------------------------------------------

_OP_FIELDS = ("kind", "layer", "api", "aten_op", "origin", "kernels", "direction",
              "src_mem", "dst_mem", "payload_bytes", "mechanism", "composite",
              "loaded", "stream", "frame")


def to_record(seq: OpSequence) -> dict:
    """The banked artifact: order, mechanisms and positions, no durations."""
    cp = seq.code_path
    ops = []
    for op in seq.ops:
        d = {f: getattr(op, f) for f in _OP_FIELDS}
        d["kind"] = op.kind.value
        d["kernels"] = list(op.kernels)
        ops.append(d)
    return {"schema": SCHEMA,
            "code_path": {"arm": cp.arm, "gpu": cp.gpu,
                          "implementation": cp.implementation,
                          "libraries": [list(x) for x in cp.libraries],
                          "python_optimize": cp.python_optimize},
            "context_length": seq.context_length, "batch": seq.batch,
            "streams": list(seq.streams),
            "cross_stream_waits": seq.cross_stream_waits, "ops": ops}


def record_sha256(record: Mapping) -> str:
    blob = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def from_record(record: Mapping) -> OpSequence:
    if record.get("schema") != SCHEMA:
        raise SequenceError(f"schema {record.get('schema')!r}, expected {SCHEMA!r}")
    c = record["code_path"]
    cp = CodePath(arm=c["arm"], gpu=c["gpu"], implementation=c["implementation"],
                  libraries=tuple(tuple(x) for x in c["libraries"]),
                  python_optimize=int(c["python_optimize"]))
    ops = []
    for d in record["ops"]:
        if set(d) != set(_OP_FIELDS):
            raise SequenceError(f"operation fields {sorted(d)}")
        ops.append(SeqOp(**{**d, "kind": SeqKind(d["kind"]),
                            "kernels": tuple(d["kernels"])}))
    return OpSequence(code_path=cp, context_length=int(record["context_length"]),
                      batch=int(record["batch"]), ops=tuple(ops),
                      streams=tuple(record["streams"]),
                      cross_stream_waits=int(record["cross_stream_waits"]))
