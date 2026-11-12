"""Device components by replay (T2.4; plan Sections 5.1, 5.3 step 5, 5.6).

c_dev(i) is "GPU execution time of operation i, from replaying the arm's
whole kernel sequence in order on the target GPU, with the masks recorded
for its code path, as a CUDA graph (primary; keeps cache state between
kernels and has no host gaps). Isolated per-operation replay is the
cross-check, and the difference between the two is reported as the cache
effect."

**In-order replay.** The arm's own prefill code is captured as one CUDA
graph, on the recorded prompt, with two things held fixed from the
recording (`recorder.record_arm_prefill`):

  - every host-device interaction, served from the tape (`sync_tape`), so
    the capture holds no synchronisation and no host-to-device copy;
  - the mask the block-sparse kernel receives at each call, the banked one
    (`mask_bank.MaskCapture(substitute=...)`). The estimator's own kernels
    still run, so their device cost is in the replay; the mask they produce
    is kept and compared with the bank (`mask_agreement`), which must be
    bitwise on the code path that recorded it.

**The live masks are read after a replay** (researcher's ruling, 1 Nov
2026). A graph's capture records kernels and runs none, so a live mask read
straight after it was never written (Shakedown 3: all-False in all 28 layers,
agreement 1 - 7380/49152 by construction). Each live mask is therefore copied,
inside the captured prefill, into a buffer that holds `NEVER_WRITTEN` from
before the capture (`live_buffers`), the graph is replayed once and
synchronised, and a call whose buffer still holds the marker is refused
before anything is compared (`check_masks`, `mask_agreement`). The copies
are work of the check, not of the arm, so the check has a graph of its own,
dropped before the timed graph is captured: the timed graph and its node
match hold the arm's kernels only.

The graph is replayed under the profiler once to read every node's device
time (`replay_nodes`), and without it `reps` times with CUDA events for the
total (`time_replay`), so the profiler's effect on the node times is
visible as the ratio of the two.

**Positions.** `match` assigns replayed nodes to the positions of the
recorded sequence by structure only (researcher's ruling 1 of 6 Nov 2026,
after the debug session found the name rule refusing every sparse cell): an
activity's class, stream, order and payload class, the strict "same
sequence" definition's terms, so a renamed kernel (a CUDA graph runs a device
copy as `memcpy32_post`) or a copy's other launch variant is not a mismatch.
Names only place the comparison (`match`'s docstring). A host-device copy and
a synchronisation have no device cost in the recursion (l_i and s_0 are
T2.5's), and the tape keeps them out of the graph. Whatever does not match is
reported: positions with no node (the tape's substitutions, `nonzero_static`
for `nonzero`, are the expected case) and nodes with no position.
`device_costs` refuses an incomplete match.

**Pricing** (plan Section 5.1, Appendix A, 27 Oct and 7 Nov 2026). A position
the graph does not hold as recorded is priced from an isolated replay of the
original operators: `isolated_replay(..., substitute=False)` runs the prefill
eagerly without the tape, so `nonzero` and the host-to-device copy run as
recorded, each alone on a cold cache (step 6 runs it segmented and compacted,
`isolated_replay_compact` with `trace_stream.DEVICE_FIELDS`).
`price_positions` labels every position by how it was priced (`GRAPH`,
`ISOLATED_ORIGINAL`, `VARIANT_DIFFERS`) and counts the nodes it dropped
(`EXTRA_NODE_DROPPED`). It refuses nothing: `pricing_gate` takes, per arm
over its cells, the share of priced device time priced from the isolated
original (structure the graph does not hold as eager), and an arm above the
T4.1 lock value (working 5%), or with a position neither replay matches, is
not priced and is labelled `PRICING_NOT_VALIDATED`.

**Isolated replay.** The same prefill, eagerly, from the tape and the bank,
with the device synchronised and the L2 cache flushed before every aten
operator (`IsolationMode`), under the profiler. Each operation then runs
alone on a cold cache. Kernels that no aten operator issues (a Triton kernel
called from Python) are not separated from their neighbours. The flush's own
kernels are excluded by their annotation.

**The replay-versus-trace check** (Gate A, Section 5.6 "Replay mismatch")
compares replayed device time with the device time in a recorded trace, per
operation family (`op_sequence.operation_family`), and reports the
positions `match` leaves unmatched (by structure, since 7 Nov 2026; the key
keeps its old name, `kernel_name_mismatch`). The bound is fixed at lock, so
`replay_vs_trace` reports and `check_passes` takes the bound as an argument.

The graph is behind `GraphBackend` so the bookkeeping is tested on CPU with
`EagerBackend`; `CudaGraphBackend` is the real one.
"""

from __future__ import annotations

import math
import bisect
import difflib
import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
from torch.autograd.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode

from . import mask_bank, op_sequence
from .op_sequence import OpSequence, SeqKind, operation_family
from .sync_tape import SyncTape, TapeReplayer

REPLAY_REGION = "attnbench::replay"
FLUSH = "attnbench::isolation_flush"
SCHEMA = "attnbench.replay/2"
SCHEMAS_READ = ("attnbench.replay/1", SCHEMA)
GRAPH, ISOLATED_ORIGINAL = "graph", "isolated_original"
NEVER_WRITTEN = 2          # in a live-mask buffer: no replay has written it (a mask is 0 or 1)
# The aten operators issuing the device work at a position the tape
# substitutes (`sync_tape.SUBSTITUTED_KINDS`): `nonzero` (and `torch.where`
# with one argument, through `nonzero_numpy`), boolean indexing and
# assignment, and a host-to-device `copy_`. Read from PyTorch's dispatch, not
# yet from an L4 trace: T2.27's replay step reports the positions it prices.
SUBSTITUTED_ATEN_OPS = frozenset({
    "aten::nonzero", "aten::nonzero_numpy", "aten::index", "aten::index_put_",
    "aten::_index_put_impl_", "aten::copy_",
    # the tape's `__setitem__` entry serves a boolean-mask assignment with an
    # `index_put_`; eagerly the same assignment of a scalar is `masked_fill_`
    # (found by the debug session, 2 Nov 2026: 28 of them in the oracle's
    # recording, none in its graph)
    "aten::masked_fill_"})
# Where the tape serves a boolean mask (`nonzero`, boolean indexing and
# assignment) the graph runs its stand-in (`nonzero_static`, an `index_put_`),
# so a node of another name matched one for one to such a position is the
# stand-in's kernel, not the position's work (`stand_in_positions`; ruling B3
# of 8 Nov 2026: T3.2's eager `masked_fill_` after an `eq` was priced at the
# stand-in's `index_put_`, 3.2 to 4.0 times its traced time). A copy is left
# out: a graph's copy kernel does the copy the eager one does.
STAND_IN_ATEN_OPS = SUBSTITUTED_ATEN_OPS - {"aten::copy_"}


class ReplayError(RuntimeError):
    pass


# --- backends ------------------------------------------------------------------

class CudaGraphBackend:
    """Capture with `torch.cuda.graph`; time with CUDA events."""

    def capture(self, body: Callable[[], None]):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        return g

    def warmup(self, body: Callable[[], None]) -> None:
        """PyTorch's recipe: warm up on a side stream before capture, so lazy
        initialisation is not captured."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

    def replay(self, g) -> None:
        g.replay()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def time_once(self, g) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e-3


class EagerBackend:
    """Stands in for the graph on CPU: "capture" runs the body once and keeps
    it; "replay" runs it again, under a fresh tape replayer each time."""

    def __init__(self):
        self.replays = 0

    def capture(self, body):
        body()
        return body

    def warmup(self, body) -> None:
        body()

    def replay(self, body) -> None:
        self.replays += 1
        body()

    def synchronize(self) -> None:
        pass

    def time_once(self, body) -> float:
        t0 = time.perf_counter()
        self.replay(body)
        return time.perf_counter() - t0


# --- capture -----------------------------------------------------------------------

@dataclass
class Captured:
    handle: object
    backend: object
    tape: SyncTape
    banked: Optional[list[torch.Tensor]]
    live_masks: list[torch.Tensor] = field(default_factory=list)   # the check graph's buffers
    staged: dict = field(default_factory=dict)
    mask_check: Optional["MaskCheck"] = None

    def replay(self) -> None:
        self.backend.replay(self.handle)


@dataclass(frozen=True)
class MaskCheck:
    """The live masks of the check graph after its one replay (module
    docstring): the agreement with the bank, or why there is none."""

    agreement: Optional[dict]
    unwritten: tuple            # kernel calls whose buffer no replay wrote
    replays: int


def _bank_capture(banked, kernel_module, into=None) -> mask_bank.MaskCapture:
    return mask_bank.MaskCapture(module=kernel_module, substitute=banked,
                                 optional=banked is None, into=into)


def live_buffers(banked: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """One buffer per banked call, at its shape, holding `NEVER_WRITTEN`."""
    return [torch.full(tuple(b.shape), NEVER_WRITTEN, dtype=torch.uint8, device=b.device)
            for b in banked]


def unwritten(buffers: Sequence[torch.Tensor]) -> list[int]:
    """The kernel calls whose live-mask buffer still holds `NEVER_WRITTEN`."""
    return [i for i, m in enumerate(buffers) if bool((m == NEVER_WRITTEN).any())]


def _check_calls(cap: mask_bank.MaskCapture, banked) -> None:
    n = len(banked) if banked is not None else 0
    if len(cap.calls) != n:
        raise ReplayError(f"the replay made {len(cap.calls)} kernel calls; "
                          f"{n} masks are banked")


def _body(run_prefill, tape, staged, banked, kernel_module, into=None):
    def body():
        rep = TapeReplayer(tape, staged)
        with rep, _bank_capture(banked, kernel_module, into) as cap:
            run_prefill()
        rep.finish()
        _check_calls(cap, banked)
    return body


def check_masks(run_prefill, tape, staged, banked, backend, *, kernel_module=None,
                replays: int = 1) -> tuple[MaskCheck, list[torch.Tensor]]:
    """The check graph: the prefill captured with each live mask copied into
    a `live_buffers` buffer, replayed `replays` times (one; zero only to show
    what a capture alone writes), synchronised, then read (module
    docstring). Its graph is dropped here, before the timed one is
    captured."""
    buffers = live_buffers(banked)
    handle = backend.capture(_body(run_prefill, tape, staged, banked, kernel_module, buffers))
    for _ in range(replays):
        backend.replay(handle)
    backend.synchronize()
    del handle
    missing = unwritten(buffers)
    agreement = None if missing else mask_bank.agreement([b.bool() for b in buffers], banked)
    return MaskCheck(agreement, tuple(missing), replays), buffers


def capture(run_prefill: Callable[[], None], tape: SyncTape, *, device,
            masks: Optional[mask_bank.MaskRecord] = None, backend=None,
            warmup: int = 1, kernel_module=None) -> Captured:
    """The arm's prefill as one graph, from the tape and the bank. The modes
    are entered inside the capture, so the backend's own synchronisation
    around it is not taken for the arm's. With a bank, the mask check runs
    first on a graph of its own (`check_masks`), so the timed graph holds the
    arm's kernels and nothing of the check's."""
    backend = backend or CudaGraphBackend()
    if tape.unhandled:
        raise ReplayError(f"the tape has unhandled interactions at {tape.unhandled}: "
                          f"{tape.entries[tape.unhandled[0]].note}")
    staged = tape.stage(device)
    banked = masks.tensors(device) if masks is not None else None
    body = _body(run_prefill, tape, staged, banked, kernel_module)
    for _ in range(warmup):
        backend.warmup(body)
    check, buffers = (None, [])
    if banked is not None:
        check, buffers = check_masks(run_prefill, tape, staged, banked, backend,
                                     kernel_module=kernel_module)
    handle = backend.capture(body)
    return Captured(handle, backend, tape, banked, buffers, staged, check)


def mask_agreement(cap: Captured) -> Optional[dict]:
    """The estimator's live masks against the bank, read after one replay of
    the check graph and a synchronise (module docstring). Refuses a call
    whose mask no replay wrote."""
    if cap.banked is None:
        return None
    if cap.mask_check.unwritten:
        raise ReplayError(
            f"the live masks of kernel calls {list(cap.mask_check.unwritten)} were never "
            f"written after a replay and a synchronise; nothing is compared with the bank")
    return cap.mask_check.agreement


def time_replay(cap: Captured, reps: int) -> dict:
    """Total device seconds per replay, unprofiled, over `reps` replays."""
    if reps < 1:
        raise ValueError("reps must be >= 1")
    xs = sorted(cap.backend.time_once(cap.handle) for _ in range(reps))
    return {"reps": reps, "median_s": statistics.median(xs), "min_s": xs[0],
            "p90_s": xs[min(len(xs) - 1, int(0.9 * len(xs)))]}


def profile_replay(cap: Captured, path) -> Path:
    """One replay under the profiler, inside REPLAY_REGION."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    cap.backend.synchronize()
    with profile(activities=acts) as prof:
        with record_function(REPLAY_REGION):
            cap.replay()
        cap.backend.synchronize()
    prof.export_chrome_trace(str(path))
    return Path(path)


# --- reading replayed nodes ----------------------------------------------------------

@dataclass(frozen=True)
class Node:
    cat: str
    name: str
    duration_s: float
    correlation: Optional[int]
    stream: Optional[int] = None       # the device's stream id, as the trace gives it
    bytes: Optional[int] = None        # a copy's or a memset's, as the trace gives it


def replay_nodes(trace: Union[str, Path, Mapping], *, region: str = REPLAY_REGION,
                 exclude: Optional[str] = None) -> list[Node]:
    """Device activities of the host calls inside `region`, in device order;
    those whose host call sits inside an `exclude` annotation are dropped.
    A compact stream (`trace_stream.is_compact`, written with
    `DEVICE_FIELDS`) is read lazily, in passes, never held whole."""
    rows = _rows(trace)
    keep, _ = _kept_launches(rows, region, exclude)
    acts = [(ts, cat, name, end, corr, stream, nbytes)
            for cat, name, pid, tid, ts, end, corr, stream, nbytes in rows()
            if cat in op_sequence._DEVICE_CATS and corr is not None and int(corr) in keep]
    acts.sort(key=lambda a: a[0])
    return [Node(cat, name, (end - ts) * 1e-6, int(corr), stream, nbytes)
            for ts, cat, name, end, corr, stream, nbytes in acts]


# The host calls that put work on the device, each with an activity of its own.
_ACTIVITY_LAUNCHES = op_sequence._LAUNCH_PREFIXES + ("cudaMemcpy", "cudaMemset",
                                                     "cuMemcpy", "cuMemset")


def launches_without_activity(trace: Union[str, Path, Mapping], *, region: str = REPLAY_REGION,
                              exclude: Optional[str] = None) -> list[int]:
    """The correlations of the launches `replay_nodes` keeps (a kernel launch,
    a copy or a memset, inside `region` and outside every `exclude`) that no
    device activity in the trace answers: work the device did and the trace
    lost (researcher's ruling 2 of 7 Nov 2026, per segment in
    `segment_report`)."""
    rows = _rows(trace)
    keep, names = _kept_launches(rows, region, exclude)
    answered = {int(corr) for cat, _, _, _, _, _, corr, *_ in rows()
                if cat in op_sequence._DEVICE_CATS and corr is not None}
    return sorted(c for c in keep - answered if names[c].startswith(_ACTIVITY_LAUNCHES))


def _rows(trace) -> Callable:
    """A fresh iterator of (cat, name, pid, tid, ts, end, correlation, stream,
    bytes) per call, from a trace or a compact stream with `DEVICE_FIELDS`.
    `end` is `ts + dur` as `op_sequence._events` takes it, so both readers
    give a node the same duration, `end - ts`."""
    from . import trace_stream
    if trace_stream.is_compact(trace):
        if trace_stream.fields_of(trace) != trace_stream.DEVICE_FIELDS:
            raise ReplayError(f"{Path(trace).name}: a compact stream without the device "
                              f"fields; its nodes cannot be read")
        return lambda: ((c, n, pid, tid, float(ts), float(ts) + float(dur), *rest)
                        for _, c, n, pid, tid, ts, dur, *rest in trace_stream.iter_rows(trace))
    trace = op_sequence.load_trace(trace)
    arg = op_sequence._int_arg
    rows = [(e.cat, e.name, e.pid, e.tid, e.ts, e.end, arg(e.args, "correlation"),
             arg(e.args, "stream"), arg(e.args, "bytes")) for e in op_sequence._events(trace)]
    return lambda: iter(rows)


def _kept_launches(rows: Callable, region: str, exclude: Optional[str]) -> tuple[set, dict]:
    """The correlations of the host calls inside `region` on its thread and
    outside every `exclude`, and each one's call name, in two passes over
    `rows()`: the region and the skips, then the calls."""
    regions, skips = [], []
    for cat, name, pid, tid, ts, end, *_ in rows():
        if cat == "user_annotation":
            if name == region:
                regions.append((pid, tid, ts, end))
            elif exclude and name == exclude:
                skips.append((ts, end))
    if len(regions) != 1:
        raise ReplayError(f"{len(regions)} {region!r} regions in the trace, expected one")
    rpid, rtid, r0, r1 = regions[0]
    eps = op_sequence._EPS
    skips.sort()
    starts = [a for a, _ in skips]
    reach, far = [], float("-inf")      # reach[i]: the latest end among the first i+1 skips
    for _, b in skips:
        far = max(far, b)
        reach.append(far)
    del skips
    # an isolated replay has one flush per operator, so a scan of every skip for every
    # host event is quadratic (an 8192-token oracle cell ran 14 minutes in it, 2 Nov 2026)
    names: dict = {}
    for cat, name, pid, tid, a, b, corr, *_ in rows():
        if cat not in op_sequence._HOST_CATS or (pid, tid) != (rpid, rtid) or corr is None:
            continue
        if a < r0 - eps or b > r1 + eps:
            continue
        i = bisect.bisect_right(starts, a)
        if i > 0 and b <= reach[i - 1]:
            continue
        names[int(corr)] = name
    return set(names), names


# --- matching nodes to positions -------------------------------------------------------

@dataclass(frozen=True)
class Match:
    device_s: dict                 # position -> device seconds (matched nodes only)
    unmatched_positions: tuple     # positions with at least one activity not matched
    extra_nodes: tuple             # replayed node names with no position
    families_unmatched: dict       # family -> count of unmatched positions
    n_expected: int
    n_nodes: int
    variant_matched: tuple = ()    # positions matched with a name that differs (VARIANT_DIFFERS)
    extra_s: float = 0.0           # the device seconds of the extra nodes
    by_segment: tuple = ()         # `match_segmented`: per segment, what it holds and lost

    @property
    def complete(self) -> bool:
        return not self.unmatched_positions and not self.extra_nodes


def expected_nodes(seq: OpSequence) -> list[tuple[int, str]]:
    """(position, activity name) for every device activity of the sequence."""
    return [(i, name) for i, op in seq.device_ops() for name in op.kernels]


# A CUDA graph runs a captured device-to-device copy as a copy kernel, which the
# trace files as a kernel of this name (`memcpy32_post`, `memcpy128`): the debug
# session's 84 oracle and 336 XAttention graph nodes, 2 Nov 2026, every one of
# them in place of an eager "Memcpy DtoD (Device -> Device)".
_GRAPH_COPY_KERNEL = re.compile(r"^memcpy\d+(?:_post)?$")


def activity_class(name: str, cat: Optional[str] = None) -> str:
    """The structural class of one device activity: a memset, a device copy,
    a host-device copy by direction, or a kernel. From the trace's category
    where there is one (a replayed node), else from the activity's name as
    the trace writes a copy and a memset (a sequence position keeps names
    only); a graph's copy kernel is the device copy it runs."""
    if cat == "gpu_memset" or name.startswith("Memset ("):
        return "memset"
    if cat == "gpu_memcpy" or name.startswith("Memcpy "):
        try:
            direction = op_sequence.parse_memcpy_name(name)[0]
        except op_sequence.SequenceError:
            direction = "unknown"
        return "device_copy:DtoD" if direction == "DtoD" else f"copy:{direction}"
    if _GRAPH_COPY_KERNEL.match(name):
        return "device_copy:DtoD"
    return "kernel"


def _stream_index(streams):
    first: dict = {}
    return [None if s is None else first.setdefault(s, len(first)) for s in streams]


def structural_keys(seq: OpSequence) -> list[tuple]:
    """(class, stream by order of first use, payload class) for each activity of
    `expected_nodes`: what a position is matched by (researcher's ruling 1 of
    6 Nov 2026), the strict definition's terms for one activity. A device
    position's payload class is the sequence's ("none": payloads are a
    host-device copy's)."""
    exp = [(seq.ops[i], name) for i, name in expected_nodes(seq)]
    idx = _stream_index([op.stream for op, _ in exp])
    return [(activity_class(name), k, op_sequence.payload_class(op.payload_bytes))
            for (op, name), k in zip(exp, idx)]


def node_keys(nodes: Sequence[Node]) -> list[tuple]:
    """`structural_keys` for replayed nodes: a payload class for a host-device
    copy only, as the sequence keeps one for those only."""
    idx = _stream_index([n.stream for n in nodes])
    out = []
    for n, k in zip(nodes, idx):
        cls = activity_class(n.name, n.cat)
        nbytes = n.bytes if cls.startswith("copy:") else 0
        out.append((cls, k, op_sequence.payload_class(nbytes or 0)))
    return out


def _same_activity(name: str) -> str:
    """The name two activities are anchored by: a memset's memory kind is left
    out ("Memset (Device)" eagerly, "Memset (Unknown)" for a graph's memset
    node, which has no allocation to name; the debug session, 2 Nov 2026)."""
    if name.startswith("Memset (") and name.endswith(")"):
        return "Memset"
    return name


def match(seq: OpSequence, nodes: Sequence[Node]) -> Match:
    """Replayed nodes to the sequence's positions, by structure only
    (researcher's ruling 1 of 6 Nov 2026): an activity is matched by its
    class, stream, order and payload class (`structural_keys`), and a name
    or launch variant that differs is not a mismatch.

    Names place the comparison, never decide it. Where the two lists agree
    by name (`difflib`'s matching blocks) each pair must also agree in
    structure; between two such blocks, the stretch of positions and the
    stretch of nodes are paired in order only if they agree in structure
    one for one, as a whole, and are otherwise all unmatched. A kernel stream
    is one class on one stream, so pairing by structure alone across a stretch
    whose counts differ (`nonzero`'s three kernels against the graph's eight
    for `nonzero_static`) would give positions the time of other kernels.
    A stretch with equal counts and structure (a copy's other launch
    variant, a graph's copy kernel for a device copy, or a stand-in of one
    kernel for one) is matched, and its positions marked `variant_matched`.
    A host-device copy is never a device position (T2.5 prices it), so such
    nodes (an eager run's) are left out of the comparison and are extra."""
    return _match(seq, nodes, None)


def match_segmented(seq: OpSequence, nodes: Sequence[Node], segments: Sequence[Mapping]) -> Match:
    """`match` within each segment of a replay run in blocks of layers
    (`isolated_replay_compact`; researcher's ruling 2 of 7 Nov 2026): a
    segment's positions (`position_segments`) are compared only with the
    nodes of the profiler session that ran them, told by their correlations
    (`segments[k]["correlations"]`, as `trace_stream.stitch_compact` offset
    them), so no stretch runs across a cut and a loss is reported in the
    segment that lost it. The layers are one structure repeated, so a match
    of the whole stream reports a layer's worth of nodes lost anywhere at
    the last layers (the exploratory session's oracle cells, 7 Nov 2026).
    `by_segment` holds each segment's layers, positions, nodes and unmatched
    positions. A node no segment's correlations hold is refused."""
    where = position_segments(seq, segments)
    owner = _node_segments(nodes, segments)
    exp = expected_nodes(seq)
    groups = [([i for i, (p, _) in enumerate(exp) if where[p] == k],
               [j for j in range(len(nodes)) if owner[j] == k]) for k in range(len(segments))]
    return _match(seq, nodes, groups, segments)


def position_segments(seq: OpSequence, segments: Sequence[Mapping]) -> dict[int, int]:
    """Each position's segment: its layer's, for a layered position; for one
    outside the layers, the segment of the last layered position before it
    (an operator between two layers runs before the roll at the next), or the
    first segment. Refused unless the segments' layers are the sequence's
    layers in order, each in one segment."""
    layers = seq.layers
    bounds = [tuple(int(x) for x in s["layers"]) for s in segments]
    ok = (bool(bounds) and bool(layers) and bounds[0][0] == layers[0]
          and bounds[-1][1] == layers[-1] and all(a <= b for a, b in bounds)
          and all(bounds[k + 1][0] == bounds[k][1] + 1 for k in range(len(bounds) - 1)))
    if not ok:
        raise ReplayError(f"segments of layers {bounds} are not the sequence's layers "
                          f"{layers[0] if layers else None} to {layers[-1] if layers else None} "
                          f"in order")
    seg_of = {l: k for k, (a, b) in enumerate(bounds) for l in range(a, b + 1)}
    out, cur = {}, 0
    for p, op in enumerate(seq.ops):
        if op.layer is not None:
            cur = seg_of[op.layer]
        out[p] = cur
    return out


def _node_segments(nodes: Sequence[Node], segments: Sequence[Mapping]) -> list[int]:
    ranges = sorted((int(s["correlations"][0]), int(s["correlations"][1]), k)
                    for k, s in enumerate(segments) if s.get("correlations"))
    starts = [a for a, _, _ in ranges]
    out = []
    for n in nodes:
        i = bisect.bisect_right(starts, n.correlation) - 1 if n.correlation is not None else -1
        if i < 0 or n.correlation > ranges[i][1]:
            raise ReplayError(f"node {n.name!r} (correlation {n.correlation}): no segment holds "
                              f"its correlation")
        out.append(ranges[i][2])
    return out


def segments_from_counts(counts: Mapping) -> Optional[list[dict]]:
    """The segments of `isolated_replay_compact`'s counts, for
    `match_segmented`; None for a replay in one session."""
    if "segment_layer_ranges" not in counts:
        return None
    layers, corrs = counts["segment_layer_ranges"], counts.get("segment_correlations")
    if corrs is None or len(corrs) != len(layers) or len(layers) != counts.get("segments", len(layers)):
        raise ReplayError(f"{len(layers)} segments of layers, "
                          f"{None if corrs is None else len(corrs)} of correlations: "
                          f"the segments cannot be told apart")
    return [{"layers": list(l), "correlations": None if c is None else list(c)}
            for l, c in zip(layers, corrs)]


def segment_report(m: Match, segments: Sequence[Mapping], orphans: Sequence[int]) -> list[dict]:
    """`m.by_segment` with each segment's count of launches no device activity
    answered (`launches_without_activity`), by correlation."""
    out = []
    for s, seg in zip(m.by_segment, segments):
        c = seg.get("correlations")
        n = sum(c[0] <= o <= c[1] for o in orphans) if c else 0
        out.append(dict(s, launches_without_activity=n))
    return out


def _match(seq: OpSequence, nodes: Sequence[Node], groups, segments=None) -> Match:
    """`match` over `groups`, (indices into `expected_nodes`, indices into
    `nodes`) compared one group at a time; None is one group of everything."""
    exp = expected_nodes(seq)
    ka, kb_all = structural_keys(seq), node_keys(nodes)
    copy = {k for k, key in enumerate(kb_all) if key[0].startswith("copy:")}
    if groups is None:
        groups = [(list(range(len(exp))), list(range(len(nodes))))]
    got = [False] * len(exp)
    used = [False] * len(nodes)
    device: dict = defaultdict(float)
    varied: set = set()
    n_cand = 0
    for ea, nb in groups:
        nb = [j for j in nb if j not in copy]
        n_cand += len(nb)
        a = [_same_activity(exp[i][1]) for i in ea]
        b = [_same_activity(nodes[j].name) for j in nb]

        def pair(x, y, named):
            i, j = ea[x], nb[y]
            device[exp[i][0]] += nodes[j].duration_s
            got[i] = used[j] = True
            if not named:
                varied.add(exp[i][0])

        ia = ib = 0
        for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
            ga, gb = range(ia, blk.a), range(ib, blk.b)
            if len(ga) == len(gb) and all(ka[ea[x]] == kb_all[nb[y]] for x, y in zip(ga, gb)):
                for x, y in zip(ga, gb):
                    pair(x, y, False)
            for k in range(blk.size):
                if ka[ea[blk.a + k]] == kb_all[nb[blk.b + k]]:
                    pair(blk.a + k, blk.b + k, True)
            ia, ib = blk.a + blk.size, blk.b + blk.size
    missing = sorted({exp[k][0] for k in range(len(exp)) if not got[k]})
    fam: dict = defaultdict(int)
    for p in missing:
        fam[operation_family(seq.ops[p])] += 1
    for p in missing:
        device.pop(p, None)
    extra = [k for k in range(len(nodes)) if not used[k]]
    by_segment = () if segments is None else tuple(
        {"segment": k, "layers": list(segments[k]["layers"]),
         "positions": len({exp[i][0] for i in ea}), "nodes": len([j for j in nb if j not in copy]),
         "unmatched": sorted({exp[i][0] for i in ea if not got[i]})}
        for k, (ea, nb) in enumerate(groups))
    return Match(dict(device), tuple(missing), tuple(nodes[k].name for k in extra),
                 dict(fam), len(exp), n_cand, tuple(sorted(varied - set(missing))),
                 sum(nodes[k].duration_s for k in extra), by_segment)


def device_costs(m: Match) -> dict[int, float]:
    """d_i by sequence position, for `op_sequence.to_operations(device=...)`."""
    if not m.complete:
        raise ReplayError(
            f"the replay does not match the sequence: {len(m.unmatched_positions)} "
            f"positions unmatched ({m.families_unmatched}), {len(m.extra_nodes)} extra "
            f"nodes; no device cost is taken from a different kernel")
    return dict(m.device_s)


# How each position was priced, and what was dropped (researcher's ruling 1 of 6 Nov 2026).
VARIANT_DIFFERS = "variant_differs"          # matched by structure, its name differs
EXTRA_NODE_DROPPED = "extra_node_dropped"    # a node no position owns, priced nowhere
PRICING_NOT_VALIDATED = "pricing not validated"


@dataclass(frozen=True)
class Pricing:
    device_s: dict                 # position -> device seconds, every priced position
    source: dict                   # position -> GRAPH or ISOLATED_ORIGINAL
    substitute_nodes: tuple        # the graph's nodes no position owns, priced nowhere
    label: dict = field(default_factory=dict)   # position -> GRAPH, ISOLATED_ORIGINAL or VARIANT_DIFFERS
    unpriced: tuple = ()           # device positions neither replay matches
    dropped: dict = field(default_factory=dict)  # replay -> {"nodes", "s"}, EXTRA_NODE_DROPPED

    def counts(self) -> dict[str, int]:
        out: dict = defaultdict(int)
        for v in self.source.values():
            out[v] += 1
        return dict(out)

    def label_counts(self) -> dict[str, int]:
        out: dict = defaultdict(int)
        for i, src in self.source.items():
            out[f"{src}:{self.label.get(i, src)}"] += 1
        return dict(out)

    @property
    def priced_s(self) -> float:
        return sum(self.device_s.values())

    @property
    def mismatch_s(self) -> float:
        """Priced device seconds at positions whose structure the graph does
        not hold as the eager sequence has it: those priced from the isolated
        original."""
        return sum(v for i, v in self.device_s.items() if self.source[i] == ISOLATED_ORIGINAL)

    def summary(self, seq: OpSequence) -> dict:
        """What step 6 banks per cell and the aggregate gate reads."""
        iso = [i for i, s in self.source.items() if s == ISOLATED_ORIGINAL]
        by_family: dict = defaultdict(lambda: {"n": 0, "s": 0.0})
        for i in iso:
            f = by_family[operation_family(seq.ops[i])]
            f["n"] += 1
            f["s"] += self.device_s[i]
        return {"priced_s": self.priced_s, "mismatch_s": self.mismatch_s,
                "positions": len(self.source), "labels": self.label_counts(),
                "isolated_substitutions": sum(seq.ops[i].aten_op in SUBSTITUTED_ATEN_OPS
                                              for i in iso),
                "isolated_families": dict(sorted(by_family.items())),
                "unpriced": list(self.unpriced),
                "unpriced_families": sorted({operation_family(seq.ops[i]) for i in self.unpriced}),
                EXTRA_NODE_DROPPED: dict(self.dropped)}


def _dropped(m: Match) -> dict:
    return {"nodes": len(m.extra_nodes), "s": m.extra_s}


def stand_in_positions(seq: OpSequence, graph: Match) -> tuple:
    """Positions the graph matched only to its stand-in's kernel
    (`STAND_IN_ATEN_OPS`): priced from the isolated original instead."""
    return tuple(i for i in graph.variant_matched if seq.ops[i].aten_op in STAND_IN_ATEN_OPS)


def needs_original(seq: OpSequence, graph: Match) -> bool:
    """Whether step 6 must run the isolated replay of the original operators."""
    return bool(graph.unmatched_positions or stand_in_positions(seq, graph))


def price_positions(seq: OpSequence, graph: Match, original: Optional[Match]) -> Pricing:
    """c_dev for every device position (researcher's ruling 1 of 6 Nov 2026):
    from the in-order graph where it matches the position by structure
    (`match`), else from the isolated replay of the original operators
    where that matches it, and each position labelled by how: `GRAPH`,
    `ISOLATED_ORIGINAL`, or `VARIANT_DIFFERS` where the matched activity's
    name differs (its `source` says which replay). Nodes no position owns are
    dropped and counted (`EXTRA_NODE_DROPPED`): the graph's stand-ins, and an
    eager run's host-device copies, which T2.5 prices. A position the graph
    matched only to its stand-in's kernel (`stand_in_positions`) is priced as
    one the graph does not hold. A position neither matches is `unpriced`;
    nothing is refused here: the aggregate gate (`pricing_gate`) decides
    whether the arm is priced at all."""
    stand_in = set(stand_in_positions(seq, graph))
    missing = tuple(sorted(set(graph.unmatched_positions) | stand_in))
    if missing and original is None:
        raise ReplayError(f"{len(missing)} positions the graph does not hold need the "
                          f"isolated replay of the original operators, which was not run")
    held = {i: v for i, v in graph.device_s.items() if i not in stand_in}
    costs, source = dict(held), {i: GRAPH for i in held}
    label = {i: VARIANT_DIFFERS if i in graph.variant_matched else GRAPH for i in held}
    dropped = {GRAPH: _dropped(graph)}
    unpriced = []
    if missing:
        varied = set(original.variant_matched)
        for i in missing:
            if i in original.device_s:
                costs[i], source[i] = original.device_s[i], ISOLATED_ORIGINAL
                label[i] = VARIANT_DIFFERS if i in varied else ISOLATED_ORIGINAL
            else:
                unpriced.append(i)
        dropped[ISOLATED_ORIGINAL] = _dropped(original)
    order = sorted(costs)
    return Pricing({i: costs[i] for i in order}, {i: source[i] for i in order},
                   tuple(graph.extra_nodes), {i: label[i] for i in order}, tuple(unpriced),
                   dropped)


def pricing_gate(cells: Sequence[Mapping], max_share: float) -> dict:
    """The aggregate gate (researcher's ruling 1 of 6 Nov 2026), per arm over
    its cells: the share of priced device time whose structure the graph does
    not hold (`Pricing.mismatch_s` over `priced_s`, each cell's `summary`)
    must be at most `max_share`, a T4.1 lock value (working 5%), and every
    device position must be priced. An arm that fails is not priced: it is
    labelled `PRICING_NOT_VALIDATED` and kept out of the hypothesis claims.
    `cells` are step 6's entries: "arm" and "pricing" (a `summary`). Never
    widened to pass. The nodes step 6 dropped (`EXTRA_NODE_DROPPED`) are
    counted per arm for the step's report and decide nothing (adviser's
    ruling 3, second set of 12 Nov 2026)."""
    if not 0 < max_share < 1:
        raise ValueError("the share is a fraction in (0, 1), fixed at lock")
    arms: dict = {}
    for e in cells:
        a = arms.setdefault(e["arm"], {"cells": 0, "priced_s": 0.0, "mismatch_s": 0.0,
                                       "unpriced": 0, "dropped_nodes": 0, "dropped_s": 0.0})
        p = e["pricing"]
        a["cells"] += 1
        a["priced_s"] += p["priced_s"]
        a["mismatch_s"] += p["mismatch_s"]
        a["unpriced"] += len(p["unpriced"])
        for d in (p.get(EXTRA_NODE_DROPPED) or {}).values():
            a["dropped_nodes"] += int(d["nodes"])
            a["dropped_s"] += float(d["s"])
    out = {}
    for arm, a in sorted(arms.items()):
        share = a["mismatch_s"] / a["priced_s"] if a["priced_s"] > 0 else None
        why = []
        if a["unpriced"]:
            why.append(f"{a['unpriced']} device positions neither replay matches")
        if share is None:
            why.append("no priced device time")
        elif share > max_share:
            why.append(f"mismatch share {share:.4f} above {max_share}")
        out[arm] = {**a, "share": share, "max_share": max_share, "validated": not why,
                    **({"label": PRICING_NOT_VALIDATED, "why": why} if why else {})}
    return out


# --- isolated replay ---------------------------------------------------------------------

class IsolationMode(TorchDispatchMode):
    """Synchronise and flush the L2 cache before every aten operator."""

    def __init__(self, flush: Callable[[], None], sync: Callable[[], None]):
        super().__init__()
        self.flush, self.sync = flush, sync
        self.operators = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operators += 1
        with record_function(FLUSH):
            self.sync()
            self.flush()
            self.sync()
        return func(*args, **(kwargs or {}))


def l2_flush(device, nbytes: Optional[int] = None) -> Callable[[], None]:
    """A write over twice the L2 cache (from the device's properties)."""
    if nbytes is None:
        props = torch.cuda.get_device_properties(device)
        nbytes = 2 * int(getattr(props, "L2_cache_size", 0) or (64 << 20))
    buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
    return lambda: buf.zero_()


def isolated_replay(run_prefill: Callable[[], None], tape: SyncTape, path, *, device,
                    masks: Optional[mask_bank.MaskRecord] = None,
                    flush: Optional[Callable[[], None]] = None,
                    sync: Optional[Callable[[], None]] = None,
                    kernel_module=None, profile=True, substitute: bool = True) -> Path:
    """The prefill once, eagerly, each aten operator alone on a cold cache,
    under the profiler; returns the trace. Read it with `replay_nodes(...,
    exclude=FLUSH)`. With `substitute=False` the tape is not served: the
    original operators run, synchronisations included, which is how the
    substituted positions are priced (`price_positions`)."""
    from contextlib import nullcontext
    from torch.profiler import ProfilerActivity, profile as _profile
    sync = sync or torch.cuda.synchronize          # taken before the tape patches it
    flush = flush or l2_flush(device)
    banked = masks.tensors(device) if masks is not None else None
    rep = TapeReplayer(tape, tape.stage(device)) if substitute else None
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    sync()
    with _profile(activities=acts) as prof:
        with record_function(REPLAY_REGION):
            with rep or nullcontext(), _bank_capture(banked, kernel_module) as cap, \
                    IsolationMode(flush, sync):
                run_prefill()
            if rep is not None:
                rep.finish()
            _check_calls(cap, banked)
        sync()
    prof.export_chrome_trace(str(path))
    return Path(path)


class _Rolled:
    """The isolated replay's profiler, rolled at layer boundaries (researcher's
    ruling, 2 Nov 2026): one profiler session, with its own replay region,
    per block of `per` layers, each exported, compacted and deleted before
    the next opens, so the profiler holds a block's events and not the
    prefill's (the oracle's trace at 32768 tokens is 3.1 GB at density 0.25,
    and the profiler's host memory about five times that).

    Each roll records the layer it was made before (`starts`), and
    `check_cuts` refuses a replay whose rolls were not made once each before
    layers `per`, `2 per`, ... in order (researcher's ruling 2 of 7 Nov 2026:
    a cut only at a layer boundary), so each segment is the whole layers
    `layer_ranges` names.

    A roll is made before layer `j` for j a multiple of `per`, from the
    layer's forward pre-hook: in Python, between operators, with no aten
    call open, so no operator's event is cut. The dispatch mode is paused
    for it (`_disable_current_modes`), as the profiler's own calls are not
    the prefill's: left on, the mode puts its own events into the trace."""

    def __init__(self, path, acts, sync, layers, per, *, cats=None, fields=None):
        from torch.profiler import profile as _profile
        from . import trace_stream
        self._profile, self.acts, self.sync = _profile, acts, sync
        self.cats = trace_stream.ISOLATED_CATS if cats is None else cats
        self.fields = trace_stream.FIELDS if fields is None else fields
        self.path, self.layers, self.per = Path(path), list(layers), int(per)
        self.parts, self.exported, self.hooks, self.starts = [], [], [], []
        self.prof = self.region = None

    def open(self):
        self.prof = self._profile(activities=self.acts)
        self.prof.__enter__()
        self.region = record_function(REPLAY_REGION)
        self.region.__enter__()

    def close(self):
        import gc
        from . import trace_stream
        self.region.__exit__(None, None, None)
        self.sync()
        self.prof.__exit__(None, None, None)
        part = self.path.with_name(f"{self.path.stem}.seg{len(self.parts):03d}.json")
        self.prof.export_chrome_trace(str(part))
        self.prof = self.region = None
        gc.collect()
        info = trace_stream.write_compact(part, cats=self.cats, fields=self.fields)
        part.unlink()
        self.exported.append(info)
        self.parts.append(Path(info["path"]))

    def roll(self, layer: int):
        from torch.utils._python_dispatch import _disable_current_modes
        self.starts.append(layer)
        with _disable_current_modes():
            self.close()
            self.open()

    def attach(self):
        for j, layer in enumerate(self.layers):
            if j and j % self.per == 0:
                self.hooks.append(layer.register_forward_pre_hook(lambda m, a, j=j: self.roll(j)))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def layer_ranges(self) -> list[list[int]]:
        n = len(self.layers)
        return [[lo, min(lo + self.per, n) - 1] for lo in range(0, n, self.per)]

    def check_cuts(self):
        want = list(range(self.per, len(self.layers), self.per))
        if self.starts != want:
            raise ReplayError(f"the replay rolled before layers {self.starts}, not {want}: "
                              f"a cut not at its layer boundary, or a layer run other than "
                              f"once in order")


def isolated_replay_compact(run_prefill: Callable[[], None], tape: SyncTape, path, *, device,
                            layers, segment_layers: int,
                            masks: Optional[mask_bank.MaskRecord] = None,
                            flush: Optional[Callable[[], None]] = None,
                            sync: Optional[Callable[[], None]] = None,
                            kernel_module=None, device_nodes: bool = False) -> tuple:
    """`isolated_replay(..., substitute=False)` with the profiler rolled every
    `segment_layers` layers (`_Rolled`), the segments' compact streams
    stitched (`trace_stream.stitch_compact`) into one at
    `compact_path(path)`; returns (that stream, its counts). The operators,
    flushes and synchronisations are the unsegmented replay's, and each
    operator is priced alone as before, so the stream is the whole replay's
    events in order. Refused with fewer than two blocks of layers, where
    the whole replay is the segment. With `device_nodes` (step 6's isolated
    original, 6 Nov 2026 ruling 3) the stream keeps what `replay_nodes`
    reads (`trace_stream.REPLAY_CATS`, `DEVICE_FIELDS`) in place of what
    `host_profile.read_isolated` reads. The counts name each segment's
    layers and correlations (`segment_layer_ranges`,
    `segment_correlations`), which `segments_from_counts` gives
    `match_segmented`; a roll not at its layer boundary is refused
    (`_Rolled.check_cuts`)."""
    from torch.profiler import ProfilerActivity
    from . import trace_stream
    layers = list(layers)
    if not segment_layers >= 1 or len(layers) <= segment_layers:
        raise ReplayError(f"{len(layers)} layers in blocks of {segment_layers}: not segmented")
    sync = sync or torch.cuda.synchronize
    flush = flush or l2_flush(device)
    banked = masks.tensors(device) if masks is not None else None
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    mode = IsolationMode(flush, sync)
    roller = _Rolled(path, acts, sync, layers, segment_layers,
                     **({"cats": trace_stream.REPLAY_CATS, "fields": trace_stream.DEVICE_FIELDS}
                        if device_nodes else {}))
    sync()
    try:
        roller.open()
        roller.attach()
        with _bank_capture(banked, kernel_module) as cap, mode:
            run_prefill()
        _check_calls(cap, banked)
        roller.check_cuts()
        roller.close()
    except BaseException:
        if roller.prof is not None:      # a failed replay leaves no profiler running
            roller.region.__exit__(None, None, None)
            roller.prof.__exit__(None, None, None)
        for p in roller.parts:
            p.unlink(missing_ok=True)
        raise
    finally:
        roller.detach()
    dst = trace_stream.compact_path(Path(path))
    info = trace_stream.stitch_compact(roller.parts, dst, region=REPLAY_REGION)
    for p in roller.parts:
        p.unlink()
    return dst, {"stream": dst.name, "segments": len(roller.parts), "segment_layers": segment_layers,
                 "segment_layer_ranges": roller.layer_ranges(),
                 "segment_correlations": info["correlations"],
                 "events": sum(i["events"] for i in roller.exported),
                 "kept": info["kept"],
                 "source_bytes": sum(i["source_bytes"] for i in roller.exported),
                 "bytes": info["bytes"]}


# --- reports -------------------------------------------------------------------------------

def by_family(seq: OpSequence, per_position: Mapping[int, float]) -> dict[str, dict]:
    out: dict = defaultdict(lambda: {"n": 0, "s": 0.0})
    for i, v in per_position.items():
        f = operation_family(seq.ops[i])
        out[f]["n"] += 1
        out[f]["s"] += v
    return dict(out)


def cache_effect(seq: OpSequence, in_order: Mapping[int, float],
                 isolated: Mapping[int, float]) -> dict:
    """Isolated minus in-order device time, per position, family and total."""
    common = sorted(set(in_order) & set(isolated))
    fam_in, fam_iso = by_family(seq, {i: in_order[i] for i in common}), \
        by_family(seq, {i: isolated[i] for i in common})
    tot_in = sum(in_order[i] for i in common)
    tot_iso = sum(isolated[i] for i in common)
    return {
        "positions": len(common),
        "missing_in_order": sorted(set(isolated) - set(in_order)),
        "missing_isolated": sorted(set(in_order) - set(isolated)),
        "in_order_s": tot_in, "isolated_s": tot_iso,
        "effect_s": tot_iso - tot_in,
        "isolated_over_in_order": tot_iso / tot_in if tot_in > 0 else None,
        "families": {f: {"n": fam_in[f]["n"], "in_order_s": fam_in[f]["s"],
                         "isolated_s": fam_iso[f]["s"],
                         "effect_s": fam_iso[f]["s"] - fam_in[f]["s"]} for f in fam_in},
    }


def replay_vs_trace(seq: OpSequence, trace_s: Mapping[int, float],
                    replay_s: Mapping[int, float], m: Optional[Match] = None) -> dict:
    """Per operation family: positions, trace device time, replayed device
    time, and their relative difference (replay - trace) / trace."""
    common = sorted(set(trace_s) & set(replay_s))
    ft = by_family(seq, {i: trace_s[i] for i in common})
    fr = by_family(seq, {i: replay_s[i] for i in common})
    fams = {f: {"n": ft[f]["n"], "trace_s": ft[f]["s"], "replay_s": fr[f]["s"],
                "relative_difference": ((fr[f]["s"] - ft[f]["s"]) / ft[f]["s"]
                                        if ft[f]["s"] > 0 else None)} for f in ft}
    return {"families": fams,
            "positions_compared": len(common),
            "positions_without_replay": sorted(set(trace_s) - set(replay_s)),
            "kernel_name_mismatch": None if m is None else {
                "unmatched_positions": len(m.unmatched_positions),
                "families": m.families_unmatched, "extra_nodes": len(m.extra_nodes)}}


def check_passes(report: Mapping, bound: float) -> tuple[bool, list[str]]:
    """Gate A's family check at an explicit bound (a lock value). Fails on
    a report whose structure evidence was not fed (`kernel_name_mismatch`
    None: no `Match` behind it), a position with no replay, a family whose
    relative difference is missing or not finite, or nothing compared
    (adviser's ruling 2d of 11 Nov 2026; T4.3 read A, finding 7b-c)."""
    if not bound > 0:
        raise ValueError("the bound is a positive relative difference, fixed at lock")
    why = []
    km = report.get("kernel_name_mismatch")
    if km is None:
        why.append("no structure evidence: the replay's match was not fed to the report")
    elif km["unmatched_positions"] or km["extra_nodes"]:
        why.append(f"the replay does not match the sequence by structure: {km}")
    if report["positions_without_replay"]:
        why.append(f"{len(report['positions_without_replay'])} positions have no replay")
    if not report["families"]:
        why.append("no operation family was compared")
    for f, r in report["families"].items():
        d = r["relative_difference"]
        if d is None or not math.isfinite(d) or abs(d) > bound:
            why.append(f"{f}: relative difference {d}")
    return not why, why


def to_record(*, seq: OpSequence, prompt_id: str, m: Match, timing: Optional[dict],
              agreement: Optional[dict], tape: SyncTape,
              masks: Optional[mask_bank.MaskRecord],
              isolated: Optional[Mapping[int, float]] = None,
              pricing: Optional[Pricing] = None) -> dict:
    """The banked device components of one arm, prompt and cell."""
    cp = seq.code_path
    node_sum = sum(m.device_s.values())
    return {
        "schema": SCHEMA, "code_path": mask_bank._code_path_dict(cp),
        "context_length": seq.context_length, "batch": seq.batch, "prompt_id": prompt_id,
        "sequence_sha256": op_sequence.record_sha256(op_sequence.to_record(seq)),
        "masks_sha256": masks.sha256 if masks else None,
        "tape_entries": len(tape), "tape_substitutions": tape.substitutions,
        "complete": m.complete, "unmatched_positions": list(m.unmatched_positions),
        "families_unmatched": m.families_unmatched, "extra_nodes": list(m.extra_nodes),
        "device_s": {str(k): v for k, v in sorted(m.device_s.items())},
        "isolated_s": None if isolated is None else {str(k): v for k, v in sorted(isolated.items())},
        "node_sum_s": node_sum, "graph_timing": timing,
        "profiled_over_unprofiled": (node_sum / timing["median_s"]
                                     if timing and timing["median_s"] > 0 else None),
        "mask_agreement": agreement,
        "priced_s": None if pricing is None else {str(k): v for k, v in pricing.device_s.items()},
        "cost_source": None if pricing is None else {str(k): v for k, v in pricing.source.items()},
        "substitute_nodes": None if pricing is None else list(pricing.substitute_nodes),
    }


def device_costs_from_record(record: Mapping) -> dict[int, float]:
    """c_dev by position: the priced costs where the record has them (every
    position with its source), else the graph's, which must be complete."""
    if record.get("schema") not in SCHEMAS_READ:
        raise ReplayError(f"schema {record.get('schema')!r}")
    if record.get("priced_s") is not None:
        if set(record["priced_s"]) != set(record.get("cost_source") or {}):
            raise ReplayError("the priced positions and their cost sources differ")
        return {int(k): float(v) for k, v in record["priced_s"].items()}
    if not record["complete"]:
        raise ReplayError("this replay did not match its sequence; it has no device costs")
    return {int(k): float(v) for k, v in record["device_s"].items()}


def cost_sources_from_record(record: Mapping) -> dict[int, str]:
    """Each priced position's cost source; all GRAPH for a record that has
    only the graph's complete match."""
    if record.get("cost_source") is not None:
        return {int(k): v for k, v in record["cost_source"].items()}
    return {i: GRAPH for i in device_costs_from_record(record)}
