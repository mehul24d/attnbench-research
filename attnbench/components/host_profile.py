"""Host components by a null-kernel pass over the whole operation sequence
(T2.2; plan Sections 5.1, 5.3 step 4 and 5.4).

c_host(i) is "host time to issue operation i (framework dispatch, library
front end, launch floor) plus host-side compute such as CPU mask building.
Measured by adapting TaxBreak's null-kernel protocol, which applies null
kernels to the launch path, to the whole operation sequence (device work
replaced by null kernels), and by running host-side compute directly at
target sizes. The adaptation is not claimed as a contribution."

**The pass.** The arm's own prefill is run once per repetition on the target
with the device work taken out:

  - every host-device interaction is served from the tape (`sync_tape`), as
    in the in-order replay, so the pass waits for nothing and takes the path
    the recorded prefill took;
  - every aten call that would launch device work is replaced, at the
    dispatcher (`NullDeviceMode`), by one null kernel (`torch.cuda._sleep(0)`,
    the kernel T2.1's probe times) and an output allocated at the shape the
    call would give. The real call is not run, so no device work is done, the
    launch queue never fills, and the device cannot make the host wait;
  - calls that launch nothing (views, metadata) run for real, so the host
    time of the arm's Python and of the dispatcher around them is in the
    measurement;
  - the entry points that are not aten calls (the block-sparse kernel, a
    Triton kernel called from Python) are replaced in the same way, one null
    launch each (`NullEntry`). An entry that is not registered runs for real,
    and the check below reports it as a missing launch.

**What is measured, per launching call.** The *gap*: the host time from the
end of the previous call to the start of this one, which is the arm's Python
between the two calls and the dispatcher up to this call, less any
host-compute region between them. And the *launch*: the time of the null
launch itself. The instrument's own work (shape inference, allocating the
output) lies between the call's start and its end and is in neither.

**Which position gets which cost.** A recorded call can launch several
kernels (a sort, a library GEMM with a split-K reduce). Its first operation
carries gap + launch; each further operation of the same call carries one
launch, priced at the pass's median null launch (`launch_floor_s`). The
recorded sequence does not say which operations came from one call, so calls
are matched by *runs*: the maximal stretch of consecutive issued operations
with one aten operator in one layer. A run of r operations met by c launching
calls gives k = r / c operations per call, and r must be a multiple of c. A
run that is not (a call that launched no kernel this time, an entry that was
not nulled, a path that differs from the recording) is a `ProfileMismatch`,
and no cost is given to a different sequence. When the calls of a run launch
different numbers of kernels the division spreads them evenly; the run's
total is right and the split inside it is not, and that is the one thing this
alignment cannot see.

**Host-side compute** (`attnbench::host_compute`, `trace_recording.
host_compute()`) is measured directly in the same pass: the time between the
region's entry and exit, at the size the pass runs at. Its inputs come from
the tape, so the work is the recorded work. A region's duration is the cost of
its marker position, and it is taken out of the gap it falls in, so it is
charged once (`op_sequence.to_operations` adds it to the next operation).

**The embedding lookup** (researcher's ruling, 31 Oct 2026). Shakedown 3's
t22 pass was refused at its first position: the recording names the
kernel's operator, `aten::gather`, outside the layers, while the dispatcher
shows the pass `aten::embedding`, whose own kernel then runs below it
unseen. A position whose operator is in `ISOLATED_OUTSIDE_LAYERS` and that
lies outside the layers is therefore priced from an isolated replay of the
original operators (`isolated_host_s`: the host time of the call that
issued it, run alone after a synchronise, the flush taken out), and its cost
source is flagged `ISOLATED_REPLAY` (`HostProfile.cost_source`); every other
position keeps the null pass's. In the pass that call is replaced with no
launch, and its own time is kept out of the next gap. Any other position the
pass does not meet is refused, as before.

**What this does not cover**, refused rather than approximated: a CUDA-graph
launch in the sequence (a compiled arm's pass has its own null form, T2.21),
and a synchronisation, which has no host cost in the recursion (s_fixed is
T2.5's).

**Known limits of the instrument** (to be checked on the L4, T2.27 step 1):
the pass runs under the tape's `TorchFunctionMode`, which a real run does
not, and the replaced call skips the C++ front end of the kernel it stands
for, so a gap can be a little long and a launch a little short. The pass
reports the median null launch so the second can be compared with T2.1's.

The device operations are behind `NullDevice`, and the clock is a parameter,
so the bookkeeping is tested on CPU with a counter for a clock; `cuda_device`
builds the real one.
"""

from __future__ import annotations

import contextlib
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch.autograd.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from . import op_sequence
from .op_sequence import HOST_COMPUTE, OpSequence, SeqKind
from .sync_tape import SyncTape, TapeReplayer

SCHEMA = "attnbench.host_profile/1"
NULL_PASS, ISOLATED_REPLAY = "null_pass", "isolated_replay"
# Operators outside the layers priced by the isolated replay (module
# docstring): the embedding lookup's gather on the L4.
ISOLATED_OUTSIDE_LAYERS = frozenset({"aten::gather"})


class ProfileMismatch(RuntimeError):
    """The null pass did not follow the recorded sequence; no host cost is
    given to a different one."""


class NullPassError(RuntimeError):
    """The pass could not run: an op it cannot give an output for, or
    a marker it cannot pair."""


# --- the device ----------------------------------------------------------------

@dataclass
class NullDevice:
    """What the pass needs of the device. `launch` issues one null kernel;
    `device_type` names the tensors that count as device work ("cuda")."""

    launch: Callable[[], None]
    device_type: str = "cuda"


def cuda_device(device_type: str = "cuda") -> NullDevice:
    """The real one: `torch.cuda._sleep(0)`, a kernel that spins for zero
    cycles (T2.1's null kernel)."""
    return NullDevice(launch=lambda: torch.cuda._sleep(0), device_type=device_type)


# --- what the pass records -------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    """One launching call as the pass saw it."""

    name: Optional[str]        # the aten operator, None for an entry point
    layer: Optional[int]
    gap_s: float
    launch_s: float


@dataclass
class PassLog:
    issues: list = field(default_factory=list)
    markers: list = field(default_factory=list)       # host-compute durations, in order
    launches: list = field(default_factory=list)      # every null launch, in order
    isolated: list = field(default_factory=list)      # calls priced by the isolated replay


def aten_name(func) -> str:
    """'aten::mm' for aten.mm.default."""
    return func._schema.name


class NullEntry:
    """A non-aten entry point whose device work is replaced: while the pass
    runs, `module.attr` is a function that takes the call's start, issues one
    null launch and returns `make_output(*args, **kwargs)`."""

    def __init__(self, module, attr: str, make_output: Callable):
        self.module, self.attr, self.make_output = module, attr, make_output


def block_sparse_entry(module=None) -> NullEntry:
    """The block-sparse kernel's extension call, `block_sparse_attn_cuda.
    fwd_block`, below the library's Python wrapper: returns the extension's
    four outputs, `out` shaped like q and the rest None (inference reads only
    `out`).

    Not `block_sparse_attn.block_sparse_attn_func`: nulling the wrapper
    skipped its Python (`replace_ones_with_count`,
    `convert_blockmask_row_reverse`: three synchronisations per layer), so
    Shakedown 2's t22 replay used 0 of 84 recorded interactions, and the
    wrapper's host cost, which is the arm's, would have been lost. At the
    pinned commit the wrapper reaches the extension through the custom op
    `flash_attn::_block_sparse_attn_forward` (torch >= 2.4), which the null
    pass's dispatch mode nulls first when the sequence names it; this entry
    is for a build without that op."""
    if module is None:
        import importlib
        module = importlib.import_module("block_sparse_attn_cuda")
    return NullEntry(module, "fwd_block",
                     lambda q, *a, **kw: (torch.empty_like(q), None, None, None))


class NullDeviceMode(TorchDispatchMode):
    """Replace every launching aten call by one null launch (module
    docstring). `launching` is the set of aten operator names the recorded
    sequence launches device work under; a call outside it, or one that touches
    no device tensor, runs for real."""

    def __init__(self, device: NullDevice, launching: Iterable[str], log: PassLog,
                 state: "_State", clock: Callable[[], float],
                 isolated: Iterable[str] = ()):
        super().__init__()
        self.device, self.launching = device, frozenset(launching)
        self.isolated = frozenset(isolated)
        self.log, self.state, self.clock = log, state, clock

    def _touches_device(self, args, kwargs) -> bool:
        leaves = pytree.tree_leaves((args, kwargs))
        return any(isinstance(t, torch.Tensor) and t.device.type == self.device.device_type
                   for t in leaves)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = aten_name(func)
        if name in self.isolated and self.state.layer is None and \
                self._touches_device(args, kwargs):
            t0 = self.clock()      # priced by the isolated replay: no launch, out of the gap
            out = _null_outputs(func, args, kwargs, self.device.device_type)
            d = self.clock() - t0
            self.state.marker_s += d
            self.log.isolated.append(d)
            return out
        if name not in self.launching or not self._touches_device(args, kwargs):
            return func(*args, **kwargs)
        entry = self.clock()      # after the instrument's own classification
        out = _null_outputs(func, args, kwargs, self.device.device_type)
        self.state.issue(self.log, name, entry, self.device, self.clock)
        return out


def _meta_like(t):
    if isinstance(t, torch.Tensor):
        return torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype,
                                   device="meta")
    return t


def _null_outputs(func, args, kwargs, device_type: str):
    """What the call would return, allocated and uninitialised. Shapes come
    from running the call on meta tensors; an output that is one of the
    inputs (an in-place call, `out=`) is that input."""
    real = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
    metas = {id(t): _meta_like(t) for t in real}
    margs, mkwargs = pytree.tree_map(
        lambda t: metas[id(t)] if isinstance(t, torch.Tensor) else t, (args, kwargs))
    try:
        mout = func(*margs, **mkwargs)
    except Exception as e:
        raise NullPassError(
            f"{aten_name(func)} has no output on meta tensors ({e!r}); its output "
            f"shape cannot be given without running it") from e
    by_meta = {id(m): t for t, m in zip(real, (metas[id(t)] for t in real))}
    dev = next((t.device for t in real if t.device.type == device_type), None)

    def alloc(m):
        if not isinstance(m, torch.Tensor):
            return m
        if id(m) in by_meta:
            return by_meta[id(m)]
        return torch.empty_strided(tuple(m.shape), tuple(m.stride()), dtype=m.dtype,
                                   device=dev)
    return pytree.tree_map(alloc, mout)


class _State:
    """The pass's running stamps: when the last launching call ended, how
    much host-compute time has run since, and the current layer."""

    def __init__(self, clock, start: float):
        self.last_exit = start
        self.marker_s = 0.0
        self.layer: Optional[int] = None
        self.clock = clock

    def issue(self, log: PassLog, name, entry: float, device: NullDevice, clock) -> None:
        gap = entry - self.last_exit - self.marker_s
        t0 = clock()
        device.launch()
        launch = clock() - t0
        log.issues.append(Issue(name=name, layer=self.layer, gap_s=max(gap, 0.0),
                                launch_s=launch))
        log.launches.append(launch)
        self.marker_s = 0.0
        self.last_exit = clock()


@contextlib.contextmanager
def _entries(entries: Sequence[NullEntry], device: NullDevice, log: PassLog,
             state: _State, clock):
    saved = []
    try:
        for e in entries:
            original = getattr(e.module, e.attr)
            saved.append((e, original))

            def null(*args, _e=e, **kwargs):
                entry = clock()
                out = _e.make_output(*args, **kwargs)
                state.issue(log, None, entry, device, clock)
                return out
            setattr(e.module, e.attr, null)
        yield
    finally:
        for e, original in saved:
            setattr(e.module, e.attr, original)


@contextlib.contextmanager
def _host_compute_regions(log: PassLog, state: _State, clock):
    """Time each `attnbench::host_compute` region directly (module docstring)
    by wrapping `record_function`'s enter and exit for that name only."""
    enter, exit_ = record_function.__enter__, record_function.__exit__
    open_: list = []

    def __enter__(self):
        r = enter(self)
        if self.name == HOST_COMPUTE:
            if open_:
                raise NullPassError("a host-compute region inside another one")
            open_.append(clock())
        return r

    def __exit__(self, *a):
        if self.name == HOST_COMPUTE:
            if not open_:
                raise NullPassError("a host-compute region closed that was not open")
            d = clock() - open_.pop()
            log.markers.append(d)
            state.marker_s += d
        return exit_(self, *a)

    record_function.__enter__, record_function.__exit__ = __enter__, __exit__
    try:
        yield
        if open_:
            raise NullPassError("a host-compute region never closed")
    finally:
        record_function.__enter__, record_function.__exit__ = enter, exit_


@contextlib.contextmanager
def _layer_hooks(layers: Sequence[torch.nn.Module], state: _State):
    handles = []
    try:
        for i, m in enumerate(layers):
            handles.append(m.register_forward_pre_hook(
                lambda mod, args, _i=i: setattr(state, "layer", _i)))
            handles.append(m.register_forward_hook(
                lambda mod, args, out, _i=i: setattr(state, "layer", None)))
        yield
    finally:
        for h in handles:
            h.remove()


def launching_names(seq: OpSequence) -> frozenset:
    """The aten operators the recorded sequence issues device work under."""
    names = set()
    for op in seq.ops:
        if _is_issue(op):
            _refuse_graph(op)
            if op.aten_op is not None:
                names.add(op.aten_op)
    return frozenset(names)


def _is_issue(op) -> bool:
    return op.is_device or op.kind is SeqKind.COPY


def _refuse_graph(op) -> None:
    if op.kind is SeqKind.GRAPH:
        raise ProfileMismatch(
            "the sequence launches a CUDA graph, whose host cost the null pass "
            "does not measure (a compiled arm has its own form, T2.21)")


def isolated_positions(seq: OpSequence,
                       ops: Iterable[str] = ISOLATED_OUTSIDE_LAYERS) -> tuple[int, ...]:
    """Positions priced by the isolated replay: an issued operation outside
    the layers whose operator is in `ops` (module docstring)."""
    ops = frozenset(ops)
    return tuple(i for i, op in enumerate(seq.ops)
                 if _is_issue(op) and op.layer is None and op.aten_op in ops)


def _isolated_origins(seq: OpSequence, positions: Sequence[int]) -> frozenset:
    out = set()
    for i in positions:
        if seq.ops[i].origin is None:
            raise ProfileMismatch(f"position {i} ({seq.ops[i].aten_op}) names no issuing call; "
                                  f"the pass cannot tell which call to keep out")
        out.add(seq.ops[i].origin)
    return frozenset(out)


def null_pass(run_prefill: Callable[[], None], tape: SyncTape, seq: OpSequence, *,
              device: NullDevice, staged_device=None,
              entries: Sequence[NullEntry] = (),
              layers: Sequence[torch.nn.Module] = (),
              clock: Callable[[], float] = time.perf_counter,
              isolated_ops: Iterable[str] = ISOLATED_OUTSIDE_LAYERS) -> PassLog:
    """One null-kernel pass of the arm's prefill (module docstring)."""
    launching = launching_names(seq)
    origins = _isolated_origins(seq, isolated_positions(seq, isolated_ops))
    staged = tape.stage(staged_device) if staged_device is not None else tape.stage(
        torch.device(device.device_type))
    log = PassLog()
    state = _State(clock, clock())
    mode = NullDeviceMode(device, launching, log, state, clock, isolated=origins)
    rep = TapeReplayer(tape, staged)
    with rep, _entries(entries, device, log, state, clock), \
            _host_compute_regions(log, state, clock), _layer_hooks(layers, state), mode:
        run_prefill()
    rep.finish()
    return log


# --- from a pass to the sequence's positions ----------------------------------------

@dataclass(frozen=True)
class Run:
    name: Optional[str]
    layer: Optional[int]
    positions: tuple[int, ...]


def runs_of(seq: OpSequence, skip: Iterable[int] = ()) -> list[Run]:
    """Maximal stretches of consecutive issued operations with one aten
    operator in one layer. A synchronisation or a host-compute marker ends a
    run (it is a different host event). Positions in `skip` (priced by the
    isolated replay) are passed over."""
    out: list[Run] = []
    cur: Optional[tuple] = None
    pos: list[int] = []
    skip = frozenset(skip)
    for i, op in enumerate(seq.ops):
        if i in skip:
            continue
        if not _is_issue(op):
            if cur is not None:
                out.append(Run(cur[0], cur[1], tuple(pos)))
                cur, pos = None, []
            continue
        _refuse_graph(op)
        key = (op.aten_op, op.layer)
        if key != cur:
            if cur is not None:
                out.append(Run(cur[0], cur[1], tuple(pos)))
            cur, pos = key, []
        pos.append(i)
    if cur is not None:
        out.append(Run(cur[0], cur[1], tuple(pos)))
    return out


def _merge_equal_neighbours(runs: Sequence[Run]) -> list[Run]:
    """Adjacent runs with one key (a synchronisation between two identical
    operators) are met by one stretch of identical calls, and the pass cannot
    say where one ends, so they are divided together."""
    out: list[Run] = []
    for r in runs:
        if out and (out[-1].name, out[-1].layer) == (r.name, r.layer):
            out[-1] = Run(r.name, r.layer, out[-1].positions + r.positions)
        else:
            out.append(r)
    return out


def assign(seq: OpSequence, log: PassLog, *, isolated: Sequence[int] = ()) -> dict[int, float]:
    """Position -> host seconds for one pass: every issued operation but the
    `isolated` ones, and every host-compute marker. Raises `ProfileMismatch`
    where the pass did not meet the sequence exactly (module docstring)."""
    if isolated and not log.isolated:
        raise ProfileMismatch(
            f"the recording issues {len(isolated)} operation(s) priced by the isolated replay "
            f"({seq.ops[isolated[0]].aten_op} outside the layers), and the pass met no call "
            f"to {seq.ops[isolated[0]].origin} there")
    if len(isolated) % max(len(log.isolated), 1) or (log.isolated and not isolated):
        raise ProfileMismatch(
            f"the pass kept out {len(log.isolated)} call(s) for {len(isolated)} position(s) "
            f"priced by the isolated replay")
    runs = _merge_equal_neighbours(runs_of(seq, skip=isolated))
    floor = statistics.median(log.launches) if log.launches else 0.0
    host: dict[int, float] = {}
    issues = log.issues
    j = 0
    for run in runs:
        c = 0
        while j + c < len(issues) and (issues[j + c].name, issues[j + c].layer) == \
                (run.name, run.layer):
            c += 1
        r = len(run.positions)
        where = f"{run.name or 'an entry point'} in layer {run.layer}"
        if c == 0:
            raise ProfileMismatch(
                f"the recording issues {r} operation(s) under {where}, and the pass "
                f"made no launching call there (next seen: "
                f"{issues[j].name if j < len(issues) else 'nothing'} in layer "
                f"{issues[j].layer if j < len(issues) else None}); an entry point "
                f"that was not nulled would have run for real")
        if r % c:
            raise ProfileMismatch(
                f"the recording issues {r} operation(s) under {where} and the pass "
                f"made {c} launching call(s); {r} is not a multiple of {c}")
        k = r // c
        for n in range(c):
            ev = issues[j + n]
            first, *rest = run.positions[n * k:(n + 1) * k]
            host[first] = ev.gap_s + ev.launch_s
            for p in rest:
                host[p] = floor
        j += c
    if j != len(issues):
        raise ProfileMismatch(
            f"the pass made {len(issues) - j} launching call(s) after the "
            f"recorded sequence ended (first: {issues[j].name} in layer "
            f"{issues[j].layer})")
    marks = [i for i, op in enumerate(seq.ops) if op.kind is SeqKind.HOST_COMPUTE]
    if len(marks) != len(log.markers):
        raise ProfileMismatch(
            f"the recording has {len(marks)} host-compute marker(s) and the pass "
            f"timed {len(log.markers)}")
    for i, d in zip(marks, log.markers):
        host[i] = d
    return host


# --- the profile -----------------------------------------------------------------------

@dataclass(frozen=True)
class HostProfile:
    """Host seconds per position over `reps` passes."""

    host_s: Mapping[int, float]                     # the median over repetitions
    reps_s: Mapping[int, tuple[float, ...]]         # every repetition
    launch_floor_s: float                           # median null launch, all passes
    reps: int
    sequence_sha256: str
    cost_source: Mapping[int, str] = field(default_factory=dict)   # absent: NULL_PASS

    def source(self, i: int) -> str:
        return self.cost_source.get(i, NULL_PASS)

    def sources(self) -> dict[str, int]:
        out = {NULL_PASS: 0, ISOLATED_REPLAY: 0}
        for i in self.host_s:
            out[self.source(i)] += 1
        return out

    @property
    def total_s(self) -> float:
        return sum(self.host_s.values())


def profile(run_prefill: Callable[[], None], tape: SyncTape, seq: OpSequence, *,
            device: NullDevice, reps: int, warmup: int = 1,
            isolated: Optional[Mapping[int, float]] = None,
            isolated_ops: Iterable[str] = ISOLATED_OUTSIDE_LAYERS, **kw) -> HostProfile:
    """`warmup` untimed passes, then `reps` passes; per-position median.
    `isolated` holds the isolated replay's host seconds for exactly the
    positions `isolated_positions` names (`isolated_host_s`)."""
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be an int >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    iso = isolated_positions(seq, isolated_ops)
    if set(iso) != set(isolated or {}):
        raise ProfileMismatch(
            f"positions {list(iso)} are priced by the isolated replay; the isolated costs "
            f"given cover {sorted(isolated or {})}")
    kw = dict(kw, isolated_ops=isolated_ops)
    for _ in range(warmup):
        assign(seq, null_pass(run_prefill, tape, seq, device=device, **kw), isolated=iso)
    per, launches = [], []
    for _ in range(reps):
        log = null_pass(run_prefill, tape, seq, device=device, **kw)
        per.append(assign(seq, log, isolated=iso))
        launches.extend(log.launches)
    keys = set(per[0])
    for h in per[1:]:
        if set(h) != keys:
            raise ProfileMismatch("two passes gave costs for different positions")
    reps_s = {i: tuple(h[i] for h in per) for i in sorted(keys)}
    host_s = {i: float(statistics.median(v)) for i, v in reps_s.items()}
    host_s.update({i: float(isolated[i]) for i in iso})
    return HostProfile(
        host_s=dict(sorted(host_s.items())),
        reps_s=reps_s, launch_floor_s=float(statistics.median(launches)) if launches else 0.0,
        reps=reps,
        sequence_sha256=op_sequence.record_sha256(op_sequence.to_record(seq)),
        cost_source={i: ISOLATED_REPLAY for i in iso})


def isolated_host_s(trace, seq: OpSequence, positions: Sequence[int]) -> dict[int, float]:
    """Host seconds for `positions` from an isolated replay of the original
    operators (`replay.isolated_replay(..., substitute=False)`): the kernel
    launches in its region, flushes left out, are the sequence's kernel
    positions in order; a position's cost is the duration of the innermost
    call to its issuing operator (`origin`) around its launch, less any
    flush inside it. Refused where the replay's launches are not the
    recording's."""
    from .replay import FLUSH, REPLAY_REGION
    events = op_sequence._events(op_sequence.load_trace(trace))
    regions = [e for e in events if e.cat == "user_annotation" and e.name == REPLAY_REGION]
    if len(regions) != 1:
        raise ProfileMismatch(f"{len(regions)} replay regions in the isolated trace, expected one")
    reg = regions[0]
    thread = (reg.pid, reg.tid)
    on_thread = [e for e in events if (e.pid, e.tid) == thread]
    flushes = [e for e in on_thread if e.cat == "user_annotation" and e.name == FLUSH]
    inside = lambda e, c: c.ts - op_sequence._EPS <= e.ts and e.end <= c.end + op_sequence._EPS
    launches = sorted((e for e in on_thread if e.cat in op_sequence._HOST_CATS
                       and op_sequence.api_class(e.name) == "launch" and inside(e, reg)
                       and not any(inside(e, f) for f in flushes)), key=lambda e: e.ts)
    kernels = [i for i, op in enumerate(seq.ops) if op.kind is SeqKind.KERNEL]
    if len(launches) != len(kernels):
        raise ProfileMismatch(f"the isolated replay launched {len(launches)} kernels and the "
                              f"recording has {len(kernels)}; not the recorded path")
    calls = op_sequence._containers([e for e in on_thread if e.cat == "cpu_op"], launches)
    out = {}
    for i in positions:
        op = seq.ops[i]
        if op.kind is not SeqKind.KERNEL:
            raise ProfileMismatch(f"position {i} is a {op.kind.value}, not a kernel")
        ev = launches[kernels.index(i)]
        chain = calls.get(ev.idx, [])
        if not chain or chain[-1].name != op.aten_op:
            raise ProfileMismatch(
                f"position {i}: the isolated replay's launch is under "
                f"{chain[-1].name if chain else 'no operator'}, the recording's under {op.aten_op}")
        issuing = [c for c in chain if c.name == op.origin]
        if not issuing:
            raise ProfileMismatch(f"position {i}: no call to {op.origin} around its launch")
        c = issuing[-1]
        flushed = sum(f.end - f.ts for f in flushes if inside(f, c))
        out[i] = max(c.end - c.ts - flushed, 0.0) * 1e-6
    return out


def host_costs(p: HostProfile, seq: OpSequence) -> dict[int, float]:
    """The `host` argument of `op_sequence.to_operations`: refuses a profile
    of another sequence."""
    if p.sequence_sha256 != op_sequence.record_sha256(op_sequence.to_record(seq)):
        raise ProfileMismatch("this profile was taken against another sequence")
    return dict(p.host_s)


def to_record(p: HostProfile) -> dict:
    return {"schema": SCHEMA, "sequence_sha256": p.sequence_sha256, "reps": p.reps,
            "launch_floor_s": p.launch_floor_s,
            "host_s": {str(i): v for i, v in sorted(p.host_s.items())},
            "reps_s": {str(i): list(v) for i, v in sorted(p.reps_s.items())},
            "cost_source": {str(i): v for i, v in sorted(p.cost_source.items())}}


def from_record(rec: Mapping) -> HostProfile:
    if rec.get("schema") != SCHEMA:
        raise ProfileMismatch(f"schema {rec.get('schema')!r}, expected {SCHEMA!r}")
    return HostProfile(
        host_s={int(i): float(v) for i, v in rec["host_s"].items()},
        reps_s={int(i): tuple(float(x) for x in v) for i, v in rec["reps_s"].items()},
        launch_floor_s=float(rec["launch_floor_s"]), reps=int(rec["reps"]),
        sequence_sha256=rec["sequence_sha256"],
        cost_source={int(i): str(v) for i, v in (rec.get("cost_source") or {}).items()})


def dumps(p: HostProfile) -> str:
    return json.dumps(to_record(p), indent=1, sort_keys=True)
