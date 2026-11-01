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
kernels (a sort, a library GEMM with a cuBLAS memset). Its first operation
carries gap + launch; each further operation of the same call carries one
launch, priced at the pass's median null launch (`launch_floor_s`). Which
operations came from one call is recorded (researcher's ruling, 1 Nov 2026):
each operation keeps the call instance of every operator around its launch
(`SeqOp.calls`; the launch's correlation ties it to its kernel, the calls
around it on the host thread are its instance; an entry point's is its
Python call), and the operations the pass
met at one call instance are that call's. The pass's launching calls are
matched in order to those calls, one to one, by operator and layer. A call
the pass did not make, or made where the recording has none (an entry that
was not nulled, a path that differs from the recording), is a
`ProfileMismatch`, and no cost is given to a different sequence. A position
with no recorded call instance is refused: positions are never divided
evenly among calls (Shakedown 4's t22 pass: three `addmm` calls launched 2,
2 and 1 kernels).

**Empty calls** (researcher's ruling, 1 Nov 2026). A call whose outputs, or
whose in-place target, are all empty (numel 0) launches nothing on a CUDA
build (Shakedown 4: `fill_` on an empty slice, `Xattention.py:221`). The pass
makes no launch for it and gives it no position; its dispatch up to the call
stays in the next launching call's gap, so its host cost is priced there with
no device cost, and the profile flags that position (`HostProfile.
empty_calls`).

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
launch, and its own time is kept out of the next gap.

**Composite and factory calls** (researcher's ruling, 1 Nov 2026). The
dispatcher shows the pass one call where the recording names the kernels
below it: `aten::_to_copy` where the recording has `aten::copy_`, `aten::mean`
where it has `aten::fill_` and `aten::copy_`. The recording keeps every aten
operator around each launch (`SeqOp.chain`, outermost first), so the pass
nulls any call whose name is in a recorded chain, and a position is met at
the outermost operator of its chain the pass issued (`met_at`). The call's
gap and launch go to the first of its positions and the launch floor to each
further one, as for a call that launches several kernels; such a position's
cost source is `CHAIN`. A call that names the device only by its `device`
argument (a factory such as `aten::arange`; a copy to the device) is a launch
when that argument is the device and host-only otherwise; its positions'
source is `DEVICE_ARGUMENT`. Under any dispatch mode PyTorch's own
composites take the out-of-place form of an in-place call
(`areAnyTensorSubclassLike`: SDPA's math path adds its mask with `aten::add`,
not the recorded `aten::add_`), so the pass's mode changes the call it meets;
a recorded in-place operator met at its out-of-place form is priced as met
there and flagged `OUT_OF_PLACE`. A composite whose chain is not recorded (a record
from before 1 Nov 2026: `aten_op` and `origin` differ and `chain` is empty)
is priced by the isolated replay at its `origin`, like the embedding, and is
refused only where neither route reaches it. `chain_check` holds the
composites' costs against their host time measured directly in the isolated
replay, in aggregate within a T4.1 lock value (ruling, 1 Nov 2026). Any
other position the pass does not meet is refused, as before.

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
CHAIN, DEVICE_ARGUMENT, OUT_OF_PLACE = "chain", "device_argument", "out_of_place"
SOURCES = (NULL_PASS, CHAIN, DEVICE_ARGUMENT, OUT_OF_PLACE, ISOLATED_REPLAY)
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
    device_argument: bool = False   # met by its `device` argument alone


@dataclass
class PassLog:
    issues: list = field(default_factory=list)
    markers: list = field(default_factory=list)       # host-compute durations, in order
    empty: list = field(default_factory=list)         # (operator, layer, issues before it)
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


def _named_device(func, kwargs, tensors) -> Optional[torch.device]:
    """The device a call's `device` argument names; for a factory call with
    none, the default device. None for an operator with no `device` argument
    (`aten::promote_types` takes no tensor and makes none)."""
    if not any(a.name == "device" for a in func._schema.arguments):
        return None
    dev = kwargs.get("device")
    if dev is None and not tensors:
        return torch.get_default_device()
    return None if dev is None else torch.device(dev)


class NullDeviceMode(TorchDispatchMode):
    """Replace every launching aten call by one null launch (module
    docstring). `launching` is the set of aten operator names in the recorded
    sequence's chains; a call outside it, or one that touches no device, runs
    for real. `isolated` holds the (operator, layer) pairs priced by the
    isolated replay."""

    def __init__(self, device: NullDevice, launching: Iterable[str], log: PassLog,
                 state: "_State", clock: Callable[[], float],
                 isolated: Iterable[tuple] = ()):
        super().__init__()
        self.device, self.launching = device, frozenset(launching)
        self.isolated = frozenset(isolated)
        self.log, self.state, self.clock = log, state, clock

    def _device_call(self, func, args, kwargs) -> Optional[bool]:
        """None if the call touches no device; else whether only its `device`
        argument names the device (module docstring)."""
        tensors = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
        if any(t.device.type == self.device.device_type for t in tensors):
            return False
        dev = _named_device(func, kwargs, tensors)
        return True if dev is not None and dev.type == self.device.device_type else None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = aten_name(func)
        how = self._device_call(func, args, kwargs)
        if (name, self.state.layer) in self.isolated and how is not None:
            t0 = self.clock()      # priced by the isolated replay: no launch, out of the gap
            out = _null_outputs(func, args, kwargs, self.device.device_type)
            d = self.clock() - t0
            self.state.marker_s += d
            self.log.isolated.append(d)
            return out
        if name not in self.launching or how is None:
            return func(*args, **kwargs)
        entry = self.clock()      # after the instrument's own classification
        out = _null_outputs(func, args, kwargs, self.device.device_type)
        if all_empty(out):
            # No launch, no position; its dispatch stays in the next gap and
            # the instrument's own time is taken out (module docstring).
            self.state.marker_s += self.clock() - entry
            self.log.empty.append((name, self.state.layer, len(self.log.issues)))
            return out
        self.state.issue(self.log, name, entry, self.device, self.clock, device_argument=how)
        return out


def all_empty(out) -> bool:
    """Whether a call's outputs (for an in-place call, its target) are all
    empty tensors: at least one tensor, and none with an element."""
    ts = [t for t in pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
    return bool(ts) and all(t.numel() == 0 for t in ts)


def _meta_like(t):
    if isinstance(t, torch.Tensor):
        return torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype,
                                   device="meta")
    return t


def _null_outputs(func, args, kwargs, device_type: str):
    """What the call would return, allocated and uninitialised. Shapes come
    from running the call on meta tensors; an output that is one of the
    inputs (an in-place call, `out=`) is that input. A new output goes to the
    device the call's `device` argument names, else to its device input's."""
    real = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
    metas = {id(t): _meta_like(t) for t in real}
    margs, mkwargs = pytree.tree_map(
        lambda t: metas[id(t)] if isinstance(t, torch.Tensor) else t, (args, kwargs))
    named = _named_device(func, kwargs, real)
    if named is not None:
        mkwargs = {**mkwargs, "device": torch.device("meta")}
        mkwargs.pop("pin_memory", None)
    try:
        mout = func(*margs, **mkwargs)
    except Exception as e:
        raise NullPassError(
            f"{aten_name(func)} has no output on meta tensors ({e!r}); its output "
            f"shape cannot be given without running it") from e
    by_meta = {id(m): t for t, m in zip(real, (metas[id(t)] for t in real))}
    dev = named if named is not None else \
        next((t.device for t in real if t.device.type == device_type), None)

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

    def issue(self, log: PassLog, name, entry: float, device: NullDevice, clock, *,
              device_argument: bool = False) -> None:
        gap = entry - self.last_exit - self.marker_s
        t0 = clock()
        device.launch()
        launch = clock() - t0
        log.issues.append(Issue(name=name, layer=self.layer, gap_s=max(gap, 0.0),
                                launch_s=launch, device_argument=device_argument))
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


def chain_of(op) -> tuple:
    """The operators the pass may meet `op` at, outermost first: its recorded
    chain, or for a record without one its `aten_op`."""
    return op.chain or (op.aten_op,)


def launching_names(seq: OpSequence, skip: Iterable[int] = ()) -> frozenset:
    """Every aten operator in the chains of the recorded sequence's issued
    operations, and the out-of-place form of each in-place one, but those at
    `skip` (priced by the isolated replay)."""
    names = set()
    skip = frozenset(skip)
    for i, op in enumerate(seq.ops):
        if _is_issue(op):
            _refuse_graph(op)
            if i not in skip:
                names.update(n for n in chain_of(op) if n is not None)
                names.update(filter(None, map(out_of_place, chain_of(op))))
    return frozenset(names)


def _is_issue(op) -> bool:
    return op.is_device or op.kind is SeqKind.COPY


def _refuse_graph(op) -> None:
    if op.kind is SeqKind.GRAPH:
        raise ProfileMismatch(
            "the sequence launches a CUDA graph, whose host cost the null pass "
            "does not measure (a compiled arm has its own form, T2.21)")


def unrecorded_composite(op) -> bool:
    """A composite whose chain the record does not keep (module docstring)."""
    return not op.chain and op.aten_op is not None and op.origin is not None \
        and op.origin != op.aten_op


def isolated_positions(seq: OpSequence,
                       ops: Iterable[str] = ISOLATED_OUTSIDE_LAYERS) -> tuple[int, ...]:
    """Positions priced by the isolated replay: an issued operation outside
    the layers whose operator is in `ops`, and a composite whose chain is not
    recorded (module docstring)."""
    ops = frozenset(ops)
    return tuple(i for i, op in enumerate(seq.ops)
                 if _is_issue(op) and ((op.layer is None and op.aten_op in ops)
                                       or unrecorded_composite(op)))


def _isolated_origins(seq: OpSequence, positions: Sequence[int]) -> frozenset:
    """The (issuing operator, layer) pairs the pass keeps out."""
    out = set()
    for i in positions:
        if seq.ops[i].origin is None:
            raise ProfileMismatch(f"position {i} ({seq.ops[i].aten_op}) names no issuing call; "
                                  f"the pass cannot tell which call to keep out")
        out.add((seq.ops[i].origin, seq.ops[i].layer))
    return frozenset(out)


def null_pass(run_prefill: Callable[[], None], tape: SyncTape, seq: OpSequence, *,
              device: NullDevice, staged_device=None,
              entries: Sequence[NullEntry] = (),
              layers: Sequence[torch.nn.Module] = (),
              clock: Callable[[], float] = time.perf_counter,
              isolated_ops: Iterable[str] = ISOLATED_OUTSIDE_LAYERS) -> PassLog:
    """One null-kernel pass of the arm's prefill (module docstring)."""
    iso = isolated_positions(seq, isolated_ops)
    launching = launching_names(seq, skip=iso)
    origins = _isolated_origins(seq, iso)
    staged = tape.stage(staged_device) if staged_device is not None else tape.stage(
        torch.device(device.device_type))
    log = PassLog()
    state = _State(clock, clock())
    mode = NullDeviceMode(device, launching, log, state, clock, isolated=origins)
    rep = TapeReplayer(tape, staged, serve_setitem=False)
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


def out_of_place(name: Optional[str]) -> Optional[str]:
    """'aten::add' for 'aten::add_'; None for an operator that is not in-place."""
    if name and name.endswith("_") and not name.endswith("__"):
        return name[:-1]
    return None


def met_at(seq: OpSequence, issued: Iterable[Optional[str]],
           skip: Iterable[int] = ()) -> dict[int, Optional[str]]:
    """Position -> the operator the pass met it at: the outermost of its chain
    that the pass issued (`issued`, the names of its launching calls), else the
    out-of-place form of the outermost in-place one it issued, else its
    `aten_op` (module docstring)."""
    issued, skip = set(issued), frozenset(skip)

    def one(op):
        chain = chain_of(op)
        return next((n for n in chain if n in issued),
                    next((out_of_place(n) for n in chain if out_of_place(n) in issued),
                         op.aten_op))
    return {i: one(op) for i, op in enumerate(seq.ops) if _is_issue(op) and i not in skip}


def runs_of(seq: OpSequence, skip: Iterable[int] = (),
            met: Optional[Mapping[int, Optional[str]]] = None) -> list[Run]:
    """Maximal stretches of consecutive issued operations met at one aten
    operator (`met`; by default their own) in one layer. A synchronisation or
    a host-compute marker ends a run (it is a different host event).
    Positions in `skip` (priced by the isolated replay) are passed over."""
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
        key = (met[i] if met is not None else op.aten_op, op.layer)
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
    return align(seq, log, isolated=isolated)[0]


def call_instance(op, met: Optional[str]) -> Optional[int]:
    """The recorded call instance of the operator the pass met `op` at: the
    outermost in its chain with that name, else the outermost in-place one
    whose out-of-place form it is (`met_at`); for a launch with no operator
    around it (an entry point), the Python call recorded as its instance.
    None when no instance is recorded."""
    if not op.chain and len(op.calls) == 1 and met is None:
        return op.calls[0]
    if len(op.calls) != len(op.chain) or not op.calls:
        return None
    for n, c in zip(op.chain, op.calls):
        if n == met:
            return c
    for n, c in zip(op.chain, op.calls):
        if out_of_place(n) == met:
            return c
    return None


def recorded_calls(seq: OpSequence, met: Mapping[int, Optional[str]],
                   skip: Iterable[int] = ()) -> list[tuple[Optional[str], Optional[int], tuple]]:
    """(operator met at, layer, positions) per recorded call the pass meets,
    in order of the call's first position: the positions met at one call
    instance (module docstring). Refuses a position with no recorded
    instance."""
    skip = frozenset(skip)
    calls: dict = {}
    for i, op in enumerate(seq.ops):
        if i in skip or not _is_issue(op):
            continue
        _refuse_graph(op)
        c = call_instance(op, met[i])
        if c is None:
            raise ProfileMismatch(
                f"position {i} ({op.aten_op or op.api} in layer {op.layer}) records no call "
                f"instance, so the pass's calls cannot be aligned to it; positions are not "
                f"divided evenly among calls (ruling, 1 Nov 2026). Re-record the sequence")
        calls.setdefault((met[i], c), (met[i], op.layer, []))[2].append(i)
    return [(n, layer, tuple(pos)) for n, layer, pos in calls.values()]


def align(seq: OpSequence, log: PassLog, *, isolated: Sequence[int] = ()) -> tuple[dict, dict, dict]:
    """(host seconds, cost source, operator met at) per position, for one
    pass (`assign`)."""
    if isolated and not log.isolated:
        raise ProfileMismatch(
            f"the recording issues {len(isolated)} operation(s) priced by the isolated replay "
            f"({seq.ops[isolated[0]].aten_op} outside the layers), and the pass met no call "
            f"to {seq.ops[isolated[0]].origin} there")
    if len(isolated) % max(len(log.isolated), 1) or (log.isolated and not isolated):
        raise ProfileMismatch(
            f"the pass kept out {len(log.isolated)} call(s) for {len(isolated)} position(s) "
            f"priced by the isolated replay")
    met = met_at(seq, (e.name for e in log.issues), skip=isolated)
    calls = recorded_calls(seq, met, skip=isolated)
    floor = statistics.median(log.launches) if log.launches else 0.0
    host: dict[int, float] = {}
    source: dict[int, str] = {}
    issues = log.issues
    for n, (name, layer, pos) in enumerate(calls):
        where = f"{name or 'an entry point'} in layer {layer}"
        if n >= len(issues):
            raise ProfileMismatch(
                f"the recording issues {len(pos)} operation(s) from a call to {where} "
                f"(position {pos[0]}), and the pass made no launching call there (it made "
                f"{len(issues)} in all); an entry point that was not nulled would have run "
                f"for real")
        ev = issues[n]
        if (ev.name, ev.layer) != (name, layer):
            raise ProfileMismatch(
                f"the recording's call {n} is to {where} (position {pos[0]}, "
                f"{len(pos)} operation(s)), and the pass's launching call {n} was to "
                f"{ev.name or 'an entry point'} in layer {ev.layer}")
        first, *rest = pos
        host[first] = ev.gap_s + ev.launch_s
        for p in rest:
            host[p] = floor
        for p in pos:
            source[p] = DEVICE_ARGUMENT if ev.device_argument else \
                NULL_PASS if met[p] == seq.ops[p].aten_op else \
                CHAIN if met[p] in chain_of(seq.ops[p]) else OUT_OF_PLACE
    if len(issues) > len(calls):
        extra = issues[len(calls)]
        raise ProfileMismatch(
            f"the pass made {len(issues) - len(calls)} launching call(s) after the "
            f"recorded sequence ended (first: {extra.name} in layer {extra.layer})")
    marks = [i for i, op in enumerate(seq.ops) if op.kind is SeqKind.HOST_COMPUTE]
    if len(marks) != len(log.markers):
        raise ProfileMismatch(
            f"the recording has {len(marks)} host-compute marker(s) and the pass "
            f"timed {len(log.markers)}")
    for i, d in zip(marks, log.markers):
        host[i] = d
        source[i] = NULL_PASS
    return host, source, {p: met[p] for p in source if p in met}


def empty_at(seq: OpSequence, log: PassLog, *, isolated: Sequence[int] = ()) -> tuple:
    """The pass's empty calls (module docstring), each with the position
    whose gap holds its dispatch: the first position of the next launching
    call, or None after the last one."""
    met = met_at(seq, (e.name for e in log.issues), skip=isolated)
    firsts = [pos[0] for _, _, pos in recorded_calls(seq, met, skip=isolated)]
    return tuple({"operator": name, "layer": layer,
                  "position": firsts[n] if n < len(firsts) else None}
                 for name, layer, n in log.empty)


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
    met_at: Mapping[int, str] = field(default_factory=dict)       # every source but NULL_PASS
    empty_calls: tuple = ()       # {operator, layer, position whose gap holds it}

    def source(self, i: int) -> str:
        return self.cost_source.get(i, NULL_PASS)

    def sources(self) -> dict[str, int]:
        out = dict.fromkeys(SOURCES, 0)
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
    per, launches, how = [], [], None
    for _ in range(reps):
        log = null_pass(run_prefill, tape, seq, device=device, **kw)
        h, src, met = align(seq, log, isolated=iso)
        empty = empty_at(seq, log, isolated=iso)
        if how is not None and how != (src, met, empty):
            raise ProfileMismatch("two passes met the sequence at different calls")
        how = (src, met, empty)
        per.append(h)
        launches.extend(log.launches)
    keys = set(per[0])
    for h in per[1:]:
        if set(h) != keys:
            raise ProfileMismatch("two passes gave costs for different positions")
    src, met, empty = how
    priced = {i: s for i, s in src.items() if s != NULL_PASS}
    reps_s = {i: tuple(h[i] for h in per) for i in sorted(keys)}
    host_s = {i: float(statistics.median(v)) for i, v in reps_s.items()}
    host_s.update({i: float(isolated[i]) for i in iso})
    return HostProfile(
        host_s=dict(sorted(host_s.items())),
        reps_s=reps_s, launch_floor_s=float(statistics.median(launches)) if launches else 0.0,
        reps=reps,
        sequence_sha256=op_sequence.record_sha256(op_sequence.to_record(seq)),
        cost_source=dict(sorted({**priced, **{i: ISOLATED_REPLAY for i in iso}}.items())),
        met_at={i: met[i] for i in sorted(priced)}, empty_calls=empty)


@dataclass(frozen=True)
class _Isolated:
    """An isolated replay's trace, read against a sequence: `launch[i]` is the
    launch of kernel position i, `calls[e.idx]` the aten calls around launch
    e (outermost first), `flushes` the isolation flushes."""

    launch: dict
    calls: dict
    flushes: list


def _read_isolated(trace, seq: OpSequence) -> _Isolated:
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
    return _Isolated(dict(zip(kernels, launches)), calls, flushes)


def _flushed(iso: _Isolated, t0: float, t1: float) -> float:
    return sum(f.end - f.ts for f in iso.flushes
               if t0 - op_sequence._EPS <= f.ts and f.end <= t1 + op_sequence._EPS)


def isolated_host_s(trace, seq: OpSequence, positions: Sequence[int]) -> dict[int, float]:
    """Host seconds for `positions` from an isolated replay of the original
    operators (`replay.isolated_replay(..., substitute=False)`): the kernel
    launches in its region, flushes left out, are the sequence's kernel
    positions in order; a position's cost is the duration of the innermost
    call to its issuing operator (`origin`) around its launch, less any
    flush inside it. Refused where the replay's launches are not the
    recording's."""
    iso = _read_isolated(trace, seq)
    out = {}
    for i in positions:
        op = seq.ops[i]
        if op.kind is not SeqKind.KERNEL:
            raise ProfileMismatch(f"position {i} is a {op.kind.value}, not a kernel")
        ev = iso.launch[i]
        chain = iso.calls.get(ev.idx, [])
        if not chain or chain[-1].name != op.aten_op:
            raise ProfileMismatch(
                f"position {i}: the isolated replay's launch is under "
                f"{chain[-1].name if chain else 'no operator'}, the recording's under {op.aten_op}")
        issuing = [c for c in chain if c.name == op.origin]
        if not issuing:
            raise ProfileMismatch(f"position {i}: no call to {op.origin} around its launch")
        c = issuing[-1]
        out[i] = max(c.end - c.ts - _flushed(iso, c.ts, c.end), 0.0) * 1e-6
    return out


def chain_check(trace, seq: OpSequence, p: HostProfile, *, aggregate_tol: float) -> dict:
    """The ruling's validation (1 Nov 2026): for every call the pass met at a
    composite or by its device argument (`CHAIN`, `DEVICE_ARGUMENT`; not
    `OUT_OF_PLACE`, whose call the isolated replay makes in place), the sum
    of the costs it gave the call's positions against the call's host time
    measured directly in the isolated replay of the original operators
    (`replay.isolated_replay(..., substitute=False)`, the trace
    `isolated_host_s` reads).

    Like for like with the pass's gap and launch, the direct time of a call
    runs from the end of the previous issuing call to the end of this one, the
    isolation flushes in between taken out: the arm's Python, the dispatcher
    and the composite's whole body with its launches. A call is checked when
    its positions and the previous call's are kernels with nothing between
    them in the sequence and the previous call was not kept out of the pass
    (`ISOLATED_REPLAY`); the rest are counted as unchecked.

    The gate is in aggregate (researcher's ruling, 1 Nov 2026, after
    Shakedown 4's first GPU figures): the sum over checked calls of |chain -
    direct| must be at most `aggregate_tol` (a T4.1 lock value, working 10%)
    of the profile's whole host time. Each call's ratio chain / direct is
    reported (`calls`), and each operator's sums; neither is a gate."""
    iso = _read_isolated(trace, seq)

    def call_of(i: int):
        op = seq.ops[i]
        name = op.origin if p.source(i) == ISOLATED_REPLAY else p.met_at.get(i, op.aten_op)
        hit = [c for c in iso.calls.get(iso.launch[i].idx, []) if c.name == name]
        if not hit and p.source(i) in (CHAIN, DEVICE_ARGUMENT):
            raise ProfileMismatch(f"position {i}: no call to {name} around its launch in the "
                                  f"isolated replay")
        return hit[0] if hit else None      # the outermost: the one the pass meets

    groups: list = []                       # [call event or None, positions]
    for i, op in enumerate(seq.ops):
        if op.kind is not SeqKind.KERNEL or i not in p.host_s:
            groups.append([None, [i]])
            continue
        c = call_of(i)
        if c is None:
            groups.append([None, [i]])
        elif groups and groups[-1][0] is not None and groups[-1][0].idx == c.idx:
            groups[-1][1].append(i)
        else:
            groups.append([c, [i]])
    per: dict = {}
    each: list = []
    unchecked = 0
    for prev, (c, pos) in zip([[None, []]] + groups, groups):
        src = p.source(pos[0])
        if c is None or src not in (CHAIN, DEVICE_ARGUMENT):
            continue
        if prev[0] is None or p.source(prev[1][0]) == ISOLATED_REPLAY:
            unchecked += 1
            continue
        direct = max(c.end - prev[0].end - _flushed(iso, prev[0].end, c.end), 0.0) * 1e-6
        row = per.setdefault(p.met_at[pos[0]], {"calls": 0, "chain_s": 0.0, "direct_s": 0.0,
                                                "sources": set()})
        chain = sum(p.host_s[j] for j in pos)
        row["calls"] += 1
        row["chain_s"] += chain
        row["direct_s"] += direct
        row["abs_diff_s"] = row.get("abs_diff_s", 0.0) + abs(chain - direct)
        row["sources"].add(src)
        each.append({"position": pos[0], "operator": p.met_at[pos[0]], "chain_s": chain,
                     "direct_s": direct, "ratio": chain / direct if direct > 0 else None})
    for name, row in per.items():
        row["sources"] = sorted(row["sources"])
        row["ratio"] = row["chain_s"] / row["direct_s"] if row["direct_s"] > 0 else None
    diff = sum(r["abs_diff_s"] for r in per.values())
    host = p.total_s
    share = diff / host if host > 0 else None
    return {"aggregate_tol": aggregate_tol,
            "checked_calls": sum(r["calls"] for r in per.values()),
            "unchecked_calls": unchecked, "operators": dict(sorted(per.items())),
            "calls": each, "abs_diff_s": diff, "host_s": host, "share": share,
            # Over no checked call the gate would pass by comparing nothing.
            "passed": bool(each) and share is not None and share <= aggregate_tol}


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
            "cost_source": {str(i): v for i, v in sorted(p.cost_source.items())},
            "met_at": {str(i): v for i, v in sorted(p.met_at.items())},
            "empty_calls": [dict(e) for e in p.empty_calls]}


def from_record(rec: Mapping) -> HostProfile:
    if rec.get("schema") != SCHEMA:
        raise ProfileMismatch(f"schema {rec.get('schema')!r}, expected {SCHEMA!r}")
    return HostProfile(
        host_s={int(i): float(v) for i, v in rec["host_s"].items()},
        reps_s={int(i): tuple(float(x) for x in v) for i, v in rec["reps_s"].items()},
        launch_floor_s=float(rec["launch_floor_s"]), reps=int(rec["reps"]),
        sequence_sha256=rec["sequence_sha256"],
        cost_source={int(i): str(v) for i, v in (rec.get("cost_source") or {}).items()},
        met_at={int(i): str(v) for i, v in (rec.get("met_at") or {}).items()},
        empty_calls=tuple(dict(e) for e in rec.get("empty_calls") or ()))


def dumps(p: HostProfile) -> str:
    return json.dumps(to_record(p), indent=1, sort_keys=True)
