"""Host prices by a bare timer (T2.2; researcher's ruling, 1 Nov 2026, after
Shakedown 5).

The null pass (`host_profile`) measures each launching call under its own
instruments: the tape's `TorchFunctionMode` and the null device's
`TorchDispatchMode`. On Shakedown 4's L4 cells it priced leaf calls at 2.1 to
2.5 times the recorded prefill's own host time per call, and the isolated
replay's direct time was inflated as well. So the prediction's host costs
are taken uninstrumented (Appendix A, 1 Nov 2026): every call the null pass
meets, a leaf or a composite, is timed alone with nothing around it.

**The bare timer** (`time_bare`). The call is re-issued at the inputs it had
in the prefill, at least `MIN_REPS` (1000) times, each time after a
synchronise taken outside the timer, so it meets an empty launch queue and
waits for no earlier work; the price is the median of `perf_counter_ns`
around the call, less the timer's own overhead (the median of an empty pair,
`timer_overhead_ns`), floored at zero. One untimed call goes first. Torch
function handling is off while it runs and no dispatch mode is on the stack.
An input the call writes (an in-place target, an `out=` tensor) is cloned
once, so the prefill's own tensors are never changed by the repetitions.

**The bare pass** (`bare_pass`). The arm's prefill runs once for real under
the same tape as the null pass (`TapeReplayer(serve_setitem=False)`), so it
takes the recorded path, and under `BareMode`, a dispatch mode that meets
calls exactly where `NullDeviceMode` does (the recorded chains' operators,
by the same device rule). Each call it meets runs for real under a second
mode that records the call's direct children that launch, then is priced:
its own bare time (`direct`) and, where it has launching children (a
composite), the sum of their bare times (`chain`). An entry point that is
not an aten call (`NullEntry`) is priced the same way, as a leaf. Prices are
cached by signature (operator, the structure of its arguments, each
tensor's shape, stride, dtype and device, every other argument's value), so
a call repeated across layers, cells or arms is timed once in a process. A
call with an empty output gives no position, as in the null pass, and its
bare time is carried into the next call's.

**From calls to positions** (`bare_profile`). The pass's calls are aligned
to the recorded call instances as the null pass's are (`recorded_calls`),
and refused where they differ. A call's price goes to its first position,
less the bare launch floor (the null kernel, bare-timed) for each further
position, which gets the floor: the call's sum is its price. A leaf is
priced by its direct time (`bare_leaf`); a composite by its chain
(`bare_chain`), which the 1 Nov 2026 ruling makes the route of record. A
composite here is a call every one of whose positions a child launched; a
call that also launches its own kernel beside its children (`addmm` beside
its bias copy; `host_profile` costs such a position `null_pass`) has no
chain that adds up to it, and is priced as a leaf.

**The chain gate, in the zoo** (rulings of 1 Nov 2026). Over the
composites, the summed |chain - direct| must be at most the T4.1 lock value
(row 32, working 10%) of the cell's bare host time. A cell that fails is not
refused: its composites are priced by their direct time
(`bare_direct_fallback`) and the cell carries `FALLBACK_LABEL` (oracle
re-evaluation and fallback labelling, researcher's ruling). A cell with no
composite compares nothing and has no verdict.

**Not bare-timed.** A host-compute region keeps the null pass's direct time
(`host_compute_null_pass`): it is not a call. A position the isolated replay
prices (the embedding, a composite whose chain is not recorded) keeps that
price. Both routes are labelled per position (`HostProfile.price_route`).
The arm's Python between calls, which the null pass's gap held, is in no
bare price; the instrumented/bare ratio, per call, per operator and in
aggregate, is recorded beside the prices (`bare_prices.json`).
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from .host_profile import (ISOLATED_OUTSIDE_LAYERS, ISOLATED_REPLAY, NULL_PASS, HostProfile,
                           NullDevice, NullEntry, ProfileMismatch, _isolated_origins, _layer_hooks,
                           _State, all_empty, aten_name, device_call, isolated_positions,
                           launching_names, met_at, recorded_calls)
from .op_sequence import OpSequence, SeqKind
from .sync_tape import SyncTape, TapeReplayer

SCHEMA = "attnbench.bare_prices/1"
MIN_REPS = 1000
BARE_TIMER = "bare_timer"
LEAF, CHAIN_ROUTE, DIRECT_FALLBACK = "bare_leaf", "bare_chain", "bare_direct_fallback"
HOST_COMPUTE_ROUTE = "host_compute_null_pass"
PRICE_ROUTES = (LEAF, CHAIN_ROUTE, DIRECT_FALLBACK, ISOLATED_REPLAY, HOST_COMPUTE_ROUTE)
FALLBACK_LABEL = ("chain gate failed: composites priced by their own bare time, "
                  "not by their leaves")


class BareTimerError(RuntimeError):
    """A call the bare timer could not re-issue, or fewer repetitions than
    the ruling's minimum."""


# --- the timer -----------------------------------------------------------------------

def timer_overhead_ns(reps: int, clock: Callable[[], int] = time.perf_counter_ns) -> float:
    """The median of `reps` empty timer pairs, in nanoseconds."""
    xs = []
    for _ in range(reps):
        t0 = clock()
        t1 = clock()
        xs.append(t1 - t0)
    return float(statistics.median(xs))


def time_bare(fn: Callable[[], object], *, reps: int, overhead_ns: float,
              sync: Callable[[], None],
              clock: Callable[[], int] = time.perf_counter_ns) -> float:
    """Seconds (module docstring): one untimed call, then `reps` calls, each
    after a synchronise outside the timer; the median less the overhead."""
    sync()
    out = fn()
    del out
    xs = []
    for _ in range(reps):
        sync()
        t0 = clock()
        out = fn()
        t1 = clock()
        del out
        xs.append(t1 - t0)
    sync()
    return max(float(statistics.median(xs)) - overhead_ns, 0.0) * 1e-9


def _leaf_key(x):
    if isinstance(x, torch.Tensor):
        return ("tensor", tuple(x.shape), tuple(x.stride()), str(x.dtype), x.device.type)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
        return str(x)
    return type(x).__name__


def signature(name: Optional[str], args, kwargs) -> tuple:
    """What a cached price is keyed on (module docstring)."""
    leaves, spec = pytree.tree_flatten((args, kwargs))
    return (name, str(spec), tuple(_leaf_key(x) for x in leaves))


def _clone(tree):
    return pytree.tree_map(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, tree)


def reissue(func, args, kwargs) -> Callable[[], object]:
    """The call again, at its inputs, with every input it writes cloned once."""
    schema = getattr(func, "_schema", None)
    if schema is not None:
        written = {i for i, a in enumerate(schema.arguments)
                   if a.alias_info is not None and a.alias_info.is_write}
        names = {schema.arguments[i].name for i in written}
        args = tuple(_clone(a) if i in written else a for i, a in enumerate(args))
        kwargs = {k: _clone(v) if k in names else v for k, v in kwargs.items()}
    return lambda: func(*args, **kwargs)


class Pricer:
    """Bare prices by signature, cached for the life of the process."""

    def __init__(self, *, reps: int, sync: Callable[[], None],
                 clock: Callable[[], int] = time.perf_counter_ns, min_reps: int = MIN_REPS):
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < min_reps:
            raise BareTimerError(f"the bare timer takes at least {min_reps} repetitions "
                                 f"(ruling, 1 Nov 2026); got {reps!r}")
        self.reps, self.sync, self.clock = reps, sync, clock
        self.overhead_ns = timer_overhead_ns(max(reps, MIN_REPS), clock)
        self.prices: dict = {}

    def time(self, key: tuple, make: Callable[[], Callable[[], object]]) -> float:
        if key not in self.prices:
            try:
                with _disable_current_modes(), torch._C.DisableTorchFunction():
                    self.prices[key] = time_bare(make(), reps=self.reps,
                                                 overhead_ns=self.overhead_ns,
                                                 sync=self.sync, clock=self.clock)
            except Exception as e:
                raise BareTimerError(f"{key[0] or 'an entry point'} could not be re-issued "
                                     f"bare: {e!r}") from e
        return self.prices[key]

    def call(self, func, args, kwargs) -> float:
        return self.time(signature(aten_name(func), args, kwargs),
                         lambda: reissue(func, args, kwargs))

    def entry(self, name: str, fn, args, kwargs) -> float:
        return self.time(signature(name, args, kwargs), lambda: (lambda: fn(*args, **kwargs)))

    def floor(self, launch: Callable[[], None]) -> float:
        """The bare launch floor: the null kernel, bare-timed."""
        return self.time(("null launch",), lambda: launch)


# --- the pass --------------------------------------------------------------------------

@dataclass(frozen=True)
class BareCall:
    name: Optional[str]           # the aten operator, None for an entry point
    layer: Optional[int]
    direct_s: float
    chain_s: Optional[float]      # None for a leaf
    children: tuple = ()          # the launching children's operators, in order
    device_argument: bool = False


@dataclass
class BarePass:
    calls: list = field(default_factory=list)
    empty: list = field(default_factory=list)       # (operator, layer)
    floor_s: float = 0.0
    reps: int = 0
    overhead_ns: float = 0.0
    signatures: int = 0


_AFTER_PYTHON = torch._C._dispatch_keyset_full_after(torch._C.DispatchKey.Python)


def _launches(name: str, launching: Optional[frozenset]) -> bool:
    """In the recorded sequence's chains; with no sequence (the zoo's own
    entries), any operator that may launch by the stand-in's rule (not a
    view, an allocation or a composite PyTorch decomposes, `cpu_boundary`),
    or that does its device work through a child (`_to_copy`, `clone`)."""
    if launching is not None:
        return name in launching
    from attnbench.cpu_boundary import PREPARING_CHILDREN, _may_launch
    return _may_launch(name) or name in PREPARING_CHILDREN


def _scalar_for_tensor(func, args, kwargs) -> bool:
    """A Python number where the schema takes a Tensor (`mul.Tensor` by 1.0):
    the boxed redispatch does not wrap it as the call itself does."""
    for i, a in enumerate(func._schema.arguments):
        v = args[i] if i < len(args) else kwargs.get(a.name)
        if isinstance(v, (bool, int, float, complex)) and isinstance(a.type, torch.TensorType):
            return True
    return False


def _below_python(func, args, kwargs):
    """The inputs' dispatch keys below the Python key: redispatching a call
    there runs its own kernel with a mode still on the stack, so the calls it
    makes are seen and it is not. None for a call with no tensor input (a
    factory) or a number in a Tensor argument, either of which runs as a
    leaf."""
    if _scalar_for_tensor(func, args, kwargs):
        return None
    keys = None
    for t in pytree.tree_leaves((args, kwargs)):
        if isinstance(t, torch.Tensor):
            k = torch._C._dispatch_keys(t)
            keys = k if keys is None else keys | k
    return None if keys is None else keys & _AFTER_PYTHON


class _Children(TorchDispatchMode):
    """Records a call's direct children that launch (one level down: inside
    this mode's handler the mode is off, so grandchildren are not seen). A
    call to the parent's own operator is a redispatch to itself, not a
    child."""

    def __init__(self, launching: Optional[frozenset], device_type: str,
                 parent: Optional[str] = None):
        super().__init__()
        self.launching, self.device_type, self.parent = launching, device_type, parent
        self.calls: list = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if _launches(aten_name(func), self.launching) and aten_name(func) != self.parent and \
                device_call(func, args, kwargs, self.device_type) is not None and \
                not all_empty(out):
            self.calls.append((func, args, kwargs))
        return out


class BareMode(TorchDispatchMode):
    """Meets launching calls where `NullDeviceMode` does, runs each for real
    and prices it (module docstring)."""

    def __init__(self, pricer: Pricer, launching: Optional[Iterable[str]], device_type: str,
                 state: _State, out: BarePass, isolated: Iterable[tuple] = ()):
        super().__init__()
        self.pricer = pricer
        self.launching = None if launching is None else frozenset(launching)
        self.device_type, self.state, self.out = device_type, state, out
        self.isolated = frozenset(isolated)
        self.carried = 0.0
        self.inside = 0         # an entry point reached inside a met call is its child

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = aten_name(func)
        how = device_call(func, args, kwargs, self.device_type)
        if how is None or not _launches(name, self.launching) or \
                (name, self.state.layer) in self.isolated:
            return func(*args, **kwargs)
        kids = _Children(self.launching, self.device_type, name)
        keys = _below_python(func, args, kwargs)
        self.inside += 1
        try:
            if keys is None:            # run as a leaf (`_below_python`)
                out = func(*args, **kwargs)
            else:
                with kids:
                    out = func.redispatch(keys, *args, **kwargs)
        finally:
            self.inside -= 1
        direct = self.pricer.call(func, args, kwargs)
        chain = sum(self.pricer.call(f, a, k) for f, a, k in kids.calls) if kids.calls else None
        if all_empty(out):
            self.carried += direct
            self.out.empty.append((name, self.state.layer))
            return out
        self.add(BareCall(name, self.state.layer, direct + self.carried,
                          None if chain is None else chain + self.carried,
                          tuple(aten_name(f) for f, _, _ in kids.calls), bool(how)))
        return out

    def add(self, call: BareCall) -> None:
        self.out.calls.append(call)
        self.carried = 0.0


@contextlib.contextmanager
def _bare_entries(entries: Sequence[NullEntry], mode: BareMode):
    saved = []
    try:
        for e in entries:
            original = getattr(e.module, e.attr)
            saved.append((e, original))

            def bare(*args, _orig=original,
                     _name=f"{getattr(e.module, '__name__', type(e.module).__name__)}.{e.attr}",
                     **kwargs):
                if mode.inside:
                    return _orig(*args, **kwargs)
                with _disable_current_modes():
                    out = _orig(*args, **kwargs)
                price = mode.pricer.entry(_name, _orig, args, kwargs)
                mode.add(BareCall(None, mode.state.layer, price + mode.carried, None))
                return out
            setattr(e.module, e.attr, bare)
        yield
    finally:
        for e, original in saved:
            setattr(e.module, e.attr, original)


def bare_pass(run_prefill: Callable[[], None], tape: SyncTape, seq: OpSequence, *,
              device: NullDevice, pricer: Pricer, entries: Sequence[NullEntry] = (),
              layers: Sequence[torch.nn.Module] = (), staged_device=None,
              isolated_ops: Iterable[str] = ISOLATED_OUTSIDE_LAYERS) -> BarePass:
    """The prefill once, for real, under the tape, every call it meets priced
    bare (module docstring)."""
    iso = isolated_positions(seq, isolated_ops)
    launching = launching_names(seq, skip=iso)
    origins = _isolated_origins(seq, iso)
    staged = tape.stage(staged_device if staged_device is not None
                        else torch.device(device.device_type))
    before = len(pricer.prices)
    out = BarePass(reps=pricer.reps, overhead_ns=pricer.overhead_ns)
    out.floor_s = pricer.floor(device.launch)
    state = _State(time.perf_counter, time.perf_counter())
    mode = BareMode(pricer, launching, device.device_type, state, out, isolated=origins)
    rep = TapeReplayer(tape, staged, serve_setitem=False)
    with rep, _bare_entries(entries, mode), _layer_hooks(layers, state), mode:
        run_prefill()
    rep.finish()
    out.signatures = len(pricer.prices) - before
    return out


def price_closure(fn: Callable[[], object], pricer: Pricer, device_type: str) -> list[BareCall]:
    """Every call `fn` makes that touches the device, priced as the bare pass
    prices a met call (direct, and chain where it has launching children):
    the pricing zoo's entries, which have no recorded sequence."""
    out = BarePass(reps=pricer.reps, overhead_ns=pricer.overhead_ns)
    state = _State(time.perf_counter, time.perf_counter())
    with BareMode(pricer, None, device_type, state, out):
        fn()
    return list(out.calls)


# --- from the pass to the positions -------------------------------------------------------

def _spread(total: float, positions: Sequence[int], floor: float) -> dict[int, float]:
    """A call's price over its positions: the floor to each further one (at
    most an equal share), the rest to the first; the sum is the price."""
    first, *rest = positions
    each = min(floor, total / len(positions)) if rest else 0.0
    out = {p: each for p in rest}
    out[first] = total - each * len(rest)
    return out


def bare_profile(seq: OpSequence, bp: BarePass, null: HostProfile, *,
                 aggregate_tol: float) -> tuple[HostProfile, dict]:
    """(the bare profile the prediction reads, the record written beside it).
    `null` is the same recording's null-pass profile: its isolated and
    host-compute positions are kept, labelled, and every call's instrumented
    price is set beside the bare one."""
    iso = tuple(i for i, s in null.cost_source.items() if s == ISOLATED_REPLAY)
    met = met_at(seq, (c.name for c in bp.calls), skip=iso)
    calls = recorded_calls(seq, met, skip=iso)
    if len(calls) != len(bp.calls):
        raise ProfileMismatch(f"the bare pass made {len(bp.calls)} launching calls and the "
                              f"recording has {len(calls)}")
    for n, ((name, layer, pos), c) in enumerate(zip(calls, bp.calls)):
        if (c.name, c.layer) != (name, layer):
            raise ProfileMismatch(
                f"the recording's call {n} is to {name or 'an entry point'} in layer {layer} "
                f"(position {pos[0]}), and the bare pass's was to "
                f"{c.name or 'an entry point'} in layer {c.layer}")
    if len(bp.empty) != len(null.empty_calls):
        raise ProfileMismatch(f"the bare pass met {len(bp.empty)} empty call(s), the null "
                              f"pass {len(null.empty_calls)}")

    # A composite whose own kernel is among its positions (it launches beside
    # its children: `addmm` beside its bias copy) has no chain that adds up
    # to it, and is priced as a leaf.
    pure = [c.chain_s is not None and all(null.source(p) != NULL_PASS for p in pos)
            for (_, _, pos), c in zip(calls, bp.calls)]
    composites = [c for c, ok in zip(bp.calls, pure) if ok]
    diff = sum(abs(c.chain_s - c.direct_s) for c in composites)
    chain_total = sum(c.chain_s if ok else c.direct_s for c, ok in zip(bp.calls, pure))
    kept = sum(v for i, v in null.host_s.items()
               if i in iso or seq.ops[i].kind is SeqKind.HOST_COMPUTE)
    host_total = chain_total + kept
    share = diff / host_total if host_total > 0 else None
    passed = None if not composites else (share is not None and share <= aggregate_tol)

    host: dict[int, float] = {}
    route: dict[int, str] = {}
    rows, ops = [], {}
    for (name, layer, pos), c, ok in zip(calls, bp.calls, pure):
        if not ok:
            how, total = LEAF, c.direct_s
        elif passed is False:
            how, total = DIRECT_FALLBACK, c.direct_s
        else:
            how, total = CHAIN_ROUTE, c.chain_s
        host.update(_spread(total, pos, bp.floor_s))
        route.update(dict.fromkeys(pos, how))
        instrumented = sum(null.host_s[p] for p in pos)
        rows.append({"position": pos[0], "positions": len(pos), "operator": name,
                     "layer": layer, "route": how, "direct_s": c.direct_s, "chain_s": c.chain_s,
                     "children": list(c.children), "bare_s": total,
                     "instrumented_s": instrumented,
                     "ratio": instrumented / total if total > 0 else None})
        o = ops.setdefault(name or "entry point", {"calls": 0, "composite": ok,
                                                   "bare_s": 0.0, "instrumented_s": 0.0,
                                                   "direct_s": 0.0, "chain_s": 0.0})
        o["calls"] += 1
        o["bare_s"] += total
        o["instrumented_s"] += instrumented
        o["direct_s"] += c.direct_s
        o["chain_s"] += c.chain_s if ok else c.direct_s
    for o in ops.values():
        o["ratio"] = o["instrumented_s"] / o["bare_s"] if o["bare_s"] > 0 else None
    for i, v in null.host_s.items():
        if i in iso:
            host[i], route[i] = v, ISOLATED_REPLAY
        elif seq.ops[i].kind is SeqKind.HOST_COMPUTE:
            host[i], route[i] = v, HOST_COMPUTE_ROUTE
    if set(host) != set(null.host_s):
        raise ProfileMismatch(f"the bare prices cover {len(host)} positions and the null pass "
                              f"{len(null.host_s)}")
    profile = HostProfile(
        host_s=dict(sorted(host.items())), reps_s={}, launch_floor_s=bp.floor_s, reps=bp.reps,
        sequence_sha256=null.sequence_sha256, cost_source=dict(null.cost_source),
        met_at=dict(null.met_at), empty_calls=null.empty_calls, pricing=BARE_TIMER,
        price_route=dict(sorted(route.items())))
    instr = sum(r["instrumented_s"] for r in rows)
    bare = sum(r["bare_s"] for r in rows)
    record = {
        "schema": SCHEMA, "reps": bp.reps, "timer_overhead_ns": bp.overhead_ns,
        "launch_floor_s": bp.floor_s, "signatures_timed": bp.signatures,
        "calls_priced": len(rows), "empty_calls": len(bp.empty),
        "routes": {r: sum(1 for v in route.values() if v == r) for r in PRICE_ROUTES},
        "host_total_s": profile.total_s,
        "chain_gate": {"aggregate_tol": aggregate_tol, "composite_calls": len(composites),
                       "abs_diff_s": diff, "host_s": host_total, "share": share,
                       "passed": passed,
                       "fallback": FALLBACK_LABEL if passed is False else None},
        "instrumented_over_bare": {"instrumented_s": instr, "bare_s": bare,
                                   "ratio": instr / bare if bare > 0 else None},
        "operators": dict(sorted(ops.items())), "calls": rows}
    return profile, record


def gate_summary(record: Mapping) -> dict:
    """The cell index's view of a bare record."""
    g = record["chain_gate"]
    return {"pricing": BARE_TIMER, "passed": g["passed"], "share": g["share"],
            "aggregate_tol": g["aggregate_tol"], "composite_calls": g["composite_calls"],
            "fallback": g["fallback"],
            "instrumented_over_bare": record["instrumented_over_bare"]["ratio"]}
