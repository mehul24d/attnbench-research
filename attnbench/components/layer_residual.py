"""The per-layer residual: the arm's own Python between calls, which the bare
prices leave out (T2.2; researcher's ruling, 1 Nov 2026, after Shakedown 6;
measured with the device idle since the adviser's ruling B1 of 8 Nov 2026).

The bare timer (`bare_timer`) prices each call alone, so the host time the
arm spends between calls (its Python, the dispatcher around calls that
launch nothing) is in no price. The residual adds it back per layer.

**Why the device is idle.** The first method (schema 1) timed each layer's
whole span with real device work running and took the bare prices off. On
both P1 repeats every residual layer was one where the device fell behind
the host, so a span held the host's waits for the device: 5,021 ms at the
oracle's 32768 cell against a measured prefill of 2,988 ms, charged as host
time and serialised again by the recursion at every synchronisation (Gate A
v1, Appendix A, 8 Nov 2026). This method times only the gaps between calls,
each one opened after a synchronise, so no wait for the device can be in it.

**The gaps** (`idle_gaps`). The arm's prefill runs for real under the tape
(`TapeReplayer(serve_setitem=False)`, so it takes the recorded path), under
`_GapMode`, a dispatch mode that meets calls where the bare pass does (the
recorded chains' operators, by the same device rule): every call it meets,
every interaction the tape serves, every non-aten entry point (`NullEntry`)
and every host-compute region is a call, each a priced position. Between
calls the clock runs; at a call it stops, the call runs, the device is
synchronised outside the clock, and the clock starts again. A layer's gaps
run from its forward pre-hook to its forward hook, so a layer has one
interval more than it has calls. A call nested in a met one is part of it.

**Host-side functions** (`host_side`; adviser's ruling 2 of 9 Nov 2026,
seventh paste). A function declared host-side (`cpu_boundary.
HOST_SIDE_FUNCTIONS`: the oracle's reference mask builder) runs inside a
gap, but with every Python mode suspended (the tape's torch-function mode
and `_GapMode`), its own time read by the clock around it and added to the
layer's gap. Under the modes each of its CPU operators paid both modes'
dispatch: on T3.2c the oracle's builder, some 2,000 to 8,000 assignments a
layer, read 3.8 to 6.5 times the layer's whole measured time (schema 2).
The function touches no device (`cpu_boundary.launch_check` checks that in
the recording), so the tape has nothing in it to serve, and a departure
would be refused at the tape's next entry.

**The timer's overhead.** Each run first issues `CALIBRATION_CALLS`
back-to-back one-element `add_` calls under the same stack; the median gap
between two of them is the instrument's own time per interval (the modes'
entry and exit, the clock reads, one Python statement), and it is taken off
each interval of that run.

**Reps.** One untimed run, then at least `MIN_REPS` timed ones, and more
until the cell's median total is stable: the median of all n runs within
`STABLE_REL` of the median of the first n - 2, at most `MAX_REPS` runs. A
cell that does not settle keeps the median and is labelled
(`UNSTABLE_LABEL`), as the spread tolerance does: labelled, never refused.

**The residual** (`residual`). Per layer, the median over runs of its net
gaps. It is added to the layer's call positions in proportion to the null
pass's gaps there (each position's null-pass cost less the null launch
floor; equal shares if they are all zero): the instrumented pass says where
in the layer the Python is, the idle gaps say how much. A layer whose median
is negative (its gaps are within the timer's overhead) adds nothing and is
labelled (`NEGATIVE_LABEL`). Positions outside the layers get none.

**The tolerance** (T4.1 lock table row 34, working 10%). A layer's spread is
the interquartile range of its runs over their median. A layer above the
tolerance keeps its residual and labels the cell (`SPREAD_LABEL`).

**What it leaves out.** Python inside a call that a torch function wraps
(`F.softmax`'s argument checks) runs inside the call and is in no gap; the
bare price, which re-issues the aten operator, leaves it out too.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch.autograd.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode

from .host_profile import (ISOLATED_REPLAY, HostProfile, NullEntry, aten_name, device_call,
                           launching_names)
from .op_sequence import HOST_COMPUTE, OpSequence, SeqKind
from .sync_tape import SyncTape, TapeReplayer, classify

SCHEMA = "attnbench.layer_residual/3"
METHOD = "idle_device_gaps_bare_host"
MIN_REPS, MAX_REPS, STABLE_REL = 7, 21, 0.01
CALIBRATION_CALLS = 64
SPREAD_LABEL = "per-layer residual spread above its T4.1 tolerance"
NEGATIVE_LABEL = "the layer's gaps are within the timer's overhead: no residual added"
UNSTABLE_LABEL = f"the cell's median residual did not settle within {MAX_REPS} runs"
_CALIBRATION = "calibration"


@dataclass
class Gaps:
    """Per layer, its net gaps in each timed run (seconds, the run's overhead
    taken off each interval) and its intervals per run; each run's overhead
    per interval; and whether the cell's median settled."""

    gaps_s: dict = field(default_factory=dict)           # layer -> (net gaps per run)
    host_side_s: dict = field(default_factory=dict)      # layer -> (host-side time per run)
    intervals: dict = field(default_factory=dict)        # layer -> intervals in a run
    overhead_s: tuple = ()
    stable: bool = True


class _Timer:
    """The gap clock (module docstring): `stop` at a call, `start` after it,
    the device synchronised before the clock starts again."""

    def __init__(self, clock, sync):
        self.clock, self.sync = clock, sync
        self.layer = None
        self.last: Optional[float] = None
        self.sum, self.n, self.inside = 0.0, 0, 0
        self.host_side = 0.0
        self.calibration: list = []

    def stop(self) -> None:
        if self.layer is not None and self.last is not None:
            d = self.clock() - self.last
            if self.layer == _CALIBRATION:
                self.calibration.append(d)
            self.sum += d
            self.n += 1
        self.last = None

    def start(self) -> None:
        if self.layer is None:
            return
        self.sync()
        self.last = self.clock()

    @contextlib.contextmanager
    def call(self):
        self.stop()
        self.inside += 1
        try:
            yield
        finally:
            self.inside -= 1
            self.start()


class _GapMode(TorchDispatchMode):
    """Meets calls where the bare pass does (`bare_timer.BareMode`) and runs
    each as a call of the gap clock."""

    def __init__(self, launching: Optional[frozenset], device_type: str, timer: _Timer):
        super().__init__()
        self.launching, self.device_type, self.timer = launching, device_type, timer

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        t = self.timer
        if t.inside or (t.layer != _CALIBRATION and self.launching is not None
                        and aten_name(func) not in self.launching) or \
                device_call(func, args, kwargs, self.device_type) is None:
            return func(*args, **kwargs)
        with t.call():
            return func(*args, **kwargs)


class _GapReplayer(TapeReplayer):
    """The tape, with each interaction it serves a call of the gap clock."""

    def __init__(self, tape: SyncTape, staged, timer: _Timer):
        super().__init__(tape, staged, serve_setitem=False)
        self.timer = timer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if self.timer.inside or \
                classify(func, args, kwargs or {}, self.tape.device_type) is None:
            return super().__torch_function__(func, types, args, kwargs)
        with self.timer.call():
            return super().__torch_function__(func, types, args, kwargs)


@contextlib.contextmanager
def _entries(entries: Sequence[NullEntry], timer: _Timer):
    saved = []
    try:
        for e in entries:
            original = getattr(e.module, e.attr)
            saved.append((e, original))

            def timed(*args, _orig=original, **kwargs):
                if timer.inside:
                    return _orig(*args, **kwargs)
                with timer.call():
                    return _orig(*args, **kwargs)
            setattr(e.module, e.attr, timed)
        yield
    finally:
        for e, original in saved:
            setattr(e.module, e.attr, original)


@contextlib.contextmanager
def _no_modes():
    """Every Python mode suspended: the torch-function modes and the dispatch modes."""
    from torch.utils._python_dispatch import _disable_current_modes
    with torch._C.DisableTorchFunction(), _disable_current_modes():
        yield


@contextlib.contextmanager
def _host_side(fns: Sequence[tuple], timer: _Timer):
    """Each (module, attribute) in `fns` runs bare inside its gap (module docstring)."""
    saved = []
    try:
        for module, attr in fns:
            original = getattr(module, attr)
            saved.append((module, attr, original))

            def bare(*args, _orig=original, **kwargs):
                if timer.inside or timer.layer in (None, _CALIBRATION):
                    return _orig(*args, **kwargs)
                timer.stop()
                timer.inside += 1
                try:
                    with _no_modes():
                        t0 = timer.clock()
                        out = _orig(*args, **kwargs)
                        d = timer.clock() - t0
                    timer.sum += d
                    timer.host_side += d
                finally:
                    timer.inside -= 1
                    timer.start()
                return out
            setattr(module, attr, bare)
        yield
    finally:
        for module, attr, original in reversed(saved):
            setattr(module, attr, original)


@contextlib.contextmanager
def _host_compute(timer: _Timer):
    """Each `attnbench::host_compute` region is a call: its time is priced
    at its position (`bare_timer`, `host_compute_null_pass`)."""
    enter, exit_ = record_function.__enter__, record_function.__exit__

    def __enter__(self):
        r = enter(self)
        if self.name == HOST_COMPUTE and not timer.inside:
            timer.stop()
            timer.inside += 1
            self._attnbench_gap = True
        return r

    def __exit__(self, *a):
        if getattr(self, "_attnbench_gap", False):
            timer.inside -= 1
            timer.start()
        return exit_(self, *a)

    record_function.__enter__, record_function.__exit__ = __enter__, __exit__
    try:
        yield
    finally:
        record_function.__enter__, record_function.__exit__ = enter, exit_


@contextlib.contextmanager
def _hooks(layers: Sequence[torch.nn.Module], timer: _Timer, out: dict):
    handles = []

    def pre(i):
        def h(mod, args):
            timer.layer, timer.sum, timer.n, timer.host_side = i, 0.0, 0, 0.0
            timer.start()
        return h

    def post(i):
        def h(mod, args, result):
            timer.stop()
            out[i] = (timer.sum, timer.n, timer.host_side)
            timer.layer = None
        return h
    try:
        for i, m in enumerate(layers):
            handles.append(m.register_forward_pre_hook(pre(i)))
            handles.append(m.register_forward_hook(post(i)))
        yield
    finally:
        for h in handles:
            h.remove()


def _calibrate(timer: _Timer, device_type: str, calls: int) -> float:
    x = torch.zeros(1, device=device_type)
    timer.layer, timer.calibration = _CALIBRATION, []
    timer.start()
    for _ in range(calls):
        x.add_(1)
    timer.stop()
    timer.layer = None
    return float(statistics.median(timer.calibration[1:-1] or timer.calibration))


def _one_run(run_prefill, tape, staged, layers, launching, entries, host_side, *, sync,
             clock, device_type, calls) -> tuple[dict, float]:
    timer = _Timer(clock, sync)
    raw: dict = {}
    rep = _GapReplayer(tape, staged, timer)
    with _host_side(host_side, timer), rep, _entries(entries, timer), _host_compute(timer), \
            _hooks(layers, timer, raw), _GapMode(launching, device_type, timer):
        overhead = _calibrate(timer, device_type, calls)
        run_prefill()
    rep.finish()
    sync()
    return {i: (s - n * overhead, n, h) for i, (s, n, h) in raw.items()}, overhead


def _settled(totals: Sequence[float], min_reps: int, rel: float) -> bool:
    if len(totals) < max(min_reps, 3):
        return False
    a, b = statistics.median(totals), statistics.median(totals[:-2])
    return abs(a - b) <= rel * abs(a) if a else a == b


def idle_gaps(run_prefill: Callable[[], None], tape: SyncTape,
              layers: Sequence[torch.nn.Module], *, seq: Optional[OpSequence] = None,
              entries: Sequence[NullEntry] = (), host_side: Sequence[tuple] = (),
              device_type: str = "cuda",
              staged_device=None, sync: Callable[[], None] = torch.cuda.synchronize,
              clock: Callable[[], float] = time.perf_counter, min_reps: int = MIN_REPS,
              max_reps: int = MAX_REPS, stable_rel: float = STABLE_REL,
              calibration_calls: int = CALIBRATION_CALLS) -> Gaps:
    """One untimed run, then timed runs until the median settles (module
    docstring). With no `seq`, every call that touches the device is met.
    `host_side`: (module, attribute) pairs run bare (module docstring)."""
    for name, v in (("min_reps", min_reps), ("max_reps", max_reps)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be an int >= 1")
    if max_reps < min_reps:
        raise ValueError("max_reps is below min_reps")
    if not layers:
        raise ValueError("no layers: the residual is per layer")
    launching = None if seq is None else launching_names(seq)
    staged = tape.stage(staged_device if staged_device is not None
                        else torch.device(device_type))
    kw = dict(sync=sync, clock=clock, device_type=device_type, calls=calibration_calls)
    _one_run(run_prefill, tape, staged, layers, launching, entries, host_side, **kw)
    runs, overheads, totals = [], [], []
    while len(runs) < max_reps and not _settled(totals, min_reps, stable_rel):
        r, ov = _one_run(run_prefill, tape, staged, layers, launching, entries, host_side,
                         **kw)
        runs.append(r)
        overheads.append(ov)
        totals.append(sum(v for v, _, _ in r.values()))
    out = Gaps(overhead_s=tuple(overheads), stable=_settled(totals, min_reps, stable_rel))
    for i in sorted(runs[0]):
        out.gaps_s[i] = tuple(r[i][0] for r in runs)
        out.host_side_s[i] = tuple(r[i][2] for r in runs)
        ns = {r[i][1] for r in runs}
        if len(ns) != 1:
            raise ValueError(f"layer {i} had {sorted(ns)} intervals in different runs: "
                             f"the prefill did not take one path")
        out.intervals[i] = ns.pop()
    return out


def declared_host_side() -> tuple:
    """`cpu_boundary.HOST_SIDE_FUNCTIONS` as the (module, attribute) pairs
    `idle_gaps` runs bare."""
    from attnbench import cpu_boundary, masks
    from attnbench.accuracy import model
    modules = {"model.py": model, "masks.py": masks}
    return tuple((modules[f], name) for f, name in cpu_boundary.HOST_SIDE_FUNCTIONS)


def _spread(xs: Sequence[float]) -> float:
    med = statistics.median(xs)
    if len(xs) < 2 or med <= 0:
        return 0.0
    q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return (q3 - q1) / med


def residual(seq: OpSequence, bare: HostProfile, null: HostProfile, gaps: Gaps, *,
             spread_tol: float) -> tuple[HostProfile, dict]:
    """(the bare profile with each layer's residual added, the record written
    beside it) (module docstring)."""
    by_layer: dict = {}
    for i in bare.host_s:
        layer = seq.ops[i].layer
        if layer is not None:
            by_layer.setdefault(layer, []).append(i)
    if set(gaps.gaps_s) != set(by_layer):
        raise ValueError(f"the gaps cover layers {sorted(gaps.gaps_s)} and the sequence's "
                         f"priced positions layers {sorted(by_layer)}")
    added: dict[int, float] = {}
    rows = []
    for layer, positions in sorted(by_layer.items()):
        xs = gaps.gaps_s[layer]
        rho = float(statistics.median(xs))
        calls = [i for i in positions if seq.ops[i].kind is not SeqKind.HOST_COMPUTE
                 and null.source(i) != ISOLATED_REPLAY]
        w = {i: max(null.host_s.get(i, 0.0) - null.launch_floor_s, 0.0) for i in calls}
        total_w = sum(w.values())
        labels = []
        if rho < 0:
            labels.append(NEGATIVE_LABEL)
        elif calls:
            for i in calls:
                added[i] = rho * (w[i] / total_w if total_w > 0 else 1 / len(calls))
        spread = _spread(xs)
        if spread > spread_tol:
            labels.append(SPREAD_LABEL)
        rows.append({"layer": layer, "residual_s": rho, "gaps_s": list(xs),
                     "intervals": gaps.intervals.get(layer),
                     "host_side_s": (float(statistics.median(gaps.host_side_s[layer]))
                                     if gaps.host_side_s.get(layer) else 0.0),
                     "added_s": max(rho, 0.0) if calls else 0.0, "positions": len(calls),
                     "spread": spread, "labels": labels})
    host = {i: v + added.get(i, 0.0) for i, v in bare.host_s.items()}
    profile = HostProfile(
        host_s=host, reps_s={}, launch_floor_s=bare.launch_floor_s, reps=bare.reps,
        sequence_sha256=bare.sequence_sha256, cost_source=dict(bare.cost_source),
        met_at=dict(bare.met_at), empty_calls=bare.empty_calls, pricing=bare.pricing,
        price_route=dict(bare.price_route), residual_s=dict(sorted(added.items())))
    labels = sorted({l for r in rows for l in r["labels"]} |
                    (set() if gaps.stable else {UNSTABLE_LABEL}))
    ov = gaps.overhead_s
    record = {"schema": SCHEMA, "method": METHOD,
              "reps": len(next(iter(gaps.gaps_s.values()), ())), "stable": gaps.stable,
              "min_reps": MIN_REPS, "max_reps": MAX_REPS, "stable_rel": STABLE_REL,
              "overhead_s": float(statistics.median(ov)) if ov else 0.0,
              "overhead_s_per_run": list(ov), "spread_tol": spread_tol,
              "gap_total_s": sum(r["residual_s"] for r in rows),
              "host_side_total_s": sum(r["host_side_s"] for r in rows),
              "residual_added_s": sum(added.values()),
              "layers_over_spread": [r["layer"] for r in rows if SPREAD_LABEL in r["labels"]],
              "layers_negative": [r["layer"] for r in rows if NEGATIVE_LABEL in r["labels"]],
              "labels": labels, "layers": rows}
    return profile, record


def summary(record: Mapping) -> dict:
    """The cell index's view of a residual record."""
    return {k: record[k] for k in ("method", "residual_added_s", "gap_total_s", "overhead_s",
                                   "reps", "stable", "spread_tol", "layers_over_spread",
                                   "layers_negative", "labels")}
