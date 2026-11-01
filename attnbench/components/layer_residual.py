"""The per-layer residual: the arm's own Python between calls, which the bare
prices leave out (T2.2; researcher's ruling, 1 Nov 2026, after Shakedown 6).

The bare timer (`bare_timer`) prices each call alone, so the host time the
arm spends between calls (its Python, the dispatcher around calls that
launch nothing) is in no price; the null pass's gaps held it, under
instruments that put 5.5 to 11.5 times the bare total on Shakedown 6's
cells. The residual adds it back per layer, measured uninstrumented.

**The span** (`tape_spans`). The arm's prefill is run for real, device work
and all, under the tape alone (`TapeReplayer(serve_setitem=False)`): every
host-device interaction is served from the tape, so the host waits for no
read and takes the recorded path, and no dispatch mode is on the stack. The
only instrument is two clock reads per layer, at its forward pre-hook and
forward hook; a layer's span is the time between them. A boolean-mask write
(`x[mask] = v`) runs as written and waits for the device inside its
`nonzero`; around it the layer's clock stops, the device is synchronised,
and the write itself is timed and counted, as the bare timer times a call
after a synchronise outside its timer, so no wait for earlier device work is
in a span. One untimed run goes first, then `reps` timed ones, the median
per layer.

**The residual** (`residual`). Per layer, the median span less the sum of
the bare prices of the layer's positions (host-compute regions keep the
null pass's direct time there, as in the bare profile). It is added to the
layer's call positions in proportion to the null pass's gaps there (each
position's null-pass cost less the null launch floor; equal shares if they
are all zero): the instrumented pass says where in the layer the Python is,
the uninstrumented span says how much. A negative residual (the bare prices
exceed the span) adds nothing and labels the layer (`NEGATIVE_LABEL`).
Positions outside the layers get none: the hooks see no span there.

**The tolerance** (T4.1 lock table row 34, working 10%). A layer's spread is
the interquartile range of its timed spans over their median. A layer above
the tolerance keeps its residual and labels the cell (`SPREAD_LABEL`), as
the chain gate's fallback does: labelled, never refused.

**The device check.** One more run records a device event at each hook. A
layer through which the device's elapsed time since the first layer's
pre-hook exceeds the host's by more than `DEVICE_BEHIND_MARGIN` is one where
the device fell behind the host, and its span may hold launch-queue waits
the recursion prices elsewhere: the layer is labelled (`DEVICE_BEHIND_LABEL`)
and keeps its residual. Behind the CPU boundary there is no device clock and
no check.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch
from torch.overrides import TorchFunctionMode

from .host_profile import ISOLATED_REPLAY, HostProfile
from .op_sequence import OpSequence, SeqKind
from .sync_tape import SyncTape, TapeReplayer, classify

SCHEMA = "attnbench.layer_residual/1"
RESIDUAL_REPS = 5
DEVICE_BEHIND_MARGIN = 0.05
SPREAD_LABEL = "per-layer residual spread above its T4.1 tolerance"
NEGATIVE_LABEL = "bare prices exceed the layer's uninstrumented span: no residual added"
DEVICE_BEHIND_LABEL = "the device fell behind the host in this layer: its span may hold waits"


@dataclass
class Spans:
    """Per layer, the host span of each timed run (seconds); the boolean-mask
    writes timed in each; and, if a device clock was given, the device's and
    the host's elapsed time from the first layer's pre-hook to each layer's
    end in the device check's run."""

    spans_s: dict = field(default_factory=dict)          # layer -> (span per run)
    setitems: dict = field(default_factory=dict)         # layer -> writes per run
    device_cum_s: Optional[dict] = None
    host_cum_s: Optional[dict] = None


class _PauseAtSetitem(TorchFunctionMode):
    """Stops the layer's clock around a boolean-mask write, synchronises, and
    times the write alone (module docstring)."""

    def __init__(self, device_type: str, sync: Callable[[], None], state: dict, clock):
        super().__init__()
        self.device_type, self.sync, self.state, self.clock = device_type, sync, state, clock

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if classify(func, args, kwargs, self.device_type) != "bool_setitem" or \
                self.state.get("open") is None:
            return func(*args, **kwargs)
        paused = self.clock()
        self.sync()
        t0 = self.clock()
        out = func(*args, **kwargs)
        self.state["excluded"] += t0 - paused        # the synchronise, out of the span
        self.state["writes"] += 1
        return out


@contextlib.contextmanager
def _hooks(layers: Sequence[torch.nn.Module], state: dict, clock, on_enter=None, on_exit=None):
    handles = []

    def pre(i):
        def h(mod, args):
            if on_enter is not None:
                on_enter(i)
            state.update(open=i, t0=clock(), excluded=0.0, writes=0)
        return h

    def post(i):
        def h(mod, args, out):
            t1 = clock()
            state["spans"][i] = t1 - state["t0"] - state["excluded"]
            state["writes_by_layer"][i] = state["writes"]
            state["open"] = None
            if on_exit is not None:
                on_exit(i)
        return h
    try:
        for i, m in enumerate(layers):
            handles.append(m.register_forward_pre_hook(pre(i)))
            handles.append(m.register_forward_hook(post(i)))
        yield
    finally:
        for h in handles:
            h.remove()


def _one_run(run_prefill, tape: SyncTape, staged, layers, *, sync, clock, device_type,
             on_enter=None, on_exit=None) -> dict:
    state = {"open": None, "spans": {}, "writes_by_layer": {}, "excluded": 0.0, "writes": 0}
    rep = TapeReplayer(tape, staged, serve_setitem=False)
    with rep, _PauseAtSetitem(device_type, sync, state, clock), \
            _hooks(layers, state, clock, on_enter, on_exit):
        run_prefill()
    rep.finish()
    sync()
    return state


def tape_spans(run_prefill: Callable[[], None], tape: SyncTape,
               layers: Sequence[torch.nn.Module], *, reps: int = RESIDUAL_REPS,
               device_type: str = "cuda", staged_device=None,
               sync: Callable[[], None] = torch.cuda.synchronize,
               clock: Callable[[], float] = time.perf_counter,
               event: Optional[Callable[[], object]] = None) -> Spans:
    """One untimed run, `reps` timed runs and, given a device `event`
    factory (`torch.cuda.Event(enable_timing=True)`), one device-check run
    (module docstring)."""
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be an int >= 1")
    if not layers:
        raise ValueError("no layers: the residual is per layer")
    staged = tape.stage(staged_device if staged_device is not None
                        else torch.device(device_type))
    kw = dict(sync=sync, clock=clock, device_type=device_type)
    _one_run(run_prefill, tape, staged, layers, **kw)
    out = Spans()
    for _ in range(reps):
        s = _one_run(run_prefill, tape, staged, layers, **kw)
        for i, v in s["spans"].items():
            out.spans_s.setdefault(i, []).append(v)
            out.setitems.setdefault(i, []).append(s["writes_by_layer"][i])
    out.spans_s = {i: tuple(v) for i, v in sorted(out.spans_s.items())}
    out.setitems = {i: tuple(v) for i, v in sorted(out.setitems.items())}
    if event is not None:
        marks: dict = {}
        start = {}

        def enter(i):
            if not start:
                ev = event()
                ev.record()
                start.update(event=ev, host=clock())

        def leave(i):
            ev = event()
            ev.record()
            marks[i] = (ev, clock())
        _one_run(run_prefill, tape, staged, layers, on_enter=enter, on_exit=leave, **kw)
        out.device_cum_s = {i: start["event"].elapsed_time(ev) * 1e-3
                            for i, (ev, _) in sorted(marks.items())}
        out.host_cum_s = {i: h - start["host"] for i, (_, h) in sorted(marks.items())}
    return out


def _spread(xs: Sequence[float]) -> float:
    med = statistics.median(xs)
    if len(xs) < 2 or med <= 0:
        return 0.0
    q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return (q3 - q1) / med


def residual(seq: OpSequence, bare: HostProfile, null: HostProfile, spans: Spans, *,
             spread_tol: float) -> tuple[HostProfile, dict]:
    """(the bare profile with each layer's residual added, the record written
    beside it) (module docstring)."""
    by_layer: dict = {}
    for i in bare.host_s:
        layer = seq.ops[i].layer
        if layer is not None:
            by_layer.setdefault(layer, []).append(i)
    if set(spans.spans_s) != set(by_layer):
        raise ValueError(f"the spans cover layers {sorted(spans.spans_s)} and the sequence's "
                         f"priced positions layers {sorted(by_layer)}")
    added: dict[int, float] = {}
    rows = []
    for layer, positions in sorted(by_layer.items()):
        xs = spans.spans_s[layer]
        span = float(statistics.median(xs))
        priced = sum(bare.host_s[i] for i in positions)
        rho = span - priced
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
        behind = None
        if spans.device_cum_s is not None:
            d, h = spans.device_cum_s[layer], spans.host_cum_s[layer]
            behind = d > h * (1 + DEVICE_BEHIND_MARGIN)
            if behind:
                labels.append(DEVICE_BEHIND_LABEL)
        rows.append({"layer": layer, "span_s": span, "spans_s": list(xs),
                     "bare_s": priced, "residual_s": rho,
                     "added_s": max(rho, 0.0) if calls else 0.0, "positions": len(calls),
                     "spread": spread, "setitems": list(spans.setitems.get(layer, ())),
                     "device_behind": behind, "labels": labels})
    host = {i: v + added.get(i, 0.0) for i, v in bare.host_s.items()}
    profile = HostProfile(
        host_s=host, reps_s={}, launch_floor_s=bare.launch_floor_s, reps=bare.reps,
        sequence_sha256=bare.sequence_sha256, cost_source=dict(bare.cost_source),
        met_at=dict(bare.met_at), empty_calls=bare.empty_calls, pricing=bare.pricing,
        price_route=dict(bare.price_route), residual_s=dict(sorted(added.items())))
    labels = sorted({l for r in rows for l in r["labels"]})
    record = {"schema": SCHEMA, "reps": len(next(iter(spans.spans_s.values()), ())),
              "spread_tol": spread_tol, "device_behind_margin": DEVICE_BEHIND_MARGIN,
              "device_checked": spans.device_cum_s is not None,
              "span_total_s": sum(r["span_s"] for r in rows),
              "bare_in_layers_s": sum(r["bare_s"] for r in rows),
              "residual_added_s": sum(added.values()),
              "layers_over_spread": [r["layer"] for r in rows if SPREAD_LABEL in r["labels"]],
              "layers_negative": [r["layer"] for r in rows if NEGATIVE_LABEL in r["labels"]],
              "layers_device_behind": [r["layer"] for r in rows if r["device_behind"]],
              "labels": labels, "layers": rows}
    return profile, record


def summary(record: Mapping) -> dict:
    """The cell index's view of a residual record."""
    return {k: record[k] for k in ("residual_added_s", "span_total_s", "bare_in_layers_s",
                                   "spread_tol", "layers_over_spread", "layers_negative",
                                   "layers_device_behind", "labels", "device_checked")}
