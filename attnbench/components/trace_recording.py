"""Recording side of T2.3's trace contract (`op_sequence` module docstring).

`record_prefill_trace` profiles one prefill with the annotations
`op_sequence.parse_trace` reads. `annotate_layers` marks decoder layers
through forward hooks, so no model or vendored code is edited (rule 8), and
`host_compute` marks host-side estimator work.

The **mechanism zoo** is the classifier's check on real hardware. Each zoo
layer runs one way of making the host wait, chosen from Python with a known
answer, and `check_zoo` compares what `parse_trace` read from the trace
with that answer. It needs a CUDA device; on the CPU only its bookkeeping
is tested. Where PyTorch's behaviour was not checked on a GPU before this
was written (how `nonzero()` reads its count back, whether a tensor built
from a Python list waits), the expected answer lists every alternative and
the run records which one held.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Union

import torch
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile

from .op_sequence import (HOST_COMPUTE, REGION, OpSequence, SeqKind, SeqOp,
                          layer_annotation)


def profiler_kwargs(cuda: bool) -> dict:
    """The profiler settings every recorded trace uses.

    Shapes and memory are not recorded (the sequence does not need them, and
    they slow the host); Python frames are, so each operation names the
    function that issued it.
    """
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    return dict(activities=activities, record_shapes=False, with_stack=True,
                profile_memory=False)


@contextlib.contextmanager
def annotate_layers(layers: Sequence[torch.nn.Module]) -> Iterator[None]:
    """Wrap each decoder layer's forward in `attnbench::layer.<i>`."""
    open_: dict[int, record_function] = {}
    handles = []

    def pre(i):
        def hook(module, args):
            rf = record_function(layer_annotation(i))
            rf.__enter__()
            open_[i] = rf
        return hook

    def post(i):
        def hook(module, args, output):
            open_.pop(i).__exit__(None, None, None)
        return hook

    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(pre(i)))
            handles.append(layer.register_forward_hook(post(i)))
        yield
    finally:
        for h in handles:
            h.remove()
        for rf in open_.values():
            rf.__exit__(None, None, None)


def host_compute():
    """Mark host-side estimator work (CPU mask building)."""
    return record_function(HOST_COMPUTE)


def _sync(cuda: bool) -> None:
    if cuda:
        torch.cuda.synchronize()


def record_prefill_trace(run_prefill: Callable[[], None], path: Union[str, Path], *,
                         layers: Sequence[torch.nn.Module] = (), warmup: int = 1,
                         cuda: Optional[bool] = None) -> Path:
    """Profile one prefill and write its trace to `path`.

    `warmup` untraced runs come first. Then one more prefill is the
    profiler's own warm-up step (collected, then discarded): in Shakedown 2
    the first kernel of a trace that starts collecting with the prefill had
    no device record (meanpool's probe, Figure 1), so the recorded prefill
    is the profiler's second step. The device is synchronised before the
    region opens, so no earlier work runs inside it, and after it closes, so
    every activity is in the trace. Neither synchronise is inside the
    region; the closing synchronise of the timer is the recursion's
    (`recursion.close_prefill`).
    """
    cuda = torch.cuda.is_available() if cuda is None else cuda
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        run_prefill()
    _sync(cuda)
    path = Path(path)
    sched = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with profile(**profiler_kwargs(cuda), schedule=sched,
                 on_trace_ready=lambda p: p.export_chrome_trace(str(path))) as prof:
        run_prefill()                 # the profiler's warm-up step, not exported
        _sync(cuda)
        prof.step()
        with annotate_layers(layers):
            with record_function(REGION):
                run_prefill()
        _sync(cuda)
        prof.step()
    if not path.exists():
        raise RuntimeError(f"the profiler wrote no trace to {path}")
    return path


# Prefills `record_prefill_trace` runs beyond `warmup` before the traced one:
# the profiler's warm-up step. Callers that count calls read this.
record_prefill_trace.profiler_warmup = 1


# --- the mechanism zoo -------------------------------------------------------

def signature(op: SeqOp) -> tuple:
    """What the zoo checks for each operation that is not device work."""
    if op.kind is SeqKind.SYNC:
        return ("sync", op.mechanism)
    if op.kind is SeqKind.COPY:
        return ("copy", op.direction, op.loaded)
    return (op.kind.value,)


@dataclass(frozen=True)
class ZooStep:
    layer: int
    name: str
    # Each alternative is the tuple of signatures, in order, of the layer's
    # operations that are not device work.
    expected: tuple[tuple[tuple, ...], ...]
    composite: Optional[str] = None


_SIZE_READBACK = (
    (("sync", "copy_to_host_pinned"),),
    (("sync", "copy_to_host_pageable"),),
    (("sync", "item"),),
)

ZOO: tuple[ZooStep, ...] = (
    ZooStep(0, "explicit torch.cuda.synchronize()", ((("sync", "synchronize"),),)),
    ZooStep(1, ".item() on a GPU tensor", ((("sync", "item"),),)),
    ZooStep(2, "bool() of a GPU tensor", ((("sync", "bool"),),)),
    ZooStep(3, "assert on a GPU tensor", ((("sync", "bool"),),)),
    ZooStep(4, "sizing from nonzero()", _SIZE_READBACK, composite="aten::nonzero"),
    ZooStep(5, "boolean-mask indexing", _SIZE_READBACK, composite="aten::index"),
    ZooStep(6, ".cpu() of a GPU tensor", ((("sync", "copy_to_host_pageable"),),)),
    ZooStep(7, "blocking copy into pinned memory", ((("sync", "copy_to_host_pinned"),),)),
    ZooStep(8, "non-blocking copy into pinned memory, then synchronize",
            ((("copy", "DtoH", False), ("sync", "synchronize")),)),
    ZooStep(9, "GPU tensor built from a Python list",
            ((("sync", "copy_from_host_pageable"),),
             (("sync", "copy_from_host_pinned"),),
             (("copy", "HtoD", False),))),
    ZooStep(10, "host compute around a non-blocking upload",
            ((("host_compute",), ("copy", "HtoD", True), ("host_compute",)),)),
    ZooStep(11, "event synchronize", ((("sync", "event_synchronize"),),)),
)


def run_mechanism_zoo(device: Union[str, torch.device] = "cuda") -> None:
    """Run every zoo step once, each in its own layer annotation, inside
    the prefill region. Profile it with `record_prefill_trace(lambda:
    run_mechanism_zoo(), path, warmup=1)`."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("the mechanism zoo needs a CUDA device: every step "
                           "is a way of waiting for one")
    x = torch.linspace(-1.0, 1.0, 4096, device=device)
    pinned = torch.empty(4096, pin_memory=True)
    host = torch.arange(4096, dtype=torch.float32)
    staged = torch.empty(4096, pin_memory=True)

    def layer(i):
        return record_function(layer_annotation(i))

    with layer(0):
        y = x * 2
        torch.cuda.synchronize()
    with layer(1):
        y.sum().item()
    with layer(2):
        bool(y.sum() > 0)
    with layer(3):
        assert (y > -10).all()
    with layer(4):
        torch.nonzero(y > 0)
    with layer(5):
        y[y > 0]
    with layer(6):
        y.cpu()
    with layer(7):
        pinned.copy_(y)
    with layer(8):
        pinned.copy_(y, non_blocking=True)
        torch.cuda.synchronize()
    with layer(9):
        torch.tensor([0, 4096], dtype=torch.int32, device=device)
    with layer(10):
        with host_compute():
            torch.mul(host, 2, out=staged)
        g = staged.to(device, non_blocking=True)
        with host_compute():
            (host + 1).sum()
    with layer(11):
        g = g + 1
        ev = torch.cuda.Event()
        ev.record()
        ev.synchronize()


def run_two_stream_zoo(device: Union[str, torch.device] = "cuda") -> None:
    """Work on a second stream, which `stream_check` must flag."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("the two-stream zoo needs a CUDA device")
    x = torch.ones(4096, device=device)
    side = torch.cuda.Stream(device=device)
    with record_function(layer_annotation(0)):
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            y = x * 3
        torch.cuda.current_stream(device).wait_stream(side)
        (y + 1).sum()


@dataclass(frozen=True)
class ZooResult:
    step: ZooStep
    observed: tuple[tuple, ...]
    composites: tuple[Optional[str], ...]
    ok: bool


def check_zoo(seq: OpSequence) -> tuple[ZooResult, ...]:
    """Each zoo step's observed non-device operations against its answer."""
    results = []
    layers = set(seq.layers)
    for step in ZOO:
        ops = [op for op in seq.ops if op.layer == step.layer and not op.is_device]
        observed = tuple(signature(op) for op in ops)
        comps = tuple(op.composite for op in ops if op.kind is SeqKind.SYNC)
        ok = step.layer in layers and observed in step.expected
        if step.composite is not None:
            ok = ok and comps == (step.composite,)
        else:
            ok = ok and all(c is None for c in comps)
        results.append(ZooResult(step, observed, comps, ok))
    return tuple(results)
