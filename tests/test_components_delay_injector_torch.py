"""T2.25: the delay injector's torch side.

UNRUN: written in a container without torch or a GPU. The CPU tests need
torch only; the CUDA tests skip without a device. Neither has run yet.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from attnbench.components import delay_injector as di  # noqa: E402
from attnbench.components import host_profile  # noqa: E402
from attnbench.components.op_sequence import (  # noqa: E402
    CodePath, SeqKind, SeqOp, parse_trace, to_operations)
from attnbench.components.trace_recording import record_prefill_trace  # noqa: E402
from attnbench.predict import recursion  # noqa: E402
from attnbench.predict.delay_injection import (  # noqa: E402
    BUSY, IDLE, plan_variants)

from test_components_delay_injector import SEQ  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


class _MM(torch.nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n, n, device=device) / n ** 0.5)

    def forward(self, x):
        return torch.mm(x, self.w)


class _Toy(torch.nn.Module):
    """An add outside the layers, then one mm per layer."""

    def __init__(self, n=16, n_layers=2, device="cpu"):
        super().__init__()
        self.layers = torch.nn.ModuleList(_MM(n, device) for _ in range(n_layers))
        self.x = torch.randn(n, n, device=device)

    def forward(self):
        h = torch.add(self.x, 1.0)
        for layer in self.layers:
            h = layer(h)
        return h


# --- CPU, torch only -----------------------------------------------------------

def test_the_run_rule_is_host_profiles():
    """`delay_injector._runs` restates `host_profile`'s rule; they must agree."""
    for seq in (SEQ,
                replace(SEQ, ops=(SeqOp(SeqKind.KERNEL, layer=1, aten_op="aten::mm"),
                                  SeqOp(SeqKind.SYNC, layer=1, mechanism="item"),
                                  SeqOp(SeqKind.KERNEL, layer=1, aten_op="aten::mm")))):
        theirs = host_profile._merge_equal_neighbours(host_profile.runs_of(seq))
        ours = di._runs(seq)
        assert [(r.name, r.layer, r.positions) for r in theirs] == \
            [(r.name, r.layer, r.positions) for r in ours]


def test_the_dispatch_observer_reports_launching_calls_with_their_layer():
    toy = _Toy()
    seen = []
    observe = di.dispatch_observer(frozenset({"aten::add", "aten::mm"}),
                                   device_type="cpu", layers=toy.layers)
    with torch.no_grad():
        observe(lambda: toy(), lambda name, layer: seen.append((name, layer)))
    assert seen == [("aten::add", None), ("aten::mm", 0), ("aten::mm", 1)]
    # the hooks are gone afterwards
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in toy.layers)


def test_work_done_inside_on_call_is_not_observed():
    toy = _Toy()
    seen = []

    def on_call(name, layer):
        seen.append((name, layer))
        torch.mm(toy.x, toy.x)          # an injected aten call: must not count

    observe = di.dispatch_observer(frozenset({"aten::mm"}), device_type="cpu",
                                   layers=toy.layers)
    with torch.no_grad():
        observe(lambda: toy(), on_call)
    assert seen == [("aten::mm", 0), ("aten::mm", 1)]


def test_calls_on_another_device_type_are_not_observed():
    toy = _Toy()
    seen = []
    observe = di.dispatch_observer(frozenset({"aten::add", "aten::mm"}),
                                   device_type="cuda", layers=toy.layers)
    with torch.no_grad():
        observe(lambda: toy(), lambda name, layer: seen.append((name, layer)))
    assert seen == []


def test_cuda_mechanism_refuses_a_cpu_device():
    with pytest.raises(di.InjectionError, match="CUDA device"):
        di.cuda_mechanism("synchronize", device="cpu")


# --- CUDA ------------------------------------------------------------------------

def _cycles_for(seconds: float) -> int:
    """Rough `torch.cuda._sleep` cycles for `seconds` on this device."""
    torch.cuda.synchronize()
    probe = 10_000_000
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(probe)
    end.record()
    end.synchronize()
    return int(probe * seconds / (start.elapsed_time(end) * 1e-3))


@cuda
@pytest.mark.parametrize("mechanism", recursion.PRICED_MECHANISMS)
def test_every_mechanism_waits_for_the_work_before_it(mechanism):
    """Each injected mechanism must hold the host until the device has done
    what was issued before it: that is what makes it a synchronisation."""
    payload = 0 if mechanism in di.NO_PAYLOAD else 4096
    fire = di.cuda_mechanism(mechanism, payload)
    fire()                                   # first use outside the check
    torch.cuda.synchronize()
    cycles = _cycles_for(0.05)
    torch.cuda._sleep(cycles)
    done = torch.cuda.Event()
    done.record()
    t0 = time.perf_counter()
    fire()
    waited = time.perf_counter() - t0
    assert done.query(), f"{mechanism} returned before the device finished"
    assert waited > 0.02, f"{mechanism} waited {waited:.4f} s for a 0.05 s kernel"


@cuda
def test_an_idle_device_costs_each_mechanism_little():
    for m in recursion.PRICED_MECHANISMS:
        fire = di.cuda_mechanism(m, 0)
        fire()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fire()
        assert time.perf_counter() - t0 < 0.01, m


@cuda
def test_end_to_end_on_a_recorded_toy_prefill(tmp_path):
    """Record a toy prefill's trace, parse it, plan busy and idle variants
    on made-up costs, and run each with its injections."""
    toy = _Toy(n=256, n_layers=4, device="cuda")

    def run_prefill():
        with torch.no_grad():
            toy()

    path = record_prefill_trace(run_prefill, tmp_path / "trace.json", layers=toy.layers)
    cp = CodePath(arm="toy", gpu=torch.cuda.get_device_name(), implementation="eager")
    seq = parse_trace(path, cp, context_length=256, batch=1)
    observe = di.dispatch_observer(di.launching_names(seq), layers=toy.layers)
    calls = di.calibrate(run_prefill, seq, observe)
    assert len(calls) == 5              # the add, then one mm per layer
    host = {i: 1e-5 for i, op in enumerate(seq.ops) if op.kind is not SeqKind.SYNC}
    device = {i: 1e-4 for i, op in enumerate(seq.ops) if op.is_device}
    link = {i: 1e-5 for i, op in enumerate(seq.ops) if op.kind is SeqKind.COPY}
    base = recursion.close_prefill(to_operations(seq, host=host, device=device, link=link))
    costs = {m: recursion.MechanismCost(s_fixed=1e-5, bandwidth=1e10,
                                        loaded_bandwidth=1e10)
             for m in recursion.PRICED_MECHANISMS}
    for placement, mechanism, payload in ((BUSY, "synchronize", 0),
                                          (IDLE, "item", 0),
                                          (BUSY, "copy_to_host_pinned", 1 << 16)):
        variants = plan_variants(base, mechanism, placement, (0, 1, 2), costs, None,
                                 payload_bytes=payload)
        fire = di.cuda_mechanism(mechanism, payload)
        for s in di.schedules(variants, base, seq, calls, payload_bytes=payload):
            run = di.injected_prefill(run_prefill, s, fire, observe)
            assert run() == s.count
            torch.cuda.synchronize()


def test_the_observer_sees_the_same_calls_inside_the_null_pass():
    """Host costs come from T2.2's null pass under the observer: stacked with
    `NullDeviceMode`, the observer must see the calls a plain observed pass
    sees. On CPU, with `device_type="cpu"` and a null launch that does
    nothing."""
    toy = _Toy()
    launching = frozenset({"aten::add", "aten::mm"})
    observe = di.dispatch_observer(launching, device_type="cpu", layers=toy.layers)
    plain = []
    with torch.no_grad():
        observe(lambda: toy(), lambda n, l: plain.append(di.Call(n, l)))
    log = host_profile.PassLog()
    state = host_profile._State(time.perf_counter, time.perf_counter())
    device = host_profile.NullDevice(launch=lambda: None, device_type="cpu")
    mode = host_profile.NullDeviceMode(device, launching, log, state, time.perf_counter)
    with torch.no_grad(), mode:
        di.observed(lambda: toy(), observe, plain)()      # raises on a mismatch
    assert len(log.issues) == len(plain) == 3
