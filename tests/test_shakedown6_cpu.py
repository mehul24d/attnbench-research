"""The researcher's rulings of 1 Nov 2026 after Shakedown 5, on the real
classes (behind the stubbed GPU boundary, `attnbench.cpu_boundary`, where a
recording is needed):

  - host prices by a bare timer, for leaves and composites: at least 1000
    repetitions, the median, the timer's overhead subtracted, the
    instrumented/bare ratio recorded (`components/bare_timer.py`);
  - the chain gate evaluated in that zoo, and a cell that fails it labelled
    with its fallback, not refused (oracle re-evaluation and fallback
    labelling);
  - a compiled variant's declared breaks are its code-path identity, and a
    compiled arm that compiles after its warm-up is refused (the recompile
    gate).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from attnbench import cpu_boundary as cb
from attnbench.components import bare_timer as bt
from attnbench.components import host_profile as hp
from attnbench.components import recorder
from attnbench.components.op_sequence import CodePath

pytestmark = pytest.mark.skipif(torch.cuda.is_available(),
                                reason="the CPU boundary is for a machine without CUDA")
CP = CodePath(arm="bare", gpu=cb.STUB_DEVICE_NAME, implementation="torch")


# --- the timer -----------------------------------------------------------------------

def test_each_repetition_is_synchronised_outside_the_timer_and_the_overhead_subtracted():
    log, t = [], [0]

    def clock():
        t[0] += 100
        log.append("clock")
        return t[0]

    def fn():
        t[0] += 900                      # the call itself: 900 ns
        log.append("call")

    s = bt.time_bare(fn, reps=3, overhead_ns=100.0, sync=lambda: log.append("sync"),
                     clock=clock)
    assert s == pytest.approx(900e-9)   # (100 + 900) per pair, less the 100 overhead
    # one untimed call, then each repetition: sync, clock, call, clock
    assert log[:3] == ["sync", "call", "sync"]
    assert log[3:8] == ["clock", "call", "clock", "sync", "clock"]
    assert log.count("call") == 4 and log[-1] == "sync"


def test_the_overhead_is_the_median_empty_pair():
    t = iter(range(0, 10_000, 1))
    assert bt.timer_overhead_ns(5, clock=lambda: next(t)) == 1.0


@pytest.mark.parametrize("reps", [0, 999, True, 1000.0])
def test_fewer_than_1000_repetitions_are_refused(reps):
    with pytest.raises(bt.BareTimerError, match="at least 1000"):
        bt.Pricer(reps=reps, sync=lambda: None)


def test_repetitions_never_change_the_prefills_own_tensors():
    x = torch.ones(4)
    fn = bt.reissue(torch.ops.aten.add_.Tensor, (x, torch.ones(4)), {})
    for _ in range(5):
        fn()
    assert torch.equal(x, torch.ones(4))       # the in-place target was cloned
    out = torch.zeros(4)
    bt.reissue(torch.ops.aten.add.out, (torch.ones(4), torch.ones(4)), {"out": out})()
    assert torch.equal(out, torch.zeros(4))


def test_the_price_is_cached_by_signature_not_by_call():
    p = bt.Pricer(reps=3, sync=lambda: None, min_reps=1)
    a, b, c = torch.ones(8), torch.zeros(8), torch.ones(9)
    p.call(torch.ops.aten.mul.Tensor, (a, a), {})
    p.call(torch.ops.aten.mul.Tensor, (b, b), {})       # same shapes: cached
    assert len(p.prices) == 1
    p.call(torch.ops.aten.mul.Tensor, (c, c), {})       # another shape
    assert len(p.prices) == 2


def test_a_call_the_timer_cannot_reissue_is_named():
    p = bt.Pricer(reps=3, sync=lambda: None, min_reps=1)
    with pytest.raises(bt.BareTimerError, match="an entry point could not be re-issued"):
        p.entry(None, lambda: 1 / 0, (), {})


# --- leaves and composites -------------------------------------------------------------

def _closure(fn):
    return bt.price_closure(fn, bt.Pricer(reps=3, sync=lambda: None, min_reps=1), "cpu")


def test_a_composite_is_seen_with_its_children_and_a_leaf_alone():
    x = torch.randn(16, 16)
    (copy,) = _closure(lambda: x.to(torch.float64))
    assert copy.name == "aten::_to_copy" and copy.children == ("aten::copy_",)
    assert copy.chain_s is not None
    (mm,) = _closure(lambda: torch.mm(x, x))
    assert mm.name == "aten::mm" and mm.children == () and mm.chain_s is None


def test_a_redispatch_to_the_call_itself_is_not_a_child():
    x = torch.randn(16, 16)
    (mean,) = _closure(lambda: x.mean())
    assert mean.name == "aten::mean" and "aten::mean" not in mean.children


def test_a_boolean_index_put_is_a_composite_over_its_impl():
    def put():
        y = torch.randn(64)
        y[y > 0] = 0.0
    put_ = [c for c in _closure(put) if c.name == "aten::index_put_"]
    assert put_ and put_[0].children == ("aten::_index_put_impl_",)


def test_a_number_in_a_tensor_argument_runs_as_a_leaf():
    assert bt._below_python(torch.ops.aten.mul.Tensor, (torch.ones(2), 1.0), {}) is None
    assert bt._below_python(torch.ops.aten.mul.Tensor, (torch.ones(2), torch.ones(2)), {})
    assert bt._below_python(torch.ops.aten.arange.default, (4,), {}) is None   # a factory


# --- the bare pass on a recording --------------------------------------------------------

class Layer(torch.nn.Module):
    """A copy to another dtype (a composite: `_to_copy` over `copy_`), a
    multiply and an add (leaves)."""

    def forward(self, x):
        y = x.to(torch.float64) * 2
        return y.to(torch.float32) + 1


def _recorded(tmp_path):
    layers = [Layer(), Layer()]

    def prefill():
        x = torch.ones(64, device="cuda")
        for layer in layers:
            x = layer(x)
    with cb.installed(xattn=False):
        rec = recorder.record_arm_prefill(prefill, tmp_path, code_path=CP, context_length=64,
                                          batch=1, prompt_id="p", layers=layers)
        null = hp.profile(prefill, rec.tape, rec.sequence,
                          device=hp.NullDevice(launch=lambda: None, device_type="cpu"), reps=2,
                          layers=layers)
        bp = bt.bare_pass(prefill, rec.tape, rec.sequence,
                          device=hp.NullDevice(launch=lambda: None, device_type="cpu"),
                          pricer=bt.Pricer(reps=5, sync=lambda: None, min_reps=1),
                          layers=layers)
    return rec, null, bp


def test_bare_prices_cover_every_position_and_each_call_keeps_its_price(tmp_path):
    rec, null, bp = _recorded(tmp_path)
    prof, record = bt.bare_profile(rec.sequence, bp, null, aggregate_tol=1.0)
    assert set(prof.host_s) == set(null.host_s) and prof.pricing == bt.BARE_TIMER
    assert prof.sequence_sha256 == null.sequence_sha256
    for row in record["calls"]:
        n = row["positions"]
        got = sum(prof.host_s[p] for p in range(row["position"], row["position"] + n))
        assert got == pytest.approx(row["bare_s"])
    routes = {r["operator"]: r["route"] for r in record["calls"]}
    assert routes["aten::_to_copy"] == bt.CHAIN_ROUTE
    assert routes["aten::mul"] == bt.LEAF and routes["aten::add"] == bt.LEAF
    assert record["reps"] == 5 and record["chain_gate"]["passed"] is True
    ratio = record["instrumented_over_bare"]
    assert ratio["ratio"] == pytest.approx(ratio["instrumented_s"] / ratio["bare_s"])
    assert all(r["ratio"] is not None for r in record["calls"])
    # the record round-trips with its pricing and routes
    back = hp.from_record(hp.to_record(prof))
    assert back.pricing == bt.BARE_TIMER and back.price_route == prof.price_route


def test_a_failed_chain_gate_falls_back_to_each_composites_own_time_labelled(tmp_path):
    rec, null, bp = _recorded(tmp_path)
    # the composites' leaves now account for twice their direct time
    bp.calls = [dataclasses.replace(c, chain_s=2 * c.direct_s + 1e-3)
                if c.chain_s is not None else c for c in bp.calls]
    prof, record = bt.bare_profile(rec.sequence, bp, null, aggregate_tol=0.10)
    g = record["chain_gate"]
    assert g["passed"] is False and g["fallback"] == bt.FALLBACK_LABEL
    copies = [r for r in record["calls"] if r["operator"] == "aten::_to_copy"]
    assert copies and all(r["route"] == bt.DIRECT_FALLBACK for r in copies)
    assert all(r["bare_s"] == r["direct_s"] for r in copies)
    assert bt.gate_summary(record)["fallback"] == bt.FALLBACK_LABEL      # labelled, not refused


def test_a_call_that_launches_beside_its_children_is_priced_as_a_leaf(tmp_path):
    # `mul` launched its own kernel (its position is costed `null_pass`); give
    # it children, as `addmm` has its bias copy: its chain cannot add up to
    # it, so it is no composite, in the routes or the gate.
    rec, null, bp = _recorded(tmp_path)
    bp.calls = [dataclasses.replace(c, chain_s=c.direct_s / 10, children=("aten::copy_",))
                if c.name == "aten::mul" else c for c in bp.calls]
    _, record = bt.bare_profile(rec.sequence, bp, null, aggregate_tol=0.10)
    muls = [r for r in record["calls"] if r["operator"] == "aten::mul"]
    assert muls and all(r["route"] == bt.LEAF and r["bare_s"] == r["direct_s"] for r in muls)
    assert record["operators"]["aten::mul"]["composite"] is False
    assert record["chain_gate"]["composite_calls"] == \
        sum(1 for r in record["calls"] if r["operator"] == "aten::_to_copy")


def test_a_gate_over_no_composite_has_no_verdict(tmp_path):
    rec, null, bp = _recorded(tmp_path)
    bp.calls = [dataclasses.replace(c, chain_s=None) for c in bp.calls]
    _, record = bt.bare_profile(rec.sequence, bp, null, aggregate_tol=0.10)
    assert record["chain_gate"]["passed"] is None and record["chain_gate"]["fallback"] is None


def test_a_bare_pass_that_met_other_calls_is_refused(tmp_path):
    rec, null, bp = _recorded(tmp_path)
    bp.calls[1] = dataclasses.replace(bp.calls[1], name="aten::sub")
    with pytest.raises(hp.ProfileMismatch, match="bare pass's was to aten::sub"):
        bt.bare_profile(rec.sequence, bp, null, aggregate_tol=0.10)
    rec, null, bp = _recorded(tmp_path / "again")
    bp.calls.pop()
    with pytest.raises(hp.ProfileMismatch, match="launching calls"):
        bt.bare_profile(rec.sequence, bp, null, aggregate_tol=0.10)


def test_the_floor_goes_to_each_further_position_and_the_rest_to_the_first():
    assert bt._spread(10.0, [3, 4, 5], 1.0) == {3: 8.0, 4: 1.0, 5: 1.0}
    assert bt._spread(1.0, [3, 4], 5.0) == {3: 0.5, 4: 0.5}         # never above an equal share
    assert bt._spread(2.0, [7], 1.0) == {7: 2.0}


# --- declared breaks and the recompile gate (T2.21) ------------------------------------------

from attnbench.backends import compiled as C  # noqa: E402


XAttentionBackend = type("XAttentionBackend", (), {})
BlockSparseAttention = type("BlockSparseAttention", (), {})
KERNEL = C.GraphBreak("Attempted to call function marked as skipped",
                      ("xattention.py:316 in forward_and_mask",))


def test_a_declared_break_is_matched_by_reason_file_and_function_not_by_line():
    moved = C.GraphBreak(KERNEL.reason, ("xattention.py:401 in forward_and_mask",))
    C.check_declared([C.LayerBreaks(0, 3, (KERNEL,)), C.LayerBreaks(1, 3, (moved,))],
                     XAttentionBackend(), "default")
    C.check_declared([C.LayerBreaks(0, 3, (KERNEL,))], XAttentionBackend(), "capture")


@pytest.mark.parametrize("breaks, why", [
    ((), "declared break missing"),
    ((KERNEL, C.GraphBreak("Dynamic shape operator",
                           ("block_sparse_attn_interface.py:111 in replace_ones_with_count",))),
     "undeclared break 'Dynamic shape operator'"),
    ((C.GraphBreak(KERNEL.reason, ("xattention.py:9 in forward",)),), "undeclared break"),
])
def test_a_report_that_is_not_the_declared_code_path_is_refused(breaks, why):
    with pytest.raises(C.CompiledVariantError, match=why):
        C.check_declared([C.LayerBreaks(0, 3, breaks)], XAttentionBackend(), "default")


def test_an_empty_report_shows_no_code_path():
    with pytest.raises(C.CompiledVariantError, match="no layer"):
        C.check_declared([], XAttentionBackend(), "default")


def test_the_declared_breaks_are_per_setting_and_inherited():
    sub = type("Sub", (BlockSparseAttention,), {})()
    assert [d.function for d in C.declared_breaks(sub, "default")] == \
        ["forward", "replace_ones_with_count"]
    assert C.declared_breaks(sub, "capture") == ()          # L4: no break under capture
    with pytest.raises(C.CompiledVariantError, match="undeclared"):
        C.check_declared([C.LayerBreaks(0, 2, (KERNEL,))], sub, "capture")


class _Arm:
    """An arm whose forward Dynamo compiles once per input shape."""

    class capability:
        name = "toy"

    def forward(self, q, k, v, cfg, mask=None):
        return q * 2 + k


@pytest.fixture
def fresh_dynamo():
    C.reset_compiled_state()
    yield C
    C.reset_compiled_state()


def test_a_compiled_arm_that_compiles_after_its_warm_up_is_refused(fresh_dynamo):
    c = C.CompiledBackend(_Arm(), "default", mode=None, dynamo_backend="eager")
    q = torch.ones(4)
    c.forward(q, q, q, None)
    c.forward(q, q, q, None)
    c.mark_warm()
    for _ in range(3):
        c.forward(q, q, q, None)
    assert c.check_recompiles() == 0
    q8 = torch.ones(4, 8)                        # a new guard set: it compiles again
    c.forward(q8, q8, q8, None)
    assert c.recompiles_after_warmup >= 1
    with pytest.raises(C.CompiledVariantError, match="after its warm-up"):
        c.check_recompiles()
    assert c.identity == {"variant": "toy@compile-default", "declared_breaks": []}


def test_the_gate_needs_the_warm_up_marked(fresh_dynamo):
    c = C.CompiledBackend(_Arm(), "default", mode=None, dynamo_backend="eager")
    with pytest.raises(C.CompiledVariantError, match="mark_warm"):
        c.check_recompiles()


class _Counters:
    observes_transients = True

    def read(self):
        from attnbench import clock_sampling as cs
        return cs.ClockSample(2040, 6250, 0, {"power": 0, "thermal": 0})


class _Ids:
    shape = (1, 64)

    def detach(self):
        return self

    def to(self, *a, **k):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return type("B", (), {"tobytes": lambda self: b"ids"})()


class _Compiled:
    """A compiled backend's gate surface; `after` compiles after warm-up."""
    setting = "default"

    def __init__(self, after):
        self.after, self.marked = after, False

    def mark_warm(self):
        self.marked = True

    def check_recompiles(self):
        if self.after:
            raise C.CompiledVariantError("toy compiled 1 time(s) after its warm-up (gate 0)")
        return 0

    @property
    def identity(self):
        return {"variant": "toy@compile-default", "declared_breaks": []}


def _time(backend):
    from attnbench import clock_sampling as cs
    from attnbench import ground_truth as gt
    from attnbench.precomputed_masks import prompt_digest
    gt.reset_process_compile_setting()
    controls = gt.Controls(1e9, 0, cs.BlockSampler(_Counters(), flag_bits=cs.RESEARCHER_FLAG_BITS,
                                                   interval_s=0.001))
    t = [0.0]

    def clock():
        t[0] += 0.01
        return t[0]
    arms = [gt.Arm("dense", gt.DENSE, object(), None),
            gt.Arm("toy@compile-default", gt.PRECOMPUTED, backend, None,
                   compile_setting="default", prompt_digest=prompt_digest(_Ids()),
                   builder_kind="gpu")]

    def density_of(arm, run):
        if arm.kind == gt.DENSE:
            return {"mask_density": None, "executed_density": 1.0}
        return {"mask_density": 0.25, "executed_density": 0.25}
    wrapped = type("W", (), {"run_measured": lambda self, *a, **k: None})()
    try:
        return gt.time_arms(wrapped, _Ids(), arms, reps=2, controls=controls, clock=clock,
                            density_of=density_of)
    finally:
        gt.reset_process_compile_setting()


def test_ground_truth_records_the_recompile_count_and_the_code_path_identity():
    be = _Compiled(after=False)
    out = _time(be)
    assert be.marked
    row = next(r for r in out["rows"] if r["arm"] == "toy@compile-default")
    assert row["recompiles_after_warmup"] == 0
    assert row["code_path_identity"]["variant"] == "toy@compile-default"
    assert "recompiles_after_warmup" not in out["rows"][0]                  # the eager control


def test_ground_truth_refuses_a_compiled_arm_that_recompiled_after_warm_up():
    with pytest.raises(C.CompiledVariantError, match="after its warm-up"):
        _time(_Compiled(after=True))
