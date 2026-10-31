"""T2.2: the null-kernel pass and its assignment to a sequence's positions.

CPU tensors stand in for the device (`NullDevice(device_type="cpu")`), the
null launch is a fake that advances a scripted clock, and the arm's Python
advances the same clock to say how long it took. So every cost below is
worked by hand from the script: a gap is the time the arm's code advanced
between two launching calls, a launch is what the fake launch advances.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.components import host_profile as H
from attnbench.components import op_sequence as S
from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp
from attnbench.components.sync_tape import SyncTape, TapeRecorder
from attnbench.components.trace_recording import host_compute

CP = CodePath(arm="a", gpu="L4", implementation="torch")
LAUNCH = 1.0       # what one null launch advances the clock by


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


def device(clock, launch=LAUNCH):
    launches = []

    def go():
        launches.append(clock.now)
        clock.advance(launch)
    d = H.NullDevice(launch=go, device_type="cpu")
    d.launches = launches
    return d


def k(aten, layer=None):
    return SeqOp(SeqKind.KERNEL, layer=layer, aten_op=aten, api="cudaLaunchKernel",
                 kernels=("k",))


def sync(layer=None):
    return SeqOp(SeqKind.SYNC, layer=layer, mechanism="item", aten_op="aten::item")


def mark(layer=None):
    return SeqOp(SeqKind.HOST_COMPUTE, layer=layer)


def seq(*ops):
    return OpSequence(code_path=CP, context_length=4, batch=1, ops=tuple(ops), streams=(7,))


def run_pass(prefill, sq, clock, tape=None, **kw):
    return H.null_pass(prefill, tape or SyncTape("cpu"), sq, device=device(clock),
                       clock=clock, **kw)


W = torch.ones(3, 3)


# --------------------------------------------------------------------------
# One pass: gaps, launches, and what runs for real
# --------------------------------------------------------------------------

def test_a_gap_is_the_arms_time_between_launching_calls_and_a_launch_is_the_null_kernels():
    clock = Clock()
    x = torch.ones(3, 3)

    def prefill():
        clock.advance(3.0)
        y = torch.mm(x, W)
        clock.advance(5.0)
        torch.add(y, 1)
        clock.advance(2.0)
        torch.mm(x, W)

    sq = seq(k("aten::mm"), k("aten::add"), k("aten::mm"))
    log = run_pass(prefill, sq, clock)
    assert [(i.name, i.gap_s, i.launch_s) for i in log.issues] == \
        [("aten::mm", 3.0, 1.0), ("aten::add", 5.0, 1.0), ("aten::mm", 2.0, 1.0)]
    assert H.assign(sq, log) == {0: 4.0, 1: 6.0, 2: 3.0}


def test_the_device_work_is_not_done():
    clock = Clock()
    y = torch.zeros(3)
    a = torch.ones(3, 3)

    def prefill():
        y.add_(1)                         # in place: the output is the input
        prefill.out = torch.mm(a, W)      # allocated at its shape, not computed

    sq = seq(k("aten::add_"), k("aten::mm"))
    log = run_pass(prefill, sq, clock)
    assert torch.equal(y, torch.zeros(3))            # add_ never ran
    assert prefill.out.shape == (3, 3) and len(log.issues) == 2


def test_calls_that_launch_nothing_run_for_real_and_are_not_counted():
    clock = Clock()
    a = torch.arange(9.0).reshape(3, 3)

    def prefill():
        v = a.t().contiguous().view(9)    # views and metadata: not in the sequence
        prefill.v = v
        torch.mm(a, W)

    sq = seq(k("aten::mm"))
    log = run_pass(prefill, sq, clock)
    assert [i.name for i in log.issues] == ["aten::mm"]
    assert torch.equal(prefill.v, a.t().contiguous().view(9))   # ran for real


def test_a_call_on_host_tensors_is_not_device_work():
    clock = Clock()
    cuda_like = H.NullDevice(launch=lambda: clock.advance(1.0), device_type="cuda")
    a = torch.ones(2, 2)

    def prefill():
        prefill.r = torch.mm(a, a)        # CPU tensors, "cuda" is the device type

    log = H.null_pass(prefill, SyncTape("cuda"), seq(k("aten::mm")),
                      device=cuda_like, clock=clock, staged_device=torch.device("cpu"))
    assert log.issues == [] and torch.equal(prefill.r, a @ a)


def test_the_mode_does_not_leak_out_of_the_pass():
    clock = Clock()
    run_pass(lambda: torch.mm(W, W), seq(k("aten::mm")), clock)
    assert torch.equal(torch.mm(W, W), W @ W)


# --------------------------------------------------------------------------
# Several kernels from one call; several calls in one run
# --------------------------------------------------------------------------

def test_a_call_that_launches_two_kernels_prices_the_second_at_the_launch_floor():
    clock = Clock()

    def prefill():
        clock.advance(3.0)
        torch.mm(W, W)

    sq = seq(k("aten::mm"), k("aten::mm"))           # one call, two kernels
    log = run_pass(prefill, sq, clock)
    assert H.assign(sq, log) == {0: 4.0, 1: 1.0}     # gap 3 + launch 1; floor 1


def test_two_calls_in_a_run_of_four_operations_split_evenly():
    clock = Clock()

    def prefill():
        clock.advance(3.0)
        torch.mm(W, W)
        clock.advance(7.0)
        torch.mm(W, W)

    sq = seq(*[k("aten::mm")] * 4)
    assert H.assign(sq, run_pass(prefill, sq, clock)) == {0: 4.0, 1: 1.0, 2: 8.0, 3: 1.0}


def test_a_run_that_is_not_a_multiple_of_the_calls_is_a_mismatch():
    clock = Clock()

    def prefill():
        torch.mm(W, W)
        torch.mm(W, W)

    sq = seq(*[k("aten::mm")] * 3)
    with pytest.raises(H.ProfileMismatch, match="3 is not a multiple of 2"):
        H.assign(sq, run_pass(prefill, sq, clock))


def test_a_recorded_operation_the_pass_never_launched_is_a_mismatch():
    # e.g. a Triton kernel called from Python that was not registered as an
    # entry: it would have run for real, and the pass made no launching call.
    clock = Clock()
    sq = seq(k("aten::mm"), k(None))
    with pytest.raises(H.ProfileMismatch, match="an entry point.*no launching call"):
        H.assign(sq, run_pass(lambda: torch.mm(W, W), sq, clock))


def test_a_launching_call_the_recording_does_not_have_is_a_mismatch():
    clock = Clock()

    def prefill():
        torch.mm(W, W)
        torch.mm(W, W)

    sq = seq(k("aten::mm"))
    # Inside the run: one recorded operation, two launching calls.
    with pytest.raises(H.ProfileMismatch, match="1 is not a multiple of 2"):
        H.assign(sq, run_pass(prefill, sq, clock))
    # After the sequence ends: mm, add, then another mm.
    def extra():
        torch.mm(W, W)
        torch.add(W, 1)
        torch.mm(W, W)
    sq3 = seq(k("aten::mm"), k("aten::add"))
    with pytest.raises(H.ProfileMismatch, match="1 launching call.*after the recorded"):
        H.assign(sq3, run_pass(extra, sq3, clock))
    # A different operator is not in the launching set and runs for real, so it
    # is invisible; the order check is on the operators the recording names.
    sq2 = seq(k("aten::mm"), k("aten::add"))

    def other():
        torch.mm(W, W)
        torch.mul(W, 2)                   # not recorded as a launcher
        torch.add(W, 1)
    assert H.assign(sq2, run_pass(other, sq2, clock)).keys() == {0, 1}


def test_calls_in_another_order_than_the_recording_are_a_mismatch():
    clock = Clock()

    def prefill():
        torch.add(W, 1)
        torch.mm(W, W)

    sq = seq(k("aten::mm"), k("aten::add"))
    with pytest.raises(H.ProfileMismatch, match="made no launching call there"):
        H.assign(sq, run_pass(prefill, sq, clock))


def test_the_same_operator_in_another_layer_is_another_run():
    clock = Clock()

    class Layer(torch.nn.Module):
        def forward(self, x):
            clock.advance(2.0)
            return torch.mm(x, W)

    layers = [Layer(), Layer()]

    def prefill():
        for m in layers:
            m(W)

    sq = seq(k("aten::mm", 0), k("aten::mm", 1))
    log = run_pass(prefill, sq, clock, layers=layers)
    assert [i.layer for i in log.issues] == [0, 1]
    assert H.assign(sq, log) == {0: 3.0, 1: 3.0}
    # Without the layer hooks both calls say layer None and the recording's
    # layers do not match.
    with pytest.raises(H.ProfileMismatch):
        H.assign(sq, run_pass(prefill, sq, clock))


def test_a_synchronisation_ends_a_run_but_identical_neighbours_are_divided_together():
    clock = Clock()
    tape = SyncTape("cpu")

    def prefill():
        clock.advance(3.0)
        torch.mm(W, W)
        clock.advance(4.0)
        torch.mm(W, W)

    sq = seq(k("aten::mm"), sync(), k("aten::mm"))
    host = H.assign(sq, run_pass(prefill, sq, clock, tape=tape))
    assert host == {0: 4.0, 2: 5.0}                  # no cost at the synchronisation
    assert 1 not in host


# --------------------------------------------------------------------------
# Host-side compute is measured directly and charged once
# --------------------------------------------------------------------------

def test_a_host_compute_region_costs_its_own_duration_and_leaves_the_gap():
    clock = Clock()

    def prefill():
        clock.advance(1.0)
        torch.mm(W, W)
        clock.advance(2.0)
        with host_compute():
            clock.advance(7.0)
        clock.advance(4.0)
        torch.add(W, 1)

    sq = seq(k("aten::mm"), mark(), k("aten::add"))
    log = run_pass(prefill, sq, clock)
    assert log.markers == [7.0]
    # The add's gap is 2 + 7 + 4 = 13 on the clock, less the region's 7.
    assert [i.gap_s for i in log.issues] == [1.0, 6.0]
    host = H.assign(sq, log)
    assert host == {0: 2.0, 1: 7.0, 2: 7.0}
    # Charged once: the marker's 7 plus the add's own 6 + 1 (not 7 + 7 + 7).
    assert sum(host.values()) == 16.0


def test_marker_counts_must_match_and_regions_must_pair():
    clock = Clock()
    sq = seq(k("aten::mm"), mark())
    with pytest.raises(H.ProfileMismatch, match="1 host-compute marker.*timed 0"):
        H.assign(sq, run_pass(lambda: torch.mm(W, W), sq, clock))

    def nested():
        with host_compute():
            with host_compute():
                pass
    with pytest.raises(H.NullPassError, match="inside another"):
        run_pass(nested, seq(), clock)

    def open_region():
        host_compute().__enter__()
    with pytest.raises(H.NullPassError, match="never closed"):
        run_pass(open_region, seq(), clock)


def test_record_function_is_restored_even_when_the_prefill_raises():
    from torch.autograd.profiler import record_function
    before = (record_function.__enter__, record_function.__exit__)

    def boom():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        run_pass(boom, seq(), Clock())
    assert (record_function.__enter__, record_function.__exit__) == before


# --------------------------------------------------------------------------
# Entry points that are not aten calls
# --------------------------------------------------------------------------

class FakeKernelModule:
    def __init__(self):
        self.ran = 0

    def kern(self, q):
        self.ran += 1
        return q * 2


def test_an_entry_point_is_nulled_while_the_pass_runs_and_restored_after():
    clock = Clock()
    mod = FakeKernelModule()
    original = mod.kern
    entry = H.NullEntry(mod, "kern", lambda q: torch.empty_like(q))

    def prefill():
        clock.advance(2.0)
        prefill.out = mod.kern(W)

    sq = seq(k(None))
    log = run_pass(prefill, sq, clock, entries=[entry])
    assert mod.ran == 0 and prefill.out.shape == W.shape      # the real kernel never ran
    assert H.assign(sq, log) == {0: 3.0}
    assert mod.kern == original and mod.kern(W)[0, 0] == 2     # restored
    assert mod.ran == 1


def test_an_entry_is_restored_when_the_prefill_raises():
    mod = FakeKernelModule()
    original = mod.kern

    def boom():
        mod.kern(W)
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        run_pass(boom, seq(k(None)), Clock(),
                 entries=[H.NullEntry(mod, "kern", lambda q: torch.empty_like(q))])
    assert mod.kern == original


def test_an_unregistered_entry_runs_for_real_and_is_reported_as_missing():
    clock = Clock()
    mod = FakeKernelModule()
    sq = seq(k(None))
    log = run_pass(lambda: mod.kern(W), sq, clock)            # no entry registered
    assert mod.ran == 1
    with pytest.raises(H.ProfileMismatch, match="an entry point.*no launching call"):
        H.assign(sq, log)


# --------------------------------------------------------------------------
# Values read from the tape; shapes the dispatcher cannot give
# --------------------------------------------------------------------------

def test_a_value_the_prefill_reads_comes_from_the_tape_not_the_nulled_output():
    clock = Clock()
    x = torch.tensor([3.0])

    def prefill():
        y = torch.mm(torch.ones(1, 1), torch.ones(1, 1)) * 0 + x    # a value
        prefill.v = y.sum().item()

    with TapeRecorder("cpu") as rec:
        prefill()
    assert prefill.v == 3.0
    sq = seq(k("aten::mm"))
    log = run_pass(prefill, sq, clock, tape=rec.tape)
    assert prefill.v == 3.0                       # served from the tape; mm was nulled
    assert len(log.issues) == 1


def test_an_op_with_no_meta_implementation_is_reported_by_name():
    # A data-dependent output size cannot be given without running the call.
    # (The tape turns nonzero into nonzero_static before this point; this is
    # the dispatcher-level refusal for whatever it does not.)
    with pytest.raises(H.NullPassError, match="aten::nonzero has no output on meta"):
        H._null_outputs(torch.ops.aten.nonzero.default, (torch.ones(3),), {}, "cpu")


def test_a_cuda_graph_launch_is_refused_before_anything_runs():
    g = SeqOp(SeqKind.GRAPH, layer=0, api="cudaGraphLaunch", kernels=("a", "b"))
    ran = []
    with pytest.raises(H.ProfileMismatch, match="CUDA graph"):
        run_pass(lambda: ran.append(1), seq(g), Clock())
    assert ran == []


# --------------------------------------------------------------------------
# Repetitions, the record, and the hand-off to the recursion
# --------------------------------------------------------------------------

def test_the_profile_is_the_per_position_median_over_repetitions():
    clock = Clock()
    gaps = iter([3.0, 9.0, 5.0, 100.0])        # one warm-up, then three reps

    def prefill():
        clock.advance(next(gaps))
        torch.mm(W, W)

    sq = seq(k("aten::mm"))
    p = H.profile(prefill, SyncTape("cpu"), sq, device=device(clock), clock=clock,
                  reps=3, warmup=1)
    # warm-up 3 is dropped; reps 9, 5, 100 -> host 10, 6, 101; median 10.
    assert p.reps_s == {0: (10.0, 6.0, 101.0)}
    assert p.host_s == {0: 10.0} and p.reps == 3 and p.launch_floor_s == 1.0
    assert p.total_s == 10.0
    with pytest.raises(ValueError, match="reps"):
        H.profile(prefill, SyncTape("cpu"), sq, device=device(clock), reps=0)
    with pytest.raises(ValueError, match="warmup"):
        H.profile(prefill, SyncTape("cpu"), sq, device=device(clock), reps=1, warmup=-1)


def test_a_profile_is_for_one_sequence_and_round_trips():
    clock = Clock()
    sq = seq(k("aten::mm"), k("aten::add"))

    def prefill():
        clock.advance(2.0)
        torch.mm(W, W)
        clock.advance(1.0)
        torch.add(W, 1)

    p = H.profile(prefill, SyncTape("cpu"), sq, device=device(clock), clock=clock, reps=2)
    assert H.host_costs(p, sq) == {0: 3.0, 1: 2.0}
    other = seq(k("aten::mm"), k("aten::mul"))
    with pytest.raises(H.ProfileMismatch, match="another sequence"):
        H.host_costs(p, other)
    back = H.from_record(__import__("json").loads(H.dumps(p)))
    assert back == p
    with pytest.raises(H.ProfileMismatch, match="schema"):
        H.from_record({**H.to_record(p), "schema": "x/0"})


def test_the_costs_are_exactly_what_to_operations_asks_for():
    clock = Clock()
    sq = seq(k("aten::mm"), sync(), mark(), k("aten::add"))

    def prefill():
        clock.advance(2.0)
        torch.mm(W, W)
        with host_compute():
            clock.advance(5.0)
        clock.advance(1.0)
        torch.add(W, 1)

    p = H.profile(prefill, SyncTape("cpu"), sq, device=device(clock), clock=clock,
                  reps=1, warmup=0)
    host = H.host_costs(p, sq)
    assert set(host) == {0, 2, 3}                # not the synchronisation
    ops = S.to_operations(sq, host=host, device={0: 10.0, 3: 20.0}, link={})
    # The marker's 5 is added to the next operation's host cost: 1 + 1 + 5.
    assert [(o.kind.value, o.host) for o in ops] == [("kernel", 3.0), ("sync", 0.0),
                                                     ("kernel", 7.0)]


def test_runs_break_at_a_synchronisation_and_a_marker():
    sq = seq(k("aten::mm"), k("aten::mm"), sync(), k("aten::mm"), mark(), k("aten::add"),
             k("aten::add", 1))
    runs = H.runs_of(sq)
    assert [(r.name, r.layer, r.positions) for r in runs] == [
        ("aten::mm", None, (0, 1)), ("aten::mm", None, (3,)), ("aten::add", None, (5,)),
        ("aten::add", 1, (6,))]
    merged = H._merge_equal_neighbours(runs)
    assert [r.positions for r in merged] == [(0, 1, 3), (5,), (6,)]


# --------------------------------------------------------------------------
# Details the mutants found
# --------------------------------------------------------------------------

def test_each_gap_loses_only_the_host_compute_that_fell_inside_it():
    clock = Clock()

    def prefill():
        torch.mm(W, W)
        with host_compute():
            clock.advance(7.0)
        clock.advance(1.0)
        torch.add(W, 1)
        with host_compute():
            clock.advance(3.0)
        clock.advance(2.0)
        torch.mm(W, W)

    sq = seq(k("aten::mm"), mark(), k("aten::add"), mark(), k("aten::mm"))
    log = run_pass(prefill, sq, clock)
    # The add's gap is 7 + 1 on the clock less its region's 7; the last mm's is
    # 3 + 2 less its region's 3. A region carried into the next gap would
    # take 7 + 3 out of the second.
    assert [i.gap_s for i in log.issues][1:] == [1.0, 2.0]
    assert log.markers == [7.0, 3.0]


def test_the_launch_floor_is_the_median_not_the_slowest_launch():
    clock = Clock()
    costs = iter([1.0, 5.0, 1.0])

    def go():
        clock.advance(next(costs))
    dev = H.NullDevice(launch=go, device_type="cpu")

    def prefill():
        torch.mm(W, W)
        torch.add(W, 1)
        torch.mul(W, 2)

    sq = seq(k("aten::mm"), k("aten::mm"), k("aten::add"), k("aten::mul"))
    log = H.null_pass(prefill, SyncTape("cpu"), seq(k("aten::mm"), k("aten::add"),
                                                    k("aten::mul")),
                      device=dev, clock=clock)
    assert log.launches == [1.0, 5.0, 1.0]
    # mm is one call that launched two kernels: its second costs the median, 1.
    log.issues[0], log.issues[1], log.issues[2] = log.issues
    host = H.assign(sq, log)
    assert host[1] == 1.0


def test_a_sequence_with_a_graph_launch_is_refused_by_assign_too():
    g = SeqOp(SeqKind.GRAPH, layer=None, api="cudaGraphLaunch", kernels=("a",))
    with pytest.raises(H.ProfileMismatch, match="CUDA graph"):
        H.assign(seq(g), H.PassLog())


def test_an_in_place_call_returns_the_tensor_it_was_given():
    clock = Clock()
    y = torch.zeros(3)

    def prefill():
        prefill.r = y.add_(1)

    run_pass(prefill, seq(k("aten::add_")), clock)
    assert prefill.r is y


def test_an_output_that_is_an_input_is_that_input_not_a_new_allocation():
    # The Python methods return their own tensor either way; the dispatcher's
    # result is what must not become a second tensor, or a later call on the
    # alias would see an allocation that was never the input.
    y = torch.zeros(3)
    out = H._null_outputs(torch.ops.aten.add_.Tensor, (y, 1), {}, "cpu")
    assert out is y
    c = torch.zeros(3, 3)
    out = H._null_outputs(torch.ops.aten.mm.out, (W, W), {"out": c}, "cpu")
    assert out is c
    fresh = H._null_outputs(torch.ops.aten.mm.default, (W, W), {}, "cpu")
    assert fresh.shape == (3, 3) and fresh is not W


# --------------------------------------------------------------------------
# The block-sparse entry sits below the library's Python wrapper (t22)
# --------------------------------------------------------------------------

def test_the_block_sparse_entry_nulls_the_extension_and_keeps_the_wrapper(monkeypatch):
    """Shakedown 2's t22: the entry replaced `block_sparse_attn_func`, the
    library's Python wrapper, so its three synchronisations per layer never
    ran and the replay used 0 of 84 recorded interactions. The entry is now
    the extension call under the wrapper, so the wrapper's host work, and its
    synchronisations from the tape, are in the pass."""
    import sys
    import types
    from attnbench.components.sync_tape import TapeRecorder

    cuda = types.ModuleType("block_sparse_attn_cuda")
    cuda.ran = 0

    def fwd_block(q, *a, **kw):
        cuda.ran += 1
        return q * 2, None, None, None
    cuda.fwd_block = fwd_block
    lib = types.ModuleType("block_sparse_attn")

    def block_sparse_attn_func(q, mask):
        n = int(mask.sum().item())            # the wrapper's own synchronisation
        return sys.modules["block_sparse_attn_cuda"].fwd_block(q, n)[0]
    lib.block_sparse_attn_func = block_sparse_attn_func
    monkeypatch.setitem(sys.modules, "block_sparse_attn_cuda", cuda)
    monkeypatch.setitem(sys.modules, "block_sparse_attn", lib)

    def prefill():
        prefill.out = sys.modules["block_sparse_attn"].block_sparse_attn_func(
            W, torch.tensor([1.0, 0.0]))
    with TapeRecorder("cpu") as rec:
        prefill()
    assert len(rec.tape.entries) == 1 and cuda.ran == 1

    clock = Clock()
    run_pass(prefill, seq(k(None)), clock, tape=rec.tape, entries=[H.block_sparse_entry()])
    assert cuda.ran == 1                          # the extension never ran in the pass
    assert prefill.out.shape == W.shape
