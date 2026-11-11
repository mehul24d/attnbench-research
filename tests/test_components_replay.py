"""T2.4: device components by replay (`components.replay`).

No GPU: the graph is `EagerBackend`, the tape runs with `device_type="cpu"`
and the kernel is a stand-in module, so what is tested is the bookkeeping
around the capture (tape and bank held fixed, live masks kept, every call
accounted for), reading replayed nodes out of a Kineto-format trace,
matching them to sequence positions, and the reports. That a CUDA graph
captures an arm's prefill at all, and what its nodes cost, is the GPU
session's.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

from attnbench.components import mask_bank as B
from attnbench.components import replay as RP
from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp
from attnbench.components.replay import EagerBackend, Node, ReplayError
from attnbench.components.sync_tape import TapeMismatch, TapeRecorder

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch")


# --- capture --------------------------------------------------------------------

class Arm:
    """An estimator that reads back once and builds one mask per layer from
    the data, then calls the kernel with it."""

    def __init__(self, x, layers=2):
        self.x, self.layers, self.check_twice = x, layers, False
        self.mod = types.SimpleNamespace(seen=[])
        self.mod.block_sparse_attn_func = self._kernel

    def _kernel(self, q, k, v, cq, ck, hmt, si, mask, *a, **kw):
        self.mod.seen.append(mask.clone())
        return q

    def __call__(self):
        assert bool((self.x > -1).all())
        if self.check_twice:
            assert bool((self.x > -2).all())
        for layer in range(self.layers):
            m = (self.x.reshape(1, 1, 2, 2) + layer) > 2
            q = torch.zeros(2)
            self.mod.block_sparse_attn_func(q, q, q, None, None,
                                            torch.ones(1, dtype=torch.int32), None, m)


def recorded(arm):
    with TapeRecorder("cpu") as rec, B.MaskCapture(module=arm.mod, copy=True) as cap:
        arm()
    bank = B.from_capture(cap, code_path=CP, context_length=4096, batch=1, prompt_id="p")
    arm.mod.seen.clear()
    return rec.tape, bank


def test_capture_and_replay_run_on_the_banked_masks():
    arm = Arm(torch.arange(4.0))
    tape, bank = recorded(arm)
    be = EagerBackend()
    cap = RP.capture(arm, tape, device="cpu", masks=bank, backend=be, kernel_module=arm.mod)
    cap.replay()
    # warmup, the check graph's capture and replay, the capture, the replay
    assert be.replays == 2 and len(arm.mod.seen) == 5 * 2
    assert all(torch.equal(s, b) for s, b in zip(arm.mod.seen[-2:], bank.tensors()))
    assert RP.mask_agreement(cap)["identical"] is True


def test_the_kernel_gets_the_bank_even_when_the_estimator_disagrees():
    arm = Arm(torch.arange(4.0))
    tape, bank = recorded(arm)
    arm.x = torch.arange(4.0).flip(0)          # same shapes, other masks
    cap = RP.capture(arm, tape, device="cpu", masks=bank, backend=EagerBackend(),
                     kernel_module=arm.mod)
    assert all(torch.equal(s, b) for s, b in zip(arm.mod.seen[-2:], bank.tensors()))
    agree = RP.mask_agreement(cap)
    assert agree["identical"] is False and min(agree["per_call_agreement"]) < 1


def test_a_replay_with_fewer_kernel_calls_than_the_bank_is_refused():
    arm = Arm(torch.arange(4.0), layers=3)
    tape, bank = recorded(arm)
    arm.layers = 2
    with pytest.raises(ReplayError, match="made 2 kernel calls; 3 masks are banked"):
        RP.capture(arm, tape, device="cpu", masks=bank, backend=EagerBackend(),
                   kernel_module=arm.mod)


def test_a_replay_off_the_tape_is_refused():
    arm = Arm(torch.arange(4.0))
    tape, bank = recorded(arm)
    arm.check_twice = True                     # another path
    with pytest.raises(TapeMismatch, match="interaction 1"):
        RP.capture(arm, tape, device="cpu", masks=bank, backend=EagerBackend(),
                   kernel_module=arm.mod)


def test_a_tape_with_unhandled_interactions_is_refused_before_capture():
    x = torch.arange(4.0)
    with TapeRecorder("cpu") as rec:
        torch.unique(x)
    ran = []
    with pytest.raises(ReplayError, match=r"unhandled interactions at \[0\]"):
        RP.capture(lambda: ran.append(1), rec.tape, device="cpu", backend=EagerBackend())
    assert ran == []


def test_a_dense_arm_replays_without_a_bank_or_the_kernel_library():
    import importlib.util
    if importlib.util.find_spec("block_sparse_attn") is not None:
        pytest.skip("block_sparse_attn is installed here")
    x = torch.arange(4.0)

    def dense():
        assert bool((x > -1).all())
    with TapeRecorder("cpu") as rec:
        dense()
    cap = RP.capture(dense, rec.tape, device="cpu", backend=EagerBackend())
    assert cap.banked is None and RP.mask_agreement(cap) is None


def test_an_arm_that_reaches_the_kernel_with_no_bank_is_refused():
    arm = Arm(torch.arange(4.0))
    tape, _ = recorded(arm)
    with pytest.raises(ReplayError, match="made 2 kernel calls; 0 masks are banked"):
        RP.capture(arm, tape, device="cpu", backend=EagerBackend(), kernel_module=arm.mod)


def test_timing_takes_every_rep():
    be = EagerBackend()
    cap = RP.Captured(lambda: None, be, None, None)
    t = RP.time_replay(cap, 5)
    assert t["reps"] == 5 and be.replays == 5 and t["min_s"] <= t["median_s"] <= t["p90_s"]
    with pytest.raises(ValueError):
        RP.time_replay(cap, 0)


def test_profile_replay_writes_one_region(tmp_path):
    cap = RP.Captured(lambda: torch.ones(3).sum(), EagerBackend(), None, None)
    trace = json.loads(RP.profile_replay(cap, tmp_path / "t.json").read_text())
    names = [e.get("name") for e in trace["traceEvents"]]
    assert names.count(RP.REPLAY_REGION) == 1


# --- reading nodes ---------------------------------------------------------------

class Trace:
    def __init__(self):
        self.ev, self.corr, self.dev_t = [], 1, 0.0

    def x(self, cat, name, ts, dur, pid=1, tid=1, **args):
        self.ev.append({"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid,
                        "ts": ts, "dur": dur, "args": args})

    def launch(self, ts, *kernels, tid=1, dur=5.0):
        c = self.corr
        self.corr += 1
        self.x("cuda_runtime", "cudaLaunchKernel", ts, 1.0, tid=tid, correlation=c)
        for k in kernels:
            self.x("kernel", k, self.dev_t + 100, dur, pid=0, tid=7, correlation=c)
            self.dev_t += dur
        return c

    def dict(self):
        return {"traceEvents": self.ev}


def test_replay_nodes_reads_the_device_work_of_the_region_in_device_order():
    t = Trace()
    t.launch(1, "before_region")
    t.x("user_annotation", RP.REPLAY_REGION, 10, 100)
    t.launch(20, "a")
    t.launch(30, "g0", "g1", dur=2.0)          # a graph launch: two nodes
    t.launch(40, "other_thread", tid=2)
    t.x("user_annotation", RP.FLUSH, 49, 5)
    t.launch(50, "flush")
    t.launch(60, "b")
    t.launch(200, "after_region")
    nodes = RP.replay_nodes(t.dict(), exclude=RP.FLUSH)
    assert [n.name for n in nodes] == ["a", "g0", "g1", "b"]
    assert [n.duration_s for n in nodes] == pytest.approx([5e-6, 2e-6, 2e-6, 5e-6])
    assert nodes[1].correlation == nodes[2].correlation
    assert "flush" in [n.name for n in RP.replay_nodes(t.dict())]


def _flushed_trace(n, rng=None, nest=False):
    """n launches in the region; a flush annotation around some of them (optionally
    nested or overlapping), so some host calls fall inside one and some do not."""
    t = Trace()
    t.x("user_annotation", RP.REPLAY_REGION, 0, 10 * n + 100)
    for i in range(n):
        ts = 10 + 10 * i
        wraps = rng is None or rng.random() < 0.5
        if wraps:
            span = (rng.choice([4, 9, 25, 60]) if (rng and nest) else 8)
            t.x("user_annotation", RP.FLUSH, ts - 1, span)
        t.launch(ts, f"k{i}")
    return t


def _brute_replay_names(t, exclude):
    trace = RP.op_sequence.load_trace(t.dict())
    events = RP.op_sequence._events(trace)
    reg = [e for e in events if e.cat == "user_annotation" and e.name == RP.REPLAY_REGION][0]
    skips = [e for e in events if e.cat == "user_annotation" and e.name == exclude]
    keep = set()
    for e in events:
        if e.cat in RP.op_sequence._HOST_CATS and e.ts >= reg.ts and e.end <= reg.end:
            if not any(s.ts <= e.ts and e.end <= s.end for s in skips):
                keep.add(RP.op_sequence._int_arg(e.args, "correlation"))
    return sorted(n.name for n in RP.replay_nodes(t.dict())
                  if n.correlation in keep)


@pytest.mark.parametrize("nest", [False, True])
def test_replay_nodes_exclusion_agrees_with_a_scan_of_every_skip(nest):
    import random
    rng = random.Random(7)
    t = _flushed_trace(300, rng, nest=nest)
    got = sorted(n.name for n in RP.replay_nodes(t.dict(), exclude=RP.FLUSH))
    assert got == _brute_replay_names(t, RP.FLUSH)
    assert 0 < len(got) < 300


def test_replay_nodes_exclusion_is_not_quadratic_in_the_flushes():
    import time
    t = _flushed_trace(60000)
    t0 = time.perf_counter()
    nodes = RP.replay_nodes(t.dict(), exclude=RP.FLUSH)
    assert len(nodes) < 60000
    assert time.perf_counter() - t0 < 10     # a scan of every skip per call takes about a minute here


def test_replay_nodes_needs_exactly_one_region():
    t = Trace()
    with pytest.raises(ReplayError, match="0 'attnbench::replay' regions"):
        RP.replay_nodes(t.dict())
    t.x("user_annotation", RP.REPLAY_REGION, 0, 1)
    t.x("user_annotation", RP.REPLAY_REGION, 2, 1)
    with pytest.raises(ReplayError, match="2 'attnbench::replay' regions"):
        RP.replay_nodes(t.dict())


# --- matching, by structure only (researcher's ruling 1 of 6 Nov 2026) ----------------------

SEQ = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::mm", kernels=("gemm",)),
    SeqOp(SeqKind.COPY, layer=0, direction="HtoD", payload_bytes=8),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::nonzero", kernels=("scan", "write")),
    SeqOp(SeqKind.SYNC, layer=0, mechanism="item", payload_bytes=4),
    SeqOp(SeqKind.GRAPH, layer=1, kernels=("g0", "g1")),
    SeqOp(SeqKind.KERNEL, layer=1, aten_op="aten::mm", kernels=("gemm",)),
))


def _cat(name):
    return ("gpu_memset" if name.startswith("Memset (") else
            "gpu_memcpy" if name.startswith("Memcpy ") else "kernel")


def nodes(*pairs, stream=None):
    return [Node(_cat(n), n, d, None, stream) for n, d in pairs]


EXACT = (("gemm", 1.0), ("scan", 0.5), ("write", 0.25), ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0))


def test_a_complete_match_gives_every_device_position_its_time():
    m = RP.match(SEQ, nodes(*EXACT))
    assert m.complete and (m.n_expected, m.n_nodes) == (6, 6) and m.variant_matched == ()
    assert RP.device_costs(m) == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}
    assert RP.expected_nodes(SEQ)[1:3] == [(2, "scan"), (2, "write")]


def test_a_stand_in_of_another_kernel_count_leaves_its_position_without_a_cost():
    # nonzero_static's three kernels in place of nonzero's two
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("nz_a", 0.5), ("nz_b", 0.25), ("nz_c", 0.125),
                            ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    assert not m.complete
    assert m.unmatched_positions == (2,) and m.extra_nodes == ("nz_a", "nz_b", "nz_c")
    assert m.extra_s == pytest.approx(0.875)
    assert m.families_unmatched == {"kernel:aten::nonzero": 1}
    assert 2 not in m.device_s
    with pytest.raises(ReplayError, match="no device cost is taken from a different kernel"):
        RP.device_costs(m)


def test_a_name_or_launch_variant_that_differs_is_not_a_mismatch():
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("scan_v2", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1_renamed", 3.0), ("gemm", 4.0)))
    assert m.complete and m.variant_matched == (2, 4)
    assert RP.device_costs(m) == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}


def test_a_one_kernel_stand_in_for_one_kernel_is_matched_as_a_variant():
    # the rule's consequence: structure alone cannot tell this from a launch variant
    one = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
        SeqOp(SeqKind.KERNEL, aten_op="aten::eq", kernels=("eq",)),
        SeqOp(SeqKind.KERNEL, aten_op="aten::masked_fill_", kernels=("fill",)),
        SeqOp(SeqKind.KERNEL, aten_op="aten::mm", kernels=("gemm",))))
    m = RP.match(one, nodes(("eq", 1.0), ("index_put_kernel", 0.5), ("gemm", 4.0)))
    assert m.complete and m.variant_matched == (1,) and m.device_s[1] == 0.5


DTOD = "Memcpy DtoD (Device -> Device)"
SEQ_DCOPY = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, aten_op="aten::eq", kernels=("eq",)),
    SeqOp(SeqKind.DEVICE_COPY, aten_op="aten::copy_", direction="DtoD", kernels=(DTOD,)),
    SeqOp(SeqKind.KERNEL, aten_op="aten::mm", kernels=("gemm",))))


def test_a_graphs_copy_kernel_is_the_device_copy_it_runs():
    for name in ("memcpy32_post", "memcpy128"):
        m = RP.match(SEQ_DCOPY, nodes(("eq", 1.0), (name, 0.5), ("gemm", 4.0)))
        assert m.complete and m.variant_matched == (1,), name
    assert RP.activity_class("memcpy32_post", "kernel") == "device_copy:DtoD"
    assert RP.activity_class(DTOD) == "device_copy:DtoD"
    m = RP.match(SEQ_DCOPY, nodes(("eq", 1.0), ("copy_like_kernel", 0.5), ("gemm", 4.0)))
    assert m.unmatched_positions == (1,) and m.extra_nodes == ("copy_like_kernel",)


SEQ_MEMSET = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::addmm", kernels=("Memset (Device)", "gemm")),
    SeqOp(SeqKind.KERNEL, layer=1, aten_op="aten::addmm", kernels=("Memset (Device)", "gemm")),
))


def test_a_graph_memset_node_matches_the_eager_memset_whatever_its_memory_kind():
    # the graph's memset node has no allocation to name: "Memset (Unknown)"
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0),
                                   ("Memset (Unknown)", 0.2), ("gemm", 2.0)))
    assert m.complete and m.variant_matched == () and RP.device_costs(m) == {0: 1.1, 1: 2.2}


def test_a_stretch_of_another_class_or_count_is_unmatched():
    # a memset needs a memset: a kernel or a copy in its place is another class
    for other in ("kernel_x", "Memcpy DtoD (Device -> Device)"):
        m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0),
                                       (other, 0.2), ("gemm", 2.0)))
        assert m.unmatched_positions == (1,) and m.extra_nodes == (other,), other
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0), ("gemm", 2.0)))
    assert m.unmatched_positions == (1,) and not m.extra_nodes
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0), ("Memset (Unknown)", 0.2),
                                   ("gemm", 2.0), ("Memset (Unknown)", 0.3)))
    assert m.unmatched_positions == () and m.extra_nodes == ("Memset (Unknown)",)   # the name as seen


def test_a_stream_that_differs_in_order_of_first_use_is_a_mismatch():
    two = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7, 9), ops=(
        SeqOp(SeqKind.KERNEL, aten_op="aten::mm", kernels=("a",), stream=7),
        SeqOp(SeqKind.KERNEL, aten_op="aten::mm", kernels=("b",), stream=9),
        SeqOp(SeqKind.KERNEL, aten_op="aten::mm", kernels=("c",), stream=7)))
    same = [Node("kernel", n, 1.0, None, s) for n, s in (("a", 3), ("b", 5), ("c", 3))]
    assert RP.match(two, same).complete             # stream ids differ; their order of use does not
    one = [Node("kernel", n, 1.0, None, 3) for n in ("a", "b", "c")]
    assert RP.match(two, one).unmatched_positions == (1,)


def test_names_place_the_comparison_and_never_pair_a_stretch_whose_counts_differ():
    seq = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=tuple(
        SeqOp(SeqKind.KERNEL, aten_op=f"aten::op{k}", kernels=(n,))
        for k, n in enumerate(("gemm", "a", "b", "gemm2"))))
    m = RP.match(seq, nodes(("gemm", 1.0), ("x", 1.0), ("y", 1.0), ("z", 1.0), ("gemm2", 1.0)))
    assert m.unmatched_positions == (1, 2) and m.extra_nodes == ("x", "y", "z")


_CP = ("void at::native::{}<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::"
       "{{lambda()#3}}::operator()() const::{{lambda()#1}}::operator()() const::"
       "{{lambda({})#1}}>(at::TensorIteratorBase&, int const&)::{{lambda(int)#1}})")
COPY_EW = _CP.format("elementwise_kernel<128, 4, at::native::gpu_kernel_impl", "unsigned char")
COPY_UNROLLED = _CP.format("unrolled_elementwise_kernel", "unsigned char")

SEQ_COPY = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::eq", kernels=("eq",)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::copy_", kernels=(COPY_EW,)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::mm", kernels=("gemm",)),
))


def test_an_eager_host_device_copy_is_never_a_position_and_never_splits_a_stretch():
    # an eager run's copies are T2.5's, not device positions (the debug session's
    # oracle and XAttention originals: every extra node is one)
    eager = nodes(("eq", 1.5), ("Memcpy DtoH (Device -> Pinned)", 0.1), (COPY_UNROLLED, 0.75),
                  ("Memcpy HtoD (Pageable -> Device)", 0.2), ("gemm", 4.5))
    m = RP.match(SEQ_COPY, eager)
    assert m.unmatched_positions == () and m.variant_matched == (1,) and m.device_s[1] == 0.75
    assert m.extra_nodes == ("Memcpy DtoH (Device -> Pinned)", "Memcpy HtoD (Pageable -> Device)")
    assert m.extra_s == pytest.approx(0.3)

# --- reports -----------------------------------------------------------------------

def test_cache_effect_is_isolated_minus_in_order():
    r = RP.cache_effect(SEQ, {0: 1.0, 2: 1.0, 5: 2.0}, {0: 1.5, 2: 1.0, 4: 7.0, 5: 2.5})
    assert r["positions"] == 3 and r["missing_in_order"] == [4]
    assert (r["in_order_s"], r["isolated_s"], r["effect_s"]) == (4.0, 5.0, 1.0)
    assert r["isolated_over_in_order"] == pytest.approx(1.25)
    mm = r["families"]["kernel:aten::mm"]
    assert (mm["n"], mm["in_order_s"], mm["isolated_s"], mm["effect_s"]) == (2, 3.0, 4.0, 1.0)


def test_replay_vs_trace_and_the_gate_at_an_explicit_bound():
    trace = {0: 1.0, 2: 1.0, 4: 2.0, 5: 1.0}
    full = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                               ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    rep = RP.replay_vs_trace(SEQ, trace, {0: 1.05, 2: 1.0, 4: 2.0, 5: 1.05}, full)
    assert rep["families"]["kernel:aten::mm"]["relative_difference"] == pytest.approx(0.05)
    assert RP.check_passes(rep, 0.1) == (True, [])
    # ruling 2d (11 Nov 2026): a report with no match behind it is not a pass
    unfed = RP.replay_vs_trace(SEQ, trace, {0: 1.05, 2: 1.0, 4: 2.0, 5: 1.05})
    ok, why = RP.check_passes(unfed, 0.1)
    assert not ok and why == ["no structure evidence: the replay's match was not fed to the report"]
    nothing = RP.replay_vs_trace(SEQ, {}, {}, full)
    assert RP.check_passes(nothing, 0.1) == (False, ["no operation family was compared"])
    # Section 5.6: a family fails when its difference is above the bound; equal is not above
    # (binary-exact: 1.25 against 1.0 is 0.25)
    edge = RP.replay_vs_trace(SEQ, {0: 1.0}, {0: 1.25}, full)
    assert RP.check_passes(edge, 0.25)[0] and not RP.check_passes(edge, 0.2)[0]
    nan = dict(rep, families={"kernel:aten::mm": dict(rep["families"]["kernel:aten::mm"],
                                                      relative_difference=float("nan"))})
    assert not RP.check_passes(nan, 0.1)[0]
    ok, why = RP.check_passes(rep, 0.01)
    assert not ok and len(why) == 1 and why[0].startswith("kernel:aten::mm: relative difference 0.05")
    short = RP.replay_vs_trace(SEQ, trace, {0: 1.0, 2: 1.0, 5: 1.0}, full)
    assert short["positions_without_replay"] == [4]
    assert not RP.check_passes(short, 0.5)[0]
    m = RP.match(SEQ, nodes(("gemm", 1.0)))
    named = RP.replay_vs_trace(SEQ, trace, trace, m)
    ok, why = RP.check_passes(named, 0.5)
    assert not ok and why[0].startswith("the replay does not match the sequence by structure")
    for bad in (0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="fixed at lock"):
            RP.check_passes(rep, bad)


def test_a_family_with_no_trace_time_fails_the_gate():
    rep = RP.replay_vs_trace(SEQ, {0: 0.0}, {0: 1.0})
    assert rep["families"]["kernel:aten::mm"]["relative_difference"] is None
    assert not RP.check_passes(rep, 10.0)[0]


def test_the_banked_record_round_trips_and_refuses_an_incomplete_match():
    full = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                               ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    from attnbench.components.sync_tape import SyncTape, TapeEntry
    tape = SyncTape("cuda", [TapeEntry("nonzero", "nonzero", ((4,), "torch.bool"), 2)])
    timing = {"reps": 3, "median_s": 10.75, "min_s": 10.0, "p90_s": 11.0}
    rec = RP.to_record(seq=SEQ, prompt_id="p", m=full, timing=timing, agreement=None,
                       tape=tape, masks=None, isolated={0: 2.0})
    rec = json.loads(json.dumps(rec))
    assert RP.device_costs_from_record(rec) == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}
    assert rec["profiled_over_unprofiled"] == pytest.approx(1.0)
    assert rec["tape_substitutions"] == {"nonzero": 1} and rec["isolated_s"] == {"0": 2.0}
    part = RP.match(SEQ, nodes(("gemm", 1.0)))
    bad = RP.to_record(seq=SEQ, prompt_id="p", m=part, timing=None, agreement=None,
                       tape=tape, masks=None)
    with pytest.raises(ReplayError, match="no device costs"):
        RP.device_costs_from_record(bad)
    with pytest.raises(ReplayError, match="schema"):
        RP.device_costs_from_record({**rec, "schema": "attnbench.replay/0"})


# --- isolated replay -----------------------------------------------------------------

def test_isolation_flushes_and_synchronises_before_every_operator(tmp_path):
    log = []
    x = torch.arange(4.0)

    def prefill():
        assert bool((x > -1).all())
        (x * 2).sum()
    with TapeRecorder("cpu") as rec:
        prefill()
    path = RP.isolated_replay(prefill, rec.tape, tmp_path / "iso.json", device="cpu",
                              flush=lambda: log.append("flush"),
                              sync=lambda: log.append("sync"))
    # one sync before the profiler, (sync, flush, sync) per operator, one after
    ops = (len(log) - 2) // 3
    assert ops >= 2 and log == ["sync"] + ["sync", "flush", "sync"] * ops + ["sync"]
    trace = json.loads(path.read_text())
    names = [e.get("name") for e in trace["traceEvents"]]
    assert names.count(RP.REPLAY_REGION) == 1 and names.count(RP.FLUSH) == ops
    assert RP.replay_nodes(trace, exclude=RP.FLUSH) == []    # no device on a CPU trace


def test_isolated_replay_holds_the_bank(tmp_path):
    arm = Arm(torch.arange(4.0), layers=3)
    tape, bank = recorded(arm)
    RP.isolated_replay(arm, tape, tmp_path / "a.json", device="cpu", masks=bank,
                       flush=lambda: None, sync=lambda: None, kernel_module=arm.mod)
    assert all(torch.equal(s, b) for s, b in zip(arm.mod.seen, bank.tensors()))
    arm.layers = 2
    with pytest.raises(ReplayError, match="made 2 kernel calls; 3 masks are banked"):
        RP.isolated_replay(arm, tape, tmp_path / "b.json", device="cpu", masks=bank,
                           flush=lambda: None, sync=lambda: None, kernel_module=arm.mod)


def test_isolated_replay_holds_the_tape(tmp_path):
    x = torch.arange(4.0)
    with TapeRecorder("cpu") as rec:
        assert bool((x > -1).all())
    with pytest.raises(TapeMismatch, match="used 0 of 1"):
        RP.isolated_replay(lambda: None, rec.tape, tmp_path / "iso.json", device="cpu",
                           flush=lambda: None, sync=lambda: None)


# --- pricing and its labels (researcher's ruling 1 of 6 Nov 2026) --------------------------

GRAPH_SUB = nodes(("gemm", 1.0), ("nz_a", 0.5), ("nz_b", 0.25), ("nz_c", 0.125),
                  ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0))
ORIGINAL = nodes(("gemm", 1.5), ("Memcpy HtoD (Pageable -> Device)", 0.0625), ("scan", 0.625),
                 ("write", 0.375), ("g0", 2.5), ("g1", 3.5), ("gemm", 4.5))


def test_a_complete_graph_prices_every_position_from_the_graph():
    p = RP.price_positions(SEQ, RP.match(SEQ, nodes(*EXACT)), None)
    assert p.device_s == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}
    assert set(p.source.values()) == set(p.label.values()) == {RP.GRAPH}
    assert p.substitute_nodes == () and p.unpriced == () and p.mismatch_s == 0
    assert p.dropped == {RP.GRAPH: {"nodes": 0, "s": 0}}


def test_a_position_the_graph_does_not_hold_is_priced_from_the_original_and_labelled():
    p = RP.price_positions(SEQ, RP.match(SEQ, GRAPH_SUB), RP.match(SEQ, ORIGINAL))
    # position 2 from the isolated original (0.625 + 0.375); the rest from the graph
    assert p.device_s == {0: 1.0, 2: 1.0, 4: 5.0, 5: 4.0}
    assert p.source == {0: RP.GRAPH, 2: RP.ISOLATED_ORIGINAL, 4: RP.GRAPH, 5: RP.GRAPH}
    assert p.label == p.source
    assert p.counts() == {RP.GRAPH: 3, RP.ISOLATED_ORIGINAL: 1}
    assert p.substitute_nodes == ("nz_a", "nz_b", "nz_c")
    assert p.dropped == {RP.GRAPH: {"nodes": 3, "s": pytest.approx(0.875)},
                         RP.ISOLATED_ORIGINAL: {"nodes": 1, "s": pytest.approx(0.0625)}}
    assert (p.priced_s, p.mismatch_s) == (11.0, 1.0)


def test_a_variant_is_labelled_whichever_replay_priced_it():
    g = RP.match(SEQ, nodes(("gemm", 1.0), ("nz_a", 0.5), ("nz_b", 0.25), ("nz_c", 0.125),
                            ("g0", 2.0), ("g1_v", 3.0), ("gemm", 4.0)))
    o = RP.match(SEQ, nodes(("gemm", 1.5), ("scan_v", 0.625), ("write", 0.375),
                            ("g0", 2.5), ("g1", 3.5), ("gemm", 4.5)))
    p = RP.price_positions(SEQ, g, o)
    assert p.source == {0: RP.GRAPH, 2: RP.ISOLATED_ORIGINAL, 4: RP.GRAPH, 5: RP.GRAPH}
    assert p.label == {0: RP.GRAPH, 2: RP.VARIANT_DIFFERS, 4: RP.VARIANT_DIFFERS, 5: RP.GRAPH}
    assert p.label_counts() == {"graph:variant_differs": 1, "isolated_original:variant_differs": 1,
                                "graph:graph": 2}


MASK_SEQ = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::eq", kernels=("eq",)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::masked_fill_", kernels=("fill_masked",)),
    SeqOp(SeqKind.DEVICE_COPY, layer=0, aten_op="aten::copy_",
          kernels=("Memcpy DtoD (Device -> Device)",)),
))


def test_a_mask_write_matched_only_to_the_stand_ins_kernel_is_priced_from_the_original():
    # the graph serves `x[x == v] = s` with an index_put_: one kernel for one, another
    # name, 4x the eager masked_fill_ (T3.2, 8 Nov 2026); a copy kernel's variant stays
    g = RP.match(MASK_SEQ, nodes(("eq", 1.0), ("index_put", 4.0), ("memcpy32_post", 2.0)))
    assert g.complete and set(g.variant_matched) == {1, 2}
    assert RP.stand_in_positions(MASK_SEQ, g) == (1,) and RP.needs_original(MASK_SEQ, g)
    with pytest.raises(ReplayError, match="which was not run"):
        RP.price_positions(MASK_SEQ, g, None)
    o = RP.match(MASK_SEQ, nodes(("eq", 1.5), ("fill_masked", 1.0),
                                 ("Memcpy DtoD (Device -> Device)", 2.5)))
    p = RP.price_positions(MASK_SEQ, g, o)
    assert p.device_s == {0: 1.0, 1: 1.0, 2: 2.0}
    assert p.source == {0: RP.GRAPH, 1: RP.ISOLATED_ORIGINAL, 2: RP.GRAPH}
    assert p.label[2] == RP.VARIANT_DIFFERS
    assert not RP.needs_original(MASK_SEQ, RP.match(MASK_SEQ, nodes(
        ("eq", 1.0), ("fill_masked", 1.0), ("Memcpy DtoD (Device -> Device)", 2.0))))


def test_without_the_original_replay_a_position_the_graph_misses_is_refused():
    with pytest.raises(ReplayError, match="which was not run"):
        RP.price_positions(SEQ, RP.match(SEQ, GRAPH_SUB), None)


def test_a_position_neither_replay_matches_is_unpriced_and_not_refused():
    o = RP.match(SEQ, nodes(("gemm", 1.5), ("other", 0.625), ("g0", 2.5), ("g1", 3.5),
                            ("gemm", 4.5)))
    p = RP.price_positions(SEQ, RP.match(SEQ, GRAPH_SUB), o)
    assert p.unpriced == (2,) and 2 not in p.device_s
    sm = p.summary(SEQ)
    assert sm["unpriced"] == [2] and sm["unpriced_families"] == ["kernel:aten::nonzero"]


def test_the_summary_is_what_the_gate_reads():
    p = RP.price_positions(SEQ, RP.match(SEQ, GRAPH_SUB), RP.match(SEQ, ORIGINAL))
    sm = json.loads(json.dumps(p.summary(SEQ)))
    assert (sm["priced_s"], sm["mismatch_s"], sm["positions"]) == (11.0, 1.0, 4)
    assert sm["labels"] == {"graph:graph": 3, "isolated_original:isolated_original": 1}
    assert sm["isolated_substitutions"] == 1          # nonzero is a tape substitution
    assert sm["isolated_families"] == {"kernel:aten::nonzero": {"n": 1, "s": 1.0}}
    assert sm[RP.EXTRA_NODE_DROPPED]["graph"]["nodes"] == 3


def test_the_record_carries_each_position_and_its_cost_source():
    from attnbench.components.sync_tape import SyncTape, TapeEntry
    tape = SyncTape("cuda", [TapeEntry("nonzero", "nonzero", ((4,), "torch.bool"), 2)])
    g = RP.match(SEQ, GRAPH_SUB)
    p = RP.price_positions(SEQ, g, RP.match(SEQ, ORIGINAL))
    rec = json.loads(json.dumps(RP.to_record(seq=SEQ, prompt_id="p", m=g, timing=None,
                                             agreement=None, tape=tape, masks=None,
                                             pricing=p)))
    assert rec["schema"] == "attnbench.replay/2" and rec["complete"] is False
    assert RP.device_costs_from_record(rec) == {0: 1.0, 2: 1.0, 4: 5.0, 5: 4.0}
    assert RP.cost_sources_from_record(rec)[2] == RP.ISOLATED_ORIGINAL
    assert rec["substitute_nodes"] == ["nz_a", "nz_b", "nz_c"]
    with pytest.raises(ReplayError, match="cost sources differ"):
        RP.device_costs_from_record({**rec, "cost_source": {"0": RP.GRAPH}})
    unpriced = RP.to_record(seq=SEQ, prompt_id="p", m=g, timing=None, agreement=None,
                            tape=tape, masks=None)
    with pytest.raises(ReplayError, match="no device costs"):
        RP.device_costs_from_record(unpriced)


def test_a_version_one_record_with_a_complete_match_is_still_read():
    full = RP.match(SEQ, nodes(*EXACT))
    from attnbench.components.sync_tape import SyncTape
    rec = RP.to_record(seq=SEQ, prompt_id="p", m=full, timing=None, agreement=None,
                       tape=SyncTape("cuda", []), masks=None)
    old = {k: v for k, v in rec.items() if k not in ("priced_s", "cost_source", "substitute_nodes")}
    old["schema"] = "attnbench.replay/1"
    assert RP.device_costs_from_record(old) == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}
    assert set(RP.cost_sources_from_record(old).values()) == {RP.GRAPH}


def test_the_original_isolated_replay_runs_the_operators_not_the_tape(tmp_path):
    x = torch.arange(4.0)
    seen = []

    def prefill():
        seen.append(int(x.sum()))
    with TapeRecorder("cpu") as rec:
        prefill()
    x += 1                                          # the live value now differs
    kw = dict(device="cpu", flush=lambda: None, sync=lambda: None)
    RP.isolated_replay(prefill, rec.tape, tmp_path / "t.json", **kw)
    RP.isolated_replay(prefill, rec.tape, tmp_path / "o.json", substitute=False, **kw)
    assert seen == [6, 6, 10]                       # recorded, served from the tape, live


# --- the aggregate gate --------------------------------------------------------------------

def _cell(arm, priced, mismatch, unpriced=()):
    return {"arm": arm, "pricing": {"priced_s": priced, "mismatch_s": mismatch,
                                    "unpriced": list(unpriced)}}


def test_the_gate_is_each_arms_share_over_its_cells_not_each_cells():
    g = RP.pricing_gate([_cell("oracle", 1.0, 0.08), _cell("oracle", 9.0, 0.09),
                         _cell("xattn", 1.0, 0.04), _cell("xattn", 1.0, 0.08)], 0.05)
    assert g["oracle"]["share"] == pytest.approx(0.017) and g["oracle"]["validated"]
    assert "label" not in g["oracle"] and g["oracle"]["cells"] == 2
    x = g["xattn"]
    assert x["share"] == pytest.approx(0.06) and not x["validated"]
    assert x["label"] == RP.PRICING_NOT_VALIDATED and x["why"] == ["mismatch share 0.0600 above 0.05"]


def test_the_gate_passes_at_its_bound_and_refuses_an_unpriced_position_at_any_share():
    g = RP.pricing_gate([_cell("a", 1.0, 0.05), _cell("b", 1.0, 0.0, unpriced=(3, 4))], 0.05)
    assert g["a"]["validated"]
    assert not g["b"]["validated"] and g["b"]["why"] == ["2 device positions neither replay matches"]
    assert not RP.pricing_gate([_cell("c", 0.0, 0.0)], 0.05)["c"]["validated"]
    for bad in (0, 1, -0.1, float("nan")):
        with pytest.raises(ValueError, match="fixed at lock"):
            RP.pricing_gate([], bad)


# --- the debug session's banked cells (2 Nov 2026, not T3.1 data) ---------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# label counts, graph extra nodes, original extra nodes (every one a host-device copy)
BANKED_PRICING = {
    "oracle_d0.25_L2048_b1": ({"graph:graph": 1924, "graph:variant_differs": 112,
                               "isolated_original:isolated_original": 112}, 308, 84, 0.0017),
    "xattention_dcal_L2048_b1": ({"graph:graph": 4108, "graph:variant_differs": 308,
                                  "isolated_original:isolated_original": 336}, 756, 224, 0.0097),
}


def _banked_cell(cell):
    import gzip
    d = json.loads(gzip.decompress((FIXTURES / f"dbg_{cell}_pricing.json.gz").read_bytes()))
    names = d["names"]
    load = lambda rows: [Node(c, names[n], dur, corr, st, nb) for c, n, dur, corr, st, nb in rows]
    from attnbench.components import op_sequence
    return op_sequence.from_record(d["sequence"]), load(d["graph"]), load(d["isolated_original"])


@pytest.mark.parametrize("cell", sorted(BANKED_PRICING))
def test_every_position_of_a_banked_sparse_cell_is_priced_within_the_gate(cell):
    labels, g_extra, o_extra, share = BANKED_PRICING[cell]
    seq, graph, orig = _banked_cell(cell)
    g, o = RP.match(seq, graph), RP.match(seq, orig)
    assert o.unmatched_positions == ()                # the original holds every position
    assert len(g.extra_nodes) == g_extra and len(o.extra_nodes) == o_extra
    by_name = {n.name: n for n in orig}
    assert {RP.activity_class(n, by_name[n].cat) for n in o.extra_nodes} <= {"copy:HtoD", "copy:DtoH"}
    p = RP.price_positions(seq, g, o)
    assert p.unpriced == () and set(p.device_s) == {i for i, _ in seq.device_ops()}
    assert p.label_counts() == labels
    assert p.mismatch_s / p.priced_s == pytest.approx(share, abs=5e-5)
    gate = RP.pricing_gate([{"arm": cell, "pricing": p.summary(seq)}], 0.05)
    assert gate[cell]["validated"]


# --- compact streams for the isolated original (6 Nov 2026 ruling 3) -------------------------

def _device_trace(tmp_path, name="t.json", corr0=1):
    import random
    t = _flushed_trace(40, random.Random(7))            # about half the launches in a flush
    t.corr = corr0 + 1000
    for i, (nm, nb) in enumerate((("Memcpy HtoD (Pageable -> Device)", 65536),
                                  ("Memset (Device)", 768))):
        c = t.corr
        t.corr += 1
        t.x("cuda_runtime", "cudaMemcpyAsync", 450 + i, 1.0, correlation=c)
        t.x("gpu_memcpy" if nm.startswith("Memcpy") else "gpu_memset", nm, 900 + i, 1.0,
            pid=0, tid=7, correlation=c, stream=7, bytes=nb)
    for e in t.ev:
        if e["cat"] == "kernel":
            e["args"]["stream"] = 7
    path = tmp_path / name
    path.write_text(json.dumps(t.dict()))
    return path


def test_a_compact_stream_gives_the_nodes_the_full_trace_gives(tmp_path):
    from attnbench.components import trace_stream as ts
    full = _device_trace(tmp_path)
    info = ts.write_compact(full, cats=ts.REPLAY_CATS, fields=ts.DEVICE_FIELDS)
    want = RP.replay_nodes(full, exclude=RP.FLUSH)
    got = RP.replay_nodes(info["path"], exclude=RP.FLUSH)
    assert got == want and len(want) > 2
    assert {(n.name, n.stream, n.bytes) for n in got if not n.name.startswith("k")} == {
        ("Memcpy HtoD (Pageable -> Device)", 7, 65536), ("Memset (Device)", 7, 768)}
    host = ts.write_compact(full, tmp_path / ("h" + ts.SUFFIX))      # step 5's fields only
    with pytest.raises(ReplayError, match="without the device fields"):
        RP.replay_nodes(host["path"])
    with pytest.raises(ts.TraceStreamError, match="FIELDS or DEVICE_FIELDS"):
        ts.write_compact(full, tmp_path / ("b" + ts.SUFFIX), fields=("i", "cat"))


def test_stitched_segments_keep_each_launch_with_its_own_activity(tmp_path):
    # each segment is a profiler session of its own, its correlations from the same start
    from attnbench.components import trace_stream as ts
    parts, want = [], []
    for k in range(3):
        t = Trace()
        base = 10_000 * k
        t.dev_t = base
        t.x("user_annotation", RP.REPLAY_REGION, base, 1000)
        for i in range(5):
            t.launch(base + 10 + 10 * i, f"s{k}k{i}")
        f = tmp_path / f"seg{k}.json"
        f.write_text(json.dumps(t.dict()))
        want += [n.name for n in RP.replay_nodes(f)]
        parts.append(ts.write_compact(f, cats=ts.REPLAY_CATS, fields=ts.DEVICE_FIELDS)["path"])
    out = ts.stitch_compact(parts, tmp_path / ("all" + ts.SUFFIX), region=RP.REPLAY_REGION)
    got = RP.replay_nodes(out["path"])
    assert [n.name for n in got] == want and len({n.correlation for n in got}) == 15
    with pytest.raises(ts.TraceStreamError, match="fields differ"):
        ts.stitch_compact([parts[0], ts.write_compact(tmp_path / "seg1.json")["path"]],
                          tmp_path / ("x" + ts.SUFFIX), region=RP.REPLAY_REGION)
