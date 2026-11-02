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


# --- matching ------------------------------------------------------------------------

SEQ = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::mm", kernels=("gemm",)),
    SeqOp(SeqKind.COPY, layer=0, direction="HtoD", payload_bytes=8),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::nonzero", kernels=("scan", "write")),
    SeqOp(SeqKind.SYNC, layer=0, mechanism="item", payload_bytes=4),
    SeqOp(SeqKind.GRAPH, layer=1, kernels=("g0", "g1")),
    SeqOp(SeqKind.KERNEL, layer=1, aten_op="aten::mm", kernels=("gemm",)),
))


def nodes(*pairs):
    return [Node("kernel", n, d, None) for n, d in pairs]


def test_a_complete_match_gives_every_device_position_its_time():
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    assert m.complete and (m.n_expected, m.n_nodes) == (6, 6)
    assert RP.device_costs(m) == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}
    assert RP.expected_nodes(SEQ)[1:3] == [(2, "scan"), (2, "write")]


def test_a_substituted_kernel_leaves_its_position_without_a_cost():
    # nonzero_static replaces nonzero's scan in the replay
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("nonzero_static", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    assert not m.complete
    assert m.unmatched_positions == (2,) and m.extra_nodes == ("nonzero_static",)
    assert m.families_unmatched == {"kernel:aten::nonzero": 1}
    assert 2 not in m.device_s                 # a part-matched position gets nothing
    with pytest.raises(ReplayError, match="no device cost is taken from a different kernel"):
        RP.device_costs(m)


def test_an_extra_node_alone_makes_the_match_incomplete():
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25), ("g0", 2.0),
                            ("g1", 3.0), ("gemm", 4.0), ("stray", 9.0)))
    assert m.unmatched_positions == () and m.extra_nodes == ("stray",)
    with pytest.raises(ReplayError):
        RP.device_costs(m)


SEQ_MEMSET = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::addmm", kernels=("Memset (Device)", "gemm")),
    SeqOp(SeqKind.KERNEL, layer=1, aten_op="aten::addmm", kernels=("Memset (Device)", "gemm")),
))


def test_a_graph_memset_node_matches_the_eager_memset_whatever_its_memory_kind():
    # the graph's memset node has no allocation to name: "Memset (Unknown)"
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0),
                                   ("Memset (Unknown)", 0.2), ("gemm", 2.0)))
    assert m.complete and RP.device_costs(m) == {0: 1.1, 1: 2.2}


def test_the_memory_kind_is_the_only_thing_a_memset_match_forgives():
    # a kernel with a parenthesis is still compared whole; a memset needs a memset
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0),
                                   ("Memcpy (Unknown)", 0.2), ("gemm", 2.0)))
    assert m.unmatched_positions == (1,) and m.extra_nodes == ("Memcpy (Unknown)",)
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0), ("gemm", 2.0)))
    assert m.unmatched_positions == (1,) and not m.extra_nodes
    m = RP.match(SEQ_MEMSET, nodes(("Memset (Unknown)", 0.1), ("gemm", 1.0), ("Memset (Unknown)", 0.2),
                                   ("gemm", 2.0), ("Memset (Unknown)", 0.3)))
    assert m.unmatched_positions == () and m.extra_nodes == ("Memset (Unknown)",)   # the name as seen
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("scan (x)", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    assert m.unmatched_positions == (2,) and m.extra_nodes == ("scan (x)",)    # names reported as seen


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
    rep = RP.replay_vs_trace(SEQ, trace, {0: 1.05, 2: 1.0, 4: 2.0, 5: 1.05})
    assert rep["families"]["kernel:aten::mm"]["relative_difference"] == pytest.approx(0.05)
    assert RP.check_passes(rep, 0.1) == (True, [])
    ok, why = RP.check_passes(rep, 0.01)
    assert not ok and len(why) == 1 and why[0].startswith("kernel:aten::mm: relative difference 0.05")
    short = RP.replay_vs_trace(SEQ, trace, {0: 1.0, 2: 1.0, 5: 1.0})
    assert short["positions_without_replay"] == [4]
    assert not RP.check_passes(short, 0.5)[0]
    m = RP.match(SEQ, nodes(("gemm", 1.0)))
    named = RP.replay_vs_trace(SEQ, trace, trace, m)
    ok, why = RP.check_passes(named, 0.5)
    assert not ok and why[0].startswith("kernel names do not match")
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


# --- substituted positions (Appendix A, 27 Oct 2026) --------------------------------------

GRAPH_SUB = nodes(("gemm", 1.0), ("nonzero_static", 0.5), ("write", 0.25),
                  ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0))
ORIGINAL = nodes(("gemm", 1.5), ("scan", 0.625), ("write", 0.375),
                 ("g0", 2.5), ("g1", 3.5), ("gemm", 4.5))


def test_a_complete_graph_prices_every_position_from_the_graph():
    m = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
    p = RP.price_positions(SEQ, m, None)
    assert p.device_s == {0: 1.0, 2: 0.75, 4: 5.0, 5: 4.0}
    assert set(p.source.values()) == {RP.GRAPH} and p.substitute_nodes == ()


def test_a_substituted_position_is_priced_from_the_original_operator_and_flagged():
    p = RP.price_positions(SEQ, RP.match(SEQ, GRAPH_SUB), RP.match(SEQ, ORIGINAL))
    # position 2 from the isolated original (0.625 + 0.375); the rest from the graph
    assert p.device_s == {0: 1.0, 2: 1.0, 4: 5.0, 5: 4.0}
    assert p.source == {0: RP.GRAPH, 2: RP.ISOLATED_ORIGINAL, 4: RP.GRAPH, 5: RP.GRAPH}
    assert p.counts() == {RP.GRAPH: 3, RP.ISOLATED_ORIGINAL: 1}
    assert p.substitute_nodes == ("nonzero_static",)


SEQ_SETITEM = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::eq", kernels=("eq",)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::masked_fill_", kernels=("fill",)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::mm", kernels=("gemm",)),
))


def test_a_masked_fill_the_tape_serves_as_an_index_put_is_priced_from_the_original_operator():
    # the graph holds the tape's index_put_ in place of the eager masked_fill_
    g = RP.match(SEQ_SETITEM, nodes(("eq", 1.0), ("index_put_kernel", 0.5), ("gemm", 4.0)))
    assert g.unmatched_positions == (1,) and g.extra_nodes == ("index_put_kernel",)
    o = RP.match(SEQ_SETITEM, nodes(("eq", 1.5), ("fill", 0.75), ("gemm", 4.5)))
    p = RP.price_positions(SEQ_SETITEM, g, o)
    assert p.source == {0: RP.GRAPH, 1: RP.ISOLATED_ORIGINAL, 2: RP.GRAPH}
    assert p.device_s == {0: 1.0, 1: 0.75, 2: 4.0}


def test_substituted_positions_without_a_complete_original_replay_are_refused():
    g = RP.match(SEQ, GRAPH_SUB)
    with pytest.raises(ReplayError, match="which was not run"):
        RP.price_positions(SEQ, g, None)
    part = RP.match(SEQ, ORIGINAL[:3])
    with pytest.raises(ReplayError, match="does not match the sequence"):
        RP.price_positions(SEQ, g, part)


def test_an_unmatched_position_the_tape_does_not_substitute_is_refused():
    g = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1", 3.0), ("other_mm", 4.0)))
    assert g.unmatched_positions == (5,)
    with pytest.raises(ReplayError, match=r"not substitutions \(\['kernel:aten::mm'\]\)"):
        RP.price_positions(SEQ, g, RP.match(SEQ, ORIGINAL))


def test_extra_graph_nodes_with_nothing_substituted_are_refused():
    g = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25), ("g0", 2.0),
                            ("g1", 3.0), ("gemm", 4.0), ("stray", 9.0)))
    with pytest.raises(ReplayError, match="substitutes nothing"):
        RP.price_positions(SEQ, g, RP.match(SEQ, ORIGINAL))


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
    assert rec["substitute_nodes"] == ["nonzero_static"]
    with pytest.raises(ReplayError, match="cost sources differ"):
        RP.device_costs_from_record({**rec, "cost_source": {"0": RP.GRAPH}})
    unpriced = RP.to_record(seq=SEQ, prompt_id="p", m=g, timing=None, agreement=None,
                            tape=tape, masks=None)
    with pytest.raises(ReplayError, match="no device costs"):
        RP.device_costs_from_record(unpriced)


def test_a_version_one_record_with_a_complete_match_is_still_read():
    full = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                               ("g0", 2.0), ("g1", 3.0), ("gemm", 4.0)))
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


# --- a debug session's lenient pricing (2 Nov 2026) ---------------------------------------

_CP = ("void at::native::{}<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::"
       "{{lambda()#3}}::operator()() const::{{lambda()#1}}::operator()() const::"
       "{{lambda({})#1}}>(at::TensorIteratorBase&, int const&)::{{lambda(int)#1}})")
COPY_EW = _CP.format("elementwise_kernel<128, 4, at::native::gpu_kernel_impl", "unsigned char")
COPY_UNROLLED = _CP.format("unrolled_elementwise_kernel", "unsigned char")
COPY_OTHER_TYPE = _CP.format("unrolled_elementwise_kernel", "float")

SEQ_COPY = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::eq", kernels=("eq",)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::copy_", kernels=(COPY_EW,)),
    SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::mm", kernels=("gemm",)),
))
GRAPH_COPY = nodes(("eq", 1.0), ("stand_in_fill", 0.5), ("gemm", 4.0))


def test_a_copy_matches_its_other_launch_variant_only_when_asked_and_by_element_type():
    other = nodes(("eq", 1.5), ("Memcpy DtoH (Device -> Pinned)", 0.1), (COPY_UNROLLED, 0.75),
                  ("gemm", 4.5))
    assert RP.match(SEQ_COPY, other).unmatched_positions == (1,)
    m = RP.match(SEQ_COPY, other, variants=True)
    assert m.unmatched_positions == () and m.variant_matched == (1,)
    assert m.device_s[1] == 0.75 and m.extra_nodes == ("Memcpy DtoH (Device -> Pinned)",)
    wrong = nodes(("eq", 1.5), (COPY_OTHER_TYPE, 0.75), ("gemm", 4.5))
    assert RP.match(SEQ_COPY, wrong, variants=True).unmatched_positions == (1,)


def test_lenient_pricing_takes_a_variant_and_counts_the_originals_extra_nodes():
    g = RP.match(SEQ_COPY, GRAPH_COPY)
    assert g.unmatched_positions == (1,)
    o = RP.match(SEQ_COPY, nodes(("eq", 1.5), ("Memcpy DtoH (Device -> Pinned)", 0.1),
                                 (COPY_UNROLLED, 0.75), ("gemm", 4.5)), variants=True)
    p = RP.price_positions(SEQ_COPY, g, o, lenient=True)
    assert p.source == {0: RP.GRAPH, 1: RP.ISOLATED_VARIANT, 2: RP.GRAPH}
    assert p.device_s == {0: 1.0, 1: 0.75, 2: 4.0} and p.original_extra_nodes == 1
    with pytest.raises(ReplayError, match="does not match the sequence"):
        RP.price_positions(SEQ_COPY, g, o)                 # the strict rule is the default


def test_lenient_pricing_still_refuses_a_position_no_original_node_matches():
    g = RP.match(SEQ_COPY, GRAPH_COPY)
    o = RP.match(SEQ_COPY, nodes(("eq", 1.5), (COPY_OTHER_TYPE, 0.75), ("gemm", 4.5)),
                 variants=True)
    with pytest.raises(ReplayError, match="1 of the 1 substituted positions are not matched"):
        RP.price_positions(SEQ_COPY, g, o, lenient=True)
    with pytest.raises(ReplayError, match="which was not run"):
        RP.price_positions(SEQ_COPY, g, None, lenient=True)


def test_lenient_pricing_prices_an_unsubstituted_missing_position_from_the_original_and_says_so():
    g = RP.match(SEQ, nodes(("gemm", 1.0), ("scan", 0.5), ("write", 0.25),
                            ("g0", 2.0), ("g1", 3.0), ("other_mm", 4.0)))
    assert g.unmatched_positions == (5,)
    p = RP.price_positions(SEQ, g, RP.match(SEQ, ORIGINAL), lenient=True)
    assert p.source[5] == RP.ISOLATED_UNSUBSTITUTED and p.device_s[5] == 4.5
    assert p.source[0] == RP.GRAPH
    with pytest.raises(ReplayError, match="not substitutions"):
        RP.price_positions(SEQ, g, RP.match(SEQ, ORIGINAL))
