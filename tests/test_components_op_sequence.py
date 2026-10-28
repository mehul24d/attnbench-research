"""T2.3: operation sequence and synchronisation map, read from a trace.

No GPU here, and none of these traces was recorded on one. `TB` writes
events in the Kineto chrome-trace format (`cuda_runtime` calls with a
`correlation` id, `kernel` / `gpu_memcpy` / `gpu_memset` activities with
`stream`, `device` and `bytes`, `deviceProperties`), and one test grafts CUDA
calls onto a real CPU trace from `torch.profiler`, so the nesting and
timestamps there are PyTorch's own. The plan's acceptance test, recorded
traces of two code paths, needs a GPU session (trace_recording's zoo and the
arm list in the T2.3 report).

The independent checker is `random_program`: it writes a random program as
a trace and states, from the program alone, which operations the sequence
must hold. It shares no code with the parser.
"""

from __future__ import annotations

import contextlib
import json
import random

import pytest
import torch

from attnbench.components import op_sequence as S
from attnbench.components.op_sequence import (CodePath, SeqKind, SeqOp, OpSequence,
                                              SequenceError)
from attnbench.predict import recursion as R

GPU = "NVIDIA L4"
PID, TID = 4242, 4243
CP = CodePath(arm="xattn", gpu=GPU, implementation="xattn:torch",
              libraries=(("torch", "2.8.0"),))


class TB:
    """Writes a Kineto-format trace. Host time advances as calls are made;
    each device activity starts after its call and after the previous one."""

    def __init__(self, gpu=GPU, stream=7, device=0):
        self.ev, self.t, self.dev_t, self.corr = [], 100.0, 100.0, 1
        self.gpu, self.stream, self.device = gpu, stream, device

    def x(self, cat, name, ts, dur, pid=PID, tid=TID, **args):
        e = {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid,
             "ts": ts, "dur": dur, "args": args}
        self.ev.append(e)
        return e

    @contextlib.contextmanager
    def span(self, cat, name, tid=TID):
        start = self.t
        self.t += 1
        yield
        self.t += 1
        self.x(cat, name, start, self.t - start, tid=tid)

    def cpu(self, name):
        return self.span("cpu_op", name)

    def note(self, name, tid=TID):
        return self.span("user_annotation", name, tid=tid)

    def region(self):
        return self.note(S.REGION)

    def layer(self, i):
        return self.note(S.layer_annotation(i))

    def host_compute(self):
        return self.note(S.HOST_COMPUTE)

    def py(self, name):
        return self.span("python_function", name)

    def call(self, name, cat="cuda_runtime", tid=TID):
        c = self.corr
        self.corr += 1
        ts = self.t
        self.x(cat, name, ts, 2.0, tid=tid, correlation=c, cbid=0)
        self.t += 3
        return c, ts

    def act(self, cat, name, corr, host_ts, stream=None, dur=10.0, **args):
        stream = self.stream if stream is None else stream
        start = max(self.dev_t, host_ts + 1)
        self.dev_t = start + dur
        self.x(cat, name, start, dur, pid=self.device, tid=stream,
               correlation=corr, stream=stream, device=self.device, **args)

    def launch(self, kernel="k", api="cudaLaunchKernel", cat="cuda_runtime",
               stream=None, tid=TID):
        c, ts = self.call(api, cat, tid=tid)
        self.act("kernel", kernel, c, ts, stream)
        return c

    def memcpy(self, kind, src, dst, nbytes, api="cudaMemcpyAsync"):
        c, ts = self.call(api)
        self.act("gpu_memcpy", f"Memcpy {kind} ({src} -> {dst})", c, ts, bytes=nbytes)
        if "Async" not in api:
            self.t = max(self.t, self.dev_t + 1)
        return c

    def memset(self):
        c, ts = self.call("cudaMemsetAsync")
        self.act("gpu_memset", "Memset (Device)", c, ts, bytes=64)

    def graph(self, n):
        c, ts = self.call("cudaGraphLaunch")
        for k in range(n):
            self.act("kernel", f"graph_node_{k}", c, ts)

    def sync(self, api="cudaStreamSynchronize"):
        self.call(api)
        self.t = max(self.t, self.dev_t + 1)

    def trace(self):
        return {"schemaVersion": 1,
                "deviceProperties": [{"id": self.device, "name": self.gpu,
                                      "computeMajor": 8, "computeMinor": 9}],
                "traceEvents": list(self.ev)}


# Readbacks as PyTorch issues them (memcpy_and_sync inside the aten op).

def item(tb, nbytes=4):
    with tb.cpu("aten::item"), tb.cpu("aten::_local_scalar_dense"):
        tb.memcpy("DtoH", "Device", "Pageable", nbytes)
        tb.sync()


def boolean(tb):
    with tb.cpu("aten::is_nonzero"), tb.cpu("aten::item"), \
            tb.cpu("aten::_local_scalar_dense"):
        tb.memcpy("DtoH", "Device", "Pageable", 1)
        tb.sync()


def nonzero(tb, before=2, after=1, mem="Pinned", outer=None):
    with contextlib.ExitStack() as st:
        if outer:
            st.enter_context(tb.cpu(outer))
        st.enter_context(tb.cpu("aten::nonzero"))
        for k in range(before):
            tb.launch(f"cub_scan_{k}")
        tb.memcpy("DtoH", "Device", mem, 8)
        tb.sync()
        for k in range(after):
            tb.launch(f"write_indices_{k}")


def to_cpu(tb, nbytes, dst="Pageable"):
    with tb.cpu("aten::to"), tb.cpu("aten::_to_copy"), tb.cpu("aten::copy_"):
        tb.memcpy("DtoH", "Device", dst, nbytes)
        tb.sync()


def kernel(tb, aten="aten::mm", name="ampere_sgemm_128x64"):
    with tb.cpu(aten):
        tb.launch(name)


def parse(tb, cp=CP, **kw):
    return S.parse_trace(tb.trace(), cp, context_length=kw.pop("context_length", 2048),
                         batch=1, **kw)


def kinds(seq):
    ops = seq.ops if isinstance(seq, OpSequence) else seq
    return [op.kind for op in ops]


# --- mechanisms ---------------------------------------------------------------

def test_item_is_one_sync_with_its_payload():
    tb = TB()
    with tb.region():
        item(tb, 4)
    (op,) = parse(tb).ops
    assert (op.kind, op.mechanism, op.payload_bytes) == (SeqKind.SYNC, "item", 4)
    assert op.composite is None and op.aten_op == "aten::_local_scalar_dense"
    assert op.origin == "aten::item"


def test_bool_of_a_tensor_is_bool_not_item():
    tb = TB()
    with tb.region():
        boolean(tb)
    (op,) = parse(tb).ops
    assert (op.mechanism, op.payload_bytes, op.composite) == ("bool", 1, None)


def test_cpu_copy_is_a_pageable_copy_to_host():
    tb = TB()
    with tb.region():
        to_cpu(tb, 1024)
        to_cpu(tb, 512, dst="Pinned")
    a, b = parse(tb).ops
    assert (a.mechanism, a.payload_bytes) == ("copy_to_host_pageable", 1024)
    assert (b.mechanism, b.payload_bytes) == ("copy_to_host_pinned", 512)


def test_explicit_synchronize_carries_no_payload():
    tb = TB()
    with tb.region():
        kernel(tb)
        tb.sync("cudaDeviceSynchronize")
        tb.sync("cudaStreamSynchronize")
    k, s1, s2 = parse(tb).ops
    assert k.kind is SeqKind.KERNEL
    for s in (s1, s2):
        assert (s.kind, s.mechanism, s.payload_bytes, s.aten_op) == \
            (SeqKind.SYNC, "synchronize", 0, None)


def test_a_non_blocking_copy_then_a_synchronize_are_two_operations():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::copy_"):
            tb.memcpy("DtoH", "Device", "Pinned", 256)
        tb.sync("cudaDeviceSynchronize")
    c, s = parse(tb).ops
    assert (c.kind, c.direction, c.payload_bytes) == (SeqKind.COPY, "DtoH", 256)
    assert (s.mechanism, s.payload_bytes) == ("synchronize", 0)


def test_a_launch_between_copy_and_sync_stops_the_fold():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::copy_"):
            tb.memcpy("DtoH", "Device", "Pinned", 256)
            tb.launch("k")
            tb.sync()
    assert kinds(parse(tb)) == [SeqKind.COPY, SeqKind.KERNEL, SeqKind.SYNC]


def test_a_copy_and_sync_in_different_aten_ops_do_not_fold():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::copy_"):
            tb.memcpy("DtoH", "Device", "Pinned", 256)
        with tb.cpu("aten::copy_"):
            tb.sync()
    assert kinds(parse(tb)) == [SeqKind.COPY, SeqKind.SYNC]


def test_a_blocking_memcpy_is_a_sync_by_itself():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::copy_"):
            tb.memcpy("DtoH", "Device", "Pageable", 64, api="cudaMemcpy")
    (op,) = parse(tb).ops
    assert (op.kind, op.mechanism, op.payload_bytes) == \
        (SeqKind.SYNC, "copy_to_host_pageable", 64)


def test_waits_the_plan_does_not_price_are_kept_and_named():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::to"), tb.cpu("aten::_to_copy"), tb.cpu("aten::copy_"):
            tb.memcpy("HtoD", "Pageable", "Device", 8)
            tb.sync()
        tb.sync("cudaEventSynchronize")
        tb.sync("cudaFree")
    ops = parse(tb).ops
    assert [o.mechanism for o in ops] == ["copy_from_host_pageable",
                                          "event_synchronize", "implicit"]
    assert ops[0].payload_bytes == 8
    assert not any(o.priced for o in ops)
    assert set(S.UNPRICED_MECHANISMS) >= {o.mechanism for o in ops}
    assert not set(S.UNPRICED_MECHANISMS) & set(R.SYNC_MECHANISMS)


# --- composites -----------------------------------------------------------------

def test_nonzero_splits_into_kernels_and_one_size_readback():
    tb = TB()
    with tb.region():
        kernel(tb, "aten::gt", "gt_kernel")
        nonzero(tb, before=2, after=1)
    ops = parse(tb).ops
    assert kinds(ops) == [SeqKind.KERNEL, SeqKind.KERNEL, SeqKind.KERNEL,
                          SeqKind.SYNC, SeqKind.KERNEL]
    assert [o.composite for o in ops] == [None] + ["aten::nonzero"] * 4
    s = ops[3]
    assert (s.mechanism, s.payload_bytes) == ("copy_to_host_pinned", 8)
    # No operation carries both device work and a wait.
    assert all(not (o.is_device and o.kind is SeqKind.SYNC) for o in ops)


def test_boolean_indexing_is_a_composite_of_its_outer_op():
    tb = TB()
    with tb.region():
        nonzero(tb, before=1, after=0, mem="Pageable", outer="aten::index")
    k, s = parse(tb).ops
    assert (k.composite, s.composite) == ("aten::index", "aten::index")
    assert s.mechanism == "copy_to_host_pageable"
    assert (s.origin, s.aten_op) == ("aten::index", "aten::nonzero")


def test_a_composite_does_not_leak_into_the_next_op():
    tb = TB()
    with tb.region():
        nonzero(tb, before=1, after=0)
        kernel(tb, "aten::nonzero", "another")   # same name, other instance
    assert [o.composite for o in parse(tb).ops] == ["aten::nonzero"] * 2 + [None]


# --- other operation kinds, layers, host compute --------------------------------

def test_memset_device_copy_and_graph_are_device_work():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::zero_"):
            tb.memset()
        with tb.cpu("aten::copy_"):
            tb.memcpy("DtoD", "Device", "Device", 4096)
        tb.graph(3)
    m, d, g = parse(tb).ops
    assert (m.kind, d.kind, g.kind) == (SeqKind.MEMSET, SeqKind.DEVICE_COPY, SeqKind.GRAPH)
    assert d.payload_bytes == 0 and all(o.is_device for o in (m, d, g))
    assert g.kernels == ("graph_node_0", "graph_node_1", "graph_node_2")
    assert S.op_key(g)[-1] == (3,)


def test_layers_come_from_the_innermost_layer_annotation():
    tb = TB()
    with tb.region():
        kernel(tb, "aten::embedding")
        for i in range(2):
            with tb.layer(i):
                kernel(tb)
                with tb.note("some::other_annotation"):
                    kernel(tb, "aten::add")
        kernel(tb, "aten::linear")
    seq = parse(tb)
    assert [o.layer for o in seq.ops] == [None, 0, 0, 1, 1, None]
    assert seq.layers == (0, 1)


def test_refuses_nested_layer_annotations():
    tb = TB()
    with tb.region(), tb.layer(0), tb.layer(1):
        kernel(tb)
    with pytest.raises(SequenceError, match=r"layer annotations nested \[0, 1\]"):
        parse(tb)


def test_host_compute_markers_and_the_loaded_rule():
    tb = TB()
    with tb.region():
        with tb.host_compute():
            pass
        tb.memcpy("HtoD", "Pinned", "Device", 100)   # 1: host compute follows: loaded
        with tb.host_compute():
            pass
        tb.sync()
        tb.memcpy("HtoD", "Pinned", "Device", 200)   # 4: sync first: unloaded
        tb.sync()
        with tb.host_compute():
            pass
        tb.memcpy("HtoD", "Pinned", "Device", 300)   # 7: nothing after: unloaded
    ops = parse(tb).ops
    assert kinds(ops) == [SeqKind.HOST_COMPUTE, SeqKind.COPY, SeqKind.HOST_COMPUTE,
                          SeqKind.SYNC, SeqKind.COPY, SeqKind.SYNC,
                          SeqKind.HOST_COMPUTE, SeqKind.COPY]
    assert [o.loaded for o in ops if o.kind is SeqKind.COPY] == [True, False, False]
    assert not any(o.loaded for o in ops if o.kind is SeqKind.SYNC)


def test_a_triton_launch_through_the_driver_api_is_a_kernel():
    tb = TB()
    with tb.region():
        tb.launch("_attn_fwd", api="cuLaunchKernel", cat="cuda_driver")
    (op,) = parse(tb).ops
    assert (op.kind, op.aten_op, op.kernels) == (SeqKind.KERNEL, None, ("_attn_fwd",))


def test_calls_that_do_nothing_to_the_sequence_are_counted_not_kept():
    tb = TB()
    with tb.region():
        tb.call("cudaGetDevice")
        tb.call("cudaGetDevice")
        tb.call("cudaEventRecord")
        kernel(tb)
    seq = parse(tb)
    assert len(seq) == 1
    assert dict(seq.ignored_calls) == {"cudaGetDevice": 2, "cudaEventRecord": 1}


def test_work_before_and_after_the_region_is_left_out():
    tb = TB()
    kernel(tb, "aten::before")
    with tb.region():
        kernel(tb)
    kernel(tb, "aten::after")
    assert [o.aten_op for o in parse(tb).ops] == ["aten::mm"]


def test_occurrence_picks_one_of_several_regions():
    tb = TB()
    for name in ("aten::a", "aten::b"):
        with tb.region():
            kernel(tb, name)
    with pytest.raises(SequenceError, match="2 'attnbench::prefill' regions"):
        parse(tb)
    assert parse(tb, occurrence=1).ops[0].aten_op == "aten::b"
    with pytest.raises(SequenceError, match="occurrence 2 of 2"):
        parse(tb, occurrence=2)


def test_frames_drop_the_directory_and_skip_builtins():
    tb = TB()
    with tb.region():
        with tb.py("/opt/conda/lib/python3.11/site-packages/xattn/src/Xattention.py(301): find_blocks_chunked"):
            with tb.py("<built-in method all of Tensor object at 0x7f00deadbeef>"):
                kernel(tb, "aten::all")
    assert parse(tb).ops[0].frame == "Xattention.py(301): find_blocks_chunked"


# --- refusals -------------------------------------------------------------------

def refuses(tb, match, **kw):
    with pytest.raises(SequenceError, match=match):
        parse(tb, **kw)


def test_refuses_a_trace_without_a_region():
    tb = TB()
    kernel(tb)
    refuses(tb, "no 'attnbench::prefill' annotation")


def test_refuses_a_trace_without_cuda_devices():
    tb = TB()
    with tb.region():
        kernel(tb)
    t = tb.trace()
    t["deviceProperties"] = []
    with pytest.raises(SequenceError, match="lists no CUDA device"):
        S.parse_trace(t, CP, context_length=2048, batch=1)


def test_refuses_a_region_with_no_device_work():
    tb = TB()
    with tb.region():
        with tb.host_compute():
            pass
    refuses(tb, "no device work in the region")


def test_refuses_a_launch_with_no_kernel_or_two():
    tb = TB()
    with tb.region():
        tb.call("cudaLaunchKernel")
    refuses(tb, "has 0 device activities")
    tb = TB()
    with tb.region():
        c = tb.launch("k")
        tb.act("kernel", "k2", c, tb.t)
    refuses(tb, "has 2 device activities")


def test_refuses_device_work_with_no_host_call():
    tb = TB()
    with tb.region():
        kernel(tb)
        tb.act("kernel", "triton_kernel", 999, tb.t)
    refuses(tb, "no host call in the trace")


def test_refuses_launches_from_another_thread():
    tb = TB()
    with tb.region():
        kernel(tb)
        tb.launch("k", tid=TID + 1)
    refuses(tb, "issued from thread")


def test_refuses_a_gpu_the_code_path_does_not_name():
    tb = TB(gpu="NVIDIA A100-SXM4-40GB")
    with tb.region():
        kernel(tb)
    refuses(tb, "the code path says 'NVIDIA L4'")


@pytest.mark.parametrize("name,match", [
    ("Memcpy DtoH (Device -> Managed)", "managed memory"),
    ("Memcpy PtoP (Device -> Device)", "only HtoD, DtoH and DtoD"),
    ("Copy sideways", "unrecognised copy activity"),
])
def test_refuses_copies_outside_the_scope(name, match):
    tb = TB()
    with tb.region():
        c, ts = tb.call("cudaMemcpyAsync")
        tb.act("gpu_memcpy", name, c, ts, bytes=8)
    refuses(tb, match)


def test_refuses_a_copy_without_bytes_or_memory_kinds():
    tb = TB()
    with tb.region():
        c, ts = tb.call("cudaMemcpyAsync")
        tb.act("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", c, ts)
    refuses(tb, "records no byte count")
    tb = TB()
    with tb.region():
        with tb.cpu("aten::copy_"):
            c, ts = tb.call("cudaMemcpyAsync")
            tb.act("gpu_memcpy", "Memcpy DtoH", c, ts, bytes=8)
            tb.sync()
    refuses(tb, "pinned or pageable is needed")


def test_refuses_nested_host_compute():
    tb = TB()
    with tb.region():
        kernel(tb)
        with tb.host_compute(), tb.host_compute():
            pass
    refuses(tb, "inside another one")


def test_refuses_one_stream_out_of_issue_order():
    tb = TB()
    with tb.region():
        kernel(tb)
        kernel(tb)
    acts = [e for e in tb.ev if e["cat"] == "kernel"]
    acts[0]["ts"], acts[1]["ts"] = acts[1]["ts"], acts[0]["ts"]
    refuses(tb, "out of host issue order")


# --- streams --------------------------------------------------------------------

def test_one_stream_passes_the_check():
    tb = TB()
    with tb.region():
        kernel(tb)
    chk = S.stream_check(parse(tb))
    assert chk.single_stream and chk.streams == (7,) and chk.message == "one stream (7)"


def test_a_second_stream_or_a_cross_stream_wait_is_flagged():
    tb = TB()
    with tb.region():
        kernel(tb)
        with tb.cpu("aten::mul"):
            tb.launch("side", stream=13)
    chk = S.stream_check(parse(tb))
    assert not chk.single_stream and chk.streams == (7, 13)
    assert "2 streams [7, 13]" in chk.message
    tb = TB()
    with tb.region():
        kernel(tb)
        tb.call("cudaStreamWaitEvent")
    chk = S.stream_check(parse(tb))
    assert not chk.single_stream and chk.cross_stream_waits == 1


# --- same or different ----------------------------------------------------------

def dense_like(gemm="ampere_sgemm_128x64", extra=False, payload=4, triton="_fwd"):
    tb = TB()
    with tb.region():
        for i in range(2):
            with tb.layer(i):
                kernel(tb, "aten::mm", gemm)
                if extra:
                    kernel(tb, "aten::mm", "splitk_reduce")
                tb.launch(triton, api="cuLaunchKernel", cat="cuda_driver")
                item(tb, payload)
    return tb


def test_two_gpus_with_differently_named_gemms_are_the_same_sequence():
    a = parse(dense_like())
    b_tb = dense_like(gemm="sm90_xmma_gemm_bf16")
    b_tb.gpu = "NVIDIA H100 80GB HBM3"
    b = parse(b_tb, cp=CodePath(arm="xattn", gpu="NVIDIA H100 80GB HBM3",
                                implementation="xattn:torch"))
    c = S.compare(a, b)
    assert c.same and c.verdict == "same sequence" and c.first_difference is None
    assert c.reason == "6 operations agree"
    assert not S.compare(a, b, kernel_names=True).same


def test_a_triton_kernel_is_identified_by_its_name():
    c = S.compare(parse(dense_like()), parse(dense_like(triton="_other")))
    assert c.verdict == "different sequence" and c.first_difference == 1


def test_an_extra_kernel_makes_a_different_sequence_at_its_position():
    c = S.compare(parse(dense_like()), parse(dense_like(extra=True)))
    assert not c.same and c.first_difference == 1
    assert "kernel aten::mm (layer 0)" in c.reason
    assert c.lengths == (6, 8)


def test_the_same_operations_in_another_layer_are_a_different_sequence():
    def two_layers(split):
        tb = TB()
        with tb.region():
            with tb.layer(0):
                for _ in range(split):
                    kernel(tb)
            with tb.layer(1):
                for _ in range(3 - split):
                    kernel(tb)
        return parse(tb)
    c = S.compare(two_layers(2), two_layers(1))
    assert not c.same and c.first_difference == 1


def test_payload_counts_unless_switched_off():
    a, b = parse(dense_like(payload=4)), parse(dense_like(payload=8))
    c = S.compare(a, b)
    assert not c.same and c.first_difference == 2 and "4 B" in c.reason
    assert S.compare(a, b, payload=False).same


def test_a_truncated_sequence_differs_where_it_stops():
    a = parse(dense_like())
    b = OpSequence(a.code_path, a.context_length, a.batch, a.ops[:5], a.streams)
    c = S.compare(a, b)
    assert c.first_difference == 5 and "ends at operation 5" in c.reason


def test_compare_refuses_two_arms_or_two_cells():
    a = parse(dense_like())
    other = OpSequence(CodePath(arm="dense", gpu=GPU, implementation="eager"),
                       a.context_length, 1, a.ops, a.streams)
    with pytest.raises(SequenceError, match="compare one arm with itself"):
        S.compare(a, other)
    longer = parse(dense_like(), context_length=4096)
    with pytest.raises(SequenceError, match="different cells"):
        S.compare(a, longer)
    assert S.compare(a, longer, same_cell=False).same


# --- the sync map and the arms-table column -------------------------------------

def test_sync_map_positions_and_mechanisms():
    tb = TB()
    with tb.region():
        with tb.layer(0):
            kernel(tb)
            item(tb, 4)
            nonzero(tb, before=1, after=0)
    m = S.sync_map(parse(tb))
    assert [(e.position, e.mechanism, e.composite, e.payload_bytes, e.layer) for e in m] == \
        [(1, "item", None, 4, 0), (3, "copy_to_host_pinned", "aten::nonzero", 8, 0)]
    assert all(e.priced for e in m)


def test_sync_column_without_synchronisations():
    tb = TB()
    with tb.region():
        with tb.layer(0):
            kernel(tb)
    assert S.sync_column(parse(tb)) == "None (checked in the trace)"


def test_sync_column_when_every_layer_agrees():
    tb = TB()
    with tb.region():
        for i in range(3):
            with tb.layer(i):
                kernel(tb)
                boolean(tb)
                item(tb)
    assert S.sync_column(parse(tb)) == "2 per layer in all 3 layers: bool x1, item x1"


def test_sync_column_when_layers_differ_and_outside_layers():
    tb = TB()
    with tb.region():
        with tb.layer(0):
            kernel(tb)
            boolean(tb)
        with tb.layer(1):
            kernel(tb)
            boolean(tb)
            boolean(tb)
            nonzero(tb, before=1, after=0)
        with tb.layer(2):
            kernel(tb)
        item(tb)
        with tb.cpu("aten::to"), tb.cpu("aten::_to_copy"), tb.cpu("aten::copy_"):
            tb.memcpy("HtoD", "Pageable", "Device", 8)
            tb.sync()
    assert S.sync_column(parse(tb)) == (
        "0 to 3 per layer over 3 layers (4 in all: bool x3, "
        "copy_to_host_pinned (aten::nonzero) x1); 2 outside the layers: "
        "copy_from_host_pageable x1, item x1; unpriced: copy_from_host_pageable")


# --- asserts from the paired arms -----------------------------------------------

def assert_arm(asserts_per_layer, predicate=("aten::gt", "aten::all"), last="bool",
               extra=False):
    tb = TB()
    with tb.region():
        for i, n in enumerate(asserts_per_layer):
            with tb.layer(i):
                for _ in range(n):
                    for p in predicate:
                        kernel(tb, p, p + "_kernel")
                    boolean(tb) if last == "bool" else item(tb)
                kernel(tb, "aten::mm")
                if extra:
                    kernel(tb, "aten::add")
    return tb


def test_asserts_are_the_runs_the_stripped_arm_lacks():
    shipped = parse(assert_arm([1, 2]))
    stripped = parse(assert_arm([0, 0]))
    out = S.attribute_asserts(shipped, stripped)
    comp = [o.composite for o in out.ops]
    assert comp == ["assert"] * 3 + [None] + ["assert"] * 6 + [None]
    assert [o.mechanism for o in out.ops if o.composite == "assert"
            and o.kind is SeqKind.SYNC] == ["bool"] * 3


def test_the_alignment_finds_the_valid_removal_greedy_matching_misses():
    # shipped [gt, all, bool, gt, all, mm], stripped [gt, all, mm]: matching
    # the first gt and all leaves [bool, gt, all] removed, which is not an
    # assert. The removal has to be the first three.
    tb = TB()
    with tb.region(), tb.layer(0):
        kernel(tb, "aten::gt", "g")
        kernel(tb, "aten::all", "a")
        boolean(tb)
        kernel(tb, "aten::gt", "g")
        kernel(tb, "aten::all", "a")
        kernel(tb, "aten::mm")
    tb2 = TB()
    with tb2.region(), tb2.layer(0):
        kernel(tb2, "aten::gt", "g")
        kernel(tb2, "aten::all", "a")
        kernel(tb2, "aten::mm")
    out = S.attribute_asserts(parse(tb), parse(tb2))
    assert [o.composite for o in out.ops] == ["assert"] * 3 + [None] * 3


@pytest.mark.parametrize("stripped_kw,shipped_kw,match", [
    (dict(asserts_per_layer=[0], extra=True), dict(asserts_per_layer=[1]),
     "not the shipped arm with assert statements removed"),
    (dict(asserts_per_layer=[0]), dict(asserts_per_layer=[1], last="item"),
     "not the shipped arm"),
    (dict(asserts_per_layer=[0, 0]), dict(asserts_per_layer=[1]),
     "differ in their layer structure"),
])
def test_attribute_asserts_refuses_anything_but_removed_asserts(stripped_kw, shipped_kw, match):
    with pytest.raises(SequenceError, match=match):
        S.attribute_asserts(parse(assert_arm(**shipped_kw)), parse(assert_arm(**stripped_kw)))


def test_a_removed_run_with_a_copy_in_it_is_not_an_assert():
    tb = TB()
    with tb.region(), tb.layer(0):
        kernel(tb, "aten::gt", "g")
        with tb.cpu("aten::copy_"):
            tb.memcpy("HtoD", "Pinned", "Device", 8)
        boolean(tb)
        kernel(tb, "aten::mm")
    tb2 = TB()
    with tb2.region(), tb2.layer(0):
        kernel(tb2, "aten::mm")
    with pytest.raises(SequenceError, match="not the shipped arm"):
        S.attribute_asserts(parse(tb), parse(tb2))


def test_a_removed_kernel_without_a_sync_is_not_an_assert():
    tb = TB()
    with tb.region(), tb.layer(0):
        kernel(tb, "aten::gt", "g")
        kernel(tb, "aten::mm")
    tb2 = TB()
    with tb2.region(), tb2.layer(0):
        kernel(tb2, "aten::mm")
    with pytest.raises(SequenceError, match="not the shipped arm"):
        S.attribute_asserts(parse(tb), parse(tb2))


def test_attribute_asserts_refuses_two_gpus():
    tb = assert_arm([0])
    tb.gpu = "NVIDIA A100-SXM4-40GB"
    other = parse(tb, cp=CodePath(arm="xattn", gpu=tb.gpu, implementation="x"))
    with pytest.raises(SequenceError, match="one GPU at one cell"):
        S.attribute_asserts(parse(assert_arm([1])), other)


# --- to the recursion -----------------------------------------------------------

def mixed():
    tb = TB()
    with tb.region():
        with tb.host_compute():
            pass                                   # 0
        kernel(tb)                                 # 1
        tb.memcpy("HtoD", "Pinned", "Device", 64)  # 2: loaded (3 follows)
        with tb.host_compute():
            pass                                   # 3
        item(tb, 4)                                # 4
        kernel(tb, "aten::add")                    # 5
    return tb


def test_to_operations_hand_computed_prefill():
    seq = parse(mixed())
    ops = S.to_operations(seq, host={0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0, 5: 1.0},
                          device={1: 10.0, 5: 20.0}, link={2: 4.0})
    assert [(o.kind, o.host, o.device, o.link) for o in ops] == [
        (R.OpKind.KERNEL, 4.0, 10.0, 0.0),   # 1 + host compute 3
        (R.OpKind.COPY, 2.0, 0.0, 4.0),
        (R.OpKind.SYNC, 0.0, 0.0, 0.0),
        (R.OpKind.KERNEL, 6.0, 20.0, 0.0),   # 1 + host compute 5, past the sync
    ]
    # Section 5.1: an .item() wait has payload zero by definition, so it
    # reaches the recursion with none; the sequence keeps the 4 bytes the
    # trace recorded, for the "same sequence" comparison.
    assert ops[1].payload_bytes == 64 and ops[2].payload_bytes == 0
    assert seq.ops[4].mechanism == "item" and seq.ops[4].payload_bytes == 4
    assert ops[2].mechanism == "item" and ops[0].label == "aten::mm"
    # A bandwidth for item would charge 4/4 if the payload went through; it
    # does not, so s0 = s_fixed = 7.
    costs = {"item": R.MechanismCost(7.0, bandwidth=4.0),
             "synchronize": R.MechanismCost(2.0)}
    # H1 = 4, D1 = 14; H2 = 6, D2 = 18; sync: H3 = max(6, 18) + 7 = 25;
    # H4 = 31, D4 = max(31, 25) + 20 = 51; closing sync: max(31, 51) + 2 = 53.
    assert R.predict_prefill(R.close_prefill(ops), costs, None) == 53.0


def test_to_operations_maps_every_device_kind_to_a_kernel():
    tb = TB()
    with tb.region():
        with tb.cpu("aten::zero_"):
            tb.memset()
        with tb.cpu("aten::copy_"):
            tb.memcpy("DtoD", "Device", "Device", 8)
        tb.graph(2)
    ops = S.to_operations(parse(tb), host={0: 1.0, 1: 1.0, 2: 1.0},
                          device={0: 1.0, 1: 2.0, 2: 3.0}, link={})
    assert [o.kind for o in ops] == [R.OpKind.KERNEL] * 3
    assert [o.payload_bytes for o in ops] == [0, 0, 0]


@pytest.mark.parametrize("host,device,link,match", [
    ({0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0}, {1: 1.0, 5: 1.0}, {2: 1.0}, r"host costs: missing positions \[5\]"),
    ({0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0, 4: 1.0, 5: 1.0}, {1: 1.0, 5: 1.0}, {2: 1.0}, r"unexpected positions \[4\]"),
    ({0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0, 5: 1.0}, {1: 1.0}, {2: 1.0}, "device costs"),
    ({0: 3.0, 1: 1.0, 2: 2.0, 3: 5.0, 5: 1.0}, {1: 1.0, 5: 1.0}, {}, "link costs"),
])
def test_to_operations_refuses_missing_or_extra_costs(host, device, link, match):
    with pytest.raises(SequenceError, match=match):
        S.to_operations(parse(mixed()), host=host, device=device, link=link)


def test_to_operations_refuses_an_unpriced_wait_and_trailing_host_compute():
    tb = TB()
    with tb.region():
        kernel(tb)
        tb.sync("cudaEventSynchronize")
    with pytest.raises(SequenceError, match="'event_synchronize' wait"):
        S.to_operations(parse(tb), host={0: 1.0}, device={0: 1.0}, link={})
    tb = TB()
    with tb.region():
        kernel(tb)
        with tb.host_compute():
            pass
    with pytest.raises(SequenceError, match="no operation to be charged to"):
        S.to_operations(parse(tb), host={0: 1.0, 1: 1.0}, device={0: 1.0}, link={})


# --- the artifact ---------------------------------------------------------------

def _floats(x):
    if isinstance(x, float):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _floats(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _floats(v)


def test_the_record_round_trips_and_holds_no_durations():
    seq = parse(mixed())
    rec = S.to_record(seq)
    assert S.from_record(json.loads(json.dumps(rec))) == seq
    assert list(_floats(rec)) == []
    text = json.dumps(rec)
    assert '"ts"' not in text and '"dur"' not in text and "correlation" not in text


def test_the_record_hash_is_stable_and_sees_a_change():
    a = S.to_record(parse(mixed()))
    b = S.to_record(parse(mixed()))
    assert S.record_sha256(a) == S.record_sha256(b)
    b["ops"][1]["payload_bytes"] = 65
    assert S.record_sha256(a) != S.record_sha256(b)


def test_from_record_refuses_another_schema_or_unknown_fields():
    rec = S.to_record(parse(mixed()))
    with pytest.raises(SequenceError, match="schema"):
        S.from_record({**rec, "schema": "other/1"})
    rec["ops"][0]["duration"] = 1
    with pytest.raises(SequenceError, match="operation fields"):
        S.from_record(rec)


def test_device_durations_for_the_replay_check():
    tb = TB()
    with tb.region():
        kernel(tb)
        item(tb)
        tb.graph(3)
    seq = parse(tb)
    d = S.device_durations(tb.trace(), seq)
    assert d == pytest.approx({0: 10e-6, 2: 30e-6})
    assert [S.operation_family(op) for _, op in seq.device_ops()] == \
        ["kernel:aten::mm", "graph:graph_node_0"]


# --- the independent checker: random programs ------------------------------------

STEPS = ("kernel", "triton", "item", "bool", "nonzero", "cpu", "pinned_block",
         "async_dtoh", "async_htod", "hc", "sync", "memset", "dtod", "graph")


def random_program(rng):
    """A random program, its trace, and the operations it must parse to.

    Each expected operation is (kind, layer, mechanism, direction, payload,
    composite); `loaded` is checked separately from the rule's own words.
    """
    tb, expected = TB(), []
    n_layers = rng.randint(0, 3)

    def emit(step, layer):
        n = rng.choice((1, 2, 4, 8, 1024))
        if step == "kernel":
            kernel(tb, rng.choice(("aten::mm", "aten::add", "aten::softmax")))
            expected.append(("kernel", layer, None, None, 0, None))
        elif step == "triton":
            tb.launch("_triton_k", api="cuLaunchKernel", cat="cuda_driver")
            expected.append(("kernel", layer, None, None, 0, None))
        elif step == "item":
            item(tb, n)
            expected.append(("sync", layer, "item", "DtoH", n, None))
        elif step == "bool":
            boolean(tb)
            expected.append(("sync", layer, "bool", "DtoH", 1, None))
        elif step == "nonzero":
            b, a = rng.randint(0, 2), rng.randint(0, 2)
            mem = rng.choice(("Pinned", "Pageable"))
            nonzero(tb, before=b, after=a, mem=mem)
            expected.extend([("kernel", layer, None, None, 0, "aten::nonzero")] * b)
            expected.append(("sync", layer, f"copy_to_host_{mem.lower()}", "DtoH", 8,
                             "aten::nonzero"))
            expected.extend([("kernel", layer, None, None, 0, "aten::nonzero")] * a)
        elif step == "cpu":
            to_cpu(tb, n)
            expected.append(("sync", layer, "copy_to_host_pageable", "DtoH", n, None))
        elif step == "pinned_block":
            to_cpu(tb, n, dst="Pinned")
            expected.append(("sync", layer, "copy_to_host_pinned", "DtoH", n, None))
        elif step in ("async_dtoh", "async_htod"):
            d = "DtoH" if step == "async_dtoh" else "HtoD"
            with tb.cpu("aten::copy_"):
                tb.memcpy(d, *(("Device", "Pinned") if d == "DtoH" else ("Pinned", "Device")), n)
            expected.append(("copy", layer, None, d, n, None))
        elif step == "hc":
            with tb.host_compute():
                pass
            expected.append(("host_compute", layer, None, None, 0, None))
        elif step == "sync":
            tb.sync(rng.choice(("cudaDeviceSynchronize", "cudaStreamSynchronize")))
            expected.append(("sync", layer, "synchronize", None, 0, None))
        elif step == "memset":
            with tb.cpu("aten::zero_"):
                tb.memset()
            expected.append(("memset", layer, None, None, 0, None))
        elif step == "dtod":
            with tb.cpu("aten::copy_"):
                tb.memcpy("DtoD", "Device", "Device", n)
            expected.append(("device_copy", layer, None, "DtoD", 0, None))
        elif step == "graph":
            tb.graph(rng.randint(1, 3))
            expected.append(("graph", layer, None, None, 0, None))

    with tb.region():
        kernel(tb, "aten::embedding")
        expected.append(("kernel", None, None, None, 0, None))
        for i in range(n_layers):
            with tb.layer(i):
                for _ in range(rng.randint(1, 8)):
                    emit(rng.choice(STEPS), i)
        for _ in range(rng.randint(0, 3)):
            emit(rng.choice(STEPS), None)
    return tb, expected


@pytest.mark.parametrize("seed", range(300))
def test_random_programs_parse_to_what_the_program_says(seed):
    tb, expected = random_program(random.Random(seed))
    seq = parse(tb)
    got = [(o.kind.value, o.layer, o.mechanism, o.direction, o.payload_bytes, o.composite)
           for o in seq.ops]
    assert got == expected
    # loaded: a copy with host compute after it and before the next sync.
    for i, op in enumerate(seq.ops):
        if op.kind is SeqKind.COPY:
            nxt = [o.kind for o in seq.ops[i + 1:]]
            stop = nxt.index(SeqKind.SYNC) if SeqKind.SYNC in nxt else len(nxt)
            assert op.loaded == (SeqKind.HOST_COMPUTE in nxt[:stop])
    assert S.stream_check(seq).single_stream
    assert S.from_record(S.to_record(seq)) == seq


# --- a real PyTorch CPU trace with CUDA calls grafted on --------------------------

def test_containment_on_a_real_profiler_trace(tmp_path):
    from torch.profiler import ProfilerActivity, profile, record_function

    a = torch.randn(32, 32)

    def fn():
        with record_function(S.layer_annotation(0)):
            torch.mm(a, a)
        with record_function(S.layer_annotation(1)):
            torch.add(a, a)

    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        with record_function(S.REGION):
            fn()
    path = tmp_path / "cpu.json"
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    targets = [e for e in events if e.get("cat") == "cpu_op"
               and e["name"] in ("aten::mm", "aten::add")]
    assert sorted(e["name"] for e in targets) == ["aten::add", "aten::mm"]
    last_end = max(e["ts"] + e["dur"] for e in targets)
    for corr, e in enumerate(sorted(targets, key=lambda e: e["ts"]), start=1):
        ts = e["ts"] + e["dur"] / 4
        events.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                       "pid": e["pid"], "tid": e["tid"], "ts": ts, "dur": e["dur"] / 4,
                       "args": {"correlation": corr}})
        events.append({"ph": "X", "cat": "kernel", "name": f"k{corr}", "pid": 0,
                       "tid": 7, "ts": last_end + corr, "dur": 1.0,
                       "args": {"correlation": corr, "stream": 7, "device": 0}})
    trace["deviceProperties"] = [{"id": 0, "name": GPU}]
    seq = S.parse_trace(trace, CP, context_length=2048, batch=1)
    assert [(o.aten_op, o.layer) for o in seq.ops] == [("aten::mm", 0), ("aten::add", 1)]
    assert all(o.frame and o.frame.startswith("test_components_op_sequence.py(")
               and o.frame.endswith(": fn") for o in seq.ops)
