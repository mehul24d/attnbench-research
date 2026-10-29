"""Fixtures for the figure and table tests (C5): a hand-written Kineto trace
whose every time is chosen by hand, and a committed P1 prediction loaded the
way the analysis loads it, from `test_predict_commit`'s own helpers."""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path

from attnbench.components import op_sequence as ops
from attnbench.predict.commit import load_committed

from test_predict_commit import CP, MODEL, NO_SKEW, OPS_HOST, T0, Clock, MemoryStore, cell, \
    committed, p1_record

PID, TID, DEV, STREAM = 11, 12, 0, 7
GPU = "NVIDIA L4"


def _x(cat, name, ts, dur, pid=PID, tid=TID, **args):
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid,
            "ts": ts, "dur": dur, "args": args}


def _dev(cat, name, ts, dur, corr, **args):
    return _x(cat, name, ts, dur, pid=DEV, tid=STREAM, correlation=corr, stream=STREAM,
              device=DEV, **args)


def _call(name, ts, dur, corr):
    return _x("cuda_runtime", name, ts, dur, correlation=corr, cbid=0)


def code_path(gpu: str = GPU) -> ops.CodePath:
    return ops.CodePath(arm="oracle", gpu=gpu, implementation="oracle:torch",
                        libraries=(("torch", "2.8.0"),))


def trace(gpu: str = GPU) -> dict:
    """Layer 0: one kernel. Layer 1 (annotation at 200 us, 300 us long):
    the mask build in host_compute (210-250), a GEMM launch, an `.item()`
    (copy and wait, folded into one sync), and a softmax launch whose kernel
    runs past the layer's end. Then the closing synchronise."""
    ev = [
        _x("user_annotation", ops.REGION, 0, 1000),
        _x("user_annotation", ops.layer_annotation(0), 10, 100),
        _x("cpu_op", "aten::add", 20, 10),
        _call("cudaLaunchKernel", 21, 5, 10),
        _dev("kernel", "add_kernel", 30, 20, 10),
        _x("user_annotation", ops.layer_annotation(1), 200, 300),
        _x("user_annotation", ops.HOST_COMPUTE, 210, 40),
        _x("cpu_op", "aten::mm", 255, 10),
        _call("cudaLaunchKernel", 256, 5, 11),
        _dev("kernel", "gemm", 270, 50, 11),
        _x("cpu_op", "aten::item", 270, 60),
        _x("cpu_op", "aten::_local_scalar_dense", 271, 58),
        _call("cudaMemcpyAsync", 272, 4, 12),
        _dev("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", 320, 2, 12, bytes=4),
        _call("cudaStreamSynchronize", 277, 50, 13),
        _x("cpu_op", "aten::softmax", 339, 10),
        _call("cudaLaunchKernel", 340, 5, 14),
        _dev("kernel", "softmax", 350, 200, 14),
        _call("cudaDeviceSynchronize", 600, 10, 15),
    ]
    return {"schemaVersion": 1,
            "deviceProperties": [{"id": DEV, "name": gpu, "computeMajor": 8,
                                  "computeMinor": 9}],
            "traceEvents": ev}


def with_event(doc: dict, event: dict) -> dict:
    d = copy.deepcopy(doc)
    d["traceEvents"].append(event)
    return d


def parse(doc: dict) -> ops.OpSequence:
    gpu = doc["deviceProperties"][0]["name"]
    return ops.parse_trace(doc, code_path(gpu), context_length=16384, batch=1)


def write_pair(tmp_path: Path, doc: dict, name: str = "a", seq=None) -> tuple[Path, Path]:
    """The trace and its sequence record, as T2.27's step 11 banks them."""
    t = tmp_path / f"fig1_{name}_trace.json"
    t.write_text(json.dumps(doc))
    rec = ops.to_record(seq if seq is not None else parse(doc))
    s = tmp_path / f"fig1_{name}_sequence.json"
    s.write_text(json.dumps({**rec, "sha256": ops.record_sha256(rec)}))
    return t, s


# --- a committed prediction, loaded ---------------------------------------------------

SYNC_AT = T0 + timedelta(seconds=5)
STARTED_AT = T0 + timedelta(seconds=60)
XATTN = (MODEL, CP, "xattn", 0.5, 4, 1)      # recursion 16, sum 13, sum+s0 16, step 14
DENSE = (MODEL, CP, "dense", 1.0, 4, 1)      # recursion 8, sum 8, sum+s0 9, step 7


def loaded_p1(tmp_path: Path, xattn_ops=None):
    """(CommittedPrediction from load_committed, local file, store, remote).
    `xattn_ops` replaces the xattn cell's operations."""
    store = MemoryStore(Clock(T0))
    record = p1_record() if xattn_ops is None else p1_record(
        cells=[cell(ops=xattn_ops), cell(ops=OPS_HOST, arm="dense", density=1.0)])
    f, receipt = committed(tmp_path, record, store, SYNC_AT)
    cp = load_committed(store, receipt.remote, ground_truth_started_at=STARTED_AT,
                        clock_skew=NO_SKEW, banked_at={}, local=f.path)
    return cp, f.path, store, receipt.remote


def ground_truth(xattn_ms=20000.0, dense_ms=10000.0, **over) -> dict:
    """A `time_arms` result: 20 s for xattn and 10 s for dense (medians)."""
    rows = [{"arm": "xattn", "prefill_ms_median": xattn_ms, "prefill_ms_p90": xattn_ms,
             "prefill_ms_p99": xattn_ms},
            {"arm": "dense", "prefill_ms_median": dense_ms, "prefill_ms_p90": dense_ms,
             "prefill_ms_p99": dense_ms}]
    r = {"traced": False, "complete": True,
         "first_timed_block_started_at": STARTED_AT.timestamp(), "rows": rows}
    r.update(over)
    return r
