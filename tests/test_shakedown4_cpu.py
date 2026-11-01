"""Shakedown 4's failures and the researcher's rulings of 1 Nov 2026, on the
real classes behind the stubbed GPU boundary (`attnbench.cpu_boundary`).

  - Empty calls: XAttention's `causal_mask[:, :, :, chunk_end:] = -inf`
    (`Xattention.py:221`) with `chunk_end` at the end is a `fill_` on an
    empty slice. CUDA launches nothing for it; the null pass nulled it as a
    launch and the CPU stand-in synthesized a kernel for it, so the dry run
    agreed with itself and the L4 refused step 5. The stand-in now reads
    shapes, and a GPU session checks its prediction against the trace.
  - Kernel splits: the t22 pass's three `addmm` calls launched 2, 2 and 1
    kernels (a cuBLAS memset beside two GEMMs); positions are aligned to
    calls by recorded call instance.
"""

from __future__ import annotations

import pytest
import torch

from attnbench import cpu_boundary as cb
from attnbench.components import host_profile as hp
from attnbench.components import recorder
from attnbench.components.op_sequence import CodePath, SeqKind

pytestmark = pytest.mark.skipif(torch.cuda.is_available(),
                                reason="the CPU boundary is for a machine without CUDA")
CP = CodePath(arm="empty-slice", gpu=cb.STUB_DEVICE_NAME, implementation="torch")


class MaskLayer(torch.nn.Module):
    """XAttention's causal-mask fill: the slice past `end` is empty when
    `end` is the mask's width (the last chunk), and not otherwise."""

    def __init__(self, end):
        super().__init__()
        self.end = end

    def forward(self, x):
        m = torch.zeros(1, 1, 4, 8, device="cuda")
        m[:, :, :, self.end:] = float("-inf")
        return x + m.sum()


def _record(tmp_path, layers):
    def prefill():
        x = torch.ones(3, device="cuda")
        for layer in layers:
            x = layer(x)
    return prefill, recorder.record_arm_prefill(prefill, tmp_path, code_path=CP,
                                                context_length=8, batch=1,
                                                prompt_id="p", layers=layers)


def test_a_fill_on_an_empty_slice_is_no_kernel_in_the_stand_in_s_trace(tmp_path):
    with cb.installed(xattn=False):
        _, rec = _record(tmp_path, [MaskLayer(end=6), MaskLayer(end=8)])
    # The assignment is a fill_ issued by itself (a CPU fill_ with a tensor
    # value reaches copy_); only layer 0's slice is not empty.
    assigned = [op.layer for op in rec.sequence.ops if op.chain[:1] == ("aten::fill_",)]
    assert assigned == [0]
    assert all(len(op.calls) == len(op.chain) for op in rec.sequence.ops if op.chain)


def test_the_null_pass_meets_the_empty_fill_as_no_launch_and_flags_where_it_is_priced(tmp_path):
    with cb.installed(xattn=False):
        layers = [MaskLayer(end=6), MaskLayer(end=8)]
        prefill, rec = _record(tmp_path, layers)
        p = hp.profile(prefill, rec.tape, rec.sequence,
                       device=hp.NullDevice(launch=lambda: None, device_type="cpu"), reps=2,
                       layers=layers)
    issued = [i for i, op in enumerate(rec.sequence.ops) if op.is_device]
    assert set(issued) <= set(p.host_s)
    (empty,) = p.empty_calls
    assert (empty["operator"], empty["layer"]) == ("aten::fill_", 1)
    # Its dispatch is in the next launching call's gap: layer 1's sum.
    nxt = rec.sequence.ops[empty["position"]]
    assert (nxt.layer, nxt.chain[0]) == (1, "aten::sum")


def test_before_the_ruling_the_stand_in_launched_the_empty_fill_and_the_dry_run_hid_it(
        tmp_path, monkeypatch):
    # The failure mode, held: with shapes ignored the stand-in's trace has a
    # kernel the L4 never launches, and the pass (empty rule off) issues it,
    # so the CPU run agrees with itself.
    monkeypatch.setattr(cb, "empty_call", lambda e: False)
    monkeypatch.setattr(hp, "all_empty", lambda out: False)
    with cb.installed(xattn=False):
        layers = [MaskLayer(end=8)]
        prefill, rec = _record(tmp_path, layers)
        p = hp.profile(prefill, rec.tape, rec.sequence,
                       device=hp.NullDevice(launch=lambda: None, device_type="cpu"), reps=1,
                       layers=layers)
    assert [op.layer for op in rec.sequence.ops if op.chain[:1] == ("aten::fill_",)] == [0]
    assert p.empty_calls == ()


# --- the stand-in's prediction and the GPU session's check ---------------------------------

def _op(name, ts, dur, dims=None, tid=1):
    args = {} if dims is None else {"Input Dims": dims}
    return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": tid, "ts": ts,
            "dur": dur, "args": args}


def _launch(ts, api="cudaLaunchKernel", tid=1):
    return {"ph": "X", "cat": "cuda_runtime", "name": api, "pid": 1, "tid": tid, "ts": ts,
            "dur": 1, "args": {"correlation": int(ts * 10)}}


def _region(ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": "attnbench::prefill", "pid": 1,
            "tid": 1, "ts": ts, "dur": dur, "args": {}}


def test_an_empty_call_is_read_from_the_recorded_shapes():
    assert cb.empty_call(_op("aten::fill_", 0, 1, [[1, 1, 4, 0], []]))         # in place
    assert not cb.empty_call(_op("aten::fill_", 0, 1, [[1, 1, 4, 2], []]))
    assert cb.empty_call(_op("aten::add", 0, 1, [[0, 3], [0, 3], []]))          # all inputs
    assert not cb.empty_call(_op("aten::cat", 0, 1, [[0, 3], [2, 3]]))
    assert not cb.empty_call(_op("aten::fill_", 0, 1))                            # no shapes
    assert not cb.empty_call(_op("aten::add", 0, 1, [[], []]))                   # scalars


def test_an_operator_that_launches_beside_its_children_is_predicted_to():
    # Shakedown 4's L4: `sort` copies its input (a child copy_) and sorts; a
    # `sum` that redispatches to itself launches only in the inner call.
    ev = [_op("aten::sort", 0, 10), _op("aten::copy_", 1, 2),
          _op("aten::sum", 20, 10), _op("aten::sum", 21, 8), _op("aten::copy_", 22, 2)]
    got = {(p.event["name"], p.event["ts"]): p.issues for p in cb.predict(ev)}
    assert got == {("aten::sort", 0): "kernel", ("aten::copy_", 1): "kernel",
                   ("aten::sum", 20): None, ("aten::sum", 21): "kernel",
                   ("aten::copy_", 22): "kernel"}


def test_the_gpu_check_passes_when_the_launches_are_the_predicted_ones():
    trace = {"traceEvents": [
        _region(0, 100),
        _op("aten::mm", 1, 10, [[2, 2], [2, 2]]), _launch(5),
        _op("aten::fill_", 20, 5, [[1, 0], []]),                    # empty: nothing
        _op("aten::addmm", 30, 10, [[2], [2, 2], [2, 2]]), _launch(31, "cudaMemsetAsync"),
        _launch(35),
        _op("aten::_local_scalar_dense", 50, 5, [[]])]}             # scalar reads: left out
    r = cb.launch_check(trace)
    assert r["passed"] and r["mismatches"] == 0 and r["checked_calls"] == 3


def test_the_gpu_check_refuses_a_launch_the_stand_in_did_not_predict_and_one_it_did_wrongly():
    trace = {"traceEvents": [
        _region(0, 100),
        _op("aten::fill_", 1, 5, [[1, 4], []]),                     # predicted, none recorded
        _op("aten::cumsum", 10, 10, [[4]]), _op("aten::copy_", 11, 2, [[4], [4]]),
        _launch(12), _launch(15),                                  # beside its child: fine
        _op("aten::where", 30, 10, [[4], [4]]), _op("aten::copy_", 31, 2, [[4], [4]]),
        _launch(32), _launch(35)]}                                 # where's own launch: missed
    r = cb.launch_check(trace)
    assert not r["passed"] and r["mismatches"] == 2
    assert r["by_operator"] == {
        "aten::fill_: predicted kernel, recorded 0 call(s)": 1,
        "aten::where: predicted nothing, recorded 1 call(s)": 1}


def test_the_gpu_check_leaves_out_host_compute_and_the_named_host_side_functions():
    def py(name, ts, dur):
        return {"ph": "X", "cat": "python_function", "name": name, "pid": 1, "tid": 1,
                "ts": ts, "dur": dur, "args": {}}
    trace = {"traceEvents": [
        _region(0, 100),
        {"ph": "X", "cat": "user_annotation", "name": "attnbench::host_compute", "pid": 1,
         "tid": 1, "ts": 1, "dur": 10, "args": {}},
        _op("aten::topk", 2, 5, [[16, 16]]),                       # host tensors
        py("model.py(411): _mask_from_scores", 20, 20), _op("aten::mean", 21, 5, [[2, 4, 4]]),
        _op("aten::mm", 50, 10, [[2, 2], [2, 2]]), _launch(55)]}
    assert cb.launch_check(trace)["passed"]
    trace["traceEvents"].append(_op("aten::topk", 70, 5, [[16, 16]]))
    assert not cb.launch_check(trace)["passed"]


def test_the_gpu_check_needs_shapes():
    trace = {"traceEvents": [_region(0, 100), _op("aten::mm", 1, 10), _launch(5)]}
    with pytest.raises(cb.LaunchMismatch, match="no shapes"):
        cb.launch_check(trace)


def test_a_gpu_recording_refuses_when_the_stand_in_predicted_other_launches(tmp_path):
    # The recorder runs the check where it is asked to (on a GPU, by default)
    # and writes it before refusing. Here the stand-in's own trace is
    # checked against a copy with one predicted kernel taken out.
    import json as _json
    with cb.installed(xattn=False):
        layers = [MaskLayer(end=6)]
        prefill, rec = _record(tmp_path / "ok", layers)
        assert not (tmp_path / "ok" / "launch_check.json").exists()      # CPU: not asked

        def drop_one(run, path, **kw):
            p = recorder.record_prefill_trace(run, path, **kw)
            doc = _json.loads(p.read_text())
            first = next(i for i, e in enumerate(doc["traceEvents"])
                         if e.get("name") == "cudaLaunchKernel")
            del doc["traceEvents"][first]
            p.write_text(_json.dumps(doc))
            return p
        drop_one.profiler_warmup = getattr(recorder.record_prefill_trace, "profiler_warmup", 0)
        with pytest.raises(recorder.RecordingError, match="predicted other launches"):
            recorder.record_arm_prefill(prefill, tmp_path / "bad", code_path=CP, context_length=8,
                                        batch=1, prompt_id="p", layers=layers,
                                        record_trace=drop_one, launch_check=True,
                                        parse=lambda *a, **k: rec.sequence)
        check = _json.loads((tmp_path / "bad" / "launch_check.json").read_text())
        assert check["mismatches"] == 1 and not check["passed"]
        recorder.record_arm_prefill(prefill, tmp_path / "good", code_path=CP, context_length=8,
                                    batch=1, prompt_id="p", layers=layers, launch_check=True)
        good = _json.loads((tmp_path / "good" / "launch_check.json").read_text())
        assert good["passed"] and good["checked_calls"] > 0


# --- compiled density: the mask is an output of the compiled function --------------------

class ReusingArm:
    """An arm whose mask comes back in one buffer it rewrites every call, as
    a CUDA-graph output is rewritten by the next replay."""

    class capability:
        name = "reusing"

    def __init__(self):
        self.buf = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        self.calls = 0

    def forward_and_mask(self, q, k, v, cfg, mask=None):
        self.calls += 1
        self.buf.zero_()
        self.buf[..., : self.calls] = True          # layer n keeps n of 4 columns
        return q + 1, self.buf


@pytest.fixture
def fresh_dynamo():
    from attnbench.backends import compiled
    compiled.reset_compiled_state()
    yield compiled
    compiled.reset_compiled_state()


def test_the_twin_pass_clones_each_mask_before_the_next_forward_and_marks_each_step(
        fresh_dynamo, monkeypatch):
    marks = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(1))
    arm = ReusingArm()
    c = fresh_dynamo.CompiledBackend(arm, "default", mode=None, dynamo_backend="eager")
    q = torch.zeros(2)
    c.forward(q, q, q, None)                       # timed: no mark, no record
    assert marks == [] and c.last_layer_density == []
    with c.twin_pass():
        for _ in range(3):
            c.forward(q, q, q, None)
    assert len(marks) == 3
    from attnbench.backends.xattention import density_tensor
    # Calls 2, 3, 4 kept 2, 3, 4 columns of the one buffer, read after it was
    # rewritten: each entry is its own call's.
    want = [float(density_tensor((torch.arange(4) < n).expand(1, 1, 4, 4).contiguous(), 4, 4))
            for n in (2, 3, 4)]
    assert [float(d) for d in c.last_layer_density] == pytest.approx(want)
    assert len(set(want)) == 3
    c.reset_density()
    assert c.last_layer_density == []


def test_the_twin_pass_reads_the_density_from_a_copy_never_from_the_output_buffer(
        fresh_dynamo, monkeypatch):
    # A CPU cannot rewrite a CUDA-graph output; what it can hold is that the
    # density is never read from the output itself.
    from attnbench.backends import xattention
    arm = ReusingArm()
    read = []
    real = xattention.density_tensor
    monkeypatch.setattr(xattention, "density_tensor",
                        lambda m, *a: read.append(m) or real(m, *a))
    c = fresh_dynamo.CompiledBackend(arm, "default", mode=None, dynamo_backend="eager")
    q = torch.zeros(2)
    with c.twin_pass():
        c.forward(q, q, q, None)
    (m,) = read
    assert m is not arm.buf and m.data_ptr() != arm.buf.data_ptr()


def test_the_a_b_side_without_the_mask_output_keeps_no_density_record(fresh_dynamo):
    c = fresh_dynamo.CompiledBackend(ReusingArm(), "default", mode=None, dynamo_backend="eager",
                                     mask_output=False)
    q = torch.zeros(2)
    assert torch.equal(c.forward(q, q, q, None), q + 1)
    assert not hasattr(c, "last_layer_density") and not hasattr(c, "reset_density")
    assert not c.mask_output


COMPILED_XATTENTION = """
import json, sys, tempfile
from attnbench import cpu_boundary as cb, ground_truth as gt
from attnbench.backends import compiled
from attnbench.components import shakedown_arms as sa
with cb.installed():
    h = sa.Harness(device="cpu", dry_run=True, score_cache_dir=tempfile.mkdtemp())
    be = compiled.CompiledBackend(h._backend(sa.ARMS["xattention"]), "default",
                                  **cb.compile_kwargs())
    wrapped, ids, _, arms, score_kwargs = h.timing_arms(
        ["dense", "xattention"], 1024, densities=[None, None], backends=[None, be],
        compile_settings=[None, "default"])
    a = arms[1]
    kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)

    def run(backend=None):
        wrapped.run_measured(ids, backend if backend is not None else a.backend, cfg=a.cfg,
                             logits_to_keep=1, **kw)
    run()
    print(json.dumps(gt.kernel_densities(a, run)))
"""


@pytest.mark.skipif(not cb.have_xattn(), reason="needs the x-attention clone (ATTNBENCH_XATTN_DIR)")
def test_the_density_probe_reads_compiled_xattention_s_mask_outputs_on_the_real_arm():
    """Shakedown 4, t228 `density`: "accessing tensor output of CUDAGraphs
    that has been overwritten" at `last_layer_density.append(...)` inside the
    compiled forward. The CPU cannot reuse a CUDA-graph buffer; what is
    held here is that the real arm's record now comes from the compiled
    function's outputs, in the twin pass, and agrees with the eager twin."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", COMPILED_XATTENTION], cwd=root,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    row = json.loads(r.stdout.strip().splitlines()[-1])
    assert 0.0 < row["mask_density"] <= 1.0
    assert row["density_vs_eager"] == pytest.approx(0.0, abs=1e-12)


def test_a_launch_beside_children_is_placed_after_them_whatever_the_gaps():
    # Found by the CPU dry run's step 6: placed in the widest uncovered gap,
    # the sort's own launch fell before or after its copy by timing alone, so
    # the replay's kernel order differed from the recording's.
    for before, after in ((8, 1), (1, 8)):
        ev = [_op("aten::sort", 0, 2 + before + after),
              _op("aten::copy_", before, 2, [[4], [4]])]
        doc = cb.synthesize_device({"traceEvents": ev})
        kernels = sorted((e for e in doc["traceEvents"] if e.get("cat") == "kernel"),
                         key=lambda e: e["ts"])
        assert [k["name"] for k in kernels] == ["cpu_stub::aten::copy_", "cpu_stub::aten::sort"]
