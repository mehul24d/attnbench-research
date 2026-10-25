"""T2.3's recording side, on the CPU: real profiler traces, no CUDA.

What runs here is the annotation contract (`op_sequence` reads what these
helpers write) and the zoo's bookkeeping. The zoo itself waits for a GPU,
so its run is the GPU session's (the T2.3 report lists it).
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest
import torch

from attnbench.components import op_sequence as S
from attnbench.components import trace_recording as T
from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp


def _record(tmp_path, model, *, warmup=1, host=False):
    calls = []
    x = torch.randn(2, 8)

    def run():
        calls.append(1)
        if host:
            with T.host_compute():
                torch.ones(4).sum()
        model(x)

    path = T.record_prefill_trace(run, tmp_path / "t.json", layers=list(model),
                                  warmup=warmup, cuda=False)
    return json.loads(path.read_text())["traceEvents"], calls


def _spans(events, cat):
    return [e for e in events if e.get("cat") == cat and e.get("ph") == "X"]


def _inside(inner, outer):
    return outer["ts"] <= inner["ts"] and inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"]


def test_one_region_with_each_layer_inside_it(tmp_path):
    model = torch.nn.Sequential(*(torch.nn.Linear(8, 8) for _ in range(3)))
    events, calls = _record(tmp_path, model, warmup=2)
    assert len(calls) == 3   # two warm-up runs, one traced
    notes = _spans(events, "user_annotation")
    (region,) = [e for e in notes if e["name"] == S.REGION]
    layers = sorted((e for e in notes if S.LAYER_PATTERN.match(e["name"])),
                    key=lambda e: e["ts"])
    assert [e["name"] for e in layers] == [S.layer_annotation(i) for i in range(3)]
    assert all(_inside(e, region) for e in layers)
    # Each layer's linear op is inside that layer's annotation and no other.
    linears = sorted((e for e in _spans(events, "cpu_op") if e["name"] == "aten::linear"),
                     key=lambda e: e["ts"])
    assert len(linears) == 3
    for i, op in enumerate(linears):
        assert [j for j, l in enumerate(layers) if _inside(op, l)] == [i]


def test_host_compute_is_annotated_inside_the_region(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    events, _ = _record(tmp_path, model, host=True)
    notes = _spans(events, "user_annotation")
    (region,) = [e for e in notes if e["name"] == S.REGION]
    (hc,) = [e for e in notes if e["name"] == S.HOST_COMPUTE]
    assert _inside(hc, region)


def test_the_parser_refuses_a_cpu_only_trace(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    _record(tmp_path, model)
    cp = CodePath(arm="dense", gpu="NVIDIA L4", implementation="eager")
    with pytest.raises(S.SequenceError, match="lists no CUDA device"):
        S.parse_trace(tmp_path / "t.json", cp, context_length=8, batch=2)


def test_hooks_are_removed_afterwards_even_after_an_error():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    with T.annotate_layers(list(model)):
        assert all(len(m._forward_pre_hooks) == 1 and len(m._forward_hooks) == 1
                   for m in model)
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model)
    with pytest.raises(RuntimeError):
        with T.annotate_layers(list(model)):
            model(torch.randn(2, 3))   # wrong width
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model)


def test_profiler_settings():
    cpu, cuda = T.profiler_kwargs(False), T.profiler_kwargs(True)
    assert cpu["activities"] == [torch.profiler.ProfilerActivity.CPU]
    assert cuda["activities"] == [torch.profiler.ProfilerActivity.CPU,
                                  torch.profiler.ProfilerActivity.CUDA]
    assert cpu["with_stack"] and not cpu["record_shapes"] and not cpu["profile_memory"]


def test_negative_warmup_is_refused(tmp_path):
    with pytest.raises(ValueError, match="warmup"):
        T.record_prefill_trace(lambda: None, tmp_path / "t.json", warmup=-1, cuda=False)


# --- the zoo ----------------------------------------------------------------------

def test_the_zoos_need_a_cuda_device():
    with pytest.raises(RuntimeError, match="needs a CUDA device"):
        T.run_mechanism_zoo("cpu")
    with pytest.raises(RuntimeError, match="needs a CUDA device"):
        T.run_two_stream_zoo("cpu")


def test_the_zoo_runs_one_layer_per_step_in_order():
    """The run and the answer key cannot drift apart: each `with layer(i)`
    in the run is one ZOO step, numbered 0, 1, 2, ..."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(T.run_mechanism_zoo)))
    used = [node.items[0].context_expr.args[0].value
            for node in ast.walk(tree) if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and getattr(node.items[0].context_expr.func, "id", None) == "layer"]
    assert used == list(range(len(T.ZOO)))
    assert [s.layer for s in T.ZOO] == list(range(len(T.ZOO)))


def _op(sig, layer, composite=None):
    if sig[0] == "sync":
        return SeqOp(SeqKind.SYNC, layer=layer, mechanism=sig[1], composite=composite)
    if sig[0] == "copy":
        return SeqOp(SeqKind.COPY, layer=layer, direction=sig[1], loaded=sig[2])
    return SeqOp(SeqKind(sig[0]), layer=layer)


def _zoo_sequence(pick=0, overrides=None):
    ops = []
    for step in T.ZOO:
        ops.append(SeqOp(SeqKind.KERNEL, layer=step.layer, kernels=("k",)))
        alt = (overrides or {}).get(step.layer, step.expected[min(pick, len(step.expected) - 1)])
        ops += [_op(sig, step.layer, step.composite if sig[0] == "sync" else None)
                for sig in alt]
    cp = CodePath(arm="zoo", gpu="NVIDIA L4", implementation="eager")
    return OpSequence(cp, 0, 1, tuple(ops), (7,))


def test_check_zoo_accepts_every_listed_alternative():
    for pick in range(3):
        assert all(r.ok for r in T.check_zoo(_zoo_sequence(pick)))


def test_check_zoo_catches_a_wrong_mechanism_and_a_missing_layer():
    res = T.check_zoo(_zoo_sequence(overrides={2: (("sync", "item"),)}))
    assert [r.step.layer for r in res if not r.ok] == [2]
    seq = _zoo_sequence()
    seq = OpSequence(seq.code_path, 0, 1, tuple(o for o in seq.ops if o.layer != 6), (7,))
    assert [r.step.layer for r in T.check_zoo(seq) if not r.ok] == [6]


def test_check_zoo_checks_composites_both_ways():
    seq = _zoo_sequence()
    ops = [o if not (o.layer == 4 and o.kind is SeqKind.SYNC) else
           SeqOp(SeqKind.SYNC, layer=4, mechanism=o.mechanism) for o in seq.ops]
    res = T.check_zoo(OpSequence(seq.code_path, 0, 1, tuple(ops), (7,)))
    assert [r.step.layer for r in res if not r.ok] == [4]
    ops = [o if not (o.layer == 1 and o.kind is SeqKind.SYNC) else
           SeqOp(SeqKind.SYNC, layer=1, mechanism="item", composite="aten::nonzero")
           for o in seq.ops]
    res = T.check_zoo(OpSequence(seq.code_path, 0, 1, tuple(ops), (7,)))
    assert [r.step.layer for r in res if not r.ok] == [1]
