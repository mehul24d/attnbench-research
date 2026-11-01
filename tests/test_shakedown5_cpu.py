"""Shakedown 5's failures (1 Nov 2026), on the real classes behind the
stubbed GPU boundary (`attnbench.cpu_boundary`).

  - The launch check refused XAttention's 20480-token probe: in its chunked
    path a boolean `index_put_` with a scalar copies the scalar to the
    device, clones its target, and launches the index kernel itself, which
    the stand-in did not predict. The check also took 4 minutes per trace.
  - Compiled XAttention under `compiled_default` failed on its second
    prefill: the per-layer threshold row it caches was built inside the
    compiled graph, so the cache held a CUDA-graph output that a later
    replay overwrote.
"""

from __future__ import annotations

import random

import pytest
import torch

from attnbench import cpu_boundary as cb
from attnbench.components import op_sequence as S

pytestmark = pytest.mark.skipif(torch.cuda.is_available(),
                                reason="the CPU boundary is for a machine without CUDA")


def _op(name, ts, dur, dims=None):
    args = {} if dims is None else {"Input Dims": dims}
    return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": 1, "ts": ts,
            "dur": dur, "args": args}


def _launch(ts, api="cudaLaunchKernel"):
    return {"ph": "X", "cat": "cuda_runtime", "name": api, "pid": 1, "tid": 1, "ts": ts,
            "dur": 1, "args": {"correlation": int(ts * 10)}}


def _region(ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": "attnbench::prefill", "pid": 1,
            "tid": 1, "ts": ts, "dur": dur, "args": {}}


def test_a_boolean_index_put_launches_beside_its_preparing_children_a_long_one_does_not():
    # As the L4 recorded them at 20480 tokens (layer 0): the boolean form
    # copies the scalar, clones the target and launches the index kernel;
    # the long form hands the write to `masked_fill_`.
    boolean = [_op("aten::index_put_", 0, 100, [[1, 768, 192], [], [], []]),
               _op("aten::_index_put_impl_", 1, 98, [[1, 768, 192], [], [], [], []]),
               _op("aten::to", 2, 10, [[]]), _op("aten::_to_copy", 3, 8, [[]]),
               _op("aten::copy_", 4, 6, [[], [], []]), _launch(5, "cudaMemcpyAsync"),
               _op("aten::clone", 20, 20, [[1, 1, 768, 192], []]),
               _op("aten::copy_", 25, 10, [[1, 1, 768, 192], [1, 1, 768, 192], []]), _launch(26),
               _launch(60)]
    long = [_op("aten::index_put_", 200, 50, [[1, 12, 160, 160], [], [], []]),
            _op("aten::_index_put_impl_", 201, 48, [[1, 12, 160, 160], [], [], [], []]),
            _op("aten::masked_fill_", 210, 20, [[1, 12, 160, 160], [1, 12, 160, 160], []]),
            _launch(215)]
    got = {(p.event["name"], p.event["ts"]): p.issues for p in cb.predict(boolean + long)}
    assert got[("aten::_index_put_impl_", 1)] == "kernel"
    assert got[("aten::index_put_", 0)] is None
    assert got[("aten::_index_put_impl_", 201)] is None
    assert got[("aten::masked_fill_", 210)] == "kernel"
    r = cb.launch_check({"traceEvents": [_region(0, 300), *boolean, *long]})
    assert r["passed"], r


def _nested_trace(rng, n):
    """Random properly nested operators on one thread, launches anywhere."""
    ev = []

    def build(start, end, depth):
        t = start
        while t < end - 4 and len(ev) < n:
            dur = rng.uniform(1, min(40, end - t - 1))
            ev.append(_op(f"aten::op{rng.randrange(5)}", t, dur, [[2]]))
            if depth < 4 and dur > 6 and rng.random() < 0.5:
                build(t + 1, t + dur - 1, depth + 1)
            if rng.random() < 0.6:
                ev.append(_launch(t + rng.uniform(0, dur - 1)))
            t += dur + rng.uniform(0, 3)
    build(0, 20000, 0)
    for _ in range(50):                                   # some between operators
        ev.append(_launch(rng.uniform(0, 20000)))
    return ev


def test_the_innermost_holder_of_each_launch_is_the_brute_force_one():
    rng = random.Random(5)
    end = lambda e: e["ts"] + e.get("dur", 0)
    within = lambda e, r: r["ts"] - S._EPS <= e["ts"] and end(e) <= end(r) + S._EPS
    for _ in range(5):
        ev = _nested_trace(rng, 600)
        ops = cb._cpu_ops(ev)
        parent, _ = cb._nesting(ops)
        calls = [e for e in ev if e["cat"] == "cuda_runtime"]
        want: dict = {}
        for c in calls:
            inner = [e for e in ops if within(c, e)]
            if inner:
                k = id(max(inner, key=lambda e: (e["ts"], -e.get("dur", 0))))
                want[k] = want.get(k, 0) + 1
        assert cb._holders(ops, parent, calls, within) == want
        assert sum(want.values()) > 100


def test_a_compiled_forward_builds_xattention_s_threshold_row_outside_the_graph(monkeypatch):
    from attnbench.backends import compiled
    from attnbench.backends.xattention import ThresholdTable, XAttentionBackend
    compiled.reset_compiled_state()
    try:
        table = ThresholdTable(name="t", values=((0.9, 0.8), (0.7, 0.6)))
        arm = XAttentionBackend(threshold=table)
        seen = []

        def forward_and_mask(q, k, v, cfg, mask=None):
            seen.append((torch.compiler.is_compiling(), dict(arm._table_cache)))
            return q + 1, torch.ones(1, 1, 2, 2, dtype=torch.bool)

        built = []
        real = arm._threshold_for
        monkeypatch.setattr(arm, "_threshold_for", lambda q, h: built.append(
            torch.compiler.is_compiling()) or real(q, h))
        monkeypatch.setattr(arm, "forward_and_mask", forward_and_mask)
        c = compiled.CompiledBackend(arm, "default", mode=None, dynamo_backend="eager")
        q = torch.zeros(1, 2, 4, 8)
        for layer in (0, 1, 0):
            c.set_layer(layer)
            c.forward(q, q, q, None)
        assert built == [False, False, False]            # never inside the graph
        assert [sorted(cache) for _, cache in seen] == [
            [(0, "cpu")], [(0, "cpu"), (1, "cpu")], [(0, "cpu"), (1, "cpu")]]
        assert torch.equal(arm._table_cache[(1, "cpu")], torch.tensor([0.7, 0.6]))
    finally:
        compiled.reset_compiled_state()


def test_an_arm_without_cached_state_is_compiled_as_before():
    from attnbench.backends import compiled
    compiled.reset_compiled_state()
    try:
        class Plain:
            class capability:
                name = "plain"

            def forward(self, q, k, v, cfg, mask=None):
                return q * 2
        c = compiled.CompiledBackend(Plain(), "default", mode=None, dynamo_backend="eager")
        q = torch.ones(2)
        assert torch.equal(c.forward(q, q, q, None), q * 2)
    finally:
        compiled.reset_compiled_state()


def test_compiled_xattention_calls_the_kernel_interface_outside_dynamo(monkeypatch):
    # Shakedown 5: traced, the interface's resume frame recompiled on every
    # call and a timed loop reached the recompile limit. A stand-in module
    # records whether Dynamo was tracing when the interface ran.
    import sys
    import types
    from attnbench.backends import compiled
    from attnbench.backends.xattention import XAttentionBackend
    from attnbench.config import AttnConfig
    tracing = []

    def block_sparse_attn_func(q, k, v, *args, **kwargs):
        tracing.append(torch.compiler.is_compiling())
        return q.clone()

    monkeypatch.setitem(sys.modules, "block_sparse_attn",
                        types.SimpleNamespace(block_sparse_attn_func=block_sparse_attn_func))
    arm = XAttentionBackend()
    monkeypatch.setattr(arm, "estimate", lambda q, k: torch.ones(
        1, q.shape[1], 1, 1, dtype=torch.bool))
    compiled.reset_compiled_state()
    try:
        c = compiled.CompiledBackend(arm, "default", mode=None, dynamo_backend="eager")
        q = torch.zeros(1, 2, 8, 4)
        cfg = AttnConfig(seq_len=8, batch=1, n_heads_q=2, n_heads_kv=2, head_dim=4)
        for _ in range(3):
            assert c.forward(q, q, q, cfg).shape == q.shape
        assert tracing == [False, False, False]
    finally:
        compiled.reset_compiled_state()
