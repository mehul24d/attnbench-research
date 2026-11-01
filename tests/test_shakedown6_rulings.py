"""The researcher's rulings of 1 Nov 2026 after Shakedown 6.

The kernel interface outside Dynamo for every arm that calls it
(`block_sparse.kernel_interface`): mean-pool and the oracle carry the one
declared break under both settings, and a traced interface is refused as
another code path (behind the CPU boundary).

Once-only host pricing (`host_profile.align_calls`), on Shakedown 6's own
sequence: XAttention at 2048 tokens on the L4, recorded on the instance
during the diagnosis (`results/shakedown6_l4_20261101/results/s6_diag/
s6_diag_fixed/xattention_None_2048/recording/sequence.json`, sha256
11f03cbf...1350). Its null pass was refused there: "the recording's call 65
is to aten::where in layer 0 (position 77, 2 operation(s)), and the pass's
launching call 65 was to aten::scalar_tensor in layer 0". The committed
fixture is that sequence up to the first operation of layer 1 (187 of 4948
operations, unchanged; `tests/fixtures/s6_xattention_2048_layer0.json.gz`);
with `results/` present the whole sequence is tested too.

The pass's calls are the dispatcher's view of the recording: one per
recorded call, at the operator the pass meets it at, and before each
`where` whose recording nests a `scalar_tensor`, that `scalar_tensor`, as
the refusal showed. The conservation test: every pass call is used exactly
once, and the positions' costs sum to the pass's calls plus one floor per
further kernel; the bare prices sum to the bare calls' prices.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from attnbench import cpu_boundary as cb
from attnbench.components import bare_timer as bt
from attnbench.components import host_profile as hp
from attnbench.components import op_sequence as S

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "s6_xattention_2048_layer0.json.gz"
BANKED = (HERE.parent / "results" / "shakedown6_l4_20261101" / "results" / "s6_diag" /
          "s6_diag_fixed" / "xattention_None_2048" / "recording" / "sequence.json")
WHERE_SCALAR = ("aten::where", "aten::scalar_tensor", "aten::fill_")


def _fixture() -> S.OpSequence:
    return S.from_record(json.loads(gzip.decompress(FIXTURE.read_bytes())))


def _banked() -> S.OpSequence:
    return S.from_record(json.loads(BANKED.read_text()))


def _dispatcher_view(seq: S.OpSequence):
    """(isolated positions, the pass's launching calls as (operator, layer))."""
    iso = hp.isolated_positions(seq)
    visible = lambda ch: next((n for n in ch if not cb.composite_implicit(n)), ch[0])
    names = {visible(hp.chain_of(o)) for i, o in enumerate(seq.ops)
             if hp._is_issue(o) and i not in iso}
    met = hp.met_at(seq, names | {"aten::scalar_tensor"}, skip=iso)
    issued = []
    for name, layer, pos in hp.recorded_calls(seq, met, skip=iso):
        issued += [("aten::scalar_tensor", layer) for p in pos
                   if seq.ops[p].chain == WHERE_SCALAR]
        issued.append((name, layer))
    return iso, issued


def _log(issued) -> hp.PassLog:
    log = hp.PassLog()
    for k, (name, layer) in enumerate(issued):
        log.issues.append(hp.Issue(name=name, layer=layer, gap_s=(k + 1) * 1e-6,
                                   launch_s=3e-6))
        log.launches.append(3e-6)
    log.isolated.append(5e-6)            # the embedding's gather, kept out
    return log


def _conserves(seq: S.OpSequence) -> None:
    iso, issued = _dispatcher_view(seq)
    log = _log(issued)
    host, source, _ = hp.align(seq, log, isolated=iso)
    met = hp.met_at(seq, (e.name for e in log.issues), skip=iso)
    groups = hp.align_calls(seq, met, issued, skip=iso)
    # every pass call used exactly once, in order
    assert [i for i, _, _ in groups] == list(range(len(issued)))
    # every priced position in exactly one share
    shares = [p for _, _, share in groups for p in share]
    assert sorted(shares) == sorted(host) and len(shares) == len(set(shares))
    floor = 3e-6
    extra = len(shares) - len(issued)
    assert sum(host.values()) == pytest.approx(
        sum(e.gap_s + e.launch_s for e in log.issues) + floor * extra, rel=1e-12)

    # the bare prices, aligned the same way, sum to the bare calls' prices
    null = hp.HostProfile(host_s={**host, **{i: 5e-6 for i in iso}}, reps_s={},
                          launch_floor_s=floor, reps=1, sequence_sha256="x",
                          cost_source={**{i: s for i, s in source.items() if s != hp.NULL_PASS},
                                       **{i: hp.ISOLATED_REPLAY for i in iso}})
    bp = bt.BarePass(floor_s=1e-6, reps=bt.MIN_REPS)
    bp.calls = [bt.BareCall(n, layer, direct_s=(k + 2) * 1e-6, chain_s=None)
                for k, (n, layer) in enumerate(issued)]
    profile, record = bt.bare_profile(seq, bp, null, aggregate_tol=0.10)
    priced = sum(v for i, v in profile.host_s.items() if i not in iso)
    assert priced == pytest.approx(sum(c.direct_s for c in bp.calls), rel=1e-12)
    assert record["calls_priced"] == len(bp.calls)


def test_the_fixture_holds_shakedown_6s_refusal():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    met = hp.met_at(seq, (n for n, _ in issued), skip=iso)
    calls = hp.recorded_calls(seq, met, skip=iso)
    assert calls[65] == ("aten::where", 0, (77, 78))
    assert seq.ops[77].chain == WHERE_SCALAR
    assert issued[65] == ("aten::scalar_tensor", 0) and issued[66] == ("aten::where", 0)
    assert sum(1 for o in seq.ops if o.chain == WHERE_SCALAR) == 5


def test_each_pass_call_is_priced_once_inside_the_call_that_nests_it():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    met = hp.met_at(seq, (n for n, _ in issued), skip=iso)
    groups = {i: (n, share) for i, n, share in hp.align_calls(seq, met, issued, skip=iso)}
    assert groups[65] == (65, (77,))       # scalar_tensor: its fill_
    assert groups[66] == (65, (78,))       # where: its own kernel
    log = _log(issued)
    host, source, _ = hp.align(seq, log, isolated=iso)
    assert host[77] == pytest.approx(66e-6 + 3e-6) and host[78] == pytest.approx(67e-6 + 3e-6)
    assert source[77] == hp.CHAIN and source[78] == hp.NULL_PASS


def test_conservation_on_the_shakedown_6_fixture():
    _conserves(_fixture())


@pytest.mark.skipif(not BANKED.exists(), reason=f"needs the banked sequence {BANKED}")
def test_conservation_on_the_whole_shakedown_6_sequence():
    seq = _banked()
    assert len(seq.ops) == 4948
    assert sum(1 for o in seq.ops if o.chain == WHERE_SCALAR) == 140
    _conserves(seq)


def test_a_nested_call_with_no_call_of_its_own_after_it_is_still_refused():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    del issued[66]                         # the where after the scalar_tensor
    with pytest.raises(hp.ProfileMismatch, match="call 65 is to aten::where"):
        hp.align(seq, _log(issued), isolated=iso)


def test_a_call_the_recording_does_not_nest_there_is_still_refused():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    issued.insert(65, ("aten::sum", 0))    # sum is not in where's chains
    with pytest.raises(hp.ProfileMismatch, match="call 65 is to aten::where"):
        hp.align(seq, _log(issued), isolated=iso)


def test_no_more_nested_calls_than_the_call_has_further_kernels():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    issued.insert(65, ("aten::scalar_tensor", 0))   # two before a two-kernel call
    with pytest.raises(hp.ProfileMismatch):
        hp.align(seq, _log(issued), isolated=iso)


def test_the_stand_in_gives_scalar_tensor_its_launch():
    # On the L4 every scalar_tensor (140 per XAttention prefill) was under a
    # where, with a fill_ child: a launch. On a CPU it is a leaf.
    assert cb._may_launch("aten::scalar_tensor")


# --- the kernel interface outside Dynamo, for every arm (ruling 1) ---------------------------

def _block_sparse_layers(n=2, s=256):
    import torch
    from attnbench.config import AttnConfig
    from attnbench.masks import BlockSparseMask
    cfg = AttnConfig(seq_len=s, batch=1, n_heads_q=2, n_heads_kv=2, head_dim=16,
                     dtype="float32", mask="block_sparse", sparsity=0.5, block_size=128,
                     mask_source="importance")
    nb = s // 128
    act = torch.tril(torch.ones(nb, nb, dtype=torch.bool))
    mask = BlockSparseMask(seq_len=s, block_size=128, active=act, seed=0,
                           source="importance", causal=True)
    g = torch.Generator().manual_seed(0)
    qkv = lambda: torch.randn(1, 2, s, 16, generator=g)
    return [(i, (qkv(), qkv(), qkv(), mask), cfg) for i in range(n)]


@pytest.mark.parametrize("setting", ["default", "capture"])
def test_mean_pool_and_the_oracle_carry_only_the_declared_kernel_exclusion(setting):
    import torch
    if torch.cuda.is_available():
        pytest.skip("the CPU boundary is for a machine without CUDA")
    from attnbench.backends import compiled as C
    from attnbench.backends.block_sparse import BlockSparseAttention
    with cb.installed(xattn=False):
        arm = BlockSparseAttention()
        report = C.graph_break_report(arm, _block_sparse_layers(), setting)
    assert [len(r.breaks) for r in report] == [1, 1]
    C.check_declared(report, arm, setting)


@pytest.mark.parametrize("setting", ["default", "capture"])
def test_a_traced_kernel_interface_is_another_code_path(setting, monkeypatch):
    # The behaviour before the ruling: the interface traced. Behind the
    # boundary the stub sizes a tensor from its data, as the library does, so
    # the default settings show that break; neither setting shows the
    # declared exclusion.
    import torch
    if torch.cuda.is_available():
        pytest.skip("the CPU boundary is for a machine without CUDA")
    from attnbench.backends import block_sparse as B
    from attnbench.backends import compiled as C
    monkeypatch.setattr(B, "kernel_interface",
                        lambda *a, unsupported=False, **kw: B._call(*a, unsupported=unsupported,
                                                                     **kw))
    with cb.installed(xattn=False):
        arm = B.BlockSparseAttention()
        report = C.graph_break_report(arm, _block_sparse_layers(), setting)
    with pytest.raises(C.CompiledVariantError, match="declared break missing"):
        C.check_declared(report, arm, setting)
    if setting == "default":
        assert any(b.reason.startswith("Dynamic shape operator") for r in report for b in r.breaks)


def test_the_kernel_interface_is_the_librarys_own_call_eagerly(monkeypatch):
    from attnbench.backends import block_sparse as B
    from attnbench.backends.base import UnsupportedConfig
    import types, sys
    seen = []

    def func(*a, **kw):
        seen.append((a, kw))
        if kw.get("fail"):
            raise RuntimeError("no kernel for this")
        return "out"
    monkeypatch.setitem(sys.modules, "block_sparse_attn",
                        types.SimpleNamespace(block_sparse_attn_func=func))
    assert B.kernel_interface(1, 2, is_causal=True) == "out"
    assert seen == [((1, 2), {"is_causal": True})]       # `unsupported` is not passed on
    with pytest.raises(RuntimeError, match="no kernel"):
        B.kernel_interface(fail=True)
    with pytest.raises(UnsupportedConfig, match="block_sparse: no kernel"):
        B.kernel_interface(fail=True, unsupported=True)


# --- the per-layer residual (ruling 3) -----------------------------------------------------

from attnbench.components import layer_residual as lr  # noqa: E402


def _profiles(seq, *, gap=None):
    """A bare profile and a null-pass profile over the fixture's priced
    positions: bare 1 us per position; null-pass 1 us floor plus a gap."""
    iso = set(hp.isolated_positions(seq))
    priced = [i for i, o in enumerate(seq.ops) if hp._is_issue(o)]
    bare = hp.HostProfile(host_s={i: 1e-6 for i in priced}, reps_s={}, launch_floor_s=1e-6,
                          reps=1000, sequence_sha256="s", pricing=bt.BARE_TIMER,
                          cost_source={i: hp.ISOLATED_REPLAY for i in iso})
    gap = gap or (lambda i: (i % 7) * 1e-6)
    null = hp.HostProfile(host_s={i: 1e-6 + gap(i) for i in priced}, reps_s={},
                          launch_floor_s=1e-6, reps=5, sequence_sha256="s",
                          cost_source={i: hp.ISOLATED_REPLAY for i in iso})
    return bare, null


def test_the_residual_is_the_span_less_the_bare_prices_spread_by_the_null_gaps():
    seq = _fixture()
    bare, null = _profiles(seq)
    layer0 = [i for i in bare.host_s if seq.ops[i].layer == 0]
    span = sum(bare.host_s[i] for i in layer0) + 500e-6        # 500 us of the arm's Python
    spans = lr.Spans(spans_s={0: (span - 1e-6, span, span + 1e-6)}, setitems={0: (0, 0, 0)})
    prof, rec = lr.residual(seq, bare, null, spans, spread_tol=0.10)
    assert sum(prof.residual_s.values()) == pytest.approx(500e-6)
    assert prof.total_s == pytest.approx(bare.total_s + 500e-6)
    # in proportion to the null pass's gaps; none outside the layers
    gaps = {i: null.host_s[i] - null.launch_floor_s for i in layer0}
    for i in layer0:
        assert prof.residual_s.get(i, 0.0) == pytest.approx(500e-6 * gaps[i] / sum(gaps.values()))
    assert all(seq.ops[i].layer == 0 for i in prof.residual_s)
    assert rec["labels"] == [] and rec["layers"][0]["residual_s"] == pytest.approx(500e-6)
    assert hp.from_record(hp.to_record(prof)).residual_s == prof.residual_s


def test_a_negative_residual_adds_nothing_and_is_labelled():
    seq = _fixture()
    bare, null = _profiles(seq)
    spans = lr.Spans(spans_s={0: (1e-6,)}, setitems={0: (0,)})
    prof, rec = lr.residual(seq, bare, null, spans, spread_tol=0.10)
    assert prof.residual_s == {} and prof.host_s == bare.host_s
    assert rec["layers_negative"] == [0] and lr.NEGATIVE_LABEL in rec["labels"]


def test_a_spread_above_the_tolerance_keeps_the_residual_and_labels_the_cell():
    seq = _fixture()
    bare, null = _profiles(seq)
    base = sum(v for i, v in bare.host_s.items() if seq.ops[i].layer == 0)
    xs = tuple(base + d for d in (100e-6, 100e-6, 200e-6, 300e-6, 300e-6))
    prof, rec = lr.residual(seq, bare, null, lr.Spans(spans_s={0: xs}), spread_tol=0.10)
    assert rec["layers"][0]["spread"] > 0.10
    assert rec["layers_over_spread"] == [0] and lr.SPREAD_LABEL in rec["labels"]
    assert sum(prof.residual_s.values()) == pytest.approx(200e-6)      # kept, the median's
    _, rec = lr.residual(seq, bare, null, lr.Spans(spans_s={0: xs}), spread_tol=10.0)
    assert rec["labels"] == []


def test_a_layer_where_the_device_fell_behind_is_labelled():
    seq = _fixture()
    bare, null = _profiles(seq)
    base = sum(v for i, v in bare.host_s.items() if seq.ops[i].layer == 0)
    s = lr.Spans(spans_s={0: (base + 1e-4,)}, device_cum_s={0: 2e-3}, host_cum_s={0: 1e-3})
    _, rec = lr.residual(seq, bare, null, s, spread_tol=0.10)
    assert rec["layers_device_behind"] == [0] and lr.DEVICE_BEHIND_LABEL in rec["labels"]
    s.device_cum_s = {0: 1.04e-3}                                # inside the margin
    _, rec = lr.residual(seq, bare, null, s, spread_tol=0.10)
    assert rec["layers_device_behind"] == [] and rec["device_checked"]


def test_spans_are_taken_between_each_layers_hooks_with_a_write_timed_after_a_synchronise():
    import torch
    t = [0.0]
    log = []

    def clock():
        return t[0]

    def sync():
        log.append("sync")
        t[0] += 50.0                     # a wait for the device: out of the span

    class Layer(torch.nn.Module):
        def __init__(self, cost, write):
            super().__init__()
            self.cost, self.write = cost, write

        def forward(self, x):
            t[0] += self.cost            # the arm's own Python
            if self.write:
                x[x > 0] = 0.0           # a boolean-mask write
            return x

    layers = [Layer(3.0, False), Layer(5.0, True)]

    def prefill():
        x = torch.ones(4)
        for m in layers:
            x = m(x)
        t[0] += 100.0                    # outside the layers: no span

    from attnbench.components.sync_tape import TapeRecorder
    with TapeRecorder("cpu") as r:
        prefill()
    spans = lr.tape_spans(prefill, r.tape, layers, reps=3, device_type="cpu", sync=sync,
                          clock=clock)
    assert spans.spans_s == {0: (3.0, 3.0, 3.0), 1: (5.0, 5.0, 5.0)}
    assert spans.setitems == {0: (0, 0, 0), 1: (1, 1, 1)}
    assert spans.device_cum_s is None
    assert log.count("sync") == 4 * 2        # per run: before the write, and at its end


# --- the isolated trace read by bisection (null pass speed-up) -------------------------------

def test_the_flush_bisection_is_the_linear_rule():
    import random
    from attnbench.components.op_sequence import _EPS, _Ev
    rng = random.Random(6)
    t, flushes = 0.0, []
    for k in range(400):                        # flushes one after another
        t += rng.uniform(0.5, 5.0)
        d = rng.uniform(0.1, 3.0)
        flushes.append(_Ev(k, "user_annotation", "flush", 1, 1, t, t + d, {}))
        t += d
    f = hp._Flushes(list(reversed(flushes)))   # any order in
    inside = lambda e, c: c.ts - _EPS <= e.ts and e.end <= c.end + _EPS
    for _ in range(3000):
        a = rng.uniform(0, t)
        e = _Ev(0, "cuda_runtime", "cudaLaunchKernel", 1, 1, a, a + rng.uniform(0, 1.0), {})
        assert f.holds(e) == any(inside(e, x) for x in flushes)
        b = a + rng.uniform(0, 60)
        linear = sum(x.end - x.ts for x in flushes if a - _EPS <= x.ts and x.end <= b + _EPS)
        assert f.within(a, b) == pytest.approx(linear, abs=1e-9)
    # an edge exactly at a flush's bounds
    x = flushes[10]
    assert f.within(x.ts, x.end) == pytest.approx(x.end - x.ts)
    assert f.holds(_Ev(0, "c", "l", 1, 1, x.ts, x.end, {}))


def test_overlapping_flushes_are_refused():
    from attnbench.components.op_sequence import _Ev
    with pytest.raises(hp.ProfileMismatch, match="overlap"):
        hp._Flushes([_Ev(0, "u", "f", 1, 1, 0.0, 5.0, {}), _Ev(1, "u", "f", 1, 1, 4.0, 6.0, {})])


def test_a_cached_output_shape_is_the_meta_runs_and_the_meta_run_is_made_once(monkeypatch):
    import torch
    aten = torch.ops.aten
    a, b = torch.randn(4, 3), torch.randn(4, 3)
    calls = [(aten.add.Tensor, (a, b), {}), (aten.add_.Tensor, (a, b), {}),
             (aten.sort.default, (a,), {}), (aten.t.default, (a,), {}),
             (aten.arange.start, (0, 7), {"device": "cpu", "dtype": torch.int64}),
             (aten.mm.default, (a, b.t().contiguous()), {})]
    runs = []
    meta = hp._meta_like
    monkeypatch.setattr(hp, "_meta_like", lambda t: runs.append(1) or meta(t))
    monkeypatch.setattr(hp, "_OUTPUT_SPECS", {})

    def shape(out, inputs):
        return [("input", next(k for k, t in enumerate(inputs) if t is o))
                if any(t is o for t in inputs) else (tuple(o.shape), tuple(o.stride()), o.dtype)
                for o in torch.utils._pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
    first = [shape(hp._null_outputs(f, x, k, "cpu"), x) for f, x, k in calls]
    n = len(runs)
    again = [shape(hp._null_outputs(f, x, k, "cpu"), x) for f, x, k in calls]
    assert again == first and len(runs) == n                    # no meta run the second time
    assert first[1] == [("input", 0)]                           # in place: the input itself
    assert first[2] == [((4, 3), (3, 1), torch.float32), ((4, 3), (3, 1), torch.int64)]
    # a different shape is another signature
    hp._null_outputs(aten.add.Tensor, (torch.randn(2, 3), torch.randn(2, 3)), {}, "cpu")
    assert len(runs) > n
