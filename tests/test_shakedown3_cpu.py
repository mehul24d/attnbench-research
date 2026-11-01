"""Shakedown 3's failures, reproduced on the real classes behind the stubbed
GPU boundary (`attnbench.cpu_boundary`), then fixed (31 Oct 2026).

Each defect had passed its tests because those tests used stand-ins: a
compiled wrapper over a fake arm that had `reset_density`, a step driven
with a fake harness, a probe fed hand-written sequences with no work
outside the layers. Here the arm, backend, model wrapper, recorder and null
pass are the study's own, on the two-layer random Qwen2. Only the
x-attention clone is left out (`installed(xattn=False)`): the XAttention arms
run in the CPU dry run itself (`t31_session.py --cpu-boundary`).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from attnbench import cpu_boundary as cb
from attnbench import ground_truth as gt
from attnbench.backends import compiled
from attnbench.components import host_profile as hp
from attnbench.components import recorder, replay
from attnbench.components import shakedown_arms as sa
from attnbench.components import structural_probe as P
from attnbench.components.op_sequence import (DIFFERENT, SAME, CodePath, OpSequence, SeqKind,
                                              SeqOp)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import t31_steps as T  # noqa: E402

pytestmark = pytest.mark.skipif(torch.cuda.is_available(),
                                reason="the CPU boundary is for a machine without CUDA")
L = 256


@pytest.fixture
def boundary():
    with cb.installed(xattn=False):
        yield


@pytest.fixture
def harness(boundary, tmp_path):
    h = sa.Harness(device="cpu", dry_run=True, score_cache_dir=str(tmp_path / "sc"))
    yield h
    h.unwrap()


# --- rehearsal step 5: dense takes no density setting ---------------------------------------

def test_every_cell_of_the_working_grid_prepares_on_the_real_harness(harness):
    """Shakedown 3, rehearsal step 5: `h.ready(arm, L, density=d)` with the
    dense cell's 1.0 raised "dense: its density is not a setting"."""
    cfg = dict(T.CPU_BOUNDARY_CONFIG, lengths=[L], batches={str(L): [1, 2]},
               densities=T.working_values()[0]["densities"],
               arms=["dense", "oracle", "meanpool"])
    seen = set()
    for arm, d, length, b in T.cells(cfg):
        a = T.ready_cell(harness, arm, d, length, b)
        assert a.extra["cfg"].batch == b == a.extra["input_ids"].shape[0]
        if arm == "dense":
            assert d == 1.0 and a.extra["cfg"].sparsity is None
        else:
            assert a.extra["cfg"].sparsity == pytest.approx(1 - d)
        seen.add(arm)
    assert seen == {"dense", "oracle", "meanpool"}


@pytest.mark.parametrize("arm, density", [("oracle", 0.25), ("meanpool", 0.25)])
def test_the_block_sparse_estimators_run_at_batch_one_only(harness, arm, density):
    """Found by the CPU-boundary rehearsal: the oracle's scoring pass and the
    mean-pool estimator refuse a batch above one, so the grid's batch-4
    cells leave them out, as they leave out XAttention."""
    assert arm in T.BATCH1_ONLY
    with pytest.raises(ValueError, match="batch"):
        harness.ready(arm, 1024, density=density, batch=2)


def test_a_density_on_an_arm_that_has_none_is_still_refused(harness):
    with pytest.raises(ValueError, match="not a setting"):
        harness.ready("dense", L, density=0.5)


# --- the density probe: the compiled wrapper over mean-pool -------------------------------

@pytest.fixture
def fresh_dynamo():
    compiled.reset_compiled_state()
    yield
    compiled.reset_compiled_state()


def test_the_compiled_wrapper_has_a_density_record_only_if_its_arm_has(boundary, fresh_dynamo):
    from attnbench.backends.block_sparse import BlockSparseAttention
    c = compiled.CompiledBackend(BlockSparseAttention(), "default", mode=None,
                                 dynamo_backend="eager")
    assert not hasattr(c, "reset_density") and not hasattr(c, "last_layer_density")


COMPILED_MEANPOOL = """
import json, sys, tempfile
from attnbench import cpu_boundary as cb, ground_truth as gt
from attnbench.backends import compiled
from attnbench.components import shakedown_arms as sa
with cb.installed(xattn=False):
    h = sa.Harness(device="cpu", dry_run=True, score_cache_dir=tempfile.mkdtemp())
    be = compiled.CompiledBackend(h._backend(sa.ARMS["meanpool"]), "default",
                                  **cb.compile_kwargs())
    wrapped, ids, _, arms, score_kwargs = h.timing_arms(      # 8 blocks: a real choice
        ["dense", "meanpool"], 1024, densities=[None, 0.25], backends=[None, be],
        compile_settings=[None, "default"])
    a = arms[1]
    kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)

    def run(backend=None):
        wrapped.run_measured(ids, backend if backend is not None else a.backend, cfg=a.cfg,
                             logits_to_keep=1, **kw)
    run()
    print(json.dumps(gt.kernel_densities(a, run)))
"""


def test_the_density_probe_runs_compiled_mean_pool_on_the_real_wrapper():
    """Shakedown 3, t228 `density`: `CompiledBackend.reset_density` called
    the inner `BlockSparseAttention.reset_density`, which does not exist.
    In a process of its own, as each compile setting runs (`density-one`):
    in the full suite an earlier `torch._dynamo.explain` left state behind
    that broke this test's guards, which a session's fresh process never
    meets."""
    import json
    import subprocess
    r = subprocess.run([sys.executable, "-c", COMPILED_MEANPOOL], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    row = json.loads(r.stdout.strip().splitlines()[-1])
    assert 0.0 < row["mask_density"] < 1.0
    assert row["executed_density"] == pytest.approx(row["mask_density"])
    assert row["density_vs_eager"] == pytest.approx(0.0, abs=1e-12)
    assert row["density_vs_eager_flagged"] is False


# --- the structural extension: regions --------------------------------------------------

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch",
              libraries=(("torch", "2.9.1"),))
PER_1K = P.ChunkingRule("one chunk per 1024", lambda L: L // 1024)


def _k(layer, op="aten::mm"):
    return SeqOp(SeqKind.KERNEL, layer=layer, aten_op=op, kernels=("k",))


def with_regions(L, chunks, *, pre=("aten::gather", "aten::arange", "aten::cos"),
                 post=("aten::mean", "aten::mm"), between=()):
    """Work before the layers (embedding, rotary), in them (a unit per chunk)
    and after them (final norm, head), as Shakedown 3's XAttention recording
    had: 11 operations outside the layers at 3 and at 4 chunks."""
    ops = [_k(None, o) for o in pre]
    for layer in (0, 1):
        ops.append(_k(layer, "aten::mm"))
        ops += [_k(layer, "aten::bmm") for _ in range(chunks)]
        ops.append(_k(layer, "bsa"))
        if layer == 0:
            ops += [_k(None, o) for o in between]
    ops += [_k(None, o) for o in post]
    return OpSequence(code_path=CP, context_length=L, batch=1, ops=tuple(ops), streams=(7,))


def test_work_outside_the_layers_is_not_taken_for_a_chunked_unit():
    """Shakedown 3's probe: "layer None: no prefix + unit x chunks + suffix
    fits both lengths (11 and 11 operations, 3 and 4 chunks)"."""
    seqs = {L_: with_regions(L_, PER_1K.chunks(L_)) for L_ in (3072, 4096)}
    ext = P.extend(seqs, PER_1K, 8192)
    assert ext.extended, ext.reason
    assert P.check_against_full(ext, with_regions(8192, 8)).same


def test_the_regions_are_named_and_only_the_layers_take_the_chunk_rule():
    seq = with_regions(3072, 3)
    regions = P.regions(seq.ops)
    assert [r.kind for r in regions] == [P.PRE_LAYERS, P.IN_LAYERS, P.IN_LAYERS, P.POST_LAYERS]
    assert [r.chunked for r in regions] == [False, True, True, False]


def test_work_outside_the_layers_that_changes_with_length_is_different():
    a = with_regions(3072, 3)
    b = with_regions(4096, 4, pre=("aten::gather", "aten::arange", "aten::sin"))
    ext = P.extend({3072: a, 4096: b}, PER_1K, 8192)
    assert ext.verdict == DIFFERENT and "before the layers" in ext.reason


def test_work_between_layers_is_refused_as_its_own_region():
    seqs = {L_: with_regions(L_, PER_1K.chunks(L_), between=("aten::cat",))
            for L_ in (3072, 4096)}
    ext = P.extend(seqs, PER_1K, 8192)
    assert ext.verdict == DIFFERENT and "between layers" in ext.reason


# --- the extension test: base + chunks x body (ruling, 1 Nov 2026) ----------------------

def test_counts_that_follow_base_plus_chunks_times_body_fit_at_every_probe_length():
    seqs = {L: with_regions(L, PER_1K.chunks(L)) for L in (2048, 3072, 4096)}
    fit = P.count_fit(seqs, PER_1K)
    assert fit["fits"] and fit["fit_from"] == [2048, 3072]
    assert [(r["region"], r["base"], r["body"]) for r in fit["regions"]] == \
        [("layer 0", 2.0, 1.0), ("layer 1", 2.0, 1.0)]
    assert fit["chunks"] == {2048: 2, 3072: 3, 4096: 4}


def test_a_third_point_off_the_line_fails_the_fit_and_says_where():
    # XAttention behind the boundary: 240 and 301 operations at 3 and 4 chunks
    # (base 57, body 61), and 294 at 32768, four whole chunks with no padding.
    seqs = {L: with_regions(L, PER_1K.chunks(L)) for L in (2048, 3072)}
    padless = with_regions(4096, PER_1K.chunks(4096))
    seqs[4096] = replace(padless, ops=tuple(o for i, o in enumerate(padless.ops)
                                            if not (o.layer == 0 and o.aten_op == "aten::mm")))
    fit = P.count_fit(seqs, PER_1K)
    layer0 = fit["regions"][0]
    assert not fit["fits"] and not layer0["fits"] and fit["regions"][1]["fits"]
    assert layer0["reason"] == "4096: 6 predicted, 5 recorded"


def test_no_two_distinct_chunk_counts_cannot_fit():
    one = P.ChunkingRule("one", lambda L: 2)
    fit = P.count_fit({L: with_regions(L, 2) for L in (2048, 4096)}, one)
    assert not fit["fits"] and fit["fit_from"] is None
    assert fit["regions"][0]["reason"] == "no two probe lengths with distinct chunk counts"


# --- t22: the embedding's gather outside the layers --------------------------------------

def _record(harness, arm, tmp_path, density=None):
    a = harness.ready(arm, L, density=density)
    rec = recorder.record_arm_prefill(a.run_prefill, tmp_path / arm, code_path=a.code_path,
                                      context_length=a.length, batch=1,
                                      prompt_id=a.prompt_id, layers=a.layers)
    return a, rec


def test_the_null_pass_meets_the_embedding_lookup_outside_the_layers(harness, tmp_path):
    """Shakedown 3, t22: "the recording issues 1 operation(s) under
    aten::gather in layer None, and the pass made no launching call there".
    The dispatcher shows the pass `aten::embedding`; the recording names
    the kernel's operator (`gather` on the L4, `index_select` on a CPU)."""
    a, rec = _record(harness, "dense", tmp_path)
    first = rec.sequence.ops[0]
    assert (first.layer, first.origin, first.aten_op) == \
        (None, "aten::embedding", cb.EMBEDDING_KERNEL_OP)
    dev = hp.NullDevice(launch=lambda: None, device_type="cpu")
    ops = hp.ISOLATED_OUTSIDE_LAYERS | {cb.EMBEDDING_KERNEL_OP}
    pos = hp.isolated_positions(rec.sequence, ops)
    assert pos == (0,)
    iso = hp.isolated_host_s(replay.isolated_replay(
        a.run_prefill, rec.tape, tmp_path / "iso.json", device="cpu", substitute=False),
        rec.sequence, pos)
    assert set(iso) == {0} and iso[0] > 0
    # Past the embedding the pass met `arange` (no tensor input) by its
    # device argument and the composites (`_to_copy`, `mean`, `clone`) at
    # their recorded chains (ruling, 1 Nov 2026), so every position is priced.
    p = hp.profile(a.run_prefill, rec.tape, rec.sequence, device=dev, reps=1, layers=a.layers,
                   isolated=iso, isolated_ops=ops)
    issued = [i for i, op in enumerate(rec.sequence.ops)
              if op.is_device or op.kind is SeqKind.COPY]
    assert set(issued) <= set(p.host_s)
    first_arange = next(i for i, op in enumerate(rec.sequence.ops) if op.aten_op == "aten::arange")
    assert p.source(first_arange) == hp.DEVICE_ARGUMENT
    assert p.met_at[first_arange] == "aten::arange"
    met = {p.met_at[i] for i in p.cost_source if p.source(i) == hp.CHAIN}
    assert {"aten::_to_copy", "aten::mean"} <= met
    assert "aten::to" not in met and "aten::linear" not in met      # never shown to the pass
    src = p.sources()
    assert src[hp.ISOLATED_REPLAY] == 1 and src[hp.CHAIN] > 0 and src[hp.DEVICE_ARGUMENT] > 0


def test_the_chain_check_holds_the_composites_against_the_isolated_replay(harness, tmp_path):
    """The ruling's validation, on the real arm: each composite's chain-priced
    cost against its host time in the isolated replay of the original
    operators. Behind the boundary a CPU kernel runs inside its call, so the
    direct time holds the compute and the check is not expected to pass;
    what is checked here is that it reads every composite call."""
    a, rec = _record(harness, "dense", tmp_path)
    ops = hp.ISOLATED_OUTSIDE_LAYERS | {cb.EMBEDDING_KERNEL_OP}
    trace = replay.isolated_replay(a.run_prefill, rec.tape, tmp_path / "iso.json", device="cpu",
                                   substitute=False)
    iso = hp.isolated_host_s(trace, rec.sequence, hp.isolated_positions(rec.sequence, ops))
    p = hp.profile(a.run_prefill, rec.tape, rec.sequence,
                   device=hp.NullDevice(launch=lambda: None, device_type="cpu"), reps=1,
                   layers=a.layers, isolated=iso, isolated_ops=ops)
    c = hp.chain_check(trace, rec.sequence, p, aggregate_tol=0.10)
    priced = [i for i in p.cost_source if p.source(i) in (hp.CHAIN, hp.DEVICE_ARGUMENT)]
    calls = len({(p.met_at[i], i) for i in priced})
    assert c["checked_calls"] + c["unchecked_calls"] <= calls and c["checked_calls"] > 0
    assert {"aten::mean", "aten::add", "aten::ones"} <= set(c["operators"])
    assert c["operators"]["aten::ones"]["sources"] == [hp.DEVICE_ARGUMENT]
    assert c["operators"]["aten::mean"]["sources"] == [hp.CHAIN]
    for row in c["operators"].values():
        assert 0 < row["chain_s"] < row["direct_s"]      # the CPU's compute is in the call
    assert not c["passed"] and c["share"] > 0.10
    assert len(c["calls"]) == c["checked_calls"]


# --- the replay and T2.12: diagnosis by arm and layer (findings, no gate) -----------------

def test_the_mean_pool_masks_agree_on_every_eager_path_of_the_replay(harness, tmp_path):
    """Shakedown 3's replay read all-False live masks in every layer. On the
    eager paths (two live runs, the bank, the tape without a graph) the
    real arm's masks agree bitwise, so the L4 failure is the graph's."""
    from attnbench.components import mask_diagnosis as md
    a, rec = _record(harness, "meanpool", tmp_path, density=0.25)
    diag = md.replay_diagnosis(a.run_prefill, rec, device="cpu")
    assert set(diag) == {"live_live", "live_bank", "tape_eager_bank"}
    for path in diag.values():
        s = path["summary"]
        assert s["layers"] == 2 and s["identical_layers"] == 2 and s["a_all_false_layers"] == 0


# --- the replay reads its live masks after a replay (ruling, 1 Nov 2026) --------------------

def test_a_graph_s_live_masks_are_unwritten_after_the_capture_alone(harness, tmp_path):
    """Shakedown 3's replay compared live masks no kernel had written: a
    graph's capture runs none. Behind the boundary the graph stand-in keeps
    that property, and the buffers say so instead of reading as all-False."""
    a, rec = _record(harness, "meanpool", tmp_path, density=0.25)
    bank = rec.masks.tensors("cpu")
    check, buffers = replay.check_masks(a.run_prefill, rec.tape, rec.tape.stage("cpu"), bank,
                                        cb.GraphLikeBackend(), replays=0)
    assert len(bank) == 2 and check.unwritten == (0, 1) and check.agreement is None
    assert replay.unwritten(buffers) == [0, 1]


def test_the_mask_check_replays_once_and_then_agrees_with_the_bank(harness, tmp_path):
    a, rec = _record(harness, "meanpool", tmp_path, density=0.25)
    be = cb.GraphLikeBackend()
    cap = replay.capture(a.run_prefill, rec.tape, device="cpu", masks=rec.masks, backend=be)
    assert cap.mask_check.replays == 1 and cap.mask_check.unwritten == ()
    assert be.replays == 1                      # the check graph's; the timed graph's are later
    agreement = replay.mask_agreement(cap)
    assert agreement["identical"] and agreement["per_call_agreement"] == [1.0, 1.0]


def test_a_layer_no_replay_wrote_is_refused(harness, tmp_path):
    class NoWrite(cb.GraphLikeBackend):     # a replay that runs no kernel
        def replay(self, body):
            self.replays += 1
            self.capture(body)
    a, rec = _record(harness, "meanpool", tmp_path, density=0.25)
    cap = replay.capture(a.run_prefill, rec.tape, device="cpu", masks=rec.masks,
                         backend=NoWrite())
    with pytest.raises(replay.ReplayError, match=r"kernel calls \[0, 1\] were never written"):
        replay.mask_agreement(cap)


def test_the_timed_graph_holds_the_arm_s_kernels_and_none_of_the_check_s(harness, tmp_path):
    """The check's copies would be graph nodes with no position (the CPU dry
    run's first device step, 1 Nov 2026), so they live in a graph of their own."""
    a, rec = _record(harness, "oracle", tmp_path, density=0.25)
    cap = replay.capture(a.run_prefill, rec.tape, device="cpu", masks=rec.masks,
                         backend=cb.GraphLikeBackend())
    m = replay.match(rec.sequence, replay.replay_nodes(
        replay.profile_replay(cap, tmp_path / "replay.json")))
    assert m.extra_nodes == () and replay.mask_agreement(cap)["identical"]


def test_t212_takes_mean_pool_s_density_from_the_kernel_input_recorder(boundary, tmp_path):
    """Shakedown 3, t212: "meanpool: the replay's executed density differs
    from its source's", logits bitwise equal: the source's density record
    (`BlockSparseAttention.last_layer_density`) does not exist, so the check
    compared no layers. Ruling of 1 Nov 2026: both sides' executed density
    is the kernel-input recorder's. Through the step itself, it now passes."""
    from types import SimpleNamespace
    import t228_steps
    args = SimpleNamespace(out=str(tmp_path / "t212"), dry_run=False, cpu_boundary=True,
                           length=1024, replay_arms="meanpool", reps=2, warmup=2, max_rerun=0,
                           floor_ms=1e9, sampler_hz=10.0)
    findings = t228_steps.step_t212(args)
    f = findings["arms"]["meanpool"]
    assert f["bitwise_equal_logits"] is True and f["executed_density_equal"] is True
    assert f["density_record_layers"] == {"source": 2, "precomputed": 2}
    assert f["density_source"] == "kernel-input recorder"
    d = f["mask_diagnosis"]
    assert d["source_keeps_density_record"] is False       # why the old record was empty
    for k in ("live_live", "live_bank", "precomputed_vs_source"):
        assert d[k]["identical_layers"] == d[k]["layers"] == 2


def test_t212_refuses_a_density_comparison_over_no_layers(boundary, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import t228_steps
    from attnbench import densities as D
    monkeypatch.setattr(D.KernelInputRecorder, "per_call_density", lambda self: [])
    args = SimpleNamespace(out=str(tmp_path / "t212"), dry_run=False, cpu_boundary=True,
                           length=1024, replay_arms="meanpool", reps=2, warmup=2, max_rerun=0,
                           floor_ms=1e9, sampler_hz=10.0)
    with pytest.raises(t228_steps.StepFailed, match="source and precomputed has no layers"):
        t228_steps.step_t212(args)


# --- the boundary itself ------------------------------------------------------------------

def test_the_boundary_puts_back_everything_it_replaced():
    from attnbench.components import sync_tape
    before = (torch.cuda.synchronize, torch.cuda.get_device_properties,
              torch.profiler.profile.export_chrome_trace, sync_tape._as_device)
    with cb.installed(xattn=False):
        assert cb.active() and "block_sparse_attn" in sys.modules
        assert torch.cuda.get_device_properties(0).name == cb.STUB_DEVICE_NAME
        assert torch.ones(2, device="cuda").device.type == "cpu"
        with pytest.raises(RuntimeError, match="already installed"):
            with cb.installed(xattn=False):
                pass
    assert not cb.active() and "block_sparse_attn" not in sys.modules
    assert (torch.cuda.synchronize, torch.cuda.get_device_properties,
            torch.profiler.profile.export_chrome_trace, sync_tape._as_device) == before
    assert "triton" not in sys.modules


def test_the_tape_reads_cuda_as_the_device_behind_the_boundary(boundary):
    """XAttention's estimator calls `.to("cuda")` on a tensor already on the
    device. On the L4 the tape lets that call through; behind the boundary
    the tape saw it before `CudaToCpu` and took it for a copy to the host, so
    the null pass met the tape's own clone (1 Nov 2026)."""
    from attnbench.components import sync_tape as st
    x = torch.ones(2, device="cuda")
    for target in ("cuda", "cuda:0", torch.device("cuda"), 0):
        assert st.classify(torch.Tensor.to, (x, target), {}, "cpu") is None
    assert st.classify(torch.tensor, ([1.0],), {"device": "cuda"}, "cpu") == "to_device"


def test_an_operator_with_no_kernel_of_its_own_is_no_launch_in_the_trace_shim():
    """`F.dropout(p=0, training=False)` returns its input: on a CUDA build
    `aten::dropout` is CompositeImplicitAutograd and launches nothing."""
    assert cb.composite_implicit("aten::dropout") and cb.composite_implicit("aten::matmul")
    assert not cb.composite_implicit("aten::mm") and not cb.composite_implicit("aten::_to_copy")
    assert not cb.composite_implicit(cb.KERNEL_OP)


def test_the_stub_kernel_with_every_block_kept_is_causal_attention():
    torch.manual_seed(0)
    b, s, h, d = 2, 300, 2, 32
    q, k, v = (torch.randn(b * s, h, d) for _ in range(3))
    nb = -(-s // cb.BLOCK)
    mask = torch.ones(b, h, nb, nb, dtype=torch.bool)
    out = cb.block_sparse_attn_func(q, k, v, None, None, None, None, mask, s, s, 0.0,
                                    is_causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.reshape(b, s, h, d).transpose(1, 2) for t in (q, k, v)), is_causal=True)
    assert torch.allclose(out, ref.transpose(1, 2).reshape(b * s, h, d), atol=1e-5)
    mask[:, :, 2, 1] = False                       # a dropped block changes only its rows
    out2 = cb.block_sparse_attn_func(q, k, v, None, None, None, None, mask, s, s, 0.0,
                                     is_causal=True)
    rows = (out2 - out).abs().reshape(b, s, h, d).amax(dim=(0, 2, 3)) > 0
    assert rows[256:].all() and not rows[:256].any()


def test_the_trace_shim_launches_once_per_kernel_operator_and_never_for_a_view():
    def op(name, ts, dur):
        return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": 1,
                "ts": ts, "dur": dur, "args": {}}
    trace = {"traceEvents": [
        op("aten::embedding", 0, 10), op("aten::index_select", 1, 8), op("aten::empty", 2, 1),
        op("aten::view", 20, 1), op(cb.KERNEL_OP, 30, 10), op("aten::mm", 31, 5),
        op("aten::_local_scalar_dense", 50, 4)]}
    ev = cb.synthesize_device(trace)["traceEvents"]
    launches = [e for e in ev if e["name"] == "cudaLaunchKernel"]
    kernels = sorted(e["name"] for e in ev if e.get("cat") == "kernel")
    assert kernels == sorted(["cpu_stub::" + cb.KERNEL_OP, "cpu_stub::aten::index_select"])
    sel = next(e for e in launches if 1 <= e["ts"] <= 9)
    assert not 2 <= sel["ts"] <= 3                 # in index_select's own time, not empty's
    assert [e["name"] for e in ev if e.get("cat") == "cuda_runtime" and e["ts"] >= 50] == \
        ["cudaMemcpyAsync", "cudaStreamSynchronize"]


def test_the_cpu_boundary_config_writes_its_prediction_under_a_rehearsal_prefix(tmp_path):
    """Found by the CPU dry run's step 7 (1 Nov 2026)."""
    assert T.prediction_dir(T.cpu_boundary_config(), tmp_path).startswith(
        "predictions/P1-cpu-boundary-rehearsal")
