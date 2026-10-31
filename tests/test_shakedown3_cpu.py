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


def test_the_density_probe_runs_compiled_mean_pool_on_the_real_wrapper(harness, fresh_dynamo):
    """Shakedown 3, t228 `density`: `CompiledBackend.reset_density` called
    the inner `BlockSparseAttention.reset_density`, which does not exist."""
    names, dens = ["dense", "meanpool"], [None, 0.25]
    be = compiled.CompiledBackend(harness._backend(sa.ARMS["meanpool"]), "default",
                                  mode=None, dynamo_backend="eager")
    wrapped, ids, _, arms, score_kwargs = harness.timing_arms(     # 8 blocks: a real choice
        names, 1024, densities=dens, backends=[None, be], compile_settings=[None, "default"])
    a = arms[1]
    kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)

    def run(backend=None):
        wrapped.run_measured(ids, backend if backend is not None else a.backend, cfg=a.cfg,
                             logits_to_keep=1, **kw)
    run()
    row = gt.kernel_densities(a, run)
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
    with pytest.raises(hp.ProfileMismatch) as e:
        hp.profile(a.run_prefill, rec.tape, rec.sequence, device=dev, reps=1, layers=a.layers,
                   isolated=iso, isolated_ops=ops)
    # Past the embedding, the next call the pass does not meet is refused,
    # as the ruling has it ("refuse all other unhandled ops"): `arange`, a
    # factory call with no tensor input, so the pass never sees it touch the
    # device. The L4's recording has it at the same place (Shakedown 3,
    # position 1). Open: a ruling on the pass's unseen calls.
    assert "aten::embedding" not in str(e.value) and cb.EMBEDDING_KERNEL_OP not in str(e.value)
    assert "under aten::arange in layer None" in str(e.value)


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


def test_t212_fails_mean_pool_on_an_empty_source_density_record(boundary, tmp_path):
    """Shakedown 3, t212: "meanpool: the replay's executed density differs
    from its source's", logits bitwise equal. Reproduced through the step
    itself: the masks agree in every layer; the source's density record
    (`BlockSparseAttention.last_layer_density`) does not exist, so the check
    compares no layers with the precomputed arm's. A finding: the gate is
    unchanged until it is ruled on."""
    from types import SimpleNamespace
    import t228_steps
    args = SimpleNamespace(out=str(tmp_path / "t212"), dry_run=False, cpu_boundary=True,
                           length=1024, replay_arms="meanpool", reps=2, warmup=2, max_rerun=0,
                           floor_ms=1e9, sampler_hz=10.0)
    with pytest.raises(t228_steps.StepFailed, match="meanpool: the replay's executed density "
                                                    "differs from its source's") as e:
        t228_steps.step_t212(args)
    f = e.value.findings["arms"]["meanpool"]
    assert f["bitwise_equal_logits"] is True
    assert f["density_record_layers"] == {"source": 0, "precomputed": 2}
    d = f["mask_diagnosis"]
    assert d["source_keeps_density_record"] is False
    for k in ("live_live", "live_bank", "precomputed_vs_source"):
        assert d[k]["identical_layers"] == d[k]["layers"] == 2


# --- the boundary itself ------------------------------------------------------------------

def test_the_boundary_puts_back_everything_it_replaced():
    before = (torch.cuda.synchronize, torch.cuda.get_device_properties,
              torch.profiler.profile.export_chrome_trace)
    with cb.installed(xattn=False):
        assert cb.active() and "block_sparse_attn" in sys.modules
        assert torch.cuda.get_device_properties(0).name == cb.STUB_DEVICE_NAME
        assert torch.ones(2, device="cuda").device.type == "cpu"
        with pytest.raises(RuntimeError, match="already installed"):
            with cb.installed(xattn=False):
                pass
    assert not cb.active() and "block_sparse_attn" not in sys.modules
    assert (torch.cuda.synchronize, torch.cuda.get_device_properties,
            torch.profiler.profile.export_chrome_trace) == before
    assert "triton" not in sys.modules


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
