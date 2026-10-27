"""T2.12: the precomputed-mask ablation arm (plan Section 5.5).

CPU, a toy Llama, and a stand-in for Block-Sparse-Attention's kernel built on
`kernels.reference_prefill` (the plain-torch statement of what the kernel
computes). What these tests establish: the masks an estimator produced are the
masks the kernel receives, unchanged and already resident; a replay reproduces
its source arm's output; nothing in the arm computes a score or builds a mask;
and a mask set cannot be replayed on a prompt it was not computed from. That
the real kernel does the same on a GPU is the L4 dry run (T2.27).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from _gt_fakes import FakeClock, ScriptedReader  # noqa: E402
from attnbench import clock_sampling as cs  # noqa: E402
from attnbench import ground_truth as gt  # noqa: E402
from attnbench import kernels, masks  # noqa: E402
from attnbench import precomputed_masks as pm  # noqa: E402
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.backends.base import UnsupportedConfig  # noqa: E402
from attnbench.backends.block_masked import BlockMaskedSDPA  # noqa: E402
from attnbench.backends.impls import NaiveAttention  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402
from attnbench.densities import KernelInputRecorder, block_density  # noqa: E402

SEQ, LAYERS, HQ, HKV, D = 512, 2, 4, 2, 32
NB = SEQ // 128


def _model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=HQ * D, intermediate_size=64,
                      num_hidden_layers=LAYERS, num_attention_heads=HQ,
                      num_key_value_heads=HKV, head_dim=D,
                      max_position_embeddings=1024, attn_implementation="eager")
    return SwappableAttentionModel(LlamaForCausalLM(cfg).eval(), _sparse_cfg(),
                                   model_id="toy", finest_block_size=128)


def _sparse_cfg(**kw):
    base = dict(seq_len=SEQ, batch=1, n_heads_q=HQ, n_heads_kv=HKV, head_dim=D,
                dtype="float32", mask="block_sparse", block_size=128, sparsity=0.5,
                mask_source="importance")
    base.update(kw)
    return AttnConfig(**base)


def _causal_cfg():
    return _sparse_cfg(mask="causal", sparsity=None, block_size=None, mask_source=None)


def _ids(seed=3):
    return torch.randint(0, 64, (1, SEQ), generator=torch.Generator().manual_seed(seed))


def fake_kernel(q_, k_, v_, cu_q, cu_k, head_mask_type, streaming, base_blockmask, max_q, max_k,
                p_dropout=0.0, deterministic=False, is_causal=False, exact_streaming=False):
    """The BSA call's signature, computed with `kernels.reference_prefill`."""
    s, h, d = q_.shape
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q_, k_, v_))
    out = kernels.reference_prefill(q, k, v, base_blockmask)
    return out[0].transpose(0, 1).reshape(s, h, d)


def _record_meanpool(wrapped, ids):
    # BlockMaskedSDPA, not NaiveAttention, as the source: this was written
    # when NaiveAttention cached its dense mask under `cfg.key()` alone, so on
    # one instance a second layer's different mask got the first layer's
    # (max output difference 0.51 against a fresh instance). The cache is
    # keyed on the mask's contents since 27 Oct 2026
    # (tests/test_naive_mask_cache.py); the source is unchanged.
    rec = pm.MaskRecorder(BlockMaskedSDPA(), estimator="meanpool", code_path="inline_meanpool",
                          mask_selector="head_uniform")
    source = wrapped.run_measured(ids, rec, cfg=_sparse_cfg(), logits_to_keep=1,
                                  inline_estimator="minference_meanpool")
    return rec, source.logits


def _arm(mask_set, ids, **kw):
    return pm.PrecomputedMaskBackend(mask_set, ids, device="cpu", kernel=kw.pop("kernel", fake_kernel))


# --- recording ------------------------------------------------------------------------

def test_the_recorded_masks_are_the_ones_the_source_arm_used_and_are_not_trivial():
    ids = _ids()
    rec, _ = _record_meanpool(_model(), ids)
    ms = rec.mask_set(ids, n_heads_q=HQ)
    assert ms.n_layers == LAYERS and ms.seq_len == SEQ
    for m in ms.masks.values():
        assert m.dtype == torch.bool and tuple(m.shape) == (1, HQ, NB, NB)
        assert m.is_contiguous()
        assert torch.equal(m[:, 0], m[:, HQ - 1])
    assert 0.0 < ms.mask_density() < 1.0                          # it really is sparse


def test_a_per_head_mask_is_recorded_as_given():
    scores = torch.rand(HQ, NB, NB, generator=torch.Generator().manual_seed(1))
    ph = masks.importance_block_mask_per_head(SEQ, 128, 0.5, scores, causal=True,
                                              identity_seed="x")
    rec = pm.MaskRecorder(NaiveAttention(), estimator="per_head", code_path="p",
                          mask_selector="per_head")
    rec.inner = types.SimpleNamespace(forward=lambda *a, **k: torch.zeros(1))
    rec.set_layer(0)
    q = torch.zeros(1, HQ, SEQ, D)
    rec.forward(q, q, q, _causal_cfg(), mask=ph)
    assert torch.equal(rec.recorded[0], ph.active.unsqueeze(0))


def test_an_estimators_own_mask_is_captured_at_its_return_and_sliced_for_the_kernel():
    """XAttention's shape: `forward` calls `self.estimate`, whose mask has
    padded block dimensions, and slices it before the kernel."""
    padded = NB + 2
    gen = torch.Generator().manual_seed(2)
    est_out = torch.rand(1, HQ, padded, padded, generator=gen) > 0.5
    est_out[:, :, torch.arange(padded), torch.arange(padded)] = True    # the diagonal and
    est_out[:, :, :, 0] = True                                           # the sink are forced

    class XAShaped:
        def estimate(self, q, k_rep):
            return est_out

        def forward(self, q, k, v, cfg, mask=None):
            m = self.estimate(q, k)[:, :, :NB, :NB].contiguous()
            return kernels.reference_prefill(q, k, v, m)

    inner = XAShaped()
    rec = pm.MaskRecorder(inner, estimator="xattention", code_path="xa", mask_selector="xattn_native")
    rec.set_layer(0)
    q = torch.randn(1, HQ, SEQ, D)
    out = rec.forward(q, q, q, _causal_cfg())
    assert torch.equal(rec.recorded[0], est_out[:, :, :NB, :NB])
    assert torch.equal(out, kernels.reference_prefill(q, q, q, est_out[:, :, :NB, :NB].contiguous()))
    assert "estimate" not in vars(inner)               # the capture hook was removed


def test_a_mask_at_another_block_size_than_the_kernels_is_refused():
    n64 = SEQ // 64
    other = masks.importance_block_mask_device(SEQ, 64, 0.5, torch.rand(n64, n64), causal=True,
                                               identity_seed="b64")
    rec = pm.MaskRecorder(types.SimpleNamespace(forward=lambda *a, **k: torch.zeros(1)),
                          estimator="e", code_path="c", mask_selector="head_uniform")
    rec.set_layer(0)
    q = torch.zeros(1, HQ, SEQ, D)
    with pytest.raises(pm.MaskSetError, match="kernel's block is 128"):
        rec.forward(q, q, q, _causal_cfg(), mask=other)


def test_a_recorder_that_saw_no_mask_says_so():
    class Silent:
        def forward(self, q, k, v, cfg, mask=None):
            return q

    rec = pm.MaskRecorder(Silent(), estimator="s", code_path="s", mask_selector="none")
    rec.set_layer(0)
    q = torch.zeros(1, HQ, SEQ, D)
    with pytest.raises(pm.MaskSetError, match="nothing to record"):
        rec.forward(q, q, q, _causal_cfg())
    with pytest.raises(RuntimeError, match="set_layer"):
        pm.MaskRecorder(Silent(), estimator="s", code_path="s",
                        mask_selector="none").forward(q, q, q, _causal_cfg())


def test_the_stored_per_head_masks_pass_the_selector_gate():
    """Gate G3 on what the arm stores: with one score matrix broadcast to every
    head, the per-head selector's masks are the head-uniform (era 3) mask,
    bitwise, after recording."""
    one = torch.rand(NB, NB, generator=torch.Generator().manual_seed(5))
    uniform = masks.importance_block_mask_device(SEQ, 128, 0.5, one, causal=True,
                                                 identity_seed="gate")
    per_head = masks.importance_block_mask_per_head(SEQ, 128, 0.5, one.expand(HQ, NB, NB),
                                                    causal=True, identity_seed="gate")
    noop = types.SimpleNamespace(forward=lambda *a, **k: torch.zeros(1))
    stored = {}
    for name, m in (("uniform", uniform), ("per_head", per_head)):
        rec = pm.MaskRecorder(noop, estimator=name, code_path=name, mask_selector=name)
        rec.set_layer(0)
        q = torch.zeros(1, HQ, SEQ, D)
        rec.forward(q, q, q, _causal_cfg(), mask=m)
        stored[name] = rec.recorded[0]
    assert torch.equal(stored["uniform"], stored["per_head"])


# --- the arm --------------------------------------------------------------------------

def test_a_replay_reproduces_the_source_arms_output():
    wrapped, ids = _model(), _ids()
    rec, source = _record_meanpool(wrapped, ids)
    arm = _arm(rec.mask_set(ids, n_heads_q=HQ), ids)
    replay = wrapped.run_measured(ids, arm, cfg=_causal_cfg(), logits_to_keep=1).logits
    torch.testing.assert_close(replay, source, rtol=1e-4, atol=1e-5)
    other = wrapped.run_measured(_ids(seed=9), NaiveAttention(), cfg=_causal_cfg(),
                                 logits_to_keep=1).logits
    assert not torch.allclose(replay, other, atol=1e-3)       # the comparison could fail


def test_the_kernel_receives_the_recorded_masks_unchanged_and_already_resident():
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    ms = rec.mask_set(ids, n_heads_q=HQ)
    seen = []

    def spy(*a, **kw):
        seen.append(a[7])
        return fake_kernel(*a, **kw)
    arm = _arm(ms, ids, kernel=spy)
    for _ in range(2):
        wrapped.run_measured(ids, arm, cfg=_causal_cfg(), logits_to_keep=1)
    assert len(seen) == 2 * LAYERS
    for i, handed in enumerate(seen):
        layer = i % LAYERS
        assert torch.equal(handed, ms.masks[layer])
        assert handed.is_contiguous() and handed.dtype == torch.bool
        assert handed.data_ptr() == arm._masks[layer].data_ptr()   # no per-forward copy


def test_the_arm_is_timed_with_no_estimator_cost_and_is_labelled_an_ablation():
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    arm = gt.Arm("precomputed", gt.PRECOMPUTED, _arm(rec.mask_set(ids, n_heads_q=HQ), ids),
                 _causal_cfg(), prompt_digest=pm.prompt_digest(ids))
    dense = gt.Arm("dense", gt.DENSE, NaiveAttention(), _causal_cfg())
    clock = FakeClock()
    controls = gt.Controls(1e9, 0, cs.BlockSampler(ScriptedReader(clock),
                                                   flag_bits=cs.SW_THERMAL_SLOWDOWN))
    out = gt.time_arms(wrapped, ids, [arm, dense], reps=2, controls=controls)
    row = out["rows"][0]
    assert not row["deployable"] and row["arm_label"] == gt.PRECOMPUTED_LABEL
    assert row["score_calls_timed"] == 0 and row["mask_builds_timed"] == 0
    assert out["complete"]


def test_an_arm_that_secretly_builds_a_mask_is_refused():
    """Break-test of the harness check: a 'precomputed' arm whose forward
    builds its mask is not one."""
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)

    class Cheat(pm.PrecomputedMaskBackend):
        def forward(self, q, k, v, cfg, mask=None):
            masks.importance_block_mask_device(
                SEQ, 128, 0.5, torch.rand(NB, NB), causal=True, identity_seed="cheat")
            return super().forward(q, k, v, cfg, mask=mask)
    cheat = Cheat(rec.mask_set(ids, n_heads_q=HQ), ids, device="cpu", kernel=fake_kernel)
    arm = gt.Arm("precomputed", gt.PRECOMPUTED, cheat, _causal_cfg(),
                 prompt_digest=pm.prompt_digest(ids))
    dense = gt.Arm("dense", gt.DENSE, NaiveAttention(), _causal_cfg())
    clock = FakeClock()
    controls = gt.Controls(1e9, 0, cs.BlockSampler(ScriptedReader(clock),
                                                   flag_bits=cs.SW_THERMAL_SLOWDOWN))
    with pytest.raises(gt.TimedRegionError, match="computed offline"):
        gt.time_arms(wrapped, ids, [arm, dense], reps=2, controls=controls)


def test_masks_from_another_prompt_are_refused_by_the_arm_and_by_the_harness():
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    ms = rec.mask_set(ids, n_heads_q=HQ)
    with pytest.raises(pm.MaskSetError, match="another prompt"):
        _arm(ms, _ids(seed=9))
    arm = gt.Arm("precomputed", gt.PRECOMPUTED, _arm(ms, ids), _causal_cfg(),
                 prompt_digest=pm.prompt_digest(ids))
    dense = gt.Arm("dense", gt.DENSE, NaiveAttention(), _causal_cfg())
    clock = FakeClock()
    controls = gt.Controls(1e9, 0, cs.BlockSampler(ScriptedReader(clock),
                                                   flag_bits=cs.SW_THERMAL_SLOWDOWN))
    with pytest.raises(gt.TimedRegionError, match="another prompt"):
        gt.time_arms(wrapped, _ids(seed=9), [arm, dense], reps=2, controls=controls)


def test_a_precomputed_arm_must_name_its_prompt_and_only_it_may():
    with pytest.raises(ValueError, match="names the prompt"):
        gt.Arm("p", gt.PRECOMPUTED, None, None)
    with pytest.raises(ValueError, match="names the prompt"):
        gt.Arm("d", gt.DENSE, None, None, prompt_digest="x")


def test_the_harness_refuses_a_scoring_call_in_a_precomputed_arms_log():
    log = gt.EventLog()
    log.record("score", "x")                  # setup phase: computed before the clock
    log.open_rep = 0
    log.record("sync", "closing synchronise")
    arm = gt.Arm("p", gt.PRECOMPUTED, None, None, prompt_digest="d")
    with pytest.raises(gt.TimedRegionError, match="computed offline"):
        gt.check_timed_region(log, arm, [0])


def test_executed_density_is_recorded_at_the_kernel_and_equals_the_mask_density(monkeypatch):
    """The arm calls `kernels.bsa_prefill`, so the T2.20 recorder sees it like any
    other arm's call and reads the density off the tensor handed over."""
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    ms = rec.mask_set(ids, n_heads_q=HQ)
    fake_mod = types.ModuleType("block_sparse_attn")
    fake_mod.block_sparse_attn_func = fake_kernel
    monkeypatch.setitem(sys.modules, "block_sparse_attn", fake_mod)
    arm = pm.PrecomputedMaskBackend(ms, ids, device="cpu")        # the real import path
    with KernelInputRecorder() as krec:
        wrapped.run_measured(ids, arm, cfg=_causal_cfg(), logits_to_keep=1)
    assert krec.executed_density() == pytest.approx(ms.mask_density(), abs=1e-6)
    assert fake_mod.block_sparse_attn_func is fake_kernel         # restored


def test_the_arm_reports_a_density_per_layer_without_computing_one_in_the_forward():
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    arm = _arm(rec.mask_set(ids, n_heads_q=HQ), ids)
    wrapped.run_measured(ids, arm, cfg=_causal_cfg(), logits_to_keep=1)
    assert len(arm.last_layer_density) == LAYERS
    assert all(d is arm._density[i] for i, d in enumerate(arm.last_layer_density))
    arm.reset_density()
    assert arm.last_layer_density == []


def test_the_arm_refuses_what_it_cannot_replay():
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    arm = _arm(rec.mask_set(ids, n_heads_q=HQ), ids)
    q = torch.zeros(1, HQ, SEQ, D)
    kv = torch.zeros(1, HKV, SEQ, D)
    arm.set_layer(0)
    with pytest.raises(UnsupportedConfig, match="external mask"):
        arm.forward(q, kv, kv, _causal_cfg(), mask=object())
    with pytest.raises(UnsupportedConfig, match="expects 'causal'"):
        arm.forward(q, kv, kv, _sparse_cfg())
    with pytest.raises(UnsupportedConfig, match="batch 1"):
        arm.forward(torch.zeros(2, HQ, SEQ, D), kv, kv, _causal_cfg())
    with pytest.raises(UnsupportedConfig, match="seq_len"):
        short = torch.zeros(1, HQ, SEQ // 2, D)
        arm.forward(short, kv[:, :, :SEQ // 2], kv[:, :, :SEQ // 2], _causal_cfg())
    arm.set_layer(LAYERS)
    with pytest.raises(UnsupportedConfig, match="no recorded mask for layer"):
        arm.forward(q, kv, kv, _causal_cfg())
    fresh = _arm(rec.mask_set(ids, n_heads_q=HQ), ids)
    with pytest.raises(RuntimeError, match="set_layer"):
        fresh.forward(q, kv, kv, _causal_cfg())


# --- the mask set ---------------------------------------------------------------------

def _set():
    wrapped, ids = _model(), _ids()
    rec, _ = _record_meanpool(wrapped, ids)
    return rec.mask_set(ids, n_heads_q=HQ), ids


def test_a_mask_set_round_trips_and_is_not_overwritten(tmp_path):
    ms, _ = _set()
    path = ms.save(tmp_path)
    back = pm.MaskSet.load(path)
    assert back.estimator == ms.estimator and back.prompt_digest == ms.prompt_digest
    assert all(torch.equal(back.masks[i], ms.masks[i]) for i in ms.masks)
    with pytest.raises(pm.MaskSetError, match="already exists"):
        ms.save(tmp_path)


def test_a_tampered_or_unhashed_mask_set_is_refused(tmp_path):
    ms, _ = _set()
    path = ms.save(tmp_path)
    side = path.with_name(path.name + ".sha256")
    good = side.read_text()
    side.write_text("0" * 64 + f"  {path.name}\n")
    with pytest.raises(pm.MaskSetError, match="sidecar's sha256"):
        pm.MaskSet.load(path)
    side.write_text("junk\n")
    with pytest.raises(pm.MaskSetError, match="sha256sum format"):
        pm.MaskSet.load(path)
    side.unlink()
    with pytest.raises(pm.MaskSetError, match="no sha256 sidecar"):
        pm.MaskSet.load(path)
    side.write_text(good)
    assert pm.MaskSet.load(path).n_layers == LAYERS


def test_a_malformed_mask_set_is_refused():
    ms, _ = _set()
    base = dict(estimator="e", code_path="c", mask_selector="head_uniform",
                prompt_digest="d", seq_len=SEQ, n_heads_q=HQ)
    with pytest.raises(pm.MaskSetError, match="at least one layer"):
        pm.MaskSet(masks={}, **base)
    with pytest.raises(pm.MaskSetError, match="layers must be"):
        pm.MaskSet(masks={1: ms.masks[0]}, **base)
    with pytest.raises(pm.MaskSetError, match="expected bool"):
        pm.MaskSet(masks={0: ms.masks[0].to(torch.uint8)}, **base)
    with pytest.raises(pm.MaskSetError, match="expected bool"):
        pm.MaskSet(masks={0: ms.masks[0][:, :, :2]}, **base)


def test_the_prompt_digest_separates_prompts_and_shapes():
    a, b = _ids(), _ids(seed=9)
    assert pm.prompt_digest(a) == pm.prompt_digest(a.clone())
    assert pm.prompt_digest(a) != pm.prompt_digest(b)
    assert pm.prompt_digest(a.view(1, 2, -1)) != pm.prompt_digest(a)
    last = a.clone()
    last[0, -1] = (last[0, -1] + 1) % 64            # differs in the final token only
    assert pm.prompt_digest(last) != pm.prompt_digest(a)
