"""timing_probe.py: the empirical-TFLOPS extrapolation math, validated with
synthetic PhaseTiming/AttnConfig values -- no model or GPU needed, mirroring
test_matched.py's synthetic-array discipline for Stage 4.
"""

from __future__ import annotations

import pytest

from attnbench.accuracy.timing_probe import (
    ModelArchitecture, PhaseTiming, _with_real_heads, blended_tflops_by_category,
    corrected_grid_hours, total_grid_flops_by_category, whole_model_flops)
from attnbench.config import AttnConfig


def _cfg(seq_len=1024, **overrides):
    fields = dict(seq_len=seq_len, batch=1, n_heads_q=1, n_heads_kv=1, head_dim=128,
                  mask="causal")
    fields.update(overrides)
    return AttnConfig(**fields)


def _arch(**overrides) -> ModelArchitecture:
    # Loosely Qwen2.5-1.5B-Instruct-shaped, not exact -- real values are
    # fetched from the model's own config in production, never hardcoded.
    fields = dict(n_layers=28, hidden_size=1536, intermediate_size=8960,
                  n_heads_q=12, n_heads_kv=2, head_dim=128, vocab_size=151936)
    fields.update(overrides)
    return ModelArchitecture(**fields)


# ---------------------------------------------------------------------------
# PhaseTiming
# ---------------------------------------------------------------------------

def test_effective_tflops_arithmetic():
    p = PhaseTiming(label="x", category="measured", flops=int(2e12), wall_seconds=2.0)
    assert p.effective_tflops == pytest.approx(1.0)


def test_phase_timing_rejects_nonpositive_wall_seconds():
    with pytest.raises(ValueError):
        PhaseTiming(label="x", category="measured", flops=1, wall_seconds=0.0)


def test_phase_timing_rejects_nonpositive_flops():
    with pytest.raises(ValueError):
        PhaseTiming(label="x", category="measured", flops=0, wall_seconds=1.0)


# ---------------------------------------------------------------------------
# blended_tflops_by_category
# ---------------------------------------------------------------------------

def test_blended_tflops_weights_by_time_not_a_plain_average():
    # phase A: 1 TFLOPS over 100s (dominates the blend)
    # phase B: 100 TFLOPS over 1s
    a = PhaseTiming(label="a", category="measured", flops=int(1e12), wall_seconds=100.0)
    b = PhaseTiming(label="b", category="measured", flops=int(100e12), wall_seconds=1.0)
    blended = blended_tflops_by_category([a, b])
    # total flops = 101e12, total seconds = 101 -> ~1.0 TFLOPS, not (1+100)/2=50.5
    assert blended["measured"] == pytest.approx(1.0, rel=0.05)


def test_blended_tflops_separates_categories():
    scoring = PhaseTiming(label="s", category="scoring", flops=int(4e12), wall_seconds=2.0)
    measured = PhaseTiming(label="m", category="measured", flops=int(9e12), wall_seconds=3.0)
    blended = blended_tflops_by_category([scoring, measured])
    assert blended["scoring"] == pytest.approx(2.0)
    assert blended["measured"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# whole_model_flops / ModelArchitecture
# ---------------------------------------------------------------------------

def test_whole_model_flops_exceeds_attention_only_by_roughly_n_layers():
    """The bug this whole module exists to fix: cfg.useful_flops() alone is
    one layer's attention core, but a real forward pass runs every layer
    plus its projections and MLP. whole_model_flops must be dramatically
    larger -- on the order of n_layers times, not equal to attention alone."""
    cfg = _cfg(seq_len=2048)
    arch = _arch()
    attention_only = cfg.useful_flops()
    whole = whole_model_flops(cfg, arch, logits_to_keep=1)
    assert whole > attention_only * arch.n_layers  # MLP/QKVO add on top of the multiple


def test_whole_model_flops_scales_linearly_with_n_layers():
    cfg = _cfg(seq_len=2048)
    arch_1layer = _arch(n_layers=1)
    arch_2layer = _arch(n_layers=2)
    flops_1 = whole_model_flops(cfg, arch_1layer, logits_to_keep=1)
    flops_2 = whole_model_flops(cfg, arch_2layer, logits_to_keep=1)
    lm_head = 2 * cfg.batch * 1 * arch_1layer.hidden_size * arch_1layer.vocab_size
    # per-layer cost (whole minus the once-only lm_head) must exactly double
    assert (flops_2 - lm_head) == pytest.approx((flops_1 - lm_head) * 2, rel=1e-9)


def test_logits_to_keep_affects_only_the_lm_head_term():
    cfg = _cfg(seq_len=2048)
    arch = _arch()
    flops_keep_1 = whole_model_flops(cfg, arch, logits_to_keep=1)
    flops_keep_all = whole_model_flops(cfg, arch, logits_to_keep=cfg.seq_len)
    lm_head_1 = 2 * cfg.batch * 1 * arch.hidden_size * arch.vocab_size
    lm_head_all = 2 * cfg.batch * cfg.seq_len * arch.hidden_size * arch.vocab_size
    assert flops_keep_all - flops_keep_1 == pytest.approx(lm_head_all - lm_head_1, rel=1e-9)


def test_gqa_reduces_kv_projection_flops_versus_mha():
    """k_proj/v_proj cost scales with n_heads_kv, not n_heads_q -- a GQA
    model's KV projections are cheaper than an MHA model's, and the FLOPs
    formula must reflect that, not silently assume n_heads_q for KV too."""
    cfg = _cfg(seq_len=2048)
    gqa = _arch(n_heads_q=12, n_heads_kv=2)
    mha = _arch(n_heads_q=12, n_heads_kv=12)
    assert whole_model_flops(cfg, gqa) < whole_model_flops(cfg, mha)


# ---------------------------------------------------------------------------
# total_grid_flops_by_category
# ---------------------------------------------------------------------------

def test_scoring_pass_amortized_once_per_example_not_per_sparsity():
    """The regression this exists to catch: block_sparse contributes one
    config per sparsity level, but the scoring pass behind all of them is
    the same dense pass, computed once per example -- not once per
    sparsity level."""
    arch = _arch()
    configs_by_backend = {
        "sdpa_math": [_cfg(seq_len=1024)],
        "block_sparse": [_cfg(seq_len=1024, mask="block_sparse", sparsity=sp,
                               block_size=128, mask_source="importance")
                          for sp in (0.5, 0.75, 0.9)],
    }
    examples_by_task_length = {("niah_single", 1024): list(range(10))}  # 10 examples

    totals = total_grid_flops_by_category(configs_by_backend, examples_by_task_length, arch=arch)

    dense_cfg = configs_by_backend["sdpa_math"][0]
    expected_scoring = 10 * whole_model_flops(_with_real_heads(dense_cfg, arch), arch,
                                              logits_to_keep=1)
    assert totals["scoring"] == expected_scoring


def test_measured_flops_sums_every_backend_config_and_example():
    arch = _arch()
    dense_cfg = _cfg(seq_len=1024)
    sparse_cfg = _cfg(seq_len=1024, mask="block_sparse", sparsity=0.5,
                       block_size=128, mask_source="importance")
    configs_by_backend = {"sdpa_math": [dense_cfg], "block_sparse": [sparse_cfg]}
    examples_by_task_length = {("niah_single", 1024): list(range(4))}

    totals = total_grid_flops_by_category(configs_by_backend, examples_by_task_length, arch=arch)

    expected_measured = (4 * whole_model_flops(_with_real_heads(dense_cfg, arch), arch,
                                               logits_to_keep=1)
                         + 4 * whole_model_flops(_with_real_heads(sparse_cfg, arch), arch,
                                                 logits_to_keep=1))
    assert totals["measured"] == expected_measured


def test_total_grid_flops_uses_real_heads_not_config_placeholder_heads():
    """The regression this exists to catch: configs_by_backend's configs
    carry placeholder n_heads_q=1 (fine for cell-counting, wrong for a
    FLOPs estimate) -- the real architecture's head count must be what
    actually drives the computed total, not the placeholder on the config."""
    placeholder_cfg = _cfg(seq_len=1024, n_heads_q=1, n_heads_kv=1)
    configs_by_backend = {"sdpa_math": [placeholder_cfg]}
    examples_by_task_length = {("niah_single", 1024): [object()]}

    totals_1_head = total_grid_flops_by_category(
        configs_by_backend, examples_by_task_length,
        arch=_arch(n_heads_q=1, n_heads_kv=1))
    totals_12_head = total_grid_flops_by_category(
        configs_by_backend, examples_by_task_length,
        arch=_arch(n_heads_q=12, n_heads_kv=2))

    assert totals_12_head["measured"] > totals_1_head["measured"]


def test_seq_len_is_taken_as_a_real_token_count_with_no_inflation_applied():
    """The inflation correction is retired. accuracy/sizing.py now fits each
    generated context to its budget against the real tokenizer, so a config's
    seq_len IS a token count -- applying a ratio on top would double-count
    the very thing that was fixed at the source.
    """
    arch = _arch()
    cfg = _cfg(seq_len=8192)
    totals = total_grid_flops_by_category(
        {"sdpa_math": [cfg]}, {("niah_single", 8192): [object()]}, arch=arch)
    assert totals["measured"] == whole_model_flops(
        _with_real_heads(cfg, arch), arch, logits_to_keep=1)


def test_missing_examples_for_a_config_seq_len_contributes_zero_not_a_crash():
    configs_by_backend = {"sdpa_math": [_cfg(seq_len=2048)]}
    totals = total_grid_flops_by_category(configs_by_backend, examples_by_task_length={},
                                          arch=_arch())
    assert totals["measured"] == 0
    assert totals["scoring"] == 0


# ---------------------------------------------------------------------------
# corrected_grid_hours
# ---------------------------------------------------------------------------

def test_corrected_grid_hours_converts_flops_and_tflops_to_hours():
    # 3600e12 FLOPs at 1 TFLOPS = 3600 seconds = 1 hour
    totals = {"measured": int(3600e12), "scoring": int(1800e12)}
    measured_tflops = {"measured": 1.0, "scoring": 1.0}
    hours = corrected_grid_hours(totals, measured_tflops)
    assert hours["measured"] == pytest.approx(1.0)
    assert hours["scoring"] == pytest.approx(0.5)
    assert hours["total"] == pytest.approx(1.5)


def test_corrected_grid_hours_zero_flops_category_is_zero_without_needing_a_measurement():
    totals = {"measured": int(3600e12), "scoring": 0}
    measured_tflops = {"measured": 1.0}
    hours = corrected_grid_hours(totals, measured_tflops)
    assert hours["scoring"] == 0.0


def test_corrected_grid_hours_raises_on_missing_measurement_for_nonzero_category():
    totals = {"measured": int(3600e12), "scoring": int(1e12)}
    measured_tflops = {"measured": 1.0}  # scoring never probed
    with pytest.raises(KeyError):
        corrected_grid_hours(totals, measured_tflops)
