"""timing_probe.py: the empirical-TFLOPS extrapolation math, validated with
synthetic PhaseTiming/AttnConfig values -- no model or GPU needed, mirroring
test_matched.py's synthetic-array discipline for Stage 4.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from attnbench.accuracy.timing_probe import (
    ModelArchitecture, PhaseTiming, _with_real_heads, blended_tflops_by_category,
    corrected_grid_hours, decode_flops, total_grid_flops_by_category,
    whole_model_flops)
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
#
# The tests in this section pass `decode_steps_by_task={}` deliberately: they
# assert properties of the PREFILL term (scoring amortization, real head
# geometry, no token inflation) and adding a decode term to them would only
# make the arithmetic harder to read. The decode term has its own section
# below. `{}` is spelled out at each call rather than defaulted, which is the
# whole point of the argument being required.
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

    totals = total_grid_flops_by_category(configs_by_backend, examples_by_task_length, arch=arch, decode_steps_by_task={})

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

    totals = total_grid_flops_by_category(configs_by_backend, examples_by_task_length, arch=arch, decode_steps_by_task={})

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
        arch=_arch(n_heads_q=1, n_heads_kv=1), decode_steps_by_task={})
    totals_12_head = total_grid_flops_by_category(
        configs_by_backend, examples_by_task_length,
        arch=_arch(n_heads_q=12, n_heads_kv=2), decode_steps_by_task={})

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
        {"sdpa_math": [cfg]}, {("niah_single", 8192): [object()]}, arch=arch,
        decode_steps_by_task={})
    assert totals["measured"] == whole_model_flops(
        _with_real_heads(cfg, arch), arch, logits_to_keep=1)


def test_missing_examples_for_a_config_seq_len_contributes_zero_not_a_crash():
    configs_by_backend = {"sdpa_math": [_cfg(seq_len=2048)]}
    totals = total_grid_flops_by_category(configs_by_backend, examples_by_task_length={},
                                          arch=_arch(), decode_steps_by_task={})
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


# ---------------------------------------------------------------------------
# decode_flops -- the unit-of-work term
#
# The estimate that was wrong by ~20x had no arithmetic error in it. Every
# input was still valid; a Stage 3 row was simply not one forward pass. These
# tests are about the unit, not the number.
# ---------------------------------------------------------------------------

def test_zero_decode_steps_cost_nothing():
    assert decode_flops(_cfg(seq_len=8192), _arch(), n_steps=0) == 0


def test_decode_is_linear_in_context_where_prefill_is_quadratic():
    """The whole reason the cache makes generation affordable. Doubling
    context roughly doubles a decode step's attention but roughly quadruples
    a prefill's -- so the decode SHARE shrinks as the grid gets longer, and
    the band where decode matters most is the shortest one."""
    arch = _arch()
    share = {}
    for s in (2048, 4096, 8192):
        cfg = _cfg(seq_len=s)
        d = decode_flops(cfg, arch, n_steps=19)
        share[s] = d / whole_model_flops(_with_real_heads(cfg, arch), arch)
    assert share[2048] > share[4096] > share[8192]


def test_a_decode_step_attends_the_growing_cache_not_a_fixed_one():
    """Step i attends seq_len + i keys. If this were charged at a constant
    seq_len the error would be tiny at these step counts and invisible --
    which is exactly why it is asserted rather than eyeballed."""
    arch = _arch()
    cfg = _cfg(seq_len=1024)
    one = decode_flops(cfg, arch, n_steps=1)
    two = decode_flops(cfg, arch, n_steps=2)
    # the second step costs one more key's worth of attention than the first
    per_key = 4 * cfg.batch * arch.n_heads_q * arch.head_dim * arch.n_layers
    assert (two - one) - one == per_key


def test_a_sparse_config_is_charged_dense_decode_because_that_is_what_runs():
    """Decision C: sparsity is applied during prefill only. A block_sparse
    row's decode really does run full attention over the cache, so charging
    it the dense cost is the model being right, not the model being lazy."""
    arch = _arch()
    dense = _cfg(seq_len=4096)
    sparse = _cfg(seq_len=4096, mask="block_sparse", sparsity=0.9,
                  block_size=128, mask_source="importance")
    assert (decode_flops(dense, arch, n_steps=19)
            == decode_flops(sparse, arch, n_steps=19))
    # ... while their PREFILL costs do differ, so the test is not vacuous
    assert (whole_model_flops(_with_real_heads(sparse, arch), arch)
            < whole_model_flops(_with_real_heads(dense, arch), arch))


def test_grid_total_charges_decode_per_task_into_its_own_category():
    """Its own category, never folded into "measured".

    Folding it in was this module's own version of the bug it documents:
    corrected_grid_hours divides each category by that category's TFLOPS, and
    those are compute-bound prefill figures. On 2026-09-06 the probe printed
    "grid decode ~= 5.35 h" from the bandwidth model and "+0.03 h (0.18%)"
    from the FLOPs path, three lines apart, in the same run."""
    arch = _arch()
    cfg = _cfg(seq_len=2048)
    examples = {("niah_single", 2048): list(range(5)),
                ("vt", 2048): list(range(5))}
    configs = {"sdpa_math": [cfg],
               "block_sparse": [_cfg(seq_len=2048, mask="block_sparse", sparsity=0.5,
                                     block_size=128, mask_source="importance")]}

    prefill_only = total_grid_flops_by_category(
        configs, examples, arch=arch, decode_steps_by_task={})
    with_decode = total_grid_flops_by_category(
        configs, examples, arch=arch,
        decode_steps_by_task={"niah_single": 8, "vt": 17})

    assert with_decode["scoring"] == prefill_only["scoring"]   # scoring generates nothing
    assert with_decode["measured"] == prefill_only["measured"], (
        "decode must not be folded into the compute-bound category")
    real = _with_real_heads(cfg, arch)
    expected = 2 * 5 * (decode_flops(real, arch, n_steps=8)
                        + decode_flops(real, arch, n_steps=17))
    assert with_decode["decode"] == expected
    assert prefill_only["decode"] == 0


def test_a_task_with_examples_but_no_step_count_raises():
    """Silently costing an unlisted task at zero decode steps is a claim
    about the model's stopping behaviour. It has to be written down."""
    with pytest.raises(KeyError, match="decode_steps_by_task"):
        total_grid_flops_by_category(
            {"sdpa_math": [_cfg(seq_len=2048)]},
            {("niah_single", 2048): [object()], ("vt", 2048): [object()]},
            arch=_arch(), decode_steps_by_task={"niah_single": 8})


def test_the_cap_bound_is_strictly_worse_than_the_expected_case():
    """Both tables are priced because the expected one depends on the model
    emitting a newline where we predict it will, and the cap one does not."""
    from attnbench.accuracy.timing_probe import (
        DECODE_STEPS_BY_TASK, DECODE_STEPS_BY_TASK_AT_CAP)
    assert set(DECODE_STEPS_BY_TASK) == set(DECODE_STEPS_BY_TASK_AT_CAP)
    for task, expected in DECODE_STEPS_BY_TASK.items():
        assert DECODE_STEPS_BY_TASK_AT_CAP[task] > expected, task


# ---------------------------------------------------------------------------
# Decode is bandwidth-bound
#
# The FLOPs count above is right. Dividing it by a prefill throughput is not
# a decode time, and the error is 177x, not a rounding difference. These
# tests pin the regime distinction rather than the numbers.
# ---------------------------------------------------------------------------

QWEN_1_5B = ModelArchitecture(n_layers=28, hidden_size=1536,
                              intermediate_size=8960, n_heads_q=12,
                              n_heads_kv=2, head_dim=128, vocab_size=151936)


def test_parameter_bytes_matches_the_real_model_size():
    """Qwen2.5-1.5B is 1.54B parameters, 3.09 GB in bf16. If this drifts,
    every decode estimate built on it drifts with it and nothing else
    would notice."""
    from attnbench.accuracy.timing_probe import model_parameter_bytes
    assert 3.0e9 < model_parameter_bytes(QWEN_1_5B) < 3.2e9


def test_the_flops_view_and_the_bandwidth_view_disagree_by_two_orders():
    """The finding, asserted so it cannot quietly stop being true.

    Pricing a decode step at the prefill's measured 42.2 TFLOPS gives ~0.07
    ms. The bandwidth floor is ~14 ms. A cost model that used the first
    number would report generation as free -- which is exactly what the
    'under 1% decode tax' in docs/stage3_generation_decision.md did before
    this was caught.
    """
    from attnbench.accuracy.timing_probe import (
        PEAK_BANDWIDTH_BYTES_PER_S, decode_seconds)
    cfg = _cfg(seq_len=8192)
    as_flops = decode_flops(cfg, QWEN_1_5B, n_steps=1) / 42.151e12
    as_bandwidth = decode_seconds(
        cfg, QWEN_1_5B, n_steps=1,
        peak_bandwidth_bytes_per_s=PEAK_BANDWIDTH_BYTES_PER_S["L4"])
    assert as_bandwidth / as_flops > 100


def test_decode_cost_is_nearly_flat_in_context_while_prefill_is_quadratic():
    """Why the tax is largest where prefill is cheapest. The weight-read term
    does not depend on context, so a 16x longer context costs a decode step
    only a little more -- while the prefill it follows costs ~256x more."""
    from attnbench.accuracy.timing_probe import (
        PEAK_BANDWIDTH_BYTES_PER_S, decode_seconds)
    bw = PEAK_BANDWIDTH_BYTES_PER_S["L4"]
    short = decode_seconds(_cfg(seq_len=2048), QWEN_1_5B, n_steps=19,
                           peak_bandwidth_bytes_per_s=bw)
    long = decode_seconds(_cfg(seq_len=32768), QWEN_1_5B, n_steps=19,
                          peak_bandwidth_bytes_per_s=bw)
    assert 1.0 < long / short < 1.5, (long, short)


def test_the_kv_term_is_gqa_shaped_not_expanded():
    """SDPABackend.state_from_prefill keeps the cache un-expanded, which is
    what makes the cache-size claim honest -- so the traffic model has to
    count n_heads_kv, not n_heads_q. Counting the expanded form would
    overstate the cache read 6x on this model."""
    from attnbench.accuracy.timing_probe import decode_memory_traffic_bytes
    cfg = _cfg(seq_len=32768)
    gqa = decode_memory_traffic_bytes(cfg, QWEN_1_5B, n_steps=1)
    mha = decode_memory_traffic_bytes(
        cfg, replace(QWEN_1_5B, n_heads_kv=QWEN_1_5B.n_heads_q), n_steps=1)
    assert mha > gqa


def test_the_overhead_term_is_zero_only_when_asked_for():
    """A floor is a floor. This implementation runs a Python-level HF forward
    per step with no CUDA graphs, so real per-step cost is above it -- and
    the difference is the whole width of the estimate's bracket."""
    from attnbench.accuracy.timing_probe import (
        PEAK_BANDWIDTH_BYTES_PER_S, decode_seconds)
    bw = PEAK_BANDWIDTH_BYTES_PER_S["L4"]
    cfg = _cfg(seq_len=8192)
    floor = decode_seconds(cfg, QWEN_1_5B, n_steps=10,
                           peak_bandwidth_bytes_per_s=bw)
    real = decode_seconds(cfg, QWEN_1_5B, n_steps=10,
                          peak_bandwidth_bytes_per_s=bw,
                          per_step_overhead_s=0.015)
    assert real - floor == pytest.approx(0.15)


def test_corrected_grid_hours_refuses_to_price_decode_from_tflops():
    """The guard that makes the separation load-bearing rather than tidy.
    Dividing decode FLOPs by a prefill TFLOPS figure is wrong by ~177x on
    this model, and it produced a plausible sub-1% number that sat three
    lines from the correct 5.35 h."""
    with pytest.raises(ValueError, match="BANDWIDTH-bound"):
        corrected_grid_hours({"measured": 10**15, "decode": 10**13},
                             {"measured": 40.0, "decode": 40.0})
    # zero decode is not an error -- a prefill-only estimate is legitimate
    assert corrected_grid_hours({"measured": 10**15, "decode": 0},
                                {"measured": 40.0})["total"] > 0
