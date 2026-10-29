"""T2.13's timed region (plan Section 5.3 step 7; Appendix A, 26 Oct 2026).

Every deployable arm's timed region includes its estimator, mask build, copies
and synchronisations; the oracle may use cached scores and is labelled a
non-deployable reference (scoring cost outside the timed path). These tests run the real wrapper on a toy Llama on
CPU, with `NaiveAttention` standing in for the block-sparse kernel.

The acceptance test is `test_scores_computed_before_the_timed_block_are_refused`:
it reproduces the pattern of `run_vectorised_endtoend.py` (score first, then
time `run_measured` with the scores passed in) for a deployable arm, and the
harness must refuse it. `test_a_probe_that_sees_no_scorer_is_refused` is the
break-test for the check itself: if the probe stopped seeing the scorer the
forward calls, every check would pass vacuously, and it must not.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench import ground_truth as gt  # noqa: E402
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.backends.impls import NaiveAttention  # noqa: E402
from attnbench.backends.xattention import XAttentionBackend  # noqa: E402
from attnbench.clock_sampling import RESEARCHER_FLAG_BITS, BlockSampler, ClockSample  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

SEQ, BLOCK, LAYERS, REPS = 64, 8, 3, 3
PLAN = Path(__file__).resolve().parent.parent / "docs" / "RESEARCH_PLAN.md"


def _wrapped():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    return SwappableAttentionModel(model, _cfg(), model_id="toy", finest_block_size=BLOCK)


def _cfg(**kw):
    base = dict(seq_len=SEQ, batch=1, n_heads_q=4, n_heads_kv=2, head_dim=8,
                dtype="float32", mask="block_sparse", block_size=BLOCK,
                sparsity=0.75, mask_source="importance")
    base.update(kw)
    return AttnConfig(**base)


def _ids():
    return torch.randint(0, 64, (1, SEQ), generator=torch.Generator().manual_seed(3))


def _inline():
    return gt.Arm("meanpool", gt.INLINE, NaiveAttention(), _cfg(),
                  inline_estimator="minference_meanpool")


def _dense():
    return gt.Arm("dense", gt.DENSE, NaiveAttention(), _cfg(mask="causal", sparsity=None,
                                                             block_size=None))


def _oracle():
    return gt.Arm("oracle", gt.ORACLE, NaiveAttention(), _cfg())


class _QuietCard:
    """A card that never throttles. These tests are about the timed region;
    the controls have their own file (test_ground_truth_controls.py), so the
    floor here is too large to flag a round on a real CPU clock."""
    observes_transients = True

    def read(self):
        return ClockSample(1000, 1000, 0, {"power": 0, "thermal": 0})


def _controls():
    return gt.Controls(resolution_floor_ms=1e9, max_rerun_rounds=0,
                       sampler=BlockSampler(_QuietCard(), flag_bits=RESEARCHER_FLAG_BITS,
                                            interval_s=0.001))


class _Syncs:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1


def test_a_deployable_inline_arm_times_its_estimator():
    syncs = _Syncs()
    out = gt.time_arms(_wrapped(), _ids(), [_inline(), _dense()], reps=REPS,
                       controls=_controls(), synchronize=syncs)
    inline, dense = out["rows"]
    assert inline["deployable"] and inline["arm_label"] == gt.DEPLOYABLE_LABEL
    # One scoring call and one mask build per layer, in every timed block.
    assert inline["score_calls_timed"] == REPS * LAYERS
    assert inline["mask_builds_timed"] == REPS * LAYERS
    assert dense["score_calls_timed"] == dense["mask_builds_timed"] == 0
    assert len(inline["prefill_ms_samples"]) == REPS
    # Both sides of every timed block, and of every clocked warm-up forward.
    assert syncs.n == 2 * 2 * REPS + 2 * 2 * 3
    assert out["first_clocked_block_started_at"] is not None


def test_the_oracle_may_cache_its_scores_and_is_labelled(tmp_path):
    out = gt.time_arms(_wrapped(), _ids(), [_oracle(), _dense()], reps=REPS,
                       controls=_controls(),
                       score_kwargs=dict(task="t", example_id="e", cache_dir=str(tmp_path)))
    row = out["rows"][0]
    assert not row["deployable"]
    assert row["arm_label"] == gt.ORACLE_LABEL == ("non-deployable reference "
                                                  "(scoring cost outside the timed path)")
    assert row["score_calls_timed"] == 0              # scored before the clock
    assert row["mask_builds_timed"] == REPS * LAYERS  # masks are still timed


def test_scores_computed_before_the_timed_block_are_refused(monkeypatch, tmp_path):
    """The acceptance test: a deployable arm timed the way
    run_vectorised_endtoend.py times its arms."""
    def score_first(wrapped, input_ids, arm, *, score_kwargs):
        return {"layer_scores": wrapped.compute_importance_scores(
            input_ids, task="t", example_id="e", cache_dir=str(tmp_path))}
    monkeypatch.setattr(gt, "untimed_setup", score_first)
    with pytest.raises(gt.TimedRegionError, match="outside the timed block"):
        gt.time_arms(_wrapped(), _ids(), [_inline(), _dense()], reps=REPS,
                     controls=_controls())


def test_a_probe_that_sees_no_scorer_is_refused(monkeypatch):
    """Break-test: blind the probe and the inline arm must fail, not pass."""
    monkeypatch.setattr(gt, "SCORE_TARGETS", ())
    with pytest.raises(gt.TimedRegionError, match="computed no scores"):
        gt.time_arms(_wrapped(), _ids(), [_inline(), _dense()], reps=REPS,
                     controls=_controls())


def test_an_estimator_inside_the_backend_is_seen(monkeypatch):
    """XAttention's estimator runs in XAttentionBackend.estimate, inside the
    backend's forward. A stand-in backend calls it the same way."""
    monkeypatch.setattr(XAttentionBackend, "estimate", lambda self, q, k: None)

    class _XAStandIn(NaiveAttention):
        def forward(self, q, k, v, cfg, mask=None):
            XAttentionBackend.estimate(self, q, k)
            return super().forward(q, k, v, replace(cfg, mask="causal"))

    arm = gt.Arm("xa", gt.IN_BACKEND, _XAStandIn(), _cfg(mask="causal", sparsity=None,
                                                          block_size=None))
    row = gt.time_arms(_wrapped(), _ids(), [arm, _dense()], reps=REPS,
                       controls=_controls())["rows"][0]
    assert row["deployable"] and row["score_calls_timed"] == REPS * LAYERS


def test_a_block_without_its_closing_synchronise_is_refused():
    log = gt.EventLog()
    log.open_rep = 0
    log.record("score", "x")
    log.open_rep = None
    with pytest.raises(gt.TimedRegionError, match="closing synchronise"):
        gt.check_timed_region(log, _inline(), [0])


def test_the_oracle_may_not_cache_its_masks():
    log = gt.EventLog()
    log.record("mask_build", "x")          # setup phase
    log.open_rep = 0
    log.record("sync", "closing synchronise")
    with pytest.raises(gt.TimedRegionError, match="not its masks"):
        gt.check_timed_region(log, _oracle(), [0])


def test_the_probe_restores_what_it_patched():
    from attnbench import masks
    from attnbench.accuracy import model
    before = (model.minference_meanpool_scores_on_device, masks.mask_for,
              SwappableAttentionModel.__dict__["compute_importance_scores"],
              XAttentionBackend.__dict__["estimate"])
    with gt.ScoringProbe(gt.EventLog()):
        assert masks.mask_for is not before[1]
    after = (model.minference_meanpool_scores_on_device, masks.mask_for,
             SwappableAttentionModel.__dict__["compute_importance_scores"],
             XAttentionBackend.__dict__["estimate"])
    assert after == before


def test_quantiles_match_numpy():
    np = pytest.importorskip("numpy")
    xs = [5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 7.0]
    for q in (0.5, 0.9, 0.99):
        assert gt.quantile(xs, q) == pytest.approx(float(np.quantile(xs, q)))


def test_arm_refusals():
    with pytest.raises(ValueError):
        gt.Arm("a", "nope", None, None)
    with pytest.raises(ValueError):
        gt.Arm("a", gt.INLINE, None, None)            # inline without an estimator
    with pytest.raises(ValueError):
        gt.Arm("a", gt.DENSE, None, None, inline_estimator="minference_meanpool")


def test_the_plan_states_the_requirement():
    text = " ".join(PLAN.read_text().split())
    assert ("every deployable arm's timed region includes its estimator, mask build, "
            "copies and synchronisations") in text
    assert "non-deployable reference (scoring cost outside the timed path)" in text
    assert "upper bound" not in text.split("## Appendix A")[0]
    assert "attnbench/ground_truth.py" in text
