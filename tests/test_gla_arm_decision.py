"""The GLA arm decision, tested against the four states of the world it has
to separate.

The rule is pre-registered in docs/gla_arm_decision.md. These tests exist so
the thresholds cannot drift after the measurement exists -- a rule that can be
edited once the number is known is not a pre-registration.
"""

from __future__ import annotations

import pytest

from attnbench.accuracy.gla_arm import (
    MIN_DISTINCT_FRACTION, MIN_FORMAT_VALID_FRACTION, MIN_MEAN_SCORE,
    ArmVerdict, evaluate)

N = 100


def _dense_control(score=100.0):
    """The positive control: input-dependent, answer-shaped, retrieves."""
    return ([f"The special magic number is {1000000 + i}." for i in range(N)],
            [score] * N)


def _varied_answer_shaped(hit_rate):
    """Distinct, answer-shaped predictions that retrieve `hit_rate` of the
    time -- the 'coherent substitution' state of the world."""
    preds = [f"The special magic number is {2000000 + i}." for i in range(N)]
    scores = [100.0 if i < int(hit_rate * N) else 0.0 for i in range(N)]
    return preds, scores


def test_the_failure_actually_observed_drops_the_arm_at_gate_1():
    """Segment 1's real numbers: 15 distinct predictions across 300 prompts."""
    preds = ["uals...\n\n"] * 95 + [f"variant {i}" for i in range(5)]
    cp, cs = _dense_control()
    v = evaluate(preds, [0.0] * N, control_predictions=cp, control_scores=cs)
    assert v.verdict == "drop" and v.failed_gate == 1
    assert "TIMING-ONLY" in v.reason


def test_varied_garbage_drops_the_arm_at_gate_2():
    """The state the user named: 'garbage that merely varies'. Input-dependent
    but not attempting the task -- which a score alone cannot distinguish from
    attempting it and failing."""
    preds = [f"特别声明 curtains blamecharges polit {i}" for i in range(N)]
    cp, cs = _dense_control()
    v = evaluate(preds, [0.0] * N, control_predictions=cp, control_scores=cs)
    assert v.verdict == "drop" and v.failed_gate == 2
    assert v.distinct_fraction == 1.0, "gate 1 must have passed, or gate 2 is untested"


def test_coherent_but_not_retrieving_drops_the_arm_at_gate_3():
    """Answer-shaped, input-dependent, retrieves nothing. A real result about
    mechanism substitution -- and explicitly NOT a claim about linear
    attention, because the weights were never trained for it."""
    preds, scores = _varied_answer_shaped(hit_rate=0.0)
    cp, cs = _dense_control()
    v = evaluate(preds, scores, control_predictions=cp, control_scores=cs)
    assert v.verdict == "drop" and v.failed_gate == 3
    assert "MECHANISM SUBSTITUTION" in v.reason
    assert "never trained for it" in v.reason


def test_real_retrieval_keeps_the_arm():
    preds, scores = _varied_answer_shaped(hit_rate=0.45)
    cp, cs = _dense_control()
    v = evaluate(preds, scores, control_predictions=cp, control_scores=cs)
    assert v.verdict == "keep" and v.keeps_the_arm
    assert "ungated gate" in v.reason


def test_a_broken_control_yields_no_verdict_rather_than_a_drop():
    """A failing instrument is not evidence against the thing it was pointed
    at. If the dense arm cannot clear its own gates on the same examples in
    the same process, the GLA numbers mean nothing -- and the rule must say
    so rather than quietly recording a drop."""
    preds, scores = _varied_answer_shaped(hit_rate=0.45)
    v = evaluate(preds, scores, control_predictions=["same"] * N,
                 control_scores=[0.0] * N)
    assert v.verdict == "no_verdict"
    assert "not evidence against GLA" in v.reason


def test_an_empty_measurement_raises_instead_of_passing_every_gate():
    """Every fraction test passes vacuously on an empty set. That would be a
    verdict nothing measured -- the same shape as an empty canary reporting
    no drift."""
    cp, cs = _dense_control()
    with pytest.raises(ValueError, match="vacuously"):
        evaluate([], [], control_predictions=cp, control_scores=cs)


def test_the_thresholds_match_the_pre_registration():
    """The rule was written down before the measurement. If these constants
    move, the document and the code have to move together and the change is
    visible in one diff -- which is what makes it a pre-registration rather
    than a preference."""
    doc = open("docs/gla_arm_decision.md").read()
    assert MIN_DISTINCT_FRACTION == 0.90 and "0.90" in doc
    assert MIN_FORMAT_VALID_FRACTION == 0.50 and "0.50" in doc
    assert MIN_MEAN_SCORE == 20.0 and "20.0" in doc
    assert "PRE-REGISTERED" in doc


def test_gate_order_is_load_bearing():
    """Gates run in order and the FIRST failure decides what we report. A
    degenerate arm and a coherent-but-empty one get different write-ups, and
    reporting the wrong one would overstate what was learned."""
    cp, cs = _dense_control()
    both_bad = ["uals..."] * N          # fails gate 1 AND gate 2
    v = evaluate(both_bad, [0.0] * N, control_predictions=cp, control_scores=cs)
    assert v.failed_gate == 1, "gate 1 must decide when both fail"
