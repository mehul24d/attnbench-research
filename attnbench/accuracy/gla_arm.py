"""The GLA accuracy-arm decision, as code rather than as judgement.

The thresholds are pre-registered in docs/gla_arm_decision.md and duplicated
here only as constants. The point of putting the rule in code is that the
expected result is an ambiguous small number, and a near-zero score is
compatible with three different states of the world. Choosing between them
after seeing the number is not a decision.

Nothing here needs a GPU: `evaluate` takes predictions and scores that a
caller has already produced, so the rule itself is tested on CPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

# See docs/gla_arm_decision.md for why each number is where it is. In short:
# 0.90 sits in empty space between the observed failure (0.05) and the dense
# arm (1.00); 0.50 asks only whether answer-shaped output happens at all; and
# 20.0 is a COST threshold, not a significance one -- below it the remaining
# bands buy a constant zero at ~0.35 h each.
MIN_DISTINCT_FRACTION = 0.90
MIN_FORMAT_VALID_FRACTION = 0.50
MIN_MEAN_SCORE = 20.0

# niah_single's answer is a random 7-digit number occurring exactly once in
# the context, so emitting it without retrieving it has probability ~1e-7.
_ANSWER_SHAPE = re.compile(r"\d{7}")


@dataclass(frozen=True)
class ArmVerdict:
    verdict: str                  # "keep" | "drop" | "no_verdict"
    reason: str
    distinct_fraction: float
    format_valid_fraction: float
    mean_score: float
    n: int
    failed_gate: Optional[int] = None

    @property
    def keeps_the_arm(self) -> bool:
        return self.verdict == "keep"


def _measure(predictions: Sequence[str], scores: Sequence[float]) -> tuple:
    n = len(predictions)
    if n == 0:
        raise ValueError(
            "no predictions: an empty set passes every fraction test "
            "vacuously and would report a verdict nothing measured")
    if len(scores) != n:
        raise ValueError(f"{n} predictions but {len(scores)} scores")
    distinct = len(set(predictions)) / n
    valid = sum(bool(_ANSWER_SHAPE.search(p)) for p in predictions) / n
    return distinct, valid, sum(scores) / n, n


def evaluate(predictions: Sequence[str], scores: Sequence[float], *,
             control_predictions: Sequence[str],
             control_scores: Sequence[float]) -> ArmVerdict:
    """Apply the three gates to GLA, after checking the dense control.

    `control_*` is the dense arm measured on the SAME examples in the same
    process. A failing control means the harness is broken and the GLA numbers
    are not interpretable -- a broken instrument is not evidence against the
    thing it was pointed at.
    """
    c_distinct, c_valid, c_score, _ = _measure(control_predictions, control_scores)
    if (c_distinct < MIN_DISTINCT_FRACTION or c_valid < MIN_FORMAT_VALID_FRACTION
            or c_score < MIN_MEAN_SCORE):
        d, v, s, n = _measure(predictions, scores)
        return ArmVerdict(
            "no_verdict",
            f"the dense control failed its own gates (distinct={c_distinct:.2f}, "
            f"format_valid={c_valid:.2f}, score={c_score:.1f}). The harness is "
            f"broken; the GLA numbers below are not interpretable and this is "
            f"not evidence against GLA.",
            d, v, s, n)

    distinct, valid, score, n = _measure(predictions, scores)

    if distinct < MIN_DISTINCT_FRACTION:
        return ArmVerdict(
            "drop",
            f"only {distinct:.0%} of predictions are distinct across {n} "
            f"different contexts -- the context is still not reaching the "
            f"computation. GLA is TIMING-ONLY; no accuracy comparison is "
            f"reported.", distinct, valid, score, n, failed_gate=1)

    if valid < MIN_FORMAT_VALID_FRACTION:
        return ArmVerdict(
            "drop",
            f"only {valid:.0%} of predictions contain an answer-shaped "
            f"7-digit number -- the output varies with input but is not "
            f"attempting the task. GLA is TIMING-ONLY; no accuracy comparison "
            f"is reported.", distinct, valid, score, n, failed_gate=2)

    if score < MIN_MEAN_SCORE:
        return ArmVerdict(
            "drop",
            f"coherent but not weight-compatible: output is input-dependent "
            f"({distinct:.0%} distinct) and answer-shaped ({valid:.0%}), and "
            f"retrieves {score:.1f} against the dense arm's {c_score:.1f}. "
            f"Report as a documented negative about MECHANISM SUBSTITUTION, "
            f"not as a finding about linear attention -- these weights were "
            f"never trained for it. Below {MIN_MEAN_SCORE} the remaining "
            f"bands buy a constant zero.",
            distinct, valid, score, n, failed_gate=3)

    return ArmVerdict(
        "keep",
        f"input-dependent ({distinct:.0%}), answer-shaped ({valid:.0%}), and "
        f"retrieves {score:.1f} against the dense arm's {c_score:.1f}. Keep "
        f"the arm, captioned: ungated gate, weights trained for softmax "
        f"attention.", distinct, valid, score, n)
