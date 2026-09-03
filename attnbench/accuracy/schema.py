"""Stage 3 per-example result row. Same shape as gates.ProbeResult /
gates.CorrectnessResult / timing.Measurement: a flat dataclass with a
to_dict() for writing into a dataframe row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

# Closed today (one dense-softmax scorer exists), extended the day a cheap
# approximate estimator exists -- see accuracy/model.py's
# compute_importance_scores docstring for why this must never be silently
# assumed constant. `None` for dense (non-block-sparse) rows: no scoring
# pass ran at all for those.
ScoreSource = Literal["dense_softmax_fp32"]

# Same reasoning as ScoreSource: a real, first-class field rather than a
# docstring footnote. "noise"/"needle" mean the example used attnbench's
# dependency-free haystack substitution (see
# attnbench/_vendor/ruler/VENDORED.md) rather than RULER's own "essay"
# haystack -- a real methodological difference that makes absolute
# accuracy numbers here not comparable to published RULER results, even
# though they remain valid for comparing backends against each other on
# identical inputs. "essay" is reserved for when that path is wired up.
HaystackMode = Literal["noise", "needle", "essay"]


@dataclass
class AccuracyResult:
    """One RULER-style example, one backend, one config.

    `score_source` records how the importance ranking behind this row's
    mask was produced -- not a footnote, a first-class field, because a
    dense-softmax-derived score is a real cost/fidelity trade-off (an upper
    bound on what a deployed cheap estimator would achieve, with the
    estimator's own cost excluded from every latency number in the study).
    That caveat must travel with the row, not live only in a docstring
    someone has to go find.

    `expected` is a semicolon-joined string of the accepted answer(s) --
    ruler.score() takes the underlying list directly; this field is the
    flattened form for a parquet row, matching every other field here.
    """

    backend: str
    config_key: str
    task: str
    example_id: str
    context_length: int
    mask_source: Optional[Literal["random", "importance"]]
    sparsity: Optional[float]
    score_source: Optional[ScoreSource]
    haystack_mode: Optional[HaystackMode]
    predicted: str
    expected: str
    score: float
    correct: bool
    latency_ms: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
