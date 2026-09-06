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

# Why a generation stopped. Recorded per row rather than reconstructed
# afterwards from `len(predicted)`, which cannot distinguish "the model
# finished its answer in 7 tokens" from "the model was cut off at 7".
#
# "cap" is the one that carries information about a BACKEND rather than
# about an example: if sparse backends hit the cap more often than dense
# ones on the same prompts, that is a finding (sparsity made the model
# ramble) and not a scoring artefact -- but only if the reason is on the
# row, because a truncated answer and a wrong answer both just score 0.
StopReason = Literal["eos", "newline", "cap"]

# What a backend is DOING in this study, as opposed to what it is called.
#
# Stage 3's backends have curated, asymmetric roles rather than a uniform
# sweep: exactly one dense arm supplies the sparsity=0 reference every other
# arm is compared against, block_sparse supplies the sparsity sweep, gla the
# linear-attention point, sage the quantized one. Until 2026-09-06 that lived
# only in a docstring, so a row said `backend="sdpa_flash"` and nothing in the
# data said it was THE reference arm.
#
# That matters because the identity of the dense arm is a study-design fact
# that must hold across every band: if it ever changed mid-grid, the halves
# would not be comparable, and a reader holding only the parquet could not
# tell. With this column that question is `df.groupby("backend_role")
# ["backend"].nunique()` -- one row, one answer.
BackendRole = Literal["dense_reference", "block_sparse", "linear", "quantized"]


@dataclass(frozen=True)
class Generated:
    """What a `generate_fn` hands back to the runner.

    Was a bare `(text, latency_ms)` tuple. It stopped being one when
    generation became real: the stopping reason and the decode backend are
    facts about how a row was produced that nothing downstream can recover
    from the text, and a tuple has no room for them that does not silently
    change meaning when a field is added.

    Every field except `text` is optional so a test stub stays one line --
    but they arrive as nulls in the parquet, visibly absent, rather than
    being inferred.
    """

    text: str
    latency_ms: Optional[float] = None
    stop_reason: Optional[StopReason] = None
    n_generated: Optional[int] = None
    decode_backend: Optional[str] = None


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

    `backend` is the PREFILL backend, which is the cell's identity and the
    thing under test. `decode_backend` is separate because they differ by
    design: sparsity is applied during prefill only, so a block_sparse row
    generates its text with a dense kernel over the cache (see
    docs/limitations.md, "Sparsity is applied during prefill only"). There
    is deliberately no `prefill_backend` field duplicating `backend` -- two
    columns that must always agree eventually disagree, and then neither
    can be trusted.
    """

    backend: str
    backend_role: BackendRole
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
    # Generation facts, carried rather than reconstructed. A query for the
    # per-backend truncation rate is `df[df.stop_reason == "cap"]`, not an
    # inference from answer length against a per-task cap table that would
    # have to be kept in sync with the one generation actually used.
    stop_reason: Optional[StopReason] = None
    n_generated: Optional[int] = None
    decode_backend: Optional[str] = None
    latency_ms: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
