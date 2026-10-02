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
#
# "minference_meanpool_inline" (2026-10-01, audit C1) is the same estimator
# run inside the measured forward: each layer ranks its own q and k, which
# come from sparse earlier layers, and its cost is inside latency_ms. The
# two-pass "minference_meanpool" ranks every layer from a dense pass first,
# outside the timer. Same formula, different inputs and accounting, so
# different values.
#
# "xattention_inline" (2026-10-02) is XAttention's own antidiagonal estimator,
# inside the measured forward like the inline arm above, but selecting blocks
# per head by a mass threshold rather than to a fixed sparsity. Its rows carry
# `xattn_threshold` and `realised_density` instead of `sparsity`.
ScoreSource = Literal["dense_softmax_fp32", "minference_meanpool",
                      "minference_meanpool_inline", "xattention_inline"]

# Same reasoning as ScoreSource: a real, first-class field rather than a
# docstring footnote. "noise"/"needle" mean the example used attnbench's
# dependency-free haystack substitution (see
# attnbench/_vendor/ruler/VENDORED.md) rather than RULER's own "essay"
# haystack -- a real methodological difference that makes absolute
# accuracy numbers here not comparable to published RULER results, even
# though they remain valid for comparing backends against each other on
# identical inputs. "essay" is RULER's own prose haystack, wired 2026-10-01
# (audit T4) for the niah_multikey_1 / multivalue / multiquery presets.
# "documents" is the QA tasks' filler: real distractor documents from SQuAD
# or HotpotQA, as RULER's qa_1 / qa_2 build it.
HaystackMode = Literal["noise", "needle", "essay", "documents"]

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

# Which forget gate produced a row, for the backends that have one.
#
# The third field of this kind, and it exists for the same reason as the
# other two: a caveat that lives in a docstring does not travel with the
# data. `score_source` says a mask's ranking came from a dense pass rather
# than a cheap estimator; `haystack_mode` says the context was attnbench's
# filler rather than RULER's essays; `gate_source` says which gate a linear
# row's recurrence used -- and that is the difference between a number worth
# reading and 900 rows of fluent noise (silent_failure_patterns.md #17).
#
# "learned" is unreachable today and deliberately still listed: Qwen2.5 has
# no gate projection to borrow, so the value exists as the thing the study
# does NOT have, not as an option waiting to be selected.
GateSource = Literal["learned", "synthetic", "ungated"]

# Backends whose output depends on a forget gate, and which therefore may
# not write a row without naming it. Gated DeltaNet joins this set the day
# backends/linear.py grows it (see that module's docstring).
#
# A name set rather than `backend_role == "linear"`: an ungated linear
# backend would be role "linear" and have no gate to record, and demanding
# one from it would push a caller toward inventing a value. Kept honest from
# the other side by test_gate_source_on_rows.py, which asserts every
# registered backend carrying a `gate_source` attribute is named here.
GATED_BACKENDS = frozenset({"gla"})

# Backends that choose their own blocks from a threshold, so that how sparse
# a row was is an outcome rather than a setting. Their rows must carry the
# threshold that ran and the density it produced; every other row must carry
# neither. Same reasoning as GATED_BACKENDS: without the two columns an
# XAttention row cannot be placed against the fixed-sparsity arms at all.
SELF_SELECTING_BACKENDS = frozenset({"xattention"})


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
    # True when the caller pinned the dense-decode fallback to a historical
    # value instead of the current DENSE_DECODE_BACKEND (audit S1a). Such a
    # row replays an old regime and is NOT current-code output.
    decode_pinned: bool = False
    # Read off the backend instance that actually ran, in generation.py --
    # never passed in alongside it. A gate named by a caller can disagree
    # with the gate the recurrence used, and the disagreement is invisible
    # in the output, which is precisely how #17 happened.
    gate_source: Optional[GateSource] = None
    # Read off the backend instance, like gate_source: the threshold it ran
    # at, and after the prefill the mean over layers of the fraction of
    # causally valid blocks it kept.
    # None for every backend that does not select its own blocks.
    xattn_threshold: Optional[float] = None
    # Set instead of xattn_threshold when the backend ran a calibrated
    # per-(layer, head) table: "<name>:<sha256[:12]>" of its values.
    xattn_calibration: Optional[str] = None
    realised_density: Optional[float] = None
    realised_density_by_layer: Optional[list] = None


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
    mask_source: Optional[Literal["random", "importance",
                                  "importance_randfree"]]
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
    # See Generated.decode_pinned. Carried so a replayed-regime row can never
    # be read as current-code output.
    decode_pinned: bool = False
    # None for every backend without a forget gate, and required for every
    # backend with one -- enforced below rather than trusted, because the
    # whole point of the column is that a linear row is uninterpretable
    # without it.
    gate_source: Optional[GateSource] = None
    # Required for SELF_SELECTING_BACKENDS and None otherwise (enforced
    # below): the threshold the estimator ran at, and the mean realised
    # density over layers it produced on this example.
    xattn_threshold: Optional[float] = None
    # Exactly one of xattn_threshold and xattn_calibration is set on a
    # self-selecting row: a scalar tau, or a calibrated table's label.
    xattn_calibration: Optional[str] = None
    realised_density: Optional[float] = None
    realised_density_by_layer: Optional[list] = None
    latency_ms: Optional[float] = None
    detail: str = ""

    def __post_init__(self) -> None:
        """Refuse to exist as a row that cannot be read correctly.

        Both directions, because both have a failure mode:

        - A `gla` row with no `gate_source` is the 2026-09-06 parquet again:
          900 rows that look like every other row and mean nothing, with the
          reason recorded only in a commit message. The arm decision
          (docs/gla_arm_decision.md) can end in KEEP, and a kept arm is
          reported with a caveat that has to be attached to the data, not to
          the paper draft -- summaries drop caveats, columns survive them.

        - A non-gated backend carrying a `gate_source` asserts a mechanism
          that is not there. `sdpa_flash` has no recurrence and no gate; a
          value in that column would make a reader believe the dense arm was
          configured, and would break the one query the column exists for
          (`df.groupby("gate_source")`).

        Raised at construction, so the bad row never reaches the buffer, let
        alone the parquet.
        """
        gated = self.backend in GATED_BACKENDS
        if gated and self.gate_source is None:
            raise ValueError(
                f"backend {self.backend!r} has a forget gate, so a row from it "
                f"must record which one ran. A linear-attention score is not "
                f"interpretable without it: the same backend produced 100.0 "
                f"and 0.0 from the same weights depending on this field. See "
                f"docs/gla_arm_decision.md.")
        if not gated and self.gate_source is not None:
            raise ValueError(
                f"backend {self.backend!r} has no forget gate, so "
                f"gate_source={self.gate_source!r} on its row claims a "
                f"mechanism that did not run. Leave it None.")
        selecting = self.backend in SELF_SELECTING_BACKENDS
        setting = (self.xattn_threshold is not None) + (self.xattn_calibration is not None)
        density = self.realised_density is not None
        if selecting and (setting != 1 or not density):
            raise ValueError(
                f"backend {self.backend!r} selects its own blocks, so its row "
                f"must record exactly one setting that ran -- a threshold or "
                f"a calibration -- and the density it produced "
                f"(xattn_threshold={self.xattn_threshold!r}, "
                f"xattn_calibration={self.xattn_calibration!r}, "
                f"realised_density={self.realised_density!r}).")
        if not selecting and (setting or density
                              or self.realised_density_by_layer is not None):
            raise ValueError(
                f"backend {self.backend!r} does not select its own blocks; "
                f"xattn_threshold, xattn_calibration and the densities must be None.")

    def to_dict(self) -> dict:
        return asdict(self)
