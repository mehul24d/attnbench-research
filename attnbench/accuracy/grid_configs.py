"""Shared Stage 3 grid construction: RULER examples keyed by (task, seq_len)
and each backend's curated AttnConfig list.

Extracted out of scripts/run_accuracy.py so scripts/time_one_accuracy_example.py
(the single-example pre-flight timing probe) builds configs and examples via
the exact same functions the full sweep uses, restricted to one (task,
seq_len, n=1) slice -- rather than a second, hand-rolled definition of "the
grid" that could silently drift from the real one.
"""

from __future__ import annotations

from .. import backends
from .schema import GATED_BACKENDS
from ..backends.base import AttentionBackend
from .config import AccuracyGrid
from .ruler import RulerExample, generate_examples
from .sizing import TokenCounter
from ..config import AttnConfig

# NOTE: the old `_APPROX_TOKENS_PER_HAYSTACK_UNIT = 20` words-per-unit
# heuristic that used to live here is gone. It made a grid seq_len a target
# rather than a token count, and real contexts came out ~21% longer, which
# is what made every Stage 3 hour estimate low. Sizing is now solved against
# the real tokenizer in accuracy/sizing.py; a grid seq_len is a token
# budget and means what it says.

# SageAttention needs cfg.quant_scheme set (base.py's claims_support doesn't
# check this -- it's a forward()-time requirement, not a declared
# capability). Placeholder pending confirmation against the real
# sageattention library's scheme names and a matching gates.TOL entry.
SAGE_QUANT_SCHEME = "int8"


def build_examples_by_task_length(grid: AccuracyGrid, seed: int, *,
                                   count_tokens: TokenCounter,
                                   tasks: tuple[str, ...] | None = None,
                                   seq_lens: dict[int, int] | None = None,
                                   ) -> dict[tuple[str, int], list[RulerExample]]:
    """One generate_examples() call per (task, seq_len), keyed on the
    GRID's seq_len -- this is what makes build_cells's context-length
    matching correct.

    A grid seq_len is now passed straight through as a TOKEN BUDGET, with
    no unit conversion: sizing.fit_units_to_budget solves the filler count
    against `count_tokens` so the rendered prompt lands at that many
    tokens. Previously this divided seq_len by a words-per-unit constant,
    which is why real contexts ran ~21% over.

    `count_tokens` is required. Production callers pass the target model's
    real tokenizer; dry runs and tests pass
    `sizing.approximate_token_count` explicitly, so an estimate is never a
    silent default.

    `tasks`/`seq_lens` default to the full grid; a caller that only needs
    one (task, seq_len) slice (the timing probe) passes single-entry
    overrides rather than generating -- and paying for -- the whole grid.
    `seq_lens`, like `grid.seq_lens`, maps seq_len -> n_per_length.
    """
    out = {}
    for task in (tasks if tasks is not None else grid.tasks):
        for seq_len, n_per_length in (seq_lens if seq_lens is not None else grid.seq_lens).items():
            out[(task, seq_len)] = generate_examples(
                task, [seq_len], n_per_length, seed=seed,
                count_tokens=count_tokens)
    return out


def build_configs_by_backend(grid: AccuracyGrid, *, include_sage: bool,
                              dense_backend: str | None = None,
                              seq_lens: tuple[int, ...] | None = None,
                              include_gla: bool = True,
                              ) -> dict[str, list[AttnConfig]]:
    """Each backend's own curated config list.

    Backends have curated, asymmetric roles, not a uniform sweep grid: one
    dense baseline (every exact-math kernel computes identical attention,
    Stage 1 already verifies that against a float64 reference, so only one
    needs to run here); block_sparse contributes its sparsity sweep; gla and
    sage (--include-sage) each contribute their own single operating point.

    `seq_lens` restricts which lengths get configs -- defaults to every
    length in the grid; the timing probe passes just the longest one.

    `dense_backend` defaults to the GRID's pinned choice, not to a value
    written here. It was `"sdpa_math"` as a script default until 2026-09-06,
    while the grid's whole hour estimate rested on a `sdpa_flash` throughput
    anchor -- a 9.2x disagreement on 68% of the prefill work, hidden in a
    keyword default. The dense arm is study design and belongs in the pinned
    data with the rest of it.
    """
    if dense_backend is None:
        dense_backend = grid.dense_backend
    configs_by_backend: dict[str, list[AttnConfig]] = {
        dense_backend: [], "block_sparse": [],
    }
    # `include_gla` is what a DROP verdict from docs/gla_arm_decision.md
    # actually does. Without it the pre-registered rule could reach a
    # decision the pipeline had no way to act on -- and since gla's default
    # gate REFUSES, an unactioned DROP does not skip the arm quietly, it
    # raises UnsupportedConfig on the first gla cell and takes the rest of
    # the chained bands with it. A decision procedure whose outcome cannot
    # be expressed is not a decision procedure.
    if include_gla:
        configs_by_backend["gla"] = []
    if include_sage:
        configs_by_backend["sage"] = []

    for seq_len in (seq_lens if seq_lens is not None else grid.seq_lens):
        configs_by_backend[dense_backend].append(AttnConfig(
            seq_len=seq_len, batch=1, n_heads_q=1, n_heads_kv=1,
            head_dim=128, mask="causal"))
        if include_gla:
            configs_by_backend["gla"].append(AttnConfig(
                seq_len=seq_len, batch=1, n_heads_q=1, n_heads_kv=1,
                head_dim=128, mask="causal"))
        for sparsity in grid.sparsities:
            configs_by_backend["block_sparse"].append(AttnConfig(
                seq_len=seq_len, batch=1, n_heads_q=1, n_heads_kv=1,
                head_dim=128, mask="block_sparse", sparsity=sparsity,
                block_size=grid.finest_block_size, mask_source="importance"))
        if include_sage:
            configs_by_backend["sage"].append(AttnConfig(
                seq_len=seq_len, batch=1, n_heads_q=1, n_heads_kv=1,
                head_dim=128, mask="causal", quant_scheme=SAGE_QUANT_SCHEME))

    return configs_by_backend


# Backends whose Stage 3 rows are excluded from every accuracy ANALYSIS.
#
# `gla` is here because of the pre-registered DROP verdict
# (docs/gla_arm_decision.md, decided 2026-09-07): the synthetic gate gives a
# 1.24-token memory horizon, and Qwen2.5 has no gate projection to borrow, so
# the rows are a measurement of a mechanism that could not retrieve rather
# than a measurement of linear attention. `results/stage3_s1/INVALID_ROWS.md`
# records which rows.
#
# This lives HERE, next to the arm decision that produced it, because it was
# restated in four places -- run_pareto, run_matched_analysis,
# run_decode_confound (each `EXCLUDE_BACKENDS = ("gla",)`) and
# run_phase_timing (an inline `df.backend != "gla"`, which does not even grep
# the same way) -- with two different justifications between them. A verdict
# that four call sites each re-derive is a verdict that cannot be revisited:
# reopening the GLA arm would need three of the four found by hand. See
# docs/silent_failure_patterns.md #23 for the general form.
ACCURACY_EXCLUDED_BACKENDS = ("gla",)


# Every backend in this study that has no decode path of its own decodes
# through this one. Named, not implicit: decision C is that sparsity applies
# during prefill only and generation runs full attention over the cache, and
# a row records which backend produced its text (see docs/limitations.md,
# "Sparsity is applied during prefill only").
DENSE_DECODE_BACKEND = "sdpa_math"


def backend_instance(name: str, *, gate_source: str | None = None) -> AttentionBackend:
    """A backend by the name a result row carries.

    SDPA is one registered class with a pinned kernel per instance, so a row
    labelled `sdpa_math` and one labelled `sdpa_flash` are the same class and
    different kernels -- which is exactly the confound SDPABackend exists to
    remove, and why the kernel is part of the name rather than left to
    torch's internal dispatch.

    `gate_source` is passed through to backends that have a forget gate and
    REFUSED for those that do not, rather than accepted and ignored. An
    ignored keyword is how a run ends up believing it configured something it
    did not -- and here the thing it would believe it configured is the one
    that decides whether the rows mean anything.

    Omitting it leaves the gated backend on its own default, which is the
    value that refuses to run (backends/linear.py). That is deliberate: the
    absence of a choice must not resolve to a choice.
    """
    if name.startswith("sdpa_"):
        cls, kwargs = backends.get("sdpa"), {"kernel": name[len("sdpa_"):]}
    else:
        cls, kwargs = backends.get(name), {}
    if gate_source is not None:
        if name not in GATED_BACKENDS:
            raise ValueError(
                f"gate_source={gate_source!r} passed for backend {name!r}, "
                f"which has no forget gate. Accepting it here would silently "
                f"do nothing while reading as though the backend had been "
                f"configured.")
        kwargs["gate_source"] = gate_source
    return cls(**kwargs)


def gate_source_of(backend: AttentionBackend) -> str | None:
    """Which forget gate this backend instance is configured with, or None
    if it has no gate at all.

    Read off the OBJECT that ran, which is the whole point. The alternative
    -- threading the choice down beside the backend as a second parameter --
    creates two places that name the gate and no way to notice when they
    disagree, and a disagreement here is invisible in the output. That is
    exactly the shape of silent_failure_patterns.md #17.

    `getattr` rather than an isinstance check on GatedLinearAttention: the
    property being recorded is "this backend has a configurable gate", and a
    second gated backend (Gated DeltaNet is planned) should be picked up by
    having the attribute, not by being added to a list here as well as to
    schema.GATED_BACKENDS.
    """
    return getattr(backend, "gate_source", None)


def decode_backend_for(backend: AttentionBackend) -> AttentionBackend:
    """The backend that will generate this row's text.

    Itself where it has a decode path (`gla` keeps its own fixed-size
    recurrent state, which is the bounded-vs-unbounded comparison Stage 3
    exists to make visible); the dense fallback otherwise. Never silent:
    `generate()` refuses to choose for a backend with no decode path, so
    this function is where the choice is made and it is recorded on the row.
    """
    if type(backend).supports_decode():
        return backend
    return backend_instance(DENSE_DECODE_BACKEND)
