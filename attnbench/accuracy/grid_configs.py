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
        dense_backend: [], "block_sparse": [], "gla": [],
    }
    if include_sage:
        configs_by_backend["sage"] = []

    for seq_len in (seq_lens if seq_lens is not None else grid.seq_lens):
        configs_by_backend[dense_backend].append(AttnConfig(
            seq_len=seq_len, batch=1, n_heads_q=1, n_heads_kv=1,
            head_dim=128, mask="causal"))
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


# Every backend in this study that has no decode path of its own decodes
# through this one. Named, not implicit: decision C is that sparsity applies
# during prefill only and generation runs full attention over the cache, and
# a row records which backend produced its text (see docs/limitations.md,
# "Sparsity is applied during prefill only").
DENSE_DECODE_BACKEND = "sdpa_math"


def backend_instance(name: str) -> AttentionBackend:
    """A backend by the name a result row carries.

    SDPA is one registered class with a pinned kernel per instance, so a row
    labelled `sdpa_math` and one labelled `sdpa_flash` are the same class and
    different kernels -- which is exactly the confound SDPABackend exists to
    remove, and why the kernel is part of the name rather than left to
    torch's internal dispatch.
    """
    if name.startswith("sdpa_"):
        return backends.get("sdpa")(kernel=name[len("sdpa_"):])
    return backends.get(name)()


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
