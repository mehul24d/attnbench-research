"""Shared Stage 3 grid construction: RULER examples keyed by (task, seq_len)
and each backend's curated AttnConfig list.

Extracted out of scripts/run_accuracy.py so scripts/time_one_accuracy_example.py
(the single-example pre-flight timing probe) builds configs and examples via
the exact same functions the full sweep uses, restricted to one (task,
seq_len, n=1) slice -- rather than a second, hand-rolled definition of "the
grid" that could silently drift from the real one.
"""

from __future__ import annotations

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
                              dense_backend: str = "sdpa_math",
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
    """
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
