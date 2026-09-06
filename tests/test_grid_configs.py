"""grid_configs.py: shared Stage 3 config/example construction, locked down
with pytest instead of only a manual `--dry-run` print -- these functions
previously lived inline in scripts/run_accuracy.py with no automated
coverage of their own; the 19500/23400 cell counts were verified by eye.
"""

from __future__ import annotations

from pathlib import Path

from attnbench.accuracy.config import load_grid
from attnbench.accuracy.sizing import approximate_token_count
from attnbench.accuracy.grid_configs import (
    build_configs_by_backend, build_examples_by_task_length)

GRID_PATH = (Path(__file__).resolve().parents[1]
             / "configs" / "accuracy" / "stage3_grid.yaml")


def test_build_examples_covers_every_task_and_seq_len():
    grid = load_grid(GRID_PATH)
    examples = build_examples_by_task_length(grid, seed=0,
                                             count_tokens=approximate_token_count)
    assert set(examples) == {(task, seq_len) for task in grid.tasks
                              for seq_len in grid.seq_lens}
    for (task, seq_len), exs in examples.items():
        assert len(exs) == grid.seq_lens[seq_len]


def test_build_examples_restricts_to_requested_subset():
    grid = load_grid(GRID_PATH)
    examples = build_examples_by_task_length(
        grid, seed=0, tasks=("niah_single",), seq_lens={32768: grid.seq_lens[32768]},
        count_tokens=approximate_token_count)
    assert set(examples) == {("niah_single", 32768)}
    assert len(examples[("niah_single", 32768)]) == grid.seq_lens[32768]


def test_build_configs_by_backend_curated_lists_match_pinned_cell_count():
    """Regression pin for the dry-run's verified cell count: 3 backends x
    5 seq_lens, block_sparse contributing 3 configs/length (one per
    sparsity) vs. 1 config/length for the dense baseline and gla."""
    grid = load_grid(GRID_PATH)
    configs = build_configs_by_backend(grid, include_sage=False)

    # sdpa_flash, not sdpa_math: the dense arm is pinned in the grid as data
    # (see stage3_grid.yaml's dense_backend). It was an unexamined "sdpa_math"
    # keyword default until 2026-09-06, while the grid's whole hour estimate
    # rested on a measured sdpa_flash anchor -- 9.2x apart on 68% of the
    # prefill work, and the two never met.
    assert set(configs) == {grid.dense_backend, "block_sparse", "gla"}
    assert grid.dense_backend == "sdpa_flash"
    assert len(configs[grid.dense_backend]) == len(grid.seq_lens)
    assert len(configs["gla"]) == len(grid.seq_lens)
    assert len(configs["block_sparse"]) == len(grid.seq_lens) * len(grid.sparsities)


def test_build_configs_by_backend_include_sage_adds_a_fourth_backend():
    grid = load_grid(GRID_PATH)
    configs = build_configs_by_backend(grid, include_sage=True)
    assert "sage" in configs
    assert len(configs["sage"]) == len(grid.seq_lens)
    for cfg in configs["sage"]:
        assert cfg.quant_scheme is not None


def test_build_configs_by_backend_restricts_to_requested_seq_lens():
    grid = load_grid(GRID_PATH)
    configs = build_configs_by_backend(grid, include_sage=False, seq_lens=(32768,))
    assert all(cfg.seq_len == 32768 for cfgs in configs.values() for cfg in cfgs)
    assert len(configs["block_sparse"]) == len(grid.sparsities)


# ---------------------------------------------------------------------------
# Band-aligned segments
#
# Stage 3 runs in four segments cut on band boundaries. A segment is expressed
# by restricting which bands run, never by editing the pinned grid -- every
# segment has to agree about what the whole experiment is or the segments
# cannot be joined.
# ---------------------------------------------------------------------------

def test_a_bands_examples_are_identical_whether_other_bands_run_or_not():
    """The invariant the segmentation rests on.

    Resume works by (config_key, backend, task, example_id), so if running
    band 2048 alone produced different examples than running it alongside
    4096, segment 2 would resume-skip against rows describing different
    prompts -- and nothing downstream could tell, because the ids would still
    line up. Generation is per (task, seq_len) from the same seed, so this
    holds; it is asserted rather than assumed.
    """
    from attnbench.accuracy.config import load_grid
    from attnbench.accuracy.sizing import approximate_token_count

    grid = load_grid("configs/accuracy/stage3_grid.yaml")
    small = {2048: 4}
    together = {2048: 4, 4096: 4}

    alone = build_examples_by_task_length(
        grid, seed=0, count_tokens=approximate_token_count,
        tasks=("niah_single",), seq_lens=small)
    with_others = build_examples_by_task_length(
        grid, seed=0, count_tokens=approximate_token_count,
        tasks=("niah_single",), seq_lens=together)

    a = alone[("niah_single", 2048)]
    b = with_others[("niah_single", 2048)]
    assert [e.example_id for e in a] == [e.example_id for e in b]
    assert [e.context for e in a] == [e.context for e in b]
    assert [e.answer for e in a] == [e.answer for e in b]


def test_restricting_bands_restricts_configs_to_the_same_set():
    """A config at a band with no examples contributes no cells, so a
    mismatch here would be silent rather than an error -- it would just
    quietly plan a segment of the wrong size."""
    from attnbench.accuracy.config import load_grid
    grid = load_grid("configs/accuracy/stage3_grid.yaml")
    configs = build_configs_by_backend(grid, include_sage=False,
                                       seq_lens=(2048, 4096))
    seen = {cfg.seq_len for cfgs in configs.values() for cfg in cfgs}
    assert seen == {2048, 4096}
