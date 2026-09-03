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

    assert set(configs) == {"sdpa_math", "block_sparse", "gla"}
    assert len(configs["sdpa_math"]) == len(grid.seq_lens)
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
