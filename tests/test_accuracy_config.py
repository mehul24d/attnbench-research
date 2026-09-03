"""stage3_grid.yaml -> AccuracyGrid: the pinned Stage 3 grid must load back
out exactly, since a silent edit dropping the 32K cap or swapping in an
unvetted model ID should fail here, not three tool calls into a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attnbench.accuracy.config import load_grid

GRID_PATH = (Path(__file__).resolve().parents[1]
             / "configs" / "accuracy" / "stage3_grid.yaml")


def test_pinned_grid_loads_exact_values():
    grid = load_grid(GRID_PATH)

    assert grid.model_primary == "Qwen/Qwen2.5-1.5B-Instruct"
    assert grid.model_alternate == "meta-llama/Llama-3.2-1B-Instruct"
    assert grid.seq_lens == {2048: 300, 4096: 300, 8192: 300,
                             16384: 300, 32768: 100}
    assert grid.block_sizes == (128,)
    assert grid.sparsities == (0.5, 0.75, 0.9)
    assert grid.score_dtype == "fp16"


def test_seq_lens_capped_at_32k():
    grid = load_grid(GRID_PATH)
    assert max(grid.seq_lens) == 32768, (
        "hardware ceiling (CLAUDE.md) caps this study at 32K -- a grid "
        "edit that raises this must fail a test, not surface mid-run"
    )


def test_32k_is_marked_directional_and_underpowered():
    grid = load_grid(GRID_PATH)
    assert grid.is_directional(32768)
    assert grid.seq_lens[32768] < grid.seq_lens[16384], (
        "32768 must be a reduced-n directional point, not run at the same "
        "power-adequate n as the main claim"
    )


def test_main_lengths_are_not_directional():
    grid = load_grid(GRID_PATH)
    for seq_len in (2048, 4096, 8192, 16384):
        assert not grid.is_directional(seq_len)


def test_all_three_sparsity_levels_are_present():
    """0.75 was cut on cost alone and RESTORED 2026-09-03 once session 4
    measured real throughput (+0.74h).

    Two points can only support a line through the accuracy-vs-sparsity
    relationship; three can show curvature in it. That relationship is the
    study's headline question, so the third level is not a luxury. Matching
    Stage 2's (0.5, 0.75, 0.9) also restores cross-stage comparability at
    every operating point rather than two of three.
    """
    grid = load_grid(GRID_PATH)
    assert grid.sparsities == (0.5, 0.75, 0.9), (
        "all three levels are restored -- dropping one again is a budget "
        "decision that must be recorded in stage3_grid.yaml, not a tidy-up")


def test_16384_is_restored_to_full_n():
    """16384 went to n=200 as budget cut B and was RESTORED to 300 on
    2026-09-03 when the two-sided rule fired.

    The rule required the 16384 anchor to come in better than the 8192
    extrapolation by enough to open ~5h. Measured throughput came in at ~3x
    the 15-TFLOPS planning assumption, so it opened far more, and B is a
    main-grid point worth more than the 32768 directional one.
    """
    grid = load_grid(GRID_PATH)
    assert grid.seq_lens[16384] == 300, (
        "16384 is a main-grid point at full n since the two-sided rule "
        "resolved; reducing it again needs a recorded reason")


def test_32768_reserve_cut_is_not_applied():
    """The n=100 -> 50 cut at 32768 is held in reserve against a worse-
    than-extrapolated 16384 anchor. It is recorded as a comment in
    stage3_grid.yaml and must not be applied pre-emptively -- firing it
    early would silently underpower the only long-context point in the
    study to buy headroom nothing has yet asked for.
    """
    grid = load_grid(GRID_PATH)
    assert grid.seq_lens[32768] == 100


def test_finest_block_size_is_the_minimum():
    grid = load_grid(GRID_PATH)
    assert grid.finest_block_size == min(grid.block_sizes)


def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "bad_grid.yaml"
    bad.write_text("seq_lens: {1024: 10}\nblock_sizes: [128]\n")
    with pytest.raises(KeyError):
        load_grid(bad)


def test_missing_model_subkey_raises(tmp_path):
    bad = tmp_path / "bad_grid.yaml"
    bad.write_text(
        "model:\n  primary: some/model\n"
        "seq_lens: {1024: 10}\nblock_sizes: [128]\nsparsities: [0.5]\n"
        "tasks: [niah_single]\nscore_dtype: fp16\n"
        "score_cache_dir: results/accuracy/score_cache\n"
    )
    with pytest.raises(KeyError):
        load_grid(bad)


def test_directional_seq_len_not_in_seq_lens_raises(tmp_path):
    bad = tmp_path / "bad_grid.yaml"
    bad.write_text(
        "model:\n  primary: a\n  alternate: b\n"
        "seq_lens: {1024: 10}\ndirectional_seq_lens: [99999]\n"
        "block_sizes: [128]\nsparsities: [0.5]\ntasks: [niah_single]\n"
        "score_dtype: fp16\nscore_cache_dir: results/accuracy/score_cache\n"
    )
    with pytest.raises(KeyError):
        load_grid(bad)


def test_grid_seq_lens_are_token_budgets_that_contexts_actually_hit():
    """Since sizing.py, a grid seq_len is an exact token budget rather than
    a word-count target. That makes the pinned numbers mean something
    specific, so pin the property too: every configured length must produce
    a context at or just under it.

    Before the sizing fix a 'seq_len' of 16384 produced ~19821 real tokens.
    A regression to unit-based sizing would blow this tolerance immediately
    rather than quietly re-inflating every hour estimate by ~21%.
    """
    from attnbench.accuracy.grid_configs import build_examples_by_task_length
    from attnbench.accuracy.sizing import approximate_token_count

    grid = load_grid(GRID_PATH)
    examples = build_examples_by_task_length(
        grid, seed=0, count_tokens=approximate_token_count,
        seq_lens={s: 1 for s in grid.seq_lens})

    for (task, seq_len), exs in examples.items():
        for ex in exs:
            assert ex.token_budget == seq_len, (
                f"{task}@{seq_len}: grid length must pass through as the "
                f"token budget, got {ex.token_budget}")
            assert ex.context_length <= seq_len, (
                f"{task}@{seq_len}: context overshot its budget at "
                f"{ex.context_length} tokens")
            assert ex.context_length >= 0.95 * seq_len, (
                f"{task}@{seq_len}: context undershot badly at "
                f"{ex.context_length} tokens -- filler granularity too coarse")


def test_every_configured_length_is_buildable_for_every_task():
    """A budget too small for a task's fixed template raises
    BudgetTooSmallError. The grid's shortest length must clear that floor
    for all three tasks, or a cell would fail mid-run rather than here."""
    from attnbench.accuracy.grid_configs import build_examples_by_task_length
    from attnbench.accuracy.sizing import approximate_token_count

    grid = load_grid(GRID_PATH)
    built = build_examples_by_task_length(
        grid, seed=0, count_tokens=approximate_token_count,
        seq_lens={min(grid.seq_lens): 1})
    assert set(built) == {(t, min(grid.seq_lens)) for t in grid.tasks}
