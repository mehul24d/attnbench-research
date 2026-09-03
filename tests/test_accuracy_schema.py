"""AccuracyResult round-trips through parquet, the mask_source="random"
guard raises (symmetric to sweep.build_cells's guard), build_cells matches
examples to configs by seq_len and gives each backend only its own curated
configs, and checkpoint/resume works with an injected fake generate_fn --
no real model or RULER call needed, mirroring test_sweep_planning.py /
test_sweep_resume.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench import provenance as prov_mod
from attnbench.accuracy.ruler import RulerExample
from attnbench.accuracy.runner import build_cells, run_accuracy
from attnbench.accuracy.schema import AccuracyResult
from attnbench.config import AttnConfig


def _cfg(**overrides) -> AttnConfig:
    base = dict(seq_len=1024, batch=1, n_heads_q=8, n_heads_kv=8,
                head_dim=64, mask="causal", mask_source=None)
    base.update(overrides)
    return AttnConfig(**base)


def _example(task="niah_single", example_id="ex0", context_length=1024) -> RulerExample:
    return RulerExample(task=task, example_id=example_id, context="ctx",
                         question="", answer=["42"], context_length=context_length)


def test_accuracy_result_round_trips_through_parquet(tmp_path):
    result = AccuracyResult(
        backend="sdpa_math", config_key="abc123", task="niah_single",
        example_id="ex0", context_length=1024, mask_source="importance",
        sparsity=0.5, score_source="dense_softmax_fp32",
        haystack_mode="noise", predicted="42", expected="42", score=100.0,
        correct=True, latency_ms=12.3,
    )
    path = tmp_path / "one_row.parquet"
    pd.DataFrame([result.to_dict()]).to_parquet(path)
    loaded = pd.read_parquet(path)

    assert loaded.iloc[0]["score_source"] == "dense_softmax_fp32"
    assert loaded.iloc[0]["haystack_mode"] == "noise"
    assert loaded.iloc[0]["correct"] == True  # noqa: E712


def test_dense_config_has_no_score_source_or_mask_source():
    """A dense (non-block-sparse) row never ran a scoring pass -- both
    fields must be None, not a stale value from a different cell."""
    result = AccuracyResult(
        backend="sdpa_math", config_key="abc123", task="niah_single",
        example_id="ex0", context_length=1024, mask_source=None,
        sparsity=None, score_source=None, haystack_mode="noise",
        predicted="42", expected="42", score=100.0, correct=True,
    )
    assert result.mask_source is None
    assert result.score_source is None


def test_build_cells_rejects_random_mask_source():
    """Symmetric to sweep.build_cells rejecting mask_source="importance"
    for Stage 2 -- accuracy must never use a random mask."""
    bad_cfg = _cfg(mask="block_sparse", sparsity=0.5, block_size=128,
                    mask_source="random")
    with pytest.raises(ValueError):
        build_cells(configs_by_backend={"sdpa_math": [bad_cfg]},
                    examples_by_task_length={("niah_single", 1024): [_example()]})


def test_build_cells_crosses_backend_configs_and_matching_examples():
    cfg = _cfg()
    examples = [_example(example_id="ex0"), _example(example_id="ex1")]
    cells = build_cells(
        configs_by_backend={"a": [cfg], "b": [cfg]},
        examples_by_task_length={("niah_single", 1024): examples})
    assert len(cells) == 2 * 2  # 2 backends x 2 examples (1 config each)


def test_build_cells_only_pairs_examples_at_matching_seq_len():
    """The bug this exists to catch: a config at one seq_len must never be
    crossed with examples generated for a different seq_len. An earlier
    version crossed every config against every example in a task
    regardless of length, inflating cell counts by roughly the number of
    distinct lengths in the grid."""
    cfg_1024 = _cfg(seq_len=1024)
    cfg_2048 = _cfg(seq_len=2048)
    examples_1024 = [_example(example_id="short0", context_length=1024)]
    examples_2048 = [_example(example_id="long0", context_length=2048)]

    cells = build_cells(
        configs_by_backend={"fake": [cfg_1024, cfg_2048]},
        examples_by_task_length={
            ("niah_single", 1024): examples_1024,
            ("niah_single", 2048): examples_2048,
        })

    assert len(cells) == 2  # one cell per (config, its own matching example)
    by_seq_len = {c.cfg.seq_len: c.example_id for c in cells}
    assert by_seq_len[1024] == "short0"
    assert by_seq_len[2048] == "long0"


def test_build_cells_gives_each_backend_only_its_own_configs():
    """Backends have curated, asymmetric config lists (dense baseline vs.
    block_sparse's sparse-only configs vs. GLA/Sage's own single point) --
    a backend must never see another backend's configs."""
    dense_cfg = _cfg(mask="causal")
    sparse_cfg = _cfg(mask="block_sparse", sparsity=0.5, block_size=128,
                      mask_source="importance")
    examples = [_example()]

    cells = build_cells(
        configs_by_backend={"sdpa_math": [dense_cfg], "block_sparse": [sparse_cfg]},
        examples_by_task_length={("niah_single", 1024): examples})

    assert len(cells) == 2
    by_backend = {c.backend_name: c.cfg for c in cells}
    assert by_backend["sdpa_math"].mask == "causal"
    assert by_backend["block_sparse"].mask == "block_sparse"


def test_resume_after_simulated_crash(tmp_path):
    cfg = _cfg()
    examples = [_example(example_id=f"ex{i}") for i in range(5)]
    cells = build_cells(configs_by_backend={"fake": [cfg]},
                        examples_by_task_length={("niah_single", 1024): examples})
    examples_by_id = {("niah_single", ex.example_id): ex for ex in examples}

    calls = {"n": 0}

    def crash_after_two(cfg, backend_name, example):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated process death")
        return example.answer[0], 1.0

    with pytest.raises(RuntimeError):
        run_accuracy(cells, out_dir=tmp_path, examples_by_id=examples_by_id,
                    generate_fn=crash_after_two, checkpoint_every=1)

    partial = pd.read_parquet(tmp_path / "accuracy.parquet")
    assert len(partial) == 2

    calls2 = {"n": 0}

    def count_calls(cfg, backend_name, example):
        calls2["n"] += 1
        return example.answer[0], 1.0

    report = run_accuracy(cells, out_dir=tmp_path, examples_by_id=examples_by_id,
                          generate_fn=count_calls, checkpoint_every=1)

    assert calls2["n"] == 3           # only the 3 not-yet-done cells ran
    assert report.run == 3
    assert report.skip_done == 2

    final = pd.read_parquet(tmp_path / "accuracy.parquet")
    assert len(final) == 5
    assert all(final["correct"])      # generate_fn always returns the right answer


def test_resume_works_across_a_host_change_unlike_stage2():
    """The deliberate difference from sweep.py: accuracy doesn't key on
    host, so a cell already done on one machine is recognized as done when
    resumed on a different one -- no HostMismatchError, no re-measurement."""
    import tempfile
    from dataclasses import replace as dc_replace

    cfg = _cfg()
    examples = [_example(example_id="ex0")]
    cells = build_cells(configs_by_backend={"fake": [cfg]},
                        examples_by_task_length={("niah_single", 1024): examples})
    examples_by_id = {("niah_single", "ex0"): examples[0]}

    def gen(cfg, backend_name, example):
        return example.answer[0], 1.0

    with tempfile.TemporaryDirectory() as out_dir:
        prov_a = lambda: dc_replace(prov_mod.capture(), host="machine-a")
        prov_b = lambda: dc_replace(prov_mod.capture(), host="machine-b")

        run_accuracy(cells, out_dir=out_dir, examples_by_id=examples_by_id,
                    generate_fn=gen, provenance_fn=prov_a)

        calls = {"n": 0}

        def counting_gen(cfg, backend_name, example):
            calls["n"] += 1
            return example.answer[0], 1.0

        report = run_accuracy(cells, out_dir=out_dir, examples_by_id=examples_by_id,
                              generate_fn=counting_gen, provenance_fn=prov_b)
        assert report.skip_done == 1
        assert calls["n"] == 0  # never re-ran on the "different host"


def test_dry_run_reports_without_writing():
    cfg = _cfg()
    examples = [_example()]
    cells = build_cells(configs_by_backend={"fake": [cfg]},
                        examples_by_task_length={("niah_single", 1024): examples})

    def should_not_be_called(cfg, backend_name, example):
        raise AssertionError("generate_fn must not be called in dry_run")

    import tempfile
    with tempfile.TemporaryDirectory() as out_dir:
        report = run_accuracy(cells, out_dir=out_dir, examples_by_id={},
                              generate_fn=should_not_be_called, dry_run=True)
        assert report.total == 1
        assert report.run == 1
        import os
        assert not os.path.exists(os.path.join(out_dir, "accuracy.parquet"))
