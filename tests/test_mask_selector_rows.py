"""The `mask_selector` column on every row type (estimator-frontier
pre-registration sec. 4.5, lock gate L3; added 2026-10-03).

Era 4's per-head selector runs beside the head-uniform path, so a new row's
era is read from this column before its commit (`analysis/eras.py`). These
tests check that each row type carries it and that a wrong value is refused
at construction.
"""
from __future__ import annotations

import pandas as pd
import pytest

from attnbench import masks
from attnbench.accuracy import ruler, runner
from attnbench.accuracy.phase_timing import PhaseMeasurement
from attnbench.accuracy.schema import AccuracyResult, Generated
from attnbench.config import AttnConfig


def test_the_selector_for_each_kind_of_row():
    assert masks.mask_selector_for("sdpa_flash", sparse=False) == "none"
    assert masks.mask_selector_for("block_sparse", sparse=True) == "head_uniform"
    assert masks.mask_selector_for("block_sparse", sparse=True, per_head=True) == "per_head"
    assert masks.mask_selector_for("xattention", sparse=True) == "xattn_native"
    assert masks.mask_selector_for("xattn_estimate", sparse=False) == "xattn_native"
    assert set(masks.MASK_SELECTORS) == {"per_head", "head_uniform", "xattn_native", "none"}


def test_the_runner_writes_the_column_on_every_row(tmp_path):
    ex = ruler.RulerExample(task="niah_single", example_id="niah_single_400_0",
                            context="c", question="", answer=["42"], context_length=1)
    dense = AttnConfig(seq_len=400, batch=1, n_heads_q=1, n_heads_kv=1,
                       head_dim=128, mask="causal")
    cells = [runner.AccuracyCell(cfg=dense, backend_name="sdpa_flash",
                                 task="niah_single", example_id=ex.example_id)]
    runner.run_accuracy(cells, out_dir=tmp_path,
                        examples_by_id={("niah_single", ex.example_id): ex},
                        generate_fn=lambda *a: Generated(text="42", latency_ms=1.0),
                        allow_dirty=True)
    df = pd.read_parquet(tmp_path / "accuracy.parquet")
    assert df.mask_selector.tolist() == ["none"]


def _result(**over):
    base = dict(backend="sdpa_flash", backend_role="dense_reference",
                config_key="k", task="niah_single", example_id="e",
                context_length=1, mask_source=None, sparsity=None,
                score_source=None, haystack_mode=None, predicted="42",
                expected="42", score=100.0, correct=True)
    base.update(over)
    return AccuracyResult(**base)


def test_an_unknown_selector_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        _result(mask_selector="per-head")


def test_a_non_selecting_backend_cannot_claim_native():
    with pytest.raises(ValueError, match="xattn_native"):
        _result(mask_selector="xattn_native")
    assert _result(mask_selector="none").mask_selector == "none"


def test_phase_rows_carry_it():
    common = dict(context_length=16384, phase="prefill", ms_mean=1.0,
                  ms_median=1.0, ms_stdev=0.0, n_warmup=1, n_reps=3,
                  clocks_locked=True)
    assert PhaseMeasurement(backend="sdpa_flash", sparsity=None,
                            **common).to_dict()["mask_selector"] == "none"
    assert PhaseMeasurement(backend="block_sparse", sparsity=0.75,
                            **common).to_dict()["mask_selector"] == "head_uniform"
