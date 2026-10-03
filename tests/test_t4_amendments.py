"""T4 amendments A1-A7 (docs/t4_xattention_calibrated.md, "Amendments,
2026-10-03"; estimator-frontier pre-registration sec. 12, lock gate L2).

Each amendment that changes code is checked here on CPU. A1's own break-test
(a shared qa_1 question under another seed) is in `test_xattn_calibration.py`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench.accuracy import ruler, sizing, t4_pilot
from attnbench.accuracy.generation import _positions_over_limit
from attnbench.accuracy.schema import AccuracyResult
from attnbench.backends.xattention import triton_for_device

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "cal", ROOT / "scripts" / "calibrate_xattn_thresholds.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# A1 -----------------------------------------------------------------------

def test_a1_the_offset_moves_ids_and_seeds_and_keeps_offset_0_identical():
    kw = dict(token_budgets=[3000], n_per_length=3, seed=1,
              count_tokens=sizing.approximate_token_count)
    base = ruler.generate_examples("niah_single", **kw)
    assert base == ruler.generate_examples("niah_single", index_offset=0, **kw)
    off = ruler.generate_examples("niah_single", index_offset=2000, **kw)
    assert [e.example_id for e in off] == [f"niah_single_3000_{i}" for i in (2000, 2001, 2002)]
    assert {e.context for e in off}.isdisjoint({e.context for e in base})


def test_a1_registered_offset():
    assert t4_pilot.XATTN_CALIBRATION_INDEX_OFFSET == 2000
    assert t4_pilot.XATTN_CALIBRATION_INDEX_OFFSET > max(t4_pilot.SPARSE_PILOT_N.values())


# A2 -----------------------------------------------------------------------

def test_a2_over_length_texts_are_excluded_not_truncated():
    cal = _script()
    assert cal.excluded_over_length([100, 32768, 32769, 91574], 32768) == [2, 3]
    assert cal.excluded_over_length([10, 20], 32768) == []


# A3 -----------------------------------------------------------------------

def test_a3_the_table_is_the_max_and_the_records_are_descriptive():
    cal = _script()
    per_text = {0: torch.tensor([[0.5, 0.9]]), 1: torch.tensor([[0.6, 0.1]]),
                2: torch.tensor([[0.55, 0.2]])}
    rec = cal.descriptive_records(per_text, [0, 1, 2])
    assert rec["argmax_text"] == [[1, 0]]
    assert rec["n_max_minus_p90_above"] == 1           # 0.9 vs p90 0.76
    assert rec["gap"] == t4_pilot.XATTN_CALIBRATION_DESCRIPTIVE_GAP
    assert t4_pilot.XATTN_CALIBRATION_STATISTIC == "max over used texts, no cap"
    src = (ROOT / "scripts" / "calibrate_xattn_thresholds.py").read_text()
    assert "0.96" not in src and "clamp" not in src       # no cap anywhere


# A4 -----------------------------------------------------------------------

@pytest.mark.parametrize("name,ok", [("NVIDIA A100-SXM4-80GB", True),
                                     ("NVIDIA L4", False), ("NVIDIA H100 80GB HBM3", False)])
def test_a4_calibration_card(name, ok):
    """The script refuses unless the device is the A100 and the official
    estimator would take Triton there."""
    assert (t4_pilot.XATTN_CALIBRATION_CARD in name and triton_for_device(name)) is ok


def _row(**over):
    base = dict(backend="xattention", backend_role="block_sparse", config_key="k",
                task="qa_1", example_id="e", context_length=1, mask_source=None,
                sparsity=None, score_source="xattention_inline", haystack_mode=None,
                predicted="a", expected="a", score=100.0, correct=True,
                xattn_calibration="authors:abc", realised_density=0.3)
    base.update(over)
    return AccuracyResult(**base)


def test_a4_rows_carry_their_path():
    assert _row(xattn_path="torch_fallback").xattn_path == "torch_fallback"
    assert _row(xattn_path="triton").xattn_path == "triton"
    with pytest.raises(ValueError, match="xattn_path"):
        _row(xattn_path="cuda")
    with pytest.raises(ValueError, match="xattn_path"):
        AccuracyResult(backend="sdpa_flash", backend_role="dense_reference",
                       config_key="k", task="qa_1", example_id="e", context_length=1,
                       mask_source=None, sparsity=None, score_source=None,
                       haystack_mode=None, predicted="a", expected="a", score=100.0,
                       correct=True, xattn_path="triton")


# A7 -----------------------------------------------------------------------

def test_a7_positions_over_limit():
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=8,
                                         num_hidden_layers=1, num_attention_heads=1,
                                         num_key_value_heads=1, max_position_embeddings=32768))
    w = type("W", (), {"model": model})()
    assert _positions_over_limit(w, 32726, 42) == 0
    assert _positions_over_limit(w, 32769, 1) == 2
    assert _positions_over_limit(w, 32700, 100) == 32
    assert _positions_over_limit(type("W", (), {})(), 10, 1) is None
