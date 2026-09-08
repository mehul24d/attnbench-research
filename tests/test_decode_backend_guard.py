"""DENSE_DECODE_BACKEND changed sdpa_math -> sdpa_flash on 2026-09-08.

The change is deliberate and stamped. These assert that the stamp is
actually consulted -- the confound it fixes survived to Stage 5 precisely
because `decode_backend` was recorded on every row and nothing compared it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from attnbench.accuracy.grid_configs import (
    DENSE_DECODE_BACKEND, DENSE_DECODE_BACKEND_HISTORY)
from attnbench.analysis.decode_backend_guard import (
    MixedDecodeBackend, assert_uniform, era_of, offending_cells)

REPO = Path(__file__).resolve().parent.parent


def _rows(*decode_backends, band=8192):
    return pd.DataFrame([
        dict(backend="block_sparse", task="niah_single", _band=band,
             sparsity=0.75, latency_ms=1000.0 + i, decode_backend=d)
        for i, d in enumerate(decode_backends)])


# --------------------------------------------------------------------------
# The constant itself
# --------------------------------------------------------------------------

def test_the_fallback_is_the_kernel_the_dense_arm_already_uses():
    """The dense arm decodes through sdpa_flash on every row in the study, so
    the alternative was never speculative."""
    assert DENSE_DECODE_BACKEND == "sdpa_flash"


def test_history_records_the_value_it_replaced_and_when():
    values = [v for v, _, _ in DENSE_DECODE_BACKEND_HISTORY]
    assert values == ["sdpa_math", "sdpa_flash"]
    assert DENSE_DECODE_BACKEND_HISTORY[0][2] == "2026-09-08"
    assert DENSE_DECODE_BACKEND_HISTORY[-1][0] == DENSE_DECODE_BACKEND
    assert DENSE_DECODE_BACKEND_HISTORY[-1][2] is None   # current era is open


def test_era_of_places_a_row_by_its_own_stamp():
    assert "project start" in era_of("sdpa_math")
    assert "current" in era_of("sdpa_flash")
    assert "not a recorded" in era_of("flashinfer")


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

def test_a_uniform_cell_passes():
    assert_uniform(_rows("sdpa_math", "sdpa_math"))


def test_a_cell_pooling_both_eras_is_refused():
    """The whole point. A mean over both describes neither, and the gap is
    23-64% per decode token."""
    with pytest.raises(MixedDecodeBackend, match="sdpa_math"):
        assert_uniform(_rows("sdpa_math", "sdpa_flash"))


def test_the_refusal_names_which_cells_are_mixed_not_just_that_some_are():
    bad = offending_cells(_rows("sdpa_math", "sdpa_flash"))
    assert len(bad) == 1
    key, seen = bad[0]
    assert seen == ["sdpa_flash", "sdpa_math"]
    assert "niah_single" in key


def test_cells_are_independent_so_one_mixed_cell_does_not_hide_a_clean_one():
    df = pd.concat([_rows("sdpa_math", "sdpa_math", band=2048),
                    _rows("sdpa_math", "sdpa_flash", band=8192)])
    bad = offending_cells(df)
    assert len(bad) == 1
    assert 8192 in bad[0][0]


def test_rows_predating_the_column_cannot_be_pooled_at_all():
    """Absent is not the same as uniform. A frame with no decode_backend
    column carries rows whose kernel is unknown, and defaulting them into
    either era would invent the fact the guard exists to check."""
    df = _rows("sdpa_math").drop(columns=["decode_backend"])
    with pytest.raises(MixedDecodeBackend, match="no `decode_backend`"):
        assert_uniform(df)


def test_missing_grouping_key_is_a_KeyError_not_a_silent_pass():
    with pytest.raises(KeyError):
        assert_uniform(_rows("sdpa_math").drop(columns=["_band"]))


# --------------------------------------------------------------------------
# The banked rows belong to the OLD era, and must keep saying so.
# --------------------------------------------------------------------------

BANKED = [REPO / "results" / "stage3_s1" / "accuracy.parquet",
          REPO / "results" / "stage3_s1b" / "accuracy.parquet"]


@pytest.mark.parametrize("path", BANKED, ids=lambda p: p.parent.name)
def test_banked_stage3_rows_are_stamped_with_the_pre_change_fallback(path):
    """11,700 rows were produced when the fallback was `sdpa_math`. If a
    future run overwrites them under the new fallback without anyone saying
    so, every latency conclusion drawn from them silently changes regime.

    Skips rather than fails when the file is absent: `results/` is
    gitignored, so a fresh clone has no data and this asserts nothing there.
    """
    if not path.exists():
        pytest.skip(f"{path} not present (results/ is gitignored)")
    df = pd.read_parquet(path)
    sparse = df[df.backend == "block_sparse"]
    assert not sparse.empty
    assert sorted(sparse.decode_backend.unique()) == ["sdpa_math"], (
        "banked block_sparse rows should carry the pre-2026-09-08 fallback. "
        "If these were re-run under sdpa_flash, the analyses comparing them "
        "to the old dense rows are comparing across the change.")
    dense = df[df.backend == "sdpa_flash"]
    assert sorted(dense.decode_backend.unique()) == ["sdpa_flash"], (
        "the dense arm decoded through itself in both eras, which is why "
        "only the sparse arms move")
