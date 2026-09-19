"""The reconciliation refuses observed totals from another decode regime.

Driven with the REAL banked inputs that produced the stage5_flashdecode bias:
original Stage 3 rows (sparse arms decoded through sdpa_math) against a phase
model whose sparse arms decode through flash. The check must refuse that, and
must accept stage3_flashdecode, the matched-regime set that closes 12 of 12.
"""
from pathlib import Path

import pandas as pd
import pytest

from attnbench.accuracy.phase_timing import check_observed_decode_regime

ROOT = Path(__file__).resolve().parent.parent
FLASH = {"block_sparse": "sdpa_flash", "sdpa_flash": "sdpa_flash"}


def _frame(**by_backend):
    return pd.DataFrame([{"backend": b, "decode_backend": d}
                         for b, d in by_backend.items()])


def test_mismatched_regime_is_refused():
    with pytest.raises(SystemExit, match="block_sparse: observed decoded through sdpa_math"):
        check_observed_decode_regime(
            _frame(block_sparse="sdpa_math", sdpa_flash="sdpa_flash"), FLASH)


def test_matched_regime_passes():
    check_observed_decode_regime(
        _frame(block_sparse="sdpa_flash", sdpa_flash="sdpa_flash"), FLASH)


def test_missing_column_is_refused_not_assumed():
    with pytest.raises(SystemExit, match="no decode_backend column"):
        check_observed_decode_regime(pd.DataFrame({"backend": ["block_sparse"]}), FLASH)


@pytest.mark.skipif(not (ROOT / "results/stage3_s1b/accuracy.parquet").exists(),
                    reason="banked results not present")
def test_the_real_inputs_that_caused_the_bias_are_refused():
    df = pd.read_parquet(ROOT / "results/stage3_s1b/accuracy.parquet",
                         columns=["backend", "decode_backend"])
    with pytest.raises(SystemExit, match="sdpa_math"):
        check_observed_decode_regime(df, FLASH)


@pytest.mark.skipif(not (ROOT / "results/stage3_flashdecode/accuracy.parquet").exists(),
                    reason="banked results not present")
def test_the_real_matched_inputs_are_accepted():
    df = pd.read_parquet(ROOT / "results/stage3_flashdecode/accuracy.parquet",
                         columns=["backend", "decode_backend"])
    check_observed_decode_regime(df, FLASH)
