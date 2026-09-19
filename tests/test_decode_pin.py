"""Audit S1a: the dense-decode fallback can be pinned to a HISTORICAL value,
the pin is stamped on every row, and pinned/unpinned rows never share an
output."""
import pandas as pd
import pytest

from attnbench.accuracy.grid_configs import (DENSE_DECODE_BACKEND,
                                             backend_instance, decode_backend_for)
from attnbench.accuracy.runner import DecodePinMismatch, check_decode_pin_continuity


def test_default_fallback_is_current_constant():
    bs = backend_instance("block_sparse")
    assert decode_backend_for(bs).name == DENSE_DECODE_BACKEND


def test_pinned_fallback_replays_sdpa_math():
    bs = backend_instance("block_sparse")
    assert decode_backend_for(bs, fallback="sdpa_math").name == "sdpa_math"


def test_pin_never_overrides_a_backend_with_its_own_decode_path():
    dense = backend_instance("sdpa_flash")
    assert decode_backend_for(dense, fallback="sdpa_math").name == "sdpa_flash"


def test_a_value_never_used_is_refused():
    with pytest.raises(ValueError, match="never"):
        decode_backend_for(backend_instance("block_sparse"), fallback="sdpa_efficient")


def _ckpt(tmp_path, pinned):
    p = tmp_path / "accuracy.parquet"
    d = pd.DataFrame({"backend": ["block_sparse"], "example_id": ["x"]})
    if pinned is not None:
        d["decode_pinned"] = [pinned]
    d.to_parquet(p)
    return p


def test_fresh_output_accepts_either(tmp_path):
    check_decode_pin_continuity(tmp_path / "none.parquet", pinned=True)
    check_decode_pin_continuity(tmp_path / "none.parquet", pinned=False)


def test_same_regime_resumes(tmp_path):
    check_decode_pin_continuity(_ckpt(tmp_path, True), pinned=True)


def test_pinned_into_unpinned_refused(tmp_path):
    with pytest.raises(DecodePinMismatch):
        check_decode_pin_continuity(_ckpt(tmp_path, False), pinned=True)


def test_pre_column_rows_count_as_unpinned(tmp_path):
    p = _ckpt(tmp_path, None)
    check_decode_pin_continuity(p, pinned=False)
    with pytest.raises(DecodePinMismatch):
        check_decode_pin_continuity(p, pinned=True)

