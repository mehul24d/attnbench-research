"""scripts/residual_spread_tolerance.py: T4.1 row 10 from T3.2d alone (the
second lock-wave rulings of 12 Nov 2026, item 2)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import residual_spread_tolerance as R  # noqa: E402

S3 = R.SOURCE_SCHEMA


def write(root, cells, schema=S3):
    for cell, spreads in cells.items():
        d = root / cell
        d.mkdir(parents=True)
        (d / "layer_residual.json").write_text(json.dumps(
            {"schema": schema, "layers": [{"layer": i, "spread": s}
                                          for i, s in enumerate(spreads)]}))
    return root


def test_the_working_value_holds_when_every_spread_is_inside_it(tmp_path):
    src = write(tmp_path / "d", {"a": [0.02, 0.10], "b": [0.05]})
    out = R.derive(src)
    assert out["tolerance"] == 0.10          # inclusive: 0.10 is inside
    assert out["source"]["cell"] == "a" and out["source"]["layer"] == 1
    assert out["source"]["n_cells"] == 2


def test_above_it_the_tolerance_is_one_and_a_half_times_the_largest(tmp_path):
    src = write(tmp_path / "d", {"a": [0.02, 0.11], "b": [0.16, 0.03], "c": [0.15]})
    out = R.derive(src)
    assert out["tolerance"] == pytest.approx(0.24)
    assert (out["source"]["cell"], out["source"]["layer"]) == ("b", 0)
    assert out["source"]["spread"] == 0.16


def test_the_sensitivity_is_reported_and_decides_nothing(tmp_path):
    src = write(tmp_path / "d", {"a": [0.05]})
    sens = write(tmp_path / "c", {"a": [0.30]}, schema="attnbench.layer_residual/2")
    out = R.derive(src, sens)
    assert out["tolerance"] == 0.10 and out["sensitivity"]["spread"] == 0.30


def test_the_source_must_be_the_current_method(tmp_path):
    src = write(tmp_path / "d", {"a": [0.05]}, schema="attnbench.layer_residual/2")
    with pytest.raises(SystemExit, match="schema"):
        R.derive(src)


def test_an_empty_source_or_a_cell_without_layers_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no layer_residual"):
        R.derive(tmp_path)
    src = write(tmp_path / "d", {"a": [0.05], "b": []})
    with pytest.raises(SystemExit, match="no layers"):
        R.derive(src)


def test_the_digest_names_the_files_read(tmp_path):
    a = R.largest_spread(write(tmp_path / "x", {"a": [0.05], "b": [0.06]}))
    b = R.largest_spread(write(tmp_path / "y", {"a": [0.05], "b": [0.07]}))
    c = R.largest_spread(write(tmp_path / "z", {"a": [0.05], "c": [0.06]}))
    assert len({a["files_sha256"], b["files_sha256"], c["files_sha256"]}) == 3
