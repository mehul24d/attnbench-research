"""The T3.1 projection from Shakedown 6 (`scripts/t31_projection.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t31_projection as P  # noqa: E402
import t31_steps as st  # noqa: E402


def test_the_dense_fit_returns_its_two_anchors():
    assert P.wall_s(2048, 1) * 1e3 == pytest.approx(P.ANCHORS["dense_2048_ms"][1])
    assert P.wall_s(16384, 1) * 1e3 == pytest.approx(P.ANCHORS["dense_16384_ms"][1])
    assert P.wall_s(16384, 4) == pytest.approx(4 * P.wall_s(16384, 1))


def test_the_table_covers_every_step_and_cell_and_its_range_is_ordered():
    t = P.table()
    cfg, cells, _ = P.grid()
    assert t["cells"] == len(cells)
    from t31_session import STEPS
    assert [r[0].split(" ", 1)[1] for r in t["rows"]] == list(STEPS)
    for _, lo, mid, hi in t["rows"] + t["phases"]:
        assert 0 < lo <= mid <= hi
    lo, mid, hi = t["total_min"]
    assert lo <= mid <= hi
    assert t["total_inr"][1] == pytest.approx(mid / 60 * P.RATE_INR_H)
    assert t["halt_h"] == pytest.approx(760 / 108.64)


@pytest.mark.skipif(not P.S6.exists(), reason=f"needs the banked Shakedown 6 results {P.S6}")
def test_the_anchors_are_the_banked_results():
    assert P.check_anchors() == []


def test_a_changed_anchor_is_refused(monkeypatch, tmp_path, capsys):
    if not P.S6.exists():
        pytest.skip(f"needs the banked Shakedown 6 results {P.S6}")
    monkeypatch.setitem(P.ANCHORS, "probe_s", ("x", 100.0))
    assert P.main([]) == 2
    assert "probe_s" in capsys.readouterr().err


def test_json_output(capsys):
    if P.check_anchors():
        pytest.skip("anchors differ")
    assert P.main(["--json"]) == 0
    assert "total_min" in json.loads(capsys.readouterr().out)
