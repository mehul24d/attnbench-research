"""`scripts/hdbi_threshold.py`: T4.1's HDBI threshold at the lock (the
researcher's item 18, lock-wave rulings of 12 Nov 2026). Each sparse cell is
read against its own dense partner and its own cell's floor."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hdbi_threshold as H  # noqa: E402
from attnbench.lock.floors import FloorTable  # noqa: E402
from attnbench.lock.hdbi_fit import fit_threshold  # noqa: E402
from attnbench.predict.training import CellKey  # noqa: E402


def row(arm, L, measured, device=1.0, host=1.0, batch=1, density=0.5):
    key = CellKey("primary", f"cp-{arm}", arm, density, L, batch)
    cell = SimpleNamespace(arm=arm, key=key, pairing="P1", context_length=L, batch=batch,
                           priced=SimpleNamespace(device=[device], host=[host]))
    return SimpleNamespace(cell=cell, measured=measured)


def test_each_sparse_cell_is_read_against_its_own_dense_partner():
    rows = [row("dense", 1951, 1.00, density=1.0), row("dense", 3900, 3.00, density=1.0),
            row("oracle", 1951, 0.90, device=3, host=1), row("oracle", 3900, 3.20, device=1, host=3),
            row("dense", 3900, 9.00, batch=4, density=1.0), row("oracle", 3900, 8.00, batch=4)]
    out = {(r.cell[4], r.cell[5]): r for r in H.hdbi_rows(rows)}
    assert set(out) == {(1951, 1), (3900, 1), (3900, 4)}
    assert out[(1951, 1)].measured_difference == pytest.approx(-0.10)
    assert out[(3900, 1)].measured_difference == pytest.approx(0.20)
    assert out[(3900, 4)].measured_difference == pytest.approx(-1.00)     # batch 4's dense
    assert out[(1951, 1)].hdbi == 0.75 and out[(3900, 1)].hdbi == 0.25
    assert (out[(3900, 4)].context_length, out[(3900, 4)].batch) == (3900, 4)
    # each row reads its own cell's floor: 0.15 s at 2048 drops the first row only
    floors = FloorTable({(2048, 1): 0.15, (4096, 1): 0.05, (4096, 4): 0.5})
    f = fit_threshold(out.values(), training_pairings=["P1"], floors=floors)
    assert (f.n_scored, f.n_inside_floor) == (2, 1)


def test_a_point_without_exactly_one_dense_cell_is_refused():
    with pytest.raises(SystemExit, match="no dense cell"):
        H.hdbi_rows([row("oracle", 1951, 0.9)])
    with pytest.raises(SystemExit, match="two dense cells"):
        H.hdbi_rows([row("dense", 1951, 1.0, density=1.0), row("dense", 1951, 1.1, density=1.0)])


def test_the_fit_reads_t32d_s_own_targets():
    assert H.FIT_SET == "repeat_a_only" and H.FIT_REPEAT == "T3.2d"
    with pytest.raises(SystemExit, match="repeat a is 'T3.2c'"):
        H.fit({"features_from": "T3.2c"}, {})
