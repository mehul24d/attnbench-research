"""The P6 host-effect rule (scripts/p6_host_effect.py; Appendix A, 21 Nov 2026)."""
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "p6_host_effect", Path(__file__).resolve().parents[1] / "scripts" / "p6_host_effect.py")
he = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(he)


def _write(root: Path, cells: dict) -> Path:
    for cell, ops in cells.items():
        d = root / cell
        d.mkdir(parents=True)
        total = sum(s for s, _ in ops.values())
        d.joinpath("bare_prices.json").write_text(json.dumps({
            "host_total_s": total,
            "operators": {k: {"bare_s": s, "calls": c} for k, (s, c) in ops.items()}}))
    return root


BASE = {f"cell{i}": {"aten::mm": (0.01 * (i + 1), 4), "aten::add": (0.002, 2), "aten::idle": (0.0, 0)}
        for i in range(5)}


def _scaled(f):
    return {c: {k: (s * f, n) for k, (s, n) in ops.items()} for c, ops in BASE.items()}


def test_identical_inputs_give_zero(tmp_path):
    a = he.load(_write(tmp_path / "a", BASE))
    b = he.load(_write(tmp_path / "b", BASE))
    r = he.compare(a, b)
    assert r["cells"] == 5 and r["d"] == 0 and r["p95"] == 0 and r["total_ratio_minus_1"] == 0


def test_uniform_scaling_gives_its_size(tmp_path):
    a = he.load(_write(tmp_path / "a", _scaled(1.2)))
    b = he.load(_write(tmp_path / "b", BASE))
    r = he.compare(a, b)
    assert r["d"] == pytest.approx(0.2) and r["total_ratio_minus_1"] == pytest.approx(0.2)


def test_per_call_price_not_total(tmp_path):
    # the same seconds over twice the calls is half the per-call price
    other = {c: {k: (s, n * 2) for k, (s, n) in ops.items()} for c, ops in BASE.items()}
    r = he.compare(he.load(_write(tmp_path / "a", other)), he.load(_write(tmp_path / "b", BASE)))
    assert r["d"] == pytest.approx(0.5)


def test_only_shared_cells_and_operators(tmp_path):
    extra = dict(_scaled(1.0))
    extra["only_here"] = {"aten::mm": (9.0, 1)}
    extra["cell0"] = dict(extra["cell0"], **{"aten::new": (5.0, 1)})
    r = he.compare(he.load(_write(tmp_path / "a", extra)), he.load(_write(tmp_path / "b", BASE)))
    assert r["cells"] == 5 and r["d"] == 0


def test_no_shared_cell_refuses(tmp_path):
    a = he.load(_write(tmp_path / "a", {"x": {"aten::mm": (1.0, 1)}}))
    b = he.load(_write(tmp_path / "b", {"y": {"aten::mm": (1.0, 1)}}))
    with pytest.raises(ValueError):
        he.compare(a, b)


@pytest.mark.parametrize("d,override,want", [
    (0.1001, False, "run"),
    (0.10, False, "skip (inconclusive band)"),
    (0.05, False, "skip (inconclusive band)"),
    (0.05, True, "run (researcher's written override)"),
    (0.0341, False, "skip (inconclusive band)"),
    (0.034, False, "skip"),
    (0.034, True, "skip"),
    (0.0, False, "skip"),
])
def test_verdict_bands(d, override, want):
    assert he.verdict(d, override) == want


def test_thresholds_are_the_plans():
    assert (he.TAU_RUN, he.TAU_SKIP) == (0.10, 0.034)


def test_cli_prints_the_verdict(tmp_path, capsys):
    a = _write(tmp_path / "a", _scaled(1.5))
    b = _write(tmp_path / "b", BASE)
    assert he.main([str(a), str(b)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "run" and out["d"] == pytest.approx(0.5)


def test_median_over_cells_and_its_p95(tmp_path):
    # 20 cells, cell i scaled by 1 + i/100 except the last, an outlier at 2x:
    # d is the median (0.095), not the mean; p95 is the 19th of 20 sorted (1.0)
    base = {f"c{i:02d}": {"aten::mm": (0.01, 1)} for i in range(20)}
    other = {c: {"aten::mm": (0.01 * (1 + i / 100 if i < 19 else 2.0), 1)} for i, c in enumerate(base)}
    r = he.compare(he.load(_write(tmp_path / "a", other)), he.load(_write(tmp_path / "b", base)))
    assert r["cells"] == 20
    assert r["d"] == pytest.approx(0.095)
    assert r["p95"] == pytest.approx(1.0)


def test_median_over_a_cells_operators(tmp_path):
    # one cell, three operators: two unchanged, one at 1.9x; the cell's value is 0, not 0.3
    base = {"c": {"aten::a": (0.01, 1), "aten::b": (0.01, 1), "aten::c": (0.01, 1)}}
    other = {"c": {"aten::a": (0.01, 1), "aten::b": (0.01, 1), "aten::c": (0.019, 1)}}
    r = he.compare(he.load(_write(tmp_path / "a", other)), he.load(_write(tmp_path / "b", base)))
    assert r["d"] == 0
