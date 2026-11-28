"""The paper's generated tables and figures (paper skeleton wave, 29 Nov 2026,
item 4): they must be current, reproduce the scored values or refuse, label
every caption, and typeset no repository name or path."""

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_figures", ROOT / "scripts/paper_figures.py")
PF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PF)

SCORED = ROOT / "docs/scoring/hypotheses_scored_20261116.json"
TABLES = ROOT / "docs/paper/exploratory/paper_tables.json"
LOCK = ROOT / "docs/lock_values.json"
PLAN = ROOT / "docs/RESEARCH_PLAN.md"
THREATS = ROOT / "docs/paper/threats_to_validity.md"


def _build(out, scored=SCORED, tables=TABLES):
    return PF.build(scored, tables, LOCK, PLAN, THREATS, out)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    _build(out)
    return out


def test_committed_tables_and_figure_data_are_current(fresh):
    """Every table, macro file and figure data table equals a fresh run. The
    figure PDFs are compared byte for byte by scripts/paper_reproduce.py."""
    files = [f for f in fresh.rglob("*") if f.is_file() and f.suffix in {".tex", ".csv"}]
    assert len(files) == 9
    for f in files:
        rel = f.relative_to(fresh)
        assert (ROOT / "paper" / rel).read_bytes() == f.read_bytes(), rel
    for name in ("fig_pred_vs_meas", "fig_error_by_cell"):
        assert (fresh / "figures" / f"{name}.pdf").stat().st_size > 1000


def test_figure_data_carry_the_scored_errors(fresh):
    """The plotted per-cell errors are the ones behind H1's scored medians."""
    scored = json.loads(SCORED.read_text())["scores"]["H1"]
    rows = list(__import__("csv").DictReader((fresh / "figures/fig_error_by_cell.csv").open()))
    for p in ("P2", "P4"):
        rel = sorted(abs(float(r["signed_rel"])) for r in rows if r["pairing"] == p)
        assert len(rel) == 35
        assert rel[17] == pytest.approx(scored[p]["recursion_median_abs_relative"], abs=1e-6)
    pm = list(__import__("csv").DictReader((fresh / "figures/fig_pred_vs_meas.csv").open()))
    by_cell = {(r["pairing"], r["cell"]): r for r in pm}
    for r in rows:
        q = by_cell[(r["pairing"], r["cell"])]
        pred, meas = float(q["predicted_ms"]), float(q["measured_ms"])
        assert float(r["signed_rel"]) == pytest.approx((pred - meas) / meas, abs=1e-5)
        assert float(r["signed_error_ms"]) == pytest.approx(pred - meas, abs=1e-3)


def test_refuses_when_the_cells_do_not_reproduce_h1(tmp_path):
    t = json.loads(TABLES.read_text())
    bad = copy.deepcopy(t)
    # The median cell's prediction moves up, consistently with its rel, so only
    # the median check can catch it.
    m = sorted(bad["P2"]["miss"], key=lambda r: abs(r["rel"]))[17]
    m["predicted"] = m["measured"] * 1.5
    m["rel"] = 0.5
    path = tmp_path / "t.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(PF.Refused, match="median"):
        _build(tmp_path / "out", tables=path)
    assert not (tmp_path / "out").exists()


def test_refuses_when_a_rival_differs_from_its_score(tmp_path):
    s = json.loads(SCORED.read_text())
    s["scores"]["H1"]["P4"]["rivals_median_abs_relative"]["step_form"] += 0.01
    path = tmp_path / "s.json"
    path.write_text(json.dumps(s))
    with pytest.raises(PF.Refused, match="step_form"):
        _build(tmp_path / "out", scored=path)


def test_refuses_a_verdict_that_disagrees_with_its_median(tmp_path):
    s = json.loads(SCORED.read_text())
    s["scores"]["H1"]["P2"]["verdict"] = "holds"
    path = tmp_path / "s.json"
    path.write_text(json.dumps(s))
    with pytest.raises(PF.Refused, match="verdict"):
        _build(tmp_path / "out", scored=path)


def test_the_error_figure_leads_with_h1s_relative_error(fresh):
    """The top (larger) panel plots H1's pre-registered relative error; the ms
    panel is labelled a signed error, never an absolute one."""
    head = (fresh / "figures/fig_error_by_cell.csv").read_text().splitlines()[0]
    assert head == "pairing,cell,arm,signed_rel,rel_lo,rel_hi,signed_error_ms"
    cap = (fresh / "generated/fig_error_by_cell.tex").read_text()
    assert "Top: H1's pre-registered relative error" in cap
    assert "signed error (ms)" in cap and "absolute error" not in cap.replace(
        "median of its absolute value", "")
    src = (ROOT / "scripts/paper_figures.py").read_text()
    assert 'bot.set_ylabel("signed error (ms)")' in src
    assert 'top.set_ylabel("signed relative error")' in src


def test_every_caption_states_its_type(fresh):
    for f in sorted((fresh / "generated").glob("*.tex")):
        if f.name == "numbers.tex":
            continue
        cap = re.search(r"\\caption\{(.*)\}\n\\label", f.read_text(), re.S).group(1)
        assert re.search(r"pre-registered|exploratory", cap, re.I), f.name
        if f.name == "tab_miss.tex":
            assert cap.startswith(r"\textbf{Exploratory / post hoc.}")
        if f.name in ("tab_pairings.tex", "fig_pred_vs_meas.tex"):
            assert cap.startswith(r"\textbf{Pre-registered.}")


def test_no_repository_name_or_path_is_typeset(fresh):
    """Traceable names live only in numbers.tex macros, blanked in one place."""
    for f in sorted((fresh / "generated").glob("*.tex")):
        body = "\n".join(l for l in f.read_text().splitlines() if not l.startswith("%"))
        if f.name == "numbers.tex":
            continue
        for bad in ("docs/", "scripts/", "attnbench", "lock-20", "results/"):
            assert bad not in body, (f.name, bad)


def test_the_skeleton_checks_pass_and_a_final_build_refuses_drafting_notes():
    r = subprocess.run([sys.executable, str(ROOT / "paper/check_skeleton.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "paper/check_skeleton.py"), "--final"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "\\fact drafting note left" in r.stderr
