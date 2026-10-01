"""docs/t4_sparse_pilot.md says what attnbench/accuracy/t4_pilot.py does.

The plan is a pre-registration: if the document and the code that runs and
analyses the pilot ever disagree, one of them is not what was registered.
Every number the document states about the design is checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

from attnbench.accuracy import t4_pilot
from attnbench.analysis.exact_noninferiority import paired_lower_bound

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "t4_sparse_pilot.md").read_text()


def _table_after(heading_text: str) -> list[list[str]]:
    block = DOC[DOC.index(heading_text):]
    rows = []
    for line in block.splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
        elif rows:
            break
    return rows[2:]          # header and rule


def test_tier_table_is_the_module():
    rows = _table_after("| task | band | dense pilot")
    got = {(r[0].strip("`"), int(r[1])): (r[3], int(r[4])) for r in rows}
    want = {key: (tier, t4_pilot.SPARSE_PILOT_N[key[0]])
            for key, tier in t4_pilot.TIERS.items()}
    assert got == want
    assert {t for t, _ in got} == set(t4_pilot.SPARSE_PILOT_TASKS)
    assert {b for _, b in got} == set(t4_pilot.PILOT_BANDS)


def test_dense_pilot_column_quotes_the_dense_pilot_doc():
    dense_doc = (ROOT / "docs" / "t4_dense_pilot.md").read_text()
    for r in _table_after("| task | band | dense pilot"):
        frac = r[2].split(" =")[0]
        assert frac in dense_doc, r


def test_margin_alpha_sparsities_and_arms():
    assert f"Margin of {t4_pilot.MARGIN_PTS:g} points; one-sided alpha {t4_pilot.ALPHA}" in DOC
    assert "sparsities 0.5, 0.75 and 0.9" in DOC
    assert "test 0.5, then 0.75,\n  then 0.9" in DOC
    assert t4_pilot.SPARSITY_SEQUENCE == (0.5, 0.75, 0.9)
    for src in t4_pilot.SPARSE_PILOT_SCORE_SOURCES:
        assert f"`{src}`" in DOC


def test_attainable_bound_table_is_computed_not_typed():
    for r in _table_after("| n | b, c | lower bound"):
        n = int(r[0])
        b, c = (int(x) for x in r[1].split(","))
        quoted = float(r[2].replace("−", "-"))
        assert round(100 * paired_lower_bound(b, c, n, t4_pilot.ALPHA), 1) == quoted, r
    assert "(the bound is then\n−12.1 points)" in DOC


def test_cell_counts():
    per_band = sum(t4_pilot.SPARSE_PILOT_N[t] for t in t4_pilot.SPARSE_PILOT_TASKS)
    examples = per_band * len(t4_pilot.PILOT_BANDS)
    m = re.search(r"The oracle run is (\d+) cells: (\d+) dense and (\d+) sparse", DOC)
    assert m and tuple(map(int, m.groups())) == (
        examples * 4, examples, examples * len(t4_pilot.SPARSITY_SEQUENCE))
    assert f"The inline run is {examples * 3} cells" in DOC


def test_named_tests_and_scripts_exist():
    for path in re.findall(r"`((?:tests|scripts|attnbench)/[\w/.]+\.py)`", DOC):
        assert (ROOT / path).exists(), path
