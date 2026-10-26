"""Session B's positions past the limit (A7), derived from the banked rows.

docs/t4_xattention_calibrated.md, "Positions past the limit (A7), re-derived
26 Oct 2026", states per file and band how many rows decode past Qwen2.5's
32,768 positions. The four files are force-committed, so this runs in a
clone. The expected table is read from the doc, never from the code under
test, and recomputed here from the parquets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "t11b_session_20261025" / "results"
DOC = REPO / "docs" / "t4_xattention_calibrated.md"
TABLES = ("dense", "authors", "authors_full", "ruler_heldout")
FILES = {t: RESULTS / f"t4_xattn_cal_{t}_20261025" / "accuracy.parquet" for t in TABLES}
LIMIT = 32768
CAPS = {"qa_1": 42, "niah_multivalue": 62, "niah_multiquery": 62}

needs_banked = pytest.mark.skipif(
    not all(p.exists() for p in FILES.values()),
    reason="Session B files not in this checkout")


def _rows(table):
    pd = pytest.importorskip("pandas")
    d = pd.read_parquet(FILES[table])
    d["band"] = d.example_id.str.rsplit("_", n=2).str[-2].astype(int)
    return d


def _doc_table() -> dict:
    """{file: (16384 over, qa_1 32768, niah_mv 32768, niah_mq 32768, total)},
    each 32768 task cell as (over, n, max overshoot)."""
    section = DOC.read_text().split("re-derived 26 Oct 2026")[1]
    out = {}
    for line in section.splitlines():
        m = re.match(r"\| `?(\w+)`? \| (.+) \|$", line)
        if not m or m.group(1) not in TABLES:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        task = [tuple(int(x) for x in re.findall(r"\d+", c)) for c in cells[1:4]]
        out[m.group(1)] = (int(cells[0].split()[0]), *task,
                           int(cells[4].split()[0]))
    return out


@needs_banked
def test_the_doc_table_is_derived_from_the_banked_rows():
    want = _doc_table()
    assert set(want) == set(TABLES)
    for table in TABLES:
        d = _rows(table)
        assert len(d) == 400 and set(d.git_dirty) == {False}
        over = d.positions_over_limit > 0
        got = [int((over & (d.band == 16384)).sum())]
        for task in ("qa_1", "niah_multivalue", "niah_multiquery"):
            x = d[(d.task == task) & (d.band == 32768)]
            got.append((int((x.positions_over_limit > 0).sum()), len(x),
                        int(x.positions_over_limit.max())))
        got.append(int((over & (d.band == 32768)).sum()))
        assert tuple(got) == want[table], table


@needs_banked
def test_the_field_is_the_registered_definition_and_decoding_reaches_the_cap():
    for table in TABLES:
        d = _rows(table)
        assert ((d.context_length + d.n_generated - LIMIT).clip(lower=0)
                == d.positions_over_limit).all()
        for (task, band), x in d.groupby(["task", "band"]):
            assert int(x.n_generated.max()) == CAPS[task], (table, task, band)
            assert int(x.n_generated.max()) <= CAPS[task]
