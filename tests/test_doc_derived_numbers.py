"""Numbers the prose restates that the repository can recompute.

**The companion to `test_no_stale_figures.py`, and a different defect.** That
file guards *withdrawn measurements* — a figure a session produced, later
superseded, still stated as live. This one guards *restated counts*: a number
that was read correctly once, written into prose, and then drifted because the
thing it counts kept growing.

`docs/spend_ledger.md` opens with exactly this incident:

    ...that total was accumulated **in prose**, carried forward from one report
    to the next -- and it drifted: Rs 902, 910, 918 and 942 were all in
    circulation for the same point in the project.

The ledger fixed it for itself, by reading `session_cost.txt` off the
instance. The README then restated the ledger's total in prose and drifted
the same way: at the 2026-09-21 audit it said **Rs 7,879 across 30 priced
sessions** in a sentence whose own text warns about this failure, while the
table summed to Rs 8,279 across 34. The four sessions added on 2026-09-19/20
total exactly Rs 400, which is the entire discrepancy.

A registry entry would be the wrong fix, because the right value changes every
session. The fix is to recompute.

**These are deliberately loose about formatting and strict about the number.**
A test that breaks when someone rewords a sentence gets deleted; a test that
passes when the number is wrong is decoration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
LEDGER = REPO / "docs" / "spend_ledger.md"
LIMITATIONS = REPO / "docs" / "limitations.md"
PATTERNS = REPO / "docs" / "silent_failure_patterns.md"


def test_the_documents_exist():
    """Anti-vacuity: every check below reads one of these."""
    for p in (README, LEDGER, LIMITATIONS, PATTERNS):
        assert p.exists(), p


# --- the spend ledger -------------------------------------------------------

def ledger_rows() -> list[tuple[str, int | None]]:
    """(date, est_inr) for every dated row of the ledger's session table.

    `None` where the row carries no figure -- the 2026-09-02 validation
    session, which the ledger records as "prose only, not reconstructable"
    and the README counts separately.
    """
    rows: list[tuple[str, int | None]] = []
    for line in LEDGER.read_text().splitlines():
        if not line.startswith("| 2026-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        m = re.search(r"([\d,]+)", cells[3].replace("*", "").replace("~", ""))
        rows.append((cells[0], int(m.group(1).replace(",", "")) if m else None))
    return rows


def test_the_ledger_table_is_parseable():
    rows = ledger_rows()
    assert len(rows) > 25, f"only {len(rows)} ledger rows parsed"
    assert any(v is None for _, v in rows), (
        "expected at least one row with no figure (the prose-only validation "
        "session); the parser is probably reading the wrong column")


def test_the_readme_spend_total_matches_the_ledger():
    rows = ledger_rows()
    total = sum(v for _, v in rows if v is not None)
    priced = sum(1 for _, v in rows if v is not None)

    text = README.read_text()
    m = re.search(r"Total rented GPU time:\s*\*\*₹([\d,]+) itemised\*\*", text)
    assert m, "README no longer states a total in the expected form"
    stated = int(m.group(1).replace(",", ""))
    assert stated == total, (
        f"README says ₹{stated:,}; docs/spend_ledger.md sums to ₹{total:,}. "
        f"The README's own sentence says the figure is 'summed from the "
        f"ledger's table, not restated here' -- restate it from the table.")

    m = re.search(r"across (\d+) priced sessions", text)
    assert m, "README no longer states a session count in the expected form"
    assert int(m.group(1)) == priced, (
        f"README says {m.group(1)} priced sessions; the ledger has {priced}.")


def test_the_day_total_for_20260917_matches_its_own_rows():
    """The ledger's own arithmetic, which drifted by ₹1 at the audit: the
    'four sessions' total excluded the boot-test the same sentence includes."""
    day = [v for d, v in ledger_rows() if d == "2026-09-17" and v is not None]
    assert day, "no 2026-09-17 rows parsed"
    m = re.search(r"Day total: ₹([\d,]+) across four sessions", LEDGER.read_text())
    assert m, "the 2026-09-17 day total is no longer stated in that form"
    assert int(m.group(1).replace(",", "")) == sum(day), (
        f"day total says ₹{m.group(1)}; the four 2026-09-17 rows sum to "
        f"₹{sum(day):,} ({day}).")


# --- counts over files ------------------------------------------------------

def test_the_readme_line_count_for_limitations_is_not_stale():
    """Stated as 'over N lines'. The claim is a floor, so it only has to be
    true -- but it went from honest to absurd as the file grew past 2x N."""
    n = len(LIMITATIONS.read_text().splitlines())
    m = re.search(r"limitations\.md.{0,40}?is over (\d[\d,]*) lines",
                  README.read_text(), re.S)
    assert m, "README no longer states a line count for limitations.md"
    stated = int(m.group(1).replace(",", ""))
    assert stated <= n, f"README claims over {stated} lines; the file has {n}"
    assert n < stated * 1.5, (
        f"README says 'over {stated} lines' for a file of {n}. True, and "
        f"uninformative by a factor of {n / stated:.1f}.")


def numbered_instances() -> list[int]:
    """Both heading levels.

    Instances 1-17 are `### N.` and 18-48 are `## N.`. Scanning one level
    finds 31 of 48 and looks like a complete list, which is how the first
    draft of this test reported a numbering gap starting at 18 -- and, very
    likely, how the README's count of 46 was arrived at in the first place.
    """
    return sorted(int(m.group(1)) for m in
                  re.finditer(r"^#{2,3} (\d+)\.", PATTERNS.read_text(), re.M))


def test_the_instance_count_agrees_in_three_places():
    """`silent_failure_patterns.md` states its own count in prose, numbers its
    sections, and is counted again by the README. All three moved apart: the
    file said forty-eight, the headings ran to 48, the README said 46."""
    numbers = numbered_instances()
    assert numbers, "no numbered instance headings found"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"instance numbering has a gap or duplicate: {numbers}")
    highest = numbers[-1]

    m = re.search(r"(\d+) confirmed incidents", README.read_text())
    assert m, "README no longer states an incident count"
    assert int(m.group(1)) == highest, (
        f"README says {m.group(1)} confirmed incidents; "
        f"silent_failure_patterns.md numbers {highest}.")


def test_the_readme_suite_counts_are_internally_consistent():
    """Not a check that the numbers are right -- a check that there is only
    ONE of them.

    At the 2026-09-21 audit the README gave three figures for its own skip
    count: `10 skipped` in the quick-start block, `The 11 skips` in the
    paragraph below it, and `2 need CUDA, and 9 read banked result files` as
    the breakdown. The real number was 23. No test can run the suite from
    inside the suite, but a document that disagrees with itself about a count
    can be caught without running anything, and that is the version of this
    failure that shipped.
    """
    text = README.read_text()

    block = re.search(r"pytest tests/ -q\s*#\s*(\d+) passed, (\d+) skipped", text)
    assert block, "the quick-start block no longer reports passed/skipped"
    _, block_skips = int(block.group(1)), int(block.group(2))

    prose = re.search(r"The (\d+) skips are the honest part", text)
    assert prose, "the skips paragraph no longer states a total"
    assert int(prose.group(1)) == block_skips, (
        f"the quick-start block says {block_skips} skipped and the paragraph "
        f"below it says {prose.group(1)}. One quantity, two numbers.")

    # Every bolded number in the skips paragraph, rather than a fixed list of
    # clauses. The paragraph went from a three-way split to a four-way one on
    # 2026-09-21 when test_import_resolution.py added a skip category, and a
    # test pinned to "expected a three-way breakdown" fails on a document
    # that is perfectly consistent -- which trains people to edit the test
    # until it passes. Count the parts; do not assume how many there are.
    para = re.search(r"The \d+ skips are the honest part.*?(?=\n\n)", text, re.S)
    assert para, "the skips paragraph is no longer in the expected form"
    parts = [int(m) for m in re.findall(r"\*\*(\d+)\*\*", para.group(0))]
    assert len(parts) >= 3, f"expected a breakdown of at least three parts, parsed {parts}"
    assert sum(parts) == block_skips, (
        f"the breakdown {parts} sums to {sum(parts)}, not {block_skips}.")


WORDS = {40: "forty", 41: "forty-one", 42: "forty-two", 43: "forty-three",
         44: "forty-four", 45: "forty-five", 46: "forty-six",
         47: "forty-seven", 48: "forty-eight", 49: "forty-nine", 50: "fifty"}


def test_the_patterns_file_states_its_own_count_correctly():
    highest = numbered_instances()[-1]
    if highest not in WORDS:
        pytest.skip(f"no spelled form for {highest}; extend WORDS")
    text = PATTERNS.read_text()
    m = re.search(r"It stands at\s*\n?\*\*([a-z-]+)\*\*", text)
    assert m, "the 'It stands at N' sentence is no longer in the expected form"
    assert m.group(1) == WORDS[highest], (
        f"the file says it stands at {m.group(1)}; it numbers "
        f"{highest} ({WORDS[highest]}).")
