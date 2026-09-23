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
import subprocess
import sys
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


def _collected_test_count() -> int:
    """How many tests pytest collects from `tests/`, by asking pytest.

    A subprocess because a test cannot count the suite it is part of: reading
    `request.session.items` gives whatever this invocation selected, which is
    the whole suite in CI and one file when someone is iterating -- a guard
    that silently measures the selection instead of the suite is the shape this
    file exists to catch. `--collect-only` runs nothing, so there is no
    recursion. ~5s, which is why it backs one test rather than several.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:randomly", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if m is None:
        pytest.skip(f"could not read a collected count from pytest:\n"
                    f"{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return int(m.group(1))


def test_the_readme_suite_counts_match_the_collected_suite():
    """The check that was missing, and the reason F3 survived a pass.

    `test_the_readme_suite_counts_are_internally_consistent` pins the SKIP
    count three ways and the PASSED count nowhere, so when two commits added
    three tests both README totals went stale together -- 1109+35 and 1133+11
    both summed to 1144 while the suite collected 1147, and internal
    consistency held the whole time. A document can only be wrong about its own
    size in one way that matters: disagreeing with the suite.
    """
    total = _collected_test_count()
    text = README.read_text()

    block = re.search(r"pytest tests/ -q\s*#\s*(\d+) passed, (\d+) skipped", text)
    assert block, "the quick-start block no longer reports passed/skipped"
    # The `**` opens on the line above ("**With `results/` present the suite
    # reads"), so anchor on the closing pair only.
    prose = re.search(r"(\d+) passed, (\d+) skipped\*\*", text)
    assert prose, "the with-results paragraph no longer reports passed/skipped"

    for label, m in (("the quick-start block (fresh clone)", block),
                     ("the with-results paragraph", prose)):
        passed, skipped = int(m.group(1)), int(m.group(2))
        assert passed + skipped == total, (
            f"{label} says {passed} passed + {skipped} skipped = "
            f"{passed + skipped}, but pytest collects {total} tests from "
            f"tests/. Every collected test either passes, skips or fails, and "
            f"the README documents a green run, so these must be equal. "
            f"Difference: {total - passed - skipped:+d}.")


_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")


def spelled(n: int) -> str:
    """`52` -> "fifty-two".

    This was a hand-maintained dict from 40 to the current instance count, and
    the test below SKIPPED when the count outran it -- so the guard that keeps
    the file's prose count honest switched itself off on precisely the event it
    exists to catch: someone adding an instance. Two of the last three audit
    passes added one. Derived now, because a lookup table that has to be
    extended by the same commit that breaks it is not a guard.
    """
    if not 0 <= n < 100:
        raise ValueError(f"no spelling for {n}; this file counts instances")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def test_the_spelling_helper_is_right_where_it_used_to_be_a_table():
    """The dict it replaced ran 40..52; those are the values in live use, so
    they are the ones worth pinning, plus the boundaries around them."""
    assert spelled(40) == "forty"
    assert spelled(48) == "forty-eight"
    assert spelled(50) == "fifty"
    assert spelled(52) == "fifty-two"
    assert spelled(53) == "fifty-three"      # the one the dict lacked
    assert spelled(60) == "sixty"
    assert spelled(99) == "ninety-nine"
    assert spelled(19) == "nineteen"
    with pytest.raises(ValueError):
        spelled(100)


def test_the_patterns_file_states_its_own_count_correctly():
    """No skip branch. An unspellable count raises from `spelled()` rather than
    quietly standing down."""
    highest = numbered_instances()[-1]
    text = PATTERNS.read_text()
    m = re.search(r"It stands at\s*\n?\*\*([a-z-]+)\*\*", text)
    assert m, "the 'It stands at N' sentence is no longer in the expected form"
    assert m.group(1) == spelled(highest), (
        f"the file says it stands at {m.group(1)}; it numbers "
        f"{highest} ({spelled(highest)}).")
