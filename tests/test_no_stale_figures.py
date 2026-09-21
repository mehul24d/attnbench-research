"""A withdrawn figure must not appear anywhere as a live statement.

**The incident class.** Every serious finding in the 2026-09-21 audit was one
defect wearing five faces: a number was withdrawn or superseded, the
withdrawal was applied in `claims.md`, and nothing propagated it to the other
places the number lived -- the End-to-end section of the same file, the
conclusion paragraph, `limitations.md`, a code comment in
`accuracy/grid_configs.py`, a status block. Each correction was right where it
was made. None of them reached the next copy.

That is the guard-versus-check distinction applied to prose. A withdrawn
figure sitting in the write-up is currently found only by an audit, after it
has been in the document for a day or a week. This file makes it a test
failure instead.

**The rule.** An occurrence of a withdrawn figure is permitted only where a
reader cannot mistake it for a current claim. Two things establish that, and
either is enough:

  1. the enclosing unit carries one of a CLOSED list of markers
     (`WITHDRAWN`, `DOES NOT SURVIVE`, `SUPERSEDED`, `UNTIL 2026-09-`, ...),
     declared in `docs/withdrawn_figures.md` and in `MARKERS` below; or
  2. the enclosing unit also carries the figure's REPLACEMENT.

(`pre-fix` and `era 2` were in that list when this paragraph was first
written, and are deliberately not now -- see the comment on `MARKERS`.)

(2) is not a convenience. "12 of 31 -> 15 of 34" is the correct way to state a
correction, and a rule that forbade it would push authors toward deleting the
history instead of marking it -- which is how the provenance of a number gets
lost. The registry pairs every withdrawn figure with what replaced it, so this
check has something to look for.

**Why patterns are phrases, not numbers.** `16` and `1.06` occur on nearly
every page. `16 of 31`, `two distinct operating points`, `no cheap estimator
was implemented` occur once each per stale copy. A registry entry whose
pattern matches something innocuous is a bad entry, and
`test_registry_patterns_are_specific` says so rather than leaving the guard to
be disabled later for crying wolf.

**Scope.** `docs/*.md`, `README.md`, and the source of `attnbench/` and
`scripts/`. Code comments are in scope because the first thing this test
caught was one: `accuracy/grid_configs.py` still carried the pre-S1a
dominance counts that `claims.md` had corrected on 2026-09-20.

`tests/` is out of scope: the fixtures below contain the patterns on purpose.
`docs/withdrawn_figures.md` is out of scope for the same reason -- it is the
registry, so it contains every pattern by construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO / "docs" / "withdrawn_figures.md"

# The closed marker vocabulary. Adding to this list is a deliberate act: each
# entry is a phrase a reader recognises as "this is not a current claim". A
# loose marker (`corrected`, `see above`, `no longer`) would let any nearby
# hedge silence the check, which is how an allowlist rots into a permanent
# exemption. Matched case-insensitively as literal substrings.
MARKERS = (
    # A marker's ONLY function must be to disclaim currency. `pre-fix`,
    # `era 2` and `post-fix` read like markers and are not: they are this
    # project's ordinary vocabulary for naming a measurement regime, and they
    # appear inside live statements all the time. Seeded here at first draft,
    # they silently rescued five of the stale blocks the registry exists to
    # catch -- including `limitations.md`'s S1a status block, which says
    # "Regenerated post-fix:" in the middle of being wrong about what had been
    # regenerated. Removed, and recorded so they are not re-added.
    "WITHDRAWN",
    "DOES NOT SURVIVE",
    "SUPERSEDED",
    "UNTIL 2026-09-",
    "THE SENTENCE READ",
    "THIS PARAGRAPH READ",
    "PARAGRAPH THAT STOOD HERE",
    "THIS SUBSECTION ORIGINALLY READ",
    "THIS LINE PREVIOUSLY SAID",
    "THIS PARAGRAPH NAMED",
    "HAD SAID",
)

# A pattern matching more than this many places across the whole repository is
# too general to be a figure -- it is matching prose. See the module docstring.
MAX_OCCURRENCES_PER_ENTRY = 24


@dataclass(frozen=True)
class Entry:
    id: str
    pattern: str
    was: str
    replacement: str          # regex; "" means "withdrawn outright"
    note: str = ""
    retired: bool = False

    @property
    def pattern_re(self) -> re.Pattern:
        return re.compile(self.pattern)

    @property
    def replacement_re(self):
        return re.compile(self.replacement) if self.replacement else None


@dataclass(frozen=True)
class Violation:
    entry_id: str
    path: str
    line: int
    text: str

    def describe(self) -> str:
        return f"{self.path}:{self.line}  [{self.entry_id}]  {self.text.strip()[:110]}"


# --- the scanner ------------------------------------------------------------

def blocks(text: str) -> list[tuple[int, str]]:
    """(first_line_number, block_text) for each maximal run of non-blank lines.

    A markdown paragraph, a blockquote, a table, and a contiguous `#` comment
    block in Python are all one run of non-blank lines, which is exactly the
    unit a reader takes in as one statement. A marker in the paragraph ABOVE a
    table does not cover the table, and that is deliberate: the audit found
    `claims.md`'s dominance table correctly self-marked (its own rows say
    `pre-fix` / `rebuilt`) while the End-to-end section's Supported row, four
    hundred lines earlier, carried nothing.
    """
    out: list[tuple[int, str]] = []
    start: int | None = None
    buf: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if start is None:
                start = i
            buf.append(line)
        elif start is not None:
            out.append((start, "\n".join(buf)))
            start, buf = None, []
    if start is not None:
        out.append((start, "\n".join(buf)))
    return out


def units(first_line: int, block: str) -> list[tuple[int, str]]:
    """The spans a marker is allowed to cover, within one block.

    Normally a block is one unit: a paragraph, a blockquote, a run of `#`
    comments. **A markdown table is not.** Its rows are independent
    statements set side by side, and a marker in one of them says nothing
    about the others -- which is exactly how this check missed two stale rows
    on its first real run. `claims.md`'s "What survives the correction" table
    has `DOES NOT SURVIVE` in some of its rows, so every scan of it found a
    marker, and the whole table was permitted as one block. The two rows that
    were wrong sat inside a table whose entire purpose is tracking which
    claims have gone stale.

    A table that marks some of its rows must not immunise the rest, so a
    block containing two or more `|` lines is split line by line. A section
    heading carrying a marker still covers the whole table (`_under_marked_
    heading`), because that is a statement about everything beneath it; a
    table's own header row is not, and gets no such reach.
    """
    lines = block.splitlines()
    if not is_table(block):
        return [(first_line, block)]
    return [(first_line + i, ln) for i, ln in enumerate(lines)]


def is_table(block: str) -> bool:
    return sum(1 for ln in block.splitlines()
               if ln.lstrip().startswith("|")) >= 2


def _cells(row: str) -> list[str]:
    return row.strip().strip("|").split("|")


def _replacement_in_same_column(block: str, row: str, at: int, repl) -> bool:
    """Whether another row of this table states the replacement in the column
    the stale figure sits in.

    The one thing a marker may NOT do across rows, a replacement may -- but
    only down its own column. A before/after table is the correct way to state
    a correction:

        | | `dominated_measured` | `dominated_normalized` |
        | pre-fix | 16 / 31 (52%)        | 12 / 31 (39%)          |
        | rebuilt | **28 / 34 (82%)**    | **15 / 34 (44%)**      |

    Splitting that per row and demanding each one carry its own replacement
    would push authors to delete the `pre-fix` row, which is how the
    provenance of a number gets lost -- the same argument the module docstring
    makes for permitting `12 of 31 -> 15 of 34` inside a sentence. Column
    alignment is what makes the pairing legible to a reader, so it is what
    this check requires too: a replacement in some unrelated cell of a large
    table rescues nothing.
    """
    cells = _cells(row)
    offset, col = 0, None
    for i, c in enumerate(cells):
        offset += len(c) + 1
        if at < offset + (1 if i == 0 else 0):
            col = i
            break
    if col is None:
        return False
    for other in block.splitlines():
        if other is row or not other.lstrip().startswith("|"):
            continue
        if other.strip() == row.strip():
            continue
        oc = _cells(other)
        if col < len(oc) and repl.search(oc[col]):
            return True
    return False


def _marked(block_text: str) -> bool:
    upper = block_text.upper()
    return any(m in upper for m in MARKERS)


def marked_headings(text: str) -> list[int]:
    """Line numbers of markdown headings that carry a marker.

    A section headed "## The oracle can put sparse ABOVE dense — mostly
    withdrawn 2026-09-20" is read as withdrawn by anyone who arrived through
    the heading, so requiring every paragraph under it to repeat the marker
    would be noise. The scope ends at the NEXT heading of any level, so a
    withdrawal cannot silently annex the section after it.

    Markdown only. There is no equivalent in Python, which is why
    `accuracy/grid_configs.py` and `analysis/matched.py` get no such cover --
    correctly: a comment block is read on its own.
    """
    return [i for i, line in enumerate(text.splitlines(), start=1)
            if line.startswith("#") and _marked(line)]


def _under_marked_heading(text: str, line_no: int, is_markdown: bool) -> bool:
    if not is_markdown:
        return False
    headings = [i for i, line in enumerate(text.splitlines(), start=1)
                if line.startswith("#")]
    prior = [i for i in headings if i < line_no]
    return bool(prior) and prior[-1] in marked_headings(text)


def scan(entry: Entry, files: list[tuple[str, str]]) -> list[Violation]:
    """Unmarked occurrences of `entry` across `files` as (path, text) pairs.

    Takes the file contents rather than reading them, so the detector can be
    driven against a fixture. A checker that can only be run against the real
    tree cannot be shown to fire.
    """
    if entry.retired:
        return []
    pat, repl = entry.pattern_re, entry.replacement_re
    bad: list[Violation] = []
    for path, text in files:
        is_markdown = path.endswith(".md")
        for first_line, block in blocks(text):
            if pat.search(block) is None:
                continue        # cheap reject before splitting
            table = is_table(block)
            for unit_line, unit in units(first_line, block):
                m = pat.search(unit)
                if m is None:
                    continue
                if _marked(unit):
                    continue
                if repl is not None and repl.search(unit):
                    continue
                if (table and repl is not None
                        and _replacement_in_same_column(block, unit,
                                                        m.start(), repl)):
                    continue
                offset = unit[:m.start()].count("\n")
                line_no = unit_line + offset
                if _under_marked_heading(text, line_no, is_markdown):
                    continue
                line_text = unit.splitlines()[offset]
                bad.append(Violation(entry.id, path, line_no, line_text))
    return bad


def occurrences(entry: Entry, files: list[tuple[str, str]]) -> int:
    return sum(len(entry.pattern_re.findall(text)) for _, text in files)


# --- the registry -----------------------------------------------------------

def load_registry(path: Path = REGISTRY_PATH) -> list[Entry]:
    """Entries from the single ```json block in `docs/withdrawn_figures.md`.

    One file, human-readable and machine-readable, rather than a markdown
    table the test has to parse (regex alternation and `|` do not coexist in a
    table cell) or a second file that can drift from the prose beside it.
    JSON rather than TOML because `requires-python = ">=3.10"` and `tomllib`
    landed in 3.11 -- the instances run 3.10.12.
    """
    text = path.read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.S)
    if m is None:
        raise AssertionError(f"{path} has no ```json registry block")
    payload = json.loads(m.group(1))
    return [Entry(**row) for row in payload["entries"]]


def files_in_scope() -> list[tuple[str, str]]:
    paths: list[Path] = []
    paths += sorted(p for p in (REPO / "docs").glob("*.md")
                    if p.name != REGISTRY_PATH.name)
    paths += [REPO / "README.md"]
    paths += sorted((REPO / "attnbench").rglob("*.py"))
    paths += sorted((REPO / "scripts").glob("*.py"))
    paths += sorted((REPO / "scripts").glob("*.sh"))
    return [(str(p.relative_to(REPO)), p.read_text()) for p in paths if p.exists()]


# --- anti-vacuity -----------------------------------------------------------

def test_files_in_scope_are_actually_found():
    """If the globs return nothing, every check below passes on an empty set."""
    files = files_in_scope()
    names = {path for path, _ in files}
    assert len(files) > 60, f"only {len(files)} files in scope"
    for required in ("docs/claims.md", "docs/limitations.md", "README.md",
                     "attnbench/accuracy/grid_configs.py"):
        assert required in names, f"{required} is not being scanned"


def test_the_registry_parses_and_has_entries():
    entries = load_registry()
    assert entries, "the registry is empty: this suite proves nothing"
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


# --- the detector, shown to fire -------------------------------------------

_FIXTURE = Entry(
    id="fixture-only",
    pattern=r"17 of 41",
    was="a figure that never existed, used to exercise the detector",
    replacement=r"19 of 44",
)


def test_the_detector_catches_a_reintroduced_stale_figure():
    """A check nobody has watched fail is a guess.

    This is the prose analogue of
    `test_timed_region_setup.test_the_detector_catches_a_deliberately_
    reintroduced_rebuild`: put the defect back and require the detector to
    notice. Written before the registry was populated, so the machinery was
    shown to work against something other than its own seed data.
    """
    doc = "Some prose.\n\nThe result is 17 of 41 cells, which settles it.\n"
    bad = scan(_FIXTURE, [("fake.md", doc)])
    assert len(bad) == 1, f"detector missed an unmarked stale figure: {bad}"
    assert bad[0].line == 3, bad[0]


@pytest.mark.parametrize("marker", ["WITHDRAWN", "superseded",
                                    "until 2026-09-20", "the sentence read"])
def test_a_marked_occurrence_is_permitted(marker):
    doc = f"Some prose.\n\nIt read 17 of 41 ({marker}).\n"
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


@pytest.mark.parametrize("not_a_marker", ["pre-fix", "era 2", "post-fix",
                                          "corrected", "see above"])
def test_regime_vocabulary_is_not_a_marker(not_a_marker):
    """The mistake this file made on its first draft.

    `pre-fix` and `era 2` name a measurement regime; they appear inside live
    statements constantly. Admitting them as markers silently rescued five of
    the blocks the registry exists to catch. Asserted rather than left to a
    comment, because the pull to add them back is the pull toward a guard that
    never fails.
    """
    doc = f"Some prose.\n\nThe {not_a_marker} figure is 17 of 41 cells.\n"
    assert len(scan(_FIXTURE, [("fake.md", doc)])) == 1


def test_an_occurrence_beside_its_replacement_is_permitted():
    """Stating the correction is the point, not a loophole."""
    doc = "Some prose.\n\nThe count moved: 17 of 41 becomes 19 of 44.\n"
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


def test_a_marker_in_a_different_block_does_not_cover_this_one():
    """The failure mode the audit actually found: a correction four hundred
    lines away from the copy it was supposed to reach."""
    doc = ("The rebuilt figures are 19 of 44 and this row is WITHDRAWN.\n"
           "\n"
           "Elsewhere: the study reports 17 of 41 matched points.\n")
    bad = scan(_FIXTURE, [("fake.md", doc)])
    assert len(bad) == 1 and bad[0].line == 3, bad


# --- tables: the hole the first real run left open -------------------------
#
# A table marking SOME of its rows immunised ALL of them, because a table is
# one run of non-blank lines and therefore one block. `claims.md`'s "What
# survives the correction" table -- whose entire job is recording which claims
# have gone stale -- had `DOES NOT SURVIVE` in some rows and two stale rows in
# others, and every scan of it found a marker and stopped. The two had to be
# corrected by hand, which is the definition of a check rather than a guard.

def test_a_marker_in_one_table_row_does_not_cover_another():
    """The defect, reproduced at fixture scale."""
    doc = ("| claim | status |\n"
           "|---|---|\n"
           "| the decode figure | DOES NOT SURVIVE |\n"
           "| the matched count | 17 of 41, unchanged |\n")
    bad = scan(_FIXTURE, [("fake.md", doc)])
    assert len(bad) == 1 and bad[0].line == 4, bad


def test_a_marked_table_row_is_permitted():
    doc = ("| claim | status |\n"
           "|---|---|\n"
           "| the matched count | 17 of 41 — WITHDRAWN 2026-09-20 |\n")
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


def test_a_before_after_table_states_its_correction_down_a_column():
    """The legitimate case per-row scanning would otherwise break, and the
    first thing it flagged in the live tree (`claims.md`'s pre-fix/rebuilt
    dominance table). Demanding each row carry its own replacement pushes
    authors to delete the stale row instead of pairing it."""
    doc = ("| | measured | normalized |\n"
           "|---|---|---|\n"
           "| pre-fix | 17 of 41 | 12 of 31 |\n"
           "| rebuilt | **19 of 44** | **15 of 34** |\n")
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


def test_a_replacement_in_a_different_column_does_not_permit():
    """The narrow version of the loophole that admitting block-scoped
    replacements would have reopened: one cell somewhere in a wide table
    happens to contain the replacement string, and every stale row in it goes
    quiet. Column alignment is what a reader uses to pair the two, so it is
    what the check requires."""
    doc = ("| | measured | normalized |\n"
           "|---|---|---|\n"
           "| pre-fix | 17 of 41 | 12 of 31 |\n"
           "| rebuilt | **28 of 44** | **19 of 44** |\n")
    bad = scan(_FIXTURE, [("fake.md", doc)])
    assert len(bad) == 1 and bad[0].line == 3, bad


def test_a_marked_heading_still_covers_a_whole_table():
    """Per-row scanning narrows what a MARKER reaches, not what a heading
    does. A section headed as withdrawn is a statement about everything under
    it, including its tables. A table's own header row is not."""
    doc = ("## The dominance result — WITHDRAWN 2026-09-20\n"
           "\n"
           "| claim | status |\n"
           "|---|---|\n"
           "| the matched count | 17 of 41 |\n")
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


def test_a_table_header_row_does_not_cover_its_body():
    doc = ("| claim | status (all WITHDRAWN) |\n"
           "|---|---|\n"
           "| the matched count | 17 of 41 |\n")
    bad = scan(_FIXTURE, [("fake.md", doc)])
    assert len(bad) == 1 and bad[0].line == 3, bad


def test_a_two_line_paragraph_is_still_one_unit():
    """Only tables split. A marker on the first line of an ordinary paragraph
    still covers its second, or every wrapped sentence would need its own."""
    doc = ("This paragraph is WITHDRAWN as of 2026-09-20 and\n"
           "the count it gave was 17 of 41 cells.\n")
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


def test_a_marked_heading_covers_its_own_section():
    doc = ("## A result — mostly withdrawn 2026-09-20\n"
           "\n"
           "The effect was 17 of 41 cells and most of it is gone.\n")
    assert scan(_FIXTURE, [("fake.md", doc)]) == []


def test_a_marked_heading_does_not_cover_the_next_section():
    """A withdrawal must not annex whatever is written after it."""
    doc = ("## A result — WITHDRAWN 2026-09-20\n"
           "\n"
           "The effect was 17 of 41 cells.\n"
           "\n"
           "## Something else entirely\n"
           "\n"
           "We report 17 of 41 matched points.\n")
    bad = scan(_FIXTURE, [("fake.md", doc)])
    assert len(bad) == 1 and bad[0].line == 7, bad


def test_heading_scope_does_not_apply_to_source_files():
    """A `#` comment block is read on its own; there is no heading above it
    that a reader arrives through. This is why the grid_configs.py and
    matched.py occurrences were caught and the claims.md section intro was
    not."""
    src = "# WITHDRAWN: an old note\n\n# The count is 17 of 41 per call.\n"
    bad = scan(_FIXTURE, [("attnbench/fake.py", src)])
    assert len(bad) == 1, bad


def test_a_retired_entry_is_not_scanned():
    retired = Entry(id="x", pattern=r"17 of 41", was="", replacement="",
                    retired=True)
    assert scan(retired, [("fake.md", "17 of 41\n")]) == []


# --- the guard --------------------------------------------------------------

def test_no_withdrawn_figure_appears_unmarked():
    files = files_in_scope()
    bad: list[Violation] = []
    for entry in load_registry():
        bad.extend(scan(entry, files))
    assert not bad, (
        f"{len(bad)} withdrawn or superseded figure(s) stated as live:\n  "
        + "\n  ".join(v.describe() for v in bad)
        + "\n\nEither mark the occurrence (see MARKERS in this file), state "
          "the replacement beside it, or correct it. "
          "docs/withdrawn_figures.md says what replaced each one.")


def test_registry_patterns_are_specific():
    """A pattern that matches prose rather than a figure makes this check
    noisy, and a noisy check gets disabled. Bound it."""
    files = files_in_scope()
    too_broad = [(e.id, occurrences(e, files)) for e in load_registry()
                 if occurrences(e, files) > MAX_OCCURRENCES_PER_ENTRY]
    assert not too_broad, (
        f"pattern(s) matching more than {MAX_OCCURRENCES_PER_ENTRY} places -- "
        f"too general to be a figure: {too_broad}")


def test_no_dead_registry_entry():
    """An entry matching nowhere is either a typo'd pattern or a figure that
    has been fully expunged. The second is fine and is declared with
    `retired: true`; the first is a hole in the guard that looks like
    coverage. Same shape as `test_no_stale_allowlist`."""
    files = files_in_scope()
    dead = [e.id for e in load_registry()
            if not e.retired and occurrences(e, files) == 0]
    assert not dead, (
        f"registry entries matching nothing anywhere: {dead}. If the figure is "
        f"gone for good, set \"retired\": true; otherwise the pattern is wrong "
        f"and this entry is protecting nothing.")


def test_every_entry_names_what_replaced_it_or_says_it_was_withdrawn():
    for e in load_registry():
        assert e.was.strip(), f"{e.id}: no description of what the figure was"
        assert e.replacement.strip() or e.note.strip(), (
            f"{e.id}: no replacement and no note. A withdrawal with neither is "
            f"a figure nobody can act on.")


# --- a correction is not a free zone ---------------------------------------

def riding_on_a_disclaimer(entries: list[Entry],
                           files: list[tuple[str, str]]) -> list[Violation]:
    """Table rows that state a SUPERSEDED figure as the verdict on another.

    **The incident, and it is not the one per-row scanning fixes.** The row in
    `claims.md`'s "What survives the correction" table read:

        | *16 of 31 matched sparse points are dominated by dense* |
          **DOES NOT SURVIVE.** Normalized: **12 of 31**. |

    Every scoping rule permits that row, per-row scanning included, because
    the row carries its own marker. But the marker is about `16 of 31`. The
    figure offered as the current answer, `12 of 31`, was itself superseded by
    `15 of 34` on 2026-09-20 -- and it was invisible, because a withdrawal
    note is a permanently marked context and the next withdrawal hid inside
    the previous one's disclaimer. A correction's right-hand side has to be
    current, or the correction is the stalest thing on the page.

    **Why this is restricted to table rows.** The obvious generalisation --
    "in any marked unit, only the FIRST registry match is covered" -- was
    written and measured against the live tree before being rejected. It
    produced three violations, all on blocks quoting a withdrawn paragraph
    verbatim (`claims.md:302`, `claims.md:305`, `limitations.md:479`), which
    is the single construction this whole scheme most wants to encourage: the
    registry pairs a figure with its replacement precisely so that authors
    mark history rather than deleting it. A rule that taxes verbatim
    quotation buys a narrow catch with the mechanism's main benefit.

    A table row is different in kind from a quotation. It is a record with a
    claim cell and a verdict cell, and the verdict is asserted in the reader's
    present tense. That is why supersession tables are where this fails, and
    why the rule stops there. Restricted this way it flags nothing in the live
    tree and still catches the row above.
    """
    live = [e for e in entries if not e.retired]
    out: list[Violation] = []
    for path, text in files:
        for first_line, block in blocks(text):
            if not is_table(block):
                continue
            for unit_line, unit in units(first_line, block):
                if not _marked(unit):
                    continue
                hits = sorted((m.start(), e) for e in live
                              for m in [e.pattern_re.search(unit)] if m)
                for start, e in hits[1:]:
                    r = e.replacement_re
                    if r is not None and r.search(unit):
                        continue
                    off = unit[:start].count("\n")
                    out.append(Violation(e.id, path, unit_line + off,
                                         unit.splitlines()[off]))
    return out


def broken_chains(entries: list[Entry]) -> list[tuple[str, str]]:
    """(entry, superseding entry) where an entry's REPLACEMENT is itself a
    withdrawn figure.

    The registry half of the same defect. If `16 of 31` is recorded as
    replaced by `12 of 31`, and `12 of 31` is later withdrawn in its own
    right, the first entry now points at a dead figure -- and every block
    permitted by rule 2 for stating `12 of 31` beside `16 of 31` is permitted
    on the strength of a value that is no longer true. The moment of
    withdrawal is the only moment anyone is looking, so that is where this
    fires: adding the new entry fails the suite until the old one's
    replacement is updated to the current value.
    """
    live = [e for e in entries if not e.retired]
    bad: list[tuple[str, str]] = []
    for a in live:
        if not a.replacement:
            continue
        probe = a.replacement.replace("\\", "")
        for b in live:
            if a.id != b.id and b.pattern_re.search(probe):
                bad.append((a.id, b.id))
    return bad


def test_no_table_row_states_a_superseded_figure_as_its_verdict():
    assert riding_on_a_disclaimer(load_registry(), files_in_scope()) == []


def test_the_disclaimer_rule_catches_the_row_it_was_written_for():
    """The row as it actually stood, with the registry as it actually is."""
    reg = load_registry()
    row = ("| claim | survives the decode correction? |\n"
           "|---|---|\n"
           "| *16 of 31 matched sparse points are dominated by dense* | "
           "**DOES NOT SURVIVE.** Normalized: **12 of 31**. |\n")
    bad = riding_on_a_disclaimer(reg, [("docs/claims.md", row)])
    assert [v.entry_id for v in bad] == ["stage6-dominated-normalized"], bad


def test_a_verdict_stating_the_current_figure_is_permitted():
    reg = load_registry()
    row = ("| claim | verdict |\n"
           "|---|---|\n"
           "| *16 of 31 matched sparse points are dominated by dense* | "
           "**DOES NOT SURVIVE.** Normalized: **15 of 34**. |\n")
    assert riding_on_a_disclaimer(reg, [("docs/claims.md", row)]) == []


def test_a_verbatim_quotation_of_a_withdrawn_paragraph_is_not_flagged():
    """Asserted, not left to the restriction's comment. Widening this rule to
    prose was measured and rejected; this is what it would have cost."""
    reg = load_registry()
    prose = ("*This paragraph read: \"**16 of 31 accuracy-matched sparse "
             "operating points are dominated**. Only **two distinct "
             "operating points** are genuinely faster than dense.\" Every "
             "clause of it moved.*\n")
    assert riding_on_a_disclaimer(reg, [("docs/claims.md", prose)]) == []


def test_no_entry_is_replaced_by_a_figure_that_is_itself_withdrawn():
    bad = broken_chains(load_registry())
    assert not bad, (
        "these registry entries name a replacement that is itself a withdrawn "
        "figure: " + ", ".join(f"{a} -> {b}" for a, b in bad)
        + ". Update the first entry's replacement to the current value, then "
        "go and correct the prose that states the dead one as current -- that "
        "prose is permitted by rule 2 and nothing else will find it.")


def test_the_chain_check_fires():
    chained = [
        Entry(id="first", pattern="16 of 31", was="x", replacement="12 of 31"),
        Entry(id="second", pattern="12 of 31", was="y", replacement="15 of 34"),
    ]
    assert broken_chains(chained) == [("first", "second")]
    assert broken_chains(chained[:1]) == []


def test_the_chain_check_ignores_retired_entries():
    chained = [
        Entry(id="first", pattern="16 of 31", was="x", replacement="12 of 31"),
        Entry(id="second", pattern="12 of 31", was="y", replacement="",
              retired=True),
    ]
    assert broken_chains(chained) == []
