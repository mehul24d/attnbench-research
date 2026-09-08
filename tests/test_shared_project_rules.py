"""Project decisions must have one definition, not one per call site.

This file exists because of a class of failure this project keeps meeting
from a new angle (docs/silent_failure_patterns.md #22, #23): a rule is fixed
correctly in the place it was noticed, the fix is invisible to the next
harness, and the next harness re-derives the bug -- or, as here, re-derives
the rule and the two then drift silently.

These are source-level assertions. That is deliberate: the property being
checked is "there is one definition", and no runtime check can see a second
definition that agrees today.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((REPO / "scripts").glob("*.py"))


def test_scripts_exist_to_scan():
    """If the glob ever returns nothing, every test below passes vacuously
    -- which is exactly the shape of failure this file is about."""
    assert len(SCRIPTS) > 10


def test_the_gla_exclusion_has_exactly_one_definition():
    """The pre-registered DROP verdict (docs/gla_arm_decision.md) decides
    which rows every accuracy analysis may read.

    It was restated in four places -- run_pareto, run_matched_analysis and
    run_decode_confound each with `EXCLUDE_BACKENDS = ("gla",)`, and
    run_phase_timing with an inline `df.backend != "gla"` that does not even
    grep the same way -- carrying two different justifications between them.
    Reopening the arm would have needed three of the four found by hand.
    """
    definition = re.compile(r'^\s*EXCLUDE_BACKENDS\s*=\s*\(')
    offenders = [p.name for p in SCRIPTS
                 if any(definition.match(line)
                        for line in p.read_text().splitlines())]
    assert offenders == [], (
        f"{offenders} define their own backend-exclusion list. Import "
        f"ACCURACY_EXCLUDED_BACKENDS from attnbench.accuracy.grid_configs, "
        f"which is where the arm decision that produced it lives.")


def test_no_script_filters_gla_by_an_inline_literal():
    """The form matters as much as the value. An inline `!= "gla"` is a
    fourth copy that does not answer a grep for the constant's name, so an
    audit of the rule misses it."""
    inline = re.compile(r'backend\s*(==|!=)\s*["\']gla["\']')
    offenders = [p.name for p in SCRIPTS if inline.search(p.read_text())]
    assert offenders == [], (
        f"{offenders} filter gla by an inline literal. Use "
        f"ACCURACY_EXCLUDED_BACKENDS so the rule is greppable by name.")


def test_the_shared_constant_says_what_it_excludes_and_why():
    """A bare tuple in a shared module is worse than four local copies: it
    is authoritative and unexplained. The comment carrying the verdict is
    load-bearing, so its absence should fail."""
    src = (REPO / "attnbench" / "accuracy" / "grid_configs.py").read_text()
    i = src.index("ACCURACY_EXCLUDED_BACKENDS")
    preamble = src[max(0, i - 1400):i]
    assert "gla_arm_decision.md" in preamble
    assert "DROP" in preamble


def test_clock_lock_callers_pass_what_happened_into_the_stamp():
    """#23. `provenance.capture()` defaults clocks_locked to False, so any
    caller that locks and does not pass the result stamps a falsehood over
    its own measurement. Two call sites lock; both must pass it."""
    lockers = [p for p in SCRIPTS
               if "provenance.lock_clocks()" in p.read_text()]
    assert lockers, "no caller locks clocks -- has the API been renamed?"
    for p in lockers:
        src = p.read_text()
        assert re.search(r"capture\(\s*clocks_locked\s*=", src), (
            f"{p.name} calls lock_clocks() but never passes the result to "
            f"provenance.capture(); its rows will claim clocks_locked=False. "
            f"See docs/silent_failure_patterns.md #23.")
