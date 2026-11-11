"""T4.3's prompts (`scripts/t43_prompts.py`): one text, two orders, every
named file present, every cited section a heading of the plan."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t43_prompts as P  # noqa: E402


def test_the_doc_is_current():
    assert P.main(["--check"]) == 0


def test_both_readers_cover_all_four_areas_b_in_reverse():
    names = [a[0] for a in P.AREAS]
    for reader, order in (("A", names), ("B", names[::-1])):
        text = P.prompt(reader)
        found = re.findall(r"^### \d\. Area \d: (.+)$", text, re.M)
        assert found == order
        assert f"`T43_read_{reader}.md`" in text and P.DEADLINE in text


def test_every_named_file_exists():
    for _, _, files, _ in P.AREAS:
        for f in files:
            path = f.split(" ")[0].rstrip(":")
            assert (ROOT / path).exists(), path


def test_every_cited_section_is_a_heading_of_the_plan():
    plan = (ROOT / "docs" / "RESEARCH_PLAN.md").read_text()
    for num, title in P.SECTIONS.items():
        level = "##" if "." not in num else "###"
        assert f"\n{level} {num} {title}\n" in plan or f"\n{level} {num}. {title}\n" in plan, num


def test_the_commit_is_pinned_only_when_given():
    assert P.PIN in P.prompt("A")
    sha = "0" * 40
    pinned = P.prompt("B", sha)
    assert P.PIN not in pinned and f"git checkout {sha}" in pinned


def test_the_rules_keep_the_reads_independent_and_off_the_cloud():
    for reader in "AB":
        text = P.prompt(reader)
        assert "Do not read the other reader's report" in text
        assert "gcp_*" in text and "design leakage" in text
