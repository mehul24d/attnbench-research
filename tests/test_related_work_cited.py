"""Audit C5 (2026-10-01): the prior art the positioning rests on stays cited.

The 2026-09-30 audit found Framework Tax and TaxBreak cited only in
`limitations.md`, and Sparse Frontier v3, XAttention, BSFA, FSA, arXiv
2606.07703 and Mytkowicz et al. cited nowhere. Each was added to
`claims.md` § "Where this study sits" and copied into the write-up's
"Where the finding sits". A rewrite that drops one from either copy reopens
the audit item without anything noticing; this notices.

Identifiers, not prose: an arXiv id or DOI is what a reader resolves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CLAIMS = REPO / "docs" / "claims.md"
WRITEUP = REPO / "docs" / "writeup_input.md"

CITED = {
    "Framework Tax": "arXiv:2302.06117",
    "TaxBreak": "arXiv:2603.12465",
    "MInference": "arXiv:2407.02490",
    "XAttention": "arXiv:2503.16428",
    "BSFA": "arXiv:2512.07011",
    "FSA": "arXiv:2508.18224",
    "oracle-guided sparse prefill": "arXiv:2606.07703",
    "Sparse Frontier, Findings ACL 2026": "10.18653/v1/2026.findings-acl.1926",
    "Mytkowicz et al.": "ASPLOS '09",
}


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start:] if nxt == -1 else text[start:nxt]


@pytest.mark.parametrize("name", sorted(CITED))
def test_cited_in_the_ledger_positioning(name):
    sec = _section(CLAIMS.read_text(), "## Where this study sits")
    sec += _section(CLAIMS.read_text(), "## Positioning against Sparse Frontier")
    assert CITED[name] in sec, f"{name} ({CITED[name]}) dropped from claims.md"


@pytest.mark.parametrize("name", sorted(CITED))
def test_cited_in_the_writeup_related_work(name):
    sec = _section(WRITEUP.read_text(), "### Where the finding sits")
    assert CITED[name] in sec, f"{name} ({CITED[name]}) dropped from writeup_input.md"
