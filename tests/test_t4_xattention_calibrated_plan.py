"""docs/t4_xattention_calibrated.md says what the code that runs it does.

A pre-registration and the module the calibration, the run and the analysis
read must not disagree: if they do, one of them is not what was registered.
"""

from __future__ import annotations

import re
from pathlib import Path

from attnbench.accuracy import t4_pilot

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "t4_xattention_calibrated.md").read_text()


def test_calibrations_and_their_roles():
    roles = {"claim": "claim", "descriptive": "descriptive"}
    for name, role in t4_pilot.XATTN_CALIBRATIONS.items():
        assert re.search(rf"\| `{name}` \|[^\n]*\| {roles[role]} \|", DOC), name
    assert [n for n, r in t4_pilot.XATTN_CALIBRATIONS.items() if r == "claim"] == ["authors"]
    assert "for CAL in " + " ".join(t4_pilot.XATTN_CALIBRATIONS) + "; do" in DOC


def test_the_heldout_set_is_the_modules():
    assert (f"seed {t4_pilot.XATTN_CALIBRATION_SEED}, "
            f"{t4_pilot.XATTN_CALIBRATION_RULER_N} per (task, band)") in DOC


def test_table_paths_and_settings():
    for path in t4_pilot.XATTN_CALIBRATION_TABLES.values():
        assert path.replace("authors", "${CAL}").replace("ruler_heldout", "${CAL}") in DOC
    assert f"stride {t4_pilot.XATTN_STRIDE}" in DOC
    assert f"Margin of {t4_pilot.MARGIN_PTS:g} points; one-sided alpha {t4_pilot.ALPHA}" in DOC


def test_cell_counts():
    examples = (sum(t4_pilot.SPARSE_PILOT_N[t] for t in t4_pilot.SPARSE_PILOT_TASKS)
                * len(t4_pilot.PILOT_BANDS))
    assert (f"The dense run is {examples}\ncells, and each calibration run is "
            f"{examples} cells.") in DOC


def test_named_tests_and_scripts_exist():
    for path in re.findall(r"`((?:tests|scripts|attnbench)/[\w/.]+\.(?:py|sh))`", DOC):
        assert (ROOT / path).exists(), path


def test_committed_tables_load_and_fit_the_model():
    """Absent until the calibration session's tables are committed; from then
    on each must load (its digest matches its values), carry its own name,
    and have one row per layer and one value per query head of the model."""
    from attnbench.backends.xattention import ThresholdTable
    for name, path in t4_pilot.XATTN_CALIBRATION_TABLES.items():
        p = ROOT / path
        if not p.exists():
            continue
        t = ThresholdTable.load(p)
        assert t.name == name
        assert len(t.values) == 28 and {len(r) for r in t.values} == {12}   # Qwen2.5-1.5B


def test_the_dated_amendments_are_appended_and_match_the_module():
    """T4 amendments A1-A7 (2026-10-03): appended, dated, and holding the
    module's numbers."""
    i = DOC.index("## Amendments, 2026-10-03 (before Session A)")
    amend = " ".join(DOC[i:].split())
    for a in ("A1.", "A2.", "A3.", "A4.", "A5.", "A6.", "A7."):
        assert f"**{a}" in amend, a
    assert f"index offset {t4_pilot.XATTN_CALIBRATION_INDEX_OFFSET}" in amend
    assert f"XATTN_CALIBRATION_INDEX_OFFSET = {t4_pilot.XATTN_CALIBRATION_INDEX_OFFSET}" in amend
    assert f'XATTN_CALIBRATION_CARD = "{t4_pilot.XATTN_CALIBRATION_CARD}"' in amend
    assert "max over used texts" in amend and "no cap" in amend
    assert f"max − p90 > {t4_pilot.XATTN_CALIBRATION_DESCRIPTIVE_GAP:g}" in amend
    # The pre-registered text above the amendments is unchanged in place.
    assert DOC.index("## Sessions") < i
