"""docs/t4_xattention_pilot.md says what the code that runs it does.

The pre-registration and the module the run and the analysis read must not
disagree: if they do, one of them is not what was registered.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from attnbench.accuracy import t4_pilot
from attnbench.backends.xattention import XAttentionBackend

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "t4_xattention_pilot.md").read_text()


def test_thresholds_in_order_and_in_the_cli():
    taus = t4_pilot.XATTN_THRESHOLD_SEQUENCE
    assert taus == (0.95, 0.9, 0.8)
    assert "**Thresholds:** 0.95, 0.9 and 0.8, tested in that order" in DOC
    assert "test tau 0.95, then 0.9,\n  then 0.8" in DOC
    assert "for TAU in " + " ".join(f"{t:g}" for t in taus) + "; do" in DOC


def test_margin_alpha_and_score_source():
    assert f"Margin of {t4_pilot.MARGIN_PTS:g} points; one-sided alpha {t4_pilot.ALPHA}" in DOC
    assert f"`{t4_pilot.XATTN_SCORE_SOURCE}`" in DOC


def test_the_settings_are_the_backends_defaults():
    defaults = {k: p.default for k, p in
                inspect.signature(XAttentionBackend.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}
    assert defaults["stride"] == t4_pilot.XATTN_STRIDE == 8
    assert defaults["keep_sink"] is True and defaults["keep_recent"] is True
    assert f"stride {t4_pilot.XATTN_STRIDE}" in DOC
    assert "`keep_sink=True`, `keep_recent=True`" in DOC


def test_cell_counts():
    per_band = sum(t4_pilot.SPARSE_PILOT_N[t] for t in t4_pilot.SPARSE_PILOT_TASKS)
    examples = per_band * len(t4_pilot.PILOT_BANDS)
    assert f"The dense run is {examples} cells. Each threshold run is {examples} cells." in DOC


def test_named_tests_and_scripts_exist():
    for path in re.findall(r"`((?:tests|scripts|attnbench)/[\w/.]+\.(?:py|sh))`", DOC):
        assert (ROOT / path).exists(), path
