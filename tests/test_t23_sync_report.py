"""T2.3's report on banked sequences (`scripts/t23_sync_report.py`): the
stream check and synchronisation column per cell, the strict verdict
against an earlier session, and the asserts from the asserts-removed arm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from attnbench.components import op_sequence as O  # noqa: E402
import t23_sync_report as T  # noqa: E402
from test_components_op_sequence import assert_arm, kernel, parse, TB  # noqa: E402


def _bank(d: Path, name: str, seq) -> None:
    (d / name).mkdir(parents=True)
    (d / name / "sequence.json").write_text(json.dumps(O.to_record(seq)))


def test_the_report_counts_streams_synchronisations_and_attributed_asserts(tmp_path):
    shipped, stripped = parse(assert_arm([1, 2])), parse(assert_arm([0, 0]))
    _bank(tmp_path, "xattention_dcal_L2048_b1", shipped)
    _bank(tmp_path, "xattention_noassert_dcal_L2048_b1", stripped)
    r = T.report(tmp_path)
    c = r["cells"]["xattention_dcal_L2048_b1"]
    assert c["single_stream"] and c["synchronisations"] == 3
    assert c["sync_column"] == O.sync_column(shipped)
    assert r["cells"]["xattention_noassert_dcal_L2048_b1"]["synchronisations"] == 0
    assert list(r["asserts"]) == ["xattention_dcal_L2048_b1"]
    assert r["asserts"]["xattention_dcal_L2048_b1"]["assert_synchronisations"] == 3
    assert "against_earlier" not in c


def test_a_refused_attribution_is_reported_not_dropped(tmp_path):
    _bank(tmp_path, "xattention_dcal_L2048_b1", parse(assert_arm([1])))
    _bank(tmp_path, "xattention_noassert_dcal_L2048_b1", parse(assert_arm([0], extra=True)))
    a = T.report(tmp_path)["asserts"]["xattention_dcal_L2048_b1"]
    assert a["attributed"] is False and "not the shipped arm" in a["refusal"]


def test_each_cell_is_compared_with_the_earlier_sessions_same_cell(tmp_path):
    now, before = tmp_path / "now", tmp_path / "before"
    _bank(now, "dense_d1_L2048_b1", parse(assert_arm([1])))
    _bank(before, "dense_d1_L2048_b1", parse(assert_arm([1])))
    _bank(now, "meanpool_d0.5_L2048_b1", parse(assert_arm([1])))
    tb = TB()
    with tb.region(), tb.layer(0):
        kernel(tb, "aten::mm")
    _bank(before, "meanpool_d0.5_L2048_b1", parse(tb))
    r = T.report(now, before)["cells"]
    assert r["dense_d1_L2048_b1"]["against_earlier"]["verdict"] == O.SAME
    assert r["meanpool_d0.5_L2048_b1"]["against_earlier"]["verdict"] == O.DIFFERENT
