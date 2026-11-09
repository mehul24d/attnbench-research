"""The P4 card's built region (`scripts/t64_card.py`; adviser's ruling 7 of
the eleventh 9 Nov 2026 paste): the runner's steps with the minutes file's
sums, and Flex-start, Spot blocks and on-demand from the same numbers;
Flex-start first and Spot's four 2.3 h blocks the fallback (adviser's ruling
5 of the 10 Nov 2026 paste)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t64_blocks as B  # noqa: E402
import t64_card as C  # noqa: E402
from t64_session import T64  # noqa: E402


def _built():
    return C.build(C.load_minutes(), "2026-11-10")


def test_the_card_on_disk_is_this_build():
    assert C.main(["--check"]) == 0
    text = C.CARD.read_text()
    a, b = C.region(text)
    assert text[a:b] == _built()


def test_the_step_table_is_the_runners_steps_with_the_minutes_files_sums():
    m = C.load_minutes()
    got = B.card_numbers(C.CARD.read_text())
    assert [n for n, _ in got["steps"]] == [s.name for s in T64.steps]
    assert dict(got["steps"]) == {n: float(sum(e["minutes"])) for n, e in m["steps"].items()}
    assert (got["setup_h"], got["teardown_h"]) == (m["setup_h"]["value"], m["teardown_h"]["value"])


def test_flex_and_spot_come_from_the_same_minutes_with_each_boots_overhead():
    m = C.load_minutes()
    steps = [(s.name, float(sum(m["steps"][s.name]["minutes"]))) for s in T64.steps]
    run = sum(x for _, x in steps)
    planned = 0.6 + run / 60 + 0.15
    card = _built()
    assert f"| Expected, nothing fails | ₹{math.ceil(round(planned, 3) * C.FLEX_RATE)} |" in card
    assert f"₹{math.ceil(round(planned, 3) * C.FLEX_RATE * B.CAP_FACTOR)} (halt" in card
    blocks, refusal = C.spot_blocks(steps, 0.6, 0.15)
    boot = sum(dict(steps)[s] for s in B.BOOT["P4"])
    # every block pays setup, its boot steps and teardown again
    assert sum(b.planned_h for b in blocks) > planned + (len(blocks) - 1) * (0.75 + boot / 60) - 1e-6
    assert f"₹{math.ceil(sum(b.planned_h for b in blocks) * C.SPOT_RATE)} |" in card
    assert f"₹{sum(b.cap_inr for b in blocks)}, the caps summed" in card
    assert "not offered for `a3-highgpu-1g`" in card


def test_flex_is_first_and_the_fallback_is_four_blocks_at_2_3_hours():
    card = _built()
    steps = B.card_numbers(C.CARD.read_text())["steps"]
    blocks, refusal = C.spot_blocks(steps, 0.6, 0.15)
    assert len(blocks) == 4 and all(b.planned_h <= 2.3 + 1e-9 for b in blocks)
    assert "Flex-start first, its request submitted right after the lock (14 Nov 2026)" in card
    assert "if no grant has come by about 22 Nov 2026, Spot in 4 blocks of 2.3 h" in card
    # why not 2.2 h is stated, not hidden
    assert refusal == "host_components takes 79 min, more than a block's 78"
    assert f"at {B.MAX_BLOCK_HOURS} h the blocks are refused: {refusal}" in card
    short = [(n, 20.0 if n == "host_components" else x) for n, x in steps]
    with pytest.raises(ValueError, match="not the ruled 4"):
        C.spot_blocks(short, 0.6, 0.15)        # 3 blocks: the fallback is no longer the ruled one
    long = [(n, 3 * x if n == "ground_truth" else x) for n, x in steps]
    with pytest.raises(ValueError, match="not the ruled 4"):
        C.spot_blocks(long, 0.6, 0.15)


def test_a_minutes_file_off_the_runners_steps_is_refused(tmp_path):
    m = C.load_minutes()
    del m["steps"]["masks"]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="T64's steps are"):
        C.load_minutes(p)


def test_the_card_says_the_suite_record_is_not_in_the_setup_and_no_command_passes_it():
    text = _built()
    assert "**No suite record in the setup**" in text and "20 minutes" in text
    assert not [l for l in text.splitlines() if "gcp_bootstrap_instance.sh" in l and "--suite-record" in l]
