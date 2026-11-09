"""The P2 launch card's builder (`scripts/t61_card.py`; adviser's ruling 9 of
the tenth 9 Nov 2026 paste): the card is the runner's steps, the minutes
file and the plan's statuses, and the block planner reads it back."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t61_card as C  # noqa: E402
import t61_session as S  # noqa: E402
import t64_blocks as B  # noqa: E402


def _card(plan_text=None):
    return C.build(C.load_minutes(), plan_text if plan_text is not None else C.PLAN.read_text(),
                   "2026-11-09")


def test_the_cards_step_table_is_the_runners_steps_with_the_minutes_files_sums():
    m = C.load_minutes()
    got = B.card_numbers(_card())
    assert [n for n, _ in got["steps"]] == [s.name for s in S.T61.steps]
    assert dict(got["steps"]) == {n: float(sum(e["minutes"])) for n, e in m["steps"].items()}
    assert (got["setup_h"], got["teardown_h"]) == (m["setup_h"]["value"], m["teardown_h"]["value"])


def test_the_block_planner_reads_the_card_back_to_its_own_blocks():
    card = _card()
    n = B.card_numbers(card)
    blocks = B.plan_blocks(n["steps"], n["setup_h"], n["teardown_h"], boot=B.BOOT["P2"],
                           rate=B.SPOT_RATES["P2"])
    for i, b in enumerate(blocks, 1):
        assert f"| b{i} | {', '.join(f'`{x}`' for x in b.steps)} | {b.planned_h:.2f} | " \
               f"{b.cap_inr} | {b.halt_min} | {b.delete_min} |" in card


def test_a_minutes_file_off_the_runners_steps_is_refused(tmp_path):
    m = C.load_minutes()
    del m["steps"]["mask_gate"]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="T61's steps are"):
        C.load_minutes(p)


def test_the_prerequisites_follow_the_plan():
    plan = C.PLAN.read_text()
    done = "\n".join(f"| {r.task} | x | Done (9 Nov 2026) |" for r in S.T61.requirements)
    assert "**As the plan stands, 0 of" in _card(done)
    first = S.T61.requirements[0].task
    one_short = done.replace(f"| {first} | x | Done", f"| {first} | x | Partly built")
    assert f"| {first} | {S.T61.requirements[0].why} | Partly built (9 Nov 2026) | no |" in _card(one_short)
    assert "**As the plan stands, 1 of" in _card(one_short)
    assert plan          # the real plan builds too
