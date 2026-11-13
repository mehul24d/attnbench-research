"""The block spec in the lock file (`attnbench.lock.blocks`; researcher's
rulings, 7 Nov 2026): each block on its own boot, the order rules across
blocks, ground truth split with one anchor cell measured in every block."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from attnbench.lock import blocks as B  # noqa: E402
import t64_session as P4  # noqa: E402

NAMES = P4.T64.names
BOOT = P4.T64.boot_steps
A = {"arm": "dense", "density": 1.0, "context_length": 8192, "batch": 1}
C1 = {"arm": "xattention", "density": None, "context_length": 2048, "batch": 1}
C2 = {"arm": "oracle", "density": 0.25, "context_length": 16384, "batch": 1}
B1 = ["link", "launch_queue", "probe", "masks", "host_components"]
B2 = ["device_components", "dense_select", "host_speed_mid", "predict", "ground_truth",
      "late_check", "trace_checks"]
B3 = ["ground_truth", "late_check", "trace_checks"]


def lock(*blocks, anchor=A, max_hours=2.2):
    return {"blocks": {"P4": {"max_hours": max_hours, "anchor": anchor, "blocks": list(blocks)}}}


def good(**kw):
    return lock({"id": "b1", "steps": B1, "planned_hours": 1.9},
                {"id": "b2", "steps": B2, "cells": [C1], "planned_hours": 2.2},
                {"id": "b3", "steps": B3, "cells": [C2], "planned_hours": 1.9}, **kw)


def problems(lk, complete=True):
    return B.check(B.parse(lk, "P4"), NAMES, BOOT, complete=complete)


def test_the_planned_p4_spec_is_whole_and_each_block_runs_its_boot_steps():
    spec = B.parse(good(), "P4")
    assert B.check(spec, NAMES, BOOT) == []
    assert spec.run_steps("b3", NAMES, BOOT) == ("prereqs", "fa3_import", "preflight",
                                                 "ground_truth", "late_check", "trace_checks",
                                                 "host_speed_end")
    assert spec.predict_block() == "b2" and [b.id for b in spec.before("b3")] == ["b1", "b2"]
    assert spec.anchor == ("dense", 1.0, 8192, 1)
    assert B.parse(good(), "P4").sha256 == spec.sha256 != B.parse(good(max_hours=2.0), "P4").sha256


def _with(n, **kw):
    lk = good()
    lk["blocks"]["P4"]["blocks"][n].update(kw)
    return lk


@pytest.mark.parametrize("lk, why", [
    (_with(0, steps=["prereqs"] + B1), "names boot steps"),
    (_with(0, steps=B1[:-1]), "missing ['host_components']"),
    (_with(1, steps=["host_components"] + B2), "twice ['host_components']"),
    (_with(0, steps=list(reversed(B1))), "not in the session's order"),
    (_with(2, steps=["predict"] + B3), "predict is in 2 blocks"),
    (_with(1, steps=[s for s in B2 if s not in B3]), "ground truth follows predict"),
    (_with(2, steps=["ground_truth"], cells=[C2]), "without ['late_check', 'trace_checks']"),
    (_with(2, cells=[C1]), "is in blocks b2 and b3"),
    (_with(2, cells=[A]), "the anchor"),
    (_with(1, planned_hours=2.3), "above the block limit"),
    (_with(0, cells=[C2]), "names cells but runs no ground truth"),
    (_with(0, id="b2"), "block ids repeat"),
    (_with(0, id="Block One"), "is not lower-case"),
    (_with(0, steps=B1 + ["late_check"]), "with no ground truth"),
    (_with(2, steps=["nonsense"] + B3), "not this session's"),
])
def test_every_breach_of_the_order_rules_is_named(lk, why):
    got = problems(lk)
    assert any(why in p for p in got), got


def test_ground_truth_before_the_prediction_is_refused():
    lk = lock({"id": "b1", "steps": B1 + B3, "cells": [C2]},
              {"id": "b2", "steps": B2, "cells": [C1]})
    assert any("before the prediction is committed" in p for p in problems(lk))


def test_a_skeleton_without_cells_or_anchor_is_a_plan_and_not_a_run():
    lk = _with(1, cells=None)
    lk["blocks"]["P4"]["blocks"][2]["cells"] = None
    lk["blocks"]["P4"]["anchor"] = None
    assert problems(lk, complete=False) == []
    got = problems(lk)
    assert "b2: runs ground truth and names no cells" in got
    assert "ground truth in more than one block needs an anchor cell" in got


def test_a_lock_without_the_pairings_spec_or_a_malformed_one_is_refused():
    with pytest.raises(B.BlockSpecError, match="no block spec for P2"):
        B.parse(good(), "P2")
    with pytest.raises(B.BlockSpecError, match="malformed"):
        B.parse({"blocks": {"P4": {"blocks": [{"steps": []}]}}}, "P4")
    with pytest.raises(B.BlockSpecError, match="no block 'b9'"):
        B.parse(good(), "P4").block("b9")


def test_the_planners_skeleton_is_a_valid_plan_for_both_pairings():
    import t61_session as S
    import t64_blocks as TB
    for pairing, cfg in (("P4", P4.T64), ("P2", S.T61)):
        n = TB.card_numbers(TB.CARDS[pairing].read_text())
        # the cards' preflight before the adviser's rulings dated 13 Nov 2026, second set:
        # at its 12 minutes since, no block fits (tests/test_t64_blocks.py)
        n["steps"] = [(s, 8.0 if s == "preflight" else m) for s, m in n["steps"]]
        h = {"P2": 2.2, "P4": 2.3}[pairing]      # P4 at 2.2 h refuses (t64_card.py)
        blocks = TB.plan_blocks(n["steps"], n["setup_h"], n["teardown_h"],
                                boot=TB.BOOT[pairing], rate=TB.SPOT_RATES[pairing], max_hours=h)
        spec = B.parse({"blocks": {cfg.pairing: TB.spec(blocks, max_hours=h)}}, cfg.pairing)
        assert TB.BOOT[pairing] == cfg.boot_steps
        assert B.check(spec, cfg.names, cfg.boot_steps, complete=False) == [], pairing
