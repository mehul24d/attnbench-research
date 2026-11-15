"""P4's block spec in the lock (the researcher's decision of 15 Nov 2026: P4
on Flex-start and Spot in parallel, full scope): derived from the card's
planner and the locked grid, never typed in, and covering every grid cell
exactly once, the anchor in every ground-truth block."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import t31_steps as st  # noqa: E402
import t61_session as S  # noqa: E402
import t64_blocks as B  # noqa: E402
import t64_card as C  # noqa: E402
from t64_session import T64  # noqa: E402

from attnbench.lock import blocks as LB  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCK = json.loads((ROOT / "docs" / "lock_values.json").read_text())


def _grid():
    return st.cells(st.fill_lock_values(json.loads((ROOT / T64.components_config).read_text())))


def _planned():
    m = C.load_minutes()
    steps = [(s.name, float(sum(m["steps"][s.name]["minutes"]))) for s in T64.steps]
    blocks, _ = C.spot_blocks(steps, m["setup_h"]["value"], m["teardown_h"]["value"],
                              m["suite_record_min"]["value"])
    return blocks


def test_the_lock_holds_the_spec_the_card_and_the_grid_give():
    assert LOCK["blocks"]["P4"] == B.lock_spec(_planned(), _grid(), max_hours=C.SPOT_HOURS)
    assert set(LOCK["blocks"]) == {"P4"}            # P2 ran in one session


def test_the_spec_runs_the_p4_session_complete():
    spec = LB.parse(LOCK, "P4")
    assert LB.check(spec, T64.names, T64.boot_steps, complete=True) == []
    assert spec.ids == ("b1", "b2", "b3", "b4") and spec.predict_block() == "b3"


def test_every_grid_cell_is_measured_exactly_once_and_the_anchor_in_every_ground_truth_block():
    spec = LB.parse(LOCK, "P4")
    own = [c for b in spec.blocks if b.cells for c in b.cells]
    grid = [LB.cell_key(dict(zip(LB.CELL_FIELDS, c))) for c in _grid()]
    assert len(own) == len(set(own)) and spec.anchor not in own
    assert sorted(own + [spec.anchor], key=str) == sorted(grid, key=str)     # full scope
    assert len(grid) == 35


def test_each_ground_truth_block_passes_the_runners_grid_check():
    spec = LB.parse(LOCK, "P4")
    comp = st.fill_lock_values(json.loads((ROOT / T64.components_config).read_text()))
    for b in spec.blocks:
        if "ground_truth" in b.steps:
            run = [list(spec.anchor)] + [list(c) for c in b.cells]
            S.block_config(comp, {"id": b.id, "run_cells": run})   # raises if refused


def test_whole_length_batch_groups_go_to_a_block_by_their_share_of_the_minutes():
    blocks, grid = _planned(), _grid()
    got = B.assign_cells(blocks, grid, B.P4_ANCHOR)
    assert [sorted({(c[2], c[3]) for c in g}) for g in got] == [
        [(2048, 1), (4096, 1)], [(8192, 1), (16384, 1), (16384, 4), (32768, 1)]]
    import pytest
    with pytest.raises(ValueError, match="not a grid cell"):
        B.assign_cells(blocks, grid, {**B.P4_ANCHOR, "context_length": 1000})


def test_each_blocks_file_loads_through_the_real_lock_fill_to_its_own_cells(tmp_path):
    # the CPU pass of 15 Nov 2026: a block's file that set the grid was refused by the
    # lock fill in b3's ground truth; the file names the block's grid apart
    spec = LB.parse(LOCK, "P4")
    path = ROOT / T64.components_config
    raw = json.loads(path.read_text())
    comp = st.load_config(str(path), dry_run=False)
    for b in spec.blocks:
        if "ground_truth" not in b.steps:
            continue
        run = [list(spec.anchor)] + [list(c) for c in b.cells]
        f = S.block_file_config(raw, S.block_config(comp, {"id": b.id, "run_cells": run}))
        assert all(f.get(k) is None for k in st.LOCK_SOURCED)
        p = tmp_path / f"{b.id}.json"
        p.write_text(json.dumps(f))
        got = st.load_config(str(p), dry_run=False)
        assert sorted(map(tuple, st.cells(got)), key=str) == sorted(map(tuple, run), key=str)


def test_a_block_grid_outside_the_pairings_grid_is_refused(tmp_path):
    import pytest
    path = ROOT / T64.components_config
    f = dict(json.loads(path.read_text()), block_grid={"lengths": [1000], "batches": {"1000": [1]}})
    p = tmp_path / "x.json"
    p.write_text(json.dumps(f))
    with pytest.raises(st.StepFailed, match=r"outside the pairing's grid"):
        st.load_config(str(p), dry_run=False)
