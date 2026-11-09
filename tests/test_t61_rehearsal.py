"""The L4 rehearsal of the P2 runner (`scripts/t61_rehearsal.py`): P2's steps
and order, its two blocks and their anchor, the interrupted block discarded
alone, the exploratory marker, the same-card mask comparison, the working
learned configuration, the compiled run's refusals and the trace sizes.
Dry runs on CPU."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t61_rehearsal as R  # noqa: E402
import t61_session as S  # noqa: E402
from test_t61_session import inproc  # noqa: E402

from attnbench.analysis import block_drift as BD  # noqa: E402
from attnbench.lock import blocks as B  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REH = R.REHEARSAL
LOCK = ROOT / R.LOCK
BANKED: list = []          # a dry run takes no banking time (as P2's dry runs)


def block(bid, imports=(), dry_run=True):
    br, why = S.block_run(REH, Namespace(lock=str(LOCK), block=bid,
                                         import_block=[str(d) for d in imports], dry_run=dry_run))
    assert why is None, why
    return br


def run_block(tmp_path, bid, imports=(), calls=None, name=None):
    d = tmp_path / (name or bid)
    rc = S.run_session(REH, d, start="prereqs", extra=BANKED, dry_run=True, sync=False,
                       runner=inproc(REH, calls), block=block(bid, imports))
    return rc, d


def test_the_rehearsal_runs_p2s_steps_in_p2s_order_on_the_l4():
    assert REH.names == S.T61.names
    assert REH.gpu == "L4" and REH.instance_type == "g2-standard-8"
    assert REH.pairing == "P2-R" and REH.banked == ("P1",)
    for step in REH.names:                       # every body but four is P2's own
        if step not in ("prereqs", "mask_gate", "predict"):
            assert REH.reals[step] is S.T61.reals[step] or \
                REH.reals[step].func is S.T61.reals[step].func, step


def test_the_block_spec_is_complete_for_a_real_run_and_splits_the_grid_by_length():
    import t31_steps as st
    spec = B.parse(json.loads(LOCK.read_text()), R.PAIRING)
    assert B.check(spec, REH.names, REH.boot_steps, complete=True) == []
    comp = st.load_config(str(ROOT / REH.components_config), dry_run=False)
    grid = {tuple(c) for c in st.cells(comp)}
    owned = [set(b.cells) for b in spec.blocks]
    assert owned[0] | owned[1] | {spec.anchor} == grid and not owned[0] & owned[1]
    assert spec.anchor == ("dense", 1.0, 8192, 1)
    for b in spec.blocks:                        # each block's config passes the control rule
        run = [spec.anchor, *b.cells]
        sub = S.block_config(comp, {"id": b.id, "anchor": spec.anchor, "cells": list(b.cells),
                                    "run_cells": run})
        assert {tuple(c) for c in st.cells(sub)} == set(run)


def test_the_two_blocks_run_in_turn_and_every_session_is_marked_exploratory(tmp_path):
    ran = {"b1": [], "b2": []}
    rc1, d1 = run_block(tmp_path, "b1", calls=ran["b1"])
    rc2, d2 = run_block(tmp_path, "b2", [d1], calls=ran["b2"])
    assert (rc1, rc2) == (0, 0)
    assert ran["b2"] == ["prereqs", "preflight", "ground_truth", "late_check", "trace_checks",
                         "host_speed_end"]
    for d in (d1, d2):
        m = json.loads((d / "manifest.json").read_text())
        assert m["exploratory"] == R.EXPLORATORY_NOTE
        assert "no row here is evidence" in (d / S.EXPLORATORY).read_text()
    pre = json.loads((S.step_dir(d1, "prereqs", REH) / "result.json").read_text())
    assert pre["status"] == S.PASSED and "p2_would_refuse" in pre["findings"]
    pred = json.loads((S.step_dir(d1, "predict", REH) / "result.json").read_text())
    assert "configuration 4" in pred["findings"]["learned_config"]
    rec = json.loads((d1 / S.RECEIPT).read_text())
    assert rec["remote_path"].startswith("predictions/P2-R/")


def test_an_interrupted_b2_is_discarded_alone_and_its_rerun_stands(tmp_path, monkeypatch):
    _, d1 = run_block(tmp_path, "b1")
    _, x = run_block(tmp_path, "b2", [d1], name="b2_interrupted")
    m = json.loads((x / "manifest.json").read_text())
    (x / "manifest.json").write_text(json.dumps({**m, "dry_run": False, "boot_id": "boot-1"}))
    monkeypatch.setattr(S, "boot_id", lambda: "boot-2")
    ran = []
    assert S.run_session(REH, x, start="ground_truth", extra=BANKED, dry_run=False, sync=False,
                         runner=lambda cmd, log: ran.append(cmd) or 1,
                         block=block("b2", [d1])) == 2
    assert ran == [] and "Block b2 of P2-R" in (x / S.DISCARD).read_text()
    monkeypatch.undo()
    _, d2 = run_block(tmp_path, "b2", [d1])
    got = BD.collect([d1, x, d2], ground_truth_dir=S.step_dir(Path(), "ground_truth", REH).name)
    assert got["blocks"] == ["b1", "b2"] and [r["block"] for r in got["discarded"]] == ["b2"]
    res = BD.drift(got["rows"], order=["b1", "b2"])
    assert res["anchor"] == ("dense", 1.0, 8192, 1) and set(res["per_block"]) == {"b1", "b2"}


def test_the_mask_step_compares_p1s_masks_with_this_cards_torch_path(tmp_path, monkeypatch):
    seen = {}

    def banked(base):
        seen["base"] = base
        return [(2048, _bank_record("xattention", 2048))]

    def record(args, cells, out, paths):
        seen["paths"], seen["cells"] = paths, cells
        return {"torch": [_bank_record("xattention", 2048, dropped=1)]}
    monkeypatch.setattr(S, "banked_masks", banked)
    args = Namespace(session=str(tmp_path), out=str(tmp_path / "mg"))
    f = R.rehearsal_mask_gate(REH, args, record=record)
    assert seen["paths"] == ("torch",) and seen["base"] == tmp_path / S.P1_MASKS
    assert seen["cells"] == [("xattention", 2048, 1, "qa_1_2048_0")]
    assert f["verdict"] == "pass" and f["bound"] == 0.01
    assert "Triton" in f["not_rehearsed"]
    out = json.loads((tmp_path / "mg" / "mask_gate.json").read_text())
    assert out["rehearsal"] is True and out["bounds"]["max_density_difference"] == 0.01
    # beyond the bound on one card: the verdict says so, and the step still passes
    monkeypatch.setattr(S, "banked_masks", lambda base: [(2048, _bank_record("xattention", 2048))])
    f = R.rehearsal_mask_gate(REH, args, record=lambda a, c, o, paths: {
        "torch": [_bank_record("xattention", 2048, dropped=7)]})
    assert f["verdict"] == "fail" and f["failures"] >= 1


def _bank_record(arm, length, *, dropped=0):
    """A one-layer record on the L4's PyTorch path: 16 x 16 causal blocks
    (136 valid), `dropped` of them dropped, so the density is 1 - dropped/136."""
    import numpy as np
    from test_mask_gate import L4_TORCH, M, R
    m = np.tril(np.ones((1, 1, 16, 16), dtype=bool))
    rows, cols = np.tril_indices(16, -1)
    m[0, 0, rows[:dropped], cols[:dropped]] = False
    return R(L4_TORCH, length, 1, f"qa_1_{length}_0", (M(m),))


def test_the_compiled_run_needs_b1s_prediction_and_ground_truth(tmp_path):
    _, d1 = run_block(tmp_path, "b1")
    calls = []

    def harness(cfg, ns):
        calls.append((cfg.components_config, ns.session))
        return {"complete": True, "untraced": {"rows": []}}
    assert R.compiled(Namespace(session=str(d1), out=str(tmp_path / "c")), harness=harness) == 0
    assert calls == [(R.COMPILED_CONFIG, str(d1))]
    assert json.loads((tmp_path / "c" / "result.json").read_text())["status"] == S.PASSED
    _, d2 = run_block(tmp_path, "b2", [d1])
    assert R.compiled(Namespace(session=str(d2), out=str(tmp_path / "c2")), harness=harness) == 2
    assert R.compiled(Namespace(session=str(d1), out=str(tmp_path / "c3")),
                      harness=lambda c, n: {"complete": False}) == 1


def test_the_compiled_config_runs_the_two_compiled_groups_at_2048_and_no_eager():
    import t31_steps as st
    comp = st.load_config(str(ROOT / R.COMPILED_CONFIG), dry_run=False)
    assert comp["lengths"] == [2048] and comp["gt_groups"]["eager"] == []
    assert comp["gt_groups"]["compiled_default"] and comp["gt_groups"]["compiled_capture"]
    eager = st.load_config(str(ROOT / REH.components_config), dry_run=False)
    assert eager["gt_groups"]["compiled_default"] == [] == eager["gt_groups"]["compiled_capture"]
    assert eager["reps"] == st.load_config(str(ROOT / "configs/t31/p1_session.json"),
                                           dry_run=False)["reps"]       # protocol unchanged


def test_trace_sizes_reads_every_kineto_trace_by_arm_and_cell(tmp_path):
    t = tmp_path / "12_ground_truth" / "eager" / "traces"
    t.mkdir(parents=True)
    (t / "trace_L2048_b1_0_dense.json").write_text("x" * 10)
    (t / "trace_L2048_b1_1_oracle_d0.25.json").write_text("x" * 30)
    (t / "notes.json").write_text("{}")
    res = R.trace_sizes(tmp_path)
    assert res["total_bytes"] == 40 and len(res["traces"]) == 2
    assert res["bytes_by_arm_and_cell"] == {"dense@L2048_b1": 10, "oracle_d0.25@L2048_b1": 30}


@pytest.mark.parametrize("name", ["t61_rehearsal_a.sh", "t61_rehearsal_b.sh",
                                  "t61_rehearsal_interrupt.sh", "t61_rehearsal_watch.sh"])
def test_the_rehearsal_scripts_parse(name):
    # the Mac's bash 3.2 reads an apostrophe inside ${VAR:?message} as a quote
    import subprocess
    p = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_the_instance_a_chain_runs_the_compiled_stage_only_when_asked():
    text = (ROOT / "scripts" / "t61_rehearsal_a.sh").read_text()
    assert 'COMPILED="${ATTNBENCH_REH_COMPILED:-0}"' in text
    assert 'if [[ "$COMPILED" == 1 ]]; then' in text
