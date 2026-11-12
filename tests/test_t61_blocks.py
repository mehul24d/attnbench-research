"""The session runner in blocks (researcher's rulings, 7 Nov 2026): each
block on its own boot and in its own directory, importing every earlier
block after checking it, ground truth on the block's cells and the anchor
with the block's id in every row, an interrupted block discarded alone, and
the between-block drift check in the analysis. Dry runs on CPU, through the
runner's own `main` per step (`test_t61_session.inproc`)."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t61_session as S  # noqa: E402
import t64_session as P4  # noqa: E402
from test_lock_blocks import A, B1, B2, B3, C1, C2  # noqa: E402
from test_t61_session import inproc  # noqa: E402
from test_t64_session import SHA  # noqa: E402

from attnbench.analysis import block_drift as BD  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

T64 = P4.T64
EXTRA = ["--fa3-sha256", SHA]


def write_lock(tmp_path, *, blocks=None, name="lock.json") -> Path:
    spec = {"max_hours": 2.2, "anchor": A, "blocks": blocks or [
        {"id": "b1", "steps": B1}, {"id": "b2", "steps": B2, "cells": [C1]},
        {"id": "b3", "steps": B3, "cells": [C2]}]}
    p = tmp_path / name
    p.write_text(json.dumps({"blocks": {"P4": spec}}))
    return p


def block(cfg, lock, bid, imports=(), dry_run=True):
    br, why = S.block_run(cfg, Namespace(lock=str(lock), block=bid,
                                         import_block=[str(d) for d in imports], dry_run=dry_run))
    assert why is None, why
    return br


def run_block(cfg, tmp_path, lock, bid, imports=(), calls=None, start=None):
    d = tmp_path / bid
    br = block(cfg, lock, bid, imports)
    rc = S.run_session(cfg, d, start=start or "prereqs", extra=EXTRA if cfg is T64 else [],
                       dry_run=True, sync=False, runner=inproc(cfg, calls), block=br)
    return rc, d


def gt_rows(cfg, d):
    return json.loads((S.step_dir(d, "ground_truth", cfg) / "ground_truth.json").read_text())


def three(tmp_path, monkeypatch=None):
    lock = write_lock(tmp_path)
    ran = {b: [] for b in ("b1", "b2", "b3")}
    rc1, d1 = run_block(T64, tmp_path, lock, "b1", calls=ran["b1"])
    rc2, d2 = run_block(T64, tmp_path, lock, "b2", [d1], calls=ran["b2"])
    rc3, d3 = run_block(T64, tmp_path, lock, "b3", [d1, d2], calls=ran["b3"])
    return lock, (rc1, rc2, rc3), (d1, d2, d3), ran


def test_three_blocks_run_in_turn_on_their_own_boots_and_the_prediction_is_made_once(tmp_path):
    lock, rcs, (d1, d2, d3), ran = three(tmp_path)
    assert rcs == (0, 0, 0)
    boot = ["prereqs", "preflight", "fa3_import"]
    assert ran["b1"] == boot[:2] + ["link", "launch_queue", "probe", "masks", "host_components",
                                    "fa3_import", "host_speed_end"]
    assert ran["b2"] == ["prereqs", "preflight", "device_components", "fa3_import",
                         "dense_select", "host_speed_mid", "predict", "ground_truth",
                         "late_check", "trace_checks", "host_speed_end"]
    assert ran["b3"] == boot + ["ground_truth", "late_check", "trace_checks", "host_speed_end"]
    m3 = json.loads((d3 / "manifest.json").read_text())
    assert m3["block"]["id"] == "b3" and set(m3["block"]["imported"]) == {"b1", "b2"}
    imported = {a["step"]: a["imported_from"] for a in m3["attempts"] if a.get("imported_from")}
    assert imported["predict"] == "b2" and imported["link"] == "b1"
    # one prediction, made in b2 and reused by b3; b3's late check reads it
    assert json.loads((d3 / S.RECEIPT).read_text()) == json.loads((d2 / S.RECEIPT).read_text())
    assert S.last_status(m3)["late_check"] == "passed"
    # the rows: the block's cells and the anchor, each with the block's id
    for d, bid, cell in ((d2, "b2", C1), (d3, "b3", C2)):
        rows = gt_rows(T64, d)["untraced"]["rows"]
        assert {r["block"] for r in rows} == {bid}
        assert sorted((r["arm"], r["anchor"]) for r in rows) == sorted(
            [(A["arm"], True), (cell["arm"], False)])


def test_the_drift_check_reads_every_block_and_names_one_that_moved(tmp_path, monkeypatch):
    lock = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock, "b1")
    _, d2 = run_block(T64, tmp_path, lock, "b2", [d1])
    monkeypatch.setenv("ATTNBENCH_FAKE_DRIFT", "0.08")
    _, d3 = run_block(T64, tmp_path, lock, "b3", [d1, d2])
    got = BD.collect([d1, d2, d3], ground_truth_dir=S.step_dir(Path(), "ground_truth", T64).name)
    assert got["blocks"] == ["b1", "b2", "b3"] and got["discarded"] == []
    assert len(got["rows"]) == 4 and got["code_commit"] == "dry"
    with pytest.raises(BD.BlockDriftError, match=r"\['b1'\] have no anchor row"):
        BD.drift(got["rows"], order=["b1", "b2", "b3"], bound=0.05)
    dr = BD.drift(got["rows"], order=["b2", "b3"], bound=0.05)       # b1 runs no ground truth
    assert dr["anchor"] == ("dense", 1.0, 8192, 1)
    assert dr["per_block"]["b3"]["relative_difference"] == pytest.approx(0.08)
    assert dr["per_block"]["b2"]["relative_difference"] == pytest.approx(-0.08 / 1.08)
    assert not dr["passed"] and dr["above"] == ["b2", "b3"]     # two blocks: either moved
    assert BD.drift(got["rows"], order=["b2", "b3"], bound=0.1)["passed"]
    from attnbench.lock import blocks as LB
    spec = LB.parse(json.loads(Path(lock).read_text()), "P4")
    post = BD.pairing_drift([d1, d2, d3], spec, bound=0.05,
                            ground_truth_dir=S.step_dir(Path(), "ground_truth", T64).name)
    assert post["drift"] == dr and post["sensitivity_without"] == ["b2", "b3"]


def test_p4s_post_session_step_reads_the_drift_from_the_lock(tmp_path, monkeypatch, capsys):
    """Adviser's ruling 2c of 11 Nov 2026: the drift check is P4's
    post-session step, with the lock's bound and spec; it flags, refuses a
    lock that is not the complete one, and never runs for the rehearsal."""
    import t31_steps as st
    import t61_rehearsal as R
    import lock_fixture as LF
    from attnbench.lock import checklist as CL
    lock_path = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock_path, "b1")
    _, d2 = run_block(T64, tmp_path, lock_path, "b2", [d1])
    monkeypatch.setenv("ATTNBENCH_FAKE_DRIFT", "0.08")
    _, d3 = run_block(T64, tmp_path, lock_path, "b3", [d1, d2])
    spec_json = json.loads(lock_path.read_text())["blocks"]
    lock = LF.complete_lock(CL.PLAN.read_text(encoding="utf-8"), blocks=spec_json)
    full = tmp_path / "lock_values.json"
    full.write_text(json.dumps(lock))
    monkeypatch.setattr(st, "LOCK_FILE", full)
    out = tmp_path / "block_drift.json"
    argv = ["drift", "--lock", str(full), "--out", str(out)] + sum(
        (["--block-dir", str(d)] for d in (d1, d2, d3)), [])
    bound = lock["values"]["block_drift_bound"]["value"]
    assert bound < 0.074
    assert P4.main(argv) == 0
    got = json.loads(out.read_text())
    assert got["pairing"] == "P4" and got["drift"]["bound"] == bound
    assert got["drift"]["above"] == ["b2", "b3"] and got["sensitivity_without"] == ["b2", "b3"]
    assert "above the bound" in capsys.readouterr().out
    assert P4.main(["drift", "--lock", str(lock_path), "--out", str(out),
                    "--block-dir", str(d2), "--block-dir", str(d3)]) == 2
    assert "complete lock" in capsys.readouterr().err
    unset = json.loads(json.dumps(lock))
    unset["values"]["block_drift_bound"]["value"] = None
    bad = tmp_path / "unset.json"
    bad.write_text(json.dumps(unset))
    assert P4.main(["drift", "--lock", str(bad), "--out", str(out),
                    "--block-dir", str(d2), "--block-dir", str(d3)]) == 2
    assert S.main(["drift", "--lock", str(full), "--out", str(out), "--block-dir", str(d2)],
                  cfg=R.REHEARSAL) == 2
    assert "rehearsal" in capsys.readouterr().err
    # one session (a single Flex grant): "not applicable", never "passed"
    assert P4.main(["drift", "--lock", str(full), "--out", str(out), "--block-dir", str(d1)]) == 0
    printed = capsys.readouterr().out
    assert "block drift: not applicable" in printed and "passed" not in printed
    assert json.loads(out.read_text())["drift"]["status"] == "not applicable"


def test_a_block_refuses_an_earlier_block_it_cannot_trust(tmp_path):
    lock, _, (d1, d2, d3), _ = three(tmp_path)
    args = lambda bid, imports, lk=lock: Namespace(lock=str(lk), block=bid, dry_run=True,  # noqa: E731
                                                   import_block=[str(d) for d in imports])
    _, why = S.block_run(T64, args("b3", [d2]))
    assert "needs ['b1', 'b2'], given ['b2']" in why
    _, why = S.block_run(T64, args("b2", [d1, d1]))
    assert "given twice" in why
    # every check before anything runs: discarded, another spec, another commit, a failed step
    (d1 / S.DISCARD).write_text("restarted")
    br = block(T64, lock, "b3", [d1, d2])
    assert S.run_session(T64, tmp_path / "x", start="prereqs", extra=EXTRA, dry_run=True,
                         sync=False, runner=inproc(T64), block=br) == 2
    assert not (tmp_path / "x" / "manifest.json").exists() or \
        not json.loads((tmp_path / "x" / "manifest.json").read_text()).get("block")
    (d1 / S.DISCARD).unlink()
    other = write_lock(tmp_path, name="other.json", blocks=[
        {"id": "b1", "steps": B1}, {"id": "b2", "steps": B2, "cells": [C2]},
        {"id": "b3", "steps": B3, "cells": [C1]}])
    assert any("not b1 of" in p for p in S.verify_import(T64, block(T64, other, "b3", [d1, d2]),
                                                          "b1", d1, True))
    m = json.loads((d1 / "manifest.json").read_text())
    (d1 / "manifest.json").write_text(json.dumps({**m, "code_commit": "abc"}))
    assert any("every block runs one commit" in p
               for p in S.verify_import(T64, block(T64, lock, "b3", [d1, d2]), "b1", d1, True))
    m["attempts"].append({"step": "probe", "status": "failed"})
    (d1 / "manifest.json").write_text(json.dumps(m))
    assert any("['probe'] did not pass" in p
               for p in S.verify_import(T64, block(T64, lock, "b3", [d1, d2]), "b1", d1, True))


def test_a_later_block_on_another_cpu_stops_after_its_preflight(tmp_path, monkeypatch):
    lock = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock, "b1")
    monkeypatch.setenv("ATTNBENCH_FAKE_CPU_MODEL", "another cpu")
    calls = []
    rc, d2 = run_block(T64, tmp_path, lock, "b2", [d1], calls=calls)
    assert rc == 1 and calls == ["prereqs", "preflight"]
    a = json.loads((d2 / "manifest.json").read_text())["attempts"][-1]
    assert a["step"] == "preflight" and a["status"] == "failed"
    assert "cpu_model is 'another cpu' here and 'fake cpu' in block b1" in a["reason"]


def test_an_interrupted_block_is_discarded_alone(tmp_path, monkeypatch):
    lock = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock, "b1")
    _, d2 = run_block(T64, tmp_path, lock, "b2", [d1])
    _, d3 = run_block(T64, tmp_path, lock, "b3", [d1, d2])
    # b3 as a real session that began on boot A and is resumed on boot B
    m = json.loads((d3 / "manifest.json").read_text())
    (d3 / "manifest.json").write_text(json.dumps({**m, "dry_run": False, "boot_id": "boot-A"}))
    monkeypatch.setattr(S, "boot_id", lambda: "boot-B")
    br = S.BlockRun(block(T64, lock, "b3", [d1, d2]).spec, "b3",
                    block(T64, lock, "b3", [d1, d2]).steps, (), "x")
    ran = []
    assert S.run_session(T64, d3, start="prereqs", extra=EXTRA, dry_run=False, sync=False,
                         runner=lambda cmd, log: ran.append(cmd) or 1, block=br) == 2
    assert ran == [] and "Block b3 of P4" in (d3 / S.DISCARD).read_text()
    got = BD.collect([d1, d2, d3], ground_truth_dir=S.step_dir(Path(), "ground_truth", T64).name)
    assert got["blocks"] == ["b1", "b2"] and [x["block"] for x in got["discarded"]] == ["b3"]
    assert {r["block"] for r in got["rows"]} == {"b2"}


def test_a_block_and_a_whole_session_never_share_a_directory(tmp_path):
    lock = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock, "b1")
    assert S.run_session(T64, d1, start="prereqs", extra=EXTRA, dry_run=True, sync=False,
                         runner=inproc(T64)) == 2
    whole = tmp_path / "whole"
    S.run_session(T64, whole, start="prereqs", extra=EXTRA, dry_run=True, sync=False,
                  runner=lambda c, l: 1)
    assert S.run_session(T64, whole, start="prereqs", extra=EXTRA, dry_run=True, sync=False,
                         runner=inproc(T64), block=block(T64, lock, "b1")) == 2
    assert S.run_session(T64, d1, start="predict", extra=EXTRA, dry_run=True, sync=False,
                         runner=inproc(T64), block=block(T64, lock, "b1")) == 2   # not b1's


def test_rows_that_are_not_the_blocks_cells_are_refused(tmp_path):
    block_ = {"id": "b2", "cells": [("xattention", None, 2048, 1)], "anchor": ("dense", 1.0, 8192, 1),
              "run_cells": [("dense", 1.0, 8192, 1), ("xattention", None, 2048, 1)]}
    good = {"untraced": {"rows": [dict(zip(("arm", "density", "context_length", "batch"), c))
                                  for c in block_["run_cells"]]}}
    S.tag_rows(good, block_)
    assert [r["anchor"] for r in good["untraced"]["rows"]] == [True, False]
    bad = {"untraced": {"rows": good["untraced"]["rows"][:1]}}
    with pytest.raises(S.StepFailed, match="not this block's"):
        S.tag_rows(bad, block_)


def test_the_p2_session_runs_in_the_same_blocks(tmp_path):
    lock = tmp_path / "p2.json"
    lock.write_text(json.dumps({"blocks": {"P2": {"max_hours": 2.2, "anchor": A, "blocks": [
        {"id": "b1", "steps": ["link", "launch_queue", "probe", "mask_gate", "host_components"]},
        {"id": "b2", "steps": B2, "cells": [C1]}, {"id": "b3", "steps": B3, "cells": [C2]}]}}}))
    T61 = S.T61
    rc1, d1 = run_block(T61, tmp_path, lock, "b1")
    rc2, d2 = run_block(T61, tmp_path, lock, "b2", [d1])
    rc3, d3 = run_block(T61, tmp_path, lock, "b3", [d1, d2])
    assert (rc1, rc2, rc3) == (0, 0, 0)
    assert {r["block"] for r in gt_rows(T61, d3)["untraced"]["rows"]} == {"b3"}


# --- the drift rule (researcher's ruling 3 of 7 Nov 2026) ------------------------------------------

def _anchor_rows(medians):
    rows = []
    for b, m in medians.items():
        rows.append({"block": b, "anchor": True, "arm": "dense", "nominal_density": 1.0,
                     "context_length": 8192, "batch": 1, "median_ms": m})
        rows.append({"block": b, "anchor": False, "arm": "oracle", "nominal_density": 0.5,
                     "context_length": 16384, "batch": 1, "median_ms": 2 * m})
    return rows


def test_the_drift_bound_is_required_and_read_from_the_lock():
    """Adviser's ruling 2c of 11 Nov 2026: no default bound; the caller reads
    T4.1's block_drift_bound from the lock (working 6.77% until then)."""
    assert BD.WORKING_BOUND == 0.0677
    rows = _anchor_rows({"b1": 100.0, "b2": 106.7, "b3": 93.2})
    with pytest.raises(TypeError):
        BD.drift(rows, order=["b1", "b2", "b3"])
    for bad in (None, True, 0.0, -0.1, float("nan")):
        with pytest.raises(BD.BlockDriftError, match="positive relative difference"):
            BD.drift(rows, order=["b1", "b2", "b3"], bound=bad)
    dr = BD.drift(rows, order=["b1", "b2", "b3"], bound=BD.WORKING_BOUND)
    # each against the median of the other two: b2 +10.5%, b3 -9.8%, b1 +0.05%
    assert dr["bound"] == 0.0677 and dr["above"] == ["b2", "b3"] and not dr["passed"]
    assert BD.drift(_anchor_rows({"b1": 100.0, "b2": 106.7}), order=["b1", "b2"],
                    bound=BD.WORKING_BOUND)["passed"]
    assert not BD.drift(_anchor_rows({"b1": 100.0, "b2": 106.8}), order=["b1", "b2"],
                        bound=BD.WORKING_BOUND)["passed"]          # strictly above 6.77%
    # "a block above it": a difference equal to the bound is not above it (binary-exact)
    assert BD.drift(_anchor_rows({"b1": 100.0, "b2": 106.25}), order=["b1", "b2"],
                    bound=0.0625)["above"] == []
    assert BD.lock_bound({"values": {"block_drift_bound": {"value": 0.0677}}}) == 0.0677
    for v in (None, "TBD", True, 0, 1.5):
        with pytest.raises(BD.BlockDriftError, match="block_drift_bound"):
            BD.lock_bound({"values": {"block_drift_bound": {"value": v}}})
    with pytest.raises(BD.BlockDriftError, match="block_drift_bound"):
        BD.lock_bound({"values": {}})


def test_the_reference_is_the_median_of_the_other_blocks():
    """A's F5c (T4.3 read A, finding 6c): the odd block is flagged, not the
    two that agree; and a block's own anchor never moves its reference."""
    dr = BD.drift(_anchor_rows({"b1": 100.0, "b2": 108.0, "b3": 108.0}),
                  order=["b1", "b2", "b3"], bound=BD.WORKING_BOUND)
    assert dr["above"] == ["b1"]
    assert dr["per_block"]["b1"]["reference"] == 108.0
    assert dr["per_block"]["b2"]["reference"] == 104.0


def test_a_block_without_an_anchor_or_with_nan_is_refused():
    with pytest.raises(BD.BlockDriftError, match="no anchor row"):
        BD.drift(_anchor_rows({"b1": 100.0}), order=["b1", "b2"], bound=BD.WORKING_BOUND)
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(BD.BlockDriftError, match="finite positive"):
            BD.drift(_anchor_rows({"b1": 100.0, "b2": bad}), order=["b1", "b2"],
                     bound=BD.WORKING_BOUND)


def test_a_block_above_the_bound_is_flagged_kept_in_scoring_and_given_a_sensitivity_run():
    rows = _anchor_rows({"b1": 100.0, "b2": 108.0, "b3": 101.0})
    dr = BD.drift(rows, order=["b1", "b2", "b3"], bound=BD.WORKING_BOUND)
    assert dr["status"] == "above the bound" and dr["passed"] is False
    labelled = BD.labelled(rows, dr)
    assert {r["block"] for r in labelled if r.get("drift_label") == BD.DRIFT_LABEL} == {"b2"}
    assert all("drift_label" not in r for r in labelled if r["block"] != "b2")
    assert BD.claimable(rows, dr) == rows                       # a flag, not an exclusion
    sens = BD.sensitivity_sets(rows, dr)
    assert list(sens) == ["b2"] and {r["block"] for r in sens["b2"]} == {"b1", "b3"}
    assert all("drift_label" not in r for r in rows)            # the input is not changed


def test_with_one_block_of_the_anchor_nothing_is_compared_and_nothing_is_dropped():
    rows = _anchor_rows({"b1": 100.0})
    dr = BD.drift(rows, order=["b1"], bound=BD.WORKING_BOUND)
    assert dr["passed"] is None and dr["status"] == "not applicable"   # never "passed"
    assert BD.claimable(rows, dr) == rows and BD.labelled(rows, dr) == rows
    assert BD.sensitivity_sets(rows, dr) == {}


def test_the_lock_table_states_the_working_bound_and_the_rule():
    table = (ROOT / "docs" / "lock_value_table.md").read_text()
    row = next(l for l in table.splitlines() if l.startswith("| 36 |"))
    assert "Block drift bound" in row and "Working: 6.77%" in row
    assert BD.DRIFT_LABEL in row and "never widened to pass" in row


def test_a_redone_block_reuses_its_discarded_prediction_and_a_differing_one_stops(tmp_path,
                                                                                  monkeypatch):
    """Ruling 1c (11 Nov 2026): the pairing's ledger holds one prediction per
    cell; a redone block re-predicts bit for bit and uses the original file,
    synced before any ground truth; a re-prediction that differs stops."""
    lock = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock, "b1")
    rc, d2 = run_block(T64, tmp_path, lock, "b2", [d1])
    assert rc == 0
    first = json.loads((d2 / S.RECEIPT).read_text())
    redo = tmp_path / "b2_redo"
    br = block(T64, lock, "b2", [d1])
    assert S.run_session(T64, redo, start="prereqs", extra=EXTRA, dry_run=True, sync=False,
                         runner=inproc(T64), block=br) == 0
    again = json.loads((redo / S.RECEIPT).read_text())
    assert again["re_predicted"] and again["sha256"] == first["sha256"]
    assert again["synced_at"] == first["synced_at"]
    fake = S.fake_record
    monkeypatch.setattr(S, "fake_record",
                        lambda *a, **k: {**fake(*a, **k), "note": "a different prediction"})
    assert S.run_session(T64, tmp_path / "b2_redo2", start="prereqs", extra=EXTRA, dry_run=True,
                         sync=False, runner=inproc(T64), block=br) == 1
    m = json.loads((tmp_path / "b2_redo2" / "manifest.json").read_text())
    assert S.last_status(m)["predict"] == "failed" and "ground_truth" not in S.last_status(m)


def test_a_real_block_runs_only_from_the_complete_lock_every_step_reads(tmp_path):
    """Ruling 1d (11 Nov 2026): `checklist.check` on the --lock file, and its
    block spec equal to `docs/lock_values.json`'s."""
    import lock_fixture as LF
    from attnbench.lock import blocks as LB
    from attnbench.lock import checklist as CL
    plan = CL.PLAN.read_text(encoding="utf-8")
    spec_json = json.loads(write_lock(tmp_path).read_text())["blocks"]
    lock = LF.complete_lock(plan, blocks=spec_json)
    held = tmp_path / "lock_values.json"
    held.write_text(json.dumps(lock))
    spec = LB.parse(lock, "P4")
    assert S.real_block_lock_problem(T64, lock, spec, held) is None
    other = json.loads(json.dumps(spec_json))
    other["P4"]["max_hours"] = 2.0
    held.write_text(json.dumps({**lock, "blocks": other}))
    assert "differs" in S.real_block_lock_problem(T64, lock, spec, held)
    unset = json.loads(json.dumps(lock))
    unset["values"]["mask_gate_bound"]["value"] = "TBD"
    assert "complete lock" in S.real_block_lock_problem(T64, unset, spec, held)
    assert "complete lock" in S.real_block_lock_problem(T64, {"blocks": spec_json}, spec, held)


def test_block_identity_refuses_a_field_missing_on_either_side(tmp_path):
    def pre(d, **f):
        out = S.step_dir(d, "preflight", T64)
        out.mkdir(parents=True)
        (out / "result.json").write_text(json.dumps({"findings": f}))
    full = {k: "x" for k in S.BLOCK_IDENTITY}
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    pre(a, **full)
    pre(b, **full)
    pre(c, **{**full, "zone": None})
    from types import SimpleNamespace
    assert S.block_identity_problems(T64, b, SimpleNamespace(imports=(("b1", a),))) == []
    assert S.block_identity_problems(T64, c, SimpleNamespace(imports=(("b1", a),))) \
        == ["zone is not recorded here"]
    assert S.block_identity_problems(T64, a, SimpleNamespace(imports=(("b1", c),))) \
        == ["zone is not recorded in block b1"]
    assert {"zone", "boot_image", "driver", "bootstrap_sha256"} <= set(S.BLOCK_IDENTITY)


def test_collect_pools_no_row_of_a_block_whose_late_check_did_not_pass(tmp_path):
    lock = write_lock(tmp_path)
    _, d1 = run_block(T64, tmp_path, lock, "b1")
    _, d2 = run_block(T64, tmp_path, lock, "b2", [d1])
    m = json.loads((d2 / "manifest.json").read_text())
    m["attempts"].append({"step": "late_check", "status": "failed"})
    (d2 / "manifest.json").write_text(json.dumps(m))
    got = BD.collect([d1, d2], ground_truth_dir=S.step_dir(Path(), "ground_truth", T64).name)
    assert got["rows"] == [] and got["blocks"] == ["b1"]
    assert got["unpassed"] == [{"block": "b2", "dir": str(d2), "steps": ["late_check"]}]


def test_a_rehearsal_block_runs_from_its_own_spec_and_p4s_does_not(tmp_path):
    """The rehearsal is exploratory (its rows are never evidence) and runs
    from its own block spec, not a T4.1 lock; a P4 block from a spec alone
    is refused (ruling 1d)."""
    import t61_rehearsal as R
    br, why = S.block_run(R.REHEARSAL, Namespace(lock=str(S.ROOT / R.LOCK), block="b1",
                                                 dry_run=False, import_block=[]))
    assert why is None and br is not None
    spec_only = write_lock(tmp_path)
    _, why = S.block_run(T64, Namespace(lock=str(spec_only), block="b1", dry_run=False,
                                        import_block=[]))
    assert "complete lock" in why


@pytest.mark.parametrize("pairing", ["P2", "P4"])
def test_a_real_pairing_marked_exploratory_cannot_skip_the_lock(tmp_path, pairing):
    """Adviser's ruling 3 dated 12 Nov 2026: the skip is keyed on the
    exploratory flag and refused for P2 and P4 by name, at the block spec and
    at the training file's lock check."""
    import dataclasses
    import t61_rehearsal as R
    base = {"P2": S.T61, "P4": T64}[pairing]
    cfg = dataclasses.replace(base, exploratory="marked exploratory by mistake")
    assert S.skips_lock(R.REHEARSAL) and not S.skips_lock(base)
    for lock in (S.ROOT / R.LOCK, write_lock(tmp_path)):
        for dry in (False, True):
            br, why = S.block_run(cfg, Namespace(lock=str(lock), block="b1", dry_run=dry,
                                                 import_block=[]))
            assert br is None and f"{pairing} is a real pairing" in why
    f = tmp_path / "p1_training.json.gz"
    f.write_bytes(b"not read: refused first")
    with pytest.raises(S.StepFailed, match=f"{pairing} is a real pairing"):
        S.training_rows(cfg, path=f)
