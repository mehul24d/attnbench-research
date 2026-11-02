"""The T3.1 debug session (researcher's ruling, 2 Nov 2026): steps 5 or 6 to
11 on staged step-5 cells, every output labelled "debug, not T3.1 data", the
stuck cell not measured, every independent step run after a failure, and a
cap on the session's time.

The donor here is built by hand (a manifest, a config of record, cells with
recordings pinned by hash); a test against session 4's banked pull runs where
that pull exists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import t31_debug as dbg  # noqa: E402
import t31_session as sess  # noqa: E402
import t31_steps as st  # noqa: E402
import test_t31_session as ts  # noqa: E402
import test_t31_wiring as tw  # noqa: E402

ORDER = ts.ORDER
IDENT = tw.IDENTITY | {"git_commit": "d" * 40}
INST = dict(tw.INSTANCE, mem_total_gib=31.3)
BANKED = (ROOT / "results" / "t31_p1_s4" / "results" / "t31_p1_20261102T1208Z")


def _cfg(**kw):
    return dict(tw._grid(), lengths=[2048, 32768], **kw)


def _donor(tmp_path, cfg, *, missing=(), steps=("preflight", "link", "dense_select", "probe"),
           failed=None):
    """A donor session: steps 1 to 4 passed, every cell of the grid passed except
    `missing` (cell directory names)."""
    d = tmp_path / "donor_s"
    attempts = []
    for n, s in enumerate(steps, 1):
        st.step_dir(d, s).mkdir(parents=True)
        (st.step_dir(d, s) / "step.log").write_text(f"{s}\n")
        attempts.append({"step": s, "n": n, "status": "passed", "reason": "", "exit": 0,
                         "started": "t", "seconds": 1.0, "dir": st.step_dir(d, s).name,
                         "log": ""})
    (st.step_dir(d, "preflight") / "session_config.json").write_text(json.dumps(cfg))
    hc = st.step_dir(d, "host_components")
    hc.mkdir()
    (hc / st.PRICE_CACHE).write_bytes(b"cache")
    for c in st.cells(cfg):
        cd = st.cell_dir(hc, *c)
        if cd.name in missing:
            continue
        (cd / "recording").mkdir(parents=True)
        files = {}
        for name in ("trace.json", "sequence.json", "tape.pt"):
            (cd / "recording" / name).write_bytes(f"{cd.name}:{name}".encode())
            files[name] = dbg.sha256(cd / "recording" / name)
        (cd / "recording" / "recording.json").write_text(json.dumps({"files": files}))
        (cd / "isolated_host.events.jsonl").write_text("big\n")
        (cd / st.CELL_INDEX).write_text(json.dumps({"arm": c[0], "context_length": c[2],
                                                    "batch": c[3]}))
        (cd / st.CELL_MANIFEST).write_text(json.dumps({
            "cell": dict(zip(("arm", "density", "length", "batch"), c)), "identity": IDENT,
            "instance": INST, "inputs": None, "status": "passed", "peak_rss_mb": 900.0}))
    attempts.append({"step": "host_components", "n": 5, "status": failed or "failed",
                     "reason": "killed", "exit": 1, "started": "t", "seconds": 5.0,
                     "dir": "05_host_components", "log": ""})
    (d / "manifest.json").write_text(json.dumps({"attempts": attempts, "dry_run": False,
                                                 "rehearsal": False}))
    return d


def _names(cfg):
    return [st.cell_dir(Path("."), *c).name for c in st.cells(cfg)]


STUCK = "oracle_d0.5_L32768_b1"


def _full_cfg():
    return _cfg(arms=["dense", "oracle", "meanpool"],
                densities={"dense": [1.0], "oracle": [0.25, 0.5], "meanpool": [0.25]},
                gt_groups={"eager": ["dense", "oracle", "meanpool"]})


# --- not measured, and the debug label -----------------------------------------------------

def test_a_not_measured_cell_is_in_no_grid_and_no_ground_truth_group():
    cfg = _full_cfg()
    assert ("oracle", 0.5, 32768, 1) in st.cells(cfg)
    cfg2 = dbg.debug_config(cfg)
    assert ("oracle", 0.5, 32768, 1) not in st.cells(cfg2)
    assert ("oracle", 0.5, 2048, 1) in st.cells(cfg2)
    names, dens = st.group_arms(cfg2, "eager", 32768, 1)
    assert ("oracle", 0.5) not in list(zip(names, dens)) and ("oracle", 0.25) in zip(names, dens)
    names, dens = st.group_arms(cfg2, "eager", 2048, 1)
    assert ("oracle", 0.5) in zip(names, dens)
    assert st.not_measured(cfg2)[("oracle", 0.5, 32768, 1)].startswith("not measured")


def test_the_debug_config_is_the_donors_with_its_own_prefix_and_never_a_rehearsal():
    cfg = dbg.debug_config(dict(_full_cfg(), rehearsal=True))
    assert cfg["debug"] is True and cfg["rehearsal"] is False
    assert cfg["prediction_prefix"] == dbg.DEBUG_PREFIX == "predictions/P1-debug"
    assert dbg.debug_config(cfg) == cfg                            # naming the cell twice names it once
    with pytest.raises(st.StepFailed, match="-debug"):
        st.prediction_dir(dict(cfg, prediction_prefix="predictions/P1"), Path("s"))
    with pytest.raises(st.StepFailed, match="-debug"):
        st.prediction_dir(dict(_full_cfg(), prediction_prefix="predictions/P1-debug"), Path("s"))
    assert st.prediction_dir(cfg, Path("s")) == "predictions/P1-debug/s"


def test_a_debug_steps_result_says_it_is_not_t31_data(tmp_path):
    out = tmp_path / "o"
    args = sess_args(out, tmp_path)
    cfgp = tmp_path / "c.json"
    cfgp.write_text(json.dumps(dict(st.DRY_CONFIG, debug=True, prediction_prefix="predictions/P1-debug")))
    args.config = str(cfgp)
    st.run_step("postflight", args)
    assert st.DEBUG_LABEL == "debug, not T3.1 data"
    assert json.loads((out / "result.json").read_text()).get("debug") == st.DEBUG_LABEL
    out2 = tmp_path / "o2"
    st.run_step("postflight", sess_args(out2, tmp_path))
    assert "debug" not in json.loads((out2 / "result.json").read_text())


def sess_args(out, session):
    from types import SimpleNamespace
    return SimpleNamespace(out=str(out), session=str(session), dry_run=True, cpu_boundary=False,
                           config=None, step="postflight")


# --- staging ---------------------------------------------------------------------------------

def test_staging_copies_the_passed_cells_verified_and_records_the_donor(tmp_path):
    cfg = _full_cfg()
    donor = _donor(tmp_path, cfg, missing=(STUCK, "meanpool_d0.25_L32768_b1"))
    s = tmp_path / "debug_s"
    rec = dbg.stage(donor, s, start="host_components", donor_source="<instance>/results/x")
    hc = st.step_dir(s, "host_components")
    on_disk = sorted(p.name for p in hc.iterdir() if p.is_dir())
    assert on_disk == sorted(c["dir"] for c in rec["cells"]) and STUCK not in on_disk
    assert len(on_disk) == len(st.cells(dbg.debug_config(cfg))) - 1       # meanpool 32768 still to run
    assert rec["run_in_this_session"] == ["meanpool_d0.25_L32768_b1"]
    assert not list(hc.rglob("isolated_host.events.jsonl"))               # not staged
    for c in rec["cells"]:
        assert c["cell_index_sha256"] == dbg.sha256(hc / c["dir"] / st.CELL_INDEX)
        assert c["trace_sha256"] == dbg.sha256(hc / c["dir"] / "recording" / "trace.json")
    assert rec["donor"]["identity"] == IDENT and rec["donor"]["instance"] == INST
    assert rec["donor"]["session"] == "donor_s" and rec["debug"] == st.DEBUG_LABEL
    assert [e["reason"] for e in rec["not_measured"]] == [dbg.STUCK_CELL["reason"]]
    for p in ("01_preflight", "02_link", "03_dense_select", "04_probe"):
        assert (s / p / "step.log").exists()
    assert (hc / st.PRICE_CACHE).read_bytes() == b"cache"
    assert "debug, not t3.1 data" in (s / dbg.DEBUG_NOTE).read_text().lower()
    m = json.loads((s / "manifest.json").read_text())
    assert m["debug"] == st.DEBUG_LABEL and m["dry_run"] is False and m["rehearsal"] is False
    assert [(a["step"], a["status"], a["staged"]) for a in m["attempts"]] == [
        (x, "passed", True) for x in ORDER[:4]]
    assert json.loads((s / dbg.DEBUG_CONFIG).read_text())["debug"] is True
    assert json.loads((s / st.STAGED_INPUTS).read_text()) == rec


def test_staging_for_step_six_names_every_cell_the_donor_never_ran_as_not_measured(tmp_path):
    cfg = _full_cfg()
    donor = _donor(tmp_path, cfg, missing=(STUCK, "meanpool_d0.25_L32768_b1"))
    s = tmp_path / "debug_s"
    rec = dbg.stage(donor, s, start="device_components")
    reasons = {(e["arm"], e["density"], e["length"]): e["reason"] for e in rec["not_measured"]}
    assert set(reasons) == {("oracle", 0.5, 32768), ("meanpool", 0.25, 32768)}
    assert "step 6" in reasons[("meanpool", 0.25, 32768)] and rec["run_in_this_session"] == []
    cfg2 = json.loads((s / dbg.DEBUG_CONFIG).read_text())
    hc = st.step_dir(s, "host_components")
    index = json.loads((hc / "cells_index.json").read_text())["cells"]
    assert len(index) == len(st.cells(cfg2)) == len(rec["cells"])
    m = json.loads((s / "manifest.json").read_text())
    assert m["attempts"][-1]["step"] == "host_components" and m["attempts"][-1]["staged"]
    assert json.loads((hc / "result.json").read_text())["debug"] == st.DEBUG_LABEL


def test_staging_refuses_what_cannot_be_trusted(tmp_path):
    cfg = _full_cfg()
    donor = _donor(tmp_path, cfg, missing=(STUCK,))
    with pytest.raises(SystemExit, match="starts at one of"):
        dbg.stage(donor, tmp_path / "a", start="predict")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        dbg.stage(donor, busy)
    # a cell whose recording no longer matches its pinned hash
    hc = st.step_dir(donor, "host_components")
    (hc / "dense_d1_L2048_b1" / "recording" / "trace.json").write_text("changed")
    with pytest.raises(SystemExit, match="differs from its pinned hashes"):
        dbg.stage(donor, tmp_path / "b")
    (hc / "dense_d1_L2048_b1" / "recording" / "trace.json").write_text("dense_d1_L2048_b1:trace.json")
    # steps 1 to 4 must all have passed
    m = json.loads((donor / "manifest.json").read_text())
    m["attempts"][3]["status"] = "failed"
    (donor / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="steps 1 to 4 did not all pass"):
        dbg.stage(donor, tmp_path / "c")
    # a rehearsal, a dry run or a debug session is never the donor
    for key in ("rehearsal", "dry_run", "debug"):
        m["attempts"][3]["status"] = "passed"
        (donor / "manifest.json").write_text(json.dumps(dict(m, **{key: True})))
        with pytest.raises(SystemExit, match="real, non-rehearsal, non-debug"):
            dbg.stage(donor, tmp_path / f"d_{key}")


def test_cells_of_two_identities_are_never_staged_together(tmp_path):
    cfg = _full_cfg()
    donor = _donor(tmp_path, cfg, missing=(STUCK,))
    mp = st.step_dir(donor, "host_components") / "dense_d1_L2048_b1" / st.CELL_MANIFEST
    m = json.loads(mp.read_text())
    m["identity"] = dict(IDENT, git_commit="e" * 40)
    mp.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="more than one code identity"):
        dbg.stage(donor, tmp_path / "s")


# --- the staged cells resume, and only those ----------------------------------------------------

def _debug_step5(tmp_path, monkeypatch, *, missing=(STUCK, "meanpool_d0.25_L32768_b1")):
    cfg = _full_cfg()
    donor = _donor(tmp_path, cfg, missing=missing)
    s = tmp_path / "debug_s"
    rec = dbg.stage(donor, s, start="host_components")
    monkeypatch.setattr(st, "cell_identity", lambda: dict(tw.IDENTITY))
    monkeypatch.setattr(st, "instance_spec", lambda: dict(tw.INSTANCE, gpu="NVIDIA L4"))
    ran = []

    def spawn(args, cfg, cell, cd, *, ceiling_mb):
        ran.append(cell)
        (cd / st.CELL_INDEX).write_text(json.dumps({"arm": cell[0], "context_length": cell[2]}))
        return {"returncode": 0, "seconds": 0.0, "sampled_peak_rss_mb": 1.0,
                "rss_ceiling_mb": ceiling_mb, "over_ceiling": False}
    from types import SimpleNamespace
    args = SimpleNamespace(session=str(s), out=str(st.step_dir(s, "host_components")))
    return s, rec, args, spawn, ran, json.loads((s / dbg.DEBUG_CONFIG).read_text())


def test_a_debug_session_runs_only_the_cells_the_donor_did_not_pass(tmp_path, monkeypatch):
    s, rec, args, spawn, ran, cfg = _debug_step5(tmp_path, monkeypatch)
    out = st.guarded_cells(args, cfg, st.cells(cfg), spawn=spawn)
    assert ran == [("meanpool", 0.25, 32768, 1)]
    assert len(out["resumed"]) == len(rec["cells"]) and len(out["index"]) == len(st.cells(cfg))
    # the staged cells keep their donor's manifests
    m = json.loads((st.cell_dir(Path(args.out), "dense", 1.0, 2048, 1) / st.CELL_MANIFEST).read_text())
    assert m["identity"] == IDENT


def test_a_cell_is_staged_only_if_its_entry_is_the_one_staged(tmp_path, monkeypatch):
    s, rec, args, spawn, ran, cfg = _debug_step5(tmp_path, monkeypatch)
    cd = st.cell_dir(Path(args.out), "dense", 1.0, 2048, 1)
    (cd / st.CELL_INDEX).write_text(json.dumps({"arm": "dense", "edited": True}))
    st.guarded_cells(args, cfg, st.cells(cfg), spawn=spawn)
    assert ("dense", 1.0, 2048, 1) in ran                      # priced again, not trusted


def test_a_cell_the_staging_did_not_list_is_run_even_under_the_donors_identity(tmp_path, monkeypatch):
    s, rec, args, spawn, ran, cfg = _debug_step5(tmp_path, monkeypatch)
    cd = st.cell_dir(Path(args.out), "meanpool", 0.25, 32768, 1)       # not staged: the donor never ran it
    cd.mkdir(parents=True)
    (cd / st.CELL_INDEX).write_text("{}")
    (cd / st.CELL_MANIFEST).write_text(json.dumps({
        "identity": IDENT, "instance": INST, "inputs": None, "status": "passed"}))
    st.guarded_cells(args, cfg, st.cells(cfg), spawn=spawn)
    assert ran == [("meanpool", 0.25, 32768, 1)]


def test_a_cell_of_another_identity_resumes_only_in_a_debug_session(tmp_path, monkeypatch):
    s, rec, args, spawn, ran, cfg = _debug_step5(tmp_path, monkeypatch)
    plain = dict(cfg, debug=False, prediction_prefix="predictions/P1")
    plain.pop("not_measured")
    st.guarded_cells(args, plain, st.cells(plain), spawn=spawn)
    assert len(ran) == len(st.cells(plain))                    # the donor's identity is not this one's


def test_a_debug_session_without_staged_inputs_is_refused(tmp_path, monkeypatch):
    s, rec, args, spawn, ran, cfg = _debug_step5(tmp_path, monkeypatch)
    (s / st.STAGED_INPUTS).unlink()
    with pytest.raises(st.StepFailed, match=st.STAGED_INPUTS):
        st.guarded_cells(args, cfg, st.cells(cfg), spawn=spawn)


# --- the runner -----------------------------------------------------------------------------------

def _staged_session(tmp_path, start="host_components"):
    donor = _donor(tmp_path, _full_cfg(), missing=(STUCK,))
    s = tmp_path / "dbg"
    dbg.stage(donor, s, start=start)
    return s


def _drun(s, outcomes=None, start="host_components", **kw):
    ran = []
    rc = sess.run_session(s, start=start, extra=[], dry_run=False, sync=False, debug=True,
                          runner=ts.fake_runner(outcomes or {}, ran), **kw)
    return rc, ran


def test_keep_going_runs_every_step_whose_inputs_did_not_fail(tmp_path):
    s = _staged_session(tmp_path)
    rc, ran = _drun(s, {"device_components": ("failed", 1)}, keep_going=True)
    assert rc == 1
    # predict, hold, ground truth and verify need step 6; the postflight does not
    assert ran == ["host_components", "device_components", "postflight"]
    last = {a["step"]: a["status"] for a in json.loads((s / "manifest.json").read_text())["attempts"]}
    assert [last[x] for x in ORDER[4:]] == ["passed", "failed", "blocked", "blocked", "blocked",
                                            "blocked", "passed"]


def test_keep_going_after_a_late_failure_runs_what_does_not_need_it(tmp_path):
    s = _staged_session(tmp_path)
    rc, ran = _drun(s, {"ground_truth": ("failed", 1)}, keep_going=True)
    assert rc == 1 and ran == list(ORDER[4:9]) + ["postflight"]
    rc, ran = _drun(_staged_session(tmp_path / "x"), {"predict": ("failed", 1)}, keep_going=True)
    assert rc == 1 and ran == ["host_components", "device_components", "predict", "postflight"]


def test_without_keep_going_a_debug_session_stops_at_the_first_failure(tmp_path):
    rc, ran = _drun(_staged_session(tmp_path), {"device_components": ("failed", 1)})
    assert rc == 1 and ran == ["host_components", "device_components"]


def test_a_clean_debug_session_runs_every_step_from_where_it_starts(tmp_path):
    rc, ran = _drun(_staged_session(tmp_path), keep_going=True)
    assert rc == 0 and ran == list(ORDER[4:])
    rc, ran = _drun(_staged_session(tmp_path / "y", "device_components"),
                    start="device_components", keep_going=True)
    assert rc == 0 and ran == list(ORDER[5:])


def test_the_cap_stops_the_session_before_the_next_step(tmp_path):
    s = _staged_session(tmp_path)
    ran = []
    # each step "takes" 100 s: the session's clock reads 100 s per step run
    rc = sess.run_session(s, start="host_components", extra=[], dry_run=False, sync=False,
                          debug=True, runner=ts.fake_runner({}, ran), cap_s=150.0,
                          clock=lambda: 100.0 * len(ran))
    assert rc == 1 and ran == ["host_components", "device_components"]    # 0 s, then 100 s, then 200 s > 150
    last = json.loads((s / "manifest.json").read_text())["attempts"][-1]
    assert last["step"] == "predict" and last["status"] == "not_run" and "cap" in last["reason"]
    # a session within its cap runs on
    rc2 = sess.run_session(_staged_session(tmp_path / "w"), start="host_components", extra=[],
                           dry_run=False, sync=False, debug=True, cap_s=1e9,
                           runner=ts.fake_runner({}, []))
    assert rc2 == 0


def test_a_debug_session_and_another_never_share_a_directory(tmp_path):
    s = _staged_session(tmp_path)
    assert sess.run_session(s, start="host_components", extra=[], dry_run=False, sync=False,
                            runner=ts.fake_runner({}, [])) == 2          # not a debug run
    rc, _ = _drun(tmp_path / "plain", start="link")                      # a plain directory, debug run
    assert rc == 2
    ts_rc, _ = ts._run(tmp_path / "real")
    assert ts_rc == 0
    assert sess.run_session(tmp_path / "real", start="hold", extra=[], dry_run=False, sync=False,
                            debug=True, runner=ts.fake_runner({}, [])) == 2


def test_a_debug_cpu_check_ends_at_hold(tmp_path):
    rc, ran = _drun(_staged_session(tmp_path), rehearsal_end=True)
    assert rc == 0 and ran == ["host_components", "device_components", "predict", "hold"]
    assert "ground_truth" not in ran


def test_the_dependencies_name_real_steps_and_every_later_step_is_covered():
    assert all(k in ORDER and all(d in ORDER for d in v) for k, v in dbg.DEPENDS.items())
    assert set(dbg.DEPENDS) == set(ORDER[5:])
    for step, deps in dbg.DEPENDS.items():
        assert all(ORDER.index(d) < ORDER.index(step) for d in deps)
    assert dbg.blocked("hold", {"predict": "failed"}) == ["predict"]
    assert dbg.blocked("hold", {"predict": "passed"}) == []
    assert dbg.blocked("hold", {}) == ["predict"]          # not run is not passed


def test_the_cap_is_one_and_a_half_times_the_projected_steps():
    import t31_projection as pr
    nm = dbg.debug_config(_full_cfg())["not_measured"]
    c6 = pr.debug_cap_min("device_components", nm)
    c5 = pr.debug_cap_min("host_components", nm, ["a"] * 4)
    assert dbg.CAP_FACTOR == 1.5
    steps = pr.project(1)["steps"]
    full = sum(v for k, v in steps.items() if int(k.split()[0]) >= 6)
    assert 0.5 * 1.5 * full < c6 <= 1.5 * full                      # fewer cells than the full grid
    assert c5 > c6 and c5 - c6 == pytest.approx(1.5 * 4 * (2.0 + 0.5), rel=1e-6)


def test_the_cli_refuses_what_a_debug_session_may_not_be(tmp_path, capsys):
    s = _staged_session(tmp_path)
    base = ["--debug", "--session", str(s), "--no-sync"]
    assert sess.main(base + ["--rehearsal"]) == 2
    assert sess.main(["--debug", "--no-sync"]) == 2                        # no session
    assert sess.main(base + ["--config", "x.json"]) == 2
    assert sess.main(["--debug", "--session", str(tmp_path / "plain")]) == 2   # not staged
    assert sess.main(base + ["--from", "link"]) == 2                       # steps 1 to 4 are never rerun
    assert sess.main(base + ["--from", "device_components"]) == 2          # not where it was staged
    assert sess.main(["--keep-going", "--session", str(tmp_path / "k")]) == 2   # debug only
    assert "REFUSING" in capsys.readouterr().err


# --- against session 4's banked pull ------------------------------------------------------------------

@pytest.mark.skipif(not BANKED.exists(), reason="session 4's pull is not in this tree")
def test_session_four_stages_with_only_the_missing_cells_left_to_run(tmp_path):
    rec = dbg.stage(BANKED, tmp_path / "s4dbg", start="host_components")
    assert rec["donor"]["identity"]["git_commit"].startswith("5cb47ac")
    assert len(rec["cells"]) == 31
    assert sorted(rec["run_in_this_session"]) == sorted([
        "meanpool_d0.25_L32768_b1", "meanpool_d0.5_L32768_b1",
        "xattention_dcal_L32768_b1", "xattention_noassert_dcal_L32768_b1"])
    assert [(e["arm"], e["density"], e["length"]) for e in rec["not_measured"]] == [
        ("oracle", 0.5, 32768)]
    assert not list((tmp_path / "s4dbg").rglob("isolated_host.events.jsonl"))
    dumped = json.dumps(rec) + (tmp_path / "s4dbg" / dbg.DEBUG_NOTE).read_text()
    assert "research-" not in dumped and "gs://" not in dumped


@pytest.mark.parametrize("bucket", ["gs://attnbench-results-x", "gs://attnbench-results-x/", "attnbench-results-x"])
def test_the_fetch_command_names_the_donor_in_the_bucket_with_one_prefix(bucket, monkeypatch):
    cmd = dbg.fetch_command(bucket, "inst-1", "sess", Path("/d"))
    assert cmd[-2:] == ["gs://attnbench-results-x/inst-1/results/sess", "/d"]
    assert "-x" in cmd and cmd[cmd.index("-x") + 1].endswith(r"isolated_host\.events\.jsonl$")


def test_the_fetch_command_takes_what_results_bucket_returns(monkeypatch):
    monkeypatch.setenv("ATTNBENCH_BUCKET", "gs://attnbench-results-y")
    cmd = dbg.fetch_command(st.results_bucket(), "i", "s", Path("d"))
    assert not any("gs://gs:" in c for c in cmd) and cmd[-2] == "gs://attnbench-results-y/i/results/s"
