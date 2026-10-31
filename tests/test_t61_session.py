"""The P2 session runner (T6.1) and the engine `scripts/t64_session.py`
shares: order, the 60 s hold, the strict first-clocked-block rule, the late
check, stop at the first failure, resume, and the refusals, on CPU.

Steps run in-process through the runner's own `main` (the fast tests) and in
their own processes (one end-to-end dry run). The dry clock is virtual, so
the 60 s rule is the real 60 s without waiting. The commit, the sync and the
late check are `attnbench.predict.commit`'s own code, against a directory
store. Each guard of `check_stamp` has a mutant that must fail.

Two tests need torch and are skipped without it (listed in the C15 report as
UNRUN): the real preflight and the real launch-queue body refuse a CPU host
with a step failure, not a crash.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t61_session as S  # noqa: E402

T61 = S.T61
STEPS = ("prereqs", "preflight", "link", "launch_queue", "probe", "mask_gate",
         "host_components", "device_components", "dense_select", "host_speed_mid",
         "predict", "ground_truth", "late_check", "trace_checks", "host_speed_end")


def inproc(cfg, calls=None):
    """A runner that runs each step through the script's own `main`, in this
    process, and writes a log the way run_logged does."""
    def runner(cmd, log):
        argv = cmd[cmd.index("step"):]
        if calls is not None:
            calls.append(argv[1])
        log.parent.mkdir(parents=True, exist_ok=True)
        rc = S.main(argv, cfg=cfg)
        log.write_text(f"# $ {' '.join(cmd)}\n# exit {rc}\n")
        return rc
    return runner


def dry(cfg, session, start=None, calls=None, extra=()):
    return S.run_session(cfg, session, start=start or cfg.names[0], extra=list(extra),
                         dry_run=True, sync=False, runner=inproc(cfg, calls))


def with_harness(cfg, harness):
    fakes = {**cfg.fakes,
             "ground_truth": lambda c, a: S.body_ground_truth(c, a, harness=harness)}
    return dataclasses.replace(cfg, fakes=fakes)


def manifest(session):
    return json.loads((session / "manifest.json").read_text())


def statuses(session):
    return S.last_status(manifest(session))


def args_for(session, out, *, dry_run=True, banked=()):
    return Namespace(session=str(session), out=str(out), dry_run=dry_run,
                     banked_at=list(banked), fa3_sha256=None)


# --- the order -----------------------------------------------------------------------

def test_the_p2_steps_are_section_5_3_in_order():
    assert T61.names == STEPS
    p = T61.names.index("predict")
    assert T61.names[p + 1] == "ground_truth"
    assert T61.names.index("dense_select") < p
    assert T61.names.index("mask_gate") < p
    assert T61.provisioning == ("FLEX_START", "SPOT")
    assert T61.instance_type == "a2-ultragpu-1g" and T61.banked == ("P1",)


def moved(names, step, before):
    """`names` with `step` taken out and put back just before `before`."""
    n = [x for x in names if x != step]
    n.insert(n.index(before), step)
    return n


def inserted(names, step, before):
    n = list(names)
    n.insert(n.index(before), step)
    return n


@pytest.mark.parametrize("mutate, why", [
    (lambda n: moved(n, "preflight", "prereqs"), "prereqs must be the first step"),
    (lambda n: inserted(n, "extra_probe", "ground_truth"), "immediately after predict"),
    (lambda n: moved(n, "dense_select", "ground_truth"), "dense_select must come before predict"),
    (lambda n: moved(n, "late_check", "predict"), "late_check must come after ground_truth"),
])
def test_validate_refuses_each_order_breach(mutate, why):
    names = mutate(list(T61.names))
    steps = tuple(S.Step(n, "T", "w") for n in names)
    bodies = {n: (lambda c, a: {}) for n in names}
    cfg = dataclasses.replace(T61, steps=steps, fakes=bodies, reals=bodies)
    with pytest.raises(RuntimeError, match=why):
        S.validate(cfg)


def test_validate_refuses_a_step_without_a_body():
    fakes = dict(T61.fakes)
    del fakes["late_check"]
    with pytest.raises(RuntimeError, match="without a body"):
        S.validate(dataclasses.replace(T61, fakes=fakes))


# --- the whole dry run ------------------------------------------------------------------

def test_the_dry_run_passes_every_step_in_its_own_process(tmp_path):
    session = tmp_path / "s"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "t61_session.py"), "run",
                        "--dry-run", "--session", str(session)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    st = statuses(session)
    assert st["prereqs"] == "skipped"          # this branch's plan is not at Gate B
    assert all(st[s] == "passed" for s in STEPS[1:])
    for i, s in enumerate(STEPS, 1):
        d = session / f"{i:02d}_{s}"
        assert (d / "result.json").exists()
        log = (d / "step.log").read_text()
        assert f"t61_session.py step {s} " in log and "# exit" in log   # its own process
    gt = json.loads((session / "12_ground_truth" / "result.json").read_text())["findings"]
    late = json.loads((session / "13_late_check" / "result.json").read_text())["findings"]
    assert gt["gap_s"] >= S.PLANNED_SYNC_WAIT_S - 1e-6
    assert late["gap_s"] == pytest.approx(gt["gap_s"])
    rec = json.loads((session / S.RECEIPT).read_text())
    assert rec["remote_path"].startswith("predictions/P2/") and "gs://" not in json.dumps(rec)
    assert "holding" in p.stdout


def test_a_dry_run_in_process_matches_and_holds_on_the_virtual_clock(tmp_path):
    session = tmp_path / "s"
    calls = []
    assert dry(T61, session, calls=calls) == 0
    assert calls == list(STEPS)
    synced = S.epoch(json.loads((session / S.RECEIPT).read_text())["synced_at"])
    opened = S.epoch(json.loads((session / S.OPENED).read_text())[0]["opened_at"])
    assert opened - synced >= S.PLANNED_SYNC_WAIT_S
    assert json.loads((session / S.DRY_CLOCK).read_text())["offset_s"] >= 90


# --- the hold ---------------------------------------------------------------------------

def _receipt(session, synced_at):
    session.mkdir(parents=True, exist_ok=True)
    S.write_json(session / S.RECEIPT, {"synced_at": S.iso(synced_at)})


def test_the_hold_waits_out_the_planned_wait(tmp_path):
    session = tmp_path / "s"
    t = S.now(session, True)
    _receipt(session, t)
    opened = S.hold_for_sync(session, True)
    assert opened >= t + 90.0
    assert opened < t + 90.0 + 5.1           # it steps in at most 5 s
    assert len(json.loads((session / S.OPENED).read_text())) == 1


def test_the_hold_does_not_wait_when_the_sync_is_old(tmp_path):
    session = tmp_path / "s"
    t = S.now(session, True)
    _receipt(session, t - 1000)
    S.hold_for_sync(session, True)
    assert not (session / S.DRY_CLOCK).exists()


def test_the_hold_refuses_without_a_receipt(tmp_path):
    with pytest.raises(S.StepFailed, match="no prediction was synced"):
        S.hold_for_sync(tmp_path, True)


def test_a_real_run_refuses_a_virtual_clock(tmp_path):
    (tmp_path / S.DRY_CLOCK).write_text(json.dumps({"offset_s": 100}))
    with pytest.raises(S.StepFailed, match="dry session"):
        S.now(tmp_path, False)


# --- the strict stamp ------------------------------------------------------------------

T = 1_800_000_000.0


def stamp_result(stamp, key=S.STRICT_KEY):
    return {key: stamp, "complete": True}


def test_a_stamp_more_than_60_s_after_the_sync_passes_and_reports_the_gap():
    f = S.check_stamp(stamp_result(T + 66), synced_at=T, opened_at=T + 65.5,
                      ended_at=T + 100, earlier_start=None)
    assert f["gap_s"] == pytest.approx(65.5)      # the open is earlier than the stamp
    assert f["stamp_after_open_s"] == pytest.approx(0.5)


@pytest.mark.parametrize("gap", [59.0, 59.999, 60.0])
def test_a_first_clocked_block_within_60_s_of_the_sync_fails(gap):
    with pytest.raises(S.StepFailed, match="K9"):
        S.check_stamp(stamp_result(T + gap), synced_at=T, opened_at=T + gap,
                      ended_at=T + 100, earlier_start=None)


def test_just_over_60_s_passes():
    S.check_stamp(stamp_result(T + 60.001), synced_at=T, opened_at=T + 60.001,
                  ended_at=T + 100, earlier_start=None)


def test_the_25_october_key_is_refused():
    with pytest.raises(S.StepFailed, match="strict rule"):
        S.check_stamp(stamp_result(T + 70, key=S.OLD_KEY), synced_at=T,
                      opened_at=T + 65, ended_at=T + 100, earlier_start=None)


@pytest.mark.parametrize("result", [{}, {S.STRICT_KEY: True}, {S.STRICT_KEY: "soon"}])
def test_a_missing_or_malformed_stamp_is_refused(result):
    with pytest.raises(S.StepFailed):
        S.check_stamp(result, synced_at=T, opened_at=T + 65, ended_at=T + 100,
                      earlier_start=None)


def test_a_stamp_before_the_attempt_opened_is_refused():
    # A harness that stamped a warm-up from before the hold, or a stale result.
    with pytest.raises(S.StepFailed, match="before this attempt opened"):
        S.check_stamp(stamp_result(T + 64), synced_at=T, opened_at=T + 65,
                      ended_at=T + 100, earlier_start=None)


def test_a_stamp_after_the_harness_returned_is_refused():
    with pytest.raises(S.StepFailed, match="after the harness returned"):
        S.check_stamp(stamp_result(T + 101), synced_at=T, opened_at=T + 65,
                      ended_at=T + 100, earlier_start=None)


def test_an_earlier_attempt_fixes_the_session_start():
    # This attempt is late enough on its own; an earlier one started at +59 s.
    with pytest.raises(S.StepFailed, match="59.000 s"):
        S.check_stamp(stamp_result(T + 200), synced_at=T, opened_at=T + 199,
                      ended_at=T + 300, earlier_start=T + 59)


# --- the ground-truth step --------------------------------------------------------------

class Spy:
    def __init__(self, result=None):
        self.called, self.result = 0, result

    def __call__(self, cfg, args):
        self.called += 1
        session = Path(args.session)
        r = dict(self.result) if self.result else {"complete": True}
        r.setdefault(S.STRICT_KEY, S.now(session, True))
        return r


def test_ground_truth_refuses_inside_60_s_before_calling_the_harness(tmp_path):
    session = tmp_path / "s"
    t = S.now(session, True)
    _receipt(session, t)
    S.write_json(session / S.OPENED, [{"opened_at": S.iso(t)}])
    spy = Spy()
    with pytest.raises(S.StepFailed, match="since the prediction was synced"):
        S.body_ground_truth(T61, args_for(session, tmp_path / "o"), harness=spy)
    assert spy.called == 0


def test_ground_truth_refuses_when_the_runner_did_not_open_it(tmp_path):
    session = tmp_path / "s"
    _receipt(session, S.now(session, True) - 1000)
    with pytest.raises(S.StepFailed, match="did not record opening"):
        S.body_ground_truth(T61, args_for(session, tmp_path / "o"), harness=Spy())


def test_the_session_stops_when_the_harness_uses_the_old_key(tmp_path):
    session = tmp_path / "s"

    def old(cfg, args):
        return {S.OLD_KEY: S.now(Path(args.session), True), "complete": True}
    assert dry(with_harness(T61, old), session) == 1
    st = statuses(session)
    assert st["ground_truth"] == "failed" and "late_check" not in st


def test_an_incomplete_ground_truth_fails_and_a_rerun_keeps_the_first_start(tmp_path):
    session = tmp_path / "s"
    assert dry(with_harness(T61, Spy({"complete": False})), session) == 1
    first_open = S.epoch(json.loads((session / S.OPENED).read_text())[0]["opened_at"])
    S.advance(session, True, 300)
    assert dry(T61, session, start="ground_truth") == 0
    opened = json.loads((session / S.OPENED).read_text())
    assert len(opened) == 2
    assert S.session_start(session) == pytest.approx(first_open)
    late = json.loads((session / "13_late_check" / "result.json").read_text())["findings"]
    assert S.epoch(late["session_start"]) == pytest.approx(first_open)
    kept = [p.name for p in session.iterdir() if p.name.startswith("12_ground_truth.failed.")]
    assert len(kept) == 1                       # the failed attempt is on record


# --- resume and its refusals -----------------------------------------------------------

def test_the_first_failure_stops_the_session(tmp_path):
    session = tmp_path / "s"
    fakes = {**T61.fakes, "link": S.not_built("T2.5", "x")}
    calls = []
    assert dry(dataclasses.replace(T61, fakes=fakes), session, calls=calls) == 1
    assert calls == ["prereqs", "preflight", "link"]


@pytest.mark.parametrize("start", ["prereqs", "link", "dense_select", "predict"])
def test_from_any_step_up_to_predict_is_refused_once_predict_passed(tmp_path, start, capsys):
    session = tmp_path / "s"
    assert dry(with_harness(T61, Spy({"complete": False})), session) == 1
    assert dry(T61, session, start=start) == 2
    assert "second prediction" in capsys.readouterr().err


def test_from_needs_every_earlier_step_passed(tmp_path, capsys):
    session = tmp_path / "s"
    fakes = {**T61.fakes, "probe": S.not_built("T2.18", "x")}
    assert dry(dataclasses.replace(T61, fakes=fakes), session) == 1
    assert dry(T61, session, start="mask_gate") == 2
    assert "earlier steps have not passed" in capsys.readouterr().err
    assert dry(T61, session, start="probe") == 0


def test_predict_refuses_a_second_receipt(tmp_path):
    session = tmp_path / "s"
    _receipt(session, 0.0)
    with pytest.raises(S.StepFailed, match="already committed"):
        S.body_predict(T61, args_for(session, tmp_path / "o"))


def test_a_dry_session_is_not_resumed_as_a_real_one(tmp_path, capsys):
    session = tmp_path / "s"
    assert dry(T61, session) == 0
    rc = S.run_session(T61, session, start="ground_truth", extra=[], dry_run=False,
                       sync=False, runner=inproc(T61))
    assert rc == 2 and "dry T6.1 session" in capsys.readouterr().err


def test_a_real_start_needs_a_clean_tree_and_the_banking_times(monkeypatch):
    monkeypatch.setattr(S, "tree_clean", lambda: False)
    a = Namespace(dry_run=False, banked_at=["P1=2026-11-20T10:00:00+00:00"])
    assert "dirty" in S.start_refusal(T61, a)
    monkeypatch.setattr(S, "tree_clean", lambda: True)
    assert S.start_refusal(T61, a) is None
    assert "missing: ['P1']" in S.start_refusal(T61, Namespace(dry_run=False, banked_at=[]))
    assert "UTC offset" in S.start_refusal(
        T61, Namespace(dry_run=False, banked_at=["P1=2026-11-20T10:00:00"]))


# --- the late check ------------------------------------------------------------------

def test_the_late_check_refuses_a_prediction_altered_in_the_store(tmp_path):
    session = tmp_path / "s"
    late_fakes = {**T61.fakes, "late_check": T61.fakes["late_check"]}
    assert dry(T61, session) == 0
    rec = json.loads((session / S.RECEIPT).read_text())
    stored = session / "dry_store" / rec["remote_path"]
    stored.chmod(0o644)
    stored.write_bytes(stored.read_bytes().replace(b'"P2"', b'"P3"', 1))
    with pytest.raises(S.StepFailed, match="would refuse"):
        S.body_late_check(dataclasses.replace(T61, fakes=late_fakes),
                          args_for(session, tmp_path / "o"))


def test_the_late_check_refuses_a_start_inside_the_skew(tmp_path):
    session = tmp_path / "s"
    assert dry(T61, session) == 0
    rec = json.loads((session / S.RECEIPT).read_text())
    S.append_json(session / S.OPENED, {"opened_at": S.iso(S.epoch(rec["synced_at"]) + 30)})
    with pytest.raises(S.StepFailed, match="K9"):
        S.body_late_check(T61, args_for(session, tmp_path / "o"))


def test_the_dry_store_never_clobbers(tmp_path):
    clock = iter([10.0, 20.0])
    st = S.DirStore(tmp_path, lambda: next(clock))
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    st.put(a, "drystore://x/f")
    st.put(b, "drystore://x/f")
    assert st.get("drystore://x/f") == b"one"
    assert st.created("drystore://x/f").timestamp() == 10.0
    with pytest.raises(S.StepFailed):
        st.put(a, "gs://bucket/f")


def test_the_bucket_is_never_guessed(monkeypatch):
    monkeypatch.setenv("ATTNBENCH_BUCKET", "gs://b/")
    assert S.results_bucket() == "gs://b"
    monkeypatch.delenv("ATTNBENCH_BUCKET")
    monkeypatch.setenv("ATTNBENCH_METADATA_URL", "http://127.0.0.1:9/none")
    with pytest.raises(S.StepFailed, match="no results bucket"):
        S.results_bucket()


# --- prereqs ------------------------------------------------------------------------------

PLAN = """\
| T2.1 | probe | Code | | Runs | Done (25 Oct 2026): built |
| T2.2 | profiler | Code | T2.3 | Unit | Partly built (27 Oct 2026) |
| T4.4 | Tagged lock commit | Code | T4.3 | Tag on origin | Not started |
| T6.1 | P2 | Code prepares; R runs | Gate B | Done (1 Dec 2026) |
"""


def test_plan_statuses_read_six_and_five_column_rows():
    st = S.plan_statuses(PLAN)
    assert st == {"T2.1": "Done (25 Oct 2026): built", "T2.2": "Partly built (27 Oct 2026)",
                  "T4.4": "Not started", "T6.1": "Done (1 Dec 2026)"}


def test_only_done_meets_a_requirement():
    reqs = [S.Requirement(t, "w") for t in ("T2.1", "T2.2", "T4.4", "T6.1", "T9.9")]
    assert [r["task"] for r in S.unmet(reqs, S.plan_statuses(PLAN))] == ["T2.2", "T4.4", "T9.9"]
    assert S.unmet(reqs[:1], {"T2.1": "Built (25 Oct 2026)"})[0]["task"] == "T2.1"


def test_prereqs_refuses_a_real_session_and_skips_a_dry_one(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "RESEARCH_PLAN.md").write_text(PLAN)
    monkeypatch.setattr(S, "ROOT", tmp_path)
    cfg = dataclasses.replace(T61, requirements=(S.Requirement("T2.2", "w"),), files=())
    with pytest.raises(S.StepFailed, match="T2.2"):
        S.body_prereqs(cfg, args_for(tmp_path, tmp_path, dry_run=False))
    with pytest.raises(S.Skip):
        S.body_prereqs(cfg, args_for(tmp_path, tmp_path, dry_run=True))
    ok = dataclasses.replace(cfg, requirements=(S.Requirement("T2.1", "w"),))
    assert S.body_prereqs(ok, args_for(tmp_path, tmp_path, dry_run=False))["unmet"] == []
    bad = dataclasses.replace(ok, files=(("x.py", None),))
    with pytest.raises(S.StepFailed, match="file check"):
        S.body_prereqs(bad, args_for(tmp_path, tmp_path, dry_run=False))


def test_this_branchs_plan_is_not_ready_for_p2():
    rows = S.unmet(T61.requirements, S.plan_statuses(
        (ROOT / "docs" / "RESEARCH_PLAN.md").read_text()))
    tasks = {r["task"] for r in rows}
    assert {"T2.23", "T3.1", "T4.4"} <= tasks


def test_the_strict_stamp_is_a_file_check():
    assert ("attnbench/ground_truth.py", S.STRICT_KEY) in T61.files


# --- wiring --------------------------------------------------------------------------

def test_a_real_step_is_wrapped_in_run_phase_and_named_for_t61(tmp_path):
    cmd = S.step_command(T61, "link", tmp_path, tmp_path / "03_link", ["--banked-at", "P1=x"],
                         dry_run=False, sync=True)
    assert cmd[:4] == ["bash", str(ROOT / "scripts" / "run_phase.sh"), "t61_03_link", "--"]
    assert cmd[5].endswith("scripts/t61_session.py") and cmd[6:8] == ["step", "link"]
    assert "--dry-run" not in cmd and cmd[-2:] == ["--banked-at", "P1=x"]
    d = S.step_command(T61, "link", tmp_path, tmp_path, [], dry_run=True, sync=False)
    assert d[0] == sys.executable and d[-1] == "--dry-run"


@pytest.mark.parametrize("step, task", [
    ("probe", "T2.18"), ("mask_gate", "T2.23"), ("host_components", "T2.2"),
    ("device_components", "T2.4"), ("dense_select", "T2.16")])
def test_real_bodies_not_yet_built_refuse_naming_their_task(tmp_path, step, task):
    with pytest.raises(S.NotBuilt, match=task.replace(".", r"\.")):
        T61.reals[step](T61, args_for(tmp_path, tmp_path, dry_run=False))


def test_the_real_ground_truth_harness_is_not_built():
    with pytest.raises(S.NotBuilt, match="T2.13"):
        S.real_harness(T61, None)


def test_the_real_record_needs_p1s_banking_time_first(tmp_path):
    w = datetime(2026, 11, 20, tzinfo=timezone.utc)
    with pytest.raises(S.StepFailed, match="missing: \\['P1'\\]"):
        S.real_record(T61, args_for(tmp_path, tmp_path, dry_run=False), w)
    with pytest.raises(S.NotBuilt, match="T2.11"):
        S.real_record(T61, args_for(tmp_path, tmp_path, dry_run=False,
                                    banked=["P1=2026-11-19T00:00:00+00:00"]), w)


GOOD_ROW = {"cloud_provider": "gcp", "instance_type": "a2-ultragpu-1g",
            "instance_generation": "a2", "cpu_microcode": "0x1", "provisioning_model": "SPOT",
            "provisioning_model_problem": None, "python_optimize": 0, "git_dirty": False,
            "gpu_name": "NVIDIA A100-SXM4-80GB", "null_launch_s_median": 4e-6}


def test_the_provenance_row_passes_on_a2_flex_or_spot():
    assert S.check_provenance_row(T61, GOOD_ROW, []) == []
    assert S.check_provenance_row(T61, {**GOOD_ROW, "provisioning_model": "FLEX_START"}, []) == []


@pytest.mark.parametrize("key, value", [
    ("instance_type", "g2-standard-8"), ("provisioning_model", "STANDARD"),
    ("gpu_name", "NVIDIA L4"), ("git_dirty", True), ("python_optimize", 1),
    ("cpu_microcode", ""), ("provisioning_model_problem", "unreadable"),
    ("null_launch_s_median", None), ("cloud_provider", "aws")])
def test_each_wrong_provenance_field_is_a_problem(key, value):
    assert S.check_provenance_row(T61, {**GOOD_ROW, key: value}, [])


def test_an_integrity_problem_is_passed_through():
    assert S.check_provenance_row(T61, GOOD_ROW, ["stamp mismatch"]) == ["stamp mismatch"]


# --- torch-dependent (UNRUN without torch) ---------------------------------------------

def test_the_real_preflight_refuses_a_cpu_host(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(S.StepFailed):
        S.real_preflight(T61, args_for(tmp_path, tmp_path, dry_run=False))


def test_the_real_launch_queue_body_fails_cleanly_on_a_cpu_host(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this checks the CPU refusal")
    with pytest.raises(S.StepFailed):
        S.real_launch_queue(T61, args_for(tmp_path, tmp_path, dry_run=False))


# --- added after break-testing: the two guards no test killed --------------------------

def test_the_late_check_refuses_a_receipt_naming_another_hash(tmp_path):
    session = tmp_path / "s"
    assert dry(T61, session) == 0
    rec = json.loads((session / S.RECEIPT).read_text())
    S.write_json(session / S.RECEIPT, {**rec, "sha256": "0" * 64})
    with pytest.raises(S.StepFailed, match="the receipt says"):
        S.body_late_check(T61, args_for(session, tmp_path / "o"))


def test_a_skipped_step_fails_a_real_session(tmp_path):
    def skipping(cmd, log):
        out = Path(cmd[cmd.index("--out") + 1])
        S.write_json(out / "result.json", {"status": "skipped", "reason": "no CUDA"})
        return 3
    rc = S.run_session(T61, tmp_path / "s", start="prereqs", extra=[], dry_run=False,
                       sync=False, runner=skipping)
    assert rc == 1
    a = manifest(tmp_path / "s")["attempts"]
    assert [x["status"] for x in a] == ["failed"] and "skipped on a real run" in a[0]["reason"]
