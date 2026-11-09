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


def test_a_real_start_needs_the_exit_wrapper(monkeypatch):
    monkeypatch.setattr(S, "tree_clean", lambda: True)
    a = Namespace(dry_run=False, banked_at=["P1=2026-11-20T10:00:00+00:00"])
    monkeypatch.delenv(S.EXIT_WRAPPER_ENV, raising=False)
    assert "scripts/session_exit.sh" in S.start_refusal(T61, a)
    monkeypatch.setenv(S.EXIT_WRAPPER_ENV, "yes")
    assert "scripts/session_exit.sh" in S.start_refusal(T61, a)
    monkeypatch.setenv(S.EXIT_WRAPPER_ENV, "armed")
    assert S.start_refusal(T61, a) is None
    monkeypatch.delenv(S.EXIT_WRAPPER_ENV)
    assert S.start_refusal(T61, Namespace(dry_run=True, banked_at=[])) is None   # a dry run needs none


def test_a_real_start_needs_a_clean_tree_and_the_banking_times(monkeypatch):
    monkeypatch.setenv(S.EXIT_WRAPPER_ENV, "armed")
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
    # 10 Nov 2026 (docs/prereqs_20261109.md): T2.3 done; T2.21 waived on conditions the check
    # cannot read, T2.23 flagged, T2.17 and T2.26 in the rehearsal, T4.4 the lock
    assert tasks == {"T2.17", "T2.21", "T2.23", "T2.26", "T4.4"}


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


# --- the mask gate (T2.23; adviser's ruling 9 of the tenth 9 Nov 2026 paste) ---------------

def _mask_record(impl, gpu, masks, prompt="p0", arm="xattention", length=1951):
    import numpy as np
    import torch
    from attnbench.components import mask_bank as MB
    from attnbench.components.op_sequence import CodePath
    cp = CodePath(arm=arm, gpu=gpu, implementation=impl, libraries=(), python_optimize=0)
    return MB.MaskRecord(cp, length, 1, prompt, tuple(
        MB._bank_one(torch.from_numpy(np.array(m, dtype=bool)), True) for m in masks))


LAYERS2 = [[[1, 0], [1, 1]], [[1, 0], [0, 1]]]          # two layers, 2 x 2 blocks, causal
BOUND = {"max_density_difference": 0.01}
SAME_DENSITY = [[[1, 0], [1, 1]], [[1, 0], [1, 0]]]    # layer 2: 2 of 3 kept, other blocks


def _mask_gate_setup(tmp_path, triton=LAYERS2, prompt="p0", p1=LAYERS2, torch=LAYERS2):
    from attnbench.components import mask_bank as MB
    session = tmp_path / "s"
    cell = session / S.P1_MASKS / "xattention_dcal_L2048_b1" / "recording"
    cell.mkdir(parents=True)
    MB.save(_mask_record("xattn:torch", "NVIDIA L4", p1), cell / "masks")
    seen = []

    def record(args, cells, out):
        seen.append(cells)
        return {"torch": [_mask_record("xattn:torch", "NVIDIA A100", torch)],
                "triton": [_mask_record("xattn:triton", "NVIDIA A100", triton, prompt=prompt)]}
    args = Namespace(out=str(tmp_path / "mg"), session=str(session), dry_run=False)
    return args, record, seen


def test_the_mask_gate_compares_both_p2_paths_with_p1s_banked_masks(tmp_path):
    args, record, seen = _mask_gate_setup(tmp_path)
    r = S.real_mask_gate(T61, args, record=record, bound=BOUND)
    assert seen == [[("xattention", 2048, 1, "p0")]]        # P1's banked cell, by budget
    assert r["verdict"] == "pass" and r["cells"] == 1
    gate = json.loads((Path(args.out) / "mask_gate.json").read_text())
    assert gate["verdict"] == "pass" and gate["triton_vs_p1"]["n_layers"] == 2
    comps = json.loads((Path(args.out) / "mask_comparisons.json").read_text())
    assert set(comps) == {"torch_vs_p1", "triton_vs_p1", "triton_vs_torch_on_p2"}


def test_a_failed_mask_gate_is_a_verdict_not_a_stop(tmp_path):
    args, record, _ = _mask_gate_setup(tmp_path, triton=[[[1, 0], [1, 1]], [[1, 0], [1, 1]]])
    r = S.real_mask_gate(T61, args, record=record, bound=BOUND)
    assert r["verdict"] == "fail" and r["cross_card_flags"] == ["triton_vs_p1"]
    assert r["failures"] == {"pytorch_vs_p1": 0, "triton_vs_p1": 1, "triton_vs_pytorch_on_p2": 1}


def test_the_verdict_is_p2s_own_card_and_p1_is_a_flag(tmp_path):
    """Adviser's ruling 3 of the eleventh 9 Nov 2026 paste: both P2 paths
    agree, both differ from P1's masks beyond the bound: a pass, flagged."""
    args, record, _ = _mask_gate_setup(tmp_path, p1=[[[1, 0], [1, 1]], [[1, 0], [1, 1]]])
    r = S.real_mask_gate(T61, args, record=record, bound=BOUND)
    assert r["verdict"] == "pass"
    assert r["cross_card_flags"] == ["pytorch_vs_p1", "triton_vs_p1"]
    gate = json.loads((Path(args.out) / "mask_gate.json").read_text())
    assert gate["cross_card_flags"] == r["cross_card_flags"]


def test_block_agreement_is_reported_and_decides_nothing(tmp_path):
    args, record, _ = _mask_gate_setup(tmp_path, triton=SAME_DENSITY)
    r = S.real_mask_gate(T61, args, record=record, bound=BOUND)
    gate = json.loads((Path(args.out) / "mask_gate.json").read_text())
    assert r["verdict"] == "pass" and r["cross_card_flags"] == []
    assert gate["triton_vs_pytorch_on_p2"]["min_agreement"] == pytest.approx(1 / 3)


def test_the_mask_gate_writes_its_comparisons_then_refuses_without_a_lock(tmp_path, monkeypatch):
    import t31_steps as st
    monkeypatch.setattr(st, "LOCK_FILE", tmp_path / "lock_values.json")
    args, record, _ = _mask_gate_setup(tmp_path)
    with pytest.raises(S.StepFailed, match="the mask-gate bound is a lock value"):
        S.real_mask_gate(T61, args, record=record)
    assert (Path(args.out) / "mask_comparisons.json").exists()
    assert not (Path(args.out) / "mask_gate.json").exists()


def test_the_mask_gate_refuses_banks_on_other_prompts(tmp_path):
    args, record, _ = _mask_gate_setup(tmp_path, prompt="p1")
    with pytest.raises(S.StepFailed, match="cannot compare these banks"):
        S.real_mask_gate(T61, args, record=record, bound=BOUND)


def test_the_mask_gate_bound_is_read_from_a_complete_lock_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from attnbench.lock import checklist
    lock = tmp_path / "lock_values.json"
    complete = SimpleNamespace(complete=True)
    monkeypatch.setattr(checklist, "check", lambda lk, text: complete)
    def write(bound, flag=0.01):
        lock.write_text(json.dumps({"values": {
            "mask_gate_bound": {"value": bound},
            "compiled_versus_eager_density_flag_tolerance": {"value": flag}}}))
    write(0.01)
    assert S.mask_gate_bound(lock, plan_text="") == BOUND
    write({"min_agreement": 0.99, "max_density_difference": 0.01})
    with pytest.raises(S.StepFailed, match="not one density tolerance"):
        S.mask_gate_bound(lock, plan_text="")
    write(0.02)
    with pytest.raises(S.StepFailed, match="not its compiled-versus-eager density tolerance"):
        S.mask_gate_bound(lock, plan_text="")
    complete.complete = False
    complete.report = lambda: ""
    with pytest.raises(S.StepFailed, match="not a complete lock"):
        S.mask_gate_bound(lock, plan_text="")
    with pytest.raises(S.StepFailed, match="no XAttention masks"):
        S.banked_masks(tmp_path / "nothing")


@pytest.mark.parametrize("step", ["probe", "host_components", "device_components",
                                  "dense_select", "link"])
def test_the_component_steps_are_t31s_and_refuse_p1s_working_values(tmp_path, step):
    """T3.1's own steps on P2's config: its values are fixed at lock (T4.1),
    and until then T3.1's guard refuses to fill them with P1's."""
    body = T61.reals[step]
    assert body is S.real_link or (body.func is S._t31_step and body.args == (step,))
    with pytest.raises(S.StepFailed, match="the working values are P1's only; this session is 'P2'"):
        T61.reals[step](T61, args_for(tmp_path, tmp_path, dry_run=False))


def test_each_pairing_names_its_components_config_with_the_dropped_cell():
    from t64_session import T64
    for cfg, card in ((T61, "A100"), (T64, "H100")):
        c = json.loads((ROOT / cfg.components_config).read_text())
        assert (c["pairing"], c["card"]) == (cfg.pairing, card)
        assert c["prediction_prefix"] == f"predictions/{cfg.pairing}"
        assert [(e["arm"], e["density"], e["length"], e["batch"]) for e in c["not_measured"]] == [
            ("oracle", 0.5, 32768, 1)]
        assert "every pairing" in c["not_measured"][0]["reason"]


def test_the_step_layout_names_this_runners_directories_for_t31s_steps(tmp_path):
    import t31_steps as st
    p = S.write_step_layout(T61, tmp_path)
    lay = json.loads(p.read_text())
    assert set(lay) == set(S.T31_STEPS)
    for step in S.T31_STEPS:
        assert st.step_dir(tmp_path, step) == S.step_dir(tmp_path, step, T61)
    assert st.step_dir(tmp_path, "link") != tmp_path / "02_link"          # T3.1's own numbering
    with pytest.raises(st.StepFailed, match="names no directory for the step 'verify'"):
        st.step_dir(tmp_path, "verify")
    assert st.step_dir(tmp_path / "t31", "predict") == tmp_path / "t31" / "07_predict"
    assert S.write_step_layout(dataclasses.replace(T61, components_config=""), tmp_path / "x") is None


def test_a_real_run_writes_the_step_layout_before_its_first_step(tmp_path, monkeypatch):
    import t31_steps as st
    seen = []

    def runner(cmd, log):
        seen.append((tmp_path / "s" / st.STEP_LAYOUT).exists())
        return 1
    monkeypatch.setattr(S, "tree_clean", lambda: True)
    monkeypatch.setattr(S, "boot_id", lambda: "boot-a")
    S.run_session(T61, tmp_path / "s", start="prereqs", extra=[], dry_run=False, sync=False,
                  runner=runner)
    assert seen == [True]
    assert not (tmp_path / "d" / st.STEP_LAYOUT).exists()
    dry(T61, tmp_path / "d")
    assert not (tmp_path / "d" / st.STEP_LAYOUT).exists()               # a dry run's fakes need none


def test_the_launch_queue_reports_the_q_link_measured(tmp_path):
    d = S.step_dir(tmp_path, "link", T61)
    args = args_for(tmp_path, tmp_path / "o", dry_run=False)
    with pytest.raises(S.StepFailed, match="`link` has not passed"):
        S.real_launch_queue(T61, args)
    S.write_json(d / "recursion_inputs.json", {"queue_depth": 1022, "fake": True})
    with pytest.raises(S.StepFailed, match="`link` has not passed"):
        S.real_launch_queue(T61, args)
    S.write_json(d / "recursion_inputs.json", {"queue_depth": 1022, "fake": False})
    assert S.real_launch_queue(T61, args) == {"q": 1022, "from": "link"}




# --- the real harness (adviser's ruling 9 of the tenth 9 Nov 2026 paste) --------------------

def _harness_setup(tmp_path, monkeypatch):
    """P1's grid stands in for P2's config (whose lock values are unset), and
    a fake T3.1 step writes one eager group's untraced and traced files."""
    import t31_steps as st
    comp, grid = p1_grid()
    monkeypatch.setattr(st, "load_config", lambda path, dry_run: json.loads(Path(path).read_text())
                        if Path(path).name == "block_config.json" else comp)
    seen = []

    def step(ns, cfg, *, raise_=None):
        seen.append((ns, cfg))
        out = Path(ns.out)
        write_t31_gt(out, "eager", [t31_row("dense", "dense", None, 2048, 1, 1951)],
                     not_before=T, firsts=[T + 1], complete=raise_ is None)
        S.write_json(out / "eager" / "traced_L2048_b1.json", {
            "traced": {"rows": []}, "traces": ["traces/trace_L2048_b1_0_dense.json"]})
        if raise_:
            raise raise_
    out = tmp_path / "gt"
    out.mkdir()
    args = Namespace(out=str(out), session=str(tmp_path), dry_run=False)
    return st, comp, grid, seen, step, args


def test_the_real_harness_runs_t31s_ground_truth_on_the_pairings_config(tmp_path, monkeypatch):
    st, comp, grid, seen, step, args = _harness_setup(tmp_path, monkeypatch)
    r = S.real_harness(T61, args, step_fn=step)
    ns, cfg = seen[0]
    assert ns.config == str(ROOT / T61.components_config) and ns.dry_run is False
    assert cfg is comp and ns.out == args.out and ns.session == args.session
    assert r["complete"] is True and r[S.STRICT_KEY] == T
    assert [x["cell"] for x in r["untraced"]["rows"]] == [["dense", 1.0, 2048, 1]]
    assert r["traced_files"] == ["eager/traced_L2048_b1.json"]
    assert r["traces"] == ["eager/traces/trace_L2048_b1_0_dense.json"]


def test_the_trace_checks_read_each_trace_from_the_ground_truth_output(tmp_path, monkeypatch):
    gt = S.step_dir(tmp_path, "ground_truth", T61)
    S.write_json(gt / "traces.json", ["eager/traces/t_0_dense.json", "compiled/traces/t_0_dense.json"])
    cmds = []
    monkeypatch.setattr(S, "run", lambda cmd, *a, **k: cmds.append(cmd) or 0)
    args = Namespace(out=str(tmp_path / "tc"), session=str(tmp_path))
    assert S.real_trace_checks(T61, args) == {"traces": 2}
    assert [c[c.index("kineto") + 1] for c in cmds] == [
        str(gt / "eager/traces/t_0_dense.json"), str(gt / "compiled/traces/t_0_dense.json")]
    outs = [c[c.index("--out") + 1] for c in cmds]
    assert len(set(outs)) == 2          # two groups' traces of one arm keep their own output
    S.write_json(gt / "traces.json", [])
    with pytest.raises(S.StepFailed, match="no traces listed"):
        S.real_trace_checks(T61, args)


def test_the_real_harness_runs_a_block_on_the_blocks_grid_only(tmp_path, monkeypatch):
    st, comp, grid, seen, step, args = _harness_setup(tmp_path, monkeypatch)
    block = {"id": "b2", "anchor": grid[0], "cells": [grid[1]], "run_cells": [grid[0], grid[1]]}
    S.real_harness(T61, args, block, step_fn=step)
    ns, cfg = seen[0]
    assert Path(ns.config) == Path(args.out) / "block_config.json"
    assert cfg == json.loads(json.dumps(S.block_config(comp, block)))
    assert {k for k in map(tuple, st.cells(cfg))} == {tuple(grid[0]), tuple(grid[1])}


def test_the_real_harness_reports_an_incomplete_cell_and_passes_on_a_refusal(tmp_path, monkeypatch):
    st, comp, grid, seen, step, args = _harness_setup(tmp_path, monkeypatch)
    r = S.real_harness(T61, args, step_fn=lambda ns, cfg: step(
        ns, cfg, raise_=st.StepIncomplete("1 cell short")))
    assert r["complete"] is False
    with pytest.raises(S.StepFailed, match="group eager failed"):
        S.real_harness(T61, args, step_fn=lambda ns, cfg: step(
            ns, cfg, raise_=st.StepFailed("ground truth for group eager failed")))


def test_predict_leaves_t31s_receipt_where_t31s_steps_read_it(tmp_path):
    import t31_steps as st
    session = tmp_path / "s"
    assert dry(T61, session) == 0
    S.write_step_layout(T61, session)
    receipt, local = st._receipt(session)
    ours = json.loads((session / S.RECEIPT).read_text())
    assert receipt.sha256 == ours["sha256"] and local == Path(ours["local"])
    assert receipt.synced_at == datetime.fromisoformat(ours["synced_at"])
    assert receipt.remote.endswith(ours["remote_path"])


def t31_row(arm, label, density, budget, batch, tokens, median=1.0):
    return {"arm": label, "cell_arm": arm, "nominal_density": density, "budget": budget,
            "batch": batch, "context_length": tokens, "prefill_ms_samples": [median],
            "prefill_ms_median": median}


def test_a_t31_row_names_its_grid_cell_not_its_label_or_prompt_length():
    assert S.t31_row_cell(t31_row("dense", "dense", None, 2048, 1, 1951)) == ["dense", 1.0, 2048, 1]
    assert S.t31_row_cell(t31_row("oracle", "oracle@0.25", 0.25, 16384, 4, 16001)) == \
        ["oracle", 0.25, 16384, 4]
    assert S.t31_row_cell(t31_row("xattention", "xattention", None, 4096, 1, 4000)) == \
        ["xattention", None, 4096, 1]


def p1_grid():
    import t31_steps as st
    comp = st.load_config(str(ROOT / "configs" / "t31" / "p1_session.json"), dry_run=False)
    return comp, st.cells(comp)


def test_a_blocks_components_config_times_only_its_cells_and_the_anchor():
    import t31_steps as st
    comp, grid = p1_grid()
    assert grid[7][:3] == ("dense", 1.0, 4096) and grid[9][2] == 4096
    block = {"id": "b3", "anchor": grid[0], "cells": [grid[5], grid[7], grid[9]]}
    block["run_cells"] = [block["anchor"]] + block["cells"]
    sub = S.block_config(comp, block)
    assert st.cells(sub) == [grid[0], grid[5], grid[7], grid[9]]
    assert sub["lengths"] == [2048, 4096] and sub["batches"] == {"2048": [1], "4096": [1]}
    assert st.cells(comp) == grid                                   # the pairing's is untouched
    assert {e["reason"] for e in sub["not_measured"][len(comp["not_measured"]):]} == \
        {"another block's cell (this is block b3)"}
    with pytest.raises(S.StepFailed, match="not in the grid"):
        S.block_config(comp, dict(block, run_cells=[("dense", 1.0, 999, 1)]))


def test_a_block_with_cells_at_a_length_but_not_its_dense_control_is_refused():
    """T3.1 times the dense control at every length it visits
    (`t31_steps.group_arms`): with no dense cell at 4096 in the block, its
    rows would hold dense at 4096 and `tag_rows` would refuse them after the
    clocking. The block's config refuses first."""
    import t31_steps as st
    comp, grid = p1_grid()
    block = {"id": "b3", "anchor": grid[0], "cells": [grid[5], grid[9]]}
    block["run_cells"] = [block["anchor"]] + block["cells"]
    with pytest.raises(S.StepFailed, match=r"\[\(4096, 1\)\].*without the dense cell"):
        S.block_config(comp, block)
    sub = S.block_config(comp, dict(block, run_cells=block["run_cells"] + [grid[7]]))
    for L in sub["lengths"]:                                    # every visited length's control
        names, _ = st.group_arms(dict(sub, gt_groups={"eager": ["dense", "oracle"]}), "eager", L, 1)
        assert names[0] == "dense" and ("dense", 1.0, L, 1) in {tuple(c) for c in st.cells(sub)}


def write_t31_gt(d, group, rows, *, not_before, firsts, traced=False, complete=True):
    g = d / group
    g.mkdir(parents=True, exist_ok=True)
    S.write_json(g / "stamps.json", {"group": group, "clock_not_before": not_before,
                                     "first_clocked_block_started_at": firsts})
    S.write_json(g / "gt_L2048_b1.json", {"rows": rows, "traced": traced, "complete": complete})


def test_t31s_ground_truth_reads_as_this_runners_result_with_the_earliest_stamp(tmp_path):
    comp, grid = p1_grid()
    a = t31_row("dense", "dense", None, 2048, 1, 1951)
    b = t31_row("oracle", "oracle@0.25", 0.25, 2048, 1, 1951)
    write_t31_gt(tmp_path, "eager", [a, b], not_before=T + 5, firsts=[T + 7])
    write_t31_gt(tmp_path, "compiled", [], not_before=T + 3, firsts=[T + 4])
    r = S.t31_ground_truth_result(tmp_path)
    assert r[S.STRICT_KEY] == T + 3 and r["complete"] is True
    assert [x["cell"] for x in r["untraced"]["rows"]] == [["dense", 1.0, 2048, 1],
                                                          ["oracle", 0.25, 2048, 1]]
    block = {"id": "b1", "anchor": grid[0], "cells": [grid[1]], "run_cells": [grid[0], grid[1]]}
    S.tag_rows(r, block)
    assert [x["anchor"] for x in r["untraced"]["rows"]] == [True, False]
    with pytest.raises(S.StepFailed, match="not this block's"):
        S.tag_rows(r, dict(block, run_cells=[grid[0], grid[2]]))


def test_the_block_drift_check_reads_t31s_rows_by_their_grid_cell(tmp_path):
    from attnbench.analysis import block_drift as BD
    comp, grid = p1_grid()
    rows = []
    for bid, med, tokens in (("b1", 700.0, 8075), ("b2", 750.0, 8075)):
        d = tmp_path / bid
        write_t31_gt(d, "eager", [t31_row("dense", "dense", None, 8192, 1, tokens, med)],
                     not_before=T, firsts=[T + 1])
        r = S.t31_ground_truth_result(d)
        anchor = ("dense", 1.0, 8192, 1)
        S.tag_rows(r, {"id": bid, "anchor": anchor, "cells": [], "run_cells": [anchor]})
        rows += r["untraced"]["rows"]
    out = BD.drift(rows, order=["b1", "b2"])
    assert out["anchor"] == ("dense", 1.0, 8192, 1)
    assert out["per_block"]["b2"]["relative_difference"] == pytest.approx(50 / 700)
    assert out["above"] == ["b2"] and out["passed"] is False


def test_t31s_ground_truth_is_incomplete_if_any_group_is_and_never_traced(tmp_path):
    row = t31_row("dense", "dense", None, 2048, 1, 1951)
    write_t31_gt(tmp_path, "eager", [row], not_before=T, firsts=[T + 1])
    write_t31_gt(tmp_path, "compiled", [row], not_before=T, firsts=[T + 1], complete=False)
    assert S.t31_ground_truth_result(tmp_path)["complete"] is False
    write_t31_gt(tmp_path, "compiled", [row], not_before=T, firsts=[T + 1], traced=True)
    with pytest.raises(S.StepFailed, match="traced"):
        S.t31_ground_truth_result(tmp_path)
    with pytest.raises(S.StepFailed, match="no ground-truth rows"):
        S.t31_ground_truth_result(tmp_path / "empty")


def test_the_real_record_needs_p1s_banking_time_first(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "P1_TRAINING", tmp_path / "absent.json.gz")
    w = datetime(2026, 11, 20, tzinfo=timezone.utc)
    with pytest.raises(S.StepFailed, match="missing: \\['P1'\\]"):
        S.real_record(T61, args_for(tmp_path, tmp_path, dry_run=False), w)
    banked = args_for(tmp_path, tmp_path, dry_run=False, banked=["P1=2026-11-19T00:00:00+00:00"])
    with pytest.raises(S.StepFailed, match="no P1 training file"):
        S.real_record(T61, banked, w)


def _p1_training_file(tmp_path):
    from attnbench.predict import p1_training as PT
    import test_p1_training as TP
    a = TP.session(tmp_path, "T3.2a", TP.gt([10, 20, 30], [5, 6, 7]))
    b = TP.session(tmp_path, "T3.2b", TP.gt([40, 50, 60, 70], [8, 9]))
    path = tmp_path / "p1_training.json.gz"
    PT.dump(PT.to_record(PT.training_rows(a, b), TP.HW), path)
    return path


def test_the_real_record_fits_the_sums_on_p1s_pooled_rows_and_waits_for_the_learned_config(
        tmp_path, monkeypatch):
    from attnbench.predict import baselines as BL
    monkeypatch.setattr(S, "P1_TRAINING", _p1_training_file(tmp_path))
    rows = S.training_rows(T61)
    assert sorted(r.measured for r in rows) == [pytest.approx(0.007), pytest.approx(0.040)]
    fits = []
    real = BL.fit_fitted_sum
    monkeypatch.setattr(BL, "fit_fitted_sum", lambda ts, **kw: fits.append((ts, kw)) or real(ts, **kw))
    a = args_for(tmp_path, tmp_path, dry_run=False, banked=["P1=2026-11-19T00:00:00+00:00"])
    with pytest.raises(S.StepFailed, match="lock value \\(Gate B\\)"):
        S.real_record(T61, a, datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert [kw for _, kw in fits] == [{}, {"add_s0": True}]
    ts = fits[0][0]
    assert ts.target == "P2" and ts.trained_on == ("P1",) and len(ts.rows) == 2


def test_a_damaged_training_file_is_refused(tmp_path, monkeypatch):
    import gzip
    path = _p1_training_file(tmp_path)
    raw = gzip.decompress(path.read_bytes()).replace(b'"target_s":0.04', b'"target_s":0.05')
    path.write_bytes(gzip.compress(raw))
    monkeypatch.setattr(S, "P1_TRAINING", path)
    with pytest.raises(S.StepFailed, match="sha256"):
        S.training_rows(T61)


GOOD_ROW = {"cloud_provider": "gcp", "instance_type": "a2-ultragpu-1g",
            "instance_generation": "a2", "cpu_microcode": "0x1", "provisioning_model": "spot",
            "provisioning_model_problem": None, "python_optimize": 0, "git_dirty": False,
            "gpu_name": "NVIDIA A100-SXM4-80GB", "null_launch_s_median": 4e-6}


def test_the_provenance_row_passes_on_a2_flex_or_spot():
    assert S.check_provenance_row(T61, GOOD_ROW, []) == []
    assert S.check_provenance_row(T61, {**GOOD_ROW, "provisioning_model": "dws-flex-start"}, []) == []


@pytest.mark.parametrize("key, value", [
    ("instance_type", "g2-standard-8"), ("provisioning_model", "on-demand"), ("provisioning_model", "SPOT"),
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


# --- lightgbm in the preflight (the 10 Nov L4 rehearsal stopped at step 11 without it) ----

def _lgbm_probe(monkeypatch, rc, out=""):
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, out, ""))


def test_lightgbm_at_the_pinned_version_is_no_problem(monkeypatch):
    from attnbench.predict.learned import LIGHTGBM_VERSION
    _lgbm_probe(monkeypatch, 0, LIGHTGBM_VERSION + "\n")
    assert S.lightgbm_problem() is None


def test_a_missing_or_wrong_lightgbm_is_a_preflight_problem(monkeypatch):
    _lgbm_probe(monkeypatch, 1)
    assert "does not import" in S.lightgbm_problem()
    _lgbm_probe(monkeypatch, 0, "3.3.5\n")
    assert "pinned to" in S.lightgbm_problem()


def test_the_real_preflight_stops_on_a_missing_lightgbm_before_any_step_is_timed(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from attnbench import provenance
    monkeypatch.setattr(S, "run", lambda *a, **k: 0)
    monkeypatch.setattr(S, "_host_speed_row", lambda out, tag: GOOD_ROW)
    monkeypatch.setattr(provenance, "stamp_integrity_problems", lambda row: [])
    monkeypatch.setattr(S, "check_provenance_row", lambda cfg, row, p: list(p))
    monkeypatch.setattr(S, "run_gpu_variants", lambda out: ({}, []))
    monkeypatch.setattr(S, "lightgbm_problem", lambda: "lightgbm does not import")
    monkeypatch.setattr(S, "import_check", lambda cfg: ([], {"required": {}, "kernel_problems": []}))
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "3.4.0\n", ""))
    with pytest.raises(S.StepFailed) as e:
        S.real_preflight(T61, args_for(tmp_path, tmp_path, dry_run=False))
    assert "lightgbm does not import" in e.value.findings["problems"]


# --- T2.26's GPU variants in the preflight ----------------------------------------------

def _junit(path, cases):
    """A JUnit record with one testcase per (name, kind), kind None for a pass."""
    body = "".join(f'<testcase name="{n}">' + (f"<{k}/>" if k else "") + "</testcase>"
                   for n, k in cases)
    path.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return path


def test_five_passing_gpu_variants_are_no_problem(tmp_path):
    counts, problems = S.gpu_variant_problems(
        _junit(tmp_path / "j.xml", [(f"t{i}", None) for i in range(5)]), 0)
    assert problems == [] and counts["passed"] == 5 and counts["skipped"] == 0


@pytest.mark.parametrize("kind", ["skipped", "failure", "error"])
def test_a_skipped_failed_or_erroring_gpu_variant_is_a_problem(tmp_path, kind):
    cases = [(f"t{i}", None) for i in range(4)] + [("t4", kind)]
    counts, problems = S.gpu_variant_problems(_junit(tmp_path / "j.xml", cases), 0)
    assert problems == [f"GPU variant {kind}: t4"] and counts[kind] == 1


def test_a_gpu_variant_count_other_than_five_is_a_problem(tmp_path):
    _, problems = S.gpu_variant_problems(
        _junit(tmp_path / "j.xml", [(f"t{i}", None) for i in range(4)]), 0)
    assert problems == ["4 GPU variants ran, not 5"]


def test_no_junit_record_or_a_bare_nonzero_exit_is_a_problem(tmp_path):
    assert S.gpu_variant_problems(tmp_path / "absent.xml", 4)[1]
    _, problems = S.gpu_variant_problems(
        _junit(tmp_path / "j.xml", [(f"t{i}", None) for i in range(5)]), 1)
    assert problems == ["pytest exited 1 on the GPU variants"]


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
