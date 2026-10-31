"""The P4 session runner (T6.4) on CPU: what differs from P2 (the FA3 import
check before K2, the Triton mask recording, Flex-start only, training on P1
and P2, the null-kernel floor finding). The shared order rules are tested in
`tests/test_t61_session.py`; one end-to-end dry run here checks they hold
with P4's step list.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t61_session as S  # noqa: E402
import t64_session as P4  # noqa: E402

from test_t61_session import GOOD_ROW, inproc  # noqa: E402

T64 = P4.T64
SHA = "ab" * 32
STEPS = ("prereqs", "preflight", "link", "launch_queue", "probe", "masks",
         "host_components", "device_components", "fa3_import", "dense_select",
         "host_speed_mid", "predict", "ground_truth", "late_check", "trace_checks",
         "host_speed_end")


def args_for(tmp_path, *, sha=SHA, dry_run=False):
    return Namespace(session=str(tmp_path), out=str(tmp_path), dry_run=dry_run,
                     banked_at=[], fa3_sha256=sha)


def test_the_p4_steps_put_fa3_before_k2_and_record_masks():
    assert T64.names == STEPS
    assert T64.names.index("fa3_import") < T64.names.index("dense_select") \
        < T64.names.index("predict")
    assert "mask_gate" not in T64.names           # the gate is P2's (Section 7)
    assert T64.provisioning == ("FLEX_START",) and T64.instance_type == "a3-highgpu-1g"
    assert T64.banked == ("P1", "P2")


def test_validate_refuses_fa3_after_k2():
    names = [n for n in T64.names if n != "fa3_import"]
    names.insert(names.index("dense_select") + 1, "fa3_import")
    steps = tuple(S.Step(n, "T", "w") for n in names)
    with pytest.raises(RuntimeError, match="fa3_import must come before dense_select"):
        S.validate(dataclasses.replace(T64, steps=steps))


def test_the_dry_run_passes_in_their_own_processes(tmp_path):
    session = tmp_path / "s"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "t64_session.py"), "run",
                        "--dry-run", "--fa3-sha256", SHA, "--session", str(session)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    st = S.last_status(json.loads((session / "manifest.json").read_text()))
    assert st["prereqs"] == "skipped" and all(st[s] == "passed" for s in STEPS[1:])
    log = (session / "09_fa3_import" / "step.log").read_text()
    assert "t64_session.py step fa3_import" in log and f"--fa3-sha256 {SHA}" in log
    rec = json.loads((session / S.RECEIPT).read_text())
    assert rec["remote_path"].startswith("predictions/P4/")
    gt = json.loads((session / "13_ground_truth" / "result.json").read_text())["findings"]
    assert gt["gap_s"] > S.MIN_GAP_S


@pytest.mark.parametrize("sha", [None, "", "AB" * 32, "ab" * 31, "zz" * 32])
def test_a_session_without_the_recorded_wheel_hash_is_refused_at_the_start(sha):
    a = Namespace(dry_run=True, banked_at=[], fa3_sha256=sha)
    assert "--fa3-sha256" in S.start_refusal(T64, a)


def test_a_real_start_needs_both_banking_times(monkeypatch):
    monkeypatch.setattr(S, "tree_clean", lambda: True)
    one = Namespace(dry_run=False, banked_at=["P1=2026-11-20T00:00:00+00:00"], fa3_sha256=SHA)
    assert "missing: ['P2']" in S.start_refusal(T64, one)
    both = Namespace(dry_run=False, fa3_sha256=SHA,
                     banked_at=["P1=2026-11-20T00:00:00+00:00", "P2=2026-11-25T00:00:00+00:00"])
    assert S.start_refusal(T64, both) is None


def _fake_tree(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fa3_import_check.sh").write_text("exit 0\n")
    monkeypatch.setattr(S, "ROOT", tmp_path)
    return tmp_path / "scripts" / "fa3_import_check.sh"


def test_the_fa3_check_runs_the_import_script_with_the_hash(tmp_path, monkeypatch):
    script = _fake_tree(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr(S, "run", lambda cmd, **kw: ran.append(cmd) or 0)
    f = P4.real_fa3_import(T64, args_for(tmp_path))
    assert ran == [["bash", str(script), SHA]]
    assert f == {"fa3_wheel_sha256": SHA, "available": True}


def test_the_fa3_check_fails_the_step_when_the_script_does(tmp_path, monkeypatch):
    script = _fake_tree(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr(S, "run", lambda cmd, **kw: ran.append(cmd) or 1)
    with pytest.raises(S.StepFailed, match="dense_set_incomplete"):
        P4.real_fa3_import(T64, args_for(tmp_path))
    assert ran == [["bash", str(script), SHA]]


def test_the_fa3_check_refuses_a_tree_without_the_script(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    with pytest.raises(S.StepFailed, match="not in the deployed tree"):
        P4.real_fa3_import(T64, args_for(tmp_path))


def test_the_fa3_step_refuses_a_bad_hash_before_running_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "run", lambda cmd, **kw: pytest.fail("ran"))
    with pytest.raises(S.StepFailed, match="64-hex"):
        P4.real_fa3_import(T64, args_for(tmp_path, sha="nope"))


H100_ROW = {**GOOD_ROW, "instance_type": "a3-highgpu-1g", "instance_generation": "a3",
            "provisioning_model": "dws-flex-start", "gpu_name": "NVIDIA H100 80GB HBM3"}


def test_the_h100_row_must_be_flex_start():
    assert S.check_provenance_row(T64, H100_ROW, []) == []
    assert S.check_provenance_row(T64, {**H100_ROW, "provisioning_model": "spot"}, [])
    assert S.check_provenance_row(T64, {**H100_ROW, "gpu_name": "NVIDIA A100"}, [])


def test_the_floor_is_reported_against_both_published_figures():
    f = P4.floor_findings({"null_launch_s_median": 4.7e-6})["null_kernel_floor"]
    assert f["p4_us"] == pytest.approx(4.7)
    assert f["ratio_to_published"]["TaxBreak, Xeon 8480C with H100"] == pytest.approx(1.0)
    assert f["ratio_to_published"]["SKIP, Xeon 8468V with H100 PCIe"] == \
        pytest.approx(4.7 / 2.37)
    assert P4.floor_findings({})["null_kernel_floor"]["ratio_to_published"] is None


def test_p4_requires_p2_banked_and_the_fa3_files():
    tasks = {r.task for r in T64.requirements}
    assert {"T6.1", "T3.1", "T4.4", "T2.16", "T2.23"} <= tasks
    files = {f for f, _ in T64.files}
    assert {"scripts/fa3_import_check.sh", "attnbench/backends/fa3.py"} <= files


def test_the_masks_step_waits_for_t2_23(tmp_path):
    with pytest.raises(S.NotBuilt, match=r"T2\.23"):
        T64.reals["masks"](T64, args_for(tmp_path))


def test_the_real_record_needs_both_banking_times(tmp_path):
    from datetime import datetime, timezone
    a = args_for(tmp_path)
    a.banked_at = ["P1=2026-11-19T00:00:00+00:00"]
    with pytest.raises(S.StepFailed, match=r"missing: \['P2'\]"):
        S.real_record(T64, a, datetime(2026, 12, 1, tzinfo=timezone.utc))


def test_an_in_process_dry_run_matches(tmp_path):
    calls = []
    rc = S.run_session(T64, tmp_path / "s", start="prereqs", extra=["--fa3-sha256", SHA],
                       dry_run=True, sync=False, runner=inproc(T64, calls))
    assert rc == 0 and calls == list(STEPS)


def test_the_steps_are_named_for_t64_in_run_phase(tmp_path):
    cmd = S.step_command(T64, "fa3_import", tmp_path, tmp_path, [], dry_run=False, sync=True)
    assert cmd[2] == "t64_09_fa3_import" and cmd[5].endswith("scripts/t64_session.py")


# --- the FA3 correctness gate (`--gate-only`) ------------------------------------------

GATE = P4.T64_GATE
GATE_STEPS = ("prereqs", "fa3_import", "fa3_gate", "halt")


def test_the_gate_is_four_untimed_steps_ending_in_a_halt():
    assert GATE.names == GATE_STEPS
    assert GATE.step_args == ("--gate-only",) and GATE.requirements == () and GATE.banked == ()
    assert GATE.instance_type == T64.instance_type and GATE.provisioning == T64.provisioning
    assert not set(GATE.names) & (set(T64.names) - {"prereqs", "fa3_import"})


@pytest.mark.parametrize("step", [s for s in STEPS if s not in GATE_STEPS])
def test_every_session_step_is_refused_in_gate_mode(tmp_path, step):
    with pytest.raises(SystemExit) as e:
        P4.main(["step", step, "--gate-only", "--session", str(tmp_path), "--out", str(tmp_path),
                 "--fa3-sha256", SHA])
    assert e.value.code == 2
    assert not (tmp_path / "result.json").exists()


def test_a_gate_run_cannot_start_from_a_session_step(tmp_path):
    with pytest.raises(SystemExit):
        P4.main(["run", "--gate-only", "--from", "ground_truth", "--session", str(tmp_path),
                 "--fa3-sha256", SHA])


def test_the_gate_needs_the_wheel_hash_and_no_banking_times(monkeypatch):
    monkeypatch.setattr(S, "tree_clean", lambda: True)
    a = Namespace(dry_run=False, fa3_sha256=None, banked_at=[])
    assert "--fa3-sha256" in S.start_refusal(GATE, a)
    a.fa3_sha256 = SHA
    assert S.start_refusal(GATE, a) is None


def test_the_gate_dry_run_passes_in_their_own_processes_with_the_flag(tmp_path):
    session = tmp_path / "g"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "t64_session.py"), "run",
                        "--gate-only", "--dry-run", "--fa3-sha256", SHA, "--session", str(session)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    m = json.loads((session / "manifest.json").read_text())
    assert m["task"] == "T6.4-gate"
    st = S.last_status(m)
    assert [st[s] for s in GATE_STEPS] == ["passed"] * 4
    for i, s in enumerate(GATE_STEPS, 1):
        log = (session / f"{i:02d}_{s}" / "step.log").read_text()
        assert f"t64_session.py step {s}" in log and "--gate-only" in log
    gate = json.loads((session / "03_fa3_gate" / "result.json").read_text())["findings"]
    assert gate["check_kinds"] == ["exact"] and gate["configs_checked"] == 2
    # a gate session is not a P4 session, and the reverse
    assert S.run_session(T64, session, start="prereqs", extra=[], dry_run=True, sync=False) == 2


def _gate_files(d, rows, **stamp):
    import pandas as pd
    from attnbench.provisioning import row_name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "correctness.parquet")
    S.write_json(d / "versions.json", {**H100_ROW, "provisioning_model": row_name("FLEX_START"),
                                       **stamp})


EXACT = {"backend": "fa3", "config_key": "c1", "passed": True, "check_kind": "exact",
         "max_abs_err": 2e-3}


def test_the_gate_passes_on_exact_checks_that_all_passed(tmp_path):
    _gate_files(tmp_path, [EXACT, {**EXACT, "config_key": "c2"},
                           {**EXACT, "backend": "sdpa_flash", "passed": False}])
    f = P4.read_gate(GATE, tmp_path)
    assert f["configs_checked"] == 2 and f["failed"] == [] and f["max_abs_err"] == 2e-3


@pytest.mark.parametrize("rows, stamp, why", [
    ([{**EXACT, "backend": "sdpa_flash"}], {}, "no configuration"),
    ([EXACT, {**EXACT, "config_key": "c2", "check_kind": "cross_backend"}], {}, "float64"),
    ([EXACT, {**EXACT, "config_key": "c2", "passed": False}], {}, "failed the float64 gate"),
    ([EXACT], {"gpu_name": "NVIDIA L4"}, "not this session's instance"),
    ([EXACT], {"provisioning_model": "spot"}, "not this session's instance"),
    ([EXACT], {"git_dirty": True}, "not this session's instance"),
])
def test_the_gate_fails(tmp_path, rows, stamp, why):
    _gate_files(tmp_path, rows, **stamp)
    with pytest.raises(S.StepFailed, match=why):
        P4.read_gate(GATE, tmp_path)


def test_the_real_gate_runs_the_probe_on_fa3_alone_below_the_oracles_length(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        _gate_files(Path(cmd[cmd.index("--out") + 1]), [EXACT])
        return 0
    monkeypatch.setattr(S, "run", fake_run)
    f = P4.real_fa3_gate(GATE, args_for(tmp_path))
    cmd = calls[0]
    assert cmd[1].endswith("run_probe.py")
    assert cmd[cmd.index("--only-backends") + 1] == "fa3"
    assert cmd[cmd.index("--max-seq") + 1] == "4096"
    assert "--skip-correctness" not in cmd and f["configs_checked"] == 1
    monkeypatch.setattr(S, "run", lambda cmd, **kw: 1)
    with pytest.raises(S.StepFailed, match="exited 1"):
        P4.real_fa3_gate(GATE, args_for(tmp_path / "b"))


def test_the_halt_schedules_a_shutdown_and_fails_loudly_if_it_cannot(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "run", lambda cmd, **kw: calls.append(cmd) or 0)
    assert P4.real_halt(GATE, args_for(tmp_path)) == {"halt_in_minutes": 1}
    assert calls == [["sudo", "shutdown", "-h", "+1"]]
    monkeypatch.setattr(S, "run", lambda cmd, **kw: 1)
    with pytest.raises(S.StepFailed, match="still billing"):
        P4.real_halt(GATE, args_for(tmp_path))
