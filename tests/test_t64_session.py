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
            "provisioning_model": "FLEX_START", "gpu_name": "NVIDIA H100 80GB HBM3"}


def test_the_h100_row_must_be_flex_start():
    assert S.check_provenance_row(T64, H100_ROW, []) == []
    assert S.check_provenance_row(T64, {**H100_ROW, "provisioning_model": "SPOT"}, [])
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
