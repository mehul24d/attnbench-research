"""The L4 shakedown runner (T2.27): order, stop at the first failure, resume,
and the GPU steps' wiring, on CPU.

The runner's control flow is tested with a fake step process. The steps that
need CUDA are run here with their CUDA pieces replaced by `autospec` fakes,
which refuse any call the real functions would refuse, so a wrong argument
name or a missing attribute fails here and not on a billed L4. The steps that
run on a CPU (T2.1/T2.20, T2.5, T2.6 and the T2.21 report on the dense arm)
run for real in dry-run mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t227_shakedown as sh  # noqa: E402
import t227_steps as st  # noqa: E402

CHECKLIST = ("t21_t220", "zoo", "link", "launch_queue", "nsys", "replay", "probe",
             "assert_gate", "graph_breaks", "null_kernel", "figure1")


# --- the runner --------------------------------------------------------------

def fake_runner(outcomes: dict, ran: list):
    """A step process that writes the result.json `outcomes` names (passed
    by default) and exits as the real step would."""
    def run(cmd, log):
        step = cmd[cmd.index("--out") - 1]
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        log.write_text(f"ran {step}\n")
        status, rc = outcomes.get(step, ("passed", 0))
        if status is not None:
            (out / "result.json").write_text(json.dumps(
                {"step": step, "status": status, "reason": f"{status} by the test"}))
        ran.append(step)
        return rc
    return run


def test_the_checklist_is_the_one_asked_for_in_order():
    assert st.STEPS == CHECKLIST
    assert tuple(sh.CHECKLIST) == CHECKLIST


def test_every_step_passes_and_writes_a_log_each(tmp_path):
    ran = []
    rc = sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                        runner=fake_runner({}, ran))
    assert rc == 0 and ran == list(CHECKLIST)
    for i, s in enumerate(CHECKLIST, 1):
        assert (tmp_path / f"{i:02d}_{s}" / "step.log").read_text() == f"ran {s}\n"
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [a["status"] for a in m["attempts"]] == ["passed"] * len(CHECKLIST)


def test_it_stops_at_the_first_failure(tmp_path):
    ran = []
    rc = sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                        runner=fake_runner({"link": ("failed", 1), "probe": ("failed", 1)}, ran))
    assert rc == 1
    assert ran == ["t21_t220", "zoo", "link"]
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["attempts"][-1]["step"] == "link" and m["attempts"][-1]["status"] == "failed"


@pytest.mark.parametrize("outcome", [("skipped", 3), (None, 1), ("passed", 1)])
def test_a_skip_a_missing_result_or_a_failed_exit_stops_a_real_run(tmp_path, outcome):
    ran = []
    rc = sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                        runner=fake_runner({"zoo": outcome}, ran))
    assert rc == 1 and ran == ["t21_t220", "zoo"]


def test_a_skip_does_not_stop_a_dry_run(tmp_path):
    ran = []
    rc = sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=True, sync=False,
                        runner=fake_runner({"zoo": ("skipped", 3)}, ran))
    assert rc == 0 and ran == list(CHECKLIST)


def test_resume_refuses_when_an_earlier_step_has_not_passed(tmp_path):
    ran = []
    sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                   runner=fake_runner({"link": ("failed", 1)}, ran))
    ran.clear()
    rc = sh.run_session(tmp_path, start="launch_queue", extra=[], dry_run=False, sync=False,
                        runner=fake_runner({}, ran))
    assert rc == 2 and ran == []


def test_keep_going_runs_every_step_past_a_failure_and_still_exits_1(tmp_path):
    ran = []
    rc = sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                        runner=fake_runner({"t21_t220": ("failed", 1), "probe": ("failed", 1)},
                                           ran), keep_going=True)
    assert rc == 1 and ran == list(CHECKLIST)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [a["step"] for a in m["attempts"] if a["status"] == "failed"] == ["t21_t220", "probe"]


def test_keep_going_lets_from_skip_a_failed_earlier_step(tmp_path):
    ran = []
    sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                   runner=fake_runner({"t21_t220": ("failed", 1)}, ran))
    assert ran == ["t21_t220"]
    ran.clear()
    rc = sh.run_session(tmp_path, start="zoo", extra=[], dry_run=False, sync=False,
                        runner=fake_runner({}, ran), keep_going=True)
    assert ran == list(CHECKLIST[1:])
    assert rc == 1                              # step 1 is still failed in this session


def test_keep_going_is_a_runner_option():
    assert sh.main(["--list", "--keep-going"]) == 0


def test_resume_keeps_the_failed_attempt_beside_the_fix(tmp_path):
    ran = []
    sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                   runner=fake_runner({"nsys": ("failed", 1)}, ran))
    ran.clear()
    rc = sh.run_session(tmp_path, start="nsys", extra=[], dry_run=False, sync=False,
                        runner=fake_runner({}, ran))
    assert rc == 0 and ran == list(CHECKLIST[CHECKLIST.index("nsys"):])
    kept = [p.name for p in tmp_path.iterdir() if p.name.startswith("05_nsys.failed.")]
    assert len(kept) == 1
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [a["status"] for a in m["attempts"] if a["step"] == "nsys"] == ["failed", "passed"]


def test_a_dry_session_cannot_be_continued_as_a_real_one(tmp_path):
    sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=True, sync=False,
                   runner=fake_runner({}, []))
    assert sh.run_session(tmp_path, start=CHECKLIST[0], extra=[], dry_run=False, sync=False,
                          runner=fake_runner({}, [])) == 2


def test_each_real_step_runs_through_run_phase_so_it_is_synced():
    cmd = sh.step_command("replay", Path("/s/06_replay"), ["--length", "8192"],
                          dry_run=False, sync=True)
    assert cmd[0] == "bash" and cmd[1].endswith("scripts/run_phase.sh")
    assert cmd[2] == "t227_06_replay" and cmd[3] == "--"
    assert cmd[5].endswith("t227_steps.py") and "--dry-run" not in cmd
    dry = sh.step_command("replay", Path("/s"), [], dry_run=True, sync=False)
    assert dry[0] == sys.executable and dry[-1] == "--dry-run"


def test_a_real_run_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "tree_clean", lambda: False)
    monkeypatch.setattr(sh, "run_session", lambda *a, **k: pytest.fail("ran"))
    assert sh.main(["--session", str(tmp_path)]) == 2


def test_step_options_are_checked_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "run_session", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit):
        sh.main(["--dry-run", "--session", str(tmp_path), "--replay-arm", "nonesuch"])


# --- the CPU steps, for real --------------------------------------------------

def _args(tmp_path, **kw):
    a = st.parser().parse_args(["t21_t220", "--out", str(tmp_path), "--dry-run"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("step", ["t21_t220", "link", "launch_queue"])
def test_the_cpu_steps_pass_a_dry_run(tmp_path, step):
    assert st.run_step(step, _args(tmp_path)) == 0
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["status"] == "passed" and res["dry_run"] is True


def test_the_dense_graph_break_report_runs_on_cpu(tmp_path):
    assert st.run_step("graph_breaks", _args(tmp_path, arms="dense", gb_length=256)) == 0
    for setting in ("default", "capture"):
        rep = json.loads((tmp_path / f"dense@{setting}" / "graph_breaks.json").read_text())
        assert rep["variant"] == f"dense@compile-{setting}" and len(rep["layers"]) == 2
    res = json.loads((tmp_path / "result.json").read_text())
    assert "settings_identical" in res["findings"]["dense"]


@pytest.mark.parametrize("step", ["zoo", "nsys", "replay", "probe", "assert_gate",
                                  "null_kernel", "figure1"])
def test_the_gpu_steps_skip_a_cpu_dry_run_and_fail_a_real_one(tmp_path, step, monkeypatch):
    monkeypatch.setattr(st, "_cuda", lambda: False)
    assert st.run_step(step, _args(tmp_path)) == 3
    assert st.run_step(step, _args(tmp_path, dry_run=False)) == 1


# --- the GPU steps' wiring, with autospec fakes --------------------------------

class FakeArm(SimpleNamespace):
    pass


def fake_harness(arm):
    return SimpleNamespace(ready=lambda name, L, index=0: arm, prepare=lambda *a, **k: arm,
                           unwrap=lambda: None)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(st, "_need_cuda", lambda args, what: None)
    monkeypatch.setattr(st, "_cuda", lambda: True)
    monkeypatch.setattr(st, "_sequence_summary", lambda seq: {"ops": 3})
    import torch
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)


def _replay_fakes(monkeypatch, agreement):
    from attnbench.components import op_sequence as ops, recorder, replay
    seq = object()
    rec = SimpleNamespace(sequence=seq, directory=Path("/rec"), tape=[1, 2],
                          masks=SimpleNamespace(masks=[0]))
    rec.tape = mock.MagicMock(__len__=lambda s: 2, substitutions=1)
    m = SimpleNamespace(device_s={0: 1.0}, complete=False, unmatched_positions=(4,),
                        families_unmatched={"x": 1}, extra_nodes=())
    calls = {}
    for mod, name, ret in [
            (recorder, "record_arm_prefill", rec), (replay, "capture", "cap"),
            (replay, "mask_agreement", agreement), (replay, "time_replay", {"median_s": 1.0}),
            (replay, "profile_replay", Path("/t.json")), (replay, "replay_nodes", []),
            (replay, "match", m), (ops, "device_durations", {0: 1.0}),
            (replay, "replay_vs_trace", {"families": {}}),
            (replay, "isolated_replay", Path("/i.json")),
            (replay, "cache_effect", {"isolated_over_in_order": 1.1}),
            (replay, "price_positions", SimpleNamespace(counts=lambda: {"graph": 1})),
            (replay, "to_record", {"profiled_over_unprofiled": 1.0})]:
        calls[name] = mock.create_autospec(getattr(mod, name), return_value=ret)
        monkeypatch.setattr(mod, name, calls[name])
    return calls


def test_replay_calls_every_t24_piece_with_arguments_it_accepts(tmp_path, gpu, monkeypatch):
    arm = FakeArm(name="meanpool", run_prefill=lambda: None, code_path="cp",
                  prompt_id="qa_1-0", layers=(), length=8075)
    monkeypatch.setattr(st, "_harness", lambda args: fake_harness(arm))
    calls = _replay_fakes(monkeypatch, {"identical": True})
    assert st.run_step("replay", _args(tmp_path, dry_run=False)) == 0
    for name, c in calls.items():
        assert c.called, name
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["findings"]["complete"] is False      # recorded, not failed (a dry run)
    assert (tmp_path / "replay_record.json").exists()
    # the match is incomplete, so the original operators are replayed alone and priced
    iso = calls["isolated_replay"].call_args_list
    assert [c.kwargs.get("substitute", True) for c in iso] == [True, False]
    assert calls["price_positions"].call_args.args[2] is calls["match"].return_value
    assert calls["to_record"].call_args.kwargs["pricing"] is calls["price_positions"].return_value
    assert res["findings"]["priced"] is True and res["findings"]["cost_sources"] == {"graph": 1}
    # the prompt's own length, not the 8192 budget, is what is recorded
    assert calls["record_arm_prefill"].call_args.kwargs["context_length"] == 8075
    assert (res["findings"]["length"], res["findings"]["budget"]) == (8075, 8192)


def test_replay_fails_when_the_live_masks_differ_from_the_bank(tmp_path, gpu, monkeypatch):
    arm = FakeArm(name="meanpool", run_prefill=lambda: None, code_path="cp",
                  prompt_id="qa_1-0", layers=(), length=8075)
    monkeypatch.setattr(st, "_harness", lambda args: fake_harness(arm))
    _replay_fakes(monkeypatch, {"identical": False})
    assert st.run_step("replay", _args(tmp_path, dry_run=False)) == 1


def test_probe_extends_and_checks_each_arm(tmp_path, gpu, monkeypatch):
    from attnbench.components import op_sequence as ops, recorder, structural_probe as sp
    arm = FakeArm(name="x", run_prefill=lambda: None, code_path="cp", prompt_id="p", layers=(),
                  length=8075)
    monkeypatch.setattr(st, "_harness", lambda args: fake_harness(arm))
    rec = mock.create_autospec(recorder.record_arm_prefill, return_value=SimpleNamespace(
        sequence=SimpleNamespace(context_length=8075)))
    monkeypatch.setattr(recorder, "record_arm_prefill", rec)
    ext = SimpleNamespace(verdict=ops.SAME, reason="", chunks={2048: 1}, payload_exponents=(),
                          extended=True, sequence="ext", checked_at=None)
    monkeypatch.setattr(sp, "extend", mock.create_autospec(sp.extend, return_value=ext))
    monkeypatch.setattr(sp, "probe", mock.create_autospec(sp.probe, return_value={}))
    verdicts = iter([ops.SAME, ops.DIFFERENT])
    monkeypatch.setattr(sp, "check_against_full", mock.create_autospec(
        sp.check_against_full, side_effect=lambda e, f: SimpleNamespace(
            verdict=next(verdicts), first_difference=None, reason="")))
    monkeypatch.setattr(st, "_save_sequence", lambda seq, path: None)
    assert st.run_step("probe", _args(tmp_path, dry_run=False, arms="dense,oracle")) == 1
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["findings"]["dense"]["passed"] is True
    assert res["findings"]["oracle"]["passed"] is False
    assert "oracle" in res["reason"] and "dense" not in res["reason"]
    assert sp.probe.call_args_list[0].kwargs["lengths"] == (2048, 4096)
    assert sp.extend.call_args_list[0].args[2] == 8075      # the full recording's length


def test_probe_takes_a_chunked_arm_at_three_lengths_checked_at_the_third(tmp_path, gpu,
                                                                         monkeypatch):
    from attnbench.components import op_sequence as ops, recorder, structural_probe as sp
    arm = FakeArm(name="xattention", run_prefill=lambda: None, code_path="cp", prompt_id="p",
                  layers=(), length=8075)
    monkeypatch.setattr(st, "_harness", lambda args: fake_harness(arm))
    monkeypatch.setattr(recorder, "record_arm_prefill", mock.create_autospec(
        recorder.record_arm_prefill, return_value=SimpleNamespace(
            sequence=SimpleNamespace(context_length=8075))))
    ext = SimpleNamespace(verdict=ops.SAME, reason="", chunks={}, payload_exponents=(),
                          extended=True, sequence="ext", checked_at=34700)
    checked = mock.create_autospec(sp.extend_checked, return_value=ext)
    two = mock.create_autospec(sp.extend, return_value=ext)
    monkeypatch.setattr(sp, "extend_checked", checked)
    monkeypatch.setattr(sp, "extend", two)
    monkeypatch.setattr(sp, "probe", mock.create_autospec(sp.probe, return_value={}))
    monkeypatch.setattr(sp, "check_against_full", mock.create_autospec(
        sp.check_against_full, return_value=SimpleNamespace(
            verdict=ops.SAME, first_difference=None, reason="")))
    monkeypatch.setattr(st, "_save_sequence", lambda seq, path: None)
    assert st.run_step("probe", _args(tmp_path, dry_run=False, arms="xattention")) == 0
    assert sp.probe.call_args.kwargs["lengths"] == (20480, 28672, 34816)
    assert checked.call_args.args[2] == 8075 and not two.called
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["findings"]["xattention"]["checked_at"] == 34700


def test_figure1_records_its_pair_at_16384_not_at_the_cell_length(tmp_path, gpu, monkeypatch):
    import contextlib
    from attnbench.components import op_sequence as ops, trace_recording as tr
    arm = FakeArm(name="oracle", run_prefill=lambda: None, code_path="cp", prompt_id="p",
                  layers=(), length=16301)
    lengths = []
    monkeypatch.setattr(st, "_harness", lambda args: SimpleNamespace(
        ready=lambda name, L, index=0: lengths.append(L) or arm))
    monkeypatch.setattr(st, "_host_compute_around_mask_build",
                        lambda arm: contextlib.nullcontext())
    monkeypatch.setattr(tr, "record_prefill_trace", mock.create_autospec(
        tr.record_prefill_trace, side_effect=lambda fn, path, **k: path))
    marker = SimpleNamespace(layer=14, kind=ops.SeqKind.HOST_COMPUTE)
    parse = mock.create_autospec(ops.parse_trace,
                                 return_value=SimpleNamespace(ops=[marker]))
    monkeypatch.setattr(ops, "parse_trace", parse)
    monkeypatch.setattr(ops, "compare", mock.create_autospec(ops.compare, return_value=(
        SimpleNamespace(verdict=ops.SAME, first_difference=None, reason=""))))
    monkeypatch.setattr(st, "_crop_layer", lambda p, layer, out: {"layer_us": 1.0})
    monkeypatch.setattr(st, "_save_sequence", lambda seq, path: None)
    assert st.parser().parse_args(["figure1", "--out", "x"]).fig1_length == 16384
    assert st.run_step("figure1", _args(tmp_path, dry_run=False)) == 0
    assert lengths == [16384]                         # the budget asked for
    assert {c.kwargs["context_length"] for c in parse.call_args_list} == {16301}
    res = json.loads((tmp_path / "result.json").read_text())
    assert (res["findings"]["length"], res["findings"]["budget"]) == (16301, 16384)
    assert res["findings"]["pair"] == ops.SAME


def test_the_figure1_crop_keeps_one_layer_and_its_device_work(tmp_path):
    from attnbench.components import op_sequence as ops
    ev = [
        {"ph": "M", "name": "process_name", "pid": 1},
        {"ph": "X", "cat": "user_annotation", "name": ops.layer_annotation(0), "pid": 1, "tid": 1, "ts": 0, "dur": 10},
        {"ph": "X", "cat": "user_annotation", "name": ops.layer_annotation(1), "pid": 1, "tid": 1, "ts": 10, "dur": 10},
        {"ph": "X", "cat": "user_annotation", "name": ops.HOST_COMPUTE, "pid": 1, "tid": 1, "ts": 11, "dur": 4},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 1, "ts": 16, "dur": 1,
         "args": {"correlation": 7}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 1, "ts": 2, "dur": 1,
         "args": {"correlation": 3}},
        {"ph": "X", "cat": "kernel", "name": "k7", "pid": 0, "tid": 7, "ts": 40, "dur": 5, "args": {"correlation": 7}},
        {"ph": "X", "cat": "kernel", "name": "k3", "pid": 0, "tid": 7, "ts": 30, "dur": 5, "args": {"correlation": 3}},
    ]
    src = tmp_path / "t.json"
    src.write_text(json.dumps({"traceEvents": ev, "deviceProperties": []}))
    r = st._crop_layer(src, 1, tmp_path / "c.json")
    kept = json.loads((tmp_path / "c.json").read_text())["traceEvents"]
    names = [e["name"] for e in kept]
    assert "k7" in names and "k3" not in names and ops.HOST_COMPUTE in names
    assert r["host_compute_us"] == 4 and r["device_events"] == 1


# --- prompt length (Shakedown 1, 28 Oct 2026: "qa_1_2048_0: 1951 tokens, not 2048") ----------

def _real_prompt_harness(monkeypatch, n_tokens):
    import torch
    from attnbench.accuracy import prompting
    from attnbench.components import shakedown_arms as sa
    h = sa.Harness(device="cpu", dry_run=True)
    h.dry_run = False                       # the real-prompt path, with a stub example
    h.examples = lambda length, n, task=sa.TASK: [SimpleNamespace(example_id="qa_1_2048_0",
                                                                  context="c")] * n
    monkeypatch.setattr(prompting, "encode_prompt",
                        lambda tok, text, device: torch.ones(1, n_tokens, dtype=torch.long))
    return h


def test_a_real_prompt_below_its_budget_is_used_at_its_own_length(monkeypatch):
    h = _real_prompt_harness(monkeypatch, 1951)
    ids, pid = h.input_ids(2048)
    assert ids.shape[-1] == 1951 and pid == "qa_1_2048_0"
    arm = h.prepare("dense", 2048)
    assert (arm.length, arm.budget) == (1951, 2048)
    assert arm.extra["cfg"].seq_len == 1951 and arm.extra["input_ids"].shape[-1] == 1951


def test_a_real_prompt_over_its_budget_is_refused(monkeypatch):
    h = _real_prompt_harness(monkeypatch, 2049)
    with pytest.raises(RuntimeError, match="2049 tokens, over the 2048-token budget"):
        h.input_ids(2048)
