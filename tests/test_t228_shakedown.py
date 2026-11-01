"""Shakedown 2's steps and their runner (`scripts/t228_steps.py`, run by
`scripts/t227_shakedown.py` with a different step list).

The steps that need a card are tested here by their wiring: the library
calls are replaced by fakes, and what is asserted is what each step decides
(what fails it, what it records). Nothing here says how the real NVML reader,
kernels or builder behave on an L4; that is what the session is for.
`t213` runs for real on a CPU with a scripted reader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import t227_shakedown as sh  # noqa: E402
import t228_steps as st  # noqa: E402
from test_t227_shakedown import fake_runner  # noqa: E402

STEPS = ("nvml", "t216", "t213", "density", "mask_ab", "t212", "t22", "queue")
CFG = st._config()


# --- the runner, reused ---------------------------------------------------------

def test_the_steps_are_the_ones_asked_for_in_the_order_they_depend_on_each_other():
    # t216 records the choice before t213's first clocked block reads it.
    assert st.STEPS == STEPS and tuple(st.CHECKLIST) == STEPS
    assert STEPS.index("t216") < STEPS.index("t213")


def test_the_same_runner_runs_every_step_and_stops_at_the_first_failure(tmp_path):
    ran = []
    assert sh.run_session(tmp_path, start="nvml", extra=[], dry_run=False, sync=False,
                          runner=fake_runner({}, ran), config=CFG) == 0
    assert ran == list(STEPS)
    assert (tmp_path / "03_t213").is_dir()
    ran2 = []
    s2 = tmp_path / "again"
    assert sh.run_session(s2, start="nvml", extra=[], dry_run=False, sync=False,
                          runner=fake_runner({"t216": ("failed", 1)}, ran2), config=CFG) == 1
    assert ran2 == ["nvml", "t216"]


def test_resuming_after_a_failure_needs_the_earlier_steps_to_have_passed(tmp_path):
    sh.run_session(tmp_path, start="nvml", extra=[], dry_run=False, sync=False,
                   runner=fake_runner({"t213": ("failed", 1)}, []), config=CFG)
    assert sh.run_session(tmp_path, start="t212", extra=[], dry_run=False, sync=False,
                          runner=fake_runner({}, []), config=CFG) == 2
    ran = []
    assert sh.run_session(tmp_path, start="t213", extra=[], dry_run=False, sync=False,
                          runner=fake_runner({}, ran), config=CFG) == 0
    assert ran == ["t213", "density", "mask_ab", "t212", "t22", "queue"]
    assert any(p.name.startswith("03_t213.failed.") for p in tmp_path.iterdir())


def test_a_step_runs_through_run_phase_under_this_sessions_prefix():
    cmd = sh.step_command("t213", Path("/s/03_t213"), ["--length", "8192"],
                          dry_run=False, sync=True, config=CFG)
    assert cmd[2] == "t228_03_t213" and cmd[3] == "--" and cmd[5].endswith("t228_steps.py")


def test_shakedown_1_is_unchanged_by_the_reuse():
    import t227_steps
    cmd = sh.step_command("replay", Path("/s/06_replay"), [], dry_run=False, sync=True)
    assert cmd[2] == "t227_06_replay" and cmd[5].endswith("t227_steps.py")
    assert sh.T227.steps == t227_steps.STEPS


def test_run_lists_and_checks_options_before_anything_runs(tmp_path, monkeypatch, capsys):
    assert st.main(["run", "--list"]) == 0
    assert "t213" in capsys.readouterr().out
    monkeypatch.setattr(sh, "run_session", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit):
        st.main(["run", "--dry-run", "--session", str(tmp_path), "--profile-arm", "nonesuch"])
    monkeypatch.setattr(sh, "tree_clean", lambda: False)
    assert st.main(["run", "--session", str(tmp_path)]) == 2          # a real run, dirty tree


# --- t213 for real, on a CPU --------------------------------------------------------------

@pytest.fixture(autouse=True)
def _light_tracer(monkeypatch):
    """The profiler's start-up dominates a CPU forward; what is tested is the
    harness's use of the tracer, not Kineto."""
    import contextlib
    monkeypatch.setattr(st, "_tracer", lambda: (lambda: contextlib.nullcontext()))


def _args(tmp_path, step="t213", **kw):
    a = st.parser().parse_args([step, "--out", str(tmp_path), "--dry-run"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_t213_passes_a_cpu_dry_run_with_a_scripted_reader(tmp_path):
    a = _args(tmp_path, arms="dense", length=128, reps=2, warmup=2, max_rerun=2, floor_ms=1e6)
    assert st.run_step("t213", a, st.STEP_FUNCS) == 0
    res = json.loads((tmp_path / "result.json").read_text())
    f = res["findings"]
    assert res["status"] == "passed" and f["complete"]
    assert f["stamp"]["inside_untraced_run"] and f["stamp"]["before_the_traced_run"]
    assert f["blocks_seen_by_under_two_samples"] == []
    assert f["sampler"]["achieved_hz"] > 5
    assert f["check_control"]["passed"] is None               # a CPU dry run has no K2 record
    assert [c["arm"] for c in f["sampler_cost"]] == ["dense"]
    assert (tmp_path / "ground_truth.json").exists() and (tmp_path / "sampler_cost.json").exists()


def test_t213_refuses_a_session_without_the_dense_control_first(tmp_path):
    a = _args(tmp_path, arms="meanpool,dense")
    assert st.run_step("t213", a, st.STEP_FUNCS) == 1
    assert "dense control must come first" in json.loads(
        (tmp_path / "result.json").read_text())["reason"]


def test_a_stamp_outside_the_run_fails_t213(tmp_path, monkeypatch):
    from attnbench import ground_truth as gt
    real = gt.time_with_profiler_overhead

    def late(*a, **k):
        out = real(*a, **k)
        out["first_clocked_block_started_at"] += 10_000.0
        return out
    monkeypatch.setattr(gt, "time_with_profiler_overhead", late)
    a = _args(tmp_path, arms="dense", length=128, reps=2, warmup=2, floor_ms=1e6)
    assert st.run_step("t213", a, st.STEP_FUNCS) == 1
    assert "outside the run" in json.loads((tmp_path / "result.json").read_text())["reason"]


def test_a_sampler_cost_is_the_run_with_it_minus_the_runs_without(tmp_path, monkeypatch):
    """B - mean(A1, A2) per arm, with |A1 - A2| as the noise."""
    from attnbench import ground_truth as gt
    seq = iter([10.0, 14.0])                       # the two no-sampler passes
    real = gt.time_arms

    def scripted(*a, controls, **k):
        out = real(*a, controls=controls, **k)
        if isinstance(controls.sampler, st.NullSampler):
            out["rows"][0]["prefill_ms_median"] = next(seq)
        else:
            out["rows"][0]["prefill_ms_median"] = 15.0
        return out
    monkeypatch.setattr(gt, "time_arms", scripted)
    a = _args(tmp_path, arms="dense", length=128, reps=2, warmup=2, floor_ms=1e6)
    st.run_step("t213", a, st.STEP_FUNCS)
    c = json.loads((tmp_path / "result.json").read_text())["findings"]["sampler_cost"][0]
    assert c["cost_ms"] == pytest.approx(3.0) and c["noise_ms"] == pytest.approx(4.0)


def test_check_control_is_read_against_t216s_record(tmp_path, monkeypatch):
    from attnbench import dense_selection as ds
    seen = {}
    ch = SimpleNamespace(chosen="sdpa_flash")
    monkeypatch.setattr(st, "_find_choice", lambda args: ch)
    monkeypatch.setattr(ds, "check_control",
                        lambda c, name, stamp, *, key: seen.update(name=name, stamp=stamp,
                                                                   key=key))
    a = _args(tmp_path, arms="dense", length=128, reps=2, warmup=2, floor_ms=1e6)
    st.run_step("t213", a, st.STEP_FUNCS)
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert f["check_control"] == {"passed": True, "chosen": "sdpa_flash"}
    assert seen["name"] == "sdpa_math" and seen["stamp"] == f["stamp"]["first_clocked_block_started_at"]
    assert (seen["key"].card, seen["key"].context_length, seen["key"].batch) == ("cpu", 128, 1)

    def refuse(c, name, stamp, *, key):
        raise ds.SelectionError("the dense control is not the choice")
    monkeypatch.setattr(ds, "check_control", refuse)
    ch.recorded_at = 1.0
    st.run_step("t213", _args(tmp_path / "b", arms="dense", length=128, reps=2, warmup=2,
                              floor_ms=1e6), st.STEP_FUNCS)
    f = json.loads((tmp_path / "b" / "result.json").read_text())["findings"]
    assert f["check_control"]["passed"] is False         # a finding, not a failure of the step


# --- the card steps: skipped on a CPU, refused when real ----------------------------------

@pytest.mark.parametrize("step", ["nvml", "t216", "density", "mask_ab", "t212", "t22"])
def test_the_card_steps_skip_a_cpu_dry_run_and_fail_a_real_one(tmp_path, step, monkeypatch):
    monkeypatch.setattr(st, "_cuda", lambda: False)
    monkeypatch.setattr("t227_steps._cuda", lambda: False)
    assert st.run_step(step, _args(tmp_path, step), st.STEP_FUNCS) == 3
    assert st.run_step(step, _args(tmp_path, step, dry_run=False), st.STEP_FUNCS) == 1


# --- the card steps' wiring, with fakes ------------------------------------------------------

@pytest.fixture
def card(monkeypatch):
    import torch
    monkeypatch.setattr(st, "_need_cuda", lambda args, what: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)


def _live(args):
    args.dry_run = False
    return args


def test_nvml_fails_a_sampler_that_cannot_keep_its_rate(tmp_path, card, monkeypatch):
    import _clock_fakes as fakes
    from attnbench import clock_sampling as cs

    class Reader:
        observes_transients, _throttle = True, staticmethod(lambda h: 0)
        _throttle.__name__ = "nvmlDeviceGetCurrentClocksEventReasons"

        def read(self):
            return cs.ClockSample(2040, 6250, cs.SW_POWER_CAP, {"power": 1, "thermal": 0})
    monkeypatch.setattr(cs, "NvmlReader", lambda i: Reader())
    monkeypatch.setattr(st, "_gpu_load", lambda s: __import__("time").sleep(0.3))

    def sampler(args, hz):
        return cs.ProcessSampler(fakes.shared_bits_reader,
                                 tuple(__import__("multiprocessing").get_context("spawn").Value("q", v)
                                       for v in (cs.SW_POWER_CAP, 2040)),
                                 flag_bits=cs.RESEARCHER_FLAG_BITS, interval_s=1.0 / hz)
    monkeypatch.setattr(st, "_sampler", lambda args: sampler(args, 20.0))
    a = _live(_args(tmp_path, "nvml", sampler_seconds=0.6, sampler_hz=20.0))
    assert st.run_step("nvml", a, st.STEP_FUNCS) == 0
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert f["software_power_cap_seen_under_load"] and not f["loaded_block"]["flagged"]
    assert f["idle"]["achieved_hz"] > 10

    b = _live(_args(tmp_path / "slow", "nvml", sampler_seconds=0.6, sampler_hz=400.0))
    monkeypatch.setattr(st, "_sampler", lambda args: sampler(args, 20.0))     # asked 400, kept 20
    assert st.run_step("nvml", b, st.STEP_FUNCS) == 1


def _fake_choice(chosen="sdpa_flash"):
    from attnbench import dense_selection as ds
    names = ds.planned_candidates("NVIDIA L4")
    cands = [ds.CandidateResult(n, ds.CORRECT if chosen else ds.INCORRECT, check_kind="exact",
                                median_ms=3.0 + i if chosen else None,
                                warmup_ms=[3.0 + i] * 2 if chosen else [])
             for i, n in enumerate(names)]
    return ds.Choice(cell=ds.Cell("NVIDIA L4", 8192, 1), cfg_key="k", planned=list(names),
                     candidates=cands, rounds=2, resolution_floor_ms=1.0, recorded_at=5.0,
                     **ds.derive(cands, 1.0))


def test_t216_writes_the_record_and_reads_it_back(tmp_path, card, monkeypatch):
    from attnbench import dense_selection as ds
    monkeypatch.setattr(st, "_harness", lambda args: SimpleNamespace(_cfg=lambda spec, n: None))
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {})
    monkeypatch.setattr(ds, "select_dense_baseline", lambda *a, **k: _fake_choice())
    seen = {}
    real_write = ds.write_choice
    monkeypatch.setattr(ds, "write_choice", lambda c, d: seen.setdefault("p", real_write(c, d)))
    assert st.run_step("t216", _live(_args(tmp_path, "t216")), st.STEP_FUNCS) == 0
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert f["chosen"] == "sdpa_flash" and f["complete"] and seen["p"].exists()
    assert f["floor_note"].startswith("provisional")


def test_t216_fails_when_no_candidate_was_correct(tmp_path, card, monkeypatch):
    from attnbench import dense_selection as ds
    monkeypatch.setattr(st, "_harness", lambda args: SimpleNamespace(_cfg=lambda spec, n: None))
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {})
    monkeypatch.setattr(ds, "select_dense_baseline", lambda *a, **k: _fake_choice(chosen=None))
    assert st.run_step("t216", _live(_args(tmp_path, "t216")), st.STEP_FUNCS) == 1


def test_t213_finds_the_choice_written_by_the_sibling_t216_directory(tmp_path):
    from attnbench import dense_selection as ds
    ds.write_choice(_fake_choice(), tmp_path / "02_t216")
    (tmp_path / "02_t216" / "result.json").write_text("{}")
    a = _args(tmp_path / "03_t213")
    assert st._find_choice(a).chosen == "sdpa_flash"
    assert st._find_choice(_args(tmp_path / "x" / "03_t213")) is None


# --- t212 ---------------------------------------------------------------------------------

class _Logits:
    """Stands in for the model wrapper: logits are a function of the backend."""

    def __init__(self, shift):
        self.shift = shift

    def run_measured(self, ids, backend, *, cfg, logits_to_keep, **kw):
        import torch
        _RUNS.append(backend)
        return SimpleNamespace(logits=torch.ones(1, 4) * self.shift.get(type(backend).__name__, 0.0))


_RUNS: list = []


class _KernelInputs:
    """The kernel-input recorder's stand-in: the densities of the masks the
    run's backend handed the kernel (its `dens`, else its record)."""

    def __enter__(self):
        self.start = len(_RUNS)
        return self

    def __exit__(self, *exc):
        (be,) = _RUNS[self.start:]
        self.per_call = list(getattr(be, "dens", None) or be.last_layer_density)

    def per_call_density(self):
        return self.per_call


def _t212_rig(monkeypatch, tmp_path, *, shift):
    import contextlib
    import torch
    from attnbench import ground_truth as gt
    from attnbench import precomputed_masks as pm

    class Source:
        last_layer_density = [0.5, 0.5]

        def reset_density(self):
            pass

    class Pre:
        def __init__(self, mset, ids, device="cuda"):
            self.last_layer_density = []
            self.dens = [0.5, 0.5] if not shift.get("density") else [0.5, 0.4]

        def reset_density(self):
            self.last_layer_density = list(self.dens)

    class Rec:
        def __init__(self, inner, **kw):
            self.inner = inner

        def __getattr__(self, n):
            return getattr(self.inner, n)

        def mask_set(self, ids, n_heads_q):
            return SimpleNamespace(masks={0: 1, 1: 1}, mask_density=lambda: 0.5,  # a method
                                   prompt_digest="d", save=lambda d: None)

    from attnbench import densities as D
    monkeypatch.setattr(pm, "MaskRecorder", Rec)
    monkeypatch.setattr(pm, "PrecomputedMaskBackend", Pre)
    monkeypatch.setattr(D, "KernelInputRecorder", _KernelInputs)
    cfg = SimpleNamespace(n_heads_q=4)
    arms = [gt.Arm("dense", gt.DENSE, object(), cfg),
            gt.Arm("meanpool", gt.INLINE, Source(), cfg, inline_estimator="x", builder_kind="gpu"),
            gt.Arm("xattention", gt.IN_BACKEND, Source(), cfg, builder_kind="gpu")]
    h = SimpleNamespace(timing_arms=lambda names, n: (_Logits(shift), None, "p", arms, {}),
                        unwrap=lambda: None)
    monkeypatch.setattr(st, "_harness", lambda args: h)
    monkeypatch.setattr(st, "_sampler", lambda args: contextlib.nullcontext(st.NullSampler()))
    rows = [{"arm": n, "prefill_ms_median": m, "arm_label": "l"}
            for n, m in [("dense", 30.0), ("meanpool", 20.0), ("precomputed_meanpool", 12.0),
                         ("xattention", 22.0), ("precomputed_xattention", 14.0)]]
    monkeypatch.setattr(gt, "time_arms", lambda *a, **k: {"rows": rows, "complete": True,
                                                          "clean_rounds": 2})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)


def _run212(tmp_path):
    return st.run_step("t212", _live(_args(tmp_path, "t212", replay_arms="meanpool,xattention")),
                       st.STEP_FUNCS)


def test_t212_passes_when_the_replays_match_and_reports_the_estimators_cost(
        tmp_path, card, monkeypatch):
    _t212_rig(monkeypatch, tmp_path, shift={})
    assert _run212(tmp_path) == 0
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert f["arms"]["xattention"]["bitwise_equal_logits"] and f["arms"]["meanpool"]["executed_density_equal"]
    assert f["timing"]["estimator_cost_ms"] == {"meanpool": 8.0, "xattention": 8.0}
    # Shakedown 3 wrote the bound method, not its value (the old fake had a float)
    assert f["arms"]["meanpool"]["mask_density"] == 0.5


def test_t212_a_meanpool_mismatch_is_a_finding_and_an_xattention_one_fails(
        tmp_path, card, monkeypatch):
    _t212_rig(monkeypatch, tmp_path, shift={"Pre": 0.25})        # every replay differs
    assert _run212(tmp_path) == 1
    res = json.loads((tmp_path / "result.json").read_text())
    assert "xattention: the replay is not bitwise" in res["reason"]
    assert "meanpool: the replay is not bitwise" not in res["reason"]
    assert res["findings"]["arms"]["meanpool"]["bitwise_equal_logits"] is False


def test_t212_a_different_executed_density_fails(tmp_path, card, monkeypatch):
    _t212_rig(monkeypatch, tmp_path, shift={"density": True})
    assert _run212(tmp_path) == 1
    assert "executed density" in json.loads((tmp_path / "result.json").read_text())["reason"]


# --- t22 --------------------------------------------------------------------------------------

def _t22_rig(monkeypatch, *, builder_rc=0):
    import pandas as pd
    from attnbench.components import host_profile as hp
    from attnbench.components import recorder
    arm = SimpleNamespace(name="meanpool", spec=SimpleNamespace(name="meanpool"),
                          run_prefill=lambda: None, code_path=None, prompt_id="p", layers=[])
    monkeypatch.setattr(st, "_harness", lambda args: SimpleNamespace(ready=lambda n, L: arm,
                                                                     unwrap=lambda: None))
    monkeypatch.setattr(recorder, "record_arm_prefill",
                        lambda *a, **k: SimpleNamespace(tape=object(), masks=None,
                                                        sequence=SimpleNamespace(ops=())))
    got = {}

    def profile(run, tape, seq, **kw):
        got.update(kw)
        return hp.HostProfile(host_s={0: 1e-4, 1: 2e-4}, reps_s={0: (1e-4,), 1: (2e-4,)},
                              launch_floor_s=5e-6, reps=kw["reps"], sequence_sha256="s")
    monkeypatch.setattr(hp, "profile", profile)
    monkeypatch.setattr(hp, "block_sparse_entry", lambda: "entry")
    monkeypatch.setattr(hp, "cuda_device", lambda: "device")
    from attnbench.components import replay
    monkeypatch.setattr(replay, "isolated_replay", lambda *a, **k: "iso_trace")
    monkeypatch.setattr(hp, "isolated_host_s", lambda *a, **k: {})
    monkeypatch.setattr(hp, "chain_check", lambda *a, **k: {
        "passed": False, "share": 0.2, "aggregate_tol": k["aggregate_tol"]})

    def run(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        if builder_rc == 0:
            out.mkdir(parents=True)
            pd.DataFrame([{"seq_len": n, "unprofiled_s_per_call_median": 0.01 * n / 2048,
                           "aten_calls_per_call": 1e5, "python_share": 0.5,
                           "python_share_vs_unprofiled": 0.3, "cprofile_python_share": 0.9}
                          for n in (2048, 32768)]).to_parquet(out / "builder_profile.parquet")
        return builder_rc
    monkeypatch.setattr(st, "_run", run)
    return got


def test_t22_profiles_the_tape_and_reads_the_builders_32768_row_against_the_plans_94(
        tmp_path, card, monkeypatch):
    got = _t22_rig(monkeypatch)
    assert st.run_step("t22", _live(_args(tmp_path, "t22")), st.STEP_FUNCS) == 0
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert got["entries"] == ("entry",) and got["reps"] == 5          # a sparse arm: its entry point nulled
    assert f["positions"] == 2 and (tmp_path / "host_profile.json").exists()
    assert f["chain_check"]["passed"] is False                   # a finding: t22 does not gate
    assert f["chain_check"]["aggregate_tol"] == pytest.approx(0.10)
    c = f["against_the_plans_94pct"]
    assert c["seq_len"] == 32768 and c["plan_figure"] == 0.94
    assert (c["cprofile"], c["torch_profiler"], c["lower_bound"]) == (0.9, 0.5, 0.3)


def test_t22_fails_when_the_builder_profile_is_not_written(tmp_path, card, monkeypatch):
    _t22_rig(monkeypatch, builder_rc=2)
    assert st.run_step("t22", _live(_args(tmp_path, "t22")), st.STEP_FUNCS) == 1


# --- the density probe -------------------------------------------------------------------

def test_the_density_step_runs_one_process_per_compile_setting_and_fails_with_any(
        tmp_path, card, monkeypatch):
    from t31_steps import working_values
    groups = list(working_values()[0]["gt_groups"])
    ran = []

    def fake_run(cmd):
        g = cmd[cmd.index("--group") + 1]
        ran.append(g)
        d = Path(cmd[cmd.index("--out") + 1])
        st._write_json(d / "densities.json", [{"arm": "x", "problems": []}])
        return 1 if g == fail[0] else 0
    monkeypatch.setattr(st, "_run", fake_run)
    fail = [None]
    a = _live(_args(tmp_path, "density"))
    assert st.run_step("density", a, st.STEP_FUNCS) == 0
    assert ran == groups and len(groups) == 3
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert f["values"].startswith("rehearsal") and len(f["values_sha256"]) == 64
    fail[0], ran[:] = "compiled_default", []
    assert st.run_step("density", _live(_args(tmp_path / "b", "density")), st.STEP_FUNCS) == 1
    assert ran == groups                                  # every group is still probed


def test_the_density_arms_are_the_working_values_at_batch_1():
    from t31_steps import working_values
    vals = working_values()[0]
    names, dens = st.density_arms(vals, "eager")
    assert names == ["dense", "oracle", "oracle", "meanpool", "meanpool", "xattention",
                     "xattention_noassert"]
    assert dens == [None, 0.25, 0.5, 0.25, 0.5, None, None]
    assert st.density_arms(vals, "compiled_capture") == (
        ["dense", "meanpool", "meanpool", "xattention"], [None, 0.25, 0.5, None])


@pytest.fixture
def density_rig(monkeypatch):
    """`density_one` against a fake harness: real `ground_truth.Arm`s, a
    stand-in for `CompiledBackend` (the real one claims its setting for the
    whole process), a fake probe that refuses `refuse`."""
    import torch
    from attnbench import ground_truth as gt
    from attnbench.backends import compiled
    seen = {}
    refuse = set()

    class Compiled:
        def __init__(self, arm, setting):
            self.arm, self.setting = arm, setting
    monkeypatch.setattr(compiled, "CompiledBackend", Compiled)

    class Be:
        def __init__(self, name):
            self.name = name

    class H:
        def _backend(self, spec):
            return Be(spec.name)

        def timing_arms(self, names, length, *, densities, backends, compile_settings):
            seen.update(names=names, densities=densities, settings=compile_settings,
                        backends=backends)
            arms = []
            for n, d, be, cs in zip(names, densities, backends, compile_settings):
                kind = gt.DENSE if n == "dense" else gt.IN_BACKEND
                label = n if d is None else f"{n}@{d:g}"
                arms.append(gt.Arm(label + ("" if cs is None else f"@compile-{cs}"), kind,
                                   be or Be("dense"), None, compile_setting=cs,
                                   builder_kind=None if n == "dense" else "gpu"))
            return SimpleNamespace(run_measured=lambda *a, **k: None), torch.zeros(1, 8), "p", arms, {}

        def unwrap(self):
            pass

    def probe(arm, run):
        run()
        if arm.kind == gt.DENSE:
            return {"mask_density": None, "executed_density": 1.0}
        if arm.name in refuse:
            raise RuntimeError(f"{arm.name}'s backend keeps no per-layer mask density")
        return {"mask_density": 0.3, "executed_density": 0.31}
    monkeypatch.setattr(st, "_harness", lambda args: H())
    monkeypatch.setattr(gt, "kernel_densities", probe)
    monkeypatch.setattr(gt, "untimed_setup", lambda *a, **k: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield seen, refuse, Compiled
    gt.reset_process_compile_setting()


def test_density_one_probes_every_arm_of_a_compiled_group_under_its_setting(
        tmp_path, density_rig):
    seen, _, compiled = density_rig
    a = _args(tmp_path, "density-one", group="compiled_default", dry_run=False)
    assert st.density_one(a) == 0
    assert seen["settings"] == [None, "default", "default", "default"]
    assert all(isinstance(b, compiled) and b.setting == "default"
               for b in seen["backends"][1:])
    rows = json.loads((tmp_path / "densities.json").read_text())
    assert [r["problems"] for r in rows] == [[]] * 4
    assert rows[1]["mask_minus_nominal"] == pytest.approx(0.3 - 0.25)
    assert "mask_minus_nominal" not in rows[3]               # XAttention sets its own


def test_density_one_fails_a_backend_that_keeps_no_mask_density(tmp_path, density_rig):
    _, refuse, _ = density_rig
    refuse.add("meanpool@0.5")
    assert st.density_one(_args(tmp_path, "density-one", group="eager", dry_run=False)) == 1
    rows = {r["arm"]: r for r in json.loads((tmp_path / "densities.json").read_text())}
    assert "no per-layer mask density" in rows["meanpool@0.5"]["problems"][0]
    assert rows["meanpool@0.25"]["problems"] == []


# --- the launch queue ----------------------------------------------------------------------

def _queue_args(tmp_path):
    return _args(tmp_path, "queue")


def test_the_queue_step_records_the_adds_q_and_bound_in_every_setting(tmp_path):
    assert st.run_step("queue", _queue_args(tmp_path), st.STEP_FUNCS) == 0
    f = json.loads((tmp_path / "result.json").read_text())["findings"]
    assert list(f["settings"]) == [s["name"] for s in st.QUEUE_SETTINGS]
    assert f["settings"]["cold_2s_0.25"]["warm_launch"] is False          # the row's own record
    assert all(v["warm_launch"] for n, v in f["settings"].items() if n != "cold_2s_0.25")
    assert f["add"]["agreed_values"] == [64] and f["add"]["lower_bound"] == 64
    assert all(f["add"]["same_as_null_kernel"].values())


def test_the_queue_step_fails_only_on_the_null_kernel_in_shakedown_1s_setting(
        tmp_path, monkeypatch):
    import pandas as pd

    def fake_run(cmd, null_q=None):
        d = Path(cmd[cmd.index("--out") + 1])
        d.mkdir(parents=True)
        cold = "--cold" in cmd
        pd.DataFrame([{"sleep0_q": null_q[0] if cold else 1022, "sleep0_q_trials": "x",
                       "sleep0_q_lower_bound": 1, "add_q": None, "add_q_trials": "1,1022",
                       "add_q_lower_bound": 1}]).to_parquet(d / "launch_queue.parquet")
        return 0
    null_q = [1022]
    monkeypatch.setattr(st, "_run", lambda cmd: fake_run(cmd, null_q))
    assert st.run_step("queue", _queue_args(tmp_path / "a"), st.STEP_FUNCS) == 0
    f = json.loads((tmp_path / "a" / "result.json").read_text())["findings"]
    assert f["add"]["agreed_values"] == [] and f["add"]["lower_bound"] == 1   # a finding
    null_q[0] = None
    assert st.run_step("queue", _queue_args(tmp_path / "b"), st.STEP_FUNCS) == 1


# --- the mask-output A/B (ruling, 1 Nov 2026) ------------------------------------------------

def test_the_a_b_summary_is_b_minus_a_with_the_wider_spread():
    got = st.ab_summary({"with_mask_output": [0.0102, 0.0100, 0.0101, 0.0103],
                         "without_mask_output": [0.0100, 0.0099, 0.0100, 0.0098]})
    assert got["arms"]["with_mask_output"]["median_ms"] == pytest.approx(10.15)
    assert got["arms"]["without_mask_output"]["median_ms"] == pytest.approx(9.95)
    assert got["b_minus_a_ms"] == pytest.approx(0.2)
    assert got["b_minus_a_pct"] == pytest.approx(100 * 0.2 / 9.95)
    assert got["spread_ms"] == max(r["iqr_ms"] for r in got["arms"].values())


def test_the_a_b_step_runs_one_side_per_process_and_pools_the_rounds(tmp_path, monkeypatch):
    # Shakedown 5: both sides in one process reached Dynamo's recompile limit.
    monkeypatch.setattr(st, "_need_cuda", lambda args, what: None)
    seen = []

    def run(cmd):
        side, part = cmd[cmd.index("--side") + 1], cmd[cmd.index("--part") + 1]
        seen.append(side)
        t = 0.0101 if side == "with_mask_output" else 0.0100
        (tmp_path / f"mask_ab_{part}_{side}.json").write_text(json.dumps(
            {"side": side, "length": 8075, "prompt_id": "p", "compiles": 31,
             "rounds_s": [t, t], "if_material": "x"}))
        return 0
    monkeypatch.setattr(st, "_run", run)
    found = st.step_mask_ab(_args(tmp_path, "mask_ab", dry_run=False))
    assert seen == list(st.AB_ORDER) == ["without_mask_output", "with_mask_output",
                                         "with_mask_output", "without_mask_output"]
    assert found["rounds_s"]["with_mask_output"] == [0.0101] * 4
    assert found["b_minus_a_ms"] == pytest.approx(0.1)
    assert json.loads((tmp_path / "mask_ab.json").read_text())["b_minus_a_ms"] == \
        pytest.approx(0.1)


def test_the_a_b_step_fails_when_a_side_s_process_does(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_need_cuda", lambda args, what: None)
    monkeypatch.setattr(st, "_run", lambda cmd: 1)
    with pytest.raises(st.StepFailed, match=r"A/B failed in process 0 \(without_mask_output\)"):
        st.step_mask_ab(_args(tmp_path, "mask_ab", dry_run=False))
