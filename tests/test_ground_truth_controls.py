"""T2.13's controls: the dense control's drift check and the rerun rule, clock
and throttle sampling, traced against untraced, one compiled setting per
process, and the first clocked block's timestamp (plan Sections 5.3 steps 1, 7
and 8, 5.5 and 5.6).

Each check here is one a mutation has been watched to break (recorded in the
commit that added it); the tests that can express the defect with a fake keep
it, so the proof does not rest on a mutation someone ran once. The clock is
fake and shared with every backend and the clock reader (`_gt_fakes.py`), so
what a test asserts about drift, throttle windows and timestamps is scripted,
not observed.
"""

from __future__ import annotations

import threading

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from _gt_fakes import (LAYERS, CostedBackend, FakeClock, ScriptedReader,  # noqa: E402
                       Tracing)
from attnbench import clock_sampling as cs  # noqa: E402
from attnbench import ground_truth as gt  # noqa: E402
from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

SEQ, BLOCK, REPS, FLOOR = 64, 8, 3, 5.0
WARMUP = 3
THERMAL = cs.SW_THERMAL_SLOWDOWN | cs.HW_THERMAL_SLOWDOWN
SLOWDOWNS = THERMAL | cs.HW_SLOWDOWN | cs.HW_POWER_BRAKE_SLOWDOWN


@pytest.fixture(autouse=True)
def _fresh_process_state():
    gt.reset_process_compile_setting()
    yield
    gt.reset_process_compile_setting()


def _wrapped():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=128, attn_implementation="eager")
    return SwappableAttentionModel(LlamaForCausalLM(cfg).eval(), _cfg(), model_id="toy",
                                   finest_block_size=BLOCK)


def _cfg(**kw):
    base = dict(seq_len=SEQ, batch=1, n_heads_q=4, n_heads_kv=2, head_dim=8,
                dtype="float32", mask="block_sparse", block_size=BLOCK,
                sparsity=0.75, mask_source="importance")
    base.update(kw)
    return AttnConfig(**base)


def _ids():
    return torch.randint(0, 64, (1, SEQ), generator=torch.Generator().manual_seed(3))


class Rig:
    """A fake clock, arms whose per-forward cost is scripted, and a reader."""

    # interval_s: Controls refuse a counters-only sampler (audit C3, finding 1),
    # so the rig's sampler reads inside every block too.
    def __init__(self, costs=None, *, windows=(), flag_bits=SLOWDOWNS, interval_s=0.001,
                 reader=None, max_rerun=2, settings=None, tracing=None,
                 traced_extra_ms=0.0, read_cost_ms=0.0):
        self.clock = FakeClock()
        self.log = []
        costs = costs or {}
        settings = settings or {}
        self.reader = reader or ScriptedReader(self.clock, windows, read_cost_ms=read_cost_ms)
        self.sampler = cs.BlockSampler(self.reader, flag_bits=flag_bits, interval_s=interval_s)
        self.controls = gt.Controls(resolution_floor_ms=FLOOR, max_rerun_rounds=max_rerun,
                                    sampler=self.sampler)

        def backend(name, default):
            return CostedBackend(self.clock, costs.get(name, lambda i: default),
                                 tracing=tracing, traced_extra_ms=traced_extra_ms,
                                 setting=settings.get(name), log=self.log, name=name)
        self.arms = [
            gt.Arm("dense", gt.DENSE, backend("dense", 30.0),
                   _cfg(mask="causal", sparsity=None, block_size=None),
                   compile_setting=settings.get("dense")),
            gt.Arm("meanpool", gt.INLINE, backend("meanpool", 18.0), _cfg(),
                   inline_estimator="minference_meanpool",
                   compile_setting=settings.get("meanpool")),
        ]

    def run(self, **kw):
        kw.setdefault("reps", REPS)
        kw.setdefault("warmup", WARMUP)
        kw.setdefault("wall_clock", lambda: 1_000_000.0 + self.clock.t)
        return gt.time_arms(_wrapped(), _ids(), self.arms, controls=self.controls,
                            clock=self.clock, **kw)


def _row(out, name):
    return next(r for r in out["rows"] if r["arm"] == name)


# --- the dense control and the rerun rule -----------------------------------

def test_a_quiet_session_has_no_flagged_round():
    out = Rig().run()
    assert out["complete"] and out["rounds_run"] == REPS
    assert not any(r["flagged"] for r in out["rounds"])
    assert out["dense_control"]["reference_ms"] == pytest.approx(30.0)
    assert _row(out, "dense")["prefill_ms_median"] == pytest.approx(30.0)


def test_dense_drift_beyond_the_floor_flags_the_round_and_it_is_rerun():
    # The dense forward index of timed round 1 is WARMUP + 1.
    rig = Rig({"dense": lambda i: 30.0 + (15.0 if i == WARMUP + 1 else 0.0)})
    out = rig.run()
    assert [r["flagged"] for r in out["rounds"]] == [False, True, False, False]
    assert "dense_drift" in out["rounds"][1]["flag_reasons"][0]
    assert out["complete"] and out["clean_rounds"] == REPS and out["rounds_run"] == REPS + 1
    d = _row(out, "dense")
    assert d["prefill_ms_samples"] == [pytest.approx(30.0)] * REPS      # drift excluded
    assert d["flagged_ms_samples"] == [pytest.approx(45.0)]
    # The other arm's block in the flagged round is excluded with it.
    assert len(_row(out, "meanpool")["prefill_ms_samples"]) == REPS


@pytest.mark.parametrize("drift,flagged", [(4.0, False), (5.5, True), (-15.0, True)])
def test_drift_is_compared_to_the_floor_in_both_directions(drift, flagged):
    rig = Rig({"dense": lambda i: 30.0 + (drift if i == WARMUP else 0.0)}, max_rerun=0)
    assert rig.run(reps=1)["rounds"][0]["flagged"] is flagged


def test_a_session_that_cannot_get_enough_clean_rounds_says_so():
    rig = Rig({"dense": lambda i: 30.0 + (40.0 if i >= WARMUP else 0.0)}, max_rerun=2)
    out = rig.run()
    assert not out["complete"] and out["clean_rounds"] == 0
    assert out["rounds_run"] == REPS + 2
    d = _row(out, "dense")
    assert d["prefill_ms_median"] is None and d["prefill_ms_samples"] == []
    assert len(d["flagged_ms_samples"]) == REPS + 2


def test_the_cold_first_warmup_forward_is_not_the_reference():
    # Two warm-up forwards: with three the median would hide the cold call and
    # the test could not tell whether it was dropped.
    rig = Rig({"dense": lambda i: 200.0 if i == 0 else 30.0})
    out = rig.run(warmup=2)
    assert out["dense_control"]["reference_ms"] == pytest.approx(30.0)
    assert out["complete"] and not any(r["flagged"] for r in out["rounds"])


def test_a_run_needs_exactly_one_dense_control():
    rig = Rig()
    with pytest.raises(gt.TimedRegionError, match="exactly one dense control"):
        gt.time_arms(_wrapped(), _ids(), rig.arms[1:], reps=1, controls=rig.controls,
                     clock=rig.clock)
    twin = gt.Arm("dense2", gt.DENSE, rig.arms[0].backend, rig.arms[0].cfg)
    with pytest.raises(gt.TimedRegionError, match="exactly one dense control"):
        gt.time_arms(_wrapped(), _ids(), rig.arms + [twin], reps=1, controls=rig.controls,
                     clock=rig.clock)


def test_controls_and_warmup_have_no_silent_defaults():
    rig = Rig()
    with pytest.raises(TypeError):
        gt.time_arms(_wrapped(), _ids(), rig.arms, reps=1)       # no controls
    with pytest.raises(ValueError, match="warmup must be >= 2"):
        rig.run(warmup=1)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            gt.Controls(resolution_floor_ms=bad, max_rerun_rounds=0, sampler=rig.sampler)


def test_every_round_is_checked_for_its_timed_region_including_reruns(monkeypatch):
    seen = []
    real = gt.check_timed_region
    monkeypatch.setattr(gt, "check_timed_region",
                        lambda log, arm, reps: (seen.append(list(reps)), real(log, arm, reps))[1])
    Rig({"dense": lambda i: 30.0 + (15.0 if i == WARMUP else 0.0)}).run()
    assert seen and all(s == list(range(REPS + 1)) for s in seen)


# --- clock and throttle sampling ----------------------------------------------

def _first_block_window(rig_clock_free=True):
    """Warm-up is 3 forwards of each arm: dense 3 x 30 ms, meanpool 3 x 18 ms."""
    return WARMUP * (30.0 + 18.0) / 1e3


def test_a_block_that_overlapped_a_thermal_window_is_flagged_and_rerun():
    start = _first_block_window()
    # The window covers the first round's dense block only (30 ms from `start`).
    rig = Rig(windows=[(start + 0.001, start + 0.010, "thermal")])
    out = rig.run()
    assert out["rounds"][0]["flagged"]
    assert out["rounds"][0]["blocks"]["dense"]["flagged"]
    assert not out["rounds"][0]["blocks"]["meanpool"]["flagged"]
    assert "thermal_violation" in out["rounds"][0]["flag_reasons"][0]
    assert out["complete"] and out["rounds_run"] == REPS + 1
    assert _row(out, "dense")["clean_rounds"] == REPS


def test_a_power_cap_is_recorded_and_never_flags():
    start = _first_block_window()
    win = [(start, start + 1.0, "power")]
    quiet = Rig(windows=win, reader=None).run(reps=1)
    assert not quiet["rounds"][0]["flagged"]
    assert quiet["rounds"][0]["blocks"]["dense"]["violation_delta_ns"]["power"] > 0
    # The power counter integrates the cap and the brake together, so it never
    # flags; the brake is seen from the bits. No session may choose the cap
    # (researcher's decision, 29 Oct 2026).
    with pytest.raises(ValueError, match="software power cap"):
        Rig(windows=win, flag_bits=SLOWDOWNS | cs.SW_POWER_CAP)


def test_instantaneous_reasons_are_recorded_and_the_idle_bit_never_flags():
    rig = Rig()
    rig.reader.bits_always = cs.GPU_IDLE | cs.SW_POWER_CAP
    out = rig.run(reps=1)
    assert not out["rounds"][0]["flagged"]       # power cap not in SLOWDOWNS
    assert _row(out, "dense")["throttle_reasons_seen"] == ["gpu_idle", "sw_power_cap"]
    hot = Rig()
    hot.reader.bits_always = cs.HW_SLOWDOWN
    assert hot.run(reps=1)["rounds"][0]["flagged"]


def test_a_reader_with_no_counters_and_no_thread_is_refused():
    """The idle-GPU trap: a boundary read after the closing synchronise sees
    only the idle bit, so without counters or a thread nothing could flag."""
    reader = ScriptedReader(FakeClock(), transients=False)
    with pytest.raises(cs.ClockSamplingError, match="idle GPU"):
        cs.BlockSampler(reader, flag_bits=SLOWDOWNS)


def test_the_sampling_thread_sees_a_throttle_inside_the_block():
    threads_before = threading.active_count()
    class _WaitsForThread(CostedBackend):
        def forward(self, *a, **kw):
            if self.layer_calls >= WARMUP * LAYERS:      # timed forwards only
                assert reader.thread_seen.wait(5), "the sampling thread never read"
            return super().forward(*a, **kw)

    clock = FakeClock()
    reader = ScriptedReader(clock, transients=False, thread_bits=cs.HW_SLOWDOWN)
    sampler = cs.BlockSampler(reader, flag_bits=SLOWDOWNS, interval_s=0.001)
    controls = gt.Controls(FLOOR, 0, sampler)
    arms = [gt.Arm("dense", gt.DENSE, _WaitsForThread(clock, lambda i: 30.0, name="dense"),
                   _cfg(mask="causal", sparsity=None, block_size=None))]
    out = gt.time_arms(_wrapped(), _ids(), arms, reps=1, controls=controls, clock=clock)
    block = out["rounds"][0]["blocks"]["dense"]
    assert block["samples"] > 2 and "hw_slowdown" in block["throttle_reasons_seen"]
    assert out["rounds"][0]["flagged"]
    assert threading.active_count() == threads_before      # the thread was stopped


def test_the_sampler_reads_are_outside_the_clock():
    rig = Rig(read_cost_ms=100.0)       # every read advances the clock 100 ms
    out = rig.run(reps=1)
    assert _row(out, "dense")["prefill_ms_median"] == pytest.approx(30.0)
    assert _row(out, "meanpool")["prefill_ms_median"] == pytest.approx(18.0)


def test_a_block_that_raises_stops_its_sampling_thread():
    threads_before = threading.active_count()
    clock = FakeClock()
    reader = ScriptedReader(clock, transients=False)
    sampler = cs.BlockSampler(reader, flag_bits=SLOWDOWNS, interval_s=0.001)
    log = gt.EventLog()
    with pytest.raises(RuntimeError, match="boom"):
        with gt.timed_block(log, 0, synchronize=lambda: None, clock=clock, out=[],
                            sampler=sampler, clock_out=[]):
            raise RuntimeError("boom")
    assert threading.active_count() == threads_before
    sampler.begin()                      # reusable: its state was reset
    sampler.end()


def test_sampler_refusals():
    r = ScriptedReader(FakeClock())
    with pytest.raises(ValueError, match="not slowdowns"):
        cs.BlockSampler(r, flag_bits=cs.GPU_IDLE | cs.HW_SLOWDOWN)
    with pytest.raises(ValueError, match="empty"):
        cs.BlockSampler(r, flag_bits=0)
    with pytest.raises(ValueError):
        cs.BlockSampler(r, flag_bits=cs.HW_SLOWDOWN, interval_s=0)
    with pytest.raises(TypeError):
        cs.BlockSampler(r)                 # no default for what flags a block
    s = cs.BlockSampler(r, flag_bits=cs.HW_SLOWDOWN)
    with pytest.raises(cs.ClockSamplingError):
        s.end()


def test_a_counter_that_goes_backwards_is_refused():
    clock = FakeClock()
    reader = ScriptedReader(clock, windows=[(0.0, 10.0, "power")])
    s = cs.BlockSampler(reader, flag_bits=cs.HW_THERMAL_SLOWDOWN)
    clock.t = 5.0
    s.begin()
    reader.windows = []                  # the counter resets underneath us
    with pytest.raises(cs.ClockSamplingError, match="backwards"):
        s.end()


# --- the NVML reader, against a fake module -----------------------------------

class _Nvml:
    NVML_CLOCK_SM, NVML_CLOCK_MEM = 1, 2
    NVML_PERF_POLICY_POWER, NVML_PERF_POLICY_THERMAL = 0, 1

    class _V:
        def __init__(self, ns):
            self.violationTime = ns

    def __init__(self, *, event_name="nvmlDeviceGetCurrentClocksEventReasons",
                 violations=True):
        self.calls = []
        self._violations = violations
        if event_name:
            setattr(self, event_name, lambda h: 0x4 | 0x20)

    def nvmlInit(self):
        self.calls.append("init")

    def nvmlDeviceGetHandleByIndex(self, i):
        return ("handle", i)

    def nvmlDeviceGetClockInfo(self, h, which):
        return 1410 if which == self.NVML_CLOCK_SM else 1593

    def nvmlDeviceGetViolationStatus(self, h, policy):
        if not self._violations:
            raise RuntimeError("not supported")
        return self._V(1000 * (policy + 1))


def test_the_nvml_reader_decodes_clocks_reasons_and_counters():
    r = cs.NvmlReader(2, nvml=_Nvml())
    s = r.read()
    assert (s.sm_mhz, s.mem_mhz) == (1410, 1593)
    assert cs.reason_names(s.throttle_bits) == ["sw_power_cap", "sw_thermal_slowdown"]
    assert s.violation_ns == {"power": 1000, "thermal": 2000}
    assert r.observes_transients


def test_the_nvml_reader_accepts_the_older_throttle_name_and_missing_counters():
    r = cs.NvmlReader(nvml=_Nvml(event_name="nvmlDeviceGetCurrentClocksThrottleReasons",
                                 violations=False))
    assert not r.observes_transients and r.read().violation_ns is None
    with pytest.raises(cs.ClockSamplingError, match="idle GPU"):
        cs.BlockSampler(r, flag_bits=SLOWDOWNS)
    with pytest.raises(cs.ClockSamplingError, match="none of"):
        cs.NvmlReader(nvml=_Nvml(event_name=None))


def test_without_the_nvml_library_the_reader_says_so(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_pynvml(name, *a, **kw):
        if name == "pynvml":
            raise ImportError(name)
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_pynvml)
    with pytest.raises(cs.ClockSamplingError, match="not installed"):
        cs.NvmlReader()


# --- traced against untraced ----------------------------------------------------

def _traced_session(extra_ms):
    tracing = Tracing()
    rig = Rig(tracing=tracing, traced_extra_ms=extra_ms)
    out = gt.time_with_profiler_overhead(_wrapped(), _ids(), rig.arms, reps=REPS,
                                         controls=rig.controls, tracer=tracing,
                                         clock=rig.clock, warmup=WARMUP)
    return rig, tracing, out


def test_the_traced_overhead_is_what_the_tracer_added():
    rig, tracing, out = _traced_session(9.0)
    by = {o["arm"]: o for o in out["overhead"]}
    assert by["dense"]["overhead_ms"] == pytest.approx(9.0)
    assert by["meanpool"]["overhead_ms"] == pytest.approx(9.0)
    assert by["dense"]["overhead_ratio"] == pytest.approx(39.0 / 30.0)
    assert not by["dense"]["inside_resolution_floor"]       # 9 ms > 5 ms floor
    assert not out["untraced"]["traced"] and out["traced"]["traced"]
    assert all(r["traced"] for r in out["traced"]["rows"])
    assert not any(r["traced"] for r in out["untraced"]["rows"])


def test_untraced_ground_truth_is_measured_before_any_tracing_starts():
    rig, tracing, out = _traced_session(9.0)
    flags = [t for (_, _, _, t) in rig.log]
    assert flags.count(True) > 0 and flags.count(False) > 0
    assert flags == sorted(flags)               # every untraced forward before any traced one
    # Ground truth never saw a tracer: its times are the untraced ones.
    assert _row(out["untraced"], "dense")["prefill_ms_median"] == pytest.approx(30.0)


def test_a_tracer_that_adds_nothing_reports_no_overhead():
    _, _, out = _traced_session(0.0)
    assert all(o["overhead_ms"] == pytest.approx(0.0) and o["inside_resolution_floor"]
               for o in out["overhead"])


def test_an_overhead_inside_the_floor_is_labelled_so():
    _, _, out = _traced_session(3.0)
    assert all(o["inside_resolution_floor"] for o in out["overhead"])


def test_the_timestamp_handed_on_is_the_untraced_one():
    rig, tracing, out = _traced_session(9.0)
    assert out["first_clocked_block_started_at"] == out["untraced"]["first_clocked_block_started_at"]
    assert out["traced"]["first_clocked_block_started_at"] > out["first_clocked_block_started_at"]


def test_profiler_overhead_refuses_swapped_or_mismatched_results():
    _, _, out = _traced_session(1.0)
    with pytest.raises(gt.TimedRegionError, match="untraced result first"):
        gt.profiler_overhead(out["traced"], out["untraced"], resolution_floor_ms=FLOOR)
    short = dict(out["traced"], rows=out["traced"]["rows"][:1])
    with pytest.raises(gt.TimedRegionError, match="differ"):
        gt.profiler_overhead(out["untraced"], short, resolution_floor_ms=FLOOR)


def test_an_arm_with_no_clean_round_has_no_overhead_not_a_wrong_one():
    tracing = Tracing()
    rig = Rig({"meanpool": lambda i: 18.0}, tracing=tracing, traced_extra_ms=2.0)
    untraced = rig.run()
    # Fabricate the failure the harness can report: no clean round on one side.
    traced = dict(untraced, traced=True,
                  rows=[dict(r, traced=True, prefill_ms_median=None, clean_rounds=0)
                        for r in untraced["rows"]])
    for o in gt.profiler_overhead(untraced, traced, resolution_floor_ms=FLOOR):
        assert o["overhead_ms"] is None and o["inside_resolution_floor"] is None


# --- one compiled variant per process --------------------------------------------

def test_two_compiled_settings_in_one_call_are_refused():
    rig = Rig(settings={"dense": "default", "meanpool": "capture"})
    with pytest.raises(gt.CompileSettingError, match="Dynamo would reuse"):
        rig.run(reps=1)
    assert rig.clock.t == 0.0         # refused before anything ran


def test_a_second_setting_in_the_same_process_is_refused_across_calls():
    Rig(settings={"dense": "default"}).run(reps=1)
    with pytest.raises(gt.CompileSettingError, match="already timed the 'default'"):
        Rig(settings={"dense": "capture"}).run(reps=1)
    Rig(settings={"dense": "default"}).run(reps=1)       # the same setting again is fine
    gt.reset_process_compile_setting()                   # after Dynamo's caches are cleared
    Rig(settings={"dense": "capture"}).run(reps=1)


def test_eager_arms_may_share_a_process_with_one_compiled_setting():
    out = Rig(settings={"meanpool": "capture"}).run(reps=1)
    assert _row(out, "meanpool")["compile_setting"] == "capture"
    assert _row(out, "dense")["compile_setting"] is None


def test_a_declaration_the_backend_does_not_bear_out_is_refused():
    rig = Rig()
    rig.arms[1] = gt.Arm("meanpool", gt.INLINE, rig.arms[1].backend, _cfg(),
                         inline_estimator="minference_meanpool", compile_setting="default")
    with pytest.raises(gt.CompileSettingError, match="declares compile_setting='default'"):
        rig.run(reps=1)
    rig = Rig(settings={"meanpool": "default"})       # backend compiled, arm says eager
    rig.arms[1] = gt.Arm("meanpool", gt.INLINE, rig.arms[1].backend, _cfg(),
                         inline_estimator="minference_meanpool")
    with pytest.raises(gt.CompileSettingError, match="backend carries setting='default'"):
        rig.run(reps=1)


# --- the first clocked block's timestamp ---------------------------------------------

def test_the_timestamp_is_the_first_clocked_block_warmups_included():
    """The strict rule (Appendix A, 27 Oct 2026): the stamp is the start of the
    first block of any kind that is clocked, so it precedes every warm-up
    forward and every timed one."""
    rig = Rig()
    out = rig.run()
    stamp = out["first_clocked_block_started_at"] - 1_000_000.0
    first_clocked = min(t for (n, i, t, _) in rig.log)            # any forward, warm-up or timed
    first_timed = min(t for (n, i, t, _) in rig.log if i >= WARMUP)
    assert stamp == pytest.approx(first_clocked)
    assert stamp < first_timed - 1e-9                           # not the first timed block's


def test_a_rerun_round_does_not_move_the_timestamp():
    flagged_first = Rig({"dense": lambda i: 30.0 + (40.0 if i == WARMUP else 0.0)})
    plain = Rig()
    a, b = flagged_first.run(), plain.run()
    assert a["rounds"][0]["flagged"]
    assert a["first_clocked_block_started_at"] == pytest.approx(b["first_clocked_block_started_at"])


def test_the_timestamp_is_taken_once_and_not_per_block():
    calls = []
    rig = Rig()
    rig.run(wall_clock=lambda: calls.append(1) or 7.0)
    assert len(calls) == 1


def test_the_timestamp_is_in_the_form_the_late_prediction_check_reads():
    from attnbench.predict.commit import ground_truth_start, utc_from_epoch
    out = Rig().run()
    stamp = out["first_clocked_block_started_at"]
    assert ground_truth_start([stamp]) == utc_from_epoch(stamp)


def test_the_late_prediction_check_reads_this_timestamp_and_refuses_what_is_late(tmp_path):
    """The consumer, end to end: a prediction synced at or after the first
    clocked block started, warm-up included, is refused by `load_committed`,
    and one synced before it is accepted."""
    from datetime import datetime, timedelta, timezone

    from test_predict_commit import Clock, MemoryStore, committed, p1_record
    from attnbench.predict.commit import LatePredictionError, load_committed, utc_from_epoch

    epoch = datetime(2026, 11, 2, 3, 0, 0, tzinfo=timezone.utc).timestamp()
    rig = Rig()
    start = 10.0                       # the instance starts its first clocked block 10 s after the file
    out = rig.run(wall_clock=lambda: epoch + start + rig.clock.t)
    stamp = out["first_clocked_block_started_at"]
    warmup_s = WARMUP * (30.0 + 18.0) / 1e3
    assert stamp == pytest.approx(epoch + start)           # the first warm-up forward

    def load(sync_offset_s):
        store = MemoryStore(Clock(None))
        (tmp_path / f"s{sync_offset_s}").mkdir()
        f, receipt = committed(tmp_path / f"s{sync_offset_s}", p1_record(), store,
                               utc_from_epoch(epoch + sync_offset_s))
        return load_committed(store, receipt.remote,
                              ground_truth_started_at=utc_from_epoch(stamp),
                              clock_skew=timedelta(0), banked_at={}, local=f.path)

    assert load(0.0).pairing == "P1"                       # before any block was clocked
    # At the stamp, during the dense control's clocked warm-up, and after it
    # (the strict rule: a warm-up is a clocked block, so these are all late).
    for late in (start, start + warmup_s / 2, start + warmup_s, start + warmup_s + 0.001):
        with pytest.raises(LatePredictionError):
            load(late)
