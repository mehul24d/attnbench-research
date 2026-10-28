"""T2.2: the reference-builder profile (Python share of the loop time).

The profiler is real (CPU), on small workloads chosen so the answer is not in
doubt: a builder that is all interpreter must read near 1, one that is a
single large torch operation must read near 0. A fake profiler and clock check
the arithmetic exactly.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from attnbench.components import builder_profile as BP

REPO = Path(__file__).resolve().parent.parent


class FakeEvent:
    def __init__(self, key, self_us, count=1):
        self.key, self.self_cpu_time_total, self.count = key, self_us, count


class FakeProfiler:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def key_averages(self):
        return self.events


class Clock:
    """Each reading advances by `step` after returning; `Spy` below rewires."""

    def __init__(self, times):
        self.times = iter(times)

    def __call__(self):
        return next(self.times)


def test_the_arithmetic_on_a_scripted_clock_and_profiler():
    # warmup 1, reps 2. Unprofiled: calls take 4 s and 6 s (median 5). The
    # profiled block takes 20 s for 2 calls: 10 s per call. aten self times:
    # 1.5 s + 0.5 s = 2 s in 8 calls over 2 calls; the "python::x" event is
    # not an aten operator and is left out.
    clock = Clock([0, 4, 10, 16, 100, 120])
    prof = FakeProfiler([FakeEvent("aten::mm", 1_500_000, 5),
                         FakeEvent("aten::add", 500_000, 3),
                         FakeEvent("python::loop", 9_000_000, 1)])
    r = BP.profile_builder(lambda: None, reps=2, warmup=1, clock=clock, profiler=prof,
                           cprofile=False)
    assert r["unprofiled_s_per_call_median"] == 5.0
    assert r["profiled_s_per_call"] == 10.0
    assert r["torch_operator_s_per_call"] == 1.0           # 2 s over 2 calls
    assert r["python_overhead_s_per_call"] == 9.0
    assert r["aten_calls_per_call"] == 4.0
    assert r["python_share"] == pytest.approx(0.9)         # 1 - 1/10
    assert r["python_share_vs_unprofiled"] == pytest.approx(0.8)   # 1 - 1/5
    assert r["profiler_dilation"] == 2.0
    # The plan's units: seconds per 10 calls.
    assert r["loop_body_s_per_10_calls"] == 100.0
    assert r["torch_operators_s_per_10_calls"] == 10.0
    assert r["unprofiled_s_per_call_samples"] == [4, 6]


def test_the_plans_own_figures_give_its_own_share():
    # limitations.md: 0.212 s loop body and 0.013 s torch operations per 10
    # calls: (0.212 - 0.013) / 0.212 = 93.9%. Ten unprofiled calls of 0.0212 s,
    # then a profiled block of 0.212 s with 13,000 us of aten self time.
    clock = Clock([0, 0.0212] * 10 + [0, 0.212])
    prof = FakeProfiler([FakeEvent("aten::index", 13_000, 800)])
    r = BP.profile_builder(lambda: None, reps=10, warmup=0, clock=clock, profiler=prof,
                           cprofile=False)
    assert r["python_share"] == pytest.approx((0.212 - 0.013) / 0.212)
    assert round(100 * r["python_share"], 1) == 93.9
    assert r["loop_body_s_per_10_calls"] == pytest.approx(0.212)
    assert r["torch_operators_s_per_10_calls"] == pytest.approx(0.013)


def test_an_all_python_builder_reads_near_one_and_a_single_big_operation_near_zero():
    def python_only():
        total = 0
        for i in range(60_000):
            total += i * i
        return total

    a = torch.rand(768, 768)

    def one_matmul():
        return a @ a @ a

    py = BP.profile_builder(python_only, reps=3)
    mm = BP.profile_builder(one_matmul, reps=3)
    assert py["python_share"] > 0.95 and py["aten_calls_per_call"] == 0
    assert mm["python_share"] < 0.5 and mm["aten_calls_per_call"] >= 2


def test_the_reference_builder_row_carries_what_it_was_run_on():
    r = BP.profile_reference_builder(1024, sparsity=0.5, reps=2, seed=3)
    assert r["builder"] == "masks.importance_block_mask" and r["free_block"] == "sink"
    assert (r["seq_len"], r["block_size"], r["n_blocks"]) == (1024, 128, 8)
    assert r["sparsity"] == 0.5 and r["density"] == 0.5 and r["scores"] == "random, seed 3"
    assert 0 < r["python_share"] <= 1 and r["aten_calls_per_call"] > 0
    assert r["torch_version"] == torch.__version__
    assert r["profiler_dilation"] > 0 and len(r["unprofiled_s_per_call_samples"]) == 2


def test_the_builder_is_called_as_the_sparse_arms_call_it():
    calls = []
    real = BP.masks.importance_block_mask

    def spy(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)
    BP.masks.importance_block_mask = spy
    try:
        BP.reference_builder(1024, 128, 0.25, seed=1)()
    finally:
        BP.masks.importance_block_mask = real
    (a, kw), = calls
    assert a[:3] == (1024, 128, 0.25) and tuple(a[3].shape) == (8, 8)
    assert kw["causal"] is True and kw["free_block"] == "sink"


def test_the_scores_are_fixed_by_the_seed():
    one = BP.reference_builder(1024, 128, 0.5, seed=5)()
    two = BP.reference_builder(1024, 128, 0.5, seed=5)()
    assert torch.equal(one.active, two.active)


@pytest.mark.parametrize("kw, message", [
    (dict(seq_len=0, sparsity=0.5, reps=1), "seq_len"),
    (dict(seq_len=1024, sparsity=1.0, reps=1), "sparsity"),
    (dict(seq_len=1024, sparsity=-0.1, reps=1), "sparsity"),
    (dict(seq_len=1024, sparsity=True, reps=1), "sparsity"),
    (dict(seq_len=1024, sparsity=0.5, reps=0), "reps"),
    (dict(seq_len=1024, sparsity=0.5, reps=1, block_size=0), "block_size"),
    (dict(seq_len=1024, sparsity=0.5, reps=1, warmup=-1), "warmup"),
])
def test_bad_arguments_are_refused(kw, message):
    with pytest.raises(ValueError, match=message):
        BP.profile_reference_builder(**kw)


def test_the_script_banks_one_stamped_row_per_length_and_never_overwrites(tmp_path, capsys):
    script = REPO / "scripts" / "profile_reference_builder.py"
    ns = runpy.run_path(str(script), run_name="not_main")
    out = tmp_path / "o"
    assert ns["main"](["--out", str(out), "--seq-lens", "512,1024", "--reps", "2"]) == 0
    import pandas as pd
    df = pd.read_parquet(out / "builder_profile.parquet")
    assert list(df["seq_len"]) == [512, 1024]
    assert {"git_commit", "git_dirty"} <= set(df.columns)
    assert "python share" in capsys.readouterr().out
    assert ns["main"](["--out", str(out), "--seq-lens", "512", "--reps", "1"]) == 2
    assert len(pd.read_parquet(out / "builder_profile.parquet")) == 2


def test_cprofile_reads_a_tensor_slot_write_as_python_and_torch_profiler_does_not():
    # `a[i, j] = True` goes through a type slot that cProfile does not count as
    # a call, so its time is the loop's own; an explicit method call is a
    # builtin cProfile does count. torch.profiler sees both as aten operators.
    a = torch.zeros(64, 64, dtype=torch.bool)

    def slot_writes():
        for i in range(5000):
            a[i % 64, (i * 7) % 64] = True

    def method_calls():
        for _ in range(5000):
            a.fill_diagonal_(True)

    slot = BP.profile_builder(slot_writes, reps=2)
    meth = BP.profile_builder(method_calls, reps=2)
    assert slot["cprofile_python_share"] > 0.95
    assert slot["python_share"] < 0.8                 # the aten operators are seen
    assert slot["aten_calls_per_call"] >= 10_000
    assert meth["cprofile_python_share"] < 0.5        # the builtin is seen
    assert slot["cprofile_python_share"] - slot["python_share"] > 0.2


def test_the_cprofile_arithmetic_and_the_option_to_leave_it_out():
    r = BP.profile_builder(lambda: sum(range(20000)), reps=2)
    assert r["cprofile_python_share"] < 0.5                 # nearly all in sum()
    assert r["cprofile_builtin_calls_per_call"] >= 1        # sum() is a builtin
    assert r["cprofile_s_per_call"] > 0 and r["cprofile_dilation"] > 0
    off = BP.profile_builder(lambda: None, reps=2, cprofile=False)
    assert not any(k.startswith("cprofile") for k in off)


def test_cprofile_is_switched_off_even_when_the_build_raises():
    def boom():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        BP.cprofile_share(boom, 1)
    # A second profiler can start: the first was disabled.
    BP.cprofile_share(lambda: None, 1)


def test_the_builder_is_called_warmup_plus_each_pass_reps_times():
    calls = []
    BP.profile_builder(lambda: calls.append(1), reps=3, warmup=2)
    assert len(calls) == 2 + 3 + 3 + 3          # warm-up, unprofiled, torch.profiler, cProfile
    calls.clear()
    BP.profile_builder(lambda: calls.append(1), reps=3, warmup=0, cprofile=False)
    assert len(calls) == 3 + 3
