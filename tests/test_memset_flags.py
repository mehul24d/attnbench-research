"""Memsets with no device record (researcher's ruling, 31 Oct 2026):
tolerated, flagged and counted; host cost counted, device cost zero; an
upper-bound sensitivity at the memset zoo's duration for the same size.

Traces are written with the op-sequence tests' builder (`TB`)."""

from __future__ import annotations

import pytest

from attnbench.components import memset_flags as M
from attnbench.components import op_sequence as S
from attnbench.components.op_sequence import SeqKind, SequenceError
from test_components_op_sequence import TB, kernel, parse


def bare_memset(tb):
    """A cudaMemsetAsync the trace holds no device activity for."""
    tb.call("cudaMemsetAsync")


def sized_memset(tb, nbytes, dur=10.0):
    c, ts = tb.call("cudaMemsetAsync")
    tb.act("gpu_memset", "Memset (Device)", c, ts, dur=dur, bytes=nbytes)


def layer_with(tb, layer, bare=3, sized=()):
    with tb.layer(layer):
        for n in sized:
            with tb.cpu("aten::addmm"):
                sized_memset(tb, n)
        for _ in range(bare):
            with tb.cpu("aten::addmm"):
                bare_memset(tb)
        kernel(tb, "aten::addmm")


def xattn_like(bare=3, sized=(), layers=2):
    tb = TB()
    with tb.region():
        for i in range(layers):
            layer_with(tb, i, bare=bare, sized=sized)
    return tb


# --- the parser: tolerate, flag, count ---------------------------------------------

def test_a_memset_with_no_device_record_is_kept_and_flagged():
    seq = parse(xattn_like())
    mem = [op for op in seq.ops if op.kind is SeqKind.MEMSET]
    assert len(mem) == 6 and all(op.no_device_record for op in mem)
    assert all(op.kernels == () and op.stream is None and op.is_device for op in mem)
    flags = S.trace_flags(seq)
    assert [f["flag"] for f in flags] == [S.NO_DEVICE_RECORD] * 6
    assert all(f["costed"] == "host" for f in flags)
    assert S.flag_counts(seq) == {S.IMPLICIT_WAIT: 0, S.NO_DEVICE_RECORD: 6}
    assert M.count(seq) == 6


def test_a_memset_with_a_device_record_is_not_flagged():
    seq = parse(xattn_like(bare=0, sized=(1024,)))
    assert M.count(seq) == 0 and S.trace_flags(seq) == ()


def test_a_kernel_with_no_device_record_is_still_an_incomplete_trace():
    tb = TB()
    with tb.region():
        tb.call("cudaLaunchKernel")
        kernel(tb)
    with pytest.raises(SequenceError, match="trace is incomplete"):
        parse(tb)


def test_the_arms_sync_column_does_not_count_memsets_as_waits():
    assert "flagged" not in S.sync_column(parse(xattn_like()))


def test_a_flagged_memset_count_that_differs_reads_as_different():
    a, b = parse(xattn_like(bare=3)), parse(xattn_like(bare=2))
    assert S.compare(a, b).verdict == S.DIFFERENT
    assert S.compare(a, parse(xattn_like(bare=3))).verdict == S.SAME


def test_the_flag_survives_the_record_and_an_older_record_reads_false():
    seq = parse(xattn_like())
    rec = S.to_record(seq)
    assert all("no_device_record" in op for op in rec["ops"])
    assert S.from_record(rec) == seq
    for op in rec["ops"]:
        del op["no_device_record"]
    rec["flags"] = []
    assert not any(op.no_device_record for op in S.from_record(rec).ops)


# --- costs: host counted, device zero ----------------------------------------------

def _costs(seq, *, device=1e-5, host=2e-6):
    hosts = {i: host for i, op in enumerate(seq.ops) if op.kind is not SeqKind.SYNC}
    devs = {i: device for i, op in enumerate(seq.ops) if op.is_device and not op.no_device_record}
    return hosts, devs


def test_a_flagged_memset_takes_host_cost_and_zero_device_cost():
    seq = parse(xattn_like())
    host, dev = _costs(seq)
    out = S.to_operations(seq, host=host, device=dev, link={})
    pos = M.flagged(seq)
    assert [out[i].device for i in pos] == [0.0] * len(pos)
    assert [out[i].host for i in pos] == [2e-6] * len(pos)


def test_a_flagged_memset_is_given_no_device_cost_and_still_needs_its_host_cost():
    seq = parse(xattn_like())
    host, dev = _costs(seq)
    with pytest.raises(SequenceError, match="unexpected positions"):
        S.to_operations(seq, host=host, device={**dev, M.flagged(seq)[0]: 1e-6}, link={})
    del host[M.flagged(seq)[0]]
    with pytest.raises(SequenceError, match="host costs: missing"):
        S.to_operations(seq, host=host, device=dev, link={})


def test_the_upper_bound_prices_every_flagged_memset_or_is_refused():
    seq = parse(xattn_like())
    host, dev = _costs(seq)
    pos = M.flagged(seq)
    out = S.to_operations(seq, host=host, device=dev, link={},
                          flagged_memset_s={i: 3e-6 for i in pos})
    assert [out[i].device for i in pos] == [3e-6] * len(pos)
    with pytest.raises(SequenceError, match="flagged_memset_s covers"):
        S.to_operations(seq, host=host, device=dev, link={},
                        flagged_memset_s={pos[0]: 3e-6})


def test_device_durations_skip_what_the_trace_never_held():
    tb = xattn_like(sized=(1024,))
    seq = parse(tb)
    got = S.device_durations(tb.trace(), seq)
    assert not set(got) & set(M.flagged(seq))
    assert len(got) == sum(1 for op in seq.ops if op.is_device and not op.no_device_record)


# --- the upper-bound sensitivity ---------------------------------------------------

def test_a_flagged_memset_takes_the_size_of_its_operators_recorded_memset():
    tb = xattn_like(bare=3, sized=(1024, 256))
    seq = parse(tb)
    sizes = M.flagged_sizes(tb.trace(), seq)
    assert set(sizes) == set(M.flagged(seq)) and set(sizes.values()) == {1024}   # the largest


def test_a_flagged_memset_with_no_recorded_sibling_has_no_size():
    tb = xattn_like(bare=3)
    seq = parse(tb)
    assert set(M.flagged_sizes(tb.trace(), seq).values()) == {None}


def test_the_bound_takes_the_smallest_zoo_size_not_below_else_the_largest():
    zoo = {256: 1e-6, 1024: 2e-6, 4096: 5e-6}
    secs, src = M.upper_bound_s({1: 1000, 2: 1024, 3: None, 4: 10_000}, zoo)
    assert secs == {1: 2e-6, 2: 2e-6, 3: 5e-6, 4: 5e-6}
    assert src[1] == "zoo at 1024 bytes" and src[3] == src[4] == M.LARGEST


def test_the_zoo_reads_each_sizes_longest_memset():
    tb = TB()
    with tb.region():
        for n, dur in ((256, 5.0), (256, 3.0), (1024, 8.0)):   # the longest is not the last
            sized_memset(tb, n, dur=dur)
    assert M.zoo_durations(tb.trace()) == {256: pytest.approx(5e-6), 1024: pytest.approx(8e-6)}


def test_the_shift_is_the_prediction_with_the_bound_minus_the_model():
    seq = parse(xattn_like())
    host, dev = _costs(seq)

    def predict(operations):                      # stands in for the recursion
        return sum(o.host + o.device for o in operations)

    upper = {i: 4e-6 for i in M.flagged(seq)}
    got = M.prediction_shift(seq, host=host, device=dev, link={}, predict=predict, upper=upper)
    assert got["flagged_memsets"] == 6
    assert got["shift_s"] == pytest.approx(6 * 4e-6)
    assert got["upper_bound_s"] - got["prediction_s"] == pytest.approx(got["shift_s"])


def test_no_flagged_memset_means_no_shift():
    seq = parse(xattn_like(bare=0, sized=(64,)))
    host, dev = _costs(seq)
    got = M.prediction_shift(seq, host=host, device=dev, link={},
                             predict=lambda o: 1.0, upper={})
    assert got == {"flagged_memsets": 0, "prediction_s": 1.0, "upper_bound_s": 1.0,
                   "shift_s": 0.0}
