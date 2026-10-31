"""T2.25: the delay injector, on CPU with fakes and no torch.

The recorded sequence below is a small dense prefill with every shape the
injector has to handle: one-operation calls, two equal calls in a row, one
call that issues two operations, a host-compute marker (dropped by
`to_operations`, so recursion indices and sequence positions differ), a copy
and one of the arm's own synchronisations.

    pos  kind          aten op           layer   call   recursion index
     0   kernel        aten::embedding   None    0      0
     1   kernel        aten::mm          0       1      1
     2   kernel        aten::mm          0       2      2
     3   kernel        aten::sort        0       3      3
     4   kernel        aten::sort        0       3      4
     5   host compute                    0       -      -
     6   copy          aten::copy_       1       4      5
     7   sync (item)   aten::item        1       -      6
     8   kernel        aten::mm          1       5      7
                                                         8 closing synchronise
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from attnbench.components import delay_injector as di
from attnbench.components.delay_injector import Call, InjectionError
from attnbench.components.op_sequence import (CodePath, OpSequence, SeqKind, SeqOp,
                                              record_sha256, to_operations, to_record)
from attnbench.predict import recursion
from attnbench.predict.delay_injection import (BUSY, IDLE, START, Variant, inject,
                                               plan_variants)

from test_predict_recursion import COSTS

REPO = Path(__file__).resolve().parents[1]

K, C, S, M = SeqKind.KERNEL, SeqKind.COPY, SeqKind.SYNC, SeqKind.HOST_COMPUTE
OPS = (
    SeqOp(K, layer=None, aten_op="aten::embedding", kernels=("emb",), stream=7),
    SeqOp(K, layer=0, aten_op="aten::mm", kernels=("gemm",), stream=7),
    SeqOp(K, layer=0, aten_op="aten::mm", kernels=("gemm",), stream=7),
    SeqOp(K, layer=0, aten_op="aten::sort", kernels=("sort_a",), stream=7),
    SeqOp(K, layer=0, aten_op="aten::sort", kernels=("sort_b",), stream=7),
    SeqOp(M, layer=0),
    SeqOp(C, layer=1, aten_op="aten::copy_", direction="h2d", payload_bytes=64,
          stream=7),
    SeqOp(S, layer=1, aten_op="aten::item", mechanism="item", payload_bytes=1),
    SeqOp(K, layer=1, aten_op="aten::mm", kernels=("gemm",), stream=7),
)
SEQ = OpSequence(code_path=CodePath(arm="dense", gpu="NVIDIA L4", implementation="sdpa"),
                 context_length=2048, batch=1, ops=OPS, streams=(7,))
CALLS = (Call("aten::embedding", None), Call("aten::mm", 0), Call("aten::mm", 0),
         Call("aten::sort", 0), Call("aten::copy_", 1), Call("aten::mm", 1))


def _base(seq=SEQ):
    host = {i: 1.0 for i, op in enumerate(seq.ops) if op.kind is not S}
    device = {i: 10.0 for i, op in enumerate(seq.ops) if op.is_device}
    link = {i: 5.0 for i, op in enumerate(seq.ops) if op.kind is C}
    return recursion.close_prefill(to_operations(seq, host=host, device=device, link=link))


BASE = _base()


def _variant(sites, mechanism="synchronize", placement=BUSY, base=BASE):
    _, positions = inject(base, sites, mechanism)
    return Variant(mechanism=mechanism, placement=placement, count=len(sites),
                   sites=tuple(sites), positions=positions, predicted=1.0,
                   predicted_waits=(1.0,) * len(sites))


class FakeArm:
    """A prefill that 'launches' its calls in order and an observer that
    reports each one, logging launches and injections in one timeline."""

    def __init__(self, calls=CALLS):
        self.calls = list(calls)
        self.log: list = []

    def fire(self):
        self.log.append("fire")

    def observer(self, run_prefill, on_call):
        self._on_call = on_call
        try:
            run_prefill()
        finally:
            self._on_call = None

    def run_prefill(self):
        for c in self.calls:
            self.log.append((c.name, c.layer))
            self._on_call(c.name, c.layer)


# --- the sequence side ------------------------------------------------------------

def test_recursion_indices_skip_the_host_compute_markers():
    assert di.recursion_positions(SEQ) == (0, 1, 2, 3, 4, 6, 7, 8)
    assert len(BASE) == 9 and recursion.is_closing_sync(BASE[-1])


def test_calls_divide_their_runs_as_the_host_profile_does():
    assert di.call_positions(SEQ, CALLS) == ((0,), (1,), (2,), (3, 4), (6,), (8,))
    assert di.launching_names(SEQ) == {"aten::embedding", "aten::mm", "aten::sort",
                                       "aten::copy_"}


def test_several_calls_each_issuing_several_operations_split_their_run_in_order():
    sorts = tuple(SeqOp(K, layer=0, aten_op="aten::sort", kernels=(f"s{i}",))
                  for i in range(6))
    seq = replace(SEQ, ops=sorts)
    calls = (Call("aten::sort", 0),) * 3
    assert di.call_positions(seq, calls) == ((0, 1), (2, 3), (4, 5))
    base = _base(seq)
    s = di.schedule_for(_variant((3, 5), base=base), base, seq, calls)
    assert s.after_call == ((1, 1), (2, 1))


def test_equal_runs_split_by_a_synchronisation_are_divided_together():
    """mm, item, mm in one layer: the pass sees two identical calls in a row
    and cannot tell where the wait fell, so the two one-position runs are
    met together (as `host_profile` does), one call each."""
    ops = (SeqOp(K, layer=1, aten_op="aten::mm"),
           SeqOp(S, layer=1, aten_op="aten::item", mechanism="item"),
           SeqOp(K, layer=1, aten_op="aten::mm"))
    seq = replace(SEQ, ops=ops)
    calls = (Call("aten::mm", 1),) * 2
    assert di.call_positions(seq, calls) == ((0,), (2,))
    base = _base(seq)
    s = di.schedule_for(_variant((2,), base=base), base, seq, calls)
    assert s.after_call == ((1, 1),)


def test_calls_that_do_not_meet_the_sequence_are_refused():
    with pytest.raises(InjectionError, match="made no launching call there"):
        di.call_positions(SEQ, CALLS[:3] + CALLS[4:])
    # three sort calls cannot divide two positions
    three_sorts = CALLS[:3] + (Call("aten::sort", 0),) * 3 + CALLS[4:]
    with pytest.raises(InjectionError, match="not a multiple"):
        di.call_positions(SEQ, three_sorts)
    with pytest.raises(InjectionError, match="after the recorded sequence ended"):
        di.call_positions(SEQ, CALLS + (Call("aten::add", None),))
    # same operator, other layer
    moved = CALLS[:5] + (Call("aten::mm", 0),)
    with pytest.raises(InjectionError, match="aten::mm in layer 1"):
        di.call_positions(SEQ, moved)


def test_graph_launches_and_entry_points_are_refused():
    graph = replace(SEQ, ops=OPS[:1] + (SeqOp(SeqKind.GRAPH, aten_op=None),) + OPS[1:])
    with pytest.raises(InjectionError, match="CUDA-graph"):
        di.launching_names(graph)
    entry = replace(SEQ, ops=OPS[:1] + (SeqOp(K, layer=0, aten_op=None),) + OPS[1:])
    with pytest.raises(InjectionError, match="entry point"):
        di.call_positions(entry, CALLS)


# --- from a plan site to a call ------------------------------------------------------

def test_sites_map_to_the_call_that_issued_them():
    # recursion 0 -> pos 0 -> call 0; recursion 4 -> pos 4, the last of the
    # sort call (3); recursion 5 -> pos 6, the copy (call 4); recursion 7 ->
    # pos 8 -> call 5. A repeated site fires twice there.
    s = di.schedule_for(_variant((0, 4, 5, 7, 7)), BASE, SEQ, CALLS)
    assert s.at_start == 0
    assert s.after_call == ((0, 1), (3, 1), (4, 1), (5, 2))
    assert s.total == 5 == s.count
    assert s.sequence_sha256 == record_sha256(to_record(SEQ))


def test_start_sites_fire_before_the_first_call():
    s = di.schedule_for(_variant((START,) * 3, "item", IDLE), BASE, SEQ, CALLS)
    assert (s.at_start, s.after_call, s.total) == (3, (), 3)


def test_a_site_inside_a_call_is_refused_not_moved():
    with pytest.raises(InjectionError, match=r"inside call 3 \(aten::sort\).*\[3, 4\]"):
        di.schedule_for(_variant((3,)), BASE, SEQ, CALLS)


def test_a_site_after_one_of_the_arms_own_synchronisations_is_refused():
    with pytest.raises(InjectionError, match="sync, which no launching call issues"):
        di.schedule_for(_variant((6,), placement=IDLE), BASE, SEQ, CALLS)


def test_planned_variants_schedule_or_refuse_by_where_spread_lands():
    """Through `plan_variants` itself. Busy sites are every kernel and copy
    (recursion 0 to 5 and 7); one injection lands on recursion 3, inside the
    sort call, two land on 1 and 5."""
    _, one, two = plan_variants(BASE, "synchronize", BUSY, (0, 1, 2), COSTS, None)
    assert one.sites == (3,) and two.sites == (1, 5)
    with pytest.raises(InjectionError, match="inside call 3"):
        di.schedule_for(one, BASE, SEQ, CALLS)
    s = di.schedule_for(two, BASE, SEQ, CALLS)
    assert s.after_call == ((1, 1), (4, 1))
    idle = plan_variants(BASE, "copy_to_host_pinned", IDLE, (0, 1, 3), COSTS, None,
                         payload_bytes=256)
    got = di.schedules(idle, BASE, SEQ, CALLS, payload_bytes=256)
    assert [x.at_start for x in got] == [0, 1, 3]
    assert all(x.after_call == () and x.payload_bytes == 256 for x in got)


# --- what is refused before the GPU ----------------------------------------------------

def test_a_plan_made_on_another_base_is_refused():
    shorter = recursion.close_prefill(BASE[:-2])
    with pytest.raises(InjectionError, match="plus the closing synchronise"):
        di.schedule_for(_variant((0,)), shorter, SEQ, CALLS)
    # same length, but a kernel where the sequence has its item wait
    wrong = list(BASE)
    wrong[6] = recursion.kernel_op(1.0, 1.0)
    with pytest.raises(InjectionError, match="base operation 6 is a kernel"):
        di.schedule_for(_variant((0,), base=wrong), wrong, SEQ, CALLS)
    # a sync of another mechanism at that place
    other = list(BASE)
    other[6] = recursion.sync_op("bool")
    with pytest.raises(InjectionError, match="base operation 6 is a sync bool"):
        di.schedule_for(_variant((0,), base=other), other, SEQ, CALLS)


def test_a_variant_whose_positions_do_not_follow_from_its_sites_is_refused():
    assert _variant((0, 4)).positions == (1, 6)
    v = replace(_variant((0, 4)), positions=(0, 6))
    with pytest.raises(InjectionError, match="planned on another base"):
        di.schedule_for(v, BASE, SEQ, CALLS)
    v = replace(_variant((0, 4)), count=3)
    with pytest.raises(InjectionError, match="2 sites for count 3"):
        di.schedule_for(v, BASE, SEQ, CALLS)


def test_payloads_loaded_injections_and_unknown_mechanisms():
    with pytest.raises(InjectionError, match="adds none"):
        di.schedule_for(_variant((0,)), BASE, SEQ, CALLS, loaded=True)
    for m in ("synchronize", "event_synchronize", "item", "bool"):
        with pytest.raises(InjectionError, match="carries no payload"):
            di.schedule_for(_variant((0,), m), BASE, SEQ, CALLS, payload_bytes=8)
    with pytest.raises(InjectionError, match="not a mechanism the recursion prices"):
        di.payload_moved("assert", 0)
    for bad in (-1, 1.5, True):
        with pytest.raises(InjectionError, match="int >= 0"):
            di.payload_moved("copy_to_host_pinned", bad)


def test_the_bytes_each_mechanism_moves():
    for m in recursion.PRICED_MECHANISMS:
        if m in di.NO_PAYLOAD:
            assert di.payload_moved(m, 0) == 0
        else:
            # payload 0 is s_fixed, which T2.5 measures at its smallest payload
            assert di.payload_moved(m, 0) == di.SMALLEST_PAYLOAD_BYTES == 1
            assert di.payload_moved(m, 4096) == 4096
    assert set(di.NO_PAYLOAD) | set(di.COPY_TO_HOST) | set(di.COPY_FROM_HOST) == \
        set(recursion.PRICED_MECHANISMS)


# --- running against a fake arm ---------------------------------------------------------

def test_calibration_records_the_calls_and_checks_them():
    arm = FakeArm()
    assert di.calibrate(arm.run_prefill, SEQ, arm.observer) == CALLS
    short = FakeArm(CALLS[:-1])
    with pytest.raises(InjectionError, match="made no launching call there"):
        di.calibrate(short.run_prefill, SEQ, short.observer)


def test_injections_fire_right_after_their_calls_and_nowhere_else():
    arm = FakeArm()
    s = di.schedule_for(_variant((0, 4, 7, 7)), BASE, SEQ, CALLS)
    run = di.injected_prefill(arm.run_prefill, s, arm.fire, arm.observer)
    assert run() == 4
    assert arm.log == [("aten::embedding", None), "fire", ("aten::mm", 0),
                       ("aten::mm", 0), ("aten::sort", 0), "fire", ("aten::copy_", 1),
                       ("aten::mm", 1), "fire", "fire"]
    # the same callable is repeatable: the timer calls it once per repeat
    arm.log.clear()
    assert run() == 4 and arm.log.count("fire") == 4


def test_start_injections_come_before_the_first_launch():
    arm = FakeArm()
    s = di.schedule_for(_variant((START, START), "bool", IDLE), BASE, SEQ, CALLS)
    di.injected_prefill(arm.run_prefill, s, arm.fire, arm.observer)()
    assert arm.log[:3] == ["fire", "fire", ("aten::embedding", None)]
    assert arm.log.count("fire") == 2


def test_the_zero_count_variant_runs_under_the_observer_and_fires_nothing():
    arm = FakeArm()
    seen = []

    def observer(run_prefill, on_call):
        seen.append("observed")
        arm.observer(run_prefill, on_call)
    s = di.schedule_for(_variant(()), BASE, SEQ, CALLS)
    assert di.injected_prefill(arm.run_prefill, s, arm.fire, observer)() == 0
    assert seen == ["observed"] and "fire" not in arm.log


def test_a_pass_that_takes_another_path_is_refused():
    s = di.schedule_for(_variant((0,)), BASE, SEQ, CALLS)
    other = FakeArm(CALLS[:2] + (Call("aten::addmm", 0),) + CALLS[3:])
    with pytest.raises(InjectionError, match="call 2 is aten::addmm in layer 0"):
        di.injected_prefill(other.run_prefill, s, other.fire, other.observer)()
    short = FakeArm(CALLS[:-1])
    with pytest.raises(InjectionError, match="made 5 launching calls"):
        di.injected_prefill(short.run_prefill, s, short.fire, short.observer)()
    long = FakeArm(CALLS + (Call("aten::mm", 1),))
    with pytest.raises(InjectionError, match="past the 6"):
        di.injected_prefill(long.run_prefill, s, long.fire, long.observer)()


def test_a_schedule_whose_count_and_placements_disagree_is_refused():
    s = di.schedule_for(_variant((0, 4)), BASE, SEQ, CALLS)
    arm = FakeArm()
    with pytest.raises(InjectionError, match="2 injections for count 3"):
        di.injected_prefill(arm.run_prefill, replace(s, count=3), arm.fire, arm.observer)
    # and a fire count that drifts from the plan is caught at the end
    bad = replace(s, after_call=((0, 1), (99, 1)))
    with pytest.raises(InjectionError, match="1 injections fired; 2 planned"):
        di.injected_prefill(arm.run_prefill, bad, arm.fire, arm.observer)()


def test_the_record_names_the_variant_and_its_sequence():
    s = di.schedule_for(_variant((0, 4)), BASE, SEQ, CALLS, payload_bytes=0)
    rec = di.to_record(s)
    assert rec == {"schema": di.SCHEMA, "mechanism": "synchronize", "placement": BUSY,
                   "count": 2, "sites": [0, 4], "payload_bytes": 0, "at_start": 0,
                   "after_call": [[0, 1], [3, 1]], "n_calls": 6,
                   "sequence_sha256": record_sha256(to_record(SEQ))}


def test_the_injector_imports_without_torch():
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['torch'] = None\n"
         "import attnbench.components.delay_injector as d\n"
         "try:\n"
         "    d.cuda_mechanism('synchronize')\n"
         "except ImportError:\n"
         "    pass\n"
         "else:\n"
         "    raise SystemExit('cuda_mechanism ran with torch blocked')\n"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr + r.stdout


# --- the observer: host costs under it, and its own effect ---------------------------

def test_the_null_pass_wrapper_runs_under_the_observer_and_checks_its_calls():
    arm = FakeArm()
    di.observed(arm.run_prefill, arm.observer)()
    assert arm.log == [(c.name, c.layer) for c in CALLS] and "fire" not in arm.log
    di.observed(arm.run_prefill, arm.observer, CALLS)()          # matches: no error
    other = FakeArm(CALLS[:2] + (Call("aten::addmm", 0),) + CALLS[3:])
    with pytest.raises(InjectionError, match="first differs .* at call 2"):
        di.observed(other.run_prefill, other.observer, CALLS)()
    short = FakeArm(CALLS[:-1])
    with pytest.raises(InjectionError, match="made 5 launching calls .* at call 5"):
        di.observed(short.run_prefill, short.observer, CALLS)()


def test_the_observer_pair_is_the_c0_variant_and_the_bare_prefill():
    arm = FakeArm()
    observed_runs = []

    def observer(run_prefill, on_call):
        observed_runs.append(1)
        arm.observer(run_prefill, on_call)
    s0 = di.schedule_for(_variant(()), BASE, SEQ, CALLS)
    with_obs, bare = di.observer_pair(arm.run_prefill, s0, arm.fire, observer)
    assert with_obs() == 0 and observed_runs == [1]
    assert bare == arm.run_prefill
    with pytest.raises(InjectionError, match="read at c = 0"):
        di.observer_pair(arm.run_prefill, di.schedule_for(_variant((0,)), BASE, SEQ, CALLS),
                         arm.fire, observer)


def test_the_observer_effect_is_the_difference_of_the_medians():
    e = di.observer_effect([3.0, 1.2, 1.1], [1.0, 0.9, 5.0, 1.0])
    assert (e.observed_s, e.unobserved_s, e.repeats) == (1.2, 1.0, (3, 4))
    assert e.effect_s == pytest.approx(0.2) and e.ratio == pytest.approx(1.2)
    assert di.observer_effect([2.0], [1.0], statistic=max).effect_s == 1.0
    for bad in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(InjectionError, match="observed repeats"):
            di.observer_effect(bad, [1.0])
        with pytest.raises(InjectionError, match="unobserved repeats"):
            di.observer_effect([1.0], bad)


# --- the plan lists only sites the injector accepts (at integration) ------------------

def test_every_site_plan_variants_returns_is_accepted_by_the_injector():
    allowed = di.injectable_sites(BASE, SEQ, CALLS)
    n_busy = len(allowed)
    assert 0 < n_busy < 7           # the unfiltered plan had seven candidates, some refused
    cases = [(BUSY, "synchronize", tuple(range(n_busy + 1))),
             (IDLE, "item", (0, 1, 4))]
    for placement, mechanism, counts in cases:
        for v in plan_variants(BASE, mechanism, placement, counts, COSTS, None,
                               sites_allowed=allowed):
            di.schedule_for(v, BASE, SEQ, CALLS)


def test_without_the_filter_the_plan_can_name_a_refused_site():
    # The break-test of the filter above: the same plan, unfiltered, is refused.
    with pytest.raises(InjectionError, match="inside call"):
        for v in plan_variants(BASE, "synchronize", BUSY, tuple(range(8)), COSTS, None):
            di.schedule_for(v, BASE, SEQ, CALLS)
