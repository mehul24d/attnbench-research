"""Fixes for audit C4, findings 1 and 3 (op_sequence.py) and the lock
table's scored pairings. Written before the fixes. torch-free.
"""

from __future__ import annotations

import pytest

from attnbench.components import op_sequence as S
from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp, to_operations
from attnbench.lock import checklist as LC
from attnbench.predict import recursion as R

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch")


def K(name="k", stream=7, layer=0, aten="aten::mm"):
    return SeqOp(SeqKind.KERNEL, layer=layer, api="cudaLaunchKernel", aten_op=aten,
                 kernels=(name,), stream=stream)


def SY(mech="item", payload=4, layer=0, api="cudaStreamSynchronize"):
    return SeqOp(SeqKind.SYNC, layer=layer, api=api, mechanism=mech, payload_bytes=payload)


def seq(*ops, length=2048, streams=(7,)):
    return OpSequence(code_path=CP, context_length=length, batch=1, ops=tuple(ops),
                      streams=streams)


def _costs(s):
    host = {i: 1.0 for i, o in enumerate(s.ops) if o.kind is not SeqKind.SYNC}
    dev = {i: 10.0 for i, o in enumerate(s.ops) if o.is_device}
    link = {i: 1.0 for i, o in enumerate(s.ops) if o.kind is SeqKind.COPY}
    return dict(host=host, device=dev, link=link)


# --- finding 1: an implicit cudaFree wait is flagged, not refused, and not costed ------

IMPLICIT = seq(K("k1"), SY("implicit", 0, api="cudaFree"), K("k2"))


def test_an_implicit_cudafree_wait_is_not_costed_and_does_not_block_the_prediction():
    out = to_operations(IMPLICIT, **_costs(IMPLICIT))
    assert [o.kind.value for o in out] == ["kernel", "kernel"]


def test_the_implicit_wait_is_flagged_on_the_sequence():
    assert S.trace_flags(IMPLICIT) == ({"position": 1, "flag": "implicit_wait",
                                        "api": "cudaFree", "layer": 0,
                                        "costed": False},)
    assert S.trace_flags(seq(K(), SY())) == ()


def test_the_flag_reaches_the_banked_record_and_the_arms_table():
    rec = S.to_record(IMPLICIT)
    assert rec["flags"] == [dict(f) for f in S.trace_flags(IMPLICIT)]
    assert S.from_record(rec) == IMPLICIT
    assert "flagged: implicit_wait (not costed)" in S.sync_column(IMPLICIT)
    (e,) = S.sync_map(IMPLICIT)
    assert e.flagged and not e.priced


def test_a_record_whose_flags_do_not_follow_from_its_operations_is_refused():
    rec = S.to_record(IMPLICIT)
    rec["flags"] = []
    with pytest.raises(S.SequenceError, match="flags"):
        S.from_record(rec)


# --- C1's three mechanisms are priced ---------------------------------------------------

@pytest.mark.parametrize("mech", ["event_synchronize", "copy_from_host_pinned",
                                  "copy_from_host_pageable"])
def test_the_three_mechanisms_c1_added_are_priced(mech):
    assert mech in R.PRICED_MECHANISMS
    assert mech not in S.UNPRICED_MECHANISMS
    op = SY(mech, 0 if mech == "event_synchronize" else 4096)
    assert op.priced
    s = seq(K(), op, K("k2"))
    out = to_operations(s, **_costs(s))
    assert [(o.kind.value, o.mechanism) for o in out][1] == ("sync", mech)
    assert "unpriced" not in S.sync_column(s)


def test_nothing_the_parser_names_is_left_unpriced_but_the_flagged_wait():
    assert S.UNPRICED_MECHANISMS == ()
    assert S.FLAGGED_MECHANISMS == ("implicit",)


# --- finding 3: the strict "same sequence" definition ----------------------------------

BASE = seq(K("ampere_gemm_a", 7), SY("item", 4), K("flash_fwd", 7))


@pytest.mark.parametrize("other", [
    seq(K("sm90_gemm_b", 7), SY("item", 4), K("flash_fwd_h100", 7)),        # kernel names
    seq(K("g", 7, aten="aten::addmm"), SY("item", 8), K("f", 7)),          # aten op; 4 B vs 8 B
    seq(K("g", 9), SY("item", 4), K("f", 9), streams=(9,)),               # the stream's id
    seq(K("g", 7, layer=3), SY("item", 4, layer=3), K("f", 7, layer=3)),   # layer labels
], ids=["kernel_names", "aten_and_payload_in_class", "stream_id", "layers"])
def test_these_differences_keep_the_same_sequence(other):
    c = S.same_sequence(BASE, other)
    assert c.same, c.reason


@pytest.mark.parametrize("other, where", [
    (seq(K("g", 7), K("f", 7), SY("item", 4)), 1),                          # order
    (seq(K("g", 7), SY("item", 4), K("f", 8), streams=(7, 8)), 2),          # a second stream
    (seq(K("g", 7), SeqOp(SeqKind.COPY, layer=0, api="cudaMemcpyAsync",
                          direction="d2h", payload_bytes=4), K("f", 7)), 1),  # blocking or not
    (seq(K("g", 7), SY("item", 4 * 2 ** 20), K("f", 7)), 1),                # payload class
    (seq(K("g", 7), SY("bool", 4), K("f", 7)), 1),                          # mechanism
    (seq(K("g", 7), SY("item", 4)), 2),                                     # length
], ids=["order", "stream", "blocking", "payload_class", "mechanism", "length"])
def test_these_differences_make_a_different_sequence(other, where):
    c = S.same_sequence(BASE, other)
    assert not c.same and c.first_difference == where


def test_the_strict_key_is_class_stream_blocking_and_payload_class():
    k = S.strict_key(SY("item", 4), stream_index=0)
    assert k == ("sync:item", 0, True, "<=8B")
    assert S.strict_key(K(), stream_index=0) == ("kernel", 0, False, "none")
    assert [S.payload_class(n) for n in (0, 8, 9, 4096, 4097, 2 ** 20, 2 ** 20 + 1)] == \
        ["none", "<=8B", "<=4KiB", "<=4KiB", "<=1MiB", "<=1MiB", ">1MiB"]


# --- the lock table prints the pairings each hypothesis is scored on ------------------

PLAN = """| T4.1 | Fix every **[lock]** value: the H0 tolerance, the H2 tolerance | Code | | |
"""


def _lock(**over):
    lock = LC.skeleton(PLAN)
    for e in lock["values"].values():
        e.update(value=0.1, source="Appendix A")
    lock["scored_pairings"] = {"H0": ["P1", "P2"], "H2": ["P2", "P4"]}
    lock.update(over)
    return lock


def test_the_lock_table_prints_the_scored_pairings_per_hypothesis():
    c = LC.check(_lock(), PLAN)
    assert c.complete
    r = c.report()
    assert "scored pairings, H0: P1, P2" in r and "scored pairings, H2: P2, P4" in r


@pytest.mark.parametrize("bad", [None, {}, {"H0": ["P1"]}, {"H0": [], "H2": ["P2"]},
                                 {"H0": ["P1"], "H2": ["X2"]}, {"H9x": ["P1"], "H0": ["P1"],
                                                                "H2": ["P1"]}])
def test_a_lock_without_the_scored_pairings_is_not_complete(bad):
    c = LC.check(_lock(scored_pairings=bad), PLAN)
    assert not c.complete
    assert "scored pairings" in c.report()
