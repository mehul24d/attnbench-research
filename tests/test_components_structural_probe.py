"""T2.18: the structural probe (`components.structural_probe`).

`program(L, chunks)` writes an arm's sequence at length L from its
structure alone: per layer a prefix (a kernel and a host-to-device copy
whose payload grows as L^copy_exp), a unit (a kernel and an `item` read)
repeated once per chunk, and a suffix. It shares no code with the probe, so
the probe's extension to a target is checked against the program written
directly at the target, which is what the L4 check of record does with a
trace recorded there (`check_against_full`).
"""

from __future__ import annotations

import json
import types

import pytest
import torch

from attnbench.backends.xattention import official_chunk_size
from attnbench.components import recorder as R
from attnbench.components import structural_probe as P
from attnbench.components.op_sequence import (DIFFERENT, SAME, CodePath, OpSequence, SeqKind,
                                              SeqOp, SequenceError)

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch",
              libraries=(("torch", "2.9.1"),))
PER_1K = P.ChunkingRule("one chunk per 1024", lambda L: L // 1024)


def program(L, chunks, *, layers=(0, 1), copy_exp=1, unit_payload=lambda L, r: 0,
            extra=None, cp=CP):
    ops = []
    for layer in layers:
        ops.append(SeqOp(SeqKind.KERNEL, layer=layer, aten_op="aten::mm", kernels=("gemm",)))
        ops.append(SeqOp(SeqKind.COPY, layer=layer, api="cudaMemcpyAsync", direction="HtoD",
                         src_mem="Pageable", dst_mem="Device",
                         payload_bytes=8 * (L // 1024) ** copy_exp))
        for r in range(chunks):
            ops.append(SeqOp(SeqKind.KERNEL, layer=layer, aten_op="aten::bmm", kernels=("bmm",)))
            ops.append(SeqOp(SeqKind.SYNC, layer=layer, api="cudaStreamSynchronize",
                             aten_op="aten::_local_scalar_dense", origin="aten::item",
                             mechanism="item", payload_bytes=4 + unit_payload(L, r)))
        if extra is not None and extra(L, layer):
            ops.append(SeqOp(SeqKind.SYNC, layer=layer, mechanism="bool", payload_bytes=1))
        ops.append(SeqOp(SeqKind.KERNEL, layer=layer, kernels=("bsa_fwd",)))
    return OpSequence(code_path=cp, context_length=L, batch=1, ops=tuple(ops), streams=(7,))


def probe_seqs(rule, **kw):
    return {L: program(L, rule.chunks(L), **kw) for L in P.PROBE_LENGTHS}


@pytest.mark.parametrize("copy_exp", [0, 1, 2])
def test_an_unchunked_arm_extends_with_its_payloads_scaled(copy_exp):
    ext = P.extend(probe_seqs(P.NO_CHUNKING, copy_exp=copy_exp), P.NO_CHUNKING, 32768)
    assert ext.extended, ext.reason
    full = program(32768, 1, copy_exp=copy_exp)
    assert P.check_against_full(ext, full).same
    copies = [op.payload_bytes for op in ext.sequence.ops if op.kind is SeqKind.COPY]
    assert copies == [8 * 32 ** copy_exp] * 2
    assert sorted({e for _, _, e in ext.payload_exponents}) == sorted({copy_exp, 0})


@pytest.mark.parametrize("target", [8192, 32768, 131072])
def test_a_chunked_arm_extends_by_its_chunk_count(target):
    ext = P.extend(probe_seqs(PER_1K), PER_1K, target)
    assert ext.extended, ext.reason
    assert ext.chunks == {2048: 2, 4096: 4, target: target // 1024}
    full = program(target, target // 1024)
    assert len(ext.sequence) == len(full)
    assert P.check_against_full(ext, full).same
    assert ext.sequence.context_length == target
    assert all(op.correlation is None for op in ext.sequence.ops)


def test_the_extension_is_checked_against_the_full_sequence_with_payloads():
    ext = P.extend(probe_seqs(PER_1K), PER_1K, 32768)
    wrong_payload = program(32768, 32, copy_exp=2)
    c = P.check_against_full(ext, wrong_payload)
    assert c.verdict == DIFFERENT and c.first_difference == 1
    one_chunk_short = program(32768, 31)
    assert not P.check_against_full(ext, one_chunk_short).same


def test_a_unit_whose_repeats_differ_is_different():
    seqs = probe_seqs(PER_1K)
    b = list(seqs[4096].ops)
    b[4] = SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::addmm", kernels=("addmm",))
    seqs[4096] = OpSequence(CP, 4096, 1, tuple(b), (7,))
    ext = P.extend(seqs, PER_1K, 32768)
    assert ext.verdict == DIFFERENT and ext.sequence is None
    assert "layer 0: no prefix + unit x chunks + suffix fits" in ext.reason


def test_an_operation_only_one_short_length_has_is_different():
    seqs = probe_seqs(PER_1K, extra=lambda L, layer: L == 4096 and layer == 1)
    ext = P.extend(seqs, PER_1K, 32768)
    assert ext.verdict == DIFFERENT and "layer 1" in ext.reason


def test_a_payload_that_grows_other_than_as_a_power_of_L_is_different():
    seqs = probe_seqs(P.NO_CHUNKING)
    b = list(seqs[4096].ops)
    b[1] = SeqOp(**{**b[1].__dict__, "payload_bytes": 48})           # 16 -> 48: x3
    seqs[4096] = OpSequence(CP, 4096, 1, tuple(b), (7,))
    ext = P.extend(seqs, P.NO_CHUNKING, 32768)
    assert ext.verdict == DIFFERENT
    assert "grows from 16 to 48 bytes" in ext.reason


def test_a_payload_that_varies_by_chunk_is_not_extended():
    # each chunk's read grows with its index, as a key length that grows by
    # chunk would; repeating chunk 0 would understate every later chunk
    seqs = probe_seqs(PER_1K, unit_payload=lambda L, r: 4 * r)
    ext = P.extend(seqs, PER_1K, 32768)
    assert ext.verdict == DIFFERENT
    assert "chunk 1 carries 8 bytes where chunk 0 carries 4" in ext.reason


def test_one_chunk_count_at_both_short_lengths_cannot_check_a_different_count():
    one_then_many = P.ChunkingRule("one, then four", lambda L: 1 if L <= 4096 else 4)
    ext = P.extend(probe_seqs(one_then_many), one_then_many, 32768)
    assert ext.verdict == DIFFERENT
    assert "1 chunk(s) at both 2048 and 4096 but 4 at 32768" in ext.reason


def test_xattention_at_its_official_chunk_size_cannot_be_extended_to_32768():
    rule = P.xattention_rule()
    assert [rule.chunks(L) for L in (2048, 4096, 16384, 32768)] == [1, 1, 1, 4]
    assert official_chunk_size(32768) == 8192
    ext = P.extend(probe_seqs(rule), rule, 32768)
    assert ext.verdict == DIFFERENT and "cannot check how the unit repeats" in ext.reason
    assert P.extend(probe_seqs(rule), rule, 16384).extended


def test_fewer_chunks_at_the_longer_length_is_different():
    down = P.ChunkingRule("down", lambda L: 4 if L == 2048 else 2)
    assert P.extend(probe_seqs(down), down, 8192).verdict == DIFFERENT


def test_a_different_layer_structure_is_different():
    seqs = {2048: program(2048, 1), 4096: program(4096, 1, layers=(0, 1, 2))}
    ext = P.extend(seqs, P.NO_CHUNKING, 32768)
    assert ext.verdict == DIFFERENT and "layer structure" in ext.reason


def test_the_probe_extends_one_arm_from_exactly_two_lengths():
    other = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:triton")
    with pytest.raises(SequenceError, match="not one arm"):
        P.extend({2048: program(2048, 1), 4096: program(4096, 1, cp=other)},
                 P.NO_CHUNKING, 32768)
    with pytest.raises(SequenceError, match="exactly two"):
        P.extend({2048: program(2048, 1)}, P.NO_CHUNKING, 32768)
    zero = P.ChunkingRule("zero", lambda L: 0)
    with pytest.raises(SequenceError, match="below one"):
        P.extend(probe_seqs(P.NO_CHUNKING), zero, 32768)


def test_a_periodic_prefix_still_extends_to_the_right_length():
    # every operation of the layer has one key, so the split is ambiguous
    def same_key(L, n):
        ops = tuple(SeqOp(SeqKind.KERNEL, layer=0, aten_op="aten::bmm", kernels=("bmm",))
                    for _ in range(n + 1))
        return OpSequence(CP, L, 1, ops, (7,))
    ext = P.extend({2048: same_key(2048, 2), 4096: same_key(4096, 4)}, PER_1K, 32768)
    assert ext.extended and P.check_against_full(ext, same_key(32768, 32)).same


def test_nothing_to_check_when_the_probe_did_not_extend():
    ext = P.extend(probe_seqs(P.xattention_rule()), P.xattention_rule(), 32768)
    with pytest.raises(SequenceError, match="nothing to check"):
        P.check_against_full(ext, program(32768, 4))


# --- the probe on the recorder ------------------------------------------------

def _recorder(rule):
    mod = types.SimpleNamespace(block_sparse_attn_func=lambda *a, **k: a[0])

    def record(L, d):
        def prefill():
            q = torch.zeros(2)
            mod.block_sparse_attn_func(q, q, q, None, None, torch.ones(2, dtype=torch.int32),
                                       None, torch.ones(1, 1, 2, 2, dtype=torch.bool))
        r = R.record_arm_prefill(
            prefill, d, code_path=CP, context_length=L, batch=1, prompt_id="probe",
            cuda=False, kernel_module=mod,
            parse=lambda path, cp, *, context_length, batch: program(context_length,
                                                                     rule.chunks(context_length)))
        return r.sequence
    return record


def test_the_probe_discards_every_trace_and_keeps_the_rest_checkable(tmp_path):
    seqs = P.probe(_recorder(PER_1K), tmp_path)
    assert sorted(seqs) == [2048, 4096]
    for L in (2048, 4096):
        d = tmp_path / f"probe_{L}"
        assert not (d / "trace.json").exists()
        meta = json.loads((d / "recording.json").read_text())
        assert meta["trace_discarded"] is True and "trace.json" not in meta["files"]
        back = R.load_recording(d)
        assert back.sequence == seqs[L] and back.masks is not None
    assert P.extend(seqs, PER_1K, 32768).extended


def test_the_probe_keeps_traces_when_asked(tmp_path):
    P.probe(_recorder(PER_1K), tmp_path, keep_traces=True)
    assert (tmp_path / "probe_2048" / "trace.json").exists()
    assert "trace_discarded" not in json.loads(
        (tmp_path / "probe_2048" / "recording.json").read_text())


# --- chunked arms: three probe lengths (Appendix A, 27 Oct 2026) ----------------------

def chunked_seqs(rule, lengths=None, **kw):
    lengths = lengths or P.chunked_probe_lengths(rule)
    return {L: program(L, rule.chunks(L), **kw) for L in lengths}


# --- no probe length above the model's 32768 positions (ruling, 31 Oct 2026) ---

def test_no_probe_length_exceeds_32768():
    assert P.MAX_PROBE_LENGTH == 32768


def test_xattention_has_two_counts_so_its_third_point_is_a_full_recording_at_32768():
    rule = P.xattention_rule()
    plan = P.probe_plan(rule)
    assert plan.lengths == (20480, 28672, 32768) and plan.full_third
    assert [rule.chunks(L) for L in plan.lengths] == [3, 4, 4]
    assert "full recording at 32768" in plan.reason and "not a third count" in plan.reason
    assert all(L % P.LENGTH_STEP == 0 and L <= 32768 for L in plan.lengths)
    with pytest.raises(SequenceError, match="fewer than 3"):
        P.chunked_probe_lengths(rule)                     # three counts do not exist here


def test_a_chunk_size_that_allows_three_counts_takes_three_lengths():
    plan = P.probe_plan(PER_1K)
    assert not plan.full_third and len(plan.lengths) == 3
    assert len({PER_1K.chunks(L) for L in plan.lengths}) == 3
    assert min(PER_1K.chunks(L) for L in plan.lengths) >= 2


def test_an_unchunked_arm_takes_the_two_short_contexts():
    assert P.probe_plan(P.NO_CHUNKING).lengths == P.PROBE_LENGTHS


def test_a_rule_with_one_count_in_range_is_refused():
    one = P.ChunkingRule("one", lambda L: 1 if L < 16384 else 2)
    with pytest.raises(SequenceError, match="fewer than 2"):
        P.probe_plan(one)


def test_probe_lengths_sit_mid_band_so_a_short_prompt_keeps_its_count():
    rule = P.xattention_rule()
    for L in P.probe_plan(rule).lengths[:2]:
        assert rule.chunks(L - 1000) == rule.chunks(L) == rule.chunks(L + 1000)


def test_an_unchunked_rule_has_no_chunked_probe_lengths():
    with pytest.raises(SequenceError, match="not a chunked rule"):
        P.chunked_probe_lengths(P.NO_CHUNKING)


def test_a_rule_without_three_counts_in_range_is_refused():
    two = P.ChunkingRule("at most two", lambda L: 1 if L < 8192 else 2)
    with pytest.raises(SequenceError, match="fewer than 3"):
        P.chunked_probe_lengths(two)


@pytest.mark.parametrize("copy_exp", [0, 1, 2])
@pytest.mark.parametrize("target", [8192, 32768, 65536])
def test_a_chunked_arm_extends_from_three_lengths_checked_at_the_third(target, copy_exp):
    seqs = chunked_seqs(PER_1K, (2048, 4096, 8192), copy_exp=copy_exp)
    ext = P.extend_checked(seqs, PER_1K, target)
    assert ext.extended, ext.reason
    assert ext.checked_at == 8192 and "checked at 8192" in ext.reason
    assert ext.chunks == {2048: 2, 4096: 4, 8192: 8, target: target // 1024}
    assert P.check_against_full(ext, program(target, target // 1024, copy_exp=copy_exp)).same


def test_xattention_extends_from_its_two_lengths_checked_at_the_full_recording():
    rule = P.xattention_rule()
    plan = P.probe_plan(rule)
    seqs = chunked_seqs(rule, plan.lengths)
    ext = P.extend_checked(seqs, rule, 32768, full_third=True)
    assert ext.extended and ext.checked_at == 32768, ext.reason
    assert P.check_against_full(ext, program(32768, rule.chunks(32768))).same


def test_the_full_third_point_still_catches_a_payload_law_two_points_infer_wrongly():
    rule = P.xattention_rule()
    lengths = P.probe_plan(rule).lengths
    payload = dict(zip(lengths, (8, 16, 16)))               # L^1, then flat at 32768
    seqs = {L: replace_copy(program(L, rule.chunks(L)), payload[L]) for L in lengths}
    ext = P.extend_checked(seqs, rule, 24704, full_third=True)
    assert ext.verdict == DIFFERENT and "differs from the one recorded there" in ext.reason


def test_sharing_a_count_at_the_third_needs_the_full_third_plan():
    rule = P.xattention_rule()
    seqs = chunked_seqs(rule, P.probe_plan(rule).lengths)
    ext = P.extend_checked(seqs, rule, 24704)               # not declared full_third
    assert ext.verdict == DIFFERENT and "no two the same" in ext.reason


def test_a_payload_law_two_points_infer_wrongly_is_caught_at_the_third():
    # L^1 between 2048 and 4096, then flat: two points say L^1, the third refutes it.
    payload = {2048: 8, 4096: 16, 8192: 16}
    seqs = {L: replace_copy(program(L, PER_1K.chunks(L)), payload[L])
            for L in (2048, 4096, 8192)}
    assert P.extend({L: seqs[L] for L in (2048, 4096)}, PER_1K, 32768).extended
    ext = P.extend_checked(seqs, PER_1K, 32768)
    assert ext.verdict == DIFFERENT and "differs from the one recorded there" in ext.reason


def test_a_unit_that_changes_at_the_third_length_is_different():
    seqs = chunked_seqs(PER_1K, (2048, 4096, 8192),
                        extra=lambda L, layer: L == 8192 and layer == 1)
    ext = P.extend_checked(seqs, PER_1K, 32768)
    assert ext.verdict == DIFFERENT and ext.sequence is None


@pytest.mark.parametrize("lengths", [(1024, 2048, 4096), (2048, 2560, 4096)])
def test_probe_lengths_under_two_chunks_or_sharing_a_count_are_different(lengths):
    ext = P.extend_checked(chunked_seqs(PER_1K, lengths), PER_1K, 32768)
    assert ext.verdict == DIFFERENT and "no two the same" in ext.reason


def test_a_chunked_arm_needs_exactly_three_lengths():
    with pytest.raises(SequenceError, match="3 lengths, not 2"):
        P.extend_checked(probe_seqs(PER_1K), PER_1K, 32768)


def test_the_probe_keys_each_sequence_by_its_recorded_length(tmp_path):
    def record(L, d):
        return program(L - 97, PER_1K.chunks(L - 97))          # the prompt lands short
    seqs = P.probe(record, tmp_path, lengths=(2048, 4096))
    assert sorted(seqs) == [1951, 3999]


def test_two_budgets_recorded_at_one_length_are_refused(tmp_path):
    with pytest.raises(SequenceError, match="recorded at 2000"):
        P.probe(lambda L, d: program(2000, 1), tmp_path, lengths=(2048, 4096))


def replace_copy(seq, payload):
    from dataclasses import replace as _r
    ops = tuple(_r(o, payload_bytes=payload) if o.kind is SeqKind.COPY else o for o in seq.ops)
    return _r(seq, ops=ops)
