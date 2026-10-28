"""Figure 1's events table (C5) from a hand-written trace: every time below
is worked from `_figure_fixtures.trace()`, with layer 1's annotation at
200 us as zero."""

from __future__ import annotations

import json

import pytest

from attnbench.components import op_sequence as ops
from attnbench.figures import fig1_events as F

import _figure_fixtures as FX

EXPECTED_LAYER_1 = """\
gpu,lane,op,start_ms,end_ms,kind
L4,host,attnbench::host_compute,0.010000,0.050000,mask_build
L4,host,aten::mm,0.056000,0.061000,kernel
L4,device,gemm,0.070000,0.120000,kernel
L4,host,aten::_local_scalar_dense,0.072000,0.076000,sync
L4,host,aten::_local_scalar_dense,0.077000,0.127000,sync
L4,device,Memcpy DtoH (Device -> Pageable),0.120000,0.122000,copy
L4,host,aten::softmax,0.140000,0.145000,kernel
L4,device,softmax,0.150000,0.350000,kernel
"""


def test_one_layer_becomes_the_hand_worked_rows():
    doc = FX.trace()
    rows = F.layer_events(doc, FX.parse(doc), 1, gpu="L4")
    assert F.to_csv(rows) == EXPECTED_LAYER_1
    # The softmax kernel ends after the layer's annotation (0.3 ms): the
    # device lag the figure shows is kept, not clipped.
    assert rows[-1]["end_ms"] > 0.3


def test_microseconds_become_milliseconds_in_one_place():
    assert F.US_PER_MS == 1000.0
    assert F.us_to_ms(1500) == 1.5 and F.us_to_ms(0.001) == 0.000001
    assert F.us_to_ms(12.3456789) == 0.012346          # the nanosecond, the trace's resolution
    import inspect
    src = inspect.getsource(F)
    assert src.count("/ US_PER_MS") == 1
    for bad in ("/ 1000", "/1000", "* 1e-3", "*1e-3", "/ 1e3", "* 0.001"):
        assert bad not in src, bad


def test_kinds_follow_the_parse_and_the_device_activity():
    doc = FX.trace()
    seq = FX.parse(doc)
    rows = F.layer_events(doc, seq, 1, gpu="L4")
    for r in rows:
        op = seq.ops[r["seq_index"]]
        assert r["parsed_kind"] == op.kind.value and r["mechanism"] == (op.mechanism or "")
        assert r["kind"] in F.KINDS
        if r["lane"] == "host":
            assert r["kind"] == F.HOST_KIND[op.kind]
    # The .item(): a sync on the host lane, its copy a copy on the device lane.
    item = [(r["lane"], r["kind"]) for r in rows if r["mechanism"] == "item"]
    assert item == [("host", "sync"), ("host", "sync"), ("device", "copy")]
    # Every parsed kind has a figure kind, and only the five exist.
    assert set(F.HOST_KIND) == set(ops.SeqKind)
    assert set(F.HOST_KIND.values()) | set(F.DEVICE_KIND.values()) <= set(F.KINDS)
    assert F.KINDS == ("mask_build", "sync", "copy", "kernel", "other")


def test_the_gpu_label_must_name_the_traces_device():
    doc = FX.trace()
    for bad in ("A100", "", "L4,x", None):
        with pytest.raises(F.Fig1Error, match="gpu label|does not name"):
            F.layer_events(doc, FX.parse(doc), 1, gpu=bad)
    a100 = FX.trace(gpu="NVIDIA A100-SXM4-80GB")
    assert {r["gpu"] for r in F.layer_events(a100, FX.parse(a100), 1, gpu="A100")} == {"A100"}


def test_another_layer_holds_only_its_own_operations():
    doc = FX.trace()
    rows = F.layer_events(doc, FX.parse(doc), 0, gpu="L4")
    assert [(r["lane"], r["op"], r["start_ms"], r["kind"]) for r in rows] == [
        ("host", "aten::add", 0.011, "kernel"), ("device", "add_kernel", 0.020, "kernel")]


def test_a_layer_with_no_operations_or_two_annotations_is_refused():
    doc = FX.trace()
    with pytest.raises(F.Fig1Error, match="0 annotations"):
        F.layer_events(doc, FX.parse(doc), 5, gpu="L4")
    twice = FX.with_event(doc, FX._x("user_annotation", ops.layer_annotation(1), 700, 10))
    with pytest.raises(F.Fig1Error, match="2 annotations"):
        F.layer_events(twice, FX.parse(doc), 1, gpu="L4")


def test_a_sequence_from_a_banked_record_is_refused_without_its_trace_parse():
    doc = FX.trace()
    banked = ops.from_record(ops.to_record(FX.parse(doc)))     # no correlation ids
    with pytest.raises(F.Fig1Error, match="no correlation id"):
        F.layer_events(doc, banked, 1, gpu="L4")


def test_a_call_no_parsed_operation_accounts_for_is_refused():
    doc = FX.trace()
    seq = FX.parse(doc)
    extra = FX.with_event(doc, FX._call("cudaLaunchKernel", 400, 5, 99))
    extra = FX.with_event(extra, FX._dev("kernel", "late", 560, 5, 99))
    with pytest.raises(F.Fig1Error, match="no parsed operation accounts for"):
        F.layer_events(extra, seq, 1, gpu="L4")


def test_an_operation_missing_from_the_trace_is_refused():
    doc = FX.trace()
    seq = FX.parse(doc)
    gone = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                  if (e.get("args") or {}).get("correlation") != 14])
    with pytest.raises(F.Fig1Error, match="has no call in layer 1"):
        F.layer_events(gone, seq, 1, gpu="L4")
    no_mark = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                     if e["name"] != ops.HOST_COMPUTE])
    with pytest.raises(F.Fig1Error, match="host_compute span"):
        F.layer_events(no_mark, seq, 1, gpu="L4")


def test_a_folded_copy_needs_its_wait_in_the_layer():
    doc = FX.trace()
    seq = FX.parse(doc)
    no_wait = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                     if e["name"] != "cudaStreamSynchronize"])
    with pytest.raises(F.Fig1Error, match="the wait is not in the layer"):
        F.layer_events(no_wait, seq, 1, gpu="L4")


def test_files_are_parsed_again_and_must_give_the_banked_sequence(tmp_path):
    doc = FX.trace()
    t, s = FX.write_pair(tmp_path, doc)
    assert F.to_csv(F.events_from_files(t, s, 1, gpu="L4")) == EXPECTED_LAYER_1
    # A record banked from another trace (one launch fewer) is refused.
    other = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                   if (e.get("args") or {}).get("correlation") != 10])
    t2, s2 = FX.write_pair(tmp_path, doc, name="b", seq=FX.parse(other))
    with pytest.raises(F.Fig1Error, match="does not parse to the sequence banked"):
        F.events_from_files(t2, s2, 1, gpu="L4")


def test_a_record_with_a_wrong_or_missing_hash_is_refused(tmp_path):
    t, s = FX.write_pair(tmp_path, FX.trace())
    rec = json.loads(s.read_text())
    rec["batch"] = 2
    s.write_text(json.dumps(rec))
    with pytest.raises(F.Fig1Error, match="sha256"):
        F.events_from_files(t, s, 1, gpu="L4")
    rec.pop("sha256")
    s.write_text(json.dumps(rec))
    with pytest.raises(F.Fig1Error, match="sha256"):
        F.events_from_files(t, s, 1, gpu="L4")


def test_gpus_share_one_file_with_distinct_labels_and_it_is_never_overwritten(tmp_path):
    doc = FX.trace()
    seq = FX.parse(doc)
    a = F.layer_events(doc, seq, 1, gpu="L4")
    a100 = FX.trace(gpu="NVIDIA A100-SXM4-80GB")
    b = F.layer_events(a100, FX.parse(a100), 1, gpu="A100")
    out = F.write_events([a, b], tmp_path / "events.csv")
    lines = out.read_text().splitlines()
    assert len(lines) == 1 + len(a) + len(b)
    assert {l.split(",")[0] for l in lines[1:]} == {"L4", "A100"}
    with pytest.raises(FileExistsError):
        F.write_events([a], out)
    with pytest.raises(F.Fig1Error, match="distinct"):
        F.write_events([a, a], tmp_path / "dup.csv")
    with pytest.raises(F.Fig1Error, match="schema"):
        F.to_csv([dict(a[0], kind="launch")])


def test_the_script_writes_the_same_table(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "figures"))
    import fig1_events as script
    t, s = FX.write_pair(tmp_path, FX.trace())
    out = tmp_path / "events.csv"
    assert script.main(["--layer", "1", "--out", str(out), "--gpu", "L4", str(t), str(s)]) == 0
    assert out.read_text() == EXPECTED_LAYER_1
    assert script.main(["--layer", "1", "--out", str(out), "--gpu", "L4", str(t), str(s)]) == 2


def test_a_host_compute_span_no_operation_accounts_for_is_refused():
    doc = FX.trace()
    seq = FX.parse(doc)
    extra = FX.with_event(doc, FX._x("user_annotation", ops.HOST_COMPUTE, 400, 10))
    with pytest.raises(F.Fig1Error, match="1 host_compute spans in layer 1 with no"):
        F.layer_events(extra, seq, 1, gpu="L4")


def test_an_operations_host_calls_come_before_its_device_work():
    # The copy's device work (274 us) starts before the wait is called
    # (277 us); the rows still list both host calls first.
    doc = FX.trace()
    for e in doc["traceEvents"]:
        if e["name"] == "gemm":
            e["dur"] = 3
        if e["name"].startswith("Memcpy"):
            e["ts"] = 274
    rows = [r for r in F.layer_events(doc, FX.parse(doc), 1, gpu="L4")
            if r["seq_index"] == 3]
    assert [(r["lane"], r["start_ms"]) for r in rows] == [
        ("host", 0.072), ("host", 0.077), ("device", 0.074)]
