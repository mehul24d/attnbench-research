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
panel,seq_index,lane,kind,mechanism,category,name,start_us,end_us,dur_us,correlation,stream,payload_bytes
L4,1,host,host_compute,,user_annotation,attnbench::host_compute,10.000,50.000,40.000,,,0
L4,2,host,kernel,,cuda_runtime,cudaLaunchKernel,56.000,61.000,5.000,11,,0
L4,2,device,kernel,,kernel,gemm,70.000,120.000,50.000,11,7,0
L4,3,host,sync,item,cuda_runtime,cudaMemcpyAsync,72.000,76.000,4.000,12,,4
L4,3,host,sync,item,cuda_runtime,cudaStreamSynchronize,77.000,127.000,50.000,13,,4
L4,3,device,sync,item,gpu_memcpy,Memcpy DtoH (Device -> Pageable),120.000,122.000,2.000,12,7,4
L4,4,host,kernel,,cuda_runtime,cudaLaunchKernel,140.000,145.000,5.000,14,,0
L4,4,device,kernel,,kernel,softmax,150.000,350.000,200.000,14,7,0
"""


def test_one_layer_becomes_the_hand_worked_rows():
    doc = FX.trace()
    rows = F.layer_events(doc, FX.parse(doc), 1, panel="L4")
    assert F.to_csv(rows) == EXPECTED_LAYER_1
    # The softmax kernel ends after the layer's annotation (300 us): the
    # device lag the figure shows is kept, not clipped.
    assert rows[-1]["end_us"] > 300


def test_kinds_and_mechanisms_are_the_parsers_not_the_converters():
    doc = FX.trace()
    seq = FX.parse(doc)
    rows = F.layer_events(doc, seq, 1, panel="L4")
    for r in rows:
        op = seq.ops[r["seq_index"]]
        assert r["kind"] == op.kind.value and r["mechanism"] == (op.mechanism or "")
        assert r["payload_bytes"] == op.payload_bytes


def test_another_layer_holds_only_its_own_operations():
    doc = FX.trace()
    rows = F.layer_events(doc, FX.parse(doc), 0, panel="L4")
    assert [(r["lane"], r["name"], r["start_us"]) for r in rows] == [
        ("host", "cudaLaunchKernel", 11.0), ("device", "add_kernel", 20.0)]


def test_a_layer_with_no_operations_or_two_annotations_is_refused():
    doc = FX.trace()
    with pytest.raises(F.Fig1Error, match="0 annotations"):
        F.layer_events(doc, FX.parse(doc), 5, panel="L4")
    twice = FX.with_event(doc, FX._x("user_annotation", ops.layer_annotation(1), 700, 10))
    with pytest.raises(F.Fig1Error, match="2 annotations"):
        F.layer_events(twice, FX.parse(doc), 1, panel="L4")


def test_a_sequence_from_a_banked_record_is_refused_without_its_trace_parse():
    doc = FX.trace()
    banked = ops.from_record(ops.to_record(FX.parse(doc)))     # no correlation ids
    with pytest.raises(F.Fig1Error, match="no correlation id"):
        F.layer_events(doc, banked, 1, panel="L4")


def test_a_call_no_parsed_operation_accounts_for_is_refused():
    doc = FX.trace()
    seq = FX.parse(doc)
    extra = FX.with_event(doc, FX._call("cudaLaunchKernel", 400, 5, 99))
    extra = FX.with_event(extra, FX._dev("kernel", "late", 560, 5, 99))
    with pytest.raises(F.Fig1Error, match="no parsed operation accounts for"):
        F.layer_events(extra, seq, 1, panel="L4")


def test_an_operation_missing_from_the_trace_is_refused():
    doc = FX.trace()
    seq = FX.parse(doc)
    gone = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                  if (e.get("args") or {}).get("correlation") != 14])
    with pytest.raises(F.Fig1Error, match="has no call in layer 1"):
        F.layer_events(gone, seq, 1, panel="L4")
    no_mark = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                     if e["name"] != ops.HOST_COMPUTE])
    with pytest.raises(F.Fig1Error, match="host_compute span"):
        F.layer_events(no_mark, seq, 1, panel="L4")


def test_a_folded_copy_needs_its_wait_in_the_layer():
    doc = FX.trace()
    seq = FX.parse(doc)
    no_wait = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                     if e["name"] != "cudaStreamSynchronize"])
    with pytest.raises(F.Fig1Error, match="the wait is not in the layer"):
        F.layer_events(no_wait, seq, 1, panel="L4")


def test_files_are_parsed_again_and_must_give_the_banked_sequence(tmp_path):
    doc = FX.trace()
    t, s = FX.write_pair(tmp_path, doc)
    assert F.to_csv(F.events_from_files(t, s, 1, panel="L4")) == EXPECTED_LAYER_1
    # A record banked from another trace (one launch fewer) is refused.
    other = dict(doc, traceEvents=[e for e in doc["traceEvents"]
                                   if (e.get("args") or {}).get("correlation") != 10])
    t2, s2 = FX.write_pair(tmp_path, doc, name="b", seq=FX.parse(other))
    with pytest.raises(F.Fig1Error, match="does not parse to the sequence banked"):
        F.events_from_files(t2, s2, 1, panel="L4")


def test_a_record_with_a_wrong_or_missing_hash_is_refused(tmp_path):
    t, s = FX.write_pair(tmp_path, FX.trace())
    rec = json.loads(s.read_text())
    rec["batch"] = 2
    s.write_text(json.dumps(rec))
    with pytest.raises(F.Fig1Error, match="sha256"):
        F.events_from_files(t, s, 1, panel="L4")
    rec.pop("sha256")
    s.write_text(json.dumps(rec))
    with pytest.raises(F.Fig1Error, match="sha256"):
        F.events_from_files(t, s, 1, panel="L4")


def test_panels_share_one_file_with_distinct_names_and_it_is_never_overwritten(tmp_path):
    doc = FX.trace()
    seq = FX.parse(doc)
    a = F.layer_events(doc, seq, 1, panel="L4")
    b = F.layer_events(doc, seq, 1, panel="A100")
    out = F.write_events([a, b], tmp_path / "events.csv")
    lines = out.read_text().splitlines()
    assert len(lines) == 1 + len(a) + len(b)
    assert {l.split(",")[0] for l in lines[1:]} == {"L4", "A100"}
    with pytest.raises(FileExistsError):
        F.write_events([a], out)
    with pytest.raises(F.Fig1Error, match="distinct"):
        F.write_events([a, a], tmp_path / "dup.csv")
    with pytest.raises(F.Fig1Error, match="panel"):
        F.layer_events(doc, seq, 1, panel="a,b")


def test_the_script_writes_the_same_table(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "figures"))
    import fig1_events as script
    t, s = FX.write_pair(tmp_path, FX.trace())
    out = tmp_path / "events.csv"
    assert script.main(["--layer", "1", "--out", str(out), "--panel", "L4", str(t), str(s)]) == 0
    assert out.read_text() == EXPECTED_LAYER_1
    assert script.main(["--layer", "1", "--out", str(out), "--panel", "L4", str(t), str(s)]) == 2


def test_a_host_compute_span_no_operation_accounts_for_is_refused():
    doc = FX.trace()
    seq = FX.parse(doc)
    extra = FX.with_event(doc, FX._x("user_annotation", ops.HOST_COMPUTE, 400, 10))
    with pytest.raises(F.Fig1Error, match="1 host_compute spans in layer 1 with no"):
        F.layer_events(extra, seq, 1, panel="L4")


def test_an_operations_host_calls_come_before_its_device_work():
    # The copy's device work (274 us) starts before the wait is called
    # (277 us); the rows still list both host calls first.
    doc = FX.trace()
    for e in doc["traceEvents"]:
        if e["name"] == "gemm":
            e["dur"] = 3
        if e["name"].startswith("Memcpy"):
            e["ts"] = 274
    rows = [r for r in F.layer_events(doc, FX.parse(doc), 1, panel="L4")
            if r["seq_index"] == 3]
    assert [(r["lane"], r["start_us"]) for r in rows] == [
        ("host", 72.0), ("host", 77.0), ("device", 74.0)]
