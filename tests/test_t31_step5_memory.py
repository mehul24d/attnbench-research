"""Step 5's memory fix (researcher's ruling, 2 Nov 2026, after T3.1's third
session, whose step 5 the out-of-memory killer took at 31.6 GB): the
isolated replay kept as a compact event stream and read once
(`trace_stream`), each cell in its own process under an RSS ceiling
(`resource_guard`), each cell's manifest with its peak RSS, and per-cell
resume within one code identity and instance spec.

The parity tests read the banked isolated traces of that session
(`results/t31_p1_s3`) and skip without them.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
import t31_steps as st  # noqa: E402

from attnbench import resource_guard as rg  # noqa: E402
from attnbench.components import host_profile as hp  # noqa: E402
from attnbench.components import op_sequence as ops  # noqa: E402
from attnbench.components import trace_stream as ts  # noqa: E402

BANKED = REPO / "results" / "t31_p1_s3" / "results" / "t31_p1_20261102T0643Z" / \
    "05_host_components"


# --- reading a trace one event at a time --------------------------------------------------

def _trace(events, *, pretty=True):
    doc = {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "NVIDIA L4"}],
           "traceEvents": events, "traceName": "x"}
    return json.dumps(doc, indent=2 if pretty else None)


EVENTS = [
    {"ph": "X", "cat": "user_annotation", "name": "attnbench::replay", "pid": 1, "tid": 1,
     "ts": 0.0, "dur": 100.0, "args": {"External id": 1}},
    {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": 1, "tid": 1, "ts": 1.5,
     "dur": 10.25, "args": {"Input Dims": [[2, 2]], "s": "a ] } , [ \"q\""}},
    {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 1,
     "ts": 2.0, "dur": 3.0, "args": {"correlation": 7}},
    {"ph": "X", "cat": "kernel", "name": "gemm", "pid": 0, "tid": 7, "ts": 6.0, "dur": 4.0,
     "args": {"correlation": 7, "stream": 7}},
    {"ph": "s", "cat": "ac2g", "id": 7, "pid": 1, "tid": 1, "ts": 2.0},
    {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernel", "pid": 1, "tid": 1,
     "ts": 20, "dur": 1},
    {"ph": "X", "cat": "cpu_op", "name": "aten::bad", "pid": 1, "tid": 1, "ts": "x", "dur": 1},
    {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "python"}},
    {},
]


@pytest.mark.parametrize("chunk", [1, 7, 64, 1 << 20])
@pytest.mark.parametrize("pretty", [True, False])
def test_the_stream_gives_the_events_json_load_gives_at_any_chunk_size(tmp_path, chunk, pretty):
    p = tmp_path / "t.json"
    p.write_text(_trace(EVENTS, pretty=pretty))
    assert list(ts.iter_trace_events(p, chunk=chunk)) == list(enumerate(EVENTS))


def test_a_gzipped_trace_streams_too(tmp_path):
    p = tmp_path / "t.json.gz"
    with gzip.open(p, "wt") as f:
        f.write(_trace(EVENTS))
    assert [i for i, _ in ts.iter_trace_events(p, chunk=5)] == list(range(len(EVENTS)))


@pytest.mark.parametrize("text, why", [
    ('{"other": []}', "no traceEvents"),
    ('{"traceEvents": [{"ph": "X"}, ', "not terminated"),
    ('{"traceEvents": [{"ph": "X", }]}', "not valid JSON"),
])
def test_a_trace_that_is_not_whole_is_refused(tmp_path, text, why):
    p = tmp_path / "t.json"
    p.write_text(text)
    with pytest.raises(ts.TraceStreamError, match=why):
        list(ts.iter_trace_events(p, chunk=4))


def _fields(evs):
    return [(e.idx, e.cat, e.name, e.pid, e.tid, e.ts, e.end) for e in evs]


def test_the_compact_stream_keeps_what_read_isolated_reads_with_the_original_indices(tmp_path):
    p = tmp_path / "isolated_host.json"
    p.write_text(_trace(EVENTS))
    info = ts.write_compact(p)
    assert Path(info["path"]).name == "isolated_host.events.jsonl"
    assert info["events"] == len(EVENTS) and info["kept"] == 4
    full = [e for e in ops._events(ops.load_trace(p)) if e.cat in ts.ISOLATED_CATS]
    back = ts.load_compact(info["path"])
    assert _fields(back) == _fields(full)
    assert [e.idx for e in back] == [0, 1, 2, 5]       # the kernel and the flow are dropped
    assert all(dict(e.args) == {} for e in back)
    assert not Path(str(info["path"]) + ".part").exists()


def test_a_failed_compaction_leaves_no_partial_file(tmp_path):
    p = tmp_path / "isolated_host.json"
    p.write_text('{"traceEvents": [{"ph": "X"}, ')
    with pytest.raises(ts.TraceStreamError):
        ts.write_compact(p)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["isolated_host.json"]
    with pytest.raises(FileNotFoundError):
        ts.write_compact(tmp_path / "missing.json")
    assert sorted(x.name for x in tmp_path.iterdir()) == ["isolated_host.json"]


def test_a_file_that_is_not_a_compact_stream_is_refused(tmp_path):
    p = tmp_path / "x.events.jsonl"
    p.write_text('{"schema": "something else"}\n')
    with pytest.raises(ts.TraceStreamError, match="not a"):
        ts.load_compact(p)
    assert ts.is_compact(p) and not ts.is_compact(tmp_path / "x.json") and not ts.is_compact({})


# --- parity on the banked isolated traces ------------------------------------------------

PARITY_CELLS = ["oracle_d0.25_L2048_b1", "xattention_dcal_L2048_b1", "meanpool_d0.5_L2048_b1",
                "dense_d1_L2048_b1"]


def _banked(cell):
    d = BANKED / cell
    if not (d / "isolated_host.json").exists():
        pytest.skip(f"needs the banked cell {d}")
    return d


@pytest.mark.parametrize("cell", PARITY_CELLS)
def test_the_compact_stream_reads_as_the_full_trace_on_a_banked_cell(tmp_path, cell):
    from attnbench.components import recorder
    d = _banked(cell)
    seq = recorder.load_recording(d / "recording").sequence
    info = ts.write_compact(d / "isolated_host.json", tmp_path / ("x" + ts.SUFFIX))
    full = hp.read_isolated(d / "isolated_host.json", seq)
    comp = hp.read_isolated(info["path"], seq)
    assert hp.read_isolated(comp, seq) is comp
    assert _fields(comp.launch.values()) == _fields(full.launch.values())
    assert {k: _fields(v) for k, v in comp.calls.items()} == \
        {k: _fields(v) for k, v in full.calls.items()}
    assert _fields(comp.flushes.items) == _fields(full.flushes.items)
    pos = hp.isolated_positions(seq, st._isolated_ops())
    assert hp.isolated_host_s(comp, seq, pos) == hp.isolated_host_s(full, seq, pos)


@pytest.mark.parametrize("cell", PARITY_CELLS[:3])
def test_the_chain_check_from_the_stream_is_the_banked_one(tmp_path, cell):
    from attnbench.components import recorder
    d = _banked(cell)
    seq = recorder.load_recording(d / "recording").sequence
    null = hp.from_record(json.loads((d / "host_profile_null_pass.json").read_text()))
    banked = json.loads((d / "chain_check.json").read_text())
    info = ts.write_compact(d / "isolated_host.json", tmp_path / ("x" + ts.SUFFIX))
    got = hp.chain_check(info["path"], seq, null, aggregate_tol=banked["aggregate_tol"])
    full = hp.chain_check(d / "isolated_host.json", seq, null,
                          aggregate_tol=banked["aggregate_tol"])
    assert got == full                                     # the stream changes nothing
    # and both are the session's own, up to the order of a sum: the null
    # pass's profile is read back from its record here
    got = json.loads(json.dumps(got, default=str))
    assert _close(got, {k: v for k, v in banked.items() if k not in ("gated", "why_not_gated")})


def _close(a, b, rel=1e-12) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b, rel=rel)
    return a == b


def test_compact_isolated_deletes_the_full_trace_and_names_the_stream_only(tmp_path):
    p = tmp_path / "isolated_host.json"
    p.write_text(_trace(EVENTS))
    stream, counts = st.compact_isolated(p)
    assert not p.exists() and stream == tmp_path / "isolated_host.events.jsonl"
    assert counts["stream"] == "isolated_host.events.jsonl" and "path" not in counts
    assert counts["kept"] == 4 and counts["source_bytes"] > counts["bytes"] > 0


# --- the guard --------------------------------------------------------------------------------

def _py(code):
    return [sys.executable, "-c", code]


def test_a_child_over_the_ceiling_is_killed_and_says_so(tmp_path):
    with open(tmp_path / "log", "w") as log:
        g = rg.run_guarded(_py("import time; b = bytearray(400 << 20); b[::4096] = b'x' * "
                               "len(b[::4096]); time.sleep(60)"),
                           ceiling_mb=200, poll_s=0.05, log=log)
    assert g.over_ceiling and g.returncode != 0 and g.seconds < 30
    assert g.sampled_peak_mb >= 200
    assert g.record()["over_ceiling"] is True and g.record()["rss_ceiling_mb"] == 200


def test_a_child_under_the_ceiling_runs_to_its_end_with_its_peak_sampled(tmp_path):
    with open(tmp_path / "log", "w") as log:
        g = rg.run_guarded(_py("import time; time.sleep(0.5); print('done')"),
                           ceiling_mb=4096, poll_s=0.05, log=log)
    assert g.returncode == 0 and not g.over_ceiling and 0 < g.sampled_peak_mb < 4096
    assert (tmp_path / "log").read_text().strip() == "done"


def test_a_ceiling_must_be_positive():
    with pytest.raises(ValueError):
        rg.run_guarded(_py("pass"), ceiling_mb=0)
    assert rg.self_peak_mb() > 0 and rg.rss_mb(10 ** 7) is None


# --- the step: manifests, the ceiling, resume -------------------------------------------------

import test_t31_wiring as tw  # noqa: E402


def _step(tmp_path, monkeypatch, *, spawned=None):
    h = tw.FakeHarness()
    tw._components(monkeypatch, h)
    tw.in_process_cells(monkeypatch, spawned=spawned)
    session = tw._session(tmp_path)
    out = st.step_dir(session, "host_components")
    return SimpleNamespace(dry_run=False, session=str(session), out=str(out)), out


def test_each_cell_has_a_manifest_with_its_identity_instance_and_peak(tmp_path, monkeypatch):
    args, out = _step(tmp_path, monkeypatch)
    r = st.step_host_components(args, tw._grid())
    assert r["cells"] == 3 and r["resumed"] == [] and r["rss_ceiling_gb"] == 26.0
    for c in json.loads((out / "cells_index.json").read_text())["cells"]:
        m = json.loads((out / c["dir"] / st.CELL_MANIFEST).read_text())
        assert m["status"] == "passed" and m["error"] is None
        assert m["identity"] == tw.IDENTITY and m["instance"] == tw.INSTANCE
        assert m["sampled_peak_rss_mb"] == 900.0 and m["self_peak_rss_mb"] > 0
        assert m["peak_rss_mb"] == max(900.0, m["self_peak_rss_mb"])
        assert m["rss_ceiling_mb"] == 26.0 * 1024 and m["cell"]["arm"] == c["arm"]
        assert r["peak_rss_mb"][c["dir"]] == m["peak_rss_mb"]
    assert (out / st.PRICE_CACHE).exists()


def test_a_cell_over_the_ceiling_stops_the_step_with_its_record(tmp_path, monkeypatch):
    args, out = _step(tmp_path, monkeypatch)
    ran = []

    def spawn(args, cfg, cell, cd, *, ceiling_mb):
        ran.append(cell)
        return {"returncode": -9, "seconds": 3.0, "sampled_peak_rss_mb": ceiling_mb + 5,
                "rss_ceiling_mb": ceiling_mb, "over_ceiling": True}
    monkeypatch.setattr(st, "spawn_cell", spawn)
    with pytest.raises(st.StepFailed, match="RSS ceiling") as e:
        st.step_host_components(args, dict(tw._grid(), host_cell_rss_ceiling_gb=20))
    assert len(ran) == 1                                   # the first failure stops the step
    m = json.loads((st.cell_dir(out, *ran[0]) / st.CELL_MANIFEST).read_text())
    assert m["status"] == "over_rss_ceiling" and m["peak_rss_mb"] == 20 * 1024 + 5
    assert e.value.findings["cell_manifest"] == m and e.value.findings["cells_passed"] == 0
    assert not (out / "cells_index.json").exists()


def test_a_cell_that_fails_in_its_process_records_the_error(tmp_path, monkeypatch):
    args, out = _step(tmp_path, monkeypatch)
    cfg = dict(tw._grid(), arms=["xattention"], densities={"xattention": ["calibrated"]})

    class NoDensity(tw.FakeHarness):
        def ready(self, name, L, **kw):
            a = super().ready(name, L, **kw)
            a.backend = SimpleNamespace(last_layer_density=[])
            return a
    monkeypatch.setattr(st, "_harness", lambda args: NoDensity())
    with pytest.raises(st.StepFailed, match="keeps no mask density"):
        st.step_host_components(args, cfg)
    cd = st.cell_dir(out, "xattention", None, 2048, 1)
    m = json.loads((cd / st.CELL_MANIFEST).read_text())
    assert m["status"] == "failed" and "no mask density" in m["error"]
    assert not (cd / st.CELL_INDEX).exists()


def test_a_passed_cell_of_the_same_identity_and_instance_is_not_run_again(tmp_path, monkeypatch):
    spawned = []
    args, out = _step(tmp_path, monkeypatch, spawned=spawned)
    first = st.step_host_components(args, tw._grid())
    assert len(spawned) == 3
    spawned.clear()
    again = st.step_host_components(args, tw._grid())
    assert spawned == [] and sorted(again["resumed"]) == sorted(first["peak_rss_mb"])
    assert json.loads((out / "cells_index.json").read_text())["cells"] == \
        [json.loads((out / c / st.CELL_INDEX).read_text()) for c in again["resumed"]]


@pytest.mark.parametrize("change", ["commit", "dirty", "library", "instance", "failed"])
def test_a_cell_of_another_identity_or_instance_or_a_failed_one_is_run_afresh(
        tmp_path, monkeypatch, change):
    spawned = []
    args, out = _step(tmp_path, monkeypatch, spawned=spawned)
    st.step_host_components(args, tw._grid())
    cd = st.cell_dir(out, *st.cells(tw._grid())[0])
    (cd / "stale.txt").write_text("from the earlier identity")
    ident, inst = dict(tw.IDENTITY), dict(tw.INSTANCE)
    if change == "commit":
        ident["git_commit"] = "b" * 40
    elif change == "dirty":
        ident["git_dirty"] = True
    elif change == "library":
        ident["libraries"] = [["torch", "2.9.2"]]
    elif change == "instance":
        inst["instance_type"] = "g2-standard-16"
    else:
        m = json.loads((cd / st.CELL_MANIFEST).read_text())
        (cd / st.CELL_MANIFEST).write_text(json.dumps(dict(m, status="failed")))
    monkeypatch.setattr(st, "cell_identity", lambda: ident)
    monkeypatch.setattr(st, "instance_spec", lambda: inst)
    spawned.clear()
    r = st.step_host_components(args, tw._grid())
    if change == "failed":
        assert spawned == [st.cells(tw._grid())[0]] and len(r["resumed"]) == 2
    else:
        assert len(spawned) == 3 and r["resumed"] == []
    assert not (cd / "stale.txt").exists()                 # cleared, never mixed
    if change == "dirty":
        # a dirty tree's cells pass, and are never resumed: its code is not pinned
        spawned.clear()
        assert st.step_host_components(args, tw._grid())["resumed"] == [] and len(spawned) == 3


def test_the_bare_prices_are_one_cache_for_the_step(tmp_path):
    from attnbench.components import bare_timer as bt
    path = tmp_path / st.PRICE_CACHE
    a = bt.Pricer(reps=bt.MIN_REPS, sync=lambda: None)
    key = ("aten::mm", "spec", (("tensor", (2, 2), (2, 1), "torch.float32", "cuda"), None))
    a.prices[key] = 1.5e-6
    st.save_prices(a, path)
    b = st.load_prices(bt.Pricer(reps=bt.MIN_REPS, sync=lambda: None), path)
    assert b.prices == {key: 1.5e-6}
    assert b.time(key, lambda: pytest.fail("a cached signature is not timed again")) == 1.5e-6
    assert st.load_prices(SimpleNamespace(prices={}), tmp_path / "none.pkl").prices == {}


def test_the_child_command_names_the_cell_and_the_boundary(tmp_path):
    args = SimpleNamespace(out="o", session="s", cpu_boundary=True, config="c.json")
    cmd = st.cell_command(args, ("xattention", None, 32768, 1))
    assert cmd[1:3] == [str(st.ROOT / "scripts" / "t31_steps.py"), "host-cell-one"]
    assert cmd[cmd.index("--density") + 1] == "calibrated"
    assert cmd[-3:] == ["--cpu-boundary", "--config", "c.json"]
    ns = st.parser().parse_args(cmd[2:])
    cell = (ns.arm, st.nominal_density(ns.density), ns.length, ns.batch)
    assert cell == ("xattention", None, 32768, 1)
    ns = st.parser().parse_args(st.cell_command(
        SimpleNamespace(out="o", session="s"), ("oracle", 0.25, 2048, 1))[2:])
    assert st.cell_dir(Path("o"), ns.arm, st.nominal_density(ns.density), ns.length,
                       ns.batch).name == "oracle_d0.25_L2048_b1"
