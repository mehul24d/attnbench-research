"""Step 6's cells, each in its own process under the RSS ceiling, with
manifests and resume as step 5's (researcher's answer, 2 Nov 2026: fix
step 6 the same way before T3.1 relaunches).

The cells run in this process through the child's own body
(`run_device_cell`, `test_t31_wiring.in_process_cells`), on the wiring
tests' fakes; the guard itself is tested in `test_t31_step5_memory`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import t31_steps as st  # noqa: E402
import test_t31_wiring as tw  # noqa: E402


def _steps(tmp_path, monkeypatch, *, device_spawned=None):
    """Step 5 run on the fakes; returns step 6's args and directory."""
    tw._components(monkeypatch, tw.FakeHarness())
    tw.in_process_cells(monkeypatch, device_spawned=device_spawned)
    session = tw._session(tmp_path)
    st.step_host_components(SimpleNamespace(
        dry_run=False, session=str(session),
        out=str(st.step_dir(session, "host_components"))), tw._grid())
    out = st.step_dir(session, "device_components")
    return SimpleNamespace(dry_run=False, session=str(session), out=str(out)), out, session


def test_each_device_cell_has_a_manifest_with_its_inputs_and_peak(tmp_path, monkeypatch):
    spawned = []
    args, out, session = _steps(tmp_path, monkeypatch, device_spawned=spawned)
    r = st.step_device_components(args, tw._grid())
    assert spawned == st.cells(tw._grid())                 # step 5's cells, in its order
    assert r["cells"] == 3 and r["resumed"] == [] and r["rss_ceiling_gb"] == 26.0
    names = [st.cell_dir(out, *c).name for c in spawned]
    assert sorted(r["cost_sources"]) == sorted(names) == sorted(r["peak_rss_mb"])
    host = st.step_dir(session, "host_components")
    link = st.step_dir(session, "link") / "link" / "link.parquet"
    for cell, name in zip(spawned, names):
        m = json.loads((out / name / st.CELL_MANIFEST).read_text())
        assert m["status"] == "passed" and m["error"] is None
        assert m["identity"] == tw.IDENTITY and m["instance"] == tw.INSTANCE
        assert m["inputs"] == {"host_cell_index_sha256": st._sha256(host / name / st.CELL_INDEX),
                               "link_sha256": st._sha256(link)}
        assert None not in m["inputs"].values()
        assert m["peak_rss_mb"] == max(900.0, m["self_peak_rss_mb"])
        assert m["cell"] == dict(zip(("arm", "density", "length", "batch"), cell))
    cells = json.loads((out / "cells.json").read_text())
    assert cells["fake"] is False
    assert cells["cells"] == [json.loads((out / n / st.CELL_INDEX).read_text()) for n in names]
    assert [st.device_cell_key(c) for c in cells["cells"]] == spawned


def test_a_device_cell_over_the_ceiling_stops_the_step(tmp_path, monkeypatch):
    args, out, _ = _steps(tmp_path, monkeypatch)
    ran = []

    def spawn(args, cfg, cell, cd, *, ceiling_mb):
        ran.append(cell)
        return {"returncode": -9, "seconds": 3.0, "sampled_peak_rss_mb": ceiling_mb + 5,
                "rss_ceiling_mb": ceiling_mb, "over_ceiling": True}
    monkeypatch.setattr(st, "spawn_device_cell", spawn)
    with pytest.raises(st.StepFailed, match="RSS ceiling") as e:
        st.step_device_components(args, dict(tw._grid(), host_cell_rss_ceiling_gb=20))
    assert len(ran) == 1
    m = json.loads((st.cell_dir(out, *ran[0]) / st.CELL_MANIFEST).read_text())
    assert m["status"] == "over_rss_ceiling" and e.value.findings["cell_manifest"] == m
    assert not (out / "cells.json").exists()


def test_the_childs_mask_diagnosis_reaches_the_parents_failure(tmp_path, monkeypatch):
    from attnbench.components import mask_diagnosis as md
    from attnbench.components import replay
    args, out, _ = _steps(tmp_path, monkeypatch)
    monkeypatch.setattr(replay, "mask_agreement", lambda cap: {"identical": False})
    monkeypatch.setattr(md, "guarded", lambda *a, **k: {"layers": [3]})
    monkeypatch.setattr(md, "summaries", lambda diag: {"first_layer": diag["layers"][0]})
    with pytest.raises(st.StepFailed, match="live masks differ from the bank") as e:
        st.step_device_components(args, tw._grid())
    assert e.value.findings["mask_diagnosis"] == {"first_layer": 3}
    cd = st.cell_dir(out, *st.cells(tw._grid())[0])
    assert json.loads((cd / "mask_diagnosis.json").read_text()) == {"layers": [3]}
    assert json.loads((cd / st.CELL_EXIT).read_text())["findings"] == {
        "mask_diagnosis": {"first_layer": 3}}
    assert e.value.findings["cell_manifest"]["status"] == "failed"


def test_an_exception_in_the_child_is_its_error_with_no_findings(tmp_path, monkeypatch):
    from attnbench.components import replay
    args, out, _ = _steps(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("capture failed")
    monkeypatch.setattr(replay, "capture", boom)
    with pytest.raises(st.StepFailed, match="RuntimeError: capture failed") as e:
        st.step_device_components(args, tw._grid())
    assert "mask_diagnosis" not in e.value.findings
    cd = st.cell_dir(out, *st.cells(tw._grid())[0])
    assert json.loads((cd / st.CELL_EXIT).read_text())["findings"] is None
    assert not (cd / st.CELL_INDEX).exists()


def test_passed_device_cells_resume_and_a_changed_host_cell_reruns(tmp_path, monkeypatch):
    spawned = []
    args, out, session = _steps(tmp_path, monkeypatch, device_spawned=spawned)
    first = st.step_device_components(args, tw._grid())
    spawned.clear()
    again = st.step_device_components(args, tw._grid())
    assert spawned == [] and sorted(again["resumed"]) == sorted(first["peak_rss_mb"])
    assert json.loads((out / "cells.json").read_text())["cells"] == [
        json.loads((out / st.cell_dir(out, *c).name / st.CELL_INDEX).read_text())
        for c in st.cells(tw._grid())]
    # a host cell priced again: its device cell is priced again, the rest resume
    cell = st.cells(tw._grid())[1]
    host = st.cell_dir(st.step_dir(session, "host_components"), *cell) / st.CELL_INDEX
    host.write_text(host.read_text() + "\n")
    spawned.clear()
    r = st.step_device_components(args, tw._grid())
    assert spawned == [cell] and len(r["resumed"]) == 2
    # a new link row: every device cell is priced again
    link = st.step_dir(session, "link") / "link" / "link.parquet"
    row = pd.read_parquet(link)
    row["bw_h2d_pinned_Bps"] = 2e10
    row.to_parquet(link)
    spawned.clear()
    r = st.step_device_components(args, tw._grid())
    assert spawned == st.cells(tw._grid()) and r["resumed"] == []


def test_the_memset_zoo_is_measured_once_for_the_step(tmp_path, monkeypatch):
    from attnbench.components import memset_flags as msf
    from attnbench.components import recorder
    args, out, _ = _steps(tmp_path, monkeypatch)
    monkeypatch.setattr(recorder, "load_recording", lambda d: SimpleNamespace(
        sequence=tw.SEQ, tape=None, masks=None, directory=tmp_path / "no_trace"))
    measured = []

    def measure(path):
        measured.append(Path(path).name)
        return {64: 1e-6, 4096: 3e-6}
    seen = []

    def upper(sizes, zoo):
        seen.append(zoo)
        return {}, {}            # the fake sequence has no memset to price
    monkeypatch.setattr(msf, "measure_zoo", measure)
    monkeypatch.setattr(msf, "count", lambda seq: 1)
    monkeypatch.setattr(msf, "flagged", lambda seq: (0,))
    monkeypatch.setattr(msf, "upper_bound_s", upper)
    monkeypatch.setattr(st, "label_outside_layers", lambda ops: ops)
    st.step_device_components(args, tw._grid())
    assert measured == ["memset_zoo_trace.json"]             # once, by the first cell
    assert seen == [{64: 1e-6, 4096: 3e-6}] * 3            # sizes as ints in every cell
    assert json.loads((out / st.MEMSET_ZOO).read_text()) == {"64": 1e-6, "4096": 3e-6}
    cells = json.loads((out / "cells.json").read_text())["cells"]
    assert all(c["memset_flags"]["count"] == 0 for c in cells)


def test_the_device_child_command_parses_back_to_its_cell():
    args = SimpleNamespace(out="o", session="s", cpu_boundary=True, config="c.json")
    for cell in (("xattention", None, 32768, 1), ("oracle", 0.5, 32768, 1), ("dense", 1.0, 4096, 2)):
        cmd = st.cell_command(args, cell, "device-cell-one")
        assert cmd[2] == "device-cell-one"
        ns = st.parser().parse_args(cmd[2:])
        assert (ns.arm, st.nominal_density(ns.density), ns.length, ns.batch) == cell
        assert st.device_cell_key({"arm": ns.arm, "nominal_density": st.nominal_density(
            ns.density), "budget": ns.length, "batch": ns.batch}) == cell


def test_main_runs_a_device_cell_behind_the_boundary(tmp_path, monkeypatch):
    import contextlib
    from attnbench import cpu_boundary
    monkeypatch.setattr(cpu_boundary, "installed", contextlib.nullcontext)
    seen = []
    monkeypatch.setattr(st, "run_device_cell", lambda args, cfg, cell: seen.append(
        (args.step, cell, bool(cfg.get("cpu_boundary")))))
    monkeypatch.setattr(st, "run_cell", lambda *a: pytest.fail("the host child ran"))
    rc = st.main(["device-cell-one", "--out", str(tmp_path), "--session", str(tmp_path),
                  "--arm", "oracle", "--density", "0.5", "--length", "2048", "--batch", "1",
                  "--cpu-boundary"])
    assert rc == 0 and seen == [("device-cell-one", ("oracle", 0.5, 2048, 1), True)]
