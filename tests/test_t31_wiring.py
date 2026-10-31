"""T3.1's four component steps, wired 31 Oct 2026 (probe, host components,
device components, ground truth), and the working values they read.

The GPU pieces (recorder, null-kernel profile, CUDA-graph replay, the timed
harness) are faked at their module boundaries; what runs for real is the
orchestration: the cell grid from the working values, one recording and
profile per cell, the priced operations the prediction reads, the group's
arms and the dense control per cell, and the session floor. None of it has
run on a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import t31_steps as st  # noqa: E402

from attnbench.components import host_profile as hp  # noqa: E402
from attnbench.components import op_sequence as ops  # noqa: E402
from attnbench.components.op_sequence import CodePath  # noqa: E402

from dataclasses import replace  # noqa: E402

from test_components_delay_injector import SEQ as _SEQ  # noqa: E402  (a small recorded sequence)

# The shared fixture writes a copy's direction as "h2d" with no memory kinds;
# a parsed trace writes "HtoD" from "Pageable" or "Pinned", which is what
# `link_costs` reads.
SEQ = replace(_SEQ, ops=tuple(
    replace(op, direction="HtoD", src_mem="Pageable", dst_mem="Device")
    if op.kind is ops.SeqKind.COPY else op for op in _SEQ.ops))


# --- the working values ---------------------------------------------------------------

def test_the_repository_config_takes_the_working_values_and_leaves_floors_and_reruns_null():
    cfg = st.load_config(None, dry_run=False)
    wv = cfg["working_values"]
    assert wv["status"] == "working, pre-lock" and wv["source"] == "docs/lock_value_table.md"
    assert cfg["densities"]["oracle"] == [0.25, 0.5]
    assert cfg["densities"]["xattention"] == ["calibrated"]
    assert cfg["gt_groups"]["eager"][0] == "dense" and cfg["reps"] == 10
    assert cfg["resolution_floor_ms"] is None and cfg["max_rerun_rounds"] is None


def test_the_table_is_the_one_source(tmp_path):
    text = st.WORKING_TABLE.read_text().replace('"reps": 10', '"reps": 7')
    t = tmp_path / "table.md"
    t.write_text(text)
    base = json.loads(st.DEFAULT_CONFIG.read_text())
    assert st.fill_working_values(base, t)["reps"] == 7
    # a value the config sets is not replaced
    assert st.fill_working_values(dict(base, reps=3), t)["reps"] == 3


@pytest.mark.parametrize("pairing", ["P2", "P4"])
def test_the_working_values_are_refused_for_any_pairing_but_p1(pairing):
    base = dict(json.loads(st.DEFAULT_CONFIG.read_text()), pairing=pairing)
    with pytest.raises(st.StepFailed, match="P1's only"):
        st.fill_working_values(base)


def test_a_table_that_does_not_say_pre_lock_is_refused(tmp_path):
    t = tmp_path / "table.md"
    t.write_text(st.WORKING_TABLE.read_text().replace('"working, pre-lock"', '"locked"'))
    with pytest.raises(st.StepFailed, match="not 'working, pre-lock'"):
        st.working_values(t)


def test_the_grid_skips_xattention_at_batch_four_and_names_calibrated_density_none():
    cfg = st.load_config(None, dry_run=False)
    cs = st.cells(cfg)
    assert ("xattention", None, 2048, 1) in cs
    assert not [c for c in cs if c[0] in st.BATCH1_ONLY and c[3] > 1]
    assert ("meanpool", 0.25, 16384, 4) in cs
    # 5 lengths x (1 + 2 + 2 + 1 + 1) at batch 1, and (1 + 2 + 2) at 16384 batch 4
    assert len(cs) == 5 * 7 + 5


def test_groups_and_their_settings():
    cfg = st.load_config(None, dry_run=False)
    names, dens = st.group_arms(cfg, "compiled_default", 16384, 4)
    assert names == ["dense", "meanpool", "meanpool"] and dens == [None, 0.25, 0.5]
    assert st.group_setting("eager") is None and st.group_setting("compiled_capture") == "capture"
    with pytest.raises(st.StepFailed):
        st.group_setting("whatever")


# --- floors and reruns ----------------------------------------------------------------

def test_the_floor_is_the_config_s_or_the_session_s_never_invented(tmp_path):
    assert st.session_floor_ms({"resolution_floor_ms": 2.5}, tmp_path) == 2.5
    with pytest.raises(st.StepFailed, match="no session floor"):
        st.session_floor_ms({"resolution_floor_ms": None}, tmp_path)
    d = st.step_dir(tmp_path, "dense_select")
    d.mkdir(parents=True)
    (d / "session_floor.json").write_text(json.dumps({"floor_ms": 0.7}))
    assert st.session_floor_ms({"resolution_floor_ms": None}, tmp_path) == 0.7
    assert st.rerun_budget({"max_rerun_rounds": None}) == 0
    assert st.rerun_budget({"max_rerun_rounds": 2}) == 2


def test_the_session_floor_comes_from_the_dense_pre_pass(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    t = [0.0]
    steps = iter([1.0, 1.1, 0.9, 1.0] * 50)

    def clock():
        return t[0]

    def run():
        t[0] += next(steps) / 1e3

    h = SimpleNamespace(prepare=lambda name, L, batch=1: SimpleNamespace(run_prefill=run),
                        unwrap=lambda: None)
    cfg = {"lengths": [2048], "batches": {}, "reps": 4, "pairing": "P1",
           "resolution_floor_ms": None}
    rec = st.derive_session_floor(h, cfg, tmp_path, clock=clock)
    saved = json.loads((tmp_path / "session_floor.json").read_text())
    assert "working, pre-lock" in rec["status"] and rec["quantile"] == 0.95
    assert len(saved["series"]["dense L2048 b1"]) == 4 and 0 < rec["floor_ms"] < 0.2


# --- link costs --------------------------------------------------------------------------

def test_link_costs_divide_each_copy_by_its_direction_and_memory():
    row = {"bw_h2d_pinned_Bps": 1e10, "bw_h2d_pageable_Bps": 5e9,
           "bw_d2h_pinned_Bps": 8e9, "bw_d2h_pageable_Bps": 4e9}
    seq = SimpleNamespace(ops=[
        ops.SeqOp(ops.SeqKind.COPY, direction="HtoD", src_mem="Pageable", dst_mem="Device",
                  payload_bytes=5000),
        ops.SeqOp(ops.SeqKind.KERNEL),
        ops.SeqOp(ops.SeqKind.COPY, direction="DtoH", src_mem="Device", dst_mem="Pinned",
                  payload_bytes=800)])
    assert st.link_costs(seq, row) == {0: 5000 / 5e9, 2: 800 / 8e9}
    bad = SimpleNamespace(ops=[ops.SeqOp(ops.SeqKind.COPY, direction="HtoD",
                                         src_mem="Managed", dst_mem="Device", payload_bytes=1)])
    with pytest.raises(st.StepFailed, match="no link cost"):
        st.link_costs(bad, row)


# --- host and device components through the prediction ------------------------------------

CP = CodePath(arm="x", gpu="NVIDIA L4", implementation="eager",
              libraries=(("torch", "2.9.1"),), python_optimize=0)


class FakeArm:
    def __init__(self, name, L, density, batch):
        self.name, self.length, self.prompt_id = name, L - 10, f"p{L}"
        self.layers, self.code_path = [], CP
        self.backend = SimpleNamespace(last_layer_density=[0.3, 0.5])
        self.run_prefill = lambda: None


class FakeHarness:
    def __init__(self):
        self.seen = []

    def ready(self, name, L, *, density=None, batch=1, index=0):
        self.seen.append((name, density, L, batch))
        return FakeArm(name, L, density, batch)

    def unwrap(self):
        pass


def _grid():
    return {"pairing": "P1", "card": "NVIDIA L4", "model": "m", "lengths": [2048],
            "batches": {}, "arms": ["dense", "meanpool", "xattention"],
            "densities": {"dense": [1.0], "meanpool": [0.25], "xattention": ["calibrated"]}}


def _components(monkeypatch, h):
    from attnbench.components import recorder, replay
    monkeypatch.setattr(st, "_cuda", lambda: True)
    monkeypatch.setattr(st, "_harness", lambda args: h)
    sha = ops.record_sha256(ops.to_record(SEQ))

    def record(run, out, **kw):
        Path(out).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(sequence=SEQ, tape=None, masks=None)

    host = {i: 1e-6 for i, op in enumerate(SEQ.ops) if op.kind is not ops.SeqKind.SYNC}
    monkeypatch.setattr(recorder, "record_arm_prefill", record)
    monkeypatch.setattr(hp, "block_sparse_entry", lambda *a: "entry")
    monkeypatch.setattr(hp, "cuda_device", lambda *a: "device")
    monkeypatch.setattr(recorder, "load_recording",
                        lambda d: SimpleNamespace(sequence=SEQ, tape=None, masks=None))
    monkeypatch.setattr(hp, "profile", lambda *a, **k: hp.HostProfile(
        host_s=host, reps_s={i: (v,) for i, v in host.items()}, launch_floor_s=2e-6, reps=5,
        sequence_sha256=sha))
    dev = {i: 1e-5 for i, op in enumerate(SEQ.ops) if op.is_device}
    monkeypatch.setattr(replay, "capture", lambda *a, **k: "cap")
    monkeypatch.setattr(replay, "mask_agreement", lambda cap: None)
    monkeypatch.setattr(replay, "profile_replay", lambda cap, p: p)
    monkeypatch.setattr(replay, "replay_nodes", lambda p, **k: [])
    monkeypatch.setattr(replay, "match", lambda seq, nodes: SimpleNamespace(complete=True))
    monkeypatch.setattr(replay, "price_positions", lambda seq, m, o: SimpleNamespace(
        device_s=dev, counts=lambda: {"graph": len(dev)}))


def _session(tmp_path):
    session = tmp_path / "s"
    for step in ("link", "host_components", "device_components"):
        st.step_dir(session, step).mkdir(parents=True)
    (st.step_dir(session, "link") / "link").mkdir()
    pd.DataFrame([{"bw_h2d_pinned_Bps": 1e10, "bw_h2d_pageable_Bps": 5e9,
                   "bw_d2h_pinned_Bps": 8e9, "bw_d2h_pageable_Bps": 4e9}]).to_parquet(
        st.step_dir(session, "link") / "link" / "link.parquet")
    return session


def test_host_then_device_components_give_the_cells_the_prediction_reads(tmp_path, monkeypatch):
    h = FakeHarness()
    _components(monkeypatch, h)
    session, cfg = _session(tmp_path), _grid()
    args = lambda step: SimpleNamespace(dry_run=False, session=str(session),  # noqa: E731
                                        out=str(st.step_dir(session, step)))
    r = st.step_host_components(args("host_components"), cfg)
    assert r["cells"] == 3 and ("meanpool", 0.25, 2048, 1) in h.seen
    idx = json.loads((st.step_dir(session, "host_components") / "cells_index.json").read_text())
    xa = next(c for c in idx["cells"] if c["arm"] == "xattention")
    assert xa["nominal_density"] is None and xa["density"] == 0.4   # its mask density, measured
    assert xa["code_path"] == {"gpu": "NVIDIA L4", "implementation": "eager",
                               "libraries": [["torch", "2.9.1"]], "python_optimize": 0}
    d = st.step_device_components(args("device_components"), cfg)
    assert d["cells"] == 3
    dev = json.loads((st.step_dir(session, "device_components") / "cells.json").read_text())
    assert dev["fake"] is False and all(c["ops"] for c in dev["cells"])
    # the prediction's grid check reads the cell's setting, not the measured density
    (st.step_dir(session, "link") / "recursion_inputs.json").write_text(json.dumps(
        {"queue_depth": 8, "fake": False,
         "hardware": {"pinned_bandwidth": 1e10, "pageable_bandwidth": 5e9},
         "by_length": {"2048": st.FAKE_MECHANISMS}}))
    monkeypatch.setattr("attnbench.predict.commit.current_code_stamp",
                        lambda: __import__("attnbench.predict.commit", fromlist=["x"]).CodeStamp(
                            "0" * 40, False))
    rec = st.build_prediction(cfg, session, dry_run=False)
    assert rec is not None


def test_a_cell_the_replay_cannot_price_stops_the_step(tmp_path, monkeypatch):
    from attnbench.components import replay
    h = FakeHarness()
    _components(monkeypatch, h)
    session, cfg = _session(tmp_path), _grid()
    a = SimpleNamespace(dry_run=False, session=str(session),
                        out=str(st.step_dir(session, "host_components")))
    st.step_host_components(a, cfg)

    def refuse(*x):
        raise replay.ReplayError("position 3 has no cost")
    monkeypatch.setattr(replay, "price_positions", refuse)
    a.out = str(st.step_dir(session, "device_components"))
    with pytest.raises(st.StepFailed, match="position 3 has no cost"):
        st.step_device_components(a, cfg)


# --- ground truth --------------------------------------------------------------------------

def test_each_group_times_its_arms_with_the_k2_choice_as_the_control(tmp_path, monkeypatch):
    import torch
    from attnbench import dense_selection as ds
    from attnbench import ground_truth as gt
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cfg = dict(_grid(), lengths=[2048, 16384], batches={"16384": [1, 4]}, reps=3, warmup=1,
               gt_groups={"eager": ["dense", "meanpool", "xattention"]})
    session = tmp_path / "s"
    seen, checked = [], []
    choice = SimpleNamespace(chosen="sdpa_cudnn")
    monkeypatch.setattr(ds, "load_choice", lambda p: choice)
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {"sdpa_cudnn": "CUDNN"})
    monkeypatch.setattr(ds, "check_control", lambda c, name, stamp, key: checked.append(
        (name, key.context_length, key.batch)))

    def timing_arms(names, L, densities, batch, backends, compile_settings):
        seen.append((tuple(names), tuple(densities), L, batch, backends[0]))
        return "w", "ids", f"p{L}", [SimpleNamespace(name=n) for n in names], {}
    h = SimpleNamespace(timing_arms=timing_arms, _backend=lambda spec: spec.name,
                        unwrap=lambda: None)
    monkeypatch.setattr(gt, "time_arms", lambda w, ids, arms, **k: {
        "first_clocked_block_started_at": 1.0, "rows": [{} for _ in arms]})
    args = SimpleNamespace(group="eager", session=str(session))
    first = st._real_cells(h, gt, cfg, controls=None, args=args,
                           receipt=SimpleNamespace(sha256="abc"), out=tmp_path)
    assert first == [1.0, 1.0, 1.0]
    assert seen[0][:2] == (("dense", "meanpool", "xattention"), (None, 0.25, None))
    assert seen[2][0] == ("dense", "meanpool")        # no XAttention at batch 4
    assert all(s[4] == "CUDNN" for s in seen)        # the control is the K2 choice
    assert checked == [("sdpa_cudnn", 2048, 1), ("sdpa_cudnn", 16384, 1), ("sdpa_cudnn", 16384, 4)]
    row = json.loads((tmp_path / "gt_L16384_b4.json").read_text())["rows"][1]
    assert row["nominal_density"] == 0.25 and row["group"] == "eager"


def test_every_step_is_wired_on_a_gpu():
    assert st.NOT_BUILT == {}


def test_work_outside_the_layers_gets_its_own_step_and_unlabelled_work_inside_is_refused():
    from attnbench.predict.recursion import copy_op, kernel_op, sync_op
    ops_ = [kernel_op(1e-6, 1e-5, layer=None), kernel_op(1e-6, 1e-5, layer=0),
            sync_op("item", payload_bytes=4, layer=None), kernel_op(1e-6, 1e-5, layer=1),
            copy_op(1e-6, 1e-6, layer=None, payload_bytes=8)]
    got = [o.layer for o in st.label_outside_layers(ops_)]
    assert got == [-1, 0, None, 1, 2]
    with pytest.raises(st.StepFailed, match="between layers"):
        st.label_outside_layers([kernel_op(1e-6, 1e-5, layer=0), kernel_op(1e-6, 1e-5, layer=None),
                                 kernel_op(1e-6, 1e-5, layer=1)])
    with pytest.raises(st.StepFailed, match="no layers"):
        st.label_outside_layers([kernel_op(1e-6, 1e-5, layer=None)])
