"""The preflight's smoke check (the adviser's rulings dated 13 Nov 2026, item 3).

P2 stopped in step 7 twice on code a CPU could not run: XAttention's Triton
path (12 Nov 2026), then the asserts-removed arm's own kernels module (13 Nov
2026). The one-cell Triton smoke check of `lock-2026-11-13` ran only the
shipped XAttention arm, so it passed and the session stopped 24 minutes
later. The smoke check now runs every arm of the session at 2048, batch 1,
through step 7's and step 8's whole cells, the prediction's input and the
ground truth's harness, untimed, before anything long.

The real check runs here behind the CPU boundary on P2's and P4's configs:
the same code, the real arms, the tiny model; XAttention takes its torch
path, as on the L4 (its Triton path is `tests/test_arm_module_parity.py`'s).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from attnbench import cpu_boundary as cb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t31_steps as st  # noqa: E402

CONFIGS = {"P2": "configs/t61/p2_session.json", "P4": "configs/t64/p4_session.json"}
ARMS = ("dense", "oracle", "meanpool", "xattention", "xattention_noassert")
needs_xattn = pytest.mark.skipif(not cb.have_xattn(),
                                 reason="needs the x-attention clone (ATTNBENCH_XATTN_DIR)")


# --- the session's side -----------------------------------------------------------------

@pytest.mark.parametrize("rc, written, problem", [
    (0, {"cells": {}}, None),
    (1, {"error": "ProfileMismatch: x"}, "ProfileMismatch: x"),
    (0, {"error": "StepFailed: y"}, "StepFailed: y"),
    (1, None, "no smoke.json"),
    (0, None, "no smoke.json"),
])
def test_the_smoke_check_runs_in_its_own_process_and_a_failure_is_a_problem(
        tmp_path, monkeypatch, rc, written, problem):
    import t61_session as T61
    import t64_session as P4
    for cfg, config in ((T61.T61, CONFIGS["P2"]), (P4.T64, CONFIGS["P4"])):
        cmds = []

        def run(cmd, **kw):
            cmds.append(cmd)
            if written is not None:
                (tmp_path / "smoke" / "smoke.json").write_text(json.dumps(written))
            return rc
        monkeypatch.setattr(T61, "run", run)
        (tmp_path / "smoke" / "smoke.json").unlink(missing_ok=True)
        found, problems = T61.run_preflight_smoke(cfg, types.SimpleNamespace(session="s"),
                                                  tmp_path)
        assert cmds[0][2:5] == [st.SMOKE, "--out", str(tmp_path / "smoke")]
        assert cmds[0][-1].endswith(config)
        assert (problems == []) if problem is None else (problem in problems[0])


def test_the_smoke_check_is_a_cli_step_the_cpu_boundary_runs():
    assert st.parser().parse_args([st.SMOKE, "--out", "o", "--session", "s"]).step == st.SMOKE
    assert st.parser().parse_args([st.SMOKE, "--out", "o", "--session", "s",
                                   "--cpu-boundary"]).cpu_boundary


@pytest.mark.parametrize("pairing", sorted(CONFIGS))
def test_the_smoke_cells_are_every_arm_at_2048_batch_1_dense_first(pairing):
    cfg = st.load_config(ROOT / CONFIGS[pairing], dry_run=False)
    cells = st.smoke_cells(cfg)
    assert [c[0] for c in cells] == list(ARMS)
    assert {(c[2], c[3]) for c in cells} == {(2048, 1)}
    assert all(c[1] is None for c in cells if c[0].startswith("xattention"))
    with pytest.raises(st.StepFailed, match="dense arm first"):
        st.smoke_cells({**cfg, "arms": list(reversed(cfg["arms"]))})


def test_a_planned_fa3_that_is_not_available_is_refused(monkeypatch):
    from attnbench import dense_selection as ds
    monkeypatch.setattr(st, "_boundary", lambda: False)
    monkeypatch.setattr(ds, "registry_candidates",
                        lambda card: {"sdpa_flash": object(), "fa2": None, "fa3": None})
    with pytest.raises(st.StepFailed, match="fa3 is planned on H100 and not available"):
        st.smoke_controls("H100")
    monkeypatch.setattr(ds, "registry_candidates",
                        lambda card: {"sdpa_flash": "be", "fa2": None, "fa3": "fa3-be"})
    assert st.smoke_controls("H100") == {"sdpa_flash": "be", "fa3": "fa3-be"}
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {"sdpa_flash": "be", "fa2": None})
    assert st.smoke_controls("A100") == {"sdpa_flash": "be"}


def test_an_xattention_cell_off_the_triton_path_is_refused_on_a_gpu(tmp_path, monkeypatch):
    from attnbench.components import recorder
    seen = []
    monkeypatch.setattr(st, "_need_gpu", lambda what: None)
    monkeypatch.setattr(st, "_boundary", lambda: False)
    monkeypatch.setattr(st, "_harness", lambda args: types.SimpleNamespace(unwrap=lambda: None))
    monkeypatch.setattr(st, "bare_pricer", lambda: "pricer")
    monkeypatch.setattr(st, "host_cell", lambda h, cfg, out, cell, pricer: seen.append(cell[0]) or {
        "dir": st.cell_dir(Path(), *cell).name, "host_cost_sources": {},
        "code_path": {"implementation": "eager" if cell[0] == "dense" else "xattn:torch"}})
    monkeypatch.setattr(recorder, "load_recording", lambda d: types.SimpleNamespace(
        sequence=types.SimpleNamespace(ops=[])))
    monkeypatch.setattr(st, "ready_cell", lambda h, *cell: object())
    monkeypatch.setattr(st, "_device_cell_priced", lambda *a: {
        "arm": "dense", "ops": None, "cost_sources": {},
        "pricing": {"mismatch_s": 0.0, "priced_s": 1.0, "unpriced": [1]}})
    cfg = {"arms": ["dense", "xattention"], "densities": {"dense": [1.0], "xattention": [st.CALIBRATED]},
           "pricing_mismatch_share": 0.05, "pairing": "P2", "model": "m", "card": "A100"}
    with pytest.raises(st.StepFailed, match="xattention_dcal_L2048_b1 took xattn:torch"):
        st.preflight_smoke(types.SimpleNamespace(out=str(tmp_path), session="s"), cfg)
    assert seen == ["dense", "xattention"]


# --- the real check, behind the CPU boundary ---------------------------------------------

@needs_xattn
@pytest.mark.parametrize("pairing", sorted(CONFIGS))
def test_the_smoke_check_runs_every_arm_through_every_stage_behind_the_cpu_boundary(
        tmp_path, pairing):
    out, session = tmp_path / "out", tmp_path / "s"
    session.mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "t31_steps.py"), st.SMOKE,
                        "--cpu-boundary", "--out", str(out), "--session", str(session),
                        "--config", str(ROOT / CONFIGS[pairing])],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    found = json.loads((out / "smoke.json").read_text())
    cells = found["cells"]
    assert [k.split("_d")[0] for k in cells] == list(ARMS)
    assert all(k.endswith("_L2048_b1") for k in cells)
    for name, c in cells.items():
        assert c["host_cost_sources"].get("null_pass", 0) > 0, name           # step 7
        assert sum(c["device_cost_sources"].values()) > 0, name              # step 8
        assert c["unpriced"] == 0 and c["predict_input"]["priced"] > 0, name  # the input
    assert sorted(found["pricing_gate"]) == sorted(ARMS)
    assert [c["correct"] for c in found["dense_select"]["cells"].values()] == [["sdpa_math"]]
    assert all("share" in g for g in found["pricing_gate"].values())
    gt = found["ground_truth"]
    assert gt["controls"] == ["sdpa_math"]                 # the CPU card's candidates
    assert [a.split("@")[0] for a in gt["arms"]] == list(ARMS)
    for arm in ("meanpool", "xattention", "xattention_noassert"):     # deployable: scored twice
        a = next(v for k, v in gt["arms"].items() if k.startswith(arm))
        assert all(n > 0 for n in a["score_calls"]) and len(a["score_calls"]) == 2
    assert all(v["executed_density"] is not None for v in gt["arms"].values())
    assert len(gt["traces"]) == 1 + len(ARMS) and gt["sampler"]       # control + each arm
    assert all((out / "ground_truth" / "trace_checks" / Path(t).stem).is_dir()
               for t in gt["traces"])


@needs_xattn
def test_the_harness_refuses_a_deployable_arm_whose_forward_the_probe_saw_score_nothing(
        monkeypatch):
    import tempfile
    from attnbench import ground_truth as gt
    from attnbench.components import shakedown_arms as sa
    cfg = st.load_config(ROOT / CONFIGS["P2"], dry_run=False)
    monkeypatch.setattr(gt, "probe_for", lambda log, arm: gt.ScoringProbe(
        log, score_targets=[], mask_targets=[]))
    with cb.installed():
        h = sa.Harness(device="cpu", dry_run=True, score_cache_dir=tempfile.mkdtemp())
        try:
            with pytest.raises(st.StepFailed, match=r"meanpool@0.25: the probe saw \[0, 0\]"):
                st.ground_truth_smoke(h, cfg, st.smoke_cells(cfg), Path(tempfile.mkdtemp()),
                                      st.smoke_controls(cfg["card"]))
        finally:
            h.unwrap()


@needs_xattn
def test_the_smoke_refuses_when_no_dense_candidate_passes_k2_s_screen(monkeypatch):
    import tempfile
    from attnbench import dense_selection as ds
    from attnbench import gates
    from attnbench.components import shakedown_arms as sa
    cfg = st.load_config(ROOT / CONFIGS["P2"], dry_run=False)
    with cb.installed():
        h = sa.Harness(device="cpu", dry_run=True, score_cache_dir=tempfile.mkdtemp())
        try:
            found = st.dense_select_smoke(h, cfg)
            assert found["card"] == ds.CPU_CARD
            assert {k: c["correct"] for k, c in found["cells"].items()} == {
                ds.Cell(ds.CPU_CARD, 2048, 1).key: ["sdpa_math"]}
            wrong = gates.check_for_family

            def fails(backend, cfg, **kw):
                v = wrong(backend, cfg, **kw)
                return type(v)(**{**v.__dict__, "passed": False, "detail": "broken"})
            monkeypatch.setattr(gates, "check_for_family", fails)
            monkeypatch.setattr(ds.screen, "__kwdefaults__",
                                {**ds.screen.__kwdefaults__, "check": fails})
            with pytest.raises(st.StepFailed, match="no dense candidate passed its checks"):
                st.dense_select_smoke(h, cfg)
        finally:
            h.unwrap()


def test_on_a_gpu_the_smoke_screens_every_cell_k2_screens(monkeypatch):
    from attnbench import dense_selection as ds
    seen = []
    monkeypatch.setattr(st, "_boundary", lambda: False)
    monkeypatch.setattr(st, "smoke_controls", lambda card: {"sdpa_flash": "be"})
    monkeypatch.setattr(ds, "screen", lambda c, planned, cands, **kw: (
        seen.append((c.seq_len, c.batch, tuple(planned))) or
        ({"sdpa_flash": types.SimpleNamespace(status="correct", check_kind="exact",
                                              max_abs_err=0.0, reason="")}, {"sdpa_flash": "be"})))
    for pairing in sorted(CONFIGS):
        cfg = st.load_config(ROOT / CONFIGS[pairing], dry_run=False)
        seen.clear()
        st.dense_select_smoke(_Cfg(), cfg)
        assert [(L, b) for L, b, _ in seen] == st.dense_select_cells(cfg)
        assert (16384, 4) in st.dense_select_cells(cfg)
        assert {p for *_, p in seen} == {ds.planned_candidates(cfg["card"])}


class _Cfg:
    """A harness whose dense config is a real `AttnConfig` (`replace` needs one)."""

    def __init__(self):
        from attnbench.config import AttnConfig
        self._cfg = lambda spec, L: AttnConfig(seq_len=L, batch=1, n_heads_q=12, n_heads_kv=2,
                                               head_dim=128, mask="causal")


def test_the_harness_is_smoked_only_with_the_controls_k2_s_screen_found_correct_at_2048():
    from attnbench import dense_selection as ds
    key = ds.Cell(card="A100", context_length=2048, batch=1).key
    other = ds.Cell(card="A100", context_length=16384, batch=1).key
    dense = {"card": "A100", "cells": {key: {"correct": ["sdpa_flash", "sdpa_cudnn"]},
                                       other: {"correct": ["sdpa_flash", "fa2"]}}}
    controls = {"sdpa_flash": "f", "sdpa_cudnn": "c", "sdpa_efficient": "e", "fa2": "2"}
    assert st.smoke_gt_controls(dense, controls) == {"sdpa_flash": "f", "sdpa_cudnn": "c"}


def test_the_smoke_hands_the_harness_the_screen_s_controls(tmp_path, monkeypatch):
    from attnbench.components import recorder
    got = {}
    monkeypatch.setattr(st, "_need_gpu", lambda what: None)
    monkeypatch.setattr(st, "_boundary", lambda: False)
    monkeypatch.setattr(st, "_harness", lambda args: types.SimpleNamespace(unwrap=lambda: None))
    monkeypatch.setattr(st, "bare_pricer", lambda: "pricer")
    monkeypatch.setattr(st, "host_cell", lambda h, cfg, out, cell, pricer: {
        "dir": st.cell_dir(Path(), *cell).name, "host_cost_sources": {},
        "code_path": {"implementation": "eager"}})
    monkeypatch.setattr(recorder, "load_recording", lambda d: types.SimpleNamespace(
        sequence=types.SimpleNamespace(ops=[])))
    monkeypatch.setattr(st, "ready_cell", lambda h, *cell: object())
    monkeypatch.setattr(st, "_device_cell_priced", lambda *a: {
        "arm": "dense", "ops": None, "cost_sources": {},
        "pricing": {"mismatch_s": 0.0, "priced_s": 1.0, "unpriced": []}})
    monkeypatch.setattr(st, "gate_brief", lambda g: {})
    from attnbench.components import replay
    monkeypatch.setattr(replay, "pricing_gate", lambda devs, share: None)
    from attnbench import dense_selection as ds
    key = ds.Cell(card="A100", context_length=2048, batch=1).key
    monkeypatch.setattr(st, "dense_select_smoke", lambda h, cfg: {
        "card": "A100", "cells": {key: {"correct": ["sdpa_flash"]}}})
    monkeypatch.setattr(st, "smoke_controls", lambda card: {"sdpa_flash": "f", "sdpa_efficient": "e"})
    monkeypatch.setattr(st, "ground_truth_smoke",
                        lambda h, cfg, cells, out, controls: got.update(controls) or {})
    cfg = {"arms": ["dense"], "densities": {"dense": [1.0]}, "pricing_mismatch_share": 0.05,
           "pairing": "P2", "model": "m", "card": "A100"}
    st.preflight_smoke(types.SimpleNamespace(out=str(tmp_path), session="s"), cfg)
    assert got == {"sdpa_flash": "f"}
