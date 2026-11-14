"""P4 boots P2's image, and its preflight refuses libraries other than P2's
(the adviser's rulings dated 14 Nov 2026, item 3).

`configs/t64/p2_libraries.json` is P2's library list, as every cell manifest of
P2's banked session records it; P4's preflight reads its own through
`shakedown_arms.libraries` and refuses any difference but FA3. The H100
launcher boots only the pinned image, `attnbench-env-v6-20260917`.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t61_session as S  # noqa: E402
import t64_session as P4  # noqa: E402
from test_launch_provisioning_tag import _creates, _flag  # noqa: E402
from test_t61_session import GOOD_ROW, args_for  # noqa: E402

REF = json.loads((ROOT / P4.T64.library_reference).read_text())
P2_LIBS = [list(x) for x in REF["libraries"]]


def _with(name, version):
    return [[n, version if n == name else v] for n, v in P2_LIBS]


def test_the_reference_is_p2s_banked_session_on_v6():
    assert P4.T64.library_reference == "configs/t64/p2_libraries.json"
    assert REF["pairing"] == "P2" and REF["session"] == "t61_p2_20261114T0022Z"
    assert REF["boot_image"] == "attnbench-env-v6-20260917"
    assert ("configs/t64/p2_libraries.json", None) in P4.T64.files
    assert S.T61.library_reference == ""                    # P2 has no reference


def test_p2s_libraries_are_no_problem_and_fa3_is_the_only_allowed_difference():
    assert S.library_problems(P4.T64, P2_LIBS) == []
    assert S.library_problems(P4.T64, P2_LIBS + [["flash_attn_3", "3.0.0b1"]]) == []


@pytest.mark.parametrize("got,why", [
    (_with("triton", "3.5.0"), "triton is 3.5.0 here and 3.5.1 on P2"),
    (_with("flash_attn", "2.8.3"), "flash_attn is 2.8.3 here"),           # FA2 is not FA3
    (_with("bsa_row_mask_fix", "none"), "bsa_row_mask_fix is none here"),
    ([x for x in P2_LIBS if x[0] != "xattn"], "is P2's and is not here"),
    (P2_LIBS + [["flashinfer", "0.2"]], "flashinfer 0.2 is here and not P2's"),
])
def test_any_other_difference_is_a_problem(got, why):
    problems = S.library_problems(P4.T64, got)
    assert len(problems) == 1 and why in problems[0], problems


def test_a_list_naming_a_library_twice_is_refused():
    with pytest.raises(S.StepFailed, match="twice"):
        S.library_problems(P4.T64, P2_LIBS + [["torch", "2.9.1+cu129"]])


def _stub_preflight(monkeypatch, libs):
    from attnbench import provenance
    monkeypatch.setattr(S, "run", lambda *a, **k: 0)
    monkeypatch.setattr(S, "_host_speed_row", lambda out, tag: GOOD_ROW)
    monkeypatch.setattr(provenance, "stamp_integrity_problems", lambda row: [])
    monkeypatch.setattr(S, "check_provenance_row", lambda cfg, row, p: list(p))
    monkeypatch.setattr(S, "run_gpu_variants", lambda out: ({}, []))
    monkeypatch.setattr(S, "lightgbm_problem", lambda: None)
    monkeypatch.setattr(S, "import_check", lambda cfg: ([], {"required": {}, "kernel_problems": []}))
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "3.5.1\n", ""))
    monkeypatch.setattr(S, "run_kernel_contract", lambda cfg, out: ({"status": "passed"}, []))
    monkeypatch.setattr(S, "run_preflight_smoke", lambda cfg, args, out: ({}, []))
    monkeypatch.setattr(S, "session_libraries", libs)


def test_p4s_real_preflight_refuses_other_libraries_and_passes_p2s(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg = dataclasses.replace(P4.T64, preflight_findings=None)
    _stub_preflight(monkeypatch, lambda: _with("torch", "2.9.0+cu129"))
    with pytest.raises(S.StepFailed) as e:
        S.real_preflight(cfg, args_for(tmp_path, tmp_path, dry_run=False))
    assert e.value.findings["problems"] == ["torch is 2.9.0+cu129 here and 2.9.1+cu129 on P2"]

    def unreadable():
        raise S.StepFailed("the libraries could not be read: x")
    _stub_preflight(monkeypatch, unreadable)
    with pytest.raises(S.StepFailed) as e:
        S.real_preflight(cfg, args_for(tmp_path, tmp_path, dry_run=False))
    assert "did not run" in e.value.findings["problems"][0]

    _stub_preflight(monkeypatch, lambda: P2_LIBS)
    got = S.real_preflight(cfg, args_for(tmp_path, tmp_path, dry_run=False))
    assert got["libraries"] == P2_LIBS and got["problems"] == []


def test_p2s_preflight_reads_no_reference(tmp_path, monkeypatch):
    pytest.importorskip("torch")

    def never():
        raise AssertionError("P2's preflight read the libraries")
    _stub_preflight(monkeypatch, never)
    got = S.real_preflight(S.T61, args_for(tmp_path, tmp_path, dry_run=False))
    assert got["libraries"] is None


def test_session_libraries_reads_the_manifests_list_in_its_own_process():
    pytest.importorskip("torch")
    got = S.session_libraries()
    assert got[0][0] == "torch" and all(len(x) == 2 for x in got)


def test_the_h100_launcher_boots_only_p2s_image(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir(), (tmp_path / "c").mkdir()
    for d, extra in (("a", {}), ("b", {"GCP_IMAGE": "attnbench-env-v6-20260917"})):
        creates = _creates("gcp_launch_h100.sh", extra, tmp_path / d)
        assert creates and all(_flag(c, "image") == ["attnbench-env-v6-20260917"]
                               for c in creates), creates
    assert _creates("gcp_launch_h100.sh", {"GCP_IMAGE": "attnbench-env-v5-20260905"},
                    tmp_path / "c") == []
