"""The L4 rehearsal's checks and its instance script, on the CPU
(docs/launch_cards/aws_rehearsal_g6.md; adviser's task of 23 Nov 2026, second)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aws_rehearsal_checks as C  # noqa: E402

SMI = """+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.9     |
"""
GOOD = {"gpus": ["NVIDIA L4"], "driver_cuda": [12, 9], "torch_cuda_available": True,
        "capability": [8, 9], "bf16_matmul_finite": True,
        "imports": {"flash_attn": "2.8.3.post1", "block_sparse_attn": "ok",
                    "xattn.src.Xattention": "ok"},
        "fa2_elf_archs": ["sm_80", "sm_90"]}


def test_the_driver_cuda_version_is_read_from_the_banner():
    assert C.driver_cuda(SMI) == (12, 9)
    assert C.driver_cuda("no banner") is None


def test_the_cubin_architectures_are_listed_once_each():
    lst = ("ELF file    1: flash_attn_2_cuda.1.sm_80.cubin\n"
           "ELF file    2: flash_attn_2_cuda.2.sm_90.cubin\n"
           "ELF file    3: flash_attn_2_cuda.3.sm_80.cubin\n")
    assert C.elf_archs(lst) == ["sm_80", "sm_90"]


def test_a_good_gpu_record_passes():
    assert C.gpu_problems(GOOD) == []


@pytest.mark.parametrize("change,said", [
    ({"gpus": ["NVIDIA L4", "NVIDIA L4"]}, "not exactly one"),
    ({"gpus": ["NVIDIA A10G"]}, "not exactly one"),
    ({"driver_cuda": [12, 8]}, "below"),
    ({"driver_cuda": None}, "below"),
    ({"torch_cuda_available": False}, "is_available"),
    ({"capability": [8, 6]}, "compute capability"),
    ({"bf16_matmul_finite": False}, "bf16"),
    ({"imports": {**GOOD["imports"], "flash_attn": "2.7.4"}}, "flash_attn imports"),
    ({"imports": {**GOOD["imports"], "block_sparse_attn": "FAILED ImportError"}}, "block_sparse_attn"),
    ({"imports": {**GOOD["imports"], "xattn.src.Xattention": "FAILED"}}, "xattn"),
    ({"fa2_elf_archs": ["sm_89"]}, "no proof"),
    ({"fa2_elf_archs": None}, "no proof"),
])
def test_each_failed_gpu_condition_is_named(change, said):
    p = C.gpu_problems({**GOOD, **change})
    assert len(p) == 1 and said in p[0], p


def _fa2(statuses):
    return {"cells": {f"L4_{L}_1": {"results": {"fa2": {"status": s}, "sdpa_flash": {"status": "correct"}}}
                      for L, s in zip(C.FA2_LENGTHS, statuses)}}


def test_fa2_passes_only_when_correct_at_both_lengths():
    assert C.fa2_problems(_fa2(["correct", "correct"])) == []
    assert len(C.fa2_problems(_fa2(["correct", "incorrect"]))) == 1
    assert len(C.fa2_problems(_fa2(["declined", "incorrect"]))) == 2
    assert len(C.fa2_problems({"cells": {}})) == 2


def test_the_rehearsal_config_is_p1s_values_on_one_dense_cell():
    import t31_steps as S
    cfg = S.load_config(str(ROOT / "configs" / "aws" / "rehearsal_g6.json"), dry_run=False)
    assert S.cells(cfg) == [("dense", 1.0, 8192, 1)]
    p1 = S.load_config(str(S.DEFAULT_CONFIG), dry_run=False)
    for k in ("reps", "dense_rounds", "warmup", "chain_check_tolerance",
              "residual_spread_tolerance", "pricing_mismatch_share", "card", "model"):
        assert cfg[k] == p1[k], k


# --- the instance script, with fake checks and steps ----------------------

def _tree(tmp_path, fail_at=None):
    """A clean git tree holding the script and fakes of its two Python entry points."""
    t = tmp_path / "repo"
    (t / "scripts").mkdir(parents=True)
    (t / "scripts" / "aws_rehearsal_instance.sh").write_text(
        (ROOT / "scripts" / "aws_rehearsal_instance.sh").read_text())
    fake = ("import sys\nname = sys.argv[1]\nopen('calls.txt', 'a').write(name + '\\n')\n"
            f"sys.exit(1 if name == {fail_at!r} else 0)\n")
    for f in ("aws_rehearsal_checks.py", "t31_steps.py"):
        (t / "scripts" / f).write_text(fake)
    (t / ".gitignore").write_text("calls.txt\nout/\n")
    for c in (["init", "-q"], ["add", "-A"],
              ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "t"]):
        subprocess.run(["git", *c], cwd=t, check=True)
    return t


def _run(t, tmp_path):
    py = tmp_path / "py"
    py.write_text(f'#!/bin/bash\nif [[ "$1" == -c ]]; then echo 3.10.12; exit 0; fi\nexec {sys.executable} "$@"\n')
    py.chmod(0o755)
    return subprocess.run(["bash", "scripts/aws_rehearsal_instance.sh", "out"], cwd=t,
                          capture_output=True, text=True,
                          env={**os.environ, "ATTNBENCH_PYTHON": str(py)})


def test_the_instance_script_runs_the_four_checks_in_order_and_times_each(tmp_path):
    t = _tree(tmp_path)
    p = _run(t, tmp_path)
    assert p.returncode == 0 and p.stdout.strip().endswith("REHEARSAL_OK"), p.stdout + p.stderr
    assert (t / "calls.txt").read_text().split() == ["gpu", "fa2", "dense_select", "host_components"]
    rows = [r.split("\t") for r in (t / "out" / "timings.tsv").read_text().splitlines()]
    assert rows[0] == ["check", "status", "seconds"]
    assert [r[:2] for r in rows[1:]] == [[c, "passed"] for c in
                                         ("gpu", "fa2", "dense_select", "host_components")]
    assert all(float(r[2]) >= 0 for r in rows[1:])


@pytest.mark.parametrize("fail_at", ["gpu", "fa2", "dense_select", "host_components"])
def test_the_first_failed_check_stops_the_rehearsal(tmp_path, fail_at):
    t = _tree(tmp_path, fail_at)
    p = _run(t, tmp_path)
    assert p.returncode == 1 and f"REHEARSAL_FAILED {fail_at}" in p.stdout
    order = ["gpu", "fa2", "dense_select", "host_components"]
    assert (t / "calls.txt").read_text().split() == order[:order.index(fail_at) + 1]
    last = (t / "out" / "timings.tsv").read_text().splitlines()[-1].split("\t")
    assert last[:2] == [fail_at, "failed"]


def test_a_dirty_tree_is_refused_before_any_check(tmp_path):
    t = _tree(tmp_path)
    (t / "scripts" / "t31_steps.py").write_text("changed\n")
    p = _run(t, tmp_path)
    assert p.returncode == 1 and "not clean" in p.stdout and not (t / "calls.txt").exists()


def test_the_fa2_check_builds_a_real_harness_and_screens_both_lengths(monkeypatch):
    """The rehearsal of 23 Nov 2026 failed here: `Harness(device="cuda")` left
    out the required `dry_run`, and nothing ran this path on the CPU."""
    import inspect
    import t31_steps as S
    from attnbench.components import shakedown_arms
    seen = {}

    class FakeHarness:
        def __init__(self, **kw):
            inspect.signature(shakedown_arms.Harness.__init__).bind(None, **kw)   # the real signature
            seen["kw"] = kw

        def unwrap(self):
            seen["unwrapped"] = True

    def fake_smoke(h, cfg):
        seen["lengths"], seen["batches"] = cfg["lengths"], cfg["batches"]
        return {"cells": {f"L4_{L}_1": {"results": {"fa2": {"status": "correct"}}} for L in cfg["lengths"]}}

    monkeypatch.setattr(shakedown_arms, "Harness", FakeHarness)
    monkeypatch.setattr(S, "dense_select_smoke", fake_smoke)
    rec = C.run_fa2(Path("."), None)
    assert seen["kw"] == {"device": "cuda", "dry_run": False} and seen["unwrapped"]
    assert seen["lengths"] == [2048, 8192] and seen["batches"] == {}
    assert C.fa2_problems(rec) == []
