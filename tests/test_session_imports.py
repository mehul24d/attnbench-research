"""The import preflight (`attnbench.session_imports`): the third-party modules a
session's scripts can import are found by a static scan, each is imported in a
fresh interpreter, and a missing one is a preflight problem that names the
package. The 10 Nov L4 rehearsal ran ten steps before the eleventh found
lightgbm missing; this is the check that would have refused at step 2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t61_session as S  # noqa: E402

from attnbench import session_imports as SI  # noqa: E402

NO_SITE = [sys.executable, "-S"]         # stdlib only: nothing installed unless on PYTHONPATH


def tree(tmp_path, files):
    for rel, text in files.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\nkernels = [\n  "flash-attn>=2.6",\n]\nlearned = [\n  "lightgbm==4.6.0",\n]\n')
    return tmp_path


def test_the_scan_follows_first_party_imports_and_skips_the_standard_library(tmp_path):
    root = tree(tmp_path, {
        "attnbench/__init__.py": "",
        "attnbench/a.py": "import json\nimport pandas\nfrom . import b\n",
        "attnbench/b.py": "def f():\n    import scipy.stats\n",
        "scripts/run.py": "import os\nfrom attnbench import a\nimport helper\n",
        "scripts/helper.py": "import yaml\n",
    })
    got = SI.scan(["run.py"], root=root)
    assert set(got["required"]) == {"pandas", "scipy", "yaml"}
    assert got["required"]["scipy"] == ["attnbench/b.py"]          # a lazy import inside a function counts


def test_a_try_that_re_raises_is_required_and_one_that_carries_on_is_optional(tmp_path):
    root = tree(tmp_path, {
        "attnbench/__init__.py": "",
        "scripts/run.py": (
            "try:\n    import lightgbm\nexcept ImportError as e:\n    raise RuntimeError('install it') from e\n"
            "try:\n    import nvidia\nexcept ImportError:\n    nvidia = None\n"
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy\n"),
    })
    got = SI.scan(["run.py"], root=root)
    assert set(got["required"]) == {"lightgbm"} and set(got["optional"]) == {"nvidia"}
    assert "numpy" not in got["required"] and "numpy" not in got["optional"]


def test_a_script_named_in_a_call_is_followed_but_one_named_in_a_sentence_is_not(tmp_path):
    root = tree(tmp_path, {
        "attnbench/__init__.py": "",
        "scripts/run.py": 'run(py("probe.py", "x"))\nmsg = "build it with scripts/fetch.py or by hand"\n',
        "scripts/probe.py": "import torch\n",
        "scripts/fetch.py": "import bs4\n",
    })
    assert set(SI.scan(["run.py"], root=root)["required"]) == {"torch"}


def test_the_shell_chains_scripts_are_scanned_too(tmp_path):
    root = tree(tmp_path, {
        "attnbench/__init__.py": "",
        "scripts/run.py": "import os\n",
        "scripts/gate.py": "import wonderwords\n",
        "scripts/chain.sh": "python3 scripts/run.py run\npython3 scripts/gate.py\n",
    })
    got = SI.scan(["run.py"], shell_files=[root / "scripts" / "chain.sh"], root=root)
    assert set(got["required"]) == {"wonderwords"}


def test_the_kernels_extra_is_reported_and_not_required(tmp_path):
    root = tree(tmp_path, {"attnbench/__init__.py": "", "scripts/run.py": "import flash_attn\nimport pandas\n"})
    assert SI.is_kernel("flash_attn", root) and not SI.is_kernel("pandas", root)
    assert SI.is_kernel("flash_attn_interface", root)               # the FA3 wheel's module
    problems, found = SI.session_import_problems(["run.py"], python=NO_SITE, root=root)
    assert found["kernels"] == ["flash_attn"] and any("flash-attn" in p for p in found["kernel_problems"])
    assert problems == [f"pandas is missing or broken (import pandas: ModuleNotFoundError: No module named 'pandas')"]


def stub_site(tmp_path, modules):
    site = tmp_path / "scratch_site"
    for m in modules:
        (site / m).mkdir(parents=True, exist_ok=True)
        (site / m / "__init__.py").write_text("")
    return site


def test_mutation_removing_a_package_from_a_scratch_environment_makes_the_preflight_refuse(tmp_path, monkeypatch):
    """The real scan of the real P2 session, against a scratch environment that
    holds a stub for each module the scan found. Complete, it passes; with
    lightgbm removed it refuses, names that package and no other."""
    found = SI.scan([S.T61.script])
    need = [m for m in found["required"] if not SI.is_kernel(m)]
    assert "lightgbm" in need and "torch" in need and "yaml" in need
    site = stub_site(tmp_path, need)
    monkeypatch.setenv("PYTHONPATH", str(site))
    assert SI.import_problems(need, python=NO_SITE) == []
    import shutil
    shutil.rmtree(site / "lightgbm")
    problems = SI.import_problems(need, python=NO_SITE)
    assert len(problems) == 1 and problems[0].startswith("lightgbm is missing or broken")
    shutil.rmtree(site / "yaml")
    problems = SI.import_problems(need, python=NO_SITE)
    assert [p.split(" ")[0] for p in problems] == ["lightgbm", "pyyaml"]      # the distribution, not the module


def test_a_package_that_is_present_but_fails_to_import_is_a_problem_too(tmp_path, monkeypatch):
    site = tmp_path / "s"
    (site / "broken").mkdir(parents=True)
    (site / "broken" / "__init__.py").write_text("raise OSError('libomp.so.5: cannot open shared object file')\n")
    monkeypatch.setenv("PYTHONPATH", str(site))
    problems = SI.import_problems(["broken"], python=NO_SITE)
    assert len(problems) == 1 and "libomp" in problems[0]


def test_the_real_preflight_refuses_on_a_missing_package_and_names_it(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from attnbench import provenance
    from test_t61_session import GOOD_ROW, args_for
    monkeypatch.setattr(S, "run", lambda *a, **k: 0)
    monkeypatch.setattr(S, "_host_speed_row", lambda out, tag: GOOD_ROW)
    monkeypatch.setattr(provenance, "stamp_integrity_problems", lambda row: [])
    monkeypatch.setattr(S, "check_provenance_row", lambda cfg, row, p: list(p))
    monkeypatch.setattr(S, "run_gpu_variants", lambda out: ({}, []))
    monkeypatch.setattr(S, "lightgbm_problem", lambda: None)
    monkeypatch.setattr(S, "import_check", lambda cfg: (
        ["lightgbm is missing or broken (import lightgbm: ModuleNotFoundError: No module named 'lightgbm')"],
        {"required": {"lightgbm": ["attnbench/predict/learned.py"]}, "kernel_problems": []}))
    real_run = subprocess.run
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "3.4.0\n", ""))
    try:
        with pytest.raises(S.StepFailed) as e:
            S.real_preflight(S.T61, args_for(tmp_path, tmp_path, dry_run=False))
    finally:
        monkeypatch.setattr(S.subprocess, "run", real_run)
    assert any(p.startswith("lightgbm is missing or broken") for p in e.value.findings["problems"])


def test_the_session_scan_names_the_modules_each_session_needs_and_keeps_the_pinned_check():
    """P2's and the rehearsal's scans agree on what is third-party, and the
    lightgbm version check still runs beside the import check."""
    for cfg, chains in ((S.T61, []), ):
        got = SI.scan([cfg.script], shell_files=chains)
        assert {"torch", "pandas", "numpy", "lightgbm", "yaml", "pynvml"} <= set(got["required"])
        assert not {"attnbench_xattn_noassert", "xattn"} & set(got["required"])
    src = (ROOT / "scripts" / "t61_session.py").read_text()
    body = src.split("def real_preflight", 1)[1].split("\ndef ", 1)[0]
    assert "import_check(cfg)" in body and "lightgbm_problem()" in body
