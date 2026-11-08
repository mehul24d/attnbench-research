"""The timed ground-truth path's hash (`scripts/timed_path.py`): the
training rule's test of which sessions pool targets (Appendix A, 8 Nov 2026)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import timed_path as TP  # noqa: E402

T32AB = "3f50aab"           # T3.2a and T3.2b's frozen commit


def _has(commit: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO,
                          capture_output=True).returncode == 0


def test_the_path_is_the_timers_closure_and_the_runners_ground_truth_functions():
    h = TP.path_hash("HEAD")
    files = {k[5:] for k in h["items"] if k.startswith("file ")}
    funcs = {k.split("::")[1] for k in h["items"] if "::" in k}
    assert {"attnbench/ground_truth.py", "attnbench/densities.py",
            "attnbench/components/shakedown_arms.py"} <= files
    assert not any(f.startswith("attnbench/analysis/") for f in files)
    assert {"cmd_ground_truth_one", "_real_cells", "group_arms", "_sampler"} <= funcs
    assert "step_delay_injection" not in funcs and "build_prediction" not in funcs
    assert TP.compare("HEAD", "HEAD")["identical"]


@pytest.mark.skipif(not _has(T32AB), reason="needs the history back to T3.2a's commit")
def test_t32a_and_b_do_not_share_the_revised_identitys_timed_path():
    c = TP.compare(T32AB, "HEAD")
    assert not c["identical"]
    assert f"{TP.RUNNER}::_real_cells" in c["differing"]        # the traced pass
