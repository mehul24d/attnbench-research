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


# --- the P2/P4 pin (adviser's ruling 1f of 11 Nov 2026; T4.3 read B, finding 7) ---

def test_the_pin_hashes_whole_files_of_the_closure_and_the_session_configs():
    h = TP.pin_hash("HEAD")
    files = {k[5:] for k in h["items"] if k.startswith("file ")}
    assert {"scripts/t31_steps.py", "scripts/t61_session.py", "scripts/t64_session.py",
            "attnbench/ground_truth.py", "attnbench/lock/checklist.py",
            "attnbench/predict/session_order.py"} <= files
    assert {f"config {c}" for c in TP.SESSION_CONFIGS} <= set(h["items"])
    assert not any("::" in k for k in h["items"])                # no function segments
    assert TP.compare("HEAD", "HEAD", TP.pin_hash)["identical"]


@pytest.mark.parametrize("path, old, new", [
    # read B's three local commits, each of which left the 8 Nov hash unmoved
    (TP.RUNNER, b'"oracle", "meanpool")', b'"oracle")'),
    (TP.RUNNER, b"class TraceExport", b"class TraceExport  "),
    ("configs/t61/p2_session.json", b'"warmup"', b'"warmup" '),
    # and a line ending, which a text read folds
    ("attnbench/ground_truth.py", b"\n", b"\r\n"),
])
def test_a_change_anywhere_in_a_pinned_file_moves_the_pin(monkeypatch, path, old, new):
    real = TP.show_bytes
    head = TP._resolve("HEAD")
    before = TP.pin_hash(head)
    assert old in real(head, path)

    def edited(commit, p):
        raw = real(commit, p)
        return raw.replace(old, new, 1) if p == path and raw is not None else raw
    monkeypatch.setattr(TP, "show_bytes", edited)
    after = TP.pin_hash(head)
    assert after["sha256"] != before["sha256"]
    moved = [k for k in after["items"] if after["items"][k] != before["items"].get(k)]
    assert moved == [("config " if path.startswith("configs/") else "file ") + path]
