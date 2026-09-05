"""The guard that would have caught the ₹218.

Instance 12 of the silent-failure pattern: a green test with a billable side
effect. `test_a_single_zone_stockout_does_not_advise_retrying_the_list` ran the
real launcher against the real project and created a real GPU instance on every
run, four times, while passing -- because all three of its assertions read the
script's source text rather than checking anything the subprocess did.

These tests check the guard, not the fixed test. The fixed test is one line
away from being rewritten wrong again; the guard is what makes that survivable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "gcp_launch_compile_session.sh"


def _sh(cmd, **kw):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          timeout=60, **kw)


# ---------------------------------------------------------------------------
# The shims are in force
# ---------------------------------------------------------------------------

def test_gcloud_is_refused_inside_the_test_session():
    p = _sh("gcloud compute instances list")
    assert p.returncode == 127
    assert "BLOCKED IN TESTS" in p.stderr


@pytest.mark.parametrize("tool", ["gcloud", "gsutil", "aws", "ssh", "scp",
                                  "curl", "wget", "rsync", "kubectl"])
def test_every_network_cli_is_refused(tool):
    """Not just gcloud. A launcher script can reach production through any of
    these, and the failure mode -- a test that quietly talks to the outside
    world -- is identical whichever one it uses."""
    p = _sh(f"{tool} --version")
    assert p.returncode == 127, f"{tool} was NOT blocked"
    assert "BLOCKED IN TESTS" in p.stderr


def test_the_refusal_explains_itself_and_names_the_escape_hatch():
    """A guard whose message does not explain itself gets deleted by whoever
    hits it next. It must say how to stub deliberately, not how to remove."""
    p = _sh("gcloud compute instances create boom")
    assert "stub it" in p.stderr
    assert "do not remove" in p.stderr
    assert "boom" in p.stderr, "the refused argv must be visible"


def test_git_is_not_blocked():
    """The deploy-guard tests build real fixture repos. Blocking git would
    make the guard so inconvenient it gets switched off."""
    p = _sh("git --version")
    assert p.returncode == 0 and "git version" in p.stdout


def test_a_test_can_still_stub_a_tool_deliberately(tmp_path):
    """The shim must not defeat the fake-gcloud pattern the other tests use:
    a directory prepended to PATH wins over the session shim."""
    d = tmp_path / "bin"; d.mkdir()
    (d / "gcloud").write_text("#!/bin/bash\necho FAKE OK\n")
    (d / "gcloud").chmod(0o755)
    p = _sh("gcloud whatever", env={**os.environ, "PATH": f"{d}:{os.environ['PATH']}"})
    assert p.returncode == 0 and "FAKE OK" in p.stdout


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------

def test_the_launcher_cannot_reach_a_real_project_from_a_test(blocked_call_log):
    """The exact invocation that billed ₹218, replayed.

    Real PATH, no GCP_PROJECT, `launch` on stdin, a real zone name. Before the
    guard this resolved the live project and created an instance. It must now
    die at the first gcloud call, having created nothing.
    """
    p = subprocess.run(
        ["bash", str(LAUNCHER), "test-instance"],
        input="launch\n", capture_output=True, text=True, timeout=60,
        env={**os.environ, "GCP_ZONE_FALLBACKS": "asia-southeast1-c"})

    assert p.returncode != 0, "the launcher must not succeed from a test"
    calls = blocked_call_log.read_text()
    assert "gcloud" in calls, "the guard did not intercept -- it is not in force"
    assert "instances create" not in calls, (
        "a create reached the CLI layer; it was refused, but the script should "
        "have stopped before it")


def test_no_test_file_invokes_the_launcher_with_the_inherited_path():
    """Static backstop, so the pattern cannot come back in a NEW test.

    The runtime guard blocks the call; this catches the shape at review time.
    A test may pass `env=` with a stubbed PATH, or use `_run`, but it must not
    hand the launcher `os.environ["PATH"]` unmodified.
    """
    offenders = []
    for f in sorted((REPO / "tests").glob("test_*.py")):
        src = f.read_text()
        if "gcp_launch_compile_session" not in src and "gcp_deploy_source" not in src:
            continue
        if f.name == "test_no_test_reaches_production.py":
            continue          # this file does it on purpose, under the guard
        if '"PATH": os.environ["PATH"]' in src or "'PATH': os.environ['PATH']" in src:
            offenders.append(f.name)
    assert not offenders, (
        f"{offenders} hand the real PATH to a script that shells out to a "
        f"cloud CLI -- the 2026-09-05 shape")
