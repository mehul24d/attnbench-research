"""procmatch.sh: match processes without matching the matcher.

The bug this removes has cost two sessions, in opposite directions:

  2026-09-03  `pkill -f 'pip install'` killed the invoking SSH command, whose
              own command line contained the pattern. The build died with it.
  2026-09-04  `pgrep -f run_probe.py` matched the polling command, so a probe
              that had already died to an Xid 31 MMU fault reported RUNNING
              for six more minutes.

Both were already "documented" -- the `[n]vcc` bracket trick appears twice in
the compile-session runbook -- and were still missed twice, because the rule
has to be remembered at every call site. The tests below run the script for
real, with the pattern deliberately present in the *invoking* command line,
because that is the only condition under which the bug appears.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "procmatch.sh"
pytestmark = pytest.mark.skipif(shutil.which("pgrep") is None,
                                reason="pgrep unavailable")


def run(action: str, pattern: str, *extra: str):
    return subprocess.run(["bash", str(SCRIPT), action, pattern, *extra],
                          capture_output=True, text=True)


def test_a_pattern_matching_only_the_caller_reports_not_running():
    """The 2026-09-04 failure exactly. The pattern appears in this very
    command line -- an unguarded `pgrep -f` reports RUNNING here."""
    pattern = "attnbench_unique_marker_no_such_process"
    out = run("status", pattern)
    assert out.stdout.strip() == "NOT_RUNNING", out.stdout
    assert out.returncode == 1


def _pgrep_stub(tmp_path, pids):
    """A `pgrep` on PATH that returns exactly `pids`.

    macOS `pgrep -f` cannot see an ancestor shell's command line at all --
    `ps -o args=` shows it plainly and `pgrep -f` returns nothing -- so the
    scenario that actually bit on Linux is unconstructable on the workstation.
    An integration test for it here is vacuous BY PLATFORM, which is exactly
    the "green tests can test nothing" trap.

    Stubbing pgrep tests OUR filtering rather than the platform's matcher,
    deterministically and everywhere: we hand the script a pid list containing
    an ancestor and require it to be dropped.
    """
    stub = tmp_path / "pgrep"
    stub.write_text("#!/usr/bin/env bash\n"
                    + "".join(f"echo {p}\n" for p in pids))
    stub.chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}


def test_an_ancestor_returned_by_pgrep_is_excluded(tmp_path):
    """The 2026-09-04 shape: the poller's own parent shell carries the pattern
    and pgrep dutifully returns it. It must not count as a live match."""
    parent = os.getpid()          # ancestor of the bash the script runs in
    env = _pgrep_stub(tmp_path, [parent])
    out = subprocess.run(["bash", str(SCRIPT), "status", "anything"],
                         capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "NOT_RUNNING", (
        f"an ancestor pid was reported as a live match: {out.stdout!r}")
    assert out.returncode == 1


def test_kill_never_targets_an_ancestor(tmp_path):
    """The 2026-09-03 shape: `pkill -f` killed its own invoking shell and took
    the build with it. Killing this pid would kill the test runner."""
    env = _pgrep_stub(tmp_path, [os.getpid()])
    out = subprocess.run(["bash", str(SCRIPT), "kill", "anything"],
                         capture_output=True, text=True, env=env)
    assert "NOT_RUNNING" in out.stdout, out.stdout
    assert out.returncode == 1
    assert str(os.getpid()) not in out.stdout


def test_a_non_ancestor_pid_from_pgrep_is_kept(tmp_path):
    """The other half: a filter that drops everything is equally useless."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        env = _pgrep_stub(tmp_path, [os.getpid(), proc.pid])
        out = subprocess.run(["bash", str(SCRIPT), "pids", "anything"],
                             capture_output=True, text=True, env=env)
        assert [int(p) for p in out.stdout.split()] == [proc.pid]
    finally:
        proc.kill()
        proc.wait()


def test_a_real_process_is_found():
    """The other half: a matcher that never matches is also useless."""
    marker = "attnbench_procmatch_live_marker"
    proc = subprocess.Popen([sys.executable, "-c",
                             f"import time; _={marker!r}; time.sleep(30)"])
    try:
        time.sleep(0.7)
        out = run("status", marker)
        assert out.stdout.startswith("RUNNING"), out.stdout
        assert str(proc.pid) in out.stdout
        assert out.returncode == 0
    finally:
        proc.kill()
        proc.wait()


def test_the_reported_pids_exclude_the_matcher_itself():
    marker = "attnbench_procmatch_pids_marker"
    proc = subprocess.Popen([sys.executable, "-c",
                             f"import time; _={marker!r}; time.sleep(30)"])
    try:
        time.sleep(0.7)
        out = run("pids", marker)
        pids = [int(p) for p in out.stdout.split()]
        assert pids == [proc.pid], f"expected only the target, got {pids}"
    finally:
        proc.kill()
        proc.wait()


def test_kill_refuses_when_only_the_caller_matches():
    """The 2026-09-03 failure. Killing here would have killed the caller."""
    out = run("kill", "attnbench_unique_marker_no_such_process")
    assert "NOT_RUNNING" in out.stdout
    assert out.returncode == 1


def test_kill_terminates_a_real_match():
    marker = "attnbench_procmatch_kill_marker"
    proc = subprocess.Popen([sys.executable, "-c",
                             f"import time; _={marker!r}; time.sleep(30)"])
    try:
        time.sleep(0.7)
        out = run("kill", marker, "KILL")
        assert "killing" in out.stdout, out.stdout
        proc.wait(timeout=10)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_an_unknown_action_is_refused():
    out = run("obliterate", "whatever")
    assert out.returncode == 2
