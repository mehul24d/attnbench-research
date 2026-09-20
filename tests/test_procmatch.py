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
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)
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
                             f"import time; _={marker!r}; time.sleep(30)"],
                            start_new_session=True)
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
                             f"import time; _={marker!r}; time.sleep(30)"],
                            start_new_session=True)
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
                             f"import time; _={marker!r}; time.sleep(30)"],
                            start_new_session=True)
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


def test_a_sibling_in_our_process_group_is_excluded(tmp_path):
    """The residual gap ancestor-walking left, observed live on 2026-09-04:
    `RUNNING pids=1338 5634`, where 5634 was a forked subshell of the polling
    command. Subshells inherit the parent's command line, so they carry the
    pattern while being nobody's ancestor. When 1338 died, that phantom held
    the answer at RUNNING for three minutes."""
    import subprocess as sp
    sibling = sp.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        env = _pgrep_stub(tmp_path, [sibling.pid])
        out = subprocess.run(["bash", str(SCRIPT), "status", "anything"],
                             capture_output=True, text=True, env=env)
        assert out.stdout.strip() == "NOT_RUNNING", (
            f"a sibling in our process group was reported live: {out.stdout!r}")
    finally:
        sibling.kill()
        sibling.wait()


def test_a_detached_process_is_still_found(tmp_path):
    """The other side, and the actual use case: work started with `nohup` over
    ssh gets its own session and must still be reported. An exclusion that
    also hid the target would be worse than the phantom it removes."""
    import subprocess as sp
    detached = sp.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                        start_new_session=True)
    try:
        env = _pgrep_stub(tmp_path, [detached.pid])
        out = subprocess.run(["bash", str(SCRIPT), "status", "anything"],
                             capture_output=True, text=True, env=env)
        assert out.stdout.startswith("RUNNING"), out.stdout
        assert str(detached.pid) in out.stdout
    finally:
        detached.kill()
        detached.wait()


def test_the_blind_spot_is_deliberate_and_stated(tmp_path):
    """Process-group exclusion has a cost: a target started by this same shell
    and NOT detached is missed. That is the right way to be wrong here -- a
    false NOT_RUNNING gets investigated, a false RUNNING gets believed -- but
    it is a real limitation and is asserted so it cannot be forgotten or
    silently "fixed" into a phantom-producing state again."""
    import subprocess as sp
    attached = sp.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        env = _pgrep_stub(tmp_path, [attached.pid])
        out = subprocess.run(["bash", str(SCRIPT), "status", "anything"],
                             capture_output=True, text=True, env=env)
        assert out.stdout.strip() == "NOT_RUNNING", (
            "documented blind spot changed: a same-process-group target is "
            "now reported. Re-check the phantom case before accepting this.")
    finally:
        attached.kill()
        attached.wait()


def test_pattern_from_a_file_matches_the_same_way(tmp_path):
    """`--file` exists to remove the problem at the root: if the pattern never
    appears in ANY command line, no ancestor, sibling, subshell, or leftover
    from a previous invocation can carry it, and there is nothing for pgrep to
    falsely match. The filters remain for callers who pass it directly."""
    marker = "attnbench_patternfile_marker"
    pf = tmp_path / "pattern.txt"
    pf.write_text(marker + "\n")
    proc = subprocess.Popen(
        [sys.executable, "-c", f"_={marker!r}; import time; time.sleep(30)"],
        start_new_session=True)
    try:
        time.sleep(0.7)
        out = subprocess.run(["bash", str(SCRIPT), "status", "--file", str(pf)],
                             capture_output=True, text=True)
        assert out.stdout.startswith("RUNNING"), out.stdout
        assert str(proc.pid) in out.stdout
    finally:
        proc.kill()
        proc.wait()


def test_an_unreadable_pattern_file_is_refused(tmp_path):
    out = subprocess.run(["bash", str(SCRIPT), "status", "--file",
                          str(tmp_path / "nope.txt")],
                         capture_output=True, text=True)
    assert out.returncode == 2


def test_an_empty_pattern_file_is_refused(tmp_path):
    """An empty pattern would match every process on the box -- and for
    `kill`, that is catastrophic rather than merely wrong."""
    pf = tmp_path / "empty.txt"
    pf.write_text("")
    out = subprocess.run(["bash", str(SCRIPT), "kill", "--file", str(pf)],
                         capture_output=True, text=True)
    assert out.returncode == 2
    assert "empty" in (out.stderr + out.stdout).lower()


def _a_pid_that_has_exited() -> int:
    """A pid that certainly existed and certainly does not now."""
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    return dead.pid


def test_a_pid_that_has_already_exited_is_not_a_live_match(tmp_path):
    """The 2026-09-20 hole, and why all three integration tests above passed
    here and failed on the instance.

    `pgrep` returns a pid -- in practice a subshell this very command forked
    for a pipeline, carrying our argv -- and by the time the filters ask `ps`
    about it, it has exited. `ps` prints nothing, and nothing satisfied BOTH
    filters: an empty pgid is not equal to ours, and an empty command line
    does not contain "procmatch.sh". So the phantom fell through and was
    reported as a genuine match. `status` said RUNNING for a pattern matching
    nothing, and `kill` signalled it.

    The integration tests could not catch this on the workstation because
    macOS `pgrep -f` cannot see an ancestor shell's command line at all, so
    the phantom is never produced here. This one is deterministic on every
    platform: hand the stub a pid that has already exited. Whatever `ps` says
    about it, it is not a live process, and the filters must fail CLOSED.
    """
    env = _pgrep_stub(tmp_path, [_a_pid_that_has_exited()])
    out = subprocess.run(["bash", str(SCRIPT), "status", "anything"],
                         capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "NOT_RUNNING", out.stdout
    assert out.returncode == 1


def test_kill_does_not_signal_a_pid_that_has_already_exited(tmp_path):
    """The same hole on the path that does damage. The pid is dead here, so
    the signal lands nowhere; on the instance the equivalent phantom was a
    subshell of the CALLER, which is the 2026-09-03 incident -- `pkill -f`
    taking the invoking SSH command with it."""
    env = _pgrep_stub(tmp_path, [_a_pid_that_has_exited()])
    out = subprocess.run(["bash", str(SCRIPT), "kill", "anything"],
                         capture_output=True, text=True, env=env)
    assert "killing" not in out.stdout, out.stdout
    assert "NOT_RUNNING" in out.stdout, out.stdout
    assert out.returncode == 1


def test_an_unreadable_own_pgid_refuses_rather_than_answering(tmp_path):
    """The sibling filter is built on our own pgid. If that cannot be read the
    filter is inert, and an inert filter here does not degrade gracefully --
    it reports this command's own subshells as live. Refusing is the only
    answer that cannot be believed wrongly."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "ps").write_text("#!/usr/bin/env bash\nexit 0\n")   # prints nothing
    (stub_dir / "ps").chmod(0o755)
    (stub_dir / "pgrep").write_text("#!/usr/bin/env bash\necho 1\n")
    (stub_dir / "pgrep").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}"}
    out = subprocess.run(["bash", str(SCRIPT), "status", "anything"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 2, (out.stdout, out.stderr)
    assert "refusing" in out.stderr, out.stderr
