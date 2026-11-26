"""The instance's halts in scripts/aws_p3_instance.sh, on this Mac, with real
processes (26 Nov 2026, after the aborted session of 25 Nov 2026).

On 25 Nov 2026 block 1's idle halt, a `sleep` started after `trap '' TERM`,
inherited the ignored TERM; block 2's `kill` did nothing, and the instance
halted itself 30 min after block 1 ended, in block 2. These tests run the real
script as the driver starts it (`nohup bash -c '...' &`, so with TERM ignored
in its exit handler, as on the instance), in a real git checkout, each block a
real process (a `sleep` standing in for t31_session.py), with a short minute
(ATTNBENCH_P3_MINUTE_S). Only the machine's power is not real: `shutdown`
below stands in for systemd's scheduler, as logind does it: one pending
shutdown (in a `scheduled` file, as /run/systemd/shutdown/scheduled), which a
new `shutdown` replaces and `shutdown -c` removes, fired by a timer that is not
our process (a new session, its signals reset). A power-off is recorded, and
after it every `shutdown` is a no-op (the machine is off). As shutdown(8) says,
5 minutes before a scheduled halt /run/nologin is written, and from then new
logins are refused; the fake writes it (ATTNBENCH_P3_NOLOGIN) and, the worst
case, never removes it when the halt is replaced.

P3_INSTANCE_SCRIPT runs them on another version of the script (the break
test: the version deployed on 25 Nov 2026 must fail them).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("P3_INSTANCE_SCRIPT", ROOT / "scripts" / "aws_p3_instance.sh"))

FAKE_SHUTDOWN = r'''#!PYTHON
import os, signal, subprocess, sys, time, uuid
F = os.environ["ATTNBENCH_P3_SHUTDOWN_SCHEDULED"]; RUN = os.environ["FAKE_RUN"]
MIN = float(os.environ["FAKE_SHUTDOWN_MINUTE_S"])
DOWN = os.path.join(RUN, "down")
def log(name, msg):
    with open(os.path.join(RUN, name), "a") as fh: fh.write(f"{time.time():.3f} {msg}\n")
a = sys.argv[1:]
if a[:1] == ["--nologin"]:
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(s, signal.SIG_DFL)
    token, usec = a[1], int(a[2])
    time.sleep(max(0.0, usec / 1e6 - time.time()))
    try: cur = open(F).read()
    except FileNotFoundError: sys.exit(0)
    if f"TOKEN={token}\n" in cur and not os.path.exists(DOWN) and os.environ.get("ATTNBENCH_P3_NOLOGIN"):
        open(os.environ["ATTNBENCH_P3_NOLOGIN"], "w").write("System is going down.\n"); log("nologin.log", f"NOLOGIN {token}")
    sys.exit(0)
if a[:1] == ["--timer"]:
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(s, signal.SIG_DFL)
    token, usec = a[1], int(a[2])
    time.sleep(max(0.0, usec / 1e6 - time.time()))
    try: cur = open(F).read()
    except FileNotFoundError: sys.exit(0)
    if f"TOKEN={token}\n" in cur and not os.path.exists(DOWN):
        open(DOWN, "w").write(token); log("poweroff.log", f"POWEROFF {token}"); os.remove(F)
    sys.exit(0)
log("shutdown_calls.log", "shutdown " + " ".join(a))
if os.path.exists(DOWN): sys.exit(0)
if "-c" in a:
    if os.path.exists(F): os.remove(F)
    sys.exit(0)
when = next(x for x in a if x == "now" or x.startswith("+"))
usec = int((time.time() + (0 if when == "now" else int(when[1:])) * MIN) * 1e6)
token = uuid.uuid4().hex[:8]
with open(F + ".tmp", "w") as fh: fh.write(f"USEC={usec}\nMODE=poweroff\nTOKEN={token}\n")
os.replace(F + ".tmp", F)
log("shutdown_calls.log", f"scheduled {token} {usec / 1e6:.3f}")
subprocess.Popen([sys.executable, __file__, "--timer", token, str(usec)], start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.Popen([sys.executable, __file__, "--nologin", token, str(int(usec - 5 * MIN * 1e6))], start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
'''
FAKE_SUDO = '#!/bin/bash\n[[ "$1" == -n ]] && shift\nexec "$@"\n'
FAKE_LOGGER = '#!/bin/bash\n:\n'
# The image's python3: the version guard, else the block's work (a real process).
FAKE_PY = r'''#!/bin/bash
if [[ "$1" == -c ]]; then echo 3.10.12; exit 0; fi
case "$*" in
  *"--from host_components"*) exec sleep "$FAKE_B2_S" ;;
  *) exec sleep "$FAKE_B1_S" ;;
esac
'''


def _setup(tmp: Path, *, halt: int, idle: int, b1: float, b2: float) -> dict:
    run, home, repo, bin_ = tmp / "run", tmp / "home", tmp / "repo", tmp / "bin"
    for d in (run, home, repo / "scripts", bin_):
        d.mkdir(parents=True)
    for name, body in (("shutdown", FAKE_SHUTDOWN.replace("PYTHON", sys.executable)),
                       ("sudo", FAKE_SUDO), ("logger", FAKE_LOGGER), ("python3", FAKE_PY)):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    shutil.copy(SCRIPT, repo / "scripts" / "aws_p3_instance.sh")
    g = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-qm", "x"], check=True)
    boot = int(time.time())
    env = {"PATH": f"{bin_}:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home), "LANG": "en_US.UTF-8",
           "FAKE_RUN": str(run), "FAKE_SHUTDOWN_MINUTE_S": "1", "FAKE_B1_S": str(b1), "FAKE_B2_S": str(b2),
           "ATTNBENCH_PYTHON": str(bin_ / "python3"), "ATTNBENCH_P3_STATE": str(home / "p3"),
           "ATTNBENCH_P3_SHUTDOWN_SCHEDULED": str(run / "scheduled"),
           "ATTNBENCH_P3_MINUTE_S": "1", "ATTNBENCH_P3_BOOT_EPOCH": str(boot),
           "ATTNBENCH_P3_HALT_MIN": str(halt), "ATTNBENCH_P3_IDLE_HALT_MIN": str(idle),
           "ATTNBENCH_P3_WATCH_S": "0.3", "ATTNBENCH_P3_NOLOGIN": str(run / "nologin"),
           "ATTNBENCH_P3_IDLE_HALT_S": str(idle)}       # the 25 Nov 2026 version's setting
    # The launcher's user data, at boot: the planned halt.
    subprocess.run([str(bin_ / "shutdown"), "-h", f"+{halt}"], env=env, check=True)
    return {"run": run, "home": home, "repo": repo, "bin": bin_, "env": env, "planned": boot + halt}


def _start(f: dict, block: str) -> None:
    """As the driver starts a block over SSH: detached, under nohup."""
    cmd = (f"nohup bash -c 'cd {f['repo']} && bash scripts/aws_p3_instance.sh {block} {f['home']}/p3_session'"
           " >/dev/null 2>&1 < /dev/null &")
    subprocess.run(["bash", "-c", cmd], env=f["env"], check=True)


def _wait(path: Path, timeout: float) -> float:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.read_text().strip():
            return time.time()
        time.sleep(0.05)
    raise AssertionError(f"{path} did not appear in {timeout} s")


def _poweroffs(f: dict) -> list[float]:
    p = f["run"] / "poweroff.log"
    return [float(ln.split()[0]) for ln in p.read_text().splitlines()] if p.exists() else []


def _run_b1_then_b2(f: dict) -> tuple[float, float, float]:
    st = f["home"] / "p3"
    _start(f, "b1")
    b1_end = _wait(st / "b1.exit", 20)
    assert (st / "b1.exit").read_text().split("\t")[0] == "0", (st / "b1.log").read_text()
    _start(f, "b2")
    b2_start = _wait(st / "b2.started", 10)
    return b1_end, b2_start, st


def test_the_idle_halt_never_fires_while_the_next_block_runs_and_fires_after_it(tmp_path):
    """Block 1 ends, its idle halt is set (4 min, a minute being 1 s); block 2
    starts at once and runs 8 s: no power-off while it runs. Block 2 ends with
    no next block: its idle halt fires 4 s later, before the planned halt."""
    f = _setup(tmp_path, halt=60, idle=4, b1=0.5, b2=8)
    b1_end, b2_start, st = _run_b1_then_b2(f)
    calls = [ln.split(" ", 1) for ln in (f["run"] / "shutdown_calls.log").read_text().splitlines()]
    idle1 = [float(t) for t, c in calls if c == "shutdown -P --no-wall +4"]
    assert idle1 and idle1[0] < b1_end, "block 1's exit file came before its idle halt was set"
    b2_end = _wait(st / "b2.exit", 30)
    during = [t for t in _poweroffs(f) if b2_start - 0.5 <= t <= b2_end]
    assert not during, f"powered off {[round(t - b1_end, 1) for t in during]} s after block 1 ended, in block 2"
    time.sleep(4 + 2)
    after = _poweroffs(f)
    assert len(after) == 1 and 3.0 <= after[0] - b2_end <= 5.5, [round(t - b2_end, 1) for t in after]
    assert after[0] < f["planned"]
    assert "shutdown -c" not in (f["run"] / "shutdown_calls.log").read_text()


def test_the_planned_halt_fires_in_a_running_block_and_nothing_cancels_or_delays_it(tmp_path):
    """The planned halt (boot + 10 min, a minute being 1 s) comes while block 2
    runs: it fires at its time (re-armed in whole minutes, so up to a minute
    early, never more than 2 s late), not at block 1's idle halt."""
    f = _setup(tmp_path, halt=10, idle=4, b1=0.5, b2=16)
    b1_end, b2_start, st = _run_b1_then_b2(f)
    time.sleep(max(0.0, f["planned"] - time.time()) + 1.5)
    offs = _poweroffs(f)
    assert offs, "the planned halt did not fire"
    # the script accepts the pending halt up to 2 s past boot + HALT_MIN (here the
    # test's boot is the second the user data's halt was set in, floored)
    assert f["planned"] - 1.5 <= offs[0] <= f["planned"] + 2.0, \
        f"first power-off at planned {offs[0] - f['planned']:+.1f} s (block 1 ended at {b1_end - f['planned']:+.1f} s)"
    assert not (st / "b2.exit").exists()          # block 2 was still running
    assert "shutdown -c" not in (f["run"] / "shutdown_calls.log").read_text()


def test_an_earlier_shutdown_set_while_a_block_runs_is_replaced_by_the_planned_halt(tmp_path):
    """The cancel file's check: a shutdown 1 min (1 s) away is set from outside
    while block 2 runs; the watchdog puts the planned halt back before it fires."""
    f = _setup(tmp_path, halt=60, idle=3, b1=0.5, b2=6)
    b1_end, b2_start, st = _run_b1_then_b2(f)
    time.sleep(1.0)
    subprocess.run([str(f["bin"] / "shutdown"), "-P", "+1"], env=f["env"], check=True)
    b2_end = _wait(st / "b2.exit", 30)
    during = [t for t in _poweroffs(f) if b2_start - 0.5 <= t <= b2_end]
    assert not during, "the outside shutdown fired while block 2 ran"
    assert "P3_HALT_WATCHDOG b2 re-armed the planned halt" in (st / "b2.log").read_text()


def _push(f: dict, delete_at: float | None = None) -> subprocess.CompletedProcess:
    """As the driver runs push-halt over SSH at a pull's start (26 Nov 2026)."""
    a = ["bash", "scripts/aws_p3_instance.sh", "push-halt"] + ([str(int(delete_at))] if delete_at else [])
    return subprocess.run(a, cwd=f["repo"], env=f["env"], capture_output=True, text=True, timeout=20)


def _pending(f: dict) -> float:
    for ln in (f["run"] / "scheduled").read_text().splitlines():
        if ln.startswith("USEC="):
            return int(ln[5:]) / 1e6
    raise AssertionError("no pending shutdown")


def _login(f: dict, what: str) -> None:
    """A new SSH login (the driver's ssh or scp): refused while /run/nologin exists."""
    assert not (f["run"] / "nologin").exists(), f"{what}: the login was refused (/run/nologin)"


def test_a_long_final_pull_is_not_cut_by_the_idle_halt_and_its_logins_stay_open(tmp_path):
    """P3's run, 26 Nov 2026: block 3's pull ended 6 min before its idle halt, and
    the final re-pull's scp, a new login inside the halt's last 5 min, was refused.
    Here (a minute being 1 s) the last block ends, its idle halt is 12 min away;
    a pull starts, packs, copies for 20 min (longer than the idle halt) with the
    halt pushed back every 4 min, then the teardown logs in. Nothing powers off
    and no login is refused until the teardown is done; then the halt fires 12 min
    after the last push, before the planned halt."""
    f = _setup(tmp_path, halt=120, idle=12, b1=0.5, b2=0.5)
    st = f["home"] / "p3"
    _start(f, "b1")
    b1_end = _wait(st / "b1.exit", 20)
    time.sleep(0.5)
    _login(f, "pull start")
    assert _push(f).returncode == 0
    time.sleep(1.0)                                    # the pack
    _login(f, "scp")
    t0 = time.time()
    while time.time() - t0 < 20:                       # the copy, with the driver's re-push
        time.sleep(4)
        _login(f, "re-push")
        assert _push(f).returncode == 0
    _login(f, "teardown")
    last = time.time(); assert _push(f).returncode == 0
    end = time.time()
    assert not [t for t in _poweroffs(f) if b1_end - 0.5 <= t <= end], "powered off during the pull"
    time.sleep(max(0.0, _pending(f) - time.time()) + 1.5)
    offs = _poweroffs(f)
    assert len(offs) == 1 and 10.0 <= offs[0] - last <= 13.5, [round(t - last, 1) for t in offs]
    assert offs[0] < f["planned"]
    assert "P3_HALT_PUSH pending" in (st / "halt.log").read_text()
    assert "shutdown -c" not in (f["run"] / "shutdown_calls.log").read_text()


def test_the_push_never_goes_past_the_planned_halt_or_the_delete(tmp_path):
    """The push is bounded by the planned halt (boot + HALT_MIN) and by the hard
    delete less ATTNBENCH_P3_DELETE_MARGIN_MIN (10)."""
    f = _setup(tmp_path, halt=22, idle=20, b1=0.5, b2=0.5)
    st = f["home"] / "p3"
    _start(f, "b1"); _wait(st / "b1.exit", 20)
    time.sleep(2.0)
    assert _push(f).returncode == 0
    assert f["planned"] - 1.5 <= _pending(f) <= f["planned"] + 2.0, _pending(f) - f["planned"]
    g = _setup(tmp_path / "d", halt=600, idle=20, b1=0.5, b2=0.5)
    _start(g, "b1"); b1_end = _wait(g["home"] / "p3" / "b1.exit", 20)
    idle_at = _pending(g)
    time.sleep(2.0)
    delete_at = b1_end + 21 + 10          # bound: the delete less 10 min, 1 min short of now + 20
    assert _push(g, delete_at).returncode == 0
    p = _pending(g)
    assert idle_at - 0.5 <= p <= delete_at - 10 + 0.5, (p - b1_end, idle_at - b1_end)
    # never earlier than it was: a bound before the pending halt leaves it
    assert _push(g, b1_end + 5 + 10).returncode == 0 and abs(_pending(g) - p) < 0.01


def test_the_push_changes_nothing_while_a_block_runs(tmp_path):
    """While a block runs its watchdog keeps the planned halt; push-halt leaves it."""
    f = _setup(tmp_path, halt=60, idle=4, b1=0.5, b2=6)
    b1_end, b2_start, st = _run_b1_then_b2(f)
    time.sleep(0.5)
    before = _pending(f)
    r = _push(f)
    assert r.returncode == 0 and abs(_pending(f) - before) < 0.01
    assert "unchanged: b2 runs" in (st / "halt.log").read_text()
