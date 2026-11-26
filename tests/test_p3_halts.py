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
after it every `shutdown` is a no-op (the machine is off).

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
           "ATTNBENCH_P3_WATCH_S": "0.3",
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
