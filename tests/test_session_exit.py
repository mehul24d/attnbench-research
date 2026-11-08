"""The session's last act on any exit is the sync, then the halt
(`scripts/session_exit.sh`; researcher's item 4, 8 Nov 2026).

A CPU dry run of the P2 runner (`t61_session.py run --dry-run`) under the
wrapper, through the real `run_phase.sh --sync-only`: the bucket is a
directory behind a fake `gsutil`, and `sudo` and `logger` are fakes that log
their arguments, so "halted" means `sudo -n shutdown -h now` was called. It
passes, is refused, is killed, is hung up on, and loses its bucket; every
time the logs and manifests reach the bucket first (when it is there), and
the VM is halted last.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WRAP = REPO / "scripts" / "session_exit.sh"
T61 = REPO / "scripts" / "t61_session.py"
T31 = REPO / "scripts" / "t31_session.py"

FAKE_GSUTIL = r"""#!/usr/bin/env bash
# gs://fake-bucket/<path> is $FAKE_GCS/<path>
printf '%s\n' "$*" >> "$GSUTIL_LOG"
[[ -n "${GSUTIL_FAIL:-}" ]] && exit 1
args=(); for a in "$@"; do case "$a" in -q|-m|-r) ;; *) args+=("$a");; esac; done
map() { echo "${1/gs:\/\/fake-bucket/$FAKE_GCS}"; }
case "${args[0]}" in
  rsync) src="${args[-2]}"; dst="$(map "${args[-1]}")"; mkdir -p "$dst"; cp -R "$src"/. "$dst"/ ;;
  ls) find "$(map "${args[-1]}")" -type f ;;
  cp) dst="$(map "${args[-1]}")"; mkdir -p "$dst"; cp "${args[@]:1:${#args[@]}-2}" "$dst"/ ;;
esac
"""
FAKE_LOG = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$HALT_LOG"
"""


def env_for(tmp_path, **extra):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("gsutil", FAKE_GSUTIL), ("sudo", FAKE_LOG), ("logger", FAKE_LOG)):
        (bin_dir / name).write_text(body.replace("args[-2]", "args[${#args[@]}-2]")
                                    .replace("args[-1]", "args[${#args[@]}-1]"))
        (bin_dir / name).chmod(0o755)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "bootstrap_xattention.log").write_text("bootstrap ok\n")
    work = tmp_path / "work"             # not a git repository: no tracked results to leave out
    (work / "results").mkdir(parents=True, exist_ok=True)
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATTNBENCH_", "GCP_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), HOME=str(home),
               GSUTIL_LOG=str(tmp_path / "gsutil.log"), HALT_LOG=str(tmp_path / "halt.log"),
               FAKE_GCS=str(tmp_path / "gcs"), ATTNBENCH_BUCKET="gs://fake-bucket",
               ATTNBENCH_SESSION="inst", ATTNBENCH_EXIT_RETRY_WAIT_S="0",
               ATTNBENCH_EXIT_STOP_GRACE_S="2", **extra)
    return env, work, home


def wrapped(*cmd):
    return ["bash", str(WRAP), "--log", "~/p2.log", "--", *cmd]


def run(tmp_path, cmd, **extra):
    env, work, home = env_for(tmp_path, **extra)
    c = [a.replace("~", str(home)) for a in wrapped(*cmd)]
    p = subprocess.run(c, cwd=work, env=env, capture_output=True, text=True, timeout=300)
    return p, tmp_path / "gcs" / "inst", home


def halted(tmp_path) -> list:
    log = tmp_path / "halt.log"
    return [l for l in log.read_text().splitlines() if "shutdown" in l] if log.exists() else []


def bucket_marker(gcs):
    return json.loads((gcs / "logs" / "session_exit.json").read_text())


def dry_session(name, tmp_path):
    return [sys.executable, str(T61), "run", "--dry-run", "--session",
            str(tmp_path / "work" / "results" / name)]


def test_a_passing_dry_run_is_synced_then_halted(tmp_path):
    p, gcs, home = run(tmp_path, dry_session("t61_dry", tmp_path))
    assert p.returncode == 0, p.stderr
    assert halted(tmp_path) == ["-n shutdown -h now"]
    m = json.loads((gcs / "results" / "t61_dry" / "manifest.json").read_text())
    assert m["attempts"] and all(a["status"] in ("passed", "skipped") for a in m["attempts"])
    assert (gcs / "results" / "t61_dry" / "11_late_check").exists() or \
        any((gcs / "results" / "t61_dry").glob("*_late_check"))
    log = (gcs / "logs" / "home" / "p2.log").read_text()
    assert "SESSION_EXIT=0 " in log and "synced on try 1" in log
    assert (gcs / "logs" / "home" / "bootstrap_xattention.log").exists()
    assert bucket_marker(gcs)["synced"] is True and bucket_marker(gcs)["rc"] == 0
    # the halt comes after the last line the log got before it
    assert (home / "p2.log").read_text().rstrip().splitlines()[-1].startswith("--- session_exit: halting")


def test_a_refused_session_is_synced_and_halted_with_its_exit_code(tmp_path):
    p, gcs, _ = run(tmp_path, dry_session("t61_new", tmp_path) + ["--from", "late_check"])
    assert p.returncode == 2
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert bucket_marker(gcs)["rc"] == 2 and bucket_marker(gcs)["synced"] is True
    assert "REFUSING" in (gcs / "logs" / "home" / "p2.log").read_text()


def test_a_crash_after_real_work_is_synced_and_halted(tmp_path):
    crash = ["bash", "-c", " ".join(dry_session("t61_crash", tmp_path)) + " >/dev/null 2>&1; kill -KILL $$"]
    p, gcs, _ = run(tmp_path, crash)
    assert p.returncode == 137
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert (gcs / "results" / "t61_crash" / "manifest.json").exists()
    assert bucket_marker(gcs)["rc"] == 137
    assert "SESSION_EXIT=137 " in (gcs / "logs" / "home" / "p2.log").read_text()


def test_a_hang_up_stops_the_session_then_syncs_and_halts(tmp_path):
    env, work, home = env_for(tmp_path)
    pidfile = tmp_path / "child.pid"
    cmd = [a.replace("~", str(home)) for a in
           wrapped("bash", "-c", f"echo $$ > {pidfile}; exec sleep 120")]
    w = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    child = int(pidfile.read_text())
    w.send_signal(signal.SIGHUP)
    assert w.wait(timeout=60) == 129
    try:
        os.kill(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive
    gcs = tmp_path / "gcs" / "inst"
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert bucket_marker(gcs)["rc"] == 129


def test_with_no_bucket_it_retries_and_still_halts(tmp_path):
    p, gcs, home = run(tmp_path, dry_session("t61_nobucket", tmp_path), GSUTIL_FAIL="1",
                       ATTNBENCH_EXIT_SYNC_TRIES="2")
    assert p.returncode == 0
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert not gcs.exists()
    log = (home / "p2.log").read_text()
    assert "SYNC FAILED on try 1 of 2" in log and "SYNC FAILED on try 2 of 2" in log
    local = json.loads((tmp_path / "work" / "logs" / "session_exit.json").read_text())
    assert local["synced"] is False and local["sync_try"] == 2
    serial = (tmp_path / "halt.log").read_text()
    assert "synced=0; halting" in serial            # the outcome reaches the serial console


def test_the_runner_under_the_wrapper_sees_it_armed(tmp_path):
    env, work, home = env_for(tmp_path)
    out = tmp_path / "env.txt"
    cmd = [a.replace("~", str(home)) for a in
           wrapped("bash", "-c", f'echo "$ATTNBENCH_SESSION_EXIT" > {out}')]
    assert subprocess.run(cmd, cwd=work, env=env, timeout=60).returncode == 0
    assert out.read_text().strip() == "armed"


# --- the P1 runner (T3.2c; ruling B4 of 8 Nov 2026) -------------------------------------

def t31_dry(name, tmp_path):
    return [sys.executable, str(T31), "--dry-run", "--session",
            str(tmp_path / "work" / "results" / name)]


def test_a_p1_dry_run_is_synced_then_halted(tmp_path):
    p, gcs, home = run(tmp_path, t31_dry("t31_dry", tmp_path))
    assert p.returncode == 0, p.stderr
    assert halted(tmp_path) == ["-n shutdown -h now"]
    m = json.loads((gcs / "results" / "t31_dry" / "manifest.json").read_text())
    last = {a["step"]: a["status"] for a in m["attempts"]}
    assert len(last) == 11 and set(last.values()) == {"passed"}      # steps 1 to 11
    assert bucket_marker(gcs)["rc"] == 0 and bucket_marker(gcs)["synced"] is True
    log = (gcs / "logs" / "home" / "p2.log").read_text()
    assert "SESSION_EXIT=0 " in log and "synced on try 1" in log
    assert (home / "p2.log").read_text().rstrip().splitlines()[-1].startswith(
        "--- session_exit: halting")


def test_a_p1_session_that_fails_after_real_work_is_synced_and_halted(tmp_path):
    # the simulated failure: the dry run's steps all run, then the session dies
    # of an exception, as a step's uncaught error would end it
    fail = ["bash", "-c", " ".join(t31_dry("t31_fail", tmp_path)) +
            " && python3 -c 'raise RuntimeError(\"simulated failure\")'"]
    p, gcs, _ = run(tmp_path, fail)
    assert p.returncode == 1
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert (gcs / "results" / "t31_fail" / "manifest.json").exists()   # the work reached the bucket
    assert bucket_marker(gcs)["rc"] == 1 and bucket_marker(gcs)["synced"] is True
    log = (gcs / "logs" / "home" / "p2.log").read_text()
    assert "simulated failure" in log and "SESSION_EXIT=1 " in log


def test_a_p1_session_killed_mid_step_is_synced_and_halted(tmp_path):
    crash = ["bash", "-c", " ".join(t31_dry("t31_crash", tmp_path)) + " >/dev/null 2>&1; kill -KILL $$"]
    p, gcs, _ = run(tmp_path, crash)
    assert p.returncode == 137
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert bucket_marker(gcs)["rc"] == 137


def test_a_refused_p1_repeat_is_synced_and_halted(tmp_path):
    # a T3.2c start in a work tree that is not a clean clone is refused, and still halts
    p, gcs, _ = run(tmp_path, [sys.executable, str(T31), "--repeat", "c", "--groups", "eager",
                               "--session", str(tmp_path / "work" / "results" / "t32c")])
    assert p.returncode == 2
    assert halted(tmp_path) == ["-n shutdown -h now"]
    assert bucket_marker(gcs)["rc"] == 2 and "REFUSING" in (gcs / "logs" / "home" / "p2.log").read_text()
