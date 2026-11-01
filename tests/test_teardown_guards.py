"""gcp_teardown_session.sh, run end to end against a fake `gcloud`.

Two failures from the T4 sparse pilot (2026-10-01/02), each now refused by
the script itself rather than by the operator noticing:

  - a wrong zone made every step "fail, continuing", reported the delete as
    "not found", and left the instance RUNNING (it was in -c, not -a);
  - fetching the serial log with a plain redirect truncates the saved log
    before the fetch can fail, so a re-run over a finished session's
    directory would have emptied it.

The fake records every call, so "touched nothing" is checked, not assumed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gcp_teardown_session.sh"

FAKE = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
zone=""
for a in "$@"; do case "$a" in --zone=*) zone="${a#--zone=}";; esac; done
case "$*" in
  "config get-value project") echo "test-project"; exit 0;;
  "compute instances describe"*)
    [[ -n "$FAKE_ZONE" && "$zone" == "$FAKE_ZONE" ]] && { echo inst; exit 0; }; exit 1;;
  "compute instances list"*) [[ -n "$FAKE_ZONE" ]] && echo "$FAKE_ZONE"; exit 0;;
  "compute instances get-serial-port-output"*) exit 1;;
  "compute ssh"*"for f in ~/*.log"*)
    # FAKE_HOME_LOGS="a.log:5 b.log:7": the instance's home logs and sizes;
    # unset, SSH is dead.
    [[ -n "${FAKE_HOME_LOGS+x}" ]] || exit 1
    for e in $FAKE_HOME_LOGS; do echo "${e%%:*} ${e##*:}"; done; echo END; exit 0;;
  "compute scp inst:~/"*.log*)
    [[ -n "${FAKE_HOME_LOGS+x}" ]] || exit 1
    src="${3##*/}"; dest="$4"
    for e in $FAKE_HOME_LOGS; do
      if [[ "${e%%:*}" == "$src" ]]; then
        n="${e##*:}"; [[ "$src" == "$FAKE_SHORT" ]] && n=$((n - 1))
        head -c "$n" /dev/zero > "$dest"; exit 0
      fi
    done; exit 1;;
  *) exit 1;;
esac
"""


def _run(tmp_path, *, actual_zone, given_zone, extra=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "gcloud"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    out_dir = tmp_path / "session"
    out_dir.mkdir(exist_ok=True)
    # The real cloud CLIs are dropped from PATH, not just shadowed, so no
    # path through the script (or the cleanup check it calls) can reach them.
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items() if k != "TEARDOWN_DELETE_WITHOUT_HOME_LOGS"}
    env.update({"PATH": os.pathsep.join([str(bin_dir), *real]),
                "FAKE_LOG": str(log), "FAKE_ZONE": actual_zone,
                "ATTNBENCH_GCP_PROJECT": "test-project", "GCP_RATE_INR_HR": "108.64",
                **(extra or {})})
    # step 5 calls gcp_cleanup_check.sh next to the script; the fake gcloud
    # makes it harmless, and its exit status is not under test here.
    p = subprocess.run(["bash", str(SCRIPT), "inst", given_zone, str(out_dir)],
                       env=env, capture_output=True, text=True, timeout=60)
    return p, log.read_text().splitlines(), out_dir


def test_a_wrong_zone_is_refused_before_anything_runs(tmp_path):
    p, calls, _ = _run(tmp_path, actual_zone="asia-northeast1-c",
                       given_zone="asia-northeast1-a")
    assert p.returncode == 2
    assert "it is in asia-northeast1-c" in p.stderr
    assert not any("delete" in c or "scp" in c or "serial" in c or "ssh" in c
                   for c in calls), calls


def test_a_failed_serial_fetch_keeps_the_saved_log(tmp_path):
    out_dir = tmp_path / "session"
    out_dir.mkdir()
    (out_dir / "serial_console.log").write_text("the first run's log\n")
    p, calls, out_dir = _run(tmp_path, actual_zone="asia-northeast1-a",
                             given_zone="asia-northeast1-a")
    assert any("get-serial-port-output" in c for c in calls)
    assert (out_dir / "serial_console.log").read_text() == "the first run's log\n"
    assert not (out_dir / "serial_console.log.partial").exists()


def test_the_right_zone_proceeds_to_delete(tmp_path):
    zone = "asia-northeast1-a"
    p, calls, out = _run(tmp_path, actual_zone=zone, given_zone=zone,
                         extra={"FAKE_HOME_LOGS": "t31_session.log:5 t31_driver.log:7"})
    assert any(c.startswith("compute instances delete inst") for c in calls), calls
    assert sorted(f.name for f in (out / "home_logs").iterdir()) == [
        "listing.txt", "t31_driver.log", "t31_session.log"]


# --- step 2b: the instance's ~/*.log (ruling, 2 Nov 2026) ---------------------

def _deleted(calls):
    return any(c.startswith("compute instances delete") for c in calls)


def test_dead_ssh_refuses_the_delete(tmp_path):
    """T3.1's first session lost ~/t31_session.log and ~/t31_driver.log to
    the delete. Without them the delete is refused."""
    zone = "asia-northeast1-a"
    p, calls, out = _run(tmp_path, actual_zone=zone, given_zone=zone)
    assert p.returncode == 3
    assert "REFUSING TO DELETE" in p.stderr
    assert not _deleted(calls), calls
    # the evidence steps before it still ran
    assert any("get-serial-port-output" in c for c in calls)


def test_a_short_copy_refuses_the_delete(tmp_path):
    zone = "asia-northeast1-a"
    p, calls, _ = _run(tmp_path, actual_zone=zone, given_zone=zone,
                       extra={"FAKE_HOME_LOGS": "a.log:5 b.log:7", "FAKE_SHORT": "b.log"})
    assert p.returncode == 3 and not _deleted(calls)
    assert "FAILED: ~/b.log" in p.stdout


def test_no_logs_on_the_instance_is_a_pull(tmp_path):
    zone = "asia-northeast1-a"
    p, calls, _ = _run(tmp_path, actual_zone=zone, given_zone=zone,
                       extra={"FAKE_HOME_LOGS": ""})
    assert _deleted(calls), p.stdout + p.stderr


def test_the_override_deletes_and_says_why(tmp_path):
    zone = "asia-northeast1-a"
    p, calls, out = _run(tmp_path, actual_zone=zone, given_zone=zone,
                         extra={"TEARDOWN_DELETE_WITHOUT_HOME_LOGS": "sshd OOM-killed"})
    assert _deleted(calls)
    assert "sshd OOM-killed" in (out / "home_logs_missing.txt").read_text()


def test_no_instance_needs_no_logs(tmp_path):
    p, calls, out = _run(tmp_path, actual_zone="", given_zone="asia-northeast1-a")
    assert "REFUSING TO DELETE" not in p.stderr
    assert not (out / "home_logs_missing.txt").exists()
