"""The Mac-side watch of the FA3 build VM (scripts/gcp_watch_fa3_build.sh).

On 28 Oct 2026 the watcher then in use called `timeout`, which macOS lacks:
every check failed, nothing said so, and a failed build's VM sat idle for two
hours. These tests run the real watcher against a scripted fake `gcloud`
(instance statuses in order, an ssh that can hang, bucket listings before and
after), with a `timeout` first on PATH that fails loudly if anything calls it,
the way macOS would.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "gcp_watch_fa3_build.sh"
PIN = "pinned-proj-123"
COMMIT = "49774b31c8d1172a5b28f72e9aed0f322bf4ccba"
PREFIX = f"gs://attnbench-results-{PIN}/fa3/{COMMIT}/"

FAKE_GCLOUD = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
  "config get-value project") echo "$FAKE_ACTIVE"; exit 0 ;;
  *"instances describe"*)
    [[ -n "${FAKE_DESCRIBE_RC:-}" ]] && { echo "ERROR: describe" >&2; exit "$FAKE_DESCRIBE_RC"; }
    n=$(grep -c "instances describe" "$GCLOUD_LOG")
    s=$(echo $FAKE_STATUSES | cut -d' ' -f"$n"); [[ -z "$s" ]] && s=$(echo $FAKE_STATUSES | awk '{print $NF}')
    echo "WARNING: some gcloud notice" >&2
    echo "$s"; exit 0 ;;
  *"compute ssh"*)
    [[ -n "${FAKE_SSH_HANG:-}" ]] && sleep 30
    n=$(grep -c "compute ssh" "$GCLOUD_LOG")
    [[ ",${FAKE_SSH_FAIL_ON:-}," == *",$n,"* ]] && { echo "ssh: connect failed" >&2; exit 255; }
    echo "[3/10]"; echo "START 14:00:00Z jobs=3"; echo WATCH_OK; exit 0 ;;
  *"storage ls"*) cat "$LS"; exit 0 ;;
  *"storage cat"*"started.json"*)
    [[ -s "$STARTED_JSON" ]] || { echo "ERROR: No URLs matched" >&2; exit 1; }; cat "$STARTED_JSON"; exit 0 ;;
  *"storage cat"*"manifest.json"*) cat "$MANIFEST"; exit 0 ;;
esac
exit 0
"""


START = "2026-10-29T10:00:00Z"


def _run(tmp_path, statuses="RUNNING TERMINATED", listing="", started=START, manifest_start=None,
         limit_s="2", **env_extra):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gcloud").write_text(FAKE_GCLOUD)
    (bin_dir / "timeout").write_text(
        '#!/bin/bash\necho "timeout: command not found (macOS)" >&2\nexit 127\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    (tmp_path / "ls.txt").write_text(listing)
    (tmp_path / "started.json").write_text(
        f'{{"started": "{started}", "commit": "{COMMIT}", "tag": "v2.8.3.post2"}}\n' if started else "")
    (tmp_path / "manifest.json").write_text(
        f'{{\n  "commit": "{COMMIT}",\n  "started_utc": "{manifest_start}"\n}}\n' if manifest_start else "")
    log = tmp_path / "gcloud.log"
    log.write_text("")
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GCP_", "ATTNBENCH_", "CLOUDSDK_", "FAKE_", "FA3_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), GCLOUD_LOG=str(log),
               FAKE_ACTIVE=PIN, ATTNBENCH_GCP_PROJECT=PIN, FAKE_STATUSES=statuses,
               LS=str(tmp_path / "ls.txt"), STARTED_JSON=str(tmp_path / "started.json"),
               MANIFEST=str(tmp_path / "manifest.json"),
               FA3_WATCH_INTERVAL_S="0", FA3_WATCH_CALL_LIMIT_S=limit_s)
    env.update(env_extra)
    t0 = time.monotonic()
    p = subprocess.run(["bash", str(SCRIPT), "inst", "zone-a"], capture_output=True,
                       text=True, env=env, timeout=120)
    calls = [c for c in log.read_text().splitlines() if c != "config get-value project"]
    return p, calls, time.monotonic() - t0


OLD_FAILED = f"{PREFIX}failed-20261028T140651Z/\n{PREFIX}started.json\n"
NEW_FAILED = f"{PREFIX}failed-20261029T101500Z/\n"
WHEEL = f"{PREFIX}manifest.json\n{PREFIX}flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl\n"


def test_this_builds_manifest_is_the_success_verdict_once_the_vm_has_halted(tmp_path):
    p, calls, _ = _run(tmp_path, "PROVISIONING RUNNING RUNNING TERMINATED",
                       listing=OLD_FAILED + WHEEL, manifest_start=START)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"VERDICT: wheel stored and read back (manifest.json for the build started {START})" in p.stdout
    assert p.stdout.count("RUNNING: [3/10]") == 2
    assert PIN not in p.stdout + p.stderr


def test_a_new_failed_directory_is_the_failure_verdict_and_an_old_one_is_not(tmp_path):
    p, _, _ = _run(tmp_path, listing=OLD_FAILED + NEW_FAILED)
    assert p.returncode == 2, p.stdout + p.stderr
    verdict = p.stdout.split("VERDICT")[1]
    assert "failed-20261029T101500Z" in verdict and "failed-20261028T140651Z" not in verdict


def test_an_earlier_builds_manifest_is_not_this_builds_success(tmp_path):
    p, _, _ = _run(tmp_path, listing=OLD_FAILED + WHEEL + NEW_FAILED,
                   manifest_start="2026-10-20T08:00:00Z")
    assert p.returncode == 2 and "failed-20261029T101500Z" in p.stdout


def test_a_build_that_failed_before_the_watch_began_is_still_its_verdict(tmp_path):
    """The failure is dated against the build's own start, not the watch's."""
    p, _, _ = _run(tmp_path, "TERMINATED", listing=OLD_FAILED + NEW_FAILED)
    assert p.returncode == 2


@pytest.mark.parametrize("listing,started,match", [
    (OLD_FAILED, START, "neither a wheel nor logs from the build started"),
    (OLD_FAILED, None, "never reached its build step"),
])
def test_a_halted_vm_with_no_outcome_of_this_build_is_unknown(tmp_path, listing, started, match):
    p, _, _ = _run(tmp_path, listing=listing, started=started)
    assert p.returncode == 5 and match in p.stderr


def test_a_hanging_ssh_is_cut_off_without_timeout_and_counted(tmp_path):
    """Three hung checks in a row stop the watch with exit 3, each cut at the
    call limit (2 s here), and the `timeout` on PATH is never called."""
    p, calls, elapsed = _run(tmp_path, "RUNNING", FAKE_SSH_HANG="1")
    assert p.returncode == 3, p.stdout + p.stderr
    assert p.stderr.count("CHECK FAILED") == 3 and "status 124" in p.stderr
    assert "cannot see the VM" in p.stderr
    assert "command not found" not in p.stderr
    assert elapsed < 25, elapsed       # 3 x 2 s, not 3 x 30 s


def test_a_failing_describe_is_reported_every_time_and_stops_the_watch(tmp_path):
    p, calls, _ = _run(tmp_path, FAKE_DESCRIBE_RC="1", FA3_WATCH_MAX_FAILURES="2")
    assert p.returncode == 3
    assert re.findall(r"CHECK FAILED \((\d)/2\): instances describe", p.stderr) == ["1", "2"]


def test_a_check_that_recovers_resets_the_count(tmp_path):
    """Failures must be in a row: ssh fails on polls 1 and 3, succeeds on 2,
    so two failures never meet and the watch runs out of polls instead."""
    p, calls, _ = _run(tmp_path, "RUNNING", FA3_WATCH_MAX_POLLS="4",
                       FA3_WATCH_MAX_FAILURES="2", FAKE_SSH_FAIL_ON="1,3")
    assert p.returncode == 4 and "still running" in p.stderr, p.stderr
    assert p.stderr.count("CHECK FAILED (1/2)") == 2


def test_the_watcher_is_read_only_and_never_calls_timeout():
    text = SCRIPT.read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"(^|[\s;(|&])timeout\s", code)
    for verb in ("create", "delete", "stop", "start", "reset", "suspend"):
        assert not re.search(rf"gcloud compute instances {verb}\b", code), verb
