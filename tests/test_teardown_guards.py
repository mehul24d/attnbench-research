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
  "compute instances describe"*)
    [[ -n "$FAKE_ZONE" && "$zone" == "$FAKE_ZONE" ]] && { echo inst; exit 0; }; exit 1;;
  "compute instances list"*) [[ -n "$FAKE_ZONE" ]] && echo "$FAKE_ZONE"; exit 0;;
  "compute instances get-serial-port-output"*) exit 1;;
  *) exit 1;;
esac
"""


def _run(tmp_path, *, actual_zone, given_zone):
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
    env = {**os.environ, "PATH": os.pathsep.join([str(bin_dir), *real]),
           "FAKE_LOG": str(log), "FAKE_ZONE": actual_zone}
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
    p, calls, _ = _run(tmp_path, actual_zone=zone, given_zone=zone)
    assert any(c.startswith("compute instances delete inst") for c in calls), calls
