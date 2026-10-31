"""A self-halted VM gets a cost file from gcloud's create and stop times.

The FA3 build VM and any in-guest halt leave the instance TERMINATED, so the
teardown can neither SSH in nor read its boot clock; before this it ended
without a `session_cost.txt` (the second FA3 build had no cost file). The span
now comes from `creationTimestamp` and `lastStopTimestamp`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gcp_teardown_session.sh"

FAKE = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
case "$*" in
  "config get-value project") echo "test-project"; exit 0;;
  *"--format=value(status,creationTimestamp,lastStopTimestamp)"*)
    printf '%s\t%s\t%s\n' "$FAKE_STATUS" "$FAKE_CREATED" "$FAKE_STOPPED"; exit 0;;
  "compute instances describe"*) echo inst; exit 0;;
  *) exit 1;;     # serial port, scp and ssh all fail: the VM is halted
esac
"""


def _run(tmp_path, *, status="TERMINATED", created="2026-10-28T03:00:00.000-07:00",
         stopped="2026-10-28T04:30:30.000-07:00", rate="297.47"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gcloud"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    out = tmp_path / "session"
    out.mkdir()
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {**os.environ, "PATH": os.pathsep.join([str(bin_dir), *real]), "FAKE_LOG": str(log),
           "FAKE_STATUS": status, "FAKE_CREATED": created, "FAKE_STOPPED": stopped,
           "ATTNBENCH_GCP_PROJECT": "test-project", "GCP_RATE_INR_HR": rate}
    p = subprocess.run(["bash", str(SCRIPT), "inst", "zone-a", str(out)],
                       env=env, capture_output=True, text=True, timeout=60)
    return p, out


def _cost(out):
    return dict(l.split(": ", 1) for l in (out / "session_cost.txt").read_text().splitlines())


def test_a_halted_vm_is_costed_from_its_create_and_stop_times(tmp_path):
    p, out = _run(tmp_path)
    c = _cost(out)
    assert c["minutes"] == "91"                    # 90 min 30 s, rounded up
    assert c["est_inr"] == str(round(91 * 297.47 / 60))
    assert c["rate_inr_hr"] == "297.47" and c["source"].startswith("gcloud create and stop")
    assert c["instance"] == "inst" and c["stopped"].startswith("2026-10-28T04:30")


def test_the_span_is_read_across_offsets(tmp_path):
    _, out = _run(tmp_path, created="2026-10-28T10:00:00.000+00:00",
                  stopped="2026-10-28T03:45:00.000-07:00")      # 10:45 UTC
    assert _cost(out)["minutes"] == "45"


@pytest.mark.parametrize("status, stopped", [("RUNNING", ""), ("TERMINATED", ""), ("", "")])
def test_without_a_halt_there_is_no_invented_cost_file(tmp_path, status, stopped):
    _, out = _run(tmp_path, status=status, stopped=stopped)
    assert not (out / "session_cost.txt").exists()
