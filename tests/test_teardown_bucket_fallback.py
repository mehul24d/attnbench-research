"""The teardown pulls from the bucket when the instance's home logs are gone
(`gcp_teardown_session.sh` step 2c; researcher's item 4, 8 Nov 2026).

The instance halted itself after its exit wrapper synced (or the hard delete
took it), so SSH is dead. The fake `gcloud` serves a directory as the bucket:
the logs and manifests come down, the bulk does not, and the exit marker
decides whether the delete may go ahead without an override.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gcp_teardown_session.sh"
BUCKET = "gs://attnbench-results-test-project/inst"

FAKE = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
url_to_path() { echo "${1/gs:\/\/attnbench-results-test-project/$FAKE_GCS}"; }
case "$*" in
  "config get-value project") echo "test-project"; exit 0;;
  *"--format=value(status,creationTimestamp,lastStopTimestamp)"*)
    printf 'TERMINATED\t2026-11-08T00:00:00.000+00:00\t2026-11-08T01:00:00.000+00:00\n'; exit 0;;
  "compute instances describe"*) echo inst; exit 0;;
  "compute instances delete"*) echo deleted >> "$FAKE_LOG"; exit 0;;
  "compute machine-images list"*|*" list "*) exit 0;;
  "storage ls -l "*)
    prefix="$(url_to_path "${4%/\*\*}")"
    [[ -d "$prefix" ]] || exit 1
    find "$prefix" -type f | while read -r f; do
      echo "  $(wc -c < "$f" | tr -d ' ')  2026-11-08T00:00:00Z  ${f/$FAKE_GCS/gs://attnbench-results-test-project}"
    done
    echo "TOTAL: n objects"; exit 0;;
  "storage cp "*) cp "$(url_to_path "$3")" "$4"; exit $?;;
  *) exit 1;;     # serial port, ssh and scp fail: the VM is halted
esac
"""


def bucket(tmp_path, marker):
    root = tmp_path / "gcs" / "inst"
    for rel, text in {"logs/home/p2.log": "SESSION_EXIT=0\n", "logs/t61_01_prereqs.log": "x\n",
                      "results/sess/manifest.json": "{}", "results/sess/01_prereqs/result.json": "{}",
                      "results/sess/01_prereqs/step.log": "ok\n",
                      "results/sess/05_ground_truth/rows.parquet": "BULK"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    if marker is not None:
        (root / "logs" / "session_exit.json").write_text(json.dumps(marker))


def run(tmp_path, marker, **extra):
    bucket(tmp_path, marker)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text(FAKE)
    (bin_dir / "gcloud").chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    out = tmp_path / "session"
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("TEARDOWN_")},
           "PATH": os.pathsep.join([str(bin_dir), *real]), "FAKE_LOG": str(log),
           "FAKE_GCS": str(tmp_path / "gcs"), "ATTNBENCH_GCP_PROJECT": "test-project",
           "GCP_RATE_INR_HR": "108.64", "TEARDOWN_DISK_RESERVE_GB": "0", **extra}
    p = subprocess.run(["bash", str(SCRIPT), "inst", "zone-a", str(out)],
                       env=env, capture_output=True, text=True, timeout=120)
    return p, out, log.read_text()


def test_a_halted_vm_that_synced_at_exit_is_pulled_from_the_bucket_and_deleted(tmp_path):
    p, out, calls = run(tmp_path, {"rc": 0, "synced": True})
    assert p.returncode == 0, p.stdout + p.stderr
    assert "deleted" in calls
    assert (out / "home_logs" / "p2.log").read_text() == "SESSION_EXIT=0\n"
    assert (out / "results" / "sess" / "manifest.json").exists()
    assert (out / "results" / "sess" / "01_prereqs" / "result.json").exists()
    assert (out / "logs" / "t61_01_prereqs.log").exists()
    assert not (out / "results" / "sess" / "05_ground_truth" / "rows.parquet").exists()  # bulk stays
    assert not (out / "home_logs_missing.txt").exists()
    assert not (out / "bucket_sync_missing.txt").exists()
    assert "test-project" not in (out / "bucket_pulled_listing.txt").read_text()


@pytest.mark.parametrize("marker", [None, {"rc": 0, "synced": False}])
def test_without_a_marker_saying_it_synced_the_delete_is_refused(tmp_path, marker):
    p, out, calls = run(tmp_path, marker)
    assert p.returncode == 3 and "REFUSING TO DELETE" in p.stderr
    assert "deleted" not in calls
    assert (out / "results" / "sess" / "manifest.json").exists()     # still pulled what was there


def test_the_override_still_deletes_when_no_marker_says_it_synced(tmp_path):
    p, out, calls = run(tmp_path, None, TEARDOWN_DELETE_WITHOUT_HOME_LOGS="halted, no marker")
    assert p.returncode == 0 and "deleted" in calls
    assert "halted, no marker" in (out / "home_logs_missing.txt").read_text()
