"""run_phase.sh names its results bucket after the instance's own project.

Until 25 Oct 2026 the bucket, and so the project ID, was written in the
script as its default. Now `ATTNBENCH_BUCKET` overrides, and otherwise the
project is read from the metadata server. If neither gives a bucket the
phase is refused before it runs, because a phase whose output cannot be
synced is the 2026-09-15 loss. What the script prints is redacted.

The metadata server is replaced by a `file://` URL (curl reads it the same
way), and gsutil by a fake that logs its arguments.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_phase.sh"
PROJ = "meta-proj-123"

FAKE_GSUTIL = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GSUTIL_LOG"
case "$1" in ls) echo "${@: -1}/x/accuracy.parquet" ;; esac
exit 0
"""


def _run(tmp_path, *, metadata=PROJ, bucket=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gsutil").write_text(FAKE_GSUTIL)
    (bin_dir / "gsutil").chmod(0o755)
    work = tmp_path / "work"          # not a git repository: the dirt check is skipped
    work.mkdir()
    meta = tmp_path / "project-id"
    if metadata is not None:
        meta.write_text(metadata)
    log = tmp_path / "gsutil.log"
    log.write_text("")
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATTNBENCH_", "GCP_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), GSUTIL_LOG=str(log),
               ATTNBENCH_METADATA_URL=f"file://{meta}", ATTNBENCH_SESSION="sess")
    if bucket is not None:
        env["ATTNBENCH_BUCKET"] = bucket
    marker = tmp_path / "phase_ran"
    p = subprocess.run(["bash", str(SCRIPT), "probe", "--", "touch", str(marker)],
                       cwd=work, capture_output=True, text=True, env=env, timeout=60)
    return p, log.read_text().splitlines(), marker.exists(), work


def test_the_bucket_is_named_after_the_instances_project(tmp_path):
    p, calls, ran, work = _run(tmp_path)
    assert p.returncode == 0, p.stderr
    assert ran
    assert any(f"gs://attnbench-results-{PROJ}/sess/results" in c for c in calls), calls


def test_what_it_prints_and_logs_carries_no_project_id(tmp_path):
    p, calls, ran, work = _run(tmp_path)
    assert PROJ not in p.stdout + p.stderr
    assert "gs://attnbench-results-<project>/sess" in p.stdout
    assert PROJ not in (work / "logs" / "probe.log").read_text()


def test_no_metadata_and_no_override_refuses_before_the_phase(tmp_path):
    p, calls, ran, _ = _run(tmp_path, metadata=None)
    assert p.returncode == 4
    assert "no results bucket" in p.stderr
    assert not ran and calls == []


def test_a_malformed_project_from_the_metadata_server_refuses(tmp_path):
    p, calls, ran, _ = _run(tmp_path, metadata="<html>not a project</html>")
    assert p.returncode == 4 and not ran and calls == []


def test_the_override_wins_and_needs_no_metadata(tmp_path):
    p, calls, ran, _ = _run(tmp_path, metadata=None, bucket="gs://some-bucket")
    assert p.returncode == 0 and ran, p.stderr
    assert any("gs://some-bucket/sess/results" in c for c in calls), calls
