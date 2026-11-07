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


def _run(tmp_path, *, metadata=PROJ, bucket=None, prepare=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gsutil").write_text(FAKE_GSUTIL)
    (bin_dir / "gsutil").chmod(0o755)
    work = tmp_path / "work"          # not a git repository: the dirt check is skipped
    work.mkdir()
    if prepare:
        prepare(work)
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



def test_the_sync_leaves_out_results_the_repository_already_tracks(tmp_path):
    """6 Nov 2026: the gate's sync copied every banked results/ directory into the bucket."""
    def prepare(work):
        (work / "results" / "banked.v1" / "sub").mkdir(parents=True)
        (work / "results" / "banked.v1" / "sub" / "a.parquet").write_text("x")
        (work / "results" / "other").mkdir()
        (work / "results" / "other" / "b.parquet").write_text("x")
        (work / "results" / "README.md").write_text("x")
        (work / ".gitignore").write_text("logs/\nresults/session_new/\n")
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
            subprocess.run(cmd, cwd=work, check=True, capture_output=True)
        (work / "results" / "session_new").mkdir()
        (work / "results" / "session_new" / "c.json").write_text("x")
    p, calls, ran, work = _run(tmp_path, prepare=prepare)
    assert p.returncode == 0, p.stderr
    rsync = next(c for c in calls if "rsync" in c and "/results" in c)
    assert "-x ^(banked\\.v1|other)/" in rsync, rsync
    assert "session_new" not in rsync


def test_a_tree_with_no_banked_results_syncs_everything(tmp_path):
    p, calls, ran, work = _run(tmp_path)
    rsync = next(c for c in calls if "rsync" in c and "/results" in c)
    assert " -x " not in rsync


def _sync_only(tmp_path, fail=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gsutil").write_text(FAKE_GSUTIL.replace("exit 0", "exit 1" if fail else "exit 0"))
    (bin_dir / "gsutil").chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    (work / "untracked.txt").write_text("dirt")          # a dirty tree does not stop a sync
    log = tmp_path / "gsutil.log"
    log.write_text("")
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATTNBENCH_", "GCP_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), GSUTIL_LOG=str(log),
               ATTNBENCH_BUCKET="gs://b", ATTNBENCH_SESSION="sess")
    p = subprocess.run(["bash", str(SCRIPT), "--sync-only", "teardown"], cwd=work,
                       capture_output=True, text=True, env=env, timeout=60)
    return p, log.read_text().splitlines()


def test_sync_only_runs_the_phases_own_sync_and_nothing_else(tmp_path):
    # the teardown's last sync (6 Nov 2026 ruling: bulk results stay in the bucket)
    p, calls = _sync_only(tmp_path)
    assert p.returncode == 0, p.stderr
    assert any(c.startswith("-q -m rsync -r") and c.endswith("results gs://b/sess/results")
               for c in calls), calls
    assert "=== phase" not in p.stdout


def test_a_failed_sync_only_says_so_with_its_status(tmp_path):
    p, calls = _sync_only(tmp_path, fail=True)
    assert p.returncode == 1 and "SYNC FAILED for results" in p.stderr
