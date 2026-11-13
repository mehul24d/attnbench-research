"""scripts/tag_lock.py (T4.4) in a scratch git repository: every refusal,
then the tag, which is local and never pushed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import tag_lock  # noqa: E402
from attnbench.lock import checklist  # noqa: E402

PLAN_TEXT = checklist.PLAN.read_text(encoding="utf-8")
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
TAG = "lock-2026-11-12"
LOCK = "docs/lock_values.json"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, env=ENV,
                          capture_output=True, text=True).stdout


def lock_json(complete=True) -> str:
    # Typed values and every hypothesis's pairings (ruling 1b, 11 Nov 2026).
    import lock_fixture as LF
    lock = LF.complete_lock(PLAN_TEXT)
    if not complete:
        lock["values"]["mask_gate_bound"]["value"] = None
    return json.dumps(lock, indent=1)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
              "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, ENV[k])          # the script's own git calls inherit them
    r = tmp_path / "repo"
    (r / "docs").mkdir(parents=True)
    git(tmp_path, "init", "-q", str(r))
    (r / "docs" / "RESEARCH_PLAN.md").write_text(PLAN_TEXT, encoding="utf-8")
    (r / LOCK).write_text(lock_json())
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "lock values")
    return r


def run(repo, *extra):
    return tag_lock.main(["--repo", str(repo), "--lock-file", LOCK, "--tag", TAG, *extra])


def test_a_complete_clean_lock_commit_is_tagged_locally(repo, capsys):
    assert run(repo) == 0
    out = capsys.readouterr().out
    assert f"git push origin {TAG}" in out
    msg = git(repo, "tag", "-l", "-n20", TAG)
    assert "lock sha256:" in msg and "plan sha256:" in msg
    assert git(repo, "rev-list", "-n1", TAG).strip() == git(repo, "rev-parse", "HEAD").strip()
    assert git(repo, "remote").strip() == ""            # nothing to push to, nothing pushed


def test_a_dry_run_checks_and_makes_no_tag(repo, capsys):
    assert run(repo, "--dry-run") == 0
    assert "no tag made" in capsys.readouterr().out
    assert git(repo, "tag", "-l").strip() == ""


def _refused(repo, capsys, match):
    assert run(repo) == 2
    assert match in capsys.readouterr().err
    assert git(repo, "tag", "-l").strip() == ""


def test_a_dirty_tree_is_refused(repo, capsys):
    (repo / "notes.txt").write_text("untracked")
    _refused(repo, capsys, "not clean")


def test_an_incomplete_lock_file_is_refused(repo, capsys):
    (repo / LOCK).write_text(lock_json(complete=False))
    git(repo, "commit", "-q", "-am", "unset one")
    _refused(repo, capsys, "mask_gate_bound")


def test_an_untracked_lock_file_is_refused(repo, capsys):
    git(repo, "rm", "-q", "--cached", LOCK)
    git(repo, "commit", "-q", "-m", "untrack")
    (repo / ".git" / "info" / "exclude").write_text(LOCK + "\n")
    _refused(repo, capsys, "not tracked")


def test_an_existing_tag_or_a_bad_name_is_refused(repo, capsys):
    git(repo, "tag", TAG)
    assert run(repo) == 2 and "already exists" in capsys.readouterr().err
    assert tag_lock.main(["--repo", str(repo), "--lock-file", LOCK,
                          "--tag", "lock-final"]) == 2
    assert "lock-YYYY-MM-DD" in capsys.readouterr().err


@pytest.mark.parametrize("tag, ok", [
    ("lock-2026-11-13", True), ("lock-2026-11-13-r2", True), ("lock-2026-11-13-r9", True),
    ("lock-2026-11-13-r10", True), ("lock-2026-11-13-r23", True),
    ("lock-2026-11-13-r1", False), ("lock-2026-11-13-r0", False), ("lock-2026-11-13-r01", False),
    ("lock-2026-11-13-r", False), ("lock-2026-11-13r2", False), ("lock-2026-11-13-R2", False),
    ("lock-2026-11-13-r2-x", False), ("lock-2026-11-13-v2", False), ("lock-2026-11-13_r2", False),
    ("lock-2026-11-1-r2", False), ("lock-26-11-13-r2", False), ("lock-final", False),
    ("lock-2026-11-13-r2\n", False), ("xlock-2026-11-13-r2", False),
])
def test_a_re_cut_tag_takes_rn_and_every_other_form_is_refused(repo, capsys, tag, ok):
    # the adviser's rulings dated 13 Nov 2026, item 6: lock-2026-11-13 and -14 are taken
    tag = tag.replace("\\n", "\n")
    rc = tag_lock.main(["--repo", str(repo), "--lock-file", LOCK, "--tag", tag, "--dry-run"])
    err = capsys.readouterr().err
    assert (rc == 0) if ok else (rc == 2 and "lock-YYYY-MM-DD-rN" in err), (tag, rc, err)


def test_a_lock_file_for_another_plan_is_refused(repo, capsys):
    (repo / "docs" / "RESEARCH_PLAN.md").write_text(PLAN_TEXT + "\nlater\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "plan changed after the lock values")
    _refused(repo, capsys, "another version of the plan")
