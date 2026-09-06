"""The deploy refuses anything that would produce a lying provenance stamp.

Root cause of the 2026-09-04 stamp failure: source was untarred over the
instance's existing checkout. That replaced `attnbench/`, `scripts/`, `tests/`
and `docs/` and left `.git` describing the machine image, so
`provenance.capture()` reported the image's commit -- a real commit, 18 behind
the code, on all 4576 rows.

Copying source without the history that identifies it is the whole bug, so the
fix is a deploy that moves real git objects (a bundle, since this project has
no remote) and then asks the INSTANCE what commit it is at.

Two guards, and both directions matter:

  * refuse to deploy a dirty local tree -- otherwise the instance would run
    code no commit describes, stamping rows with a commit that does not match
    them, which is the original bug from the other end; and
  * refuse to proceed unless the instance reports the expected commit and a
    clean tree -- the check that would actually have caught it.

The local guard is exercised for real here against a temporary repo. The
remote guard needs an instance, so its presence is asserted against the
source: a verification step that got deleted or commented out would otherwise
leave the script looking identical to one that works.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gcp_deploy_source.sh"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


@pytest.fixture
def clean_repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "f.txt").write_text("hello\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "initial")
    return r


def _run(repo):
    """The script resolves its repo from CWD, so running it inside a fixture
    repo exercises the real guard rather than a reimplementation of it."""
    return subprocess.run(["bash", str(SCRIPT), "inst", "zone"],
                          cwd=str(repo), capture_output=True, text=True)


def test_a_dirty_tree_is_refused_before_anything_is_copied(clean_repo):
    (clean_repo / "f.txt").write_text("modified\n")
    p = _run(clean_repo)
    assert p.returncode == 1
    assert "REFUSING TO DEPLOY" in p.stderr
    assert "dirty" in p.stderr


def test_an_untracked_file_also_counts_as_dirty(clean_repo):
    """`git status --porcelain` reports untracked files too, and it must: a
    new module that exists on the instance and in no commit is exactly the
    state the stamp cannot describe."""
    (clean_repo / "new_module.py").write_text("x = 1\n")
    p = _run(clean_repo)
    assert p.returncode == 1
    assert "REFUSING TO DEPLOY" in p.stderr


def test_the_refusal_says_why_it_matters(clean_repo):
    """A guard whose message does not explain itself gets worked around."""
    (clean_repo / "f.txt").write_text("modified\n")
    p = _run(clean_repo)
    assert "commit" in p.stderr.lower()


def test_gitignored_output_does_not_trip_the_guard(clean_repo):
    """Measurement output lands in the repo on every run. If results/ tripped
    this, the guard would fire constantly and be disabled within a day."""
    (clean_repo / ".gitignore").write_text("results/\n*.parquet\n")
    _git(clean_repo, "add", "-A")
    _git(clean_repo, "commit", "-qm", "ignore results")
    (clean_repo / "results").mkdir()
    (clean_repo / "results" / "probe.parquet").write_text("data")
    p = _run(clean_repo)
    # Gets past the dirty check and fails later, at the gcloud call, which is
    # not available here -- the point is only that it did NOT refuse for dirt.
    assert "REFUSING TO DEPLOY" not in p.stderr


# ---------------------------------------------------------------------------
# The remote half, asserted against the source.
# ---------------------------------------------------------------------------

def test_it_verifies_the_commit_from_the_instance():
    src = SCRIPT.read_text()
    assert "git rev-parse HEAD" in src
    assert "DEPLOY FAILED" in src, "no refusal path on a commit mismatch"


def test_it_verifies_the_instance_tree_is_clean():
    """A matching commit with a dirty tree is the exact failure being fixed,
    so checking the commit alone would rebuild it."""
    assert "dirty=" in SCRIPT.read_text()


def test_it_does_not_deploy_by_copying_source():
    """The regression that matters. tar/rsync/scp of source trees is how the
    original stamp came to describe a machine image; only the bundle carries
    the history that makes the commit true."""
    src = SCRIPT.read_text()
    assert "git bundle create" in src
    for bad in ("tar czf", "tar xzf", "rsync"):
        assert bad not in src, f"deploy uses {bad!r}, which leaves .git stale"


def test_clean_does_not_take_x():
    """`git clean -fdx` would delete gitignored files -- which is every
    measurement result on the instance, including a checkpoint being resumed
    from. -fd keeps them."""
    src = SCRIPT.read_text()
    assert "git clean -qfd" in src
    assert "-fdx" not in src and "-xfd" not in src


# ---------------------------------------------------------------------------
# The remote git sequence, run for real against a local repo
#
# The failure this covers is in the git commands, not in the ssh wrapper, so
# the honest test runs those commands -- twice, because the bug only exists
# on the second deploy.
# ---------------------------------------------------------------------------

def _remote_sequence(script_text: str) -> list[str]:
    """The git lines the deploy script sends to the instance."""
    import re
    block = re.search(r'gcloud compute ssh "\$NAME" --zone="\$ZONE" --command="\n'
                      r'set -euo pipefail\n'
                      r'cd \$REMOTE_DIR\n'
                      r'(.*?)"\n', script_text, re.S)
    assert block, "could not find the remote checkout block in the deploy script"
    return [ln for ln in block.group(1).splitlines()
            if ln.strip() and not ln.strip().startswith("rm ")]


def test_deploying_twice_in_one_session_works(tmp_path):
    """The regression, found live on 2026-09-06 mid-session.

    git refuses to fetch into the branch a non-bare repo currently has
    checked out, and --force does not override it:

        fatal: Refusing to fetch into current branch refs/heads/deployed
               of non-bare repository

    The FIRST deploy always works, because HEAD is still on the image's own
    branch -- so this was invisible until a session deployed twice. Stage 3
    runs in four segments and re-deploys each time, so it would have fired
    regardless; it just happened to fire on a mid-session fix instead.
    """
    import subprocess

    def git(repo, *args, **kw):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, **kw)

    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    git(source, "config", "user.email", "t@t")
    git(source, "config", "user.name", "t")
    (source / "f.txt").write_text("one\n")
    git(source, "add", "-A")
    git(source, "commit", "-qm", "one")

    instance = tmp_path / "instance"
    subprocess.run(["git", "clone", "-q", str(source), str(instance)], check=True)
    git(instance, "config", "user.email", "t@t")
    git(instance, "config", "user.name", "t")

    lines = _remote_sequence(
        (Path(__file__).resolve().parents[1] / "scripts" / "gcp_deploy_source.sh").read_text())

    for round_n, content in enumerate(("two\n", "three\n"), start=1):
        (source / "f.txt").write_text(content)
        git(source, "add", "-A")
        git(source, "commit", "-qm", f"round {round_n}")
        want = git(source, "rev-parse", "HEAD").stdout.strip()

        bundle = tmp_path / f"deploy{round_n}.bundle"
        subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"],
                       capture_output=True, check=True)

        for line in lines:
            cmd = line.strip().replace("/tmp/deploy.bundle", str(bundle))
            r = subprocess.run(cmd, shell=True, cwd=instance,
                               capture_output=True, text=True)
            assert r.returncode == 0, (
                f"deploy round {round_n} failed on: {cmd}\n{r.stderr}")

        got = git(instance, "rev-parse", "HEAD").stdout.strip()
        assert got == want, f"round {round_n}: instance at {got}, wanted {want}"
        assert git(instance, "status", "--porcelain").stdout.strip() == ""
