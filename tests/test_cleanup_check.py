"""`gcp_cleanup_check.sh` must look in every place storage bills, and must not
say "Clean" while something is billing.

**Why this is a test.** On 2026-09-05 the script printed

    Clean — no instances, unattached disks, or reserved IPs found.

over a 200 GB `attnbench-env-v5-20260905` disk image it had never queried. The
machine-images section directly above it carried a comment explaining that
machine images "were previously invisible to this check entirely, which meant
an ongoing storage charge nothing was watching" -- the identical argument
applies to plain disk images and to snapshots, and had simply not been carried
across. A tool whose entire job is finding forgotten storage, reporting the
all-clear word over forgotten storage, is worse than no tool: it converts an
unchecked risk into a checked-and-cleared one.

Two distinct defects, so two distinct kinds of assertion below:
  - it did not LOOK (fixed by querying images and snapshots), asserted by
    watching what the fake gcloud is actually asked for;
  - it said CLEAN anyway (fixed by the summary naming storage kinds),
    asserted by reading the summary line under a storage-only project.

These tests never touch GCP: a fake `gcloud` is placed first on PATH, which
also satisfies the session-wide shim in conftest.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gcp_cleanup_check.sh"

# Logs every invocation to $CALL_LOG, then emits whatever the per-resource
# FAKE_* env var holds. Empty (the default) means "the project has none",
# which is what the script's own -n tests key off.
FAKE_GCLOUD = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CALL_LOG"
ARGS="$*"
case "$ARGS" in
  *"config get-value project"*)   echo "test-project"; exit 0 ;;
  *"instances list"*)             printf '%s' "${FAKE_INSTANCES:-}"; exit 0 ;;
  *"disks list"*)                 printf '%s' "${FAKE_DISKS:-}"; exit 0 ;;
  *"machine-images list"*)        printf '%s' "${FAKE_MACHINE_IMAGES:-}"; exit 0 ;;
  *"images list"*)                printf '%s' "${FAKE_DISK_IMAGES:-}"; exit 0 ;;
  *"snapshots list"*)             printf '%s' "${FAKE_SNAPSHOTS:-}"; exit 0 ;;
  *"addresses list"*)             printf '%s' "${FAKE_ADDRESSES:-}"; exit 0 ;;
esac
exit 0
"""


def _run(tmp_path, **fakes):
    """Run the check against a fake project. Returns (stdout, exit_code, calls).

    `machine-images list` must be matched before `images list` in the fake --
    the substring is contained in it -- which mirrors the ordering hazard in
    the real gcloud surface and is why the two are separate FAKE_* vars here.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gcloud"
    fake.write_text(FAKE_GCLOUD)
    fake.chmod(0o755)

    call_log = tmp_path / "calls.txt"
    call_log.write_text("")

    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        CALL_LOG=str(call_log),
        GCP_PROJECT="test-project",
        FAKE_INSTANCES="", FAKE_DISKS="", FAKE_MACHINE_IMAGES="",
        FAKE_DISK_IMAGES="", FAKE_SNAPSHOTS="", FAKE_ADDRESSES="",
    )
    env.update({k: str(v) for k, v in fakes.items()})

    proc = subprocess.run(["bash", str(SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=60)
    return proc.stdout, proc.returncode, call_log.read_text()


def _summary(out: str) -> str:
    """Just the verdict block after the final rule.

    Scoped deliberately: the banner reads "GCP Cleanup Check", so a naive
    `"Clean" not in out` passes on the header alone and would have been a
    test that could never fail -- the shape this project keeps producing.
    """
    return out.rsplit("=" * 20, 1)[-1]


# --- it must LOOK in all three storage places ------------------------------

def test_queries_every_place_storage_can_bill(tmp_path):
    """Behaviour, not source text: assert what gcloud was actually asked."""
    _, _, calls = _run(tmp_path)
    for needed in ("instances list", "disks list", "machine-images list",
                   "snapshots list", "addresses list"):
        assert needed in calls, f"never queried {needed!r}"
    # `images list` distinct from `machine-images list`: the substring relation
    # is precisely what let the disk-image gap hide behind a section that
    # looked like it covered images.
    assert any(line.strip().startswith("compute images list")
               for line in calls.splitlines()), \
        "never queried plain disk images -- the 2026-09-05 gap"


def test_disk_image_query_excludes_public_images(tmp_path):
    """Without --no-standard-images the listing is hundreds of Google-owned
    rows and the project's own image is unfindable in it -- a section that
    technically looks but cannot be read is not a fix."""
    _, _, calls = _run(tmp_path)
    images_call = next(line for line in calls.splitlines()
                       if line.strip().startswith("compute images list"))
    assert "--no-standard-images" in images_call


# --- it must not say CLEAN over billing storage ----------------------------

def test_a_lone_disk_image_is_not_reported_as_clean(tmp_path):
    """The exact 2026-09-05 state: no compute, one 200 GB disk image."""
    out, code, _ = _run(tmp_path, FAKE_DISK_IMAGES="attnbench-env-v5  READY  200")
    assert "attnbench-env-v5" in out
    summary = _summary(out)
    assert "Clean" not in summary, "said Clean over a billing 200 GB image"
    assert "storage still bills" in summary
    assert "disk images" in summary, \
        "summary must name WHICH storage is billing, not just that some is"
    # Still exit 0: an image is not an error, it is a decision to review.
    assert code == 0


def test_a_lone_snapshot_is_not_reported_as_clean(tmp_path):
    out, code, _ = _run(tmp_path, FAKE_SNAPSHOTS="transfer-snap  200  READY")
    assert "Clean" not in _summary(out)
    assert "snapshots" in _summary(out)
    assert code == 0


def test_machine_image_alone_also_names_itself_in_the_summary(tmp_path):
    """The section that already existed had the same summary defect: it
    listed machine images and then said "Clean" underneath them."""
    out, _, _ = _run(tmp_path, FAKE_MACHINE_IMAGES="attnbench-l4-image-v3  READY  21.5")
    assert "Clean" not in _summary(out)
    assert "machine images" in _summary(out)


def test_all_three_storage_kinds_are_listed_together(tmp_path):
    out, _, _ = _run(tmp_path,
                     FAKE_MACHINE_IMAGES="mi  READY  21.5",
                     FAKE_DISK_IMAGES="di  READY  200",
                     FAKE_SNAPSHOTS="sn  200  READY")
    summary = _summary(out)
    for kind in ("machine images", "disk images", "snapshots"):
        assert kind in summary
    # Joined readably -- "machine images disk images" reads as one item.
    assert "machine images, disk images, snapshots" in summary


# --- the pre-existing behaviour must survive the change --------------------

def test_a_genuinely_empty_project_is_clean(tmp_path):
    """Also the bash 3.2 empty-array case: expanding an empty STORAGE_KINDS
    under `set -u` aborts, which would turn the all-clear path into a crash."""
    out, code, _ = _run(tmp_path)
    assert "Clean" in _summary(out)
    assert code == 0
    assert "unbound variable" not in out


def test_a_running_instance_still_fails_loudly(tmp_path):
    out, code, _ = _run(tmp_path, FAKE_INSTANCES="attnbench-a100  RUNNING")
    assert "ATTENTION" in out
    assert code == 1


def test_compute_beats_storage_in_the_summary(tmp_path):
    """With both, the instance is the urgent one and must not be softened
    into the storage advisory."""
    out, code, _ = _run(tmp_path,
                        FAKE_INSTANCES="attnbench-a100  RUNNING",
                        FAKE_DISK_IMAGES="di  READY  200")
    assert "ATTENTION" in out
    assert code == 1
