"""Teardown's age manifest: which copied files predate the instance's boot.

A machine image is a disk snapshot, so files produced by the session that
CAPTURED it ride along inside and the teardown script cannot tell them from
this session's output. On 2026-09-04 it copied `anchor16k.log`,
`bsa_build.log` and `fa_build.log` off a six-minute diagnostic instance and
filed them under that day's date; all three were session-3/4 artifacts sitting
in the v4 image. Read later, they would be mistaken for that day's
measurements.

This is instance 7's hazard arriving through a new door -- the boot-time cache
clear stops stale DERIVED state from being *read*, and says nothing about
stale OUTPUTS being *copied out and misattributed*.

The classification commands are extracted from the script and executed against
a fake tree, rather than asserted on as text. A boot step nobody has watched
work is a guess -- the same discipline as test_launch_startup_script.py.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gcp_teardown_session.sh"


def _script() -> str:
    return SCRIPT.read_text()


def test_the_manifest_runs_after_the_results_are_safe():
    """Both steps need SSH. The results cannot be reconstructed and the
    manifest can, so the manifest never goes ahead of them -- and copying does
    not alter mtimes on the instance, so it loses no accuracy by waiting."""
    t = _script()
    order = [t.index(m) for m in ('== 1/5 serial console log',
                                 '== 2/5 results',
                                 '== 3/5 age manifest',
                                 '== 4/5 delete',
                                 '== 5/5 verify')]
    assert order == sorted(order), "teardown steps are out of order"


def test_mtimes_are_read_on_the_instance_not_locally():
    """scp does not preserve mtimes. Comparing them after copying would
    classify every file as 'this session' and report success always -- the
    vacuous-check failure mode."""
    t = _script()
    manifest = t[t.index('== 3/5 age manifest'):t.index('== 4/5 delete')]
    assert 'gcloud compute ssh' in manifest
    assert 'uptime -s' in manifest, "boot time must come from the instance"
    # the find runs inside the remote --command, i.e. on paths under ~
    assert '~/attnbench_scaffold/results' in manifest


@pytest.fixture
def fake_tree(tmp_path):
    """Two files either side of a boot instant, with real mtimes."""
    old = tmp_path / "image_carried.log"
    new = tmp_path / "this_session.log"
    old.write_text("from the machine image")
    new.write_text("produced today")

    now = time.time()
    boot = now - 600                       # instance booted 10 minutes ago
    os.utime(old, (boot - 3600, boot - 3600))   # an hour before boot
    os.utime(new, (boot + 60, boot + 60))       # a minute after boot
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(boot))
    return tmp_path, stamp


def _find_commands():
    """The two classification commands, lifted verbatim from the script."""
    manifest = _script()
    manifest = manifest[manifest.index('== 3/5 age manifest'):
                        manifest.index('== 4/5 delete')]
    found = re.findall(r"^\s*(find .+?\| sort)$", manifest, re.M)
    assert len(found) == 2, f"expected 2 find commands, got {found}"
    return found


def test_the_two_find_commands_classify_a_real_tree(fake_tree):
    """Execute the script's own commands. If the predicate were inverted or
    the flag misspelled, this is what notices."""
    tmp_path, stamp = fake_tree
    predates, produced = _find_commands()

    def run(cmd: str) -> list[str]:
        cmd = cmd.replace("~/attnbench_scaffold/results ~/*.log",
                          str(tmp_path)).replace("'$BOOT_UTC'", f"'{stamp}'")
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        return [Path(l).name for l in out.stdout.split() if l.strip()]

    assert run(predates) == ["image_carried.log"]
    assert run(produced) == ["this_session.log"]


def test_the_predicates_are_complements(fake_tree):
    """Every file must land in exactly one section. A file in neither would be
    silently unattributed, which is the condition being fixed."""
    tmp_path, stamp = fake_tree
    predates, produced = _find_commands()
    # Assert the negation is present before relying on removing it: with both
    # commands identical, `replace` is a no-op and the equality below passes
    # while classifying every file into both sections. Watched that happen.
    assert "! -newermt" in predates and "! -newermt" not in produced, (
        "exactly one command must be negated")
    assert predates.replace("! -newermt", "-newermt") == produced, (
        "the two commands must differ ONLY by the negation")


def test_the_predate_count_reads_only_the_first_section(tmp_path):
    """The counting pipeline, run against a manifest of the shape the script
    writes. Counting the whole file would report this session's own files as
    image-carried."""
    manifest = tmp_path / "file_age_manifest.txt"
    manifest.write_text(
        "boot_time: 2026-09-04 20:19:00\n"
        "--- PREDATES BOOT (image-carried, NOT this session) ---\n"
        "/home/u/anchor16k.log\n"
        "/home/u/bsa_build.log\n"
        "--- PRODUCED THIS SESSION ---\n"
        "/home/u/attnbench_scaffold/results/diagnostics/flex_recheck.json\n"
        "/home/u/flex_recheck.log\n")

    cmd = (f"sed -n '/PREDATES BOOT/,/PRODUCED THIS/p' {manifest} | grep -c '^/'")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert out.stdout.strip() == "2", (
        f"expected the 2 image-carried files, got {out.stdout.strip()}")


def test_a_clean_session_reports_no_carryover(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text(
        "--- PREDATES BOOT (image-carried, NOT this session) ---\n"
        "--- PRODUCED THIS SESSION ---\n"
        "/home/u/results/segment.parquet\n")
    cmd = (f"sed -n '/PREDATES BOOT/,/PRODUCED THIS/p' {manifest} | grep -c '^/'")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert out.stdout.strip() == "0"


def test_a_missing_boot_time_warns_rather_than_claiming_attribution():
    """Empty output is not proof of a clean session -- the same
    empty-stdout-is-not-failure trap as `nvidia-smi --query-compute-apps`."""
    t = _script()
    manifest = t[t.index('== 3/5 age manifest'):t.index('== 4/5 delete')]
    assert 'UNATTRIBUTED' in manifest
    assert re.search(r'if \[\[ -n "\$BOOT_UTC" \]\]', manifest)
