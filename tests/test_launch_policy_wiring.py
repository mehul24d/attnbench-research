"""Every script that creates a GCP instance applies the launch policy, and
none of them changes the date the policy reads.

**Why this file exists.** `test_launch_policy.py` checked the scripts named
`gcp_launch_*.sh`. On 2026-10-20 `gcp_boot_test_image.sh`, which also creates
an instance, did not source `scripts/_launch_policy.sh`, and on the
`t11-run` branch neither did the H100 and compile-session launchers. A name
glob cannot see a launcher with another name. So the scripts are found here
by what they do: every `scripts/*.sh` that calls `gcloud compute instances
create`.

Each one must source the policy with the call the other launchers use, and
call `launch_policy_check` before its first create. It must also leave the
date alone: the policy reads today's date only from `date` (India time), and
a script that set the last date, `PATH`, `TZ` or `BASH_ENV`, or defined a
`date` or `command` function, could change which day counts. That is checked
twice: in the text, and by running each script with a fake `date` and a fake
`gcloud` first on `PATH`, once on the day after the last launch date (it
must refuse before any create) and once on the last date itself (it must
not be refused for the date).

These tests never touch GCP: the fake `gcloud` records its arguments and
fails every create, and the real one is not on the `PATH` the scripts see.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
POLICY = SCRIPTS / "_launch_policy.sh"
CREATE = "gcloud compute instances create"

# The call every launcher uses: source the policy from the script's own
# directory, then check the zone, or the zone list where the script walks
# zones.
POLICY_CALL = re.compile(
    r'^source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/_launch_policy\.sh"\n'
    r'launch_policy_check "\$(?:ZONE|ZONE_FALLBACKS)"$', re.M)

# Anything in a launch script that could change which date the policy sees.
DATE_CHANGERS = (
    (r"LAST_GCP_LAUNCH_DATE", "sets or reads the last launch date itself"),
    (r"^\s*(?:function\s+)?date\s*\(\)", "defines a `date` function"),
    (r"^\s*(?:function\s+)?command\s*\(\)", "defines a `command` function"),
    (r"\balias\s+date\b", "aliases `date`"),
    (r"\benable\s+-n\b", "disables a builtin"),
    (r"^\s*(?:export\s+)?PATH=", "changes PATH"),
    (r"\bBASH_ENV\b", "sets BASH_ENV"),
    (r"^\s*(?:export\s+)?TZ=", "sets TZ for the whole script"),
)

FAKE_DATE = """#!/bin/bash
if [[ "$*" == "+%F" ]]; then echo {today}; else exec /bin/date "$@"; fi
"""

FAKE_GCLOUD = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
  *"config get-value project"*) echo "test-project"; exit 0 ;;
  *"instances create"*)         echo "fake gcloud: create refused" >&2; exit 1 ;;
esac
exit 1
"""


def _creating_scripts() -> list[Path]:
    return sorted(p for p in SCRIPTS.glob("*.sh") if CREATE in p.read_text())


def _last_date() -> str:
    m = re.search(r'^LAST_GCP_LAUNCH_DATE="(\d{4}-\d{2}-\d{2})"$', POLICY.read_text(), re.M)
    assert m, "LAST_GCP_LAUNCH_DATE not found in the policy file"
    return m.group(1)


def _day_after(day: str) -> str:
    import datetime
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()


def _run(script: Path, today: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("date", FAKE_DATE.format(today=today)), ("gcloud", FAKE_GCLOUD)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.touch()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
           "GCLOUD_LOG": str(log), "GCP_PROJECT": "test-project",
           "ATTNBENCH_GCP_PROJECT": "test-project",
           "GCP_RATE_INR_HR": "297.47"}
    proc = subprocess.run(["bash", str(script), "test-image"], capture_output=True,
                          text=True, env=env, cwd=tmp_path, timeout=120)
    return proc, log.read_text()


SCRIPT_IDS = [p.name for p in _creating_scripts()]


def test_the_search_finds_every_known_launch_script():
    """Anti-vacuity: the search must find at least what the name glob found,
    plus the boot test, which the glob missed."""
    found = {p.name for p in _creating_scripts()}
    assert {p.name for p in SCRIPTS.glob("gcp_launch_*.sh")} <= found
    assert "gcp_boot_test_image.sh" in found
    assert len(found) >= 5, found


@pytest.mark.parametrize("name", SCRIPT_IDS)
def test_each_launch_script_sources_the_policy_before_creating(name):
    text = (SCRIPTS / name).read_text()
    calls = POLICY_CALL.findall(text)
    assert len(calls) == 1, f"{name}: expected one shared policy call, found {len(calls)}"
    assert POLICY_CALL.search(text).start() < text.find(CREATE), \
        f"{name} creates an instance before the policy check"


@pytest.mark.parametrize("name", SCRIPT_IDS)
def test_no_launch_script_changes_which_date_counts(name):
    text = (SCRIPTS / name).read_text()
    for pattern, what in DATE_CHANGERS:
        assert not re.search(pattern, text, re.M), f"{name} {what}"


def test_the_text_check_notices_a_date_change():
    """Anti-vacuity: a script that defines `date` must be caught."""
    text = "date() { echo 2026-11-01; }\n"
    assert any(re.search(p, text, re.M) for p, _ in DATE_CHANGERS)


@pytest.mark.parametrize("name", SCRIPT_IDS)
def test_each_launch_script_refuses_the_day_after_before_any_create(name, tmp_path):
    proc, calls = _run(SCRIPTS / name, _day_after(_last_date()), tmp_path)
    assert proc.returncode != 0, f"{name} ran on the day after the last launch date"
    assert "is after the last GCP" in proc.stderr, (name, proc.stderr[-2000:])
    assert "instances create" not in calls, f"{name} reached a create: {calls}"


@pytest.mark.parametrize("name", SCRIPT_IDS)
def test_no_launch_script_is_refused_on_the_last_date_itself(name, tmp_path):
    proc, _ = _run(SCRIPTS / name, _last_date(), tmp_path)
    assert "REFUSING TO LAUNCH -- today" not in proc.stderr, (name, proc.stderr[-2000:])
    assert "cannot read today's date" not in proc.stderr, (name, proc.stderr[-2000:])
