"""The launch policy (scripts/_launch_policy.sh) and where it is applied.

docs/RESEARCH_PLAN.md, Section 6: no GCP session launches after the last
GCP launch date, and no session runs in an Australia region. Both must be
enforced before anything is created, in every launch script, not remembered
by whoever types the command.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "scripts" / "_launch_policy.sh"
LAUNCH_SCRIPTS = sorted((REPO / "scripts").glob("gcp_launch_*.sh"))


def _check(zones: str, today: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f'source "{POLICY}"; launch_policy_check {zones}; echo PASSED'],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "ATTNBENCH_TODAY": today},
        timeout=30)


def _last_date() -> str:
    m = re.search(r'^LAST_GCP_LAUNCH_DATE="(\d{4}-\d{2}-\d{2})"$', POLICY.read_text(), re.M)
    assert m, "LAST_GCP_LAUNCH_DATE not found in the policy file"
    return m.group(1)


def test_last_launch_date_matches_the_plan():
    """One date, stated in two places that must agree."""
    plan = (REPO / "docs" / "RESEARCH_PLAN.md").read_text()
    assert _last_date() == "2026-11-22"
    assert "22 November 2026" in plan


def test_the_last_launch_date_itself_is_allowed():
    proc = _check("asia-northeast1-a", _last_date())
    assert proc.returncode == 0 and "PASSED" in proc.stdout, proc.stderr


def test_the_day_after_is_refused():
    proc = _check("asia-northeast1-a", "2026-11-23")
    assert proc.returncode != 0
    assert "PASSED" not in proc.stdout
    assert "last GCP" in proc.stderr


def test_an_unreadable_date_is_refused():
    proc = _check("asia-northeast1-a", "23-11-2026")
    assert proc.returncode != 0 and "PASSED" not in proc.stdout


@pytest.mark.parametrize("zones", ["australia-southeast1-a",
                                   "asia-northeast1-a australia-southeast2-b"])
def test_an_australia_zone_anywhere_in_the_list_is_refused(zones):
    proc = _check(zones, "2026-10-04")
    assert proc.returncode != 0 and "PASSED" not in proc.stdout
    assert "Australia" in proc.stderr


def test_a_permitted_zone_list_passes():
    proc = _check("asia-northeast1-a asia-northeast1-c", "2026-10-04")
    assert proc.returncode == 0 and "PASSED" in proc.stdout, proc.stderr


def test_every_launch_script_applies_the_policy_before_creating_anything():
    assert len(LAUNCH_SCRIPTS) >= 4, "the glob found too few launch scripts to mean anything"
    for script in LAUNCH_SCRIPTS:
        text = script.read_text()
        check = text.find("launch_policy_check ")
        create = text.find("gcloud compute instances create")
        assert "_launch_policy.sh" in text, f"{script.name} does not source the policy"
        assert check != -1, f"{script.name} never calls launch_policy_check"
        assert create != -1, f"{script.name} has no create call to protect"
        assert check < create, f"{script.name} creates an instance before the policy check"


def test_l4_launch_scripts_default_to_the_pinned_zone():
    for name in ("gcp_launch_l4.sh", "gcp_launch_compile_session.sh"):
        text = (REPO / "scripts" / name).read_text()
        assert 'ZONE="${GCP_ZONE:-asia-northeast1-a}"' in text, name
