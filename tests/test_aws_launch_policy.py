"""The AWS launch policy (task T5.1): no launch while the last AWS launch date
is undecided or past (India time, the date itself allowed), never an
Australia region, and the date checked again before the create request.

The committed date is PROVISIONAL (20 Dec 2026 proposed, not confirmed), so
the date tests read whatever value the policy file holds and test against
it. The UNDECIDED state is exercised on a copy.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from _aws_fakes import (SCRIPTS, committed_last_date, launch_env, run,
                        scripts_copy)

POLICY = SCRIPTS / "_aws_launch_policy.sh"


def _check(tmp_path, places, *, last, today):
    d = scripts_copy(tmp_path, last)
    script = tmp_path / "check.sh"
    script.write_text(f'source "{d}/_aws_launch_policy.sh"\n'
                      f'aws_launch_policy_check {places}\necho PASSED\n')
    p, _ = run(script, [], tmp_path, today=today)
    return p


def _shift(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=days)).isoformat()


def test_the_committed_date_is_one_valid_date_marked_provisional():
    last = committed_last_date()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", last), last
    dt.date.fromisoformat(last)
    text = POLICY.read_text()
    assert "PROVISIONAL" in text[:text.index("LAST_AWS_LAUNCH_DATE=")]


@pytest.mark.parametrize("offset,ok", [(-1, True), (0, True), (1, False), (40, False)])
def test_the_committed_date_is_inclusive_in_india_time(tmp_path, offset, ok):
    today = _shift(committed_last_date(), offset)
    p = _check(tmp_path, "us-east-1", last=None, today=today)
    assert (p.returncode == 0 and "PASSED" in p.stdout) == ok, p.stderr
    if not ok:
        assert "after the last AWS" in p.stderr


def test_the_committed_launcher_refuses_the_day_after_before_any_aws_call(tmp_path):
    p, calls = run(SCRIPTS / "aws_launch_session.sh", ["P3", "s"], tmp_path,
                   extra=launch_env(), today=_shift(committed_last_date(), 1))
    assert p.returncode == 1 and "after the last AWS" in p.stderr
    assert calls == []


def test_the_committed_launcher_launches_on_the_last_day(tmp_path):
    p, calls = run(SCRIPTS / "aws_launch_session.sh", ["P3", "s"], tmp_path,
                   extra=launch_env(), today=committed_last_date())
    assert p.returncode == 0, p.stderr
    assert sum("run-instances" in c for c in calls) == 1


def test_undecided_refuses_every_launch_before_any_aws_call(tmp_path):
    d = scripts_copy(tmp_path, "UNDECIDED")
    p, calls = run(d / "aws_launch_session.sh", ["P3", "s"], tmp_path,
                   extra=launch_env())
    assert p.returncode == 1
    assert "no last AWS launch date has been decided" in p.stderr
    assert calls == []


def test_the_date_is_read_in_india_time():
    assert "TZ=Asia/Kolkata command date +%F" in POLICY.read_text()


def test_an_unreadable_date_refuses(tmp_path):
    p = _check(tmp_path, "us-east-1", last="2026-12-31", today="garbage")
    assert p.returncode == 1 and "cannot read today's date" in p.stderr


@pytest.mark.parametrize("place", ["ap-southeast-2", "ap-southeast-2a",
                                   "ap-southeast-2-per-1a", "ap-southeast-4",
                                   "ap-southeast-4b"])
def test_australia_is_refused(tmp_path, place):
    p = _check(tmp_path, f"us-east-1 {place}", last="2026-12-31", today="2026-10-28")
    assert p.returncode == 1 and "Australia" in p.stderr


@pytest.mark.parametrize("place", ["ap-southeast-1", "ap-south-1", "us-east-1a",
                                   "ap-northeast-1"])
def test_other_regions_pass(tmp_path, place):
    p = _check(tmp_path, place, last="2026-12-31", today="2026-10-28")
    assert p.returncode == 0, p.stderr


def test_an_australia_session_never_reaches_aws(tmp_path):
    d = scripts_copy(tmp_path, "2026-12-31")
    p, calls = run(d / "aws_launch_session.sh", ["P3", "s"], tmp_path,
                   extra={**launch_env(), "AWS_SESSION_REGION": "ap-southeast-2"})
    assert p.returncode == 1 and "Australia" in p.stderr
    assert calls == []


def test_a_launch_confirmed_after_the_last_date_is_refused(tmp_path):
    """The prompt can wait past midnight; the request is the launch. The first
    read of today is the last date, the second the day after."""
    d = scripts_copy(tmp_path, "2026-12-31")
    seq = tmp_path / "today.txt"
    seq.write_text("2026-12-31\n2027-01-01\n")
    p, calls = run(d / "aws_launch_session.sh", ["P3", "s"], tmp_path,
                   extra={**launch_env(), "FAKE_TODAY_SEQ": str(seq)})
    assert p.returncode == 1 and "after the last AWS" in p.stderr, p.stderr
    assert any("describe-images" in c for c in calls)   # it got past the first check
    assert not any("run-instances" in c for c in calls)


def test_the_same_day_launches(tmp_path):
    d = scripts_copy(tmp_path, "2026-12-31")
    p, calls = run(d / "aws_launch_session.sh", ["P3", "s"], tmp_path,
                   extra=launch_env(), today="2026-12-31")
    assert p.returncode == 0, p.stderr
    assert sum("run-instances" in c for c in calls) == 1


def test_the_launcher_checks_the_policy_before_anything_else():
    text = (SCRIPTS / "aws_launch_session.sh").read_text()
    first_check = text.index("aws_launch_policy_check")
    assert first_check < text.index("aws_account_pin")
    assert first_check < text.index("aws ec2")
    # and again between the confirmation and the create request
    confirm = text.index("read -r -p")
    create = text.index("aws ec2 run-instances")
    assert "aws_launch_policy_check" in text[confirm:create]
