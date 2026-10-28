"""scripts/aws_cleanup_check.sh against a fake `aws` (task T5.1).

Read-only, every enabled region, exit 1 for anything billed by the hour,
storage named in the summary but not an error, and a failed listing is never
read as "nothing there".
"""

from __future__ import annotations

from _aws_fakes import INSTANCE, SCRIPTS, aws_calls, run

SCRIPT = SCRIPTS / "aws_cleanup_check.sh"


def test_a_clean_account_says_clean_and_checks_every_region(tmp_path):
    p, calls = run(SCRIPT, [], tmp_path)
    assert p.returncode == 0, p.stderr
    assert "Clean -- no instances" in p.stdout
    calls = aws_calls(calls)
    for region in ("us-east-1", "us-west-2", "ap-south-1"):
        assert any("describe-instances" in c and f"--region {region}" in c for c in calls), region
        assert any("describe-volumes" in c and f"--region {region}" in c for c in calls), region


def test_an_instance_in_a_region_no_session_used_is_found(tmp_path):
    p, _ = run(SCRIPT, [], tmp_path, extra={"FAKE_LIVE_REGION": "ap-south-1"})
    assert p.returncode == 1
    assert INSTANCE in p.stdout and "ATTENTION" in p.stdout


def test_stopped_instances_count(tmp_path):
    p, calls = run(SCRIPT, [], tmp_path)
    q = [c for c in calls if "describe-instances" in c][0]
    assert "stopped" in q and "stopping" in q and "terminated" not in q


def test_an_unattached_volume_is_billable(tmp_path):
    p, _ = run(SCRIPT, [], tmp_path, extra={"FAKE_VOLUME_REGION": "us-west-2"})
    assert p.returncode == 1 and "vol-0abc" in p.stdout


def test_images_alone_are_named_but_not_an_error(tmp_path):
    p, _ = run(SCRIPT, [], tmp_path, extra={"FAKE_IMAGE_REGION": "us-east-1",
                                            "FAKE_IMAGES": "ami-0aaa\tattnbench-env\t2026"})
    assert p.returncode == 0
    assert "storage still bills: images" in p.stdout
    assert "Clean --" not in p.stdout


def test_a_failed_listing_is_not_clean(tmp_path):
    p, _ = run(SCRIPT, [], tmp_path, extra={"FAKE_FAIL_SNAPSHOTS": "1"})
    assert p.returncode == 2
    assert "INCOMPLETE" in p.stdout and "Clean --" not in p.stdout


def test_nothing_is_ever_deleted(tmp_path):
    p, calls = run(SCRIPT, [], tmp_path, extra={"FAKE_LIVE_REGION": "us-east-1",
                                                "FAKE_VOLUME_REGION": "us-east-1"})
    for c in aws_calls(calls):
        verb = c.split()[1]
        assert verb.startswith(("describe-", "list-", "get-")), c
    assert not any("UNHANDLED" in c for c in calls)
