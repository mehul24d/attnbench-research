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


def test_the_zones_covered_and_a_recently_terminated_instance_zone_are_listed(tmp_path):
    """Region-wide listings cover every zone; the zones are named, and the
    study's recently terminated instance shows the zone it ran in. Neither
    is an error (the researcher's yes of 23 Nov 2026, zone fallback)."""
    p, calls = run(SCRIPT, [], tmp_path, extra={"FAKE_TERMINATED_REGION": "ap-south-1",
                                                "FAKE_ZONE": "ap-south-1b"})
    assert p.returncode == 0, p.stdout
    assert "ap-south-1a, ap-south-1b, ap-south-1c" in p.stdout     # a separator (23 Nov 2026: none)
    assert f"{INSTANCE}\tg6.xlarge\tap-south-1b\tsess-1" in p.stdout
    assert "Clean -- no instances" in p.stdout
    q = [c for c in calls if "describe-instances" in c and "Values=terminated" in c]
    assert len(q) == 3 and all("tag-key,Values=attnbench-session" in c for c in q)


def test_a_region_the_organization_denies_is_named_and_its_error_never_printed(tmp_path):
    """The organization's policy denies EC2 outside the study's Region; the CLI's
    denial names the policy's ARN, the role and its session id. They reached
    every driver log until 26 Nov 2026. The region is named as denied, no ARN,
    session id or account number is printed, and the check is still not clean."""
    import re
    from _aws_fakes import ACCOUNT
    p, _ = run(SCRIPT, [], tmp_path, extra={"FAKE_SCP_DENIED_REGIONS": "us-east-1 us-west-2"})
    out = p.stdout + p.stderr
    assert p.returncode == 2, out
    assert "denied by the organization's policy, not listed" in p.stdout
    assert "Denied by the organization's policy (not listed): 2 region(s): us-east-1 us-west-2" in p.stdout
    assert "INCOMPLETE" in p.stdout and "Clean --" not in p.stdout
    for leak in ("arn:", "assumed-role/FakeRole", "fake-session-0123", "o-fakeorg123", "p-fake1234", ACCOUNT):
        assert leak not in out, leak
    assert not re.search(r"(?<![0-9])[0-9]{12}(?![0-9])", out)
    assert "## ap-south-1" in p.stdout        # the region not denied is still listed


def test_another_error_is_printed_masked(tmp_path):
    """An error that is not the organization's denial is still shown, masked."""
    from _aws_fakes import ACCOUNT
    p, _ = run(SCRIPT, [], tmp_path, extra={"FAKE_FAIL_SNAPSHOTS_MSG": "1"})
    assert p.returncode == 2
    assert "AccessDenied for arn:<masked>" in p.stderr and ACCOUNT not in p.stderr
