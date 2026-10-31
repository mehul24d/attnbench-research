"""scripts/aws_teardown_session.sh against a fake `aws`, `ssh` and `scp`
(task T5.1).

The GCP teardown's guarantees, checked by running it: no rate, no run; a
wrong region is refused with the right one named and nothing touched; the
console log is saved before anything else and a failed re-fetch never empties
it; terminate runs even when SSH is gone; session_cost.txt carries the
billable minutes and cost. One AWS-specific guard: an instance without the
launcher's attnbench-session tag is never deleted.
"""

from __future__ import annotations

import pytest

from _aws_fakes import BOOT, INSTANCE, REGION, SCRIPTS, run

SCRIPT = SCRIPTS / "aws_teardown_session.sh"
OURS = {"FAKE_INSTANCE_REGION": REGION, "FAKE_SESSION_TAG": "attnbench-p3-x",
        "AWS_RATE_INR_HR": "120"}


def _teardown(tmp_path, *, region=REGION, env=None, instance=INSTANCE):
    out = tmp_path / "session"
    e = {**OURS, **(env or {})}
    p, calls = run(SCRIPT, [instance, region, str(out)], tmp_path, extra=e)
    return p, calls, out


def _destructive(calls):
    return [c for c in calls if "terminate-instances" in c or c.startswith(("scp", "ssh"))
            or "get-console-output" in c or "delete-schedule" in c]


def test_no_rate_no_run(tmp_path):
    p, calls, _ = _teardown(tmp_path, env={"AWS_RATE_INR_HR": ""})
    assert p.returncode == 2 and "AWS_RATE_INR_HR" in p.stderr
    assert calls == []


@pytest.mark.parametrize("bad", ["i-xyz", "inst", "i-0123; rm -rf ~"])
def test_a_malformed_instance_id_is_refused(tmp_path, bad):
    p, calls, _ = _teardown(tmp_path, instance=bad)
    assert p.returncode == 2 and calls == []


def test_a_wrong_region_is_refused_and_the_right_one_named(tmp_path):
    p, calls, _ = _teardown(tmp_path, region="us-west-2")
    assert p.returncode == 2
    assert f"it is in {REGION}" in p.stderr
    assert _destructive(calls) == []


def test_an_instance_the_launcher_did_not_tag_is_never_deleted(tmp_path):
    p, calls, _ = _teardown(tmp_path, env={"FAKE_SESSION_TAG": "None"})
    assert p.returncode == 2 and "attnbench-session" in p.stderr
    assert _destructive(calls) == []


def test_the_full_order(tmp_path):
    p, calls, out = _teardown(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr

    def first(needle):
        return next(i for i, c in enumerate(calls) if needle in c)

    console, results, manifest = first("get-console-output"), first("scp "), first("uptime -s")
    term, wait, sched = (first("terminate-instances"), first("ec2 wait instance-terminated"),
                         first("delete-schedule"))
    assert console < results < manifest < term < wait < sched
    assert f"--name attnbench-hard-delete-{INSTANCE}" in calls[sched]
    assert (out / "console.log").read_text().startswith("boot line")
    assert "OOM kills recorded: 1" in p.stdout
    assert "PRODUCED THIS SESSION" in (out / "file_age_manifest.txt").read_text()
    # never a stop
    assert not any("stop-instances" in c for c in calls)


def test_session_cost_from_the_instance_boot_clock(tmp_path):
    """Boot 06:00:00Z, now 07:00:00Z: 60 minutes at Rs 120/h is Rs 120."""
    p, _, out = _teardown(tmp_path)
    cost = dict(line.split(": ", 1) for line in (out / "session_cost.txt").read_text().splitlines())
    assert cost["minutes"] == "60" and cost["rate_inr_hr"] == "120" and cost["est_inr"] == "120"
    assert cost["provider"] == "aws" and cost["instance"] == INSTANCE
    assert cost["region"] == REGION and cost["session"] == "attnbench-p3-x"
    assert cost["basis"].startswith("instance boot clock")


@pytest.mark.parametrize("rate,inr", [("108.64", "105"),   # 105.02 rounds down
                                      ("120.6", "117")])    # 116.58 rounds up
def test_session_cost_rounds_minutes_up_and_rupees_to_nearest(tmp_path, rate, inr):
    """Boot 06:00:00Z, now 06:57:59Z: 57 min 59 s bills as 58 minutes."""
    p, _, out = _teardown(tmp_path, env={"AWS_RATE_INR_HR": rate,
                                         "FAKE_NOW": str(BOOT + 57 * 60 + 59)})
    text = (out / "session_cost.txt").read_text()
    assert "minutes: 58\n" in text
    assert f"est_inr: {inr}\n" in text


def test_session_cost_falls_back_to_launch_time_when_ssh_is_gone(tmp_path):
    """SSH dead: no results, no manifest, but the bill is still bounded from
    the API's LaunchTime (06:00:00Z, so 60 minutes again), and terminate
    still runs."""
    p, calls, out = _teardown(tmp_path, env={"FAKE_SSH_FAIL": "1"})
    assert any("terminate-instances" in c for c in calls)
    text = (out / "session_cost.txt").read_text()
    assert "minutes: 60\n" in text and "est_inr: 120\n" in text
    assert "basis: EC2 LaunchTime" in text
    assert "UNATTRIBUTED" in p.stdout


def test_a_failed_console_fetch_keeps_the_saved_log(tmp_path):
    out = tmp_path / "session"
    out.mkdir()
    (out / "console.log").write_text("the first run's log\n")
    p, calls, out = _teardown(tmp_path, env={"FAKE_CONSOLE_FAIL": "1"})
    assert any("get-console-output" in c for c in calls)
    assert (out / "console.log").read_text() == "the first run's log\n"
    assert not (out / "console.log.partial").exists()
    assert any("terminate-instances" in c for c in calls)


def test_a_stopped_instance_is_still_terminated_without_ssh(tmp_path):
    """A halted (stopped) on-demand instance: SSH is impossible, the volume
    still bills, so it is terminated."""
    p, calls, _ = _teardown(tmp_path, env={"FAKE_STATE": "stopped"})
    assert not any(c.startswith(("ssh", "scp")) for c in calls)
    assert any("terminate-instances" in c for c in calls)


def test_a_failed_terminate_is_loud(tmp_path):
    p, _, _ = _teardown(tmp_path, env={"FAKE_TERMINATE_RC": "1"})
    assert "TERMINATE FAILED" in p.stdout


def test_an_instance_found_nowhere_touches_nothing(tmp_path):
    p, calls, _ = _teardown(tmp_path, env={"FAKE_INSTANCE_REGION": "eu-north-1"})
    assert "not found in any region" in p.stdout
    assert _destructive(calls) == []
    assert any("describe-regions" in c for c in calls)


def test_it_ends_with_the_cleanup_check(tmp_path):
    p, calls, _ = _teardown(tmp_path)
    term = max(i for i, c in enumerate(calls) if "terminate-instances" in c)
    assert any("describe-volumes" in c for c in calls[term:])
    assert "AWS Cleanup Check" in p.stdout
