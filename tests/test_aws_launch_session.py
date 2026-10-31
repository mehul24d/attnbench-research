"""scripts/aws_launch_session.sh against a fake `aws` (task T5.1).

What is checked by running the script, not by reading it: the rate, cost
cap and hard-delete limit are required and refused before any AWS call; the
worst case (rate x limit) must fit under the cap; the create request carries
the stop-on-halt, IMDSv2, encrypted delete-on-termination root volume and the
provisioning-model tag; the hard delete is a schedule at launch + limit; and
an instance whose hard delete cannot be set is terminated at once.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from _aws_fakes import INSTANCE, NOW, aws_calls, launch_env, run, scripts_copy


def _launch(tmp_path, pairing="P3", *, env=None, drop=(), stdin="launch\n"):
    d = scripts_copy(tmp_path, "2026-12-31")
    e = {k: v for k, v in launch_env().items() if k not in drop}
    e.update(env or {})
    return run(d / "aws_launch_session.sh", [pairing, "sess-1"], tmp_path,
               extra=e, stdin=stdin)


def _one(calls, needle):
    hits = [c for c in calls if needle in c]
    assert len(hits) == 1, (needle, calls)
    return hits[0]


@pytest.mark.parametrize("var", ["AWS_RATE_INR_HR", "AWS_COST_CAP_INR",
                                 "AWS_MAX_RUN_MINUTES", "AWS_SESSION_REGION",
                                 "AWS_AMI_ID", "AWS_KEY_NAME", "AWS_SECURITY_GROUP_ID"])
def test_each_required_setting_is_refused_before_any_aws_call(var, tmp_path):
    p, calls = _launch(tmp_path, drop=(var,))
    assert p.returncode == 1 and "REFUSING TO LAUNCH" in p.stderr
    assert calls == []


@pytest.mark.parametrize("rate,minutes,cap,ok", [
    ("500", "120", "1000", True),     # exactly Rs 1,000: allowed
    ("500", "120", "999", False),     # Rs 1 over
    ("500.5", "120", "1001", True),   # Rs 1,001 exactly
    ("500.5", "121", "1009", False),  # Rs 1,009.34 rounds up to 1,010
    ("462.42", "300", "2313", True),  # Rs 2,312.1 rounds up to 2,313
    ("462.42", "300", "2312", False),
])
def test_the_worst_case_must_fit_under_the_cap(tmp_path, rate, minutes, cap, ok):
    p, calls = _launch(tmp_path, env={"AWS_RATE_INR_HR": rate,
                                      "AWS_MAX_RUN_MINUTES": minutes,
                                      "AWS_COST_CAP_INR": cap})
    assert (p.returncode == 0) == ok, p.stderr
    if not ok:
        assert "exceeds the cost cap" in p.stderr
        assert calls == []


def test_the_refusal_states_the_worst_case(tmp_path):
    p, _ = _launch(tmp_path, env={"AWS_RATE_INR_HR": "500.5",
                                  "AWS_MAX_RUN_MINUTES": "121", "AWS_COST_CAP_INR": "1"})
    assert "Rs 1010 " in p.stderr


@pytest.mark.parametrize("minutes", ["29", "1441", "2h", "-5"])
def test_the_hard_delete_limit_is_bounded(tmp_path, minutes):
    p, calls = _launch(tmp_path, env={"AWS_MAX_RUN_MINUTES": minutes,
                                      "AWS_COST_CAP_INR": "100000"})
    assert p.returncode == 1 and "AWS_MAX_RUN_MINUTES" in p.stderr
    assert calls == []


@pytest.mark.parametrize("halt", ["120", "150", "19"])
def test_the_halt_must_come_before_the_hard_delete(tmp_path, halt):
    p, calls = _launch(tmp_path, env={"AWS_HALT_MINUTES": halt})
    assert p.returncode == 1 and "in-guest halt" in p.stderr
    assert calls == []


def test_an_unknown_pairing_is_refused(tmp_path):
    for pairing in ("P1", "P2", "P4", "p3", ""):
        p, calls = _launch(tmp_path, pairing)
        assert p.returncode != 0 and calls == [], pairing


def test_an_abort_at_the_prompt_creates_nothing(tmp_path):
    p, calls = _launch(tmp_path, stdin="no\n")
    assert p.returncode == 1 and "no instance created" in p.stdout
    assert not any("run-instances" in c for c in calls)


def test_a_type_not_offered_in_the_region_is_refused(tmp_path):
    p, calls = _launch(tmp_path, env={"FAKE_NOT_OFFERED": "1"})
    assert p.returncode == 1 and "not offered" in p.stderr
    assert not any("run-instances" in c for c in calls)


@pytest.mark.parametrize("pairing,itype", [("P3", "g6.2xlarge"), ("P5", "g6e.2xlarge"),
                                           ("P6", "p5.4xlarge"), ("P7", "g5.2xlarge")])
def test_the_create_request_carries_every_guarantee(tmp_path, pairing, itype):
    p, calls = _launch(tmp_path, pairing)
    assert p.returncode == 0, p.stderr
    create = _one(aws_calls(calls), "ec2 run-instances")
    assert f"--instance-type {itype}" in create
    assert "--count 1" in create
    assert "--instance-initiated-shutdown-behavior stop" in create
    assert "HttpTokens=required" in create and "InstanceMetadataTags=enabled" in create
    assert ("DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,"
            "DeleteOnTermination=true,Encrypted=true}") in create
    assert "--instance-market-options" not in create
    for tag in ("Key=attnbench-session,Value=sess-1", f"Key=attnbench-pairing,Value={pairing}",
                "Key=attnbench-provisioning-model,Value=STANDARD"):
        assert create.count(tag) == 2, tag   # on the instance and on its volume
    assert "ResourceType=instance" in create and "ResourceType=volume" in create
    assert "PROVISIONING_MODEL=STANDARD" in p.stdout
    assert f"INSTANCE_ID={INSTANCE}" in p.stdout


@pytest.mark.parametrize("model", ["spot", "SPOT", "on-demand,spot", "capacity-block", ""])
def test_anything_but_on_demand_is_refused_before_any_aws_call(tmp_path, model):
    """A Spot interruption would ruin a timed run (researcher, 28 Oct 2026).
    An empty value is refused too: it is a setting typed wrong, not a default."""
    p, calls = _launch(tmp_path, env={"AWS_PROVISIONING": model})
    assert p.returncode == 1 and "on-demand only" in p.stderr
    assert calls == []


def test_on_demand_spelled_out_launches(tmp_path):
    p, calls = _launch(tmp_path, env={"AWS_PROVISIONING": "on-demand"})
    assert p.returncode == 0, p.stderr
    create = _one(aws_calls(calls), "run-instances")
    assert "spot" not in create.lower()


def test_the_hard_delete_is_a_schedule_at_launch_plus_the_limit(tmp_path):
    p, calls = _launch(tmp_path)
    assert p.returncode == 0, p.stderr
    calls = aws_calls(calls)
    expect = dt.datetime.fromtimestamp(NOW + 120 * 60, dt.timezone.utc)
    stamp = expect.strftime("%Y-%m-%dT%H:%M:%S")
    sched = _one(calls, "scheduler create-schedule")
    assert f"--name attnbench-hard-delete-{INSTANCE}" in sched
    assert f"--schedule-expression at({stamp})" in sched
    assert "--schedule-expression-timezone UTC" in sched
    assert "--action-after-completion DELETE" in sched
    assert "aws-sdk:ec2:terminateInstances" in sched
    assert f'\\"InstanceIds\\":[\\"{INSTANCE}\\"]' in sched
    assert ":role/attnbench-hard-delete" in sched
    # the tag and the schedule carry the same time
    assert f"Key=attnbench-hard-delete-at,Value={stamp}Z" in _one(calls, "run-instances")
    assert f"HARD_DELETE_AT={stamp}Z" in p.stdout
    # order: create, then schedule, then read the schedule back
    i = [k for k, c in enumerate(calls) if "run-instances" in c][0]
    j = [k for k, c in enumerate(calls) if "create-schedule" in c][0]
    k = [k for k, c in enumerate(calls) if "get-schedule" in c][0]
    assert i < j < k
    assert not any("terminate-instances" in c for c in calls)


def test_the_user_data_halts_at_the_declared_minutes(tmp_path):
    """The startup script is a temp file removed on exit; the fake cannot read
    it after the fact, so the halt figure is read from the summary and the
    heredoc together."""
    p, _ = _launch(tmp_path, env={"AWS_HALT_MINUTES": "75"})
    assert p.returncode == 0
    assert "in-guest halt: +75 min, STOPS the instance" in p.stdout
    text = (scripts_copy(tmp_path, None) / "aws_launch_session.sh").read_text()
    assert "shutdown -h +${HALT_MINUTES}" in text


@pytest.mark.parametrize("fault,env", [
    ("could not be created", {"FAKE_SCHED_FAIL": "1"}),
    ("reads back as 'DISABLED'", {"FAKE_SCHED_STATE": "DISABLED"}),
])
def test_an_instance_without_its_hard_delete_is_terminated_at_once(tmp_path, fault, env):
    p, calls = _launch(tmp_path, env=env)
    assert p.returncode == 1
    assert fault in p.stderr
    term = _one(aws_calls(calls), "ec2 terminate-instances")
    assert f"--instance-ids {INSTANCE}" in term


def test_a_failed_terminate_after_a_failed_schedule_is_loud(tmp_path):
    p, _ = _launch(tmp_path, env={"FAKE_SCHED_FAIL": "1", "FAKE_TERMINATE_RC": "1"})
    assert p.returncode == 1 and "TERMINATE FAILED" in p.stderr


def test_a_failed_create_points_at_the_cleanup_check(tmp_path):
    p, calls = _launch(tmp_path, env={"FAKE_RUN_FAIL": "1"})
    assert p.returncode == 1
    assert "aws_cleanup_check.sh" in p.stderr
    assert not any("create-schedule" in c for c in calls)


def test_the_account_never_appears_in_the_output(tmp_path):
    from _aws_fakes import ACCOUNT
    p, calls = _launch(tmp_path, env={"FAKE_SCHED_FAIL": "1"})
    assert ACCOUNT in "".join(calls)          # the role ARN does carry it...
    assert ACCOUNT not in p.stdout + p.stderr  # ...but it is never printed


def test_a_subnet_wins_over_a_zone(tmp_path):
    p, calls = _launch(tmp_path, env={"AWS_SUBNET_ID": "subnet-0123456789abcdef0",
                                      "AWS_SESSION_AZ": "us-east-1a"})
    assert p.returncode == 0, p.stderr
    create = _one(aws_calls(calls), "run-instances")
    assert "--subnet-id subnet-0123456789abcdef0" in create
    assert "--placement" not in create


def test_a_zone_outside_the_region_is_refused(tmp_path):
    p, calls = _launch(tmp_path, env={"AWS_SESSION_AZ": "us-west-2a"})
    assert p.returncode == 1 and calls == []


def test_the_iso_helper_has_a_bsd_fallback():
    text = (scripts_copy.__globals__["SCRIPTS"] / "aws_launch_session.sh").read_text()
    assert re.search(r'date -u -d "@\$1".*\n.*date -u -r "\$1"', text)
