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
from pathlib import Path

import pytest

from _aws_fakes import INSTANCE, MY_IP, NOW, aws_calls, launch_env, run, scripts_copy


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
    for pairing in ("P1", "P2", "P4", "p3", "env", "ENV2", ""):
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
                                           ("P6", "p5.4xlarge"), ("P7", "g5.2xlarge"),
                                           ("ENV", "m7a.xlarge")])
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
    sched = _one(calls, f"--name attnbench-hard-delete-{INSTANCE} --schedule-expression")
    assert sched.startswith("scheduler create-schedule")
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
    j = [k for k, c in enumerate(calls) if "create-schedule" in c and "hard-delete-i-" in c][0]
    k = [k for k, c in enumerate(calls) if "get-schedule" in c and "hard-delete-i-" in c][0]
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
    assert not any("create-schedule" in c and "hard-delete-i-" in c for c in calls)


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


ENV_CARD = {"AWS_SESSION_REGION": "ap-southeast-2", "AWS_SESSION_AZ": "ap-southeast-2a",
            "AWS_RATE_INR_HR": "34.90", "AWS_COST_CAP_INR": "250",
            "AWS_HALT_MINUTES": "360", "AWS_MAX_RUN_MINUTES": "375"}


def test_the_environment_build_cards_settings(tmp_path):
    """docs/launch_cards/aws_env_build_m7a.md (adviser's rulings dated 21 Nov
    2026, fifth wave, item 5): one m7a.xlarge, halt at 360 min, hard delete
    at 375, worst case Rs 219 under the Rs 250 cap; an abort creates nothing."""
    p, calls = _launch(tmp_path, "ENV", env=ENV_CARD, stdin="no\n")
    assert p.returncode == 1 and "no instance created" in p.stdout
    assert "worst case   : Rs 219, cost cap Rs 250" in p.stdout
    assert not any(w in c for c in aws_calls(calls) for w in ("run-instances", "create-schedule"))
    (tmp_path / "go").mkdir()
    p, calls = _launch(tmp_path / "go", "ENV", env=ENV_CARD)
    assert p.returncode == 0, p.stderr
    calls = aws_calls(calls)
    create = _one(calls, "ec2 run-instances")
    assert "--instance-type m7a.xlarge" in create and "AvailabilityZone=ap-southeast-2a" in create
    stamp = dt.datetime.fromtimestamp(NOW + 375 * 60, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    assert f"--schedule-expression at({stamp})" in _one(calls, f"--name attnbench-hard-delete-{INSTANCE} --schedule-expression")
    assert "+360 min from boot" in p.stdout
    card = (Path(__file__).resolve().parents[1] / "docs" / "launch_cards"
            / "aws_env_build_m7a.md").read_text()
    for said in ("₹34.90/h", "| 360 min (the ruling)", "| 375 min (the ruling)",
                 "₹250 (the ruling); worst case ₹34.90 × 375 min = ₹219", "`ap-southeast-2a`"):
        assert said in card, said


def test_a_cap_short_of_the_env_cards_worst_case_is_refused(tmp_path):
    p, calls = _launch(tmp_path, "ENV", env={**ENV_CARD, "AWS_COST_CAP_INR": "218"})
    assert p.returncode == 1 and "exceeds the cost cap" in p.stderr and calls == []


# --- the credit gate (adviser's rulings dated 22 Nov 2026, credits) ----------

def _record(tmp_path):
    return tmp_path / ".attnbench" / "aws_launches.tsv"


def _no_create(calls):
    return not any(w in c for c in aws_calls(calls) for w in ("run-instances", "create-schedule"))


@pytest.mark.parametrize("env,said", [
    ({"FAKE_PLAN_TYPE": "FREE"}, "'FREE ACTIVE'"),
    ({"FAKE_PLAN_STATUS": "NOT_STARTED"}, "'PAID NOT_STARTED'"),
    ({"FAKE_PLAN_FAIL": "1"}, "'unreadable"),
    ({"FAKE_CREDIT_UNIT": "INR"}, "cannot read the remaining credits"),
    ({"FAKE_CREDITS": "None"}, "cannot read the remaining credits"),
])
def test_a_launch_needs_the_paid_plan_and_readable_credits(tmp_path, env, said):
    p, calls = _launch(tmp_path, env=env)
    assert p.returncode == 1 and said in p.stderr, p.stderr
    assert _no_create(calls) and not _record(tmp_path).exists()


def test_the_plan_is_read_with_the_pinned_profile_before_the_prompt(tmp_path):
    p, calls = _launch(tmp_path, stdin="no\n")
    plan = _one(aws_calls(calls), "freetier get-account-plan-state")
    assert "--profile pinned-profile" in plan and "--region us-east-1" in plan
    assert "credits      : $140.0 left (Rs 13438.60) > Rs 0 planned so far + Rs 1200 cap" in p.stdout


@pytest.mark.parametrize("planned,ok", [
    ("719.79", True),    # Rs 719.79 + Rs 1,200 = Rs 1,919.79 < $20 x 95.99 = Rs 1,919.80
    ("719.80", False),   # equal is refused: the credits must exceed
    ("719.81", False),
])
def test_the_credits_must_exceed_the_planned_spend_plus_this_cap(tmp_path, planned, ok):
    rec = _record(tmp_path)
    rec.parent.mkdir()
    rec.write_text(f"2026-11-23T00:00:00Z\tENV\ti-0aaa\t500\n# struck: refused, created nothing\n"
                   f"2026-11-23T08:00:00Z\tP3\ti-0bbb\t{float(planned) - 500:.2f}\n\n")
    p, calls = _launch(tmp_path, env={"FAKE_CREDITS": "20.00"})
    assert (p.returncode == 0) == ok, p.stderr
    if not ok:
        assert "do not exceed the planned AWS spend so far" in p.stderr
        assert f"Rs {planned}, plus this block's cap" in p.stderr and _no_create(calls)


def test_the_cap_alone_must_fit_under_the_credits(tmp_path):
    p, calls = _launch(tmp_path, env={"FAKE_CREDITS": "12.50"})   # Rs 1,199.88 < Rs 1,200
    assert p.returncode == 1 and "Rs 0, plus this block's cap" in p.stderr and _no_create(calls)


def test_a_malformed_record_line_refuses(tmp_path):
    rec = _record(tmp_path)
    rec.parent.mkdir()
    rec.write_text("2026-11-23T00:00:00Z\tP3\ti-0bbb\tabout 500\n")
    p, calls = _launch(tmp_path)
    assert p.returncode == 1 and "has a line it cannot read" in p.stderr and _no_create(calls)


def test_each_window_is_recorded_and_only_an_instance_or_a_doubt_is_counted(tmp_path):
    """The researcher's launch yes of 24 Nov 2026, item 3: one record per
    window, a real line only once an instance id exists; a confirmed capacity
    refusal is not counted, anything else without an id is."""
    p, _ = _launch(tmp_path)
    assert p.returncode == 0, p.stderr
    p, _ = _launch(tmp_path, env={"FAKE_RUN_FAIL": "1", "AWS_COST_CAP_INR": "1300"})
    assert p.returncode == 1
    p, _ = _launch(tmp_path, env={"FAKE_RUN_ERROR_ZONES": "us-east-1a", "FAKE_RUN_ERROR": "Unsupported",
                                  "AWS_SESSION_AZ": "us-east-1a", "AWS_COST_CAP_INR": "1100"})
    assert p.returncode == 1
    lines = [ln.split("\t") for ln in _record(tmp_path).read_text().splitlines()]
    assert [ln[1:] for ln in lines] == [
        ["P3", "window-open:sess-1", "1200"], ["P3", INSTANCE, "1200"],
        ["P3", "window-closed:sess-1:launched", "1200"],
        ["P3", "window-open:sess-1", "1300"], ["P3", "window-closed:sess-1:refused", "1300"],
        ["P3", "window-open:sess-1", "1100"], ["P3", "unconfirmed", "1100"],
        ["P3", "window-closed:sess-1:stopped", "1100"]]
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", ln[0]) for ln in lines)
    p, _ = _launch(tmp_path, stdin="no\n")
    assert "> Rs 2300.00 planned so far + Rs 1200 cap" in p.stdout     # 1,200 + 1,100
    # and the record decides: Rs 2,300 + Rs 1,200 needs more than $36.46
    p, calls = _launch(tmp_path, env={"FAKE_CREDITS": "36.46"})
    assert p.returncode == 1 and _no_create(calls)


def _gate(tmp_path, record):
    rec = _record(tmp_path)
    rec.parent.mkdir(exist_ok=True)
    rec.write_text(record)
    return _launch(tmp_path, stdin="no\n")


def test_a_window_left_open_is_counted(tmp_path):
    p, _ = _gate(tmp_path, "2026-11-24T10:00:00Z\tREH\twindow-open:s1\t220\n"
                           "2026-11-24T11:00:00Z\tREH\twindow-open:s2\t220\n"
                           "2026-11-24T11:30:00Z\tREH\twindow-closed:s2:refused\t220\n")
    assert "> Rs 220.00 planned so far" in p.stdout, p.stdout + p.stderr


def test_a_dated_void_line_uncounts_the_line_it_names(tmp_path):
    rec = ("2026-11-23T18:03:26Z\tREH\tno-id\t180\n"
           "2026-11-23T18:13:54Z\tREH\tno-id\t180\n"
           "2026-11-23T10:51:17Z\tREH\ti-09462143e8347fb80\t180\n"
           "VOID\t2026-11-23T18:03:26Z\tREH\tno-id\t2026-11-24T17:00:00Z\tcapacity refusal, nothing created\n")
    p, _ = _gate(tmp_path, rec)
    assert "> Rs 360.00 planned so far" in p.stdout, p.stdout + p.stderr


@pytest.mark.parametrize("void", [
    "VOID\t2026-11-23T18:03:27Z\tREH\tno-id\t2026-11-24T17:00:00Z\tnames no line",
    "VOID\t2026-11-23T18:03:26Z\tREH\tno-id\t2026-11-24T17:00:00Z\t",
    "VOID\t2026-11-23T18:03:26Z\tREH\tno-id\t24 Nov\tnot dated in UTC",
    "VOID\t2026-11-23T18:03:26Z\tREH\tno-id\t2026-11-24T17:00:00Z",
])
def test_a_void_line_that_is_malformed_or_names_no_earlier_line_refuses(tmp_path, void):
    p, calls = _gate(tmp_path, "2026-11-23T18:03:26Z\tREH\tno-id\t180\n" + void + "\n")
    assert p.returncode == 1 and "has a line it cannot read" in p.stderr, p.stderr
    assert "(line 2)" in p.stderr
    assert _no_create(calls)


def test_a_void_line_before_the_line_it_names_refuses(tmp_path):
    p, _ = _gate(tmp_path, "VOID\t2026-11-23T18:03:26Z\tREH\tno-id\t2026-11-24T17:00:00Z\tearly\n"
                           "2026-11-23T18:03:26Z\tREH\tno-id\t180\n")
    assert p.returncode == 1 and "(line 1)" in p.stderr


def test_an_unknown_record_word_refuses(tmp_path):
    p, _ = _gate(tmp_path, "2026-11-23T18:03:26Z\tREH\trefused\t180\n")
    assert p.returncode == 1 and "cannot read" in p.stderr


def test_an_abort_or_a_refusal_records_nothing(tmp_path):
    _launch(tmp_path, stdin="no\n")
    _launch(tmp_path, env={"FAKE_PLAN_TYPE": "FREE"})
    assert not _record(tmp_path).exists()


def test_the_launcher_notes_list_the_allowed_services():
    text = (scripts_copy.__globals__["SCRIPTS"] / "aws_launch_session.sh").read_text()
    for said in ("AWS services allowed", "EC2,", "EBS volumes, EBS snapshots and AMIs", "networking",
                 "IAM (the hard-delete role)", "EventBridge Scheduler", "Service Quotas",
                 "no S3, Lambda, Bedrock, RDS, DynamoDB", "stop before P3"):
        assert said in text, said


# --- the hard-delete role (adviser's rulings dated 22 Nov 2026, role fix) ----

GOOD_TRUST = ("1\\tAllow\\tscheduler.amazonaws.com\\tsts:AssumeRole\\tStringEquals\\t"
              "aws:SourceAccount,aws:SourceArn\\t{arn}\\t111122223333")


def _dry(calls):
    return [c for c in aws_calls(calls) if "attnbench-dryreg-" in c]


def test_the_trust_policy_is_read_before_the_prompt(tmp_path):
    p, calls = _launch(tmp_path, stdin="no\n")
    assert p.returncode == 1
    get = _one(aws_calls(calls), "iam get-role")
    assert "--profile pinned-profile" in get and "--role-name attnbench-hard-delete" in get
    assert 'Condition.StringEquals."aws:SourceArn"' in get
    assert "role trust   : attnbench-hard-delete, aws:SourceArn = schedule-group/default" in p.stdout
    assert _dry(calls) == []                      # an abort registers nothing


@pytest.mark.parametrize("env,why", [
    # the role as made on 21 Nov 2026: the condition named the schedule
    ({"FAKE_TRUST_ARN": "arn:aws:scheduler:us-east-1:111122223333:schedule/default/attnbench-hard-delete-*"},
     "schedule/default/attnbench-hard-delete-*"),
    ({"FAKE_TRUST": "1\\tAllow\\tscheduler.amazonaws.com\\tsts:AssumeRole\\tArnLike,StringEquals\\t"
                    "aws:SourceAccount\\tNone\\t111122223333"}, "ArnLike"),
    ({"FAKE_TRUST_ARN": "arn:aws:scheduler:us-west-2:111122223333:schedule-group/default"}, "us-west-2"),
    ({"FAKE_TRUST_ARN": "arn:aws:scheduler:us-east-1:444455556666:schedule-group/default"}, "444455556666"),
    ({"FAKE_TRUST": GOOD_TRUST.format(arn="x").replace("1\\t", "2\\t", 1)}, "read: '2"),
    ({"FAKE_TRUST": GOOD_TRUST.format(arn="arn:aws:scheduler:us-east-1:111122223333:schedule-group/default")
                    .replace("scheduler.amazonaws.com", "ec2.amazonaws.com")}, "ec2.amazonaws.com"),
    ({"FAKE_ROLE_FAIL": "1"}, "read: ''"),
])
def test_a_trust_policy_not_in_the_documented_form_is_refused(tmp_path, env, why):
    p, calls = _launch(tmp_path, env=env)
    assert p.returncode == 1 and "not the documented form" in p.stderr, p.stderr
    assert why in p.stderr
    assert "111122223333" not in p.stdout + p.stderr
    assert _no_create(calls) and not _record(tmp_path).exists()


def test_the_dry_registration_comes_after_the_prompt_and_before_the_create(tmp_path):
    p, calls = _launch(tmp_path)
    assert p.returncode == 0, p.stderr
    calls = aws_calls(calls)
    create, get, delete = _dry(calls)
    assert create.startswith("scheduler create-schedule") and "--state DISABLED" in create
    thirty = dt.datetime.fromtimestamp(NOW + 30 * 86400, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    assert f"--schedule-expression at({thirty})" in create
    assert '\\"InstanceIds\\":[\\"i-00000000000000000\\"]' in create
    assert ":role/attnbench-hard-delete" in create and "--region us-east-1" in create
    assert get.startswith("scheduler get-schedule") and delete.startswith("scheduler delete-schedule")
    name = re.search(r"--name (attnbench-dryreg-\S+)", create).group(1)
    assert f"--name {name}" in get and f"--name {name}" in delete
    order = [k for k, c in enumerate(calls) if c in (create, delete) or "run-instances" in c]
    assert [calls[k].split()[1] for k in order] == ["create-schedule", "delete-schedule", "run-instances"]
    assert "Dry registration accepted" in p.stdout


@pytest.mark.parametrize("env,said", [
    ({"FAKE_DRYREG_FAIL": "1"}, "refused the dry registration"),
    ({"FAKE_DRYREG_DELETE_FAIL": "1"}, "could not be deleted"),
    ({"FAKE_DRYREG_STATE": "ENABLED"}, "read back as 'ENABLED', not DISABLED"),
])
def test_a_failed_dry_registration_creates_no_instance(tmp_path, env, said):
    p, calls = _launch(tmp_path, env=env)
    assert p.returncode == 1 and said in p.stderr, p.stderr
    assert not any("run-instances" in c for c in aws_calls(calls))
    assert not _record(tmp_path).exists()
    if "FAKE_DRYREG_DELETE_FAIL" not in env:     # whatever the read, it is deleted
        assert any(c.startswith("scheduler delete-schedule") for c in _dry(calls)) or "FAKE_DRYREG_FAIL" in env


# --- the image probe (adviser's rulings dated 22 Nov 2026, probe) ------------
PROBE_CARD = {**ENV_CARD, "AWS_COST_CAP_INR": "15", "AWS_HALT_MINUTES": "6",
              "AWS_MAX_RUN_MINUTES": "8"}


def test_the_probe_cards_settings(tmp_path):
    """One m7a.xlarge, halt at 6 min, hard delete at 8, cap Rs 15: worst case
    Rs 34.90 x 8 min = Rs 4.65, rounded up to Rs 5."""
    p, calls = _launch(tmp_path, "PROBE", env=PROBE_CARD)
    assert p.returncode == 0, p.stderr
    assert "worst case   : Rs 5, cost cap Rs 15" in p.stdout
    calls = aws_calls(calls)
    assert "--instance-type m7a.xlarge" in _one(calls, "ec2 run-instances")
    stamp = dt.datetime.fromtimestamp(NOW + 8 * 60, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    assert f"--schedule-expression at({stamp})" in _one(calls, f"--name attnbench-hard-delete-{INSTANCE} --schedule-expression")
    assert "+6 min from boot" in p.stdout


REH_CARD = {**ENV_CARD, "AWS_RATE_INR_HR": "143.21", "AWS_COST_CAP_INR": "220",
            "AWS_HALT_MINUTES": "75", "AWS_MAX_RUN_MINUTES": "90"}


def test_the_rehearsal_cards_settings(tmp_path):
    """docs/launch_cards/aws_rehearsal_g6.md, re-priced for g6.2xlarge (the
    researcher's launch yes of 24 Nov 2026): halt at 75 min, hard delete at
    90, cap Rs 220: worst case Rs 143.21 x 90 min = Rs 214.82, rounded up to
    Rs 215, under the yes's Rs 250."""
    p, calls = _launch(tmp_path, "REH", env=REH_CARD)
    assert p.returncode == 0, p.stderr
    assert "worst case   : Rs 215, cost cap Rs 220" in p.stdout
    calls = aws_calls(calls)
    create = _one(calls, "ec2 run-instances")
    assert "--instance-type g6.2xlarge" in create and "AvailabilityZone=ap-southeast-2a" in create
    assert "Key=attnbench-pairing,Value=REH" in create
    stamp = dt.datetime.fromtimestamp(NOW + 90 * 60, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    assert f"--schedule-expression at({stamp})" in _one(calls, f"--name attnbench-hard-delete-{INSTANCE} --schedule-expression")
    assert "+75 min from boot" in p.stdout
    card = (Path(__file__).resolve().parents[1] / "docs" / "launch_cards"
            / "aws_rehearsal_g6.md").read_text()
    for said in ("**₹143.21/h**", "**Cap ₹220**", "**75 min**", "**90 min**", "`g6.2xlarge`"):
        assert said in card, said
    assert int("220") <= 250


def test_a_cap_short_of_the_rehearsals_worst_case_is_refused(tmp_path):
    p, calls = _launch(tmp_path, "REH", env={**REH_CARD, "AWS_COST_CAP_INR": "214"})
    assert p.returncode == 1 and "exceeds the cost cap" in p.stderr and calls == []


@pytest.mark.parametrize("pairing,minutes,halt,ok", [
    ("PROBE", "4", "3", False), ("PROBE", "16", "6", False), ("PROBE", "8", "2", False),
    ("PROBE", "8", "8", False), ("PROBE", "5", "3", True), ("PROBE", "15", "13", True),
    ("ENV", "8", "6", False),    # the short limits are the probe's alone
])
def test_the_short_limits_are_the_probes_alone(tmp_path, pairing, minutes, halt, ok):
    p, calls = _launch(tmp_path, pairing, env={**PROBE_CARD, "AWS_MAX_RUN_MINUTES": minutes,
                                               "AWS_HALT_MINUTES": halt})
    assert (p.returncode == 0) == ok, p.stderr
    if not ok:
        assert calls == []


# --- the zone fallback (the researcher's launch yes of 23 Nov 2026, third
# rehearsal attempt: 2b, then 2c, then 2a, on a no-capacity error only) ------
ZONES = "ap-southeast-2b ap-southeast-2c ap-southeast-2a"
REH_ZONES = {**{k: v for k, v in REH_CARD.items() if k != "AWS_SESSION_AZ"},
             "AWS_SESSION_AZS": ZONES}


def _zones_tried(calls):
    return [re.search(r"AvailabilityZone=(\S+)", c).group(1)
            for c in aws_calls(calls) if c.startswith("ec2 run-instances")]


def _confirm_reads(calls):
    return [k for k, c in enumerate(aws_calls(calls)) if "Name=tag:attnbench-session," in c]


def test_two_capacity_refusals_then_a_launch_in_the_third_zone(tmp_path):
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES,
                       "FAKE_RUN_REFUSE_ZONES": "ap-southeast-2b ap-southeast-2c"})
    assert p.returncode == 0, p.stderr
    assert _zones_tried(calls) == ["ap-southeast-2b", "ap-southeast-2c", "ap-southeast-2a"]
    c = aws_calls(calls)
    runs = [k for k, x in enumerate(c) if x.startswith("ec2 run-instances")]
    reads = _confirm_reads(calls)
    # an instance and a volume read after each refusal, before the next request
    assert len(reads) == 4 and runs[0] < reads[0] < reads[1] < runs[1] < reads[2] < reads[3] < runs[2]
    assert [c[k].split()[1] for k in reads] == ["describe-instances", "describe-volumes"] * 2
    assert all("--region ap-southeast-2" in c[k] for k in reads)
    assert "LAUNCH_ZONE=ap-southeast-2a" in p.stdout
    assert "ZONES_REFUSED=ap-southeast-2b,ap-southeast-2c" in p.stdout
    assert f"INSTANCE_ID={INSTANCE}" in p.stdout
    sched = _one(c, f"--name attnbench-hard-delete-{INSTANCE} --schedule-expression")
    assert c.index(sched) > runs[2]
    rec = [ln.split("\t")[1:] for ln in _record(tmp_path).read_text().splitlines()]
    assert rec == [["REH", "window-open:sess-1", "220"], ["REH", INSTANCE, "220"],
                   ["REH", "window-closed:sess-1:launched", "220"]]
    assert "ap-southeast-2b ap-southeast-2c ap-southeast-2a (in this order" in p.stdout


def test_three_capacity_refusals_stop_after_one_pass(tmp_path):
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES, "FAKE_RUN_REFUSE_ZONES": ZONES})
    assert p.returncode == 1
    assert _zones_tried(calls) == ["ap-southeast-2b", "ap-southeast-2c", "ap-southeast-2a"]
    assert len(_confirm_reads(calls)) == 6
    assert "every zone refused for capacity: ap-southeast-2b ap-southeast-2c ap-southeast-2a" in p.stderr
    assert "LAUNCH_RESULT=EVERY_ZONE_REFUSED_FOR_CAPACITY" in p.stdout
    assert "aws_cleanup_check.sh" in p.stderr and "INSTANCE_ID=" not in p.stdout
    assert not any("create-schedule" in c and "hard-delete-i-" in c for c in aws_calls(calls))
    assert [ln.split("\t")[2] for ln in _record(tmp_path).read_text().splitlines()] == [
        "window-open:sess-1", "window-closed:sess-1:refused"]


@pytest.mark.parametrize("error", ["InvalidParameterValue", "Unsupported", "UnauthorizedOperation",
                                   "InsufficientInstanceCapacityX"])
def test_a_non_capacity_error_stops_at_once(tmp_path, error):
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES, "FAKE_RUN_ERROR_ZONES": "ap-southeast-2b",
                                             "FAKE_RUN_ERROR": error})
    assert p.returncode == 1
    assert _zones_tried(calls) == ["ap-southeast-2b"]
    assert _confirm_reads(calls) == []
    assert "an error other than" in p.stderr and "no other zone is tried" in p.stderr
    assert "LAUNCH_RESULT" not in p.stdout


@pytest.mark.parametrize("env,said", [
    ({"FAKE_STRAY_INSTANCE": "i-0aaaabbbbccccdddd"}, "the session has an instance"),
    ({"FAKE_STRAY_VOLUME": "vol-0aaaabbbbccccdddd"}, "the session has a volume"),
    ({"FAKE_LIST_FAIL": "1"}, "could not list the session's instances"),
])
def test_a_refusal_that_left_something_or_cannot_be_confirmed_stops(tmp_path, env, said):
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES, **env,
                                             "FAKE_RUN_REFUSE_ZONES": "ap-southeast-2b"})
    assert p.returncode == 1 and said in p.stderr, p.stderr
    assert _zones_tried(calls) == ["ap-southeast-2b"]
    assert "could not be confirmed" in p.stderr


def test_the_policy_is_checked_again_before_each_zone(tmp_path):
    """Three reads pass (the start, after the prompt, before 2b); the fourth,
    before 2c, is the day after the last launch date: no request in 2c."""
    seq = tmp_path / "today"
    seq.write_text("2026-12-31\n2026-12-31\n2026-12-31\n2027-01-01\n")
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES, "FAKE_TODAY_SEQ": str(seq),
                                             "FAKE_RUN_REFUSE_ZONES": ZONES})
    assert p.returncode == 1 and "after the last AWS" in p.stderr
    assert _zones_tried(calls) == ["ap-southeast-2b"]


@pytest.mark.parametrize("env,said", [
    ({"AWS_SESSION_AZS": "ap-southeast-2b ap-southeast-2b"}, "appears twice"),
    ({"AWS_SESSION_AZS": "ap-southeast-2b us-east-1a"}, "not an availability zone of ap-southeast-2"),
    ({"AWS_SESSION_AZS": "ap-southeast-2-per-1a"}, "Local Zone"),
    ({"AWS_SESSION_AZS": "ap-southeast-4a"}, "other than"),
    ({"AWS_SESSION_AZS": ZONES, "AWS_SESSION_AZ": "ap-southeast-2a"}, "not both"),
    ({"AWS_SESSION_AZS": ZONES, "AWS_SUBNET_ID": "subnet-0123456789abcdef0"}, "not both"),
])
def test_a_bad_zone_list_is_refused_before_any_aws_call(tmp_path, env, said):
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES, **env})
    assert p.returncode == 1 and said in p.stderr, p.stderr
    assert calls == []


def test_every_zone_in_the_list_must_offer_the_type(tmp_path):
    p, calls = _launch(tmp_path, "REH", env={**REH_ZONES, "FAKE_NOT_OFFERED": "1"})
    assert p.returncode == 1 and "not offered in ap-southeast-2b" in p.stderr
    assert not any("run-instances" in c for c in calls)
    p, calls = _launch(tmp_path, "REH", env=REH_ZONES)     # the refusal above created nothing
    offers = [c for c in aws_calls(calls) if "describe-instance-type-offerings" in c]
    assert [re.search(r"Name=location,Values=(\S+)", c).group(1) for c in offers] == ZONES.split()


def test_a_launch_in_the_first_zone_tries_no_other(tmp_path):
    p, calls = _launch(tmp_path, "REH", env=REH_ZONES)
    assert p.returncode == 0, p.stderr
    assert _zones_tried(calls) == ["ap-southeast-2b"] and _confirm_reads(calls) == []
    assert "LAUNCH_ZONE=ap-southeast-2b" in p.stdout and "ZONES_REFUSED" not in p.stdout


def test_the_policy_script_names_the_capacity_error_once():
    from _aws_fakes import SCRIPTS
    text = (SCRIPTS / "_aws_launch_policy.sh").read_text()
    assert text.count('AWS_CAPACITY_ERROR="InsufficientInstanceCapacity"') == 1
    assert '"($AWS_CAPACITY_ERROR)"' in text


# --- the retry window (the researcher's launch yes of 24 Nov 2026, fourth
# rehearsal attempt: 2b, 2c and 2a again every 10 minutes for at most 3 hours,
# on a no-capacity error only; the security group re-checked before each) ----
def _window(tmp_path, minutes="180", **env):
    clock = tmp_path / "clock"
    clock.write_text(f"{NOW}\n")
    e = {**REH_ZONES, "AWS_LAUNCH_WINDOW_MINUTES": minutes, "FAKE_CLOCK": str(clock)}
    e.update(env)
    return _launch(tmp_path, "REH", env=e)


def _sleeps(calls):
    return [int(c.split()[1]) for c in calls if c.startswith("sleep ")]


def _sg_reads(calls):
    return [c for c in aws_calls(calls) if c.startswith("ec2 describe-security-groups")]


def _record_words(tmp_path):
    return [ln.split("\t")[2] for ln in _record(tmp_path).read_text().splitlines()]


def test_refusals_then_a_launch_inside_the_window(tmp_path):
    """Pass 1: all three refused; pass 2, ten minutes after the start: 2b and
    2c refused, 2a launches."""
    p, calls = _window(tmp_path, FAKE_RUN_REFUSE_FIRST="5")
    assert p.returncode == 0, p.stderr
    assert _zones_tried(calls) == ZONES.split() * 2
    assert _sleeps(calls) == [600]
    assert len(_confirm_reads(calls)) == 10            # an instance and a volume read per refusal
    assert len(_sg_reads(calls)) == 6 and sum(c.startswith("curl ") for c in calls) == 6
    c = aws_calls(calls)
    sg = [k for k, x in enumerate(c) if x.startswith("ec2 describe-security-groups")]
    runs = [k for k, x in enumerate(c) if x.startswith("ec2 run-instances")]
    assert all(s + 1 == r for s, r in zip(sg, runs))   # the check just before each request
    for said in ("LAUNCH_ZONE=ap-southeast-2a", "ZONES_REFUSED=ap-southeast-2b,ap-southeast-2c",
                 "WINDOW_PASSES=2", "CAPACITY_REFUSALS=5", f"INSTANCE_ID={INSTANCE}",
                 "retry window : 180 min, the zones again every 10 min"):
        assert said in p.stdout, said
    assert _record_words(tmp_path) == ["window-open:sess-1", INSTANCE, "window-closed:sess-1:launched"]


def test_the_window_ends_after_three_hours_of_refusals(tmp_path):
    """18 passes (0, 10, ..., 170 min), 54 refused requests, 17 waits; then
    the stop, with nothing counted in the credit gate."""
    p, calls = _window(tmp_path, FAKE_RUN_REFUSE_ZONES=ZONES)
    assert p.returncode == 1
    assert _zones_tried(calls) == ZONES.split() * 18
    assert _sleeps(calls) == [600] * 17
    assert "the retry window (180 min) ended" in p.stderr and "18 passes (54 refusals)" in p.stderr
    assert "LAUNCH_RESULT=EVERY_ZONE_REFUSED_FOR_CAPACITY" in p.stdout and "WINDOW_PASSES=18" in p.stdout
    assert "INSTANCE_ID=" not in p.stdout
    assert _record_words(tmp_path) == ["window-open:sess-1", "window-closed:sess-1:refused"]
    p, _ = _launch(tmp_path, stdin="no\n")
    assert "> Rs 0.00 planned so far" in p.stdout


def test_the_window_ends_by_the_clock_when_passes_run_long(tmp_path):
    """Each wait overruns by 20 minutes (a pass every 30 min): the window
    still ends at 180 min, after the pass begun at 150."""
    p, calls = _window(tmp_path, FAKE_RUN_REFUSE_ZONES=ZONES, FAKE_SLEEP_EXTRA="1200")
    assert p.returncode == 1
    assert len(_zones_tried(calls)) == 3 * 6 and "WINDOW_PASSES=6" in p.stdout


def test_the_window_is_bounded_at_three_hours(tmp_path):
    for bad in ("181", "240", "3h", "-1", "1.5"):
        p, calls = _window(tmp_path, minutes=bad)
        assert p.returncode == 1 and "AWS_LAUNCH_WINDOW_MINUTES" in p.stderr, bad
        assert calls == [], bad


def test_an_address_change_mid_window_stops_and_re_points_nothing(tmp_path):
    """Pass 1 refused; before 2b in pass 2 the Mac's address has changed: the
    window stops there, the group is not touched, and nothing is counted."""
    seq = tmp_path / "ips"
    seq.write_text(f"{MY_IP}\n" * 3 + "198.51.100.9\n")
    p, calls = _window(tmp_path, FAKE_RUN_REFUSE_ZONES=ZONES, FAKE_MY_IP_SEQ=str(seq))
    assert p.returncode == 1
    assert _zones_tried(calls) == ZONES.split()
    assert "no longer admits this Mac's current address" in p.stderr
    assert "It was not re-pointed" in p.stderr and "LAUNCH_RESULT=SSH_SOURCE_NOT_ADMITTED" in p.stdout
    assert not any(w in c for c in aws_calls(calls)
                   for w in ("authorize-security-group", "revoke-security-group", "modify-security-group"))
    assert MY_IP not in p.stdout + p.stderr and "198.51.100.9" not in p.stdout + p.stderr
    assert _record_words(tmp_path) == ["window-open:sess-1", "window-closed:sess-1:stopped"]


@pytest.mark.parametrize("env,said", [
    ({"FAKE_CURL_FAIL": "1"}, "cannot read this Mac's public address"),
    ({"FAKE_SG_FAIL": "1"}, "could not read the security group"),
    ({"FAKE_SG_CIDRS": "198.51.100.0/24"}, "no longer admits"),
    ({"FAKE_SG_CIDRS": "203.0.113.70/32"}, "no longer admits"),
])
def test_an_unadmitted_or_unreadable_source_stops_before_the_first_request(tmp_path, env, said):
    p, calls = _window(tmp_path, **env)
    assert p.returncode == 1 and said in p.stderr, p.stderr
    assert _zones_tried(calls) == []


def test_a_non_capacity_error_mid_window_stops_at_once_and_is_counted(tmp_path):
    """Pass 1 refused; the first request of pass 2 fails with another error:
    no other zone, no other pass, and the doubt is counted at the cap."""
    p, calls = _window(tmp_path, FAKE_RUN_REFUSE_FIRST="3", FAKE_RUN_ERROR_AT="4",
                       FAKE_RUN_ERROR="VcpuLimitExceeded")
    assert p.returncode == 1
    assert _zones_tried(calls) == ZONES.split() + ["ap-southeast-2b"]
    assert "an error other than" in p.stderr and "LAUNCH_RESULT" not in p.stdout
    assert _record_words(tmp_path) == ["window-open:sess-1", "unconfirmed", "window-closed:sess-1:stopped"]
    p, _ = _launch(tmp_path, stdin="no\n")
    assert "> Rs 220.00 planned so far" in p.stdout


def test_no_window_is_one_pass_and_never_waits(tmp_path):
    p, calls = _window(tmp_path, minutes="0", FAKE_RUN_REFUSE_ZONES=ZONES)
    assert p.returncode == 1 and _sleeps(calls) == [] and len(_zones_tried(calls)) == 3
    assert "retry window" not in p.stdout


def test_the_policy_script_fixes_the_interval_and_the_window_bound():
    from _aws_fakes import SCRIPTS
    text = (SCRIPTS / "_aws_launch_policy.sh").read_text()
    assert text.count("AWS_RETRY_INTERVAL_MINUTES=10\n") == 1
    assert text.count("AWS_RETRY_WINDOW_MAX_MINUTES=180\n") == 1
