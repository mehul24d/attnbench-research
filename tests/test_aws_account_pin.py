"""Account and profile pinning for the AWS scripts (task T5.1).

scripts/_aws_account.sh takes the account and profile from two variables
kept outside the repository, checks both the pinned profile and the identity
a hand-typed command would use against the pinned account, refuses on any
mismatch, never prints either value, and refuses an aws call that does not
carry the pinned profile. Run against a fake `aws`; nothing reaches AWS.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from _aws_fakes import (ACCOUNT, INSTANCE, OTHER_ACCOUNT, PROFILE, REGION, SCRIPTS,
                        aws_calls, launch_env, run, scripts_copy)

PINNED = ("aws_launch_session.sh", "aws_cleanup_check.sh", "aws_teardown_session.sh")


def _invoke(name, tmp_path, **kw):
    d = scripts_copy(tmp_path, "2026-12-31")
    args = {"aws_launch_session.sh": ["P3", "sess"],
            "aws_cleanup_check.sh": [],
            "aws_teardown_session.sh": [INSTANCE, REGION, str(tmp_path / "s")]}[name]
    extra = {**launch_env(), "FAKE_INSTANCE_REGION": REGION,
             "FAKE_SESSION_TAG": "sess", **kw.pop("extra", {})}
    return run(d / name, args, tmp_path, extra=extra, **kw)


@pytest.mark.parametrize("name", PINNED)
def test_an_unset_pin_refuses_before_any_aws_call(name, tmp_path):
    p, calls = _invoke(name, tmp_path, pin=False)
    assert p.returncode != 0
    assert "ATTNBENCH_AWS_ACCOUNT and ATTNBENCH_AWS_PROFILE" in p.stderr
    assert calls == []


@pytest.mark.parametrize("name", PINNED)
@pytest.mark.parametrize("bad", [{"ATTNBENCH_AWS_ACCOUNT": "12345"},
                                 {"ATTNBENCH_AWS_PROFILE": "bad profile;rm"}])
def test_a_malformed_pin_refuses_and_is_not_printed(name, bad, tmp_path):
    p, calls = _invoke(name, tmp_path, extra=bad)
    assert p.returncode != 0 and "REFUSING" in p.stderr
    value = next(iter(bad.values()))
    assert value not in p.stdout + p.stderr
    assert calls == []


@pytest.mark.parametrize("name", PINNED)
def test_a_profile_on_another_account_refuses_without_naming_either(name, tmp_path):
    p, calls = _invoke(name, tmp_path, extra={"FAKE_PROFILE_ACCOUNT": OTHER_ACCOUNT})
    assert p.returncode != 0
    assert "does not resolve to" in p.stderr
    for secret in (ACCOUNT, OTHER_ACCOUNT, PROFILE):
        assert secret not in p.stdout + p.stderr
    assert all(c.startswith("sts get-caller-identity") for c in aws_calls(calls)), calls


@pytest.mark.parametrize("name", PINNED)
def test_a_different_active_identity_refuses(name, tmp_path):
    """The hand-typed-command hazard: the profile is right, but AWS_PROFILE,
    the default profile or environment keys point elsewhere."""
    p, calls = _invoke(name, tmp_path, extra={"FAKE_ACTIVE_ACCOUNT": OTHER_ACCOUNT})
    assert p.returncode != 0
    assert "active AWS identity is not" in p.stderr
    for secret in (ACCOUNT, OTHER_ACCOUNT, PROFILE):
        assert secret not in p.stdout + p.stderr
    assert all(c.startswith("sts get-caller-identity") for c in aws_calls(calls)), calls


@pytest.mark.parametrize("name", PINNED)
def test_a_matching_pin_reaches_aws_with_the_profile_and_is_never_printed(name, tmp_path):
    p, calls = _invoke(name, tmp_path)
    calls = aws_calls(calls)
    assert len(calls) > 2, (p.stdout, p.stderr)
    # The first identity check uses the profile; the second, deliberately, not.
    assert calls[0].startswith("sts get-caller-identity --profile")
    assert calls[1].startswith("sts get-caller-identity") and "--profile" not in calls[1]
    # (The teardown ends by running the cleanup check, which pins again.)
    for c in calls[2:]:
        if c.startswith("sts get-caller-identity"):
            continue
        assert f"--profile {PROFILE}" in c, c
        assert "--region " in c, c
    # The fake puts the account in ARNs on stdout and stderr; both redacted.
    assert ACCOUNT not in p.stdout + p.stderr
    assert PROFILE not in p.stdout + p.stderr


def test_the_wrapper_refuses_a_call_without_the_pinned_profile(tmp_path):
    body = (f'source "{SCRIPTS}/_aws_account.sh"\n'
            "aws_account_pin\n"
            "aws ec2 describe-instances --region us-east-1; echo rc=$?\n"
            "aws ec2 describe-instances --profile someone-else --region us-east-1; echo rc=$?\n"
            'aws ec2 describe-instances --profile "$AWS_PIN_PROFILE" --region us-east-1; echo rc=$?\n')
    script = tmp_path / "w.sh"
    script.write_text(body)
    p, calls = run(script, [], tmp_path)
    assert re.findall(r"rc=(\d+)", p.stdout) == ["97", "97", "0"], (p.stdout, p.stderr)
    assert sum("describe-instances" in c for c in calls) == 1


def test_every_aws_call_in_the_scripts_names_the_profile():
    """Static check, so a new call site is caught even on a path the tests do
    not run: every `aws <service>` line passes --profile, apart from the two
    identity checks inside the pin itself (which call `command aws`)."""
    for name in PINNED:
        text = re.sub(r"\\\n\s*", " ", (SCRIPTS / name).read_text())
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#") or s.startswith("echo"):
                continue
            for m in re.finditer(r"(?:^|[\s(\"$])aws (ec2|scheduler|sts|s3)\b[^\n]*", s):
                assert "--profile" in m.group(0) or '"${P[@]}"' in m.group(0), (name, s)


def test_the_pin_file_is_never_executed_directly():
    p = subprocess.run(["bash", "-n", str(SCRIPTS / "_aws_account.sh")], capture_output=True)
    assert p.returncode == 0
    assert not (SCRIPTS / "_aws_account.sh").stat().st_mode & 0o111
