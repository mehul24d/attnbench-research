"""Project pinning for every launcher, the cleanup check and the teardown.

On 23 Oct 2026 the workstation's gcloud default project was switched to a
project of unrelated research. Every script here fell back to that default,
so on 25 Oct 2026 the cleanup check listed the wrong project and reported
this study's images as gone. The teardown deletes the instance it is named
in whatever project gcloud points at. scripts/_gcp_project.sh now takes the
project from ATTNBENCH_GCP_PROJECT (kept out of the repository), passes it to
every call, refuses while the active gcloud project differs, and never prints
the ID.

These tests never touch GCP: a fake `gcloud` is first on PATH, and the real
cloud CLIs are dropped from it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
HELPER = SCRIPTS / "_gcp_project.sh"

LAUNCHERS = ("gcp_launch_l4.sh", "gcp_launch_a100.sh", "gcp_launch_h100.sh",
             "gcp_launch_compile_session.sh", "gcp_boot_test_image.sh",
             "gcp_launch_fa3_build.sh")
# The six session scripts and the bootstrap were pinned on 25 Oct 2026, after
# the first seven (PR #14).
SESSION = ("gcp_deploy_source.sh", "gcp_preflight_instance.sh",
           "gcp_verify_gcs_writable.sh", "gcp_capture_image.sh", "gcp_status.sh",
           "gcp_session_elapsed.sh", "gcp_bootstrap_instance.sh",
           "gcp_watch_fa3_build.sh")
PINNED = LAUNCHERS + ("gcp_cleanup_check.sh", "gcp_teardown_session.sh") + SESSION

PIN = "pinned-proj-123"
OTHER = "other-proj-456"

FAKE_GCLOUD = r"""#!/usr/bin/env bash
args="$*"; printf '%s\n' "${args//$'\n'/ }" >> "$GCLOUD_LOG"
proj=""
for a in "$@"; do case "$a" in --project=*) proj="${a#--project=}";; esac; done
case "$*" in
  "config get-value project") echo "$FAKE_ACTIVE"; exit 0 ;;
  "auth list"*) echo "ACCOUNT-MARKER-NOT-TO-BE-PRINTED"; exit 0 ;;
  *"sourceInstanceProperties"*) printf 'g2-standard-8\tnvidia-l4\n'; exit 0 ;;
  *"machine-images describe"*) echo READY; exit 0 ;;
  *"instances list"*) exit 0 ;;
  *"instances describe"*"value(status)"*) echo RUNNING; exit 0 ;;
  *"instances describe"*) echo inst; exit 0 ;;
  *"compute ssh"*"uptime -s"*) echo "2026-10-25 06:35:47|1792910147"; exit 0 ;;
esac
# Resource URLs carry the project, as gcloud's do, on both streams.
echo "Created [https://www.googleapis.com/compute/v1/projects/$proj/zones/z/instances/i]"
echo "WARNING: request in projects/$proj" >&2
exit 0
"""

FAKE_DATE = """#!/bin/bash
if [[ "$*" == "+%F" ]]; then echo 2026-10-25; else exec /bin/date "$@"; fi
"""

ARGS = {
    "gcp_launch_l4.sh": ["inst"],
    "gcp_launch_a100.sh": ["inst"],
    "gcp_launch_h100.sh": ["inst"],
    "gcp_launch_compile_session.sh": ["inst"],
    "gcp_launch_fa3_build.sh": ["inst"],
    "gcp_boot_test_image.sh": ["some-image", "asia-southeast1-c"],
    "gcp_cleanup_check.sh": [],
    "gcp_teardown_session.sh": ["inst", "asia-northeast1-a"],
    "gcp_deploy_source.sh": ["inst", "zone"],
    "gcp_preflight_instance.sh": ["inst", "zone"],
    "gcp_verify_gcs_writable.sh": ["inst", "zone"],
    "gcp_capture_image.sh": ["inst", "zone", "some-image"],
    "gcp_status.sh": [],
    "gcp_session_elapsed.sh": ["inst", "zone", "100"],
    "gcp_bootstrap_instance.sh": ["inst", "zone"],
    "gcp_watch_fa3_build.sh": ["inst", "zone"],
}

# The bootstrap's local steps run Python; this stands in for it so the
# generic tests reach the instance steps without the RULER data present.
FAKE_PYTHON = r"""#!/usr/bin/env bash
case "$*" in
  *"len(r.PINNED_SHA256)"*) echo 3 ;;
  *"PINNED_VERSIONS"*) echo "nltk==3.9.1 wonderwords==2.2.0" ;;
  *) cat >/dev/null; echo "$FAKE_RULER_DIR" ;;
esac
"""


def _fixture_repo(tmp_path):
    """A clean git repository, because the deploy script refuses a dirty tree
    and resolves its repository from the working directory."""
    r = tmp_path / "repo"
    r.mkdir(exist_ok=True)
    git = ["git", "-C", str(r), "-c", "user.email=t", "-c", "user.name=t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    (r / "f.txt").write_text("x\n")
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    return r


def _run(name, tmp_path, *, pin=PIN, active=PIN, rate="297.47", script=None, extra=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for fname, body in (("gcloud", FAKE_GCLOUD), ("date", FAKE_DATE)):
        (bin_dir / fname).write_text(body)
        (bin_dir / fname).chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.write_text("")
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GCP_", "ATTNBENCH_", "FRONTIER_", "CLOUDSDK_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), HOME=str(tmp_path),
               GCLOUD_LOG=str(log), FAKE_ACTIVE=active,
               ATTNBENCH_NOT_FRONTIER="pytest", GCP_ZONE_FALLBACKS="zone-a")
    if pin is not None:
        env["ATTNBENCH_GCP_PROJECT"] = pin
    if rate is not None:
        env["GCP_RATE_INR_HR"] = rate
    env.update(extra or {})
    args = list(ARGS[name])
    if name == "gcp_teardown_session.sh":
        args.append(str(tmp_path / "session"))
    cwd = tmp_path
    if name == "gcp_deploy_source.sh":
        cwd = _fixture_repo(tmp_path)
    if name == "gcp_watch_fa3_build.sh":   # two quick polls, then it stops (exit 4)
        env.setdefault("FA3_WATCH_INTERVAL_S", "0")
        env.setdefault("FA3_WATCH_MAX_POLLS", "2")
    if name == "gcp_bootstrap_instance.sh":
        cwd = REPO
        fake_py = bin_dir / "fake_python"
        fake_py.write_text(FAKE_PYTHON)
        fake_py.chmod(0o755)
        env.setdefault("ATTNBENCH_PYTHON", str(fake_py))
        env.setdefault("FAKE_RULER_DIR", str(tmp_path / "ruler"))
    p = subprocess.run(["bash", str(script or SCRIPTS / name), *args], input="launch\n",
                       capture_output=True, text=True, env=env, cwd=cwd, timeout=120)
    return p, log.read_text().splitlines()


# --- static checks ----------------------------------------------------------

def gcloud_calls(text: str) -> list[str]:
    """Every gcloud invocation in a shell script, continuation lines joined,
    echo/printf and comment lines skipped. A call ends at an unbalanced `)`
    or at a pipe, `;` or `&&`, so `$(dirname "$0")` inside a call is kept."""
    joined = re.sub(r"\\\n\s*", " ", text)
    calls = []
    for line in joined.split("\n"):
        s = line.strip()
        if s.startswith("#") or re.match(r"^(echo|printf)\b", s):
            continue
        for m in re.finditer(r"(?<![\w-])gcloud (?!auth\b)[a-z]+\b", line):
            depth, end = 0, len(line)
            for i in range(m.start(), len(line)):
                ch = line[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        end = i
                        break
                    depth -= 1
                elif depth == 0 and (ch in "|;" or line.startswith("&&", i)):
                    end = i
                    break
            calls.append(line[m.start():end])
    return calls


def missing_project(text: str) -> list[str]:
    return [c for c in gcloud_calls(text)
            if "--project=" not in c and not c.startswith("gcloud config get-value project")]


def pin_problems(text: str) -> list[str]:
    problems = []
    if "_gcp_project.sh" not in text:
        problems.append("does not source _gcp_project.sh")
    pin = text.find("\ngcp_project_pin\n")
    if pin == -1:
        problems.append("never calls gcp_project_pin at top level")
    first = min((m.start() for m in re.finditer(r"^[^#\n]*\bgcloud [a-z]+\b", text, re.M)
                 if not re.match(r"\s*(echo|printf)\b", text[m.start():])), default=-1)
    if pin != -1 and first != -1 and first < pin:
        problems.append("calls gcloud before the pin")
    if re.search(r"GCP_PROJECT:-|gcloud config get-value project", text):
        problems.append("still falls back to the gcloud default")
    if re.search(r'^\s*(echo|printf)\b[^\n]*(?<!\\)\$\{?PROJECT\b', text, re.M):
        problems.append("prints the project ID")
    return problems


@pytest.mark.parametrize("name", PINNED)
def test_each_script_pins_before_its_first_gcloud_call(name):
    assert pin_problems((SCRIPTS / name).read_text()) == []


@pytest.mark.parametrize("name", PINNED)
def test_every_gcloud_call_passes_the_project(name):
    text = (SCRIPTS / name).read_text()
    assert gcloud_calls(text), f"{name}: no gcloud call found -- the parser is blind"
    assert missing_project(text) == []


def test_the_call_counts_are_not_vacuous():
    counts = {n: len(gcloud_calls((SCRIPTS / n).read_text())) for n in PINNED}
    assert counts["gcp_teardown_session.sh"] >= 9, counts
    assert counts["gcp_cleanup_check.sh"] >= 6, counts
    assert counts["gcp_boot_test_image.sh"] >= 5, counts


def test_every_launch_script_is_pinned():
    """A new launcher is covered by naming it here, or this fails."""
    creating = {p.name for p in SCRIPTS.glob("*.sh")
                if "gcloud compute instances create" in p.read_text()}
    assert creating <= set(LAUNCHERS), creating - set(LAUNCHERS)



def test_every_gcp_script_is_pinned():
    """CLAUDE.md says every scripts/gcp_*.sh takes the pinned project; a new
    one is covered by naming it in PINNED, or this fails."""
    present = {p.name for p in SCRIPTS.glob("gcp_*.sh")}
    assert present <= set(PINNED), present - set(PINNED)


# Break-tests for the static checks: each mutation must be noticed.

def test_the_checks_notice_a_missing_project_flag():
    text = (SCRIPTS / "gcp_teardown_session.sh").read_text()
    mutated = text.replace('delete "$NAME" --zone="$ZONE" --project="$PROJECT"',
                           'delete "$NAME" --zone="$ZONE"')
    assert mutated != text
    assert len(missing_project(mutated)) == 1


def test_the_checks_notice_a_missing_pin_and_a_restored_fallback():
    text = (SCRIPTS / "gcp_cleanup_check.sh").read_text()
    no_pin = text.replace("\ngcp_project_pin\n", "\n")
    assert "never calls gcp_project_pin at top level" in pin_problems(no_pin)
    fallback = text.replace(
        "gcp_project_pin\n",
        'gcp_project_pin\nPROJECT="${GCP_PROJECT:-$(gcloud config get-value project)}"\n')
    assert "still falls back to the gcloud default" in pin_problems(fallback)


def test_the_checks_notice_a_printed_project():
    text = (SCRIPTS / "gcp_launch_l4.sh").read_text()
    mutated = text.replace('echo "  project      : ATTNBENCH_GCP_PROJECT (pinned, not printed)"',
                           'echo "  project      : $PROJECT"')
    assert mutated != text
    assert "prints the project ID" in pin_problems(mutated)


# --- behaviour --------------------------------------------------------------

@pytest.mark.parametrize("name", PINNED)
def test_an_unset_pin_refuses_before_any_gcloud_call(name, tmp_path):
    p, calls = _run(name, tmp_path, pin=None)
    assert p.returncode != 0
    assert "ATTNBENCH_GCP_PROJECT is not set" in p.stderr, p.stderr[-1500:]
    assert calls == []


@pytest.mark.parametrize("name", PINNED)
def test_a_malformed_pin_refuses_and_is_not_printed(name, tmp_path):
    p, calls = _run(name, tmp_path, pin="Bad_Project!")
    assert p.returncode != 0
    assert "not a valid GCP project ID" in p.stderr
    assert "Bad_Project" not in p.stdout + p.stderr
    assert calls == []


@pytest.mark.parametrize("name", PINNED)
def test_a_different_active_project_refuses_without_naming_either(name, tmp_path):
    p, calls = _run(name, tmp_path, active=OTHER)
    assert p.returncode != 0
    assert "active gcloud project is not ATTNBENCH_GCP_PROJECT" in p.stderr
    assert calls == ["config get-value project"], calls
    assert PIN not in p.stdout + p.stderr and OTHER not in p.stdout + p.stderr


@pytest.mark.parametrize("name", PINNED)
def test_a_matching_project_reaches_gcloud_with_the_flag_and_is_never_printed(name, tmp_path):
    p, calls = _run(name, tmp_path)
    # `gcloud auth` is not project-scoped and takes no --project.
    work = [c for c in calls if c != "config get-value project" and not c.startswith("auth ")]
    assert work, f"{name} made no call past the pin: {p.stderr[-1500:]}"
    assert all(f"--project={PIN}" in c for c in work), [c for c in work if PIN not in c]
    # The fake echoes the project on both streams; none of it may get out.
    assert PIN not in p.stdout and PIN not in p.stderr, (p.stdout + p.stderr)[-1500:]
    if name not in QUIET:
        assert "projects/<project>" in p.stdout + p.stderr


# Scripts whose gcloud output under this fake is all captured or discarded,
# so no redacted text reaches the terminal to check.
QUIET = {"gcp_session_elapsed.sh", "gcp_capture_image.sh"}


ON_INSTANCE = ("run_phase.sh", "build_fa3_wheel.sh", "fa3_import_check.sh", "_fa3_pin.sh")


@pytest.mark.parametrize("name", PINNED + ON_INSTANCE)
def test_no_project_or_bucket_is_written_in_the_script(name):
    """The project ID was the default in gcp_session_elapsed.sh, and the
    bucket named after it was the default in run_phase.sh and
    gcp_verify_gcs_writable.sh. Both are now derived."""
    text = (SCRIPTS / name).read_text()
    assert not re.search(r"attnbench-results-[a-z0-9]", text), name
    assert not re.search(r"GCP_PROJECT:-[a-z]", text), name


def test_the_literal_check_notices_a_restored_bucket():
    text = (SCRIPTS / "run_phase.sh").read_text()
    mutated = text.replace('gs://attnbench-results-${PROJECT_ID}', 'gs://attnbench-results-some-project-1')
    assert mutated != text
    assert re.search(r"attnbench-results-[a-z0-9]", mutated)


def test_the_status_check_does_not_print_the_account(tmp_path):
    """The active account is an email address; gcp_status.sh says only
    whether one is active. The fake answers `auth list` with a marker."""
    p, calls = _run("gcp_status.sh", tmp_path)
    assert any(c.startswith("auth list") for c in calls)
    assert "ACCOUNT-MARKER" not in p.stdout + p.stderr
    assert "Active account: yes (not printed)" in p.stdout


def test_an_unpinned_variable_name_is_not_a_bypass(tmp_path):
    """GCP_PROJECT, which the scripts used to read, no longer selects a project."""
    p, calls = _run("gcp_cleanup_check.sh", tmp_path, pin=None,
                    extra={"GCP_PROJECT": PIN})
    assert p.returncode != 0 and calls == []
    assert "ATTNBENCH_GCP_PROJECT is not set" in p.stderr


# --- the declared rates -----------------------------------------------------

@pytest.mark.parametrize("rate", [None, "", "abc", "297.47/h"])
def test_the_a100_launcher_requires_a_declared_rate(rate, tmp_path):
    p, calls = _run("gcp_launch_a100.sh", tmp_path, rate=rate)
    assert p.returncode != 0
    assert "declare the planning rate as GCP_RATE_INR_HR" in p.stderr
    assert not any("instances create" in c for c in calls)


def test_the_a100_launcher_states_the_declared_rate(tmp_path):
    p, calls = _run("gcp_launch_a100.sh", tmp_path, rate="297.47")
    assert "declared rate: Rs 297.47/h" in p.stdout
    assert any("instances create" in c for c in calls)


@pytest.mark.parametrize("rate", [None, "", "eighty"])
def test_the_teardown_requires_a_rate_and_touches_nothing_without_one(rate, tmp_path):
    p, calls = _run("gcp_teardown_session.sh", tmp_path, rate=rate)
    assert p.returncode == 2
    assert "set GCP_RATE_INR_HR" in p.stderr
    assert calls == []


def test_the_teardown_costs_the_session_at_the_declared_rate(tmp_path):
    p, calls = _run("gcp_teardown_session.sh", tmp_path, rate="108.64")
    cost = (tmp_path / "session" / "session_cost.txt").read_text()
    assert "rate_inr_hr: 108.64" in cost, cost
    assert any(c.startswith("compute instances delete inst") for c in calls)


def test_the_teardown_has_no_default_rate():
    text = (SCRIPTS / "gcp_teardown_session.sh").read_text()
    assert not re.search(r"GCP_RATE_INR_HR:-[0-9]", text)
