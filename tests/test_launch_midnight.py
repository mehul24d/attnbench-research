"""A launch that crosses the last GCP launch date between two attempts stops
at the boundary.

**The launch definition** (plan T0.17, adopted 21 Oct 2026): every `gcloud
compute instances create` request, retries and the Spot fallback included,
must be submitted by the end of the last GCP launch date, India time. A
request that waits for capacity counts as launched when it is submitted.

A check made once at the top of a script does not enforce that. The A100
launcher tries FLEX_START and then Spot, and the compile session walks a zone
list; a first attempt at 23:59 on the last day followed by a second at 00:01
would launch on a day the plan forbids. So each of those scripts calls
`launch_policy_check` again before every attempt, and these tests make the
day change between two attempts.

**How midnight is crossed.** The fake `date` answers `+%F` with the last
launch date until the fake `gcloud` has recorded a create, and with the next
day after that. The first create is refused for capacity, which is what
sends each script to its second attempt. The control runs the same scripts
on a date that does not cross and requires the second attempt, so a single
attempt in the crossing test is the date's doing.

These tests never touch GCP: the real gcloud is not on the PATH they use.
"""

from __future__ import annotations

import csv
import datetime
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import subprocess

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
POLICY = SCRIPTS / "_launch_policy.sh"
HAS_FRONTIER_GATE = (SCRIPTS / "frontier_gate.sh").exists()

# Scripts with more than one create attempt, and what drives the second one.
RETRYING = ("gcp_launch_a100.sh", "gcp_launch_compile_session.sh")

FAKE_DATE = """#!/bin/bash
if [[ "$*" == "+%F" ]]; then
  if grep -q "instances create" "$GCLOUD_LOG" 2>/dev/null; then echo "$DAY_AFTER_FIRST"; else echo "$DAY_OF_FIRST"; fi
else
  exec /bin/date "$@"
fi
"""

# Logs every call, then hands it to the zone-retry test's fake gcloud, which
# knows the image calls the compile session makes and refuses creates in
# FAIL_ZONES for capacity. The A100's FLEX_START create is refused the same
# way through FAKE_REFUSE_MODEL.
WRAPPER = """#!/bin/bash
printf '%s\\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
  *"instances create"*"provisioning-model=${FAKE_REFUSE_MODEL:-none}"*)
    echo "ZONE_RESOURCE_POOL_EXHAUSTED" >&2; exit 1 ;;
esac
exec "$(dirname "$0")/gcloud_inner" "$@"
"""


def _inner_gcloud() -> str:
    spec = importlib.util.spec_from_file_location(
        "zone_retry", REPO / "tests" / "test_launch_script_zone_retry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAKE_GCLOUD


def _last_date() -> str:
    m = re.search(r'^LAST_GCP_LAUNCH_DATE="(\d{4}-\d{2}-\d{2})"$', POLICY.read_text(), re.M)
    assert m, "LAST_GCP_LAUNCH_DATE not found in the policy file"
    return m.group(1)


def _next(day: str) -> str:
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()


def _frontier_env(tmp_path: Path, name: str) -> dict:
    """On a branch with the frontier budget gate, point it at scratch files
    the way tests/test_frontier_gate_hardening.py does, so the run reaches the
    creates. Rows it writes go to scratch files, never to the study's."""
    if not HAS_FRONTIER_GATE:
        return {}
    sys.path.insert(0, str(REPO))
    from attnbench.analysis import frontier_prereg as fp
    ledger, state = tmp_path / "frontier_spend.csv", tmp_path / "frontier_plan_state.json"
    with ledger.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fp.LEDGER_COLUMNS).writeheader()
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    state.write_text(json.dumps({"months_remaining": 2, "months_confirmed_on": today}))
    overrides = tmp_path / "override_rows.csv"
    overrides.write_text(",".join(fp.LEDGER_COLUMNS) + "\n")
    env = dict(ATTNBENCH_PYTHON=sys.executable, FRONTIER_LEDGER=str(ledger),
               FRONTIER_STATE=str(state), FRONTIER_GATE_EXCEPTION="pytest",
               FRONTIER_OVERRIDE_LEDGER=str(overrides))
    if name == "gcp_launch_a100.sh":
        env["FRONTIER_SESSION"] = "I1"
    else:
        env["ATTNBENCH_NOT_FRONTIER"] = "pytest"
    return env


def _run(name: str, tmp_path: Path, day_of_first: str, day_after_first: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for fname, body in (("date", FAKE_DATE), ("gcloud", WRAPPER), ("gcloud_inner", _inner_gcloud())):
        (bin_dir / fname).write_text(body)
        (bin_dir / fname).chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.touch()
    attempts = tmp_path / "attempts.txt"
    attempts.touch()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FRONTIER_", "ATTNBENCH_NOT_FRONTIER", "GCP_", "FAKE_"))}
    env.update(
        PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path),
        GCLOUD_LOG=str(log), DAY_OF_FIRST=day_of_first, DAY_AFTER_FIRST=day_after_first,
        GCP_PROJECT="test-project", ATTNBENCH_GCP_PROJECT="test-project",
        GCP_RATE_INR_HR="297.47",
        # the compile session's zone walk: zone-a refuses for capacity
        ATTEMPT_LOG=str(attempts), FAIL_ZONES="zone-a", INVALID_ZONES="",
        GCP_ZONE_FALLBACKS="zone-a zone-b", FAKE_IMAGE_KIND="machine-image",
        FAKE_IMAGE_MACHINE_TYPE="g2-standard-8", FAKE_IMAGE_ACCELERATOR="nvidia-l4",
        # the A100's FLEX_START create refuses for capacity
        FAKE_REFUSE_MODEL="FLEX_START",
    )
    env.update(_frontier_env(tmp_path, name))
    proc = subprocess.run(["bash", str(SCRIPTS / name), "test-instance"], input="launch\n",
                          capture_output=True, text=True, env=env, timeout=180)
    creates = [c for c in log.read_text().splitlines() if "instances create" in c]
    return proc, creates


def test_every_retrying_script_is_covered():
    """Anti-vacuity: a script with a create inside a loop, or more than one
    create call, must be in RETRYING."""
    for p in sorted(SCRIPTS.glob("*.sh")):
        text = p.read_text()
        n = text.count("gcloud compute instances create")
        if n == 0:
            continue
        retry = n > 1 or "create_with" in text or re.search(
            r"^\s*for [^\n]*\n(?:(?!^done)[\s\S])*?gcloud compute instances create", text, re.M)
        assert bool(retry) == (p.name in RETRYING), p.name


@pytest.mark.parametrize("name", RETRYING)
def test_a_second_attempt_after_midnight_is_refused(name, tmp_path):
    last = _last_date()
    proc, creates = _run(name, tmp_path, last, _next(last))
    assert len(creates) == 1, (name, creates, proc.stderr[-2000:])
    assert proc.returncode != 0, name
    assert "is after the last GCP" in proc.stderr, (name, proc.stderr[-2000:])
    # The last launch date (9 Dec 2026) is after the credit's last day (30 Nov
    # 2026), so the first attempt was told it is outside the credit.
    assert "outside the GCP credit" in proc.stderr, (name, proc.stderr[-2000:])


def test_the_crossing_is_the_night_of_9_to_10_december_2026():
    """The dates these tests cross, pinned, so moving the boundary is seen
    here as well as in tests/test_launch_policy.py."""
    assert (_last_date(), _next(_last_date())) == ("2026-12-09", "2026-12-10")


@pytest.mark.parametrize("name", RETRYING)
def test_without_the_crossing_the_second_attempt_is_made(name, tmp_path):
    """Control: on a day that does not change, the script does make its
    second attempt, so the refusal above comes from the date."""
    last = _last_date()
    proc, creates = _run(name, tmp_path, last, last)
    assert len(creates) == 2, (name, creates, proc.stderr[-2000:])
    assert "REFUSING TO LAUNCH -- today" not in proc.stderr, (name, proc.stderr[-2000:])


# --- the request window on the flex-start creates ---------------------------

WINDOW = {"gcp_launch_a100.sh": "15m", "gcp_launch_h100.sh": "2h"}


def _seconds(duration: str) -> int:
    units = {"h": 3600, "m": 60, "s": 1}
    parts = re.findall(r"(\d+)([hms])", duration)
    assert parts and "".join(n + u for n, u in parts) == duration, duration
    return sum(int(n) * units[u] for n, u in parts)


@pytest.mark.parametrize("name", sorted(WINDOW))
def test_the_flex_start_request_window_is_a_named_constant_in_range(name):
    """Compute Engine accepts 90 s to 2 h for a request pinned to one zone."""
    m = re.search(r'^FLEX_REQUEST_VALID_FOR="([^"]+)"$', (SCRIPTS / name).read_text(), re.M)
    assert m and m.group(1) == WINDOW[name], name
    assert 90 <= _seconds(m.group(1)) <= 7200, name


@pytest.mark.parametrize("name", sorted(WINDOW))
def test_only_the_flex_start_create_carries_the_request_window(name, tmp_path):
    last = _last_date()
    proc, creates = _run(name, tmp_path, last, last)
    flex = [c for c in creates if "provisioning-model=FLEX_START" in c]
    other = [c for c in creates if "provisioning-model=FLEX_START" not in c]
    assert flex, (name, creates, proc.stderr[-2000:])
    for c in flex:
        assert f"--request-valid-for-duration={WINDOW[name]}" in c, c
    for c in other:
        assert "--request-valid-for-duration" not in c, c
    if name == "gcp_launch_a100.sh":
        assert len(other) == 1 and "provisioning-model=SPOT" in other[0], creates
