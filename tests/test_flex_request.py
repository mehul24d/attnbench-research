"""The H100 launch path (plan T6.4): submit a flex-start request, poll it, and
cancel it at a deadline, with no long gcloud client call.

`scripts/gcp_launch_h100.sh` submits with `--async` and
`scripts/_flex_request.sh` does the waiting. What must hold, and the test
that holds it:

* the request is cancelled at the client deadline, inside the server window
  (`test_a_request_with_no_capacity_is_cancelled_at_the_deadline`);
* a cancel is confirmed before another zone is tried, and an unconfirmed
  cancel stops the script (`..._before_the_next_zone`, `..._unconfirmed...`);
* the date is checked before each submit, and a wait that crosses midnight is
  not refused (`test_the_date_is_checked_before_each_submit...`);
* no gcloud call runs unbounded (`test_a_hanging_call_is_cut_off...`);
* an interrupt cancels a live request (`test_an_interrupt_cancels...`).

**How time is faked.** A real wait is two hours, so the tests run the real
scripts against a fake clock: the fake `date` answers `+%s` and the day from
a clock file, and the fake `sleep` adds its argument to that file and
returns. No test waits. The fake `gcloud` is a small state machine in a
scratch directory: `create` records an instance with the clock at that
moment, `describe` answers PENDING until a per-zone grant time and then
RUNNING, `delete` removes it. The real `gcloud` is not on the PATH these
tests use, and `tests/conftest.py` shims the rest.

These tests never touch GCP.
"""

from __future__ import annotations

import datetime
import importlib.util
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
LAUNCHER = SCRIPTS / "gcp_launch_h100.sh"
HELPER = SCRIPTS / "_flex_request.sh"
POLICY = SCRIPTS / "_launch_policy.sh"

# A fixed epoch for the fake clock, midday on 1 Dec 2026 India time, far from
# any boundary unless a test moves the boundary.
T0 = 1_796_000_000

FAKE_DATE = r"""#!/bin/bash
now=$(cat "$FAKE_DIR/clock")
case "$*" in
  "+%s") printf '%s\n' "${now%.*}" ;;
  "+%F")
    if [ -n "${MIDNIGHT_AT:-}" ] && [ "${now%.*}" -ge "$MIDNIGHT_AT" ]; then
      echo "$DAY_AFTER"
    else
      echo "$DAY_BEFORE"
    fi ;;
  *) exec /bin/date "$@" ;;
esac
"""

# Advances the fake clock instead of waiting. With FAKE_SLEEP_REAL set it also
# really sleeps that long, so a test can interrupt a script mid-wait.
FAKE_SLEEP = r"""#!/bin/bash
awk -v a="$(cat "$FAKE_DIR/clock")" -v b="$1" 'BEGIN{printf "%.1f\n", a+b}' > "$FAKE_DIR/clock.tmp"
mv "$FAKE_DIR/clock.tmp" "$FAKE_DIR/clock"
[ -n "${FAKE_SLEEP_REAL:-}" ] && exec /bin/sleep "$FAKE_SLEEP_REAL"
exit 0
"""

# The instance file holds the clock at submit. Per-zone behaviour comes from
# the environment, with the zone's letters upper-cased and dashes as
# underscores: GRANT_AFTER_<Z> seconds until RUNNING (unset = never),
# VISIBLE_AFTER_<Z> seconds before describe can see it, CREATE_<Z> =
# capacity | invalid | hang | hang_accepted, DELETE_<Z> = ignore, and
# DESCRIBE_<Z> = hang | error.
FAKE_GCLOUD = r"""#!/bin/bash
printf '%s\n' "$*" >> "$GCLOUD_LOG"
now=$(cat "$FAKE_DIR/clock")
zone=""; name=""
for a in "$@"; do
  case "$a" in --zone=*) zone="${a#--zone=}" ;; esac
done
case "$*" in
  *"compute instances"*) name="$4" ;;
esac
key=$(printf '%s' "$zone" | tr 'a-z-' 'A-Z_')
get() { eval "printf '%s' \"\${$1_$key:-}\""; }
inst="$FAKE_DIR/inst_$zone"
notfound() { echo "ERROR: (gcloud.compute.instances.describe) Could not fetch resource: - The resource 'projects/$ATTNBENCH_GCP_PROJECT/zones/$zone/instances/$name' was not found" >&2; exit 1; }
case "$*" in
  "config get-value project") echo "$ATTNBENCH_GCP_PROJECT"; exit 0 ;;
  *"instances create"*)
    case "$(get CREATE)" in
      capacity) echo "ZONE_RESOURCE_POOL_EXHAUSTED" >&2; exit 1 ;;
      invalid)  echo "ERROR: Invalid value for field 'resource.machineType'" >&2; exit 1 ;;
      hang)     exec /bin/sleep 100000 ;;
      hang_accepted) echo "${now%.*}" > "$inst"; exec /bin/sleep 100000 ;;
    esac
    echo "${now%.*}" > "$inst"
    echo "Instance creation in progress"; exit 0 ;;
  *"instances describe"*)
    case "$(get DESCRIBE)" in
      hang)  exec /bin/sleep 100000 ;;
      error) echo "ERROR: connection reset" >&2; exit 1 ;;
    esac
    [ -f "$inst" ] || notfound
    start=$(cat "$inst"); el=$(( ${now%.*} - start ))
    window="$(get EXPIRE_AT)"; window="${window:-7200}"
    if [ "$el" -ge "$window" ]; then rm -f "$inst"; notfound; fi
    vis="$(get VISIBLE_AFTER)"
    if [ -n "$vis" ] && [ "$el" -lt "$vis" ]; then notfound; fi
    g="$(get GRANT_AFTER)"
    if [ -n "$g" ] && [ "$el" -ge "$g" ]; then echo "${FAKE_GRANTED_STATUS:-RUNNING}"; else echo PENDING; fi
    exit 0 ;;
  *"instances delete"*)
    [ "$(get DELETE)" = ignore ] || rm -f "$inst"
    exit 0 ;;
  *"operations list"*)
    printf '%s\n' "${OPS_REASON:-}"; exit 0 ;;
esac
exit 0
"""


def _policy_last_date() -> str:
    m = re.search(r'^LAST_GCP_LAUNCH_DATE="(\d{4}-\d{2}-\d{2})"$', POLICY.read_text(), re.M)
    assert m
    return m.group(1)


def _next(day: str) -> str:
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()


class Run:
    """One launcher run against the fakes."""

    def __init__(self, tmp_path: Path, **env):
        self.dir = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.fake = tmp_path / "fake"
        self.fake.mkdir()
        for name, body in (("gcloud", FAKE_GCLOUD), ("date", FAKE_DATE), ("sleep", FAKE_SLEEP)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        (self.fake / "clock").write_text(f"{T0}\n")
        self.log = tmp_path / "gcloud.log"
        self.log.touch()
        last = _policy_last_date()
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(tmp_path),
            "FAKE_DIR": str(self.fake), "GCLOUD_LOG": str(self.log),
            "ATTNBENCH_GCP_PROJECT": "test-project", "GCP_RATE_INR_HR": "462.42",
            "DAY_BEFORE": last, "DAY_AFTER": _next(last),
            # 5-minute polls keep a 2-hour wait to about 25 fake polls.
            "GCP_FLEX_POLL_SECONDS": "300",
        }
        self.env.update({k: str(v) for k, v in env.items()})

    def start(self, *args) -> subprocess.Popen:
        return subprocess.Popen(["bash", str(LAUNCHER), *(args or ("h100-test",))],
                                env=self.env, cwd=self.dir, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run(self, *args) -> subprocess.CompletedProcess:
        out, err = None, None
        p = self.start(*args)
        out, err = p.communicate(timeout=120)
        return subprocess.CompletedProcess(p.args, p.returncode, out, err)

    def calls(self, word: str) -> list[str]:
        return [l for l in self.log.read_text().splitlines() if f"instances {word}" in l]

    def clock(self) -> float:
        return float((self.fake / "clock").read_text())

    def all_calls(self) -> list[str]:
        return self.log.read_text().splitlines()


def _zone_flag(call: str) -> str:
    return re.search(r"--zone=(\S+)", call).group(1)


# --- the happy path and its shape -------------------------------------------

def test_a_granted_request_ends_the_script_and_is_not_cancelled(tmp_path):
    r = Run(tmp_path, GRANT_AFTER_US_CENTRAL1_A=1500)
    p = r.run()
    assert p.returncode == 0, p.stderr[-2000:]
    assert len(r.calls("create")) == 1 and not r.calls("delete")
    assert "Granted in us-central1-a" in p.stdout
    assert "PROVISIONING_MODEL=FLEX_START" in p.stdout
    waited = int(re.search(r"FLEX_WAIT_SECONDS=(\d+)", p.stdout).group(1))
    assert 1500 <= waited < 1500 + 300 + 5, waited


def test_the_submit_is_async_and_carries_the_pinned_flags(tmp_path):
    r = Run(tmp_path, GRANT_AFTER_US_CENTRAL1_A=0)
    r.run()
    (create,) = r.calls("create")
    for flag in ("--async", "--provisioning-model=FLEX_START", "--request-valid-for-duration=2h",
                 "--instance-termination-action=DELETE", "--max-run-duration=5h",
                 "--maintenance-policy=TERMINATE",
                 "--metadata=attnbench-provisioning-model=FLEX_START"):
        assert flag in create, (flag, create)
    assert _zone_flag(create) == "us-central1-a"


@pytest.mark.parametrize("status", ["PROVISIONING", "STAGING"])
def test_a_granted_vm_not_yet_running_ends_the_wait(tmp_path, status):
    r = Run(tmp_path, GRANT_AFTER_US_CENTRAL1_A=600, FAKE_GRANTED_STATUS=status)
    p = r.run()
    assert p.returncode == 0 and not r.calls("delete"), p.stderr[-1500:]
    assert f"VM status {status}" in p.stdout


def test_a_request_not_yet_visible_is_not_read_as_ended(tmp_path):
    """describe may say not-found for a moment after the async submit. That is
    not the request ending; the grace period covers it."""
    r = Run(tmp_path, VISIBLE_AFTER_US_CENTRAL1_A=60, GRANT_AFTER_US_CENTRAL1_A=900,
            GCP_FLEX_POLL_SECONDS=30)
    p = r.run()
    assert p.returncode == 0, p.stderr[-1500:]
    assert not r.calls("delete")


# --- cancel at the deadline -------------------------------------------------

def test_a_request_with_no_capacity_is_cancelled_at_the_deadline(tmp_path):
    r = Run(tmp_path)          # never granted
    p = r.run()
    assert p.returncode == 1, p.stderr[-2000:]
    deletes = r.calls("delete")
    assert deletes and all(_zone_flag(d) == "us-central1-a" for d in deletes)
    waited = r.clock() - T0
    # Cancelled at 6900 s (to one 300 s poll), and inside the 7200 s window.
    assert 6900 <= waited < 7200, waited
    assert "request cancelled" in p.stdout
    assert "Nothing is waiting and nothing is billing" in p.stderr


def test_the_cancel_deadline_is_inside_the_server_window():
    text = LAUNCHER.read_text()
    window = re.search(r'^FLEX_REQUEST_VALID_FOR="([^"]+)"$', text, re.M).group(1)
    secs = int(re.search(r"^FLEX_REQUEST_VALID_FOR_SECONDS=(\d+)", text, re.M).group(1))
    units = {"h": 3600, "m": 60, "s": 1}
    assert sum(int(n) * units[u] for n, u in re.findall(r"(\d+)([hms])", window)) == secs
    default = int(re.search(r'CANCEL_AFTER_SECONDS="\$\{GCP_CANCEL_AFTER_SECONDS:-(\d+)\}"', text).group(1))
    assert 120 <= default < secs and secs - default >= 300


@pytest.mark.parametrize("value", ["119", "7200", "9000", "soon", "", "-5", "6900.5"])
def test_a_bad_cancel_deadline_is_refused_before_any_call(tmp_path, value):
    r = Run(tmp_path, GCP_CANCEL_AFTER_SECONDS=value)
    p = r.run()
    if value == "":          # unset-like: falls back to the default and runs
        assert "GCP_CANCEL_AFTER_SECONDS must be" not in p.stderr
        return
    assert p.returncode == 1 and "GCP_CANCEL_AFTER_SECONDS must be" in p.stderr, p.stderr[-800:]
    assert not r.calls("create"), r.all_calls()


def test_the_server_window_ending_first_is_handled(tmp_path):
    """The client is asleep or slow and Compute Engine deletes the request at
    its own window. The script reads that as the end, not as an error."""
    r = Run(tmp_path, EXPIRE_AT_US_CENTRAL1_A=900, OPS_REASON="DONE")
    p = r.run()
    assert p.returncode == 1
    assert "ended after" in p.stdout and "nothing billed" in p.stdout
    assert not r.calls("delete"), "nothing to delete: the request was already gone"


# --- one live request at a time ---------------------------------------------

def test_a_cancel_is_confirmed_before_the_next_zone(tmp_path):
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            GRANT_AFTER_ASIA_SOUTHEAST1_B=300)
    p = r.run()
    assert p.returncode == 0, p.stderr[-2000:]
    calls = r.all_calls()
    creates = [i for i, c in enumerate(calls) if "instances create" in c]
    assert [_zone_flag(calls[i]) for i in creates] == ["us-central1-a", "asia-southeast1-b"]
    first_delete = next(i for i, c in enumerate(calls) if "instances delete" in c)
    # a describe of the first zone after the delete, before the second create
    confirm = [i for i in range(first_delete + 1, creates[1])
               if "instances describe" in calls[i] and "us-central1-a" in calls[i]]
    assert first_delete < creates[1] and confirm, calls[first_delete:creates[1] + 1]
    assert "Granted in asia-southeast1-b" in p.stdout


def test_an_unconfirmed_cancel_stops_before_any_other_zone(tmp_path):
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            DELETE_US_CENTRAL1_A="ignore", GCP_FLEX_CONFIRM_SECONDS=120,
            EXPIRE_AT_US_CENTRAL1_A=999999)
    p = r.run()
    assert p.returncode == 3, (p.returncode, p.stderr[-2000:])
    assert len(r.calls("create")) == 1, r.calls("create")
    assert "COULD NOT CONFIRM" in p.stderr
    assert "gcloud compute instances delete h100-test --zone=us-central1-a" in p.stderr


def test_a_capacity_refusal_moves_to_the_next_zone_without_a_cancel(tmp_path):
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            CREATE_US_CENTRAL1_A="capacity", GRANT_AFTER_ASIA_SOUTHEAST1_B=0)
    p = r.run()
    assert p.returncode == 0, p.stderr[-1500:]
    assert len(r.calls("create")) == 2 and not r.calls("delete")


def test_a_rejected_request_does_not_walk_the_zones(tmp_path):
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            CREATE_US_CENTRAL1_A="invalid")
    p = r.run()
    assert p.returncode == 1 and "CONFIGURATION ERROR" in p.stderr
    assert len(r.calls("create")) == 1


def test_a_success_ends_the_zone_loop(tmp_path):
    """The break after a grant: a second GPU VM must never be requested."""
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b asia-southeast1-c",
            GRANT_AFTER_US_CENTRAL1_A=0, GRANT_AFTER_ASIA_SOUTHEAST1_B=0)
    p = r.run()
    assert p.returncode == 0 and len(r.calls("create")) == 1


def test_an_operation_error_that_is_not_capacity_stops_the_walk(tmp_path):
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            EXPIRE_AT_US_CENTRAL1_A=300,
            OPS_REASON="DONE\tINVALID_FIELD_VALUE\tbad machine type")
    p = r.run()
    assert p.returncode == 1 and "not about capacity or quota" in p.stderr
    assert len(r.calls("create")) == 1


def test_a_capacity_operation_error_continues_the_walk(tmp_path):
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            EXPIRE_AT_US_CENTRAL1_A=300, GRANT_AFTER_ASIA_SOUTHEAST1_B=0,
            OPS_REASON="DONE\tZONE_RESOURCE_POOL_EXHAUSTED\tno capacity")
    p = r.run()
    assert p.returncode == 0 and len(r.calls("create")) == 2, p.stderr[-1500:]


# --- the date ---------------------------------------------------------------

def test_the_date_is_checked_before_each_submit_and_a_wait_may_cross_midnight(tmp_path):
    """The first request waits through midnight on the last launch date and is
    cancelled the next day; the second zone's submit is then refused."""
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            MIDNIGHT_AT=T0 + 3000)
    p = r.run()
    assert p.returncode == 1, p.stderr[-2000:]
    assert len(r.calls("create")) == 1
    assert "is after the last GCP" in p.stderr
    # the refusal came after the cancel was confirmed, not with a request live
    assert r.calls("delete")
    assert not [c for c in r.all_calls() if "asia-southeast1-b" in c]


def test_a_request_waiting_across_midnight_is_not_refused(tmp_path):
    """Launched when submitted: the wait itself is not date-checked."""
    r = Run(tmp_path, MIDNIGHT_AT=T0 + 600, GRANT_AFTER_US_CENTRAL1_A=1800)
    p = r.run()
    assert p.returncode == 0, p.stderr[-2000:]
    assert "is after the last GCP" not in p.stderr


def test_without_the_crossing_the_second_zone_is_tried(tmp_path):
    """Control for the test above: same run, midnight never comes."""
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            GRANT_AFTER_ASIA_SOUTHEAST1_B=0)
    p = r.run()
    assert p.returncode == 0 and len(r.calls("create")) == 2


# --- no unbounded call ------------------------------------------------------

def test_a_hanging_status_call_is_cut_off_and_the_wait_ends(tmp_path):
    """describe never answers. Each call is killed at the client limit, six in
    a row end the wait, and a cancel is attempted. The script does not hang."""
    r = Run(tmp_path, DESCRIBE_US_CENTRAL1_A="hang", GCP_FLEX_CONFIRM_SECONDS=120,
            GCP_FLEX_CALL_LIMIT=5)
    p = r.run()
    assert p.returncode == 3, (p.returncode, p.stderr[-2000:])   # cancel could not be confirmed
    assert "status checks in a row could not be read" in p.stderr
    assert r.calls("delete")
    assert len(r.calls("create")) == 1


def test_a_hanging_submit_is_cut_off_and_its_request_cancelled(tmp_path):
    """The server accepted the request and the client call never returned."""
    r = Run(tmp_path, GCP_ZONE_FALLBACKS="us-central1-a asia-southeast1-b",
            CREATE_US_CENTRAL1_A="hang_accepted", GCP_FLEX_CREATE_LIMIT=5)
    p = r.run()
    assert p.returncode == 1 and "did not answer" in p.stderr, p.stderr[-1500:]
    assert r.calls("delete")
    assert len(r.calls("create")) == 1


def test_every_gcloud_call_in_the_launch_path_goes_through_the_bounded_helper():
    """Static: no bare gcloud call, no --async-less create."""
    for path in (LAUNCHER, HELPER):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or re.match(r"^(echo|printf)\b", s):
                continue
            for m in re.finditer(r"(?<![\w-])gcloud (compute|config)\b", line):
                before = line[:m.start()].rstrip()
                assert before.endswith(("flex_bounded \"$CREATE_LIMIT\"", "flex_bounded \"$FLEX_CALL_LIMIT\"")) \
                    or "gcloud config get-value project" in line, f"{path.name}:{n}: {line}"
    assert "--async" in LAUNCHER.read_text()


def test_the_helper_has_a_client_limit_on_every_call_by_default():
    text = HELPER.read_text()
    assert re.search(r'^FLEX_CALL_LIMIT="\$\{GCP_FLEX_CALL_LIMIT:-(\d+)\}"', text, re.M)
    assert int(re.search(r"GCP_FLEX_CALL_LIMIT:-(\d+)", text).group(1)) <= 120


# --- interruption -----------------------------------------------------------

@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_an_interrupt_cancels_a_live_request(tmp_path, sig):
    r = Run(tmp_path, FAKE_SLEEP_REAL="0.05")        # never granted; slow enough to catch
    p = r.start()
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if len([c for c in r.all_calls() if "instances describe" in c]) >= 3:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the script never reached its polling loop")
        os.kill(p.pid, sig)
        out, err = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert r.calls("delete"), r.all_calls()[-5:]
    assert "leaving with a request live: cancelling" in err
    assert not (r.fake / "inst_us-central1-a").exists()


def test_a_normal_exit_after_a_grant_does_not_delete_the_vm(tmp_path):
    r = Run(tmp_path, GRANT_AFTER_US_CENTRAL1_A=0)
    r.run()
    assert not r.calls("delete")


# --- the rest of the launch path --------------------------------------------

def test_the_rate_must_be_declared(tmp_path):
    r = Run(tmp_path, GCP_RATE_INR_HR="")
    p = r.run()
    assert p.returncode == 1 and "GCP_RATE_INR_HR" in p.stderr
    assert not r.all_calls()


def test_the_project_id_is_never_printed(tmp_path):
    r = Run(tmp_path, ATTNBENCH_GCP_PROJECT="secret-proj-123", VISIBLE_AFTER_US_CENTRAL1_A=60,
            GRANT_AFTER_US_CENTRAL1_A=900, GCP_FLEX_POLL_SECONDS=30)
    p = r.run()
    assert p.returncode == 0
    assert "secret-proj-123" not in p.stdout + p.stderr


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_helper_pins_the_project_and_leaves_the_date_alone():
    pin = _load("test_gcp_project_pin")
    assert pin.missing_project(HELPER.read_text()) == []
    wiring = _load("test_launch_policy_wiring")
    text = HELPER.read_text()
    for pattern, what in wiring.DATE_CHANGERS:
        assert not re.search(pattern, text, re.M), f"_flex_request.sh {what}"
    code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
    assert "launch_policy_check" not in code, "the date is the submitter's to check"
