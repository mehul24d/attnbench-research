"""scripts/aws_p3_driver.sh end to end on the fake remote (tests/_p3_fake_remote.py).

The full P3 flow: the launch settings handed to the launcher, the deploy, the
three blocks run by scripts/aws_p3_instance.sh through the real
scripts/t31_session.py on configs/t31/p3_session.json, each block's pull with
every sha256 checked before the next block, the prediction and the leg-3 plan
witnessed by the Mac through scripts/aws_p3_gsutil.py under the real
GsutilStore, the P6 gate read after block 2 (and block 3 run whatever it
reads), the teardown. Then the ways it stops: a network drop that never comes
back (the teardown at the pull deadline, the instance halting itself), one
that does, a failed step, a block 2 short of P1's 35 cells, the launch
windows and their deadline, and the credentials' end logged at each window's start.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

import _p3_fake_remote as F

pytestmark = pytest.mark.skipif(
    len(list(F.P1_HOST.glob("*/bare_prices.json"))) != 35,
    reason="needs P1's T3.2d host components (the 17 Nov 2026 download) on this Mac")

DEADLINE = 1796236200          # 2026-12-02T18:30:00Z


def _status(fake) -> str:
    return (F.rec_dir(fake) / "status.txt").read_text()


def _windows(fake) -> list[list[str]]:
    p = fake["run"] / "windows.tsv"
    return [ln.split("\t") for ln in p.read_text().splitlines()] if p.exists() else []


@pytest.mark.parametrize("scale,verdict", [("1.05", "skip (inconclusive band)"), ("1.5", "run")])
def test_the_full_flow_runs_every_block_whatever_the_gate_reads(tmp_path, scale, verdict):
    fake = F.build(tmp_path, FAKE_HOST_SCALE=scale)
    p = F.run_driver(fake)
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-3000:]
    assert _status(fake).split(" ", 1)[1].startswith("DONE OK all three blocks passed and pulled")
    rec = F.rec_dir(fake)
    # the launcher was given P3's card: halt +473, delete +503, the cap (Rs 1,353 in all less the
    # aborted session's Rs 143), rate, window, zones
    env = (fake["run"] / "launch_env.txt").read_text().splitlines()
    for want in ("AWS_HALT_MINUTES=473", "AWS_MAX_RUN_MINUTES=503", "AWS_COST_CAP_INR=1210",
                 "AWS_RATE_INR_HR=143.21", "AWS_LAUNCH_WINDOW_MINUTES=180",
                 "AWS_SESSION_AZS=ap-southeast-2b ap-southeast-2c ap-southeast-2a",
                 "AWS_SESSION_REGION=ap-southeast-2", "AWS_AMI_ID=ami-096c5dc30364c9041",
                 "typed=launch"):
        assert want in env, (want, env)
    # each block pulled and every file checked, in order, before the next block started
    order = [ln for ln in log.splitlines() if ln.split(" ")[1:2] in (["BLOCK"], ["PULLED"], ["GATE"])]
    kinds = [ln.split(" ", 2)[1] + " " + ln.split(" ", 3)[2] for ln in order]
    assert kinds == ["BLOCK b1", "PULLED b1:", "BLOCK b2", "PULLED b2:", "GATE after",
                     "BLOCK b3", "PULLED b3:"], kinds
    # no final re-pull of every block (26 Nov 2026), and the halt pushed back as each
    # pull starts and before the teardown, on the instance, by aws_p3_instance.sh
    assert "FINAL RE-PULL skipped" in log
    pushes = [ln.split("(", 1)[1].split(")", 1)[0] for ln in log.splitlines() if " HALT pushed back (" in ln]
    assert pushes == ["pull b1", "pull b2", "pull b3", "teardown"], pushes
    assert (fake["home"] / "p3" / "halt.log").read_text().count("P3_HALT_PUSH") == 4
    for b in ("b1", "b2", "b3"):
        check = (rec / "pull" / b / "check.txt").read_text().splitlines()
        assert check and all(ln.endswith(": OK") for ln in check), check
    s = rec / "session"
    assert sorted(d.name for d in s.iterdir() if d.is_dir()) == [
        "01_preflight", "02_link", "03_dense_select", "04_probe", "05_host_components",
        "06_device_components", "07_predict", "08_hold", "09_ground_truth", "10_verify",
        "11_postflight", "12_delay_injection"]
    m = json.loads((s / "manifest.json").read_text())
    last = {a["step"]: a["status"] for a in m["attempts"]}     # ground truth also records "started" first
    assert len(last) == 12 and set(last.values()) == {"passed"}, last
    # the gate: 35 cells, P6 only; block 3 ran whatever it read
    gate = json.loads((rec / "gate.json").read_text())
    assert gate["cells"] == 35 and gate["verdict"] == verdict
    assert abs(gate["d"] - (float(scale) - 1)) < 1e-9
    assert f"verdict={verdict}" in (rec / "gate.txt").read_text()
    # the store: the prediction, its sidecar and the leg-3 plan, witnessed with this Mac's clock
    wit = [ln.split("\t") for ln in (rec / "store_witness.tsv").read_text().splitlines()]
    names = [Path(w[1]).name for w in wit]
    assert names == ["prediction_P3_stub.json", "prediction_P3_stub.json.sha256", "leg3_plan.json"]
    receipt = json.loads((s / "07_predict" / "receipt.json").read_text())
    import datetime as dt
    t = max(dt.datetime.fromisoformat(w[0].replace("Z", "+00:00")) for w in wit[:2])
    assert dt.datetime.fromisoformat(receipt["synced_at"]) == t + dt.timedelta(seconds=1)
    assert (rec / "store" / wit[0][1]).read_bytes() == (s / "07_predict" / "prediction_P3_stub.json").read_bytes()
    # one teardown, after the last pull; one launch window, ended LAUNCHED
    td = (fake["run"] / "teardown.log").read_text().splitlines()
    assert td == [f"fake teardown: {F.INSTANCE} ap-southeast-2 {rec}/teardown_1  rate=143.21"]
    w = _windows(fake)
    assert [x[0] for x in w] == ["start", "end"] and w[1][3] == f"LAUNCHED {F.INSTANCE}"
    # the driver never touched the security group's rules
    calls = (fake["run"] / "aws_calls.log").read_text()
    assert "authorize-security-group" not in calls and "revoke-security-group" not in calls


def test_a_network_drop_that_never_ends_is_torn_down_at_the_pull_deadline(tmp_path):
    """SSH lost in block 2 for good, and this Mac's address changed: the driver
    says so on every poll, never re-points the group, pulls nothing more, and
    tears down at the pull deadline; block 1 is kept, checked. The instance,
    its block ended and no next one started, halts itself."""
    fake = F.build(tmp_path, FAKE_DROP_BLOCK="b2", FAKE_DROP_AT="1", FAKE_DROP_FAILS="0",
                   P3_PULL_DEADLINE_S="15")
    (fake["run"] / "sg_cidrs").write_text(f"{F.MY_IP}/32\n")
    # the address changes when the network drops (fake curl reads it on every call)
    orig = fake["env"]["PATH"]
    p = F.run_driver(_with_ip_change(fake, orig))
    log = F.driver_log(fake)
    assert p.returncode == 1
    lost = [ln for ln in log.splitlines() if " SSH LOST b2 " in ln]
    assert len(lost) >= 3, log[-3000:]
    assert all("re-point it with sg_repoint.sh (the driver never does)" in ln for ln in lost)
    assert " DEADLINE b2 " in log
    assert _status(fake).split(" ", 1)[1].startswith("DONE FAILED at block b2 (rc 2)")
    td = (fake["run"] / "teardown.log").read_text().splitlines()
    assert len(td) == 1 and (fake["run"] / "state").read_text().strip() == "terminated"
    assert "fake cleanup check" in (fake["run"] / "cleanup.log").read_text()
    rec = F.rec_dir(fake)
    assert all(ln.endswith(": OK") for ln in (rec / "pull" / "b1" / "check.txt").read_text().splitlines())
    assert not (rec / "pull" / "b2" / "check.txt").exists()
    # block 1's idle halt was set when it ended, never cancelled, and block 2 read the
    # planned halt back as the pending shutdown (the fake sudo only records)
    sudo = (fake["run"] / "sudo.log").read_text()
    assert "sudo -n shutdown -P --no-wall +30" in sudo and "shutdown -c" not in sudo
    b2 = (fake["home"] / "p3" / "b2.log").read_text()
    assert "P3_PLANNED_HALT pending" in b2 and "P3_HALT_WARN" not in b2
    # each block was told the planned halt's minutes
    assert "ATTNBENCH_P3_HALT_MIN=473 bash scripts/aws_p3_instance.sh b2" in (fake["run"] / "ssh_calls.log").read_text().replace("\\", "")
    calls = (fake["run"] / "aws_calls.log").read_text()
    assert "authorize-security-group" not in calls and "revoke-security-group" not in calls


def _with_ip_change(fake, path):
    """curl answers the new address once the network is down."""
    b = fake["run"] / "bin_ipchange"
    b.mkdir()
    (b / "curl").write_text('#!/usr/bin/env bash\nif [[ -e "$FAKE_RUN/net_down" ]]; then echo 198.51.100.9; '
                            'else cat "$FAKE_RUN/my_ip"; fi\n')
    (b / "curl").chmod(0o755)
    fake["env"]["PATH"] = f"{b}:{path}"
    return fake


def test_a_network_drop_that_ends_goes_on(tmp_path):
    fake = F.build(tmp_path, FAKE_DROP_BLOCK="b2", FAKE_DROP_AT="1", FAKE_DROP_FAILS="4",
                   FAKE_STEP_SLEEP_S="0.5")
    p = F.run_driver(fake)
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-3000:]
    assert " SSH LOST b2 poll 1 " in log and " SSH back after " in log
    assert "the group still admits this Mac's address (network?)" in log
    assert _status(fake).split(" ", 1)[1].startswith("DONE OK")


def test_a_failed_step_stops_the_session_and_tears_down(tmp_path):
    fake = F.build(tmp_path, FAKE_FAIL_STEP="host_components")
    p = F.run_driver(fake)
    assert p.returncode == 1
    assert _status(fake).split(" ", 1)[1].startswith("DONE FAILED at block b2 (rc 1)")
    assert not (fake["home"] / "p3" / "b3.started").exists()
    assert len((fake["run"] / "teardown.log").read_text().splitlines()) == 1
    # what there is was pulled and checked: block 1, and the failed step's directory
    rec = F.rec_dir(fake)
    assert (rec / "session" / "05_host_components" / "result.json").exists()


def test_a_block_2_short_of_p1s_35_cells_is_a_no_go(tmp_path, monkeypatch):
    fake = F.build(tmp_path)
    stub = fake["run"].parent / "fr" / "stub_steps.py"
    stub.write_text(stub.read_text().replace(
        'for f in sorted(Path(os.environ["FAKE_P1_HOST"]).glob("*/bare_prices.json")):',
        'for f in sorted(Path(os.environ["FAKE_P1_HOST"]).glob("*/bare_prices.json"))[1:]:'))
    p = F.run_driver(fake)
    assert p.returncode == 1
    st = _status(fake).split(" ", 1)[1]
    assert st.startswith("DONE FAILED no-go after block 2: P3 has 34 cells"), st
    assert not (fake["home"] / "p3" / "b3.started").exists()


def test_a_window_that_created_nothing_allows_another_and_a_launch_allows_none(tmp_path):
    fake = F.build(tmp_path, FAKE_LAUNCH_MODE="refused")
    p = F.run_driver(fake)
    assert p.returncode == 1 and "NOT LAUNCHED the window ended" in _status(fake)
    assert _windows(fake)[-1][3] == "NOTHING_CREATED"
    assert not (fake["run"] / "teardown.log").exists()
    # a second window, which launches
    for d in (fake["run"] / "rec").iterdir():
        d.rename(d.with_name(d.name + ".first"))
    fake["env"]["FAKE_LAUNCH_MODE"] = "ok"
    assert F.run_driver(fake).returncode == 0
    # and then no third
    for d in (fake["run"] / "rec").glob("aws_p3_g6_*Z"):
        d.rename(d.with_name(d.name + ".second"))
    (fake["run"] / "launch_env.txt").unlink()
    p = F.run_driver(fake)
    assert p.returncode == 1 and "one P3 session" in _status(fake)
    assert not (fake["run"] / "launch_env.txt").exists()


def test_an_aborted_session_allows_one_relaunch_and_then_none(tmp_path):
    """The researcher's relaunch yes of 26 Nov 2026: a launched window with a
    dated `aborted` line naming its record allows one more session, no more."""
    fake = F.build(tmp_path)
    w = fake["run"] / "windows.tsv"
    w.write_text("start\t2026-11-25T20:43:39Z\t/x/rec\nend\t2026-11-25T20:59:03Z\t/x/rec\tLAUNCHED i-0old\n")
    p = F.run_driver(fake)
    assert p.returncode == 1 and "one P3 session" in _status(fake)      # not aborted: refused
    for d in (fake["run"] / "rec").glob("aws_p3_g6_*Z"):
        d.rename(d.with_name(d.name + ".refused"))
    with w.open("a") as fh:
        fh.write("aborted\t2026-11-26T06:00:00Z\t/x/rec\tstopped itself (idle-halt defect)\n")
    p = F.run_driver(fake)
    assert p.returncode == 0, F.driver_log(fake)[-2000:]
    for d in (fake["run"] / "rec").glob("aws_p3_g6_*Z"):
        d.rename(d.with_name(d.name + ".relaunch"))
    p = F.run_driver(fake)                                                # the relaunch launched
    assert p.returncode == 1 and "one P3 session" in _status(fake)
    for d in (fake["run"] / "rec").glob("aws_p3_g6_*Z"):
        d.rename(d.with_name(d.name + ".third"))
    recs = [ln.split("\t")[2] for ln in w.read_text().splitlines() if ln.startswith("end\t")]
    with w.open("a") as fh:                                               # a second abort line
        fh.write(f"aborted\t2026-11-26T07:00:00Z\t{recs[-1]}\tx\n")
    p = F.run_driver(fake)
    assert p.returncode == 1 and "2 P3 sessions are recorded aborted" in _status(fake)


def test_a_stopped_instance_is_said_at_once_when_ssh_drops(tmp_path):
    fake = F.build(tmp_path, FAKE_DROP_BLOCK="b2", FAKE_DROP_AT="1", FAKE_DROP_FAILS="0",
                   P3_PULL_DEADLINE_S="12")
    (fake["run"] / "state").write_text("stopped\n")
    p = F.run_driver(fake)
    log = F.driver_log(fake)
    assert p.returncode == 1
    said = [ln for ln in log.splitlines() if " INSTANCE STOPPED (stopped) in b2 " in ln]
    assert len(said) == 1, log[-3000:]                                    # once, at the first lost poll
    assert " SSH LOST b2 " not in log
    assert " DEADLINE b2 " in log


def test_a_window_left_without_its_end_line_refuses_the_next(tmp_path):
    fake = F.build(tmp_path)
    (fake["run"] / "windows.tsv").write_text("start\t2026-11-26T10:00:00Z\t/x/rec\n")
    p = F.run_driver(fake)
    assert p.returncode == 1 and "has no end line saying it created nothing" in _status(fake)
    assert not (fake["run"] / "launch_env.txt").exists()


@pytest.mark.parametrize("now,ok,window", [
    (DEADLINE - 100 * 60, True, "100"), (DEADLINE - 4 * 60, False, None),
    (DEADLINE, False, None), (DEADLINE + 3600, False, None)])
def test_the_launch_deadline_is_18_30_utc_on_2_december(tmp_path, now, ok, window):
    fake = F.build(tmp_path, P3_NOW_EPOCH=str(now), FAKE_LAUNCH_MODE="refused")
    F.run_driver(fake)
    st = _status(fake)
    if ok:
        env = (fake["run"] / "launch_env.txt").read_text()
        assert f"AWS_LAUNCH_WINDOW_MINUTES={window}" in env
    else:
        assert "REFUSED" in st and not (fake["run"] / "launch_env.txt").exists()


def test_an_address_the_group_does_not_admit_launches_nothing(tmp_path):
    fake = F.build(tmp_path)
    (fake["run"] / "sg_cidrs").write_text("198.51.100.9/32\n")
    p = F.run_driver(fake)
    assert p.returncode == 1 and "sg_repoint.sh" in _status(fake)
    assert not (fake["run"] / "launch_env.txt").exists() and _windows(fake) == []


def test_p3_variables_without_the_dry_switch_are_refused(tmp_path):
    fake = F.build(tmp_path)
    fake["env"].pop("P3_DRY")
    p = F.run_driver(fake)
    assert p.returncode == 1 and "P3_DRY is not 1" in p.stderr
    assert not (fake["run"] / "launch_env.txt").exists()


def _cred(fake):
    log = F.driver_log(fake)
    rows = (F.rec_dir(fake) / "credentials.tsv").read_text().splitlines()
    return log, [r.split("\t") for r in rows]


@pytest.mark.parametrize("ago_s,warn", [(0, False), (25 * 60, False), (35 * 60, True),
                                        (5 * 3600, True)])
def test_each_window_logs_when_the_login_session_ends_and_warns_under_11_5_h(tmp_path, ago_s, warn):
    login = int(time.time()) - ago_s
    fake = F.build(tmp_path, FAKE_LAUNCH_MODE="refused", ATTNBENCH_LOGIN_AT=str(login))
    F.run_driver(fake)
    log, rows = _cred(fake)
    end = datetime.fromtimestamp(login + 12 * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert len(rows) == 1 and rows[0][0] == "window_start" and rows[0][2] == end
    assert f"CREDENTIALS the aws login session ends by {end}" in log
    assert ("WARN CREDENTIALS end in" in log) is warn
    assert "NOT LAUNCHED the window ended" in _status(fake)   # a warning, not a refusal
    # logged before the window's launch
    assert log.index("CREDENTIALS") < log.index("LAUNCHING")


def test_no_login_time_is_a_warning_that_the_end_is_unknown(tmp_path):
    fake = F.build(tmp_path, FAKE_LAUNCH_MODE="refused")
    fake["env"].pop("ATTNBENCH_LOGIN_AT")
    F.run_driver(fake)
    log, rows = _cred(fake)
    assert rows[0][2] == "unknown" and "WARN CREDENTIALS the session's end is not known" in log


def test_an_earlier_session_expiry_in_the_login_cache_is_used_and_nothing_else_is_printed(tmp_path):
    fake = F.build(tmp_path, FAKE_LAUNCH_MODE="refused")
    now = time.time()
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()  # noqa: E731
    (fake["run"] / "login_cache" / "abc.json").write_text(json.dumps({
        "accessToken": {"accessKeyId": "AKIAFAKEKEYID", "secretAccessKey": "SECRETVALUE",
                        "expiresAt": iso(now + 900)},            # the 15-min credential: ignored
        "refreshToken": "REFRESHVALUE", "sessionExpiration": iso(now + 5 * 3600)}))
    F.run_driver(fake)
    log, rows = _cred(fake)
    want = datetime.fromtimestamp(int(now + 5 * 3600), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert rows[0][2] == want and "the login cache's session expiry" in rows[0][3]
    assert "WARN CREDENTIALS end in 5.0 h" in log or "WARN CREDENTIALS end in 4.9 h" in log
    text = log + (F.rec_dir(fake) / "credentials.tsv").read_text()
    assert not any(s in text for s in ("SECRETVALUE", "REFRESHVALUE", "AKIAFAKEKEYID"))


def test_the_final_re_pull_runs_only_when_asked(tmp_path):
    """P3_DRIVER_FINAL_REPULL=1 (allowed without the dry switch) turns it back on."""
    fake = F.build(tmp_path, FAKE_HOST_SCALE="1.5", P3_DRIVER_FINAL_REPULL="1")
    p = F.run_driver(fake)
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-3000:]
    assert "PULLED final:" in log and "FINAL RE-PULL skipped" not in log


def test_a_long_copy_pushes_the_halt_back_while_it_runs(tmp_path):
    """A slow copy (3 s here, a re-push every 1 s) pushes the halt back while it
    runs, and no pusher outlives its pull."""
    fake = F.build(tmp_path, FAKE_HOST_SCALE="1.5", P3_PUSH_EVERY_S="1", FAKE_SCP_SLEEP_S="3")
    p = F.run_driver(fake)
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-3000:]
    copying = [ln for ln in log.splitlines() if "HALT pushed back (pull b3, copying)" in ln]
    assert len(copying) >= 2, log[-2000:]
    last_b3 = max(i for i, ln in enumerate(log.splitlines()) if "PULLED b3:" in ln)
    assert not [ln for ln in log.splitlines()[last_b3 + 1:] if "copying)" in ln]


# --- P6: the same driver for one p5.4xlarge (the researcher's task of 26 Nov 2026
# after P3's push, item 2e; docs/launch_cards/aws_p6_p5.md) --------------------------

def _p6(tmp_path, **opts):
    return F.build(tmp_path, halt_min=312, FAKE_ITYPE="p5.4xlarge", **opts)


def test_p6_runs_its_three_blocks_with_a_long_final_pull(tmp_path):
    """P6's card handed to the launcher; block 1's backend check passed before any
    step; no gate after block 2; block 3 ends at postflight; and the final pull,
    slow here (3 s, a re-push every 1 s), keeps the halt pushed back, bounded by
    the planned halt (+312) and the delete (+359), until it ends."""
    fake = _p6(tmp_path, P3_PUSH_EVERY_S="1", FAKE_SCP_SLEEP_S="3")
    p = F.run_driver(fake, "P6")
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-3000:]
    rec = F.rec_dir(fake)
    assert rec.name.startswith("aws_p6_p5_")
    assert _status(fake).split(" ", 1)[1].strip() == "DONE OK all three blocks passed and pulled"
    env = (fake["run"] / "launch_env.txt").read_text().splitlines()
    for want in ("AWS_HALT_MINUTES=312", "AWS_MAX_RUN_MINUTES=359", "AWS_COST_CAP_INR=5940",
                 "AWS_RATE_INR_HR=990.2", "AWS_LAUNCH_WINDOW_MINUTES=180",
                 "AWS_SESSION_AZS=ap-southeast-2b ap-southeast-2c ap-southeast-2a",
                 "AWS_SESSION_REGION=ap-southeast-2", "AWS_AMI_ID=ami-096c5dc30364c9041",
                 "typed=launch", "args=P6"):
        assert want in env, (want, env)
    order = [ln.split(" ", 2)[1] + " " + ln.split(" ", 3)[2] for ln in log.splitlines()
             if ln.split(" ")[1:2] in (["BLOCK"], ["PULLED"], ["GATE"])]
    assert order == ["BLOCK b1", "PULLED b1:", "BLOCK b2", "PULLED b2:", "BLOCK b3", "PULLED b3:"], order
    assert not (rec / "gate.json").exists()
    s = rec / "session"
    assert sorted(d.name for d in s.iterdir() if d.is_dir()) == [
        "01_preflight", "02_link", "03_dense_select", "04_probe", "05_host_components",
        "06_device_components", "07_predict", "08_hold", "09_ground_truth", "10_verify",
        "11_postflight"]
    cfg = json.loads((s / "manifest.json").read_text())
    last = {a["step"]: a["status"] for a in cfg["attempts"]}
    assert len(last) == 11 and set(last.values()) == {"passed"} and "delay_injection" not in last
    # the abort gate ran first in block 1 and its record came back
    b1 = (fake["home"] / "p3" / "b1.log").read_text()
    assert "P6_BACKEND_GATE passed" in b1 and b1.index("P6_BACKEND_GATE") < b1.index("P3_BLOCK_EXIT")
    assert json.loads((rec / "b1_backends.json").read_text())["passed"] is True
    ssh = (fake["run"] / "ssh_calls.log").read_text().replace("\\", "")
    assert "ATTNBENCH_P3_PAIRING=P6 ATTNBENCH_P3_HALT_MIN=312 bash scripts/aws_p3_instance.sh b1" in ssh
    # the final pull is long: the halt was pushed back while it copied, and every push
    # read P6's planned halt (+312) back as the pending one, so none moved it later
    # (the bounds themselves, in fast minutes: tests/test_p3_halts.py)
    copying = [ln for ln in log.splitlines() if "HALT pushed back (pull b3, copying)" in ln]
    assert len(copying) >= 2, log[-2000:]
    halt = (fake["home"] / "p3" / "halt.log").read_text().splitlines()
    boot = int(fake["env"]["FAKE_BOOT_EPOCH"])
    planned = datetime.fromtimestamp(boot + 312 * 60, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert len(halt) >= 2 + len(copying) and all(f"pending {planned} is not before" in ln for ln in halt), halt[-3:]
    assert (fake["run"] / "teardown.log").read_text().splitlines() == [
        f"fake teardown: {F.INSTANCE} ap-southeast-2 {rec}/teardown_1  rate=990.2"]
    names = [Path(ln.split("\t")[1]).name for ln in (rec / "store_witness.tsv").read_text().splitlines()]
    assert names == ["prediction_P6_stub.json", "prediction_P6_stub.json.sha256"]


def test_p6s_backend_check_failing_stops_the_session_before_any_step(tmp_path):
    fake = _p6(tmp_path, FAKE_BACKEND_FAIL="K2 candidates not supplied on the H100: ['fa3']")
    p = F.run_driver(fake, "P6")
    log = F.driver_log(fake)
    assert p.returncode == 1
    assert _status(fake).split(" ", 1)[1].startswith("DONE FAILED at block b1"), log[-2000:]
    assert " BLOCK b1 FAILED exit 3" in log and " BLOCK b2 " not in log
    rec = F.rec_dir(fake)
    assert not (rec / "session" / "01_preflight").exists()
    assert not (fake["home"] / "p3_session" / "01_preflight").exists()
    rep = json.loads((rec / "b1_backends.json").read_text())
    assert rep["passed"] is False and "fa3" in rep["problems"][0]
    assert (rec / "b1.exit").read_text().split("\t")[0] == "3"
    assert len((fake["run"] / "teardown.log").read_text().splitlines()) == 1


def test_a_pairing_other_than_p3_or_p6_is_refused(tmp_path):
    fake = _p6(tmp_path)
    p = F.run_driver(fake, "P5")
    assert p.returncode == 1 and "REFUSING: the pairing is 'P5'" in p.stderr
    assert not (fake["run"] / "launch_env.txt").exists()


# --- P6 on Spot (the researcher's task of 27 Nov 2026; docs/launch_cards/aws_p6_p5_spot.md):
# an interrupted block is discarded in full and rerun on a new instance, in block 1's
# zone, the finished blocks restored and checked first, at most two relaunches, all
# inside P6's Rs 5,940 -----------------------------------------------------------------

def _spot(tmp_path, **opts):
    return _p6(tmp_path, FAKE_MARKET="spot", FAKE_STEP_SLEEP_S="1.5", **opts)


def _events(log):
    keep = ("BLOCK", "PULLED", "SPOT", "INTERRUPTED", "RELAUNCH", "RESTORED", "BACKEND", "DONE")
    return [ln.split(" ", 2)[1] + " " + ln.split(" ", 3)[2] for ln in log.splitlines()
            if ln.split(" ")[1:2] and ln.split(" ")[1] in keep]


def _env(fake, n):
    name = "launch_env.txt" if n == 1 else f"launch_env_{n}.txt"
    return dict(ln.split("=", 1) for ln in (fake["run"] / name).read_text().splitlines())


def test_p6_spot_hands_the_launcher_spot_first_with_the_bid(tmp_path):
    fake = _p6(tmp_path, FAKE_LAUNCH_MODE="refused")
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 1 and "NOT LAUNCHED the window ended" in _status(fake)
    env = _env(fake, 1)
    assert env["AWS_PROVISIONING"] == "spot-first" and env["AWS_SPOT_MAX_PRICE_USD"] == "8.944"
    assert env["AWS_COST_CAP_INR"] == "5940" and env["AWS_MAX_RUN_MINUTES"] == "359"
    assert "Spot then on-demand in each pass, bid at most $8.944/h" in F.driver_log(fake)


@pytest.mark.parametrize("args,said", [(("P3", "spot"), "spot is for P6 only"),
                                       (("P6", "spots"), "not spot")])
def test_spot_is_p6s_alone(tmp_path, args, said):
    fake = _p6(tmp_path)
    p = F.run_driver(fake, *args)
    assert p.returncode == 1 and said in p.stderr
    assert not (fake["run"] / "launch_env.txt").exists()


def test_p6_on_demand_never_reads_the_spot_notice(tmp_path):
    """An on-demand instance (the launcher's fallback) polls as before."""
    fake = _p6(tmp_path)
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 0, F.driver_log(fake)[-2000:]
    ssh = (fake["run"] / "ssh_calls.log").read_text()
    assert "spot/instance-action" not in ssh and "ATTNBENCH_P3_MARKET=on-demand" in ssh.replace("\\", "")


@pytest.mark.parametrize("block,how", [("b2", "state"), ("b3", "notice")])
def test_a_spot_interruption_mid_block_relaunches_restores_and_resumes(tmp_path, block, how):
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK=block, FAKE_INTERRUPT_AT="1", FAKE_INTERRUPT_HOW=how)
    p = F.run_driver(fake, "P6", "spot")
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-4000:]
    assert _status(fake).split(" ", 1)[1].strip() == "DONE OK all three blocks passed and pulled; relaunches 1"
    ev = _events(log)
    i = ev.index(f"BLOCK {block}")
    said = "SPOT INTERRUPTED" if how == "state" else "SPOT INTERRUPTION"
    assert ev[i + 1].startswith(said), ev
    assert ev[i + 2:i + 6] == ["INTERRUPTED attempt", "RELAUNCH 1", "RESTORED the", "BACKEND CHECK"], ev
    assert ev[i + 6] == f"BLOCK {block}", ev                       # the block again, from its start
    blocks = [e for e in ev if e.startswith(("BLOCK b", "PULLED b"))]
    assert [e for e in blocks if e.startswith("PULLED")] == ["PULLED b1:", "PULLED b2:", "PULLED b3:"]
    rec = F.rec_dir(fake)
    # the relaunch: block 1's zone, what is left of the cap, P6's halt and delete, a 60-min window
    first, second = _env(fake, 1), _env(fake, 2)
    assert second["AWS_SESSION_AZS"] == "ap-southeast-2a" and second["AWS_LAUNCH_WINDOW_MINUTES"] == "60"
    cost = [ln.split("\t") for ln in (rec / "cost.tsv").read_text().splitlines()]
    used = int(cost[0][6].split()[1])
    minutes = int(cost[0][5].split()[1])
    assert used == -(-990.2 * minutes // 60) and cost[0][7] == f"interrupted in {block}"
    assert second["AWS_COST_CAP_INR"] == str(5940 - used)
    delete = min(359, int((5940 - used) * 60 / 990.2))     # what is left buys, at most P6's +359
    assert second["AWS_MAX_RUN_MINUTES"] == str(delete) and delete >= 355
    assert second["AWS_HALT_MINUTES"] == str(min(312, delete - 47))
    assert second["AWS_PROVISIONING"] == "spot-first" and first["AWS_SESSION_AZS"].split() == [
        "ap-southeast-2b", "ap-southeast-2c", "ap-southeast-2a"]
    # Spot asks in 2c, 2b, 2a; a relaunch in block 1's zone alone (28 Nov 2026, item 3)
    assert first["AWS_SPOT_AZS"] == "ap-southeast-2c ap-southeast-2b ap-southeast-2a"
    assert second["AWS_SPOT_AZS"] == "ap-southeast-2a"
    # the credit record: the first instance voided and counted again at its spend,
    # so the two instances count Rs 5,940 in all (28 Nov 2026, item 2)
    assert _live_record(fake) == {F.INSTANCE: used, "i-0fa0e0000000000b2": 5940 - used}
    # two instances, each torn down once; the ledger says so
    td = (fake["run"] / "teardown.log").read_text().splitlines()
    assert [ln.split()[2] for ln in td] == [F.INSTANCE, "i-0fa0e0000000000b2"]
    w = _windows(fake)
    assert [x[0] for x in w] == ["start", "end", "relaunch"] and w[2][3] == "LAUNCHED i-0fa0e0000000000b2"
    # the finished blocks went back checked; the interrupted one's files are set aside
    restore = rec / "restore_2"
    assert (restore / "restore.sha256").exists()
    assert (rec / "attempt_1" / "instance_id.txt").read_text().strip() == F.INSTANCE
    assert (rec / "attempt_1" / f"{'b1' if block == 'b2' else 'b2'}.exit").exists()
    assert "restored, attempt 2" in (rec / "b1.exit").read_text()
    # the session: every step once, passed; the discarded block left nothing in it
    s = rec / "session"
    m = json.loads((s / "manifest.json").read_text())
    last = {a["step"]: a["status"] for a in m["attempts"]}
    assert len(last) == 11 and set(last.values()) == {"passed"}, last
    assert [a["status"] for a in m["attempts"] if a["step"] == "host_components"] == ["passed"]
    # the rerun block 3 committed its own prediction, witnessed on this Mac
    names = [Path(ln.split("\t")[1]).name for ln in (rec / "store_witness.tsv").read_text().splitlines()]
    assert names == ["prediction_P6_stub.json", "prediction_P6_stub.json.sha256"]
    # the second instance's block 1 gate ran again before the block, and the market reached the block
    assert json.loads((rec / "relaunch_backends.json").read_text())["passed"] is True
    halt = second["AWS_HALT_MINUTES"]
    assert f"ATTNBENCH_P3_MARKET=spot ATTNBENCH_P3_PAIRING=P6 ATTNBENCH_P3_HALT_MIN={halt} bash scripts/aws_p3_instance.sh {block}" \
        in (fake["run"] / "ssh_calls.log").read_text().replace("\\", "")


def _live_record(fake):
    """The launch record's lines the credit gate counts (VOID lines honoured)."""
    lines = [ln.split("\t") for ln in (fake["run"] / "aws_launches.tsv").read_text().splitlines()]
    voided = {tuple(ln[1:4]) for ln in lines if ln[0] == "VOID"}
    for ln in lines:
        if ln[0] == "VOID":
            assert len(ln) == 6 and tuple(ln[1:4]) in {tuple(x[:3]) for x in lines if x[0] != "VOID"}
    return {ln[2]: int(ln[3]) for ln in lines if ln[0] != "VOID" and tuple(ln[:3]) not in voided}


def test_two_relaunches_keep_the_session_at_its_cap_in_the_credit_record(tmp_path):
    """Two interruptions in block 2: each interrupted instance counts its spend,
    the last its cap, what was left: Rs 5,940 in all, so the credit gate never
    counts more than the session's cap."""
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK="b2", FAKE_INTERRUPT_AT="1", FAKE_INTERRUPTS="2")
    p = F.run_driver(fake, "P6", "spot")
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-4000:]
    assert _status(fake).split(" ", 1)[1].strip().endswith("relaunches 2")
    rec = F.rec_dir(fake)
    used = [int(ln.split("\t")[6].split()[1]) for ln in (rec / "cost.tsv").read_text().splitlines()
            if "interrupted in b2" in ln]
    assert len(used) == 2
    live = _live_record(fake)
    assert live == {F.INSTANCE: used[0], "i-0fa0e0000000000b2": used[1],
                    "i-0fa0e0000000000b3": 5940 - used[0] - used[1]}
    assert sum(live.values()) == 5940
    assert _env(fake, 3)["AWS_COST_CAP_INR"] == str(5940 - used[0] - used[1])
    assert sum(ln.startswith("VOID") for ln in (fake["run"] / "aws_launches.tsv").read_text().splitlines()) == 2
    assert log.count("CREDITS the interrupted instance now counts Rs ") == 2


def test_the_dry_run_needs_its_own_launch_record(tmp_path):
    fake = _p6(tmp_path)
    del fake["env"]["P3_LAUNCH_RECORD"]
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 1 and not (fake["run"] / "launch_env.txt").exists()


# --- an expired AWS login (the researcher's task of 28 Nov 2026, item 4) ---------
def test_an_expired_login_at_the_preflight_says_so_and_launches_nothing(tmp_path):
    fake = _p6(tmp_path, FAKE_LOGIN_EXPIRED="1")
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 1
    assert "REFUSED SSH preflight: AWS login expired, run aws login" in _status(fake)
    assert not (fake["run"] / "launch_env.txt").exists() and _windows(fake) == []


@pytest.mark.parametrize("mode,end", [("expired", "NOTHING_CREATED"), ("expired-at-pin", "NOTHING_CREATED"),
                                      ("expired-unconfirmed", "UNCLEAR rc=1")])
def test_a_login_that_expires_in_the_window_stops_the_driver_with_the_message(tmp_path, mode, end):
    fake = _p6(tmp_path, FAKE_LAUNCH_MODE=mode)
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 1
    st = _status(fake)
    assert "STOPPED AWS login expired, run aws login (the launcher stopped" in st, st
    assert _windows(fake)[-1][0] == "end" and _windows(fake)[-1][3] == end
    assert "no longer admits" not in F.driver_log(fake)


def test_an_interruption_in_block_1_reruns_block_1_with_nothing_to_restore(tmp_path):
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK="b1", FAKE_INTERRUPT_AT="1")
    p = F.run_driver(fake, "P6", "spot")
    log = F.driver_log(fake)
    assert p.returncode == 0, log[-4000:]
    ev = _events(log)
    i = ev.index("RELAUNCH 1")
    assert ev[i + 1] == "BLOCK b1" and "RESTORED the" not in ev and "BACKEND CHECK" not in ev
    assert "P6_BACKEND_GATE passed" in (fake["home"] / "p3" / "b1.log").read_text()


def test_a_third_interruption_stops_with_the_blocks_it_has(tmp_path):
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK="b2", FAKE_INTERRUPT_AT="1", FAKE_INTERRUPTS="3")
    p = F.run_driver(fake, "P6", "spot")
    log = F.driver_log(fake)
    assert p.returncode == 1
    assert "DONE FAILED Spot interrupted in block b2 with all 2 relaunches used" in _status(fake), log[-3000:]
    assert len((fake["run"] / "teardown.log").read_text().splitlines()) == 3
    assert (fake["run"] / "launches").read_text().strip() == "3"
    rec = F.rec_dir(fake)
    assert sorted(d.name for d in (rec / "session").iterdir() if d.is_dir()) == [
        "01_preflight", "02_link", "03_dense_select", "04_probe"]
    cost = (rec / "cost.tsv").read_text().splitlines()
    assert [ln.split("\t")[1] for ln in cost] == ["1", "2", "3"]


def test_a_relaunch_refused_for_capacity_creates_nothing_more(tmp_path):
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK="b2", FAKE_INTERRUPT_AT="1", FAKE_RELAUNCH_MODE="refused")
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 1
    assert "relaunch 1 created nothing" in _status(fake)
    assert len((fake["run"] / "teardown.log").read_text().splitlines()) == 1
    assert _windows(fake)[-1][0] == "relaunch" and _windows(fake)[-1][3].startswith("NOTHING_CREATED")


@pytest.mark.parametrize("cap,block,ok", [("3500", "b3", True), ("2700", "b2", False)])
def test_what_is_left_of_the_cap_bounds_the_relaunched_instance(tmp_path, cap, block, ok):
    """The new hard delete is what the cap left buys at the rate (at most +359),
    its halt 47 min earlier (at most +312); it must leave the blocks still to
    run their card minutes (20 for setup, b2 81, b3 99) and the last pull's 10."""
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK=block, FAKE_INTERRUPT_AT="1", P3_CAP=cap)
    p = F.run_driver(fake, "P6", "spot")
    rec = F.rec_dir(fake)
    used = int((rec / "cost.tsv").read_text().splitlines()[0].split("\t")[6].split()[1])
    left = int(cap) - used
    delete = int(left * 60 / 990.2)
    halt = delete - 47
    if ok:
        assert p.returncode == 0, F.driver_log(fake)[-3000:]
        env = _env(fake, 2)
        assert env["AWS_COST_CAP_INR"] == str(left)
        assert env["AWS_MAX_RUN_MINUTES"] == str(delete) and env["AWS_HALT_MINUTES"] == str(halt)
        assert halt - 10 >= 20 + 99
        assert f"ATTNBENCH_P3_HALT_MIN={halt} bash scripts/aws_p3_instance.sh b3" in (
            fake["run"] / "ssh_calls.log").read_text().replace("\\", "")
    else:
        assert p.returncode == 1 and halt - 10 < 20 + 81 + 99
        assert f"allows a halt at +{halt} min, short of the 200 min" in _status(fake)
        assert (fake["run"] / "launches").read_text().strip() == "1"


@pytest.mark.parametrize("opts,said", [
    ({"FAKE_RELAUNCH_GPU": "NVIDIA H100 PCIe, 999.99"}, "does not match the first"),
    ({"FAKE_RELAUNCH_BACKEND_FAIL": "fa3 incorrect at 2048 (fake)"}, "dense-backend check failed on the relaunched"),
])
def test_a_relaunched_instance_unlike_the_first_or_failing_the_gate_is_not_used(tmp_path, opts, said):
    fake = _spot(tmp_path, FAKE_INTERRUPT_BLOCK="b2", FAKE_INTERRUPT_AT="1", **opts)
    p = F.run_driver(fake, "P6", "spot")
    assert p.returncode == 1 and said in _status(fake), F.driver_log(fake)[-3000:]
    assert len((fake["run"] / "teardown.log").read_text().splitlines()) == 2
    assert not (fake["home"] / "p3" / "b2.started").exists()
