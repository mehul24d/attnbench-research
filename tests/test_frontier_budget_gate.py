"""The frontier study's budget gate (pre-registration sec. 8.3), and the
launchers' refusal without it.

The launchers are run end to end against a fake `gcloud` that only logs its
arguments. No test here can reach the real one: the fake is first on PATH and
the session-wide shim in conftest.py is behind it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from attnbench.analysis import frontier_prereg as fp

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "frontier_budget_gate.py"
LAUNCHERS = {"A100": "gcp_launch_a100.sh", "H100": "gcp_launch_h100.sh", "L4": "gcp_launch_l4.sh"}
# A never-cut session on each launcher's card.
SESSION = {"A100": "I1", "H100": "I3", "L4": "I4"}
HEADER = ",".join(fp.LEDGER_COLUMNS) + "\n"


def _row(**kw):
    base = dict(date="2026-10-04", session="", instance="", card="A100", minutes="0",
                est_inr="0", status="other", source="test")
    base.update(kw)
    return base


def _ledger(tmp_path, rows=()):
    p = tmp_path / "frontier_spend.csv"
    p.write_text(HEADER + "".join(",".join(str(r[c]) for c in fp.LEDGER_COLUMNS) + "\n"
                                   for r in rows))
    return p


def _launch(tmp_path, card, env_extra, ledger=None):
    """Run a launcher against a fake gcloud. Returns (process, gcloud calls)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "gcloud.log"
    log.unlink(missing_ok=True)
    fake = bin_dir / "gcloud"
    fake.write_text(f'#!/bin/bash\necho "$*" >> "{log}"\nexit 0\n')
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ("FRONTIER_SESSION", "ATTNBENCH_NOT_FRONTIER", "GCP_MAX_RUN",
                        "GCP_HALT_MINUTES", "FRONTIER_LEDGER", "FRONTIER_STATE")}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", GCP_PROJECT="fake-project",
               ATTNBENCH_PYTHON=sys.executable,
               FRONTIER_LEDGER=str(ledger or _ledger(tmp_path)),
               FRONTIER_STATE=str(tmp_path / "no_state.json"))
    env.update(env_extra)
    p = subprocess.run(["bash", str(REPO / "scripts" / LAUNCHERS[card]), "test-instance"],
                       capture_output=True, text=True, env=env, timeout=120,
                       input="launch\n")          # the L4 launcher asks; the others ignore it
    calls = log.read_text().splitlines() if log.exists() else []
    return p, [c for c in calls if "instances create" in c]


# --- the launchers ----------------------------------------------------------

@pytest.mark.parametrize("card", sorted(LAUNCHERS))
def test_a_launch_with_no_gate_is_refused_and_creates_nothing(tmp_path, card):
    p, created = _launch(tmp_path, card, {})
    assert p.returncode == 3 and "REFUSED" in p.stderr
    assert created == []


@pytest.mark.parametrize("card", sorted(LAUNCHERS))
def test_a_launch_whose_gate_says_stop_is_refused_and_creates_nothing(tmp_path, card):
    ledger = _ledger(tmp_path, [_row(est_inr="11900")])
    p, created = _launch(tmp_path, card, {"FRONTIER_SESSION": SESSION[card]}, ledger)
    assert p.returncode == 3 and "VERDICT=STOP" in p.stdout
    assert created == []
    assert ledger.read_text().count("\n") == 2          # nothing was recorded


@pytest.mark.parametrize("card", sorted(LAUNCHERS))
def test_a_launch_whose_gate_passes_creates_one_instance_under_the_gates_cap(tmp_path, card):
    ledger = _ledger(tmp_path)
    p, created = _launch(tmp_path, card, {"FRONTIER_SESSION": SESSION[card],
                                          "GCP_MAX_RUN": "9h", "GCP_HALT_MINUTES": "500"}, ledger)
    assert p.returncode == 0, p.stderr
    g = fp.gate(SESSION[card], card, [])
    assert len(created) == 1
    # The gate's cap wins over the launcher's default and over GCP_MAX_RUN.
    assert f"--max-run-duration={g['max_run_minutes']}m" in created[0]
    assert f"+{g['halt_minutes']} min" in p.stdout
    rows = ledger.read_text().splitlines()
    assert len(rows) == 2 and f"{SESSION[card]},test-instance,{card}," in rows[1]
    assert rows[1].split(",")[6] == "launched"
    # The same session cannot be launched again while that launch is open.
    p2, created2 = _launch(tmp_path, card, {"FRONTIER_SESSION": SESSION[card]}, ledger)
    assert p2.returncode == 3 and created2 == []


def test_a_session_on_the_wrong_card_is_refused(tmp_path):
    p, created = _launch(tmp_path, "L4", {"FRONTIER_SESSION": "I1"})
    assert p.returncode == 3 and created == [] and "runs on the A100" in p.stderr


def test_a_missing_ledger_is_refused_not_read_as_zero_spend(tmp_path):
    p, created = _launch(tmp_path, "A100", {"FRONTIER_SESSION": "I1"}, tmp_path / "absent.csv")
    assert p.returncode == 3 and created == [] and "no ledger" in p.stderr


def test_a_session_of_another_pre_registration_must_say_so(tmp_path):
    p, created = _launch(tmp_path, "L4", {"ATTNBENCH_NOT_FRONTIER": "T4 calibrated, session B"})
    assert p.returncode == 0 and len(created) == 1
    assert "outside the Rs 12,000 cap" in p.stdout
    both, created = _launch(tmp_path, "L4", {"ATTNBENCH_NOT_FRONTIER": "x", "FRONTIER_SESSION": "I4"})
    assert both.returncode == 3 and created == []


def test_every_instance_creating_gpu_launcher_sources_the_gate_before_creating():
    for name in LAUNCHERS.values():
        src = (REPO / "scripts" / name).read_text()
        assert src.index("frontier_gate ") < src.index("gcloud compute instances create"), name
        assert src.index("frontier_gate ") < src.index("STARTUP_SCRIPT="), name


# --- the gate ---------------------------------------------------------------

def test_the_gate_reserves_higher_priority_items_at_worst_case_inputs():
    assert fp.gate("I1", "A100", [])["verdict"] == "PROCEED"
    for low in ("I2b", "I2d", "I2e"):
        assert fp.gate(low, "A100", [])["verdict"] == "STOP"
    # H8's probe is covered whether or not I2c-B is still to be reserved.
    assert fp.gate("XL0", "A100", [], {"h6a_passed": True})["verdict"] == "PROCEED"
    assert fp.gate("XL0", "A100", [])["verdict"] == "PROCEED"
    assert fp.gate("SL", None, [])["verdict"] == "STOP"
    # The full-set table reserves everything ranked above it, so it waits.
    assert fp.gate("C-full", "A100", [])["verdict"] == "STOP"
    # Once the Llama runs are cut, a lower item no longer has to leave room for them.
    state = {"cut": ["XL0", "XL primary", "XL secondary", "XL 65536"]}
    assert fp.gate("I2e", "A100", [], state)["verdict"] == "PROCEED"


def test_an_open_launch_counts_at_its_hard_cap_until_the_teardown_is_recorded():
    opened = _row(session="I1", instance="i-1", est_inr="1300", status="launched")
    assert fp.read_ledger([opened])["spent"] == 1300
    closed = _row(session="I1", instance="i-1", est_inr="600", status="complete")
    led = fp.read_ledger([opened, closed])
    assert led["spent"] == 600 and led["complete"] == {"I1"} and not led["open"]
    failed = _row(session="I1", instance="i-1", est_inr="600", status="failed")
    led = fp.read_ledger([opened, failed])
    assert led["spent"] == 600 and not led["complete"]
    # A failed session's spend stays, and the session is still owed.
    g = fp.gate("I1", "A100", [opened, failed])
    assert g["spent"] == 600 and g["verdict"] == "PROCEED"


def test_a_complete_session_is_never_launched_again():
    done = [_row(session="I1", instance="i-1", est_inr="600", status="launched"),
            _row(session="I1", instance="i-1", est_inr="600", status="complete")]
    with pytest.raises(fp.GateRefusal, match="already complete"):
        fp.gate("I1", "A100", done)


@pytest.mark.parametrize("bad", [
    dict(status="done"), dict(est_inr="lots"), dict(est_inr="-5"),
    dict(status="launched", instance=""), dict(session="I9", instance="i", status="complete")])
def test_a_bad_ledger_row_is_refused(bad):
    with pytest.raises(fp.GateRefusal):
        fp.gate("I1", "A100", [_row(**bad)])


@pytest.mark.parametrize("state", [
    {"cut": ["I1"]}, {"cut": ["nope"]}, {"released": ["I1"]}, {"inputs": {"nope": 1}},
    {"surprise": 1}])
def test_a_bad_plan_state_is_refused(state):
    with pytest.raises(fp.GateRefusal):
        fp.gate("C", "A100", [], state)


def test_x_primary_is_gated_per_band_and_the_replicates_per_session():
    lines = {l.name: l for l in fp.bracket()}
    assert set(fp.session_units(lines["X primary"])) == {"X primary:16384", "X primary:32768"}
    assert sum(fp.session_units(lines["X primary"]).values()) == pytest.approx(lines["X primary"].minutes[1])
    assert len(fp.session_units(lines["R A100x3"])) == 3
    with pytest.raises(fp.GateRefusal):
        fp.gate("X primary", "A100", [])
    g = fp.gate("X primary:32768", "A100", [])
    assert g["halt_minutes"] == 389 and g["max_run_minutes"] == 399


def test_the_cli_exit_codes(tmp_path):
    ledger = _ledger(tmp_path)
    def run(*args):
        return subprocess.run([sys.executable, str(GATE), "--ledger", str(ledger),
                               "--state", str(tmp_path / "s.json"), *args],
                              capture_output=True, text=True)
    assert run("--session", "I1", "--card", "A100").returncode == 0
    assert run("--session", "I2d", "--card", "A100").returncode == 1
    assert run("--session", "I1", "--card", "L4").returncode == 2
    assert run("--fund", "SL").returncode == 1
    assert run("--session", "I1", "--card", "A100", "--record-launch", "i-9").returncode == 0
    assert run("--record-teardown", "i-9", "--minutes", "120", "--status", "complete").returncode == 0
    # 120 minutes at the A100's rate plus the boot disk.
    assert fp.read_ledger(list(__import__("csv").DictReader(ledger.open())))["spent"] == round(
        2 * (284 + 2.4))
    assert run("--init").returncode == 2                 # never overwrites a ledger
