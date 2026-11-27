"""The EC2 state reasons the P3/P6 driver branches on, and its path for each
(the researcher's audit of 28 Nov 2026; scripts/_aws_state_reasons.sh).

The reasons are the twelve AWS documents for the StateReason object
(https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_StateReason.html,
read 28 Nov 2026). Paths: relaunch (the block discarded, a relaunch under the
limit), stop (stop at once and report: no relaunch), no-relaunch (the instance
is not terminated: the driver keeps polling to the block's deadline).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "scripts" / "_aws_state_reasons.sh"

DOCUMENTED = [
    "Server.InsufficientInstanceCapacity", "Server.InternalError", "Server.ScheduledStop",
    "Server.SpotInstanceShutdown", "Server.SpotInstanceTermination",
    "Client.InstanceInitiatedShutdown", "Client.InstanceTerminated", "Client.InternalError",
    "Client.InvalidSnapshot.NotFound", "Client.UserInitiatedHibernate",
    "Client.UserInitiatedShutdown", "Client.VolumeLimitExceeded",
]
# On a Spot instance shutting down or terminated: Spot's reclaim (terminate, or the
# stop a one-time request cannot take) and its own halt relaunch; the rest stop.
SPOT_RELAUNCH = {"Server.SpotInstanceTermination", "Server.SpotInstanceShutdown",
                 "Client.InstanceInitiatedShutdown"}

TABLE = (
    [("spot", st, r, "relaunch" if r in SPOT_RELAUNCH else "stop")
     for r in DOCUMENTED for st in ("terminated", "shutting-down")]
    + [("on-demand", "terminated", r, "stop") for r in DOCUMENTED]
    + [("spot", "terminated", "none", "stop")]
    + [(m, st, r, "no-relaunch") for m in ("spot", "on-demand")
       for st in ("stopped", "stopping", "running", "pending", "unreadable")
       for r in ("Client.InstanceInitiatedShutdown", "Server.SpotInstanceTermination", "none")]
)


def _sh(fn, *args):
    p = subprocess.run(["bash", "-c", f'source "$0"; {fn} "$@"', str(LIB), *args],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_every_documented_reason_is_in_the_table():
    assert len(DOCUMENTED) == 12 and SPOT_RELAUNCH <= set(DOCUMENTED)
    for r in DOCUMENTED:
        assert r in LIB.read_text(), r


def test_each_reason_maps_to_its_path():
    got = {}
    for market, state, reason, want in TABLE:
        got[(market, state, reason)] = (_sh("aws_state_reason_path", market, state, reason), want)
    wrong = {k: v for k, v in got.items() if v[0] != v[1]}
    assert not wrong, wrong


@pytest.mark.parametrize("code,message,reason", [
    # the docs' own example: the code generic, the message specific
    ("Server.InternalError", "Client.VolumeLimitExceeded: Volume limit exceeded", "Client.VolumeLimitExceeded"),
    ("Server.SpotInstanceTermination", "Server.SpotInstanceTermination: Spot instance termination",
     "Server.SpotInstanceTermination"),
    ("Client.InstanceInitiatedShutdown", "None", "Client.InstanceInitiatedShutdown"),
    ("None", "None", "none"),
    ("", "", "none"),
])
def test_the_reason_is_the_messages_leading_token_else_the_code(code, message, reason):
    assert _sh("aws_state_reason", code, message) == reason
