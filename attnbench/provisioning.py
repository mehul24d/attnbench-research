"""The provisioning model's vocabulary (plan Section 5.2), in one place.

Two sets of names, and the one map between them:

  - **launch names**, what a launcher asks the cloud for and writes into the
    `attnbench-provisioning-model` tag: GCP's `--provisioning-model` values
    (STANDARD, SPOT, FLEX_START) as instance metadata, and on AWS the same
    tag as an instance tag, where `aws_launch_session.sh` writes STANDARD for
    on-demand;
  - **row names**, what provenance records on every row and what every
    preflight compares against: on-demand, spot, dws-flex-start (reserved is
    named; no launcher creates it).

Provenance (`provenance._provisioning_model`), the session preflights
(`t227_steps`, `t61_session`, `t64_session`) and the launcher-tag test all
read this module. Shakedown 2's `t21_t220` and the T3.1 rehearsal's preflight
failed because a check compared a row name with a launch name (31 Oct 2026).
"""

from __future__ import annotations

TAG = "attnbench-provisioning-model"

ON_DEMAND, SPOT, FLEX_START, RESERVED = "on-demand", "spot", "dws-flex-start", "reserved"

# launch name (tag value) -> row name
ROW_NAME = {
    "STANDARD": ON_DEMAND,
    "SPOT": SPOT,
    "FLEX_START": FLEX_START,
    "RESERVED": RESERVED,
}
ROW_NAMES = frozenset(ROW_NAME.values())

# AWS's `instance-life-cycle` -> GCP's `scheduling/preemptible`, so one
# cross-check serves both clouds.
AWS_LIFE_CYCLE = {"spot": "TRUE", "on-demand": "FALSE"}


def row_name(launch_name: str) -> str:
    """The row name of a launch name; unknown names are refused."""
    try:
        return ROW_NAME[launch_name.strip().upper()]
    except KeyError:
        raise KeyError(f"unknown provisioning model {launch_name!r}; launch names are "
                       f"{sorted(ROW_NAME)}") from None


def row_names(launch_names) -> tuple:
    return tuple(row_name(n) for n in launch_names)


def is_preemptible(row: str) -> bool:
    return row == SPOT
