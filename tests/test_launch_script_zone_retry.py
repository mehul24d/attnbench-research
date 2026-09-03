"""The zone-retry loop in gcp_launch_compile_session.sh must stop at the
first successful create, and must refuse to launch when an instance already
exists.

**Why this is a test and not a code comment.** On 2026-09-03 an ad-hoc
zone-retry loop created an instance in asia-south1-c and then went on to
attempt asia-south1-b and asia-south1-a as well. Both were stocked out, so
exactly one instance existed -- luck, not design. With capacity in those
zones it would have created and billed three g2-standard-8 + L4 instances
simultaneously, which is the failure mode this project treats as the worst
available.

A stockout creates nothing and costs nothing, so continuing after a FAILURE
is free; continuing after a SUCCESS bills a second GPU. In a hurried
one-liner those two cases look nearly identical -- which is exactly why the
loop belongs in a reviewed script with a test that watches the `break` work,
rather than being retyped at the terminal each session.

These tests never touch GCP: a fake `gcloud` is placed first on PATH.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gcp_launch_compile_session.sh"

# Fails for any zone whose name ends in the letter recorded in FAIL_ZONES,
# succeeds otherwise, and appends every create attempt to $ATTEMPT_LOG.
FAKE_GCLOUD = r"""#!/usr/bin/env bash
ARGS="$*"
case "$ARGS" in
  *"config get-value project"*) echo "test-project"; exit 0 ;;
  *"machine-images describe"*)  echo "READY"; exit 0 ;;
  *"machine-images list"*)      exit 0 ;;
  *"instances list"*)           printf '%s' "${FAKE_EXISTING_INSTANCES:-}"; exit 0 ;;
  *"instances create"*)
      ZONE=""
      for a in "$@"; do case "$a" in --zone=*) ZONE="${a#--zone=}" ;; esac; done
      echo "$ZONE" >> "$ATTEMPT_LOG"
      case " $FAIL_ZONES " in
        *" $ZONE "*) echo "ZONE_RESOURCE_POOL_EXHAUSTED" >&2; exit 1 ;;
      esac
      echo "Created instance in $ZONE"; exit 0 ;;
esac
exit 0
"""


def _run(tmp_path, *, fail_zones: str, zones: str, existing: str = ""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gcloud"
    fake.write_text(FAKE_GCLOUD)
    fake.chmod(0o755)

    attempts = tmp_path / "attempts.txt"
    attempts.write_text("")

    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        ATTEMPT_LOG=str(attempts),
        FAIL_ZONES=fail_zones,
        FAKE_EXISTING_INSTANCES=existing,
        GCP_ZONE_FALLBACKS=zones,
        GCP_PROJECT="test-project",
    )
    proc = subprocess.run(["bash", str(SCRIPT), "test-instance"], input="launch\n",
                          capture_output=True, text=True, env=env, timeout=60)
    tried = [z for z in attempts.read_text().split() if z]
    return proc, tried


def test_stops_at_the_first_successful_zone(tmp_path):
    """The regression: zone-a stocks out, zone-b succeeds, and zone-c must
    NEVER be attempted. A third attempt here is a third billed GPU."""
    proc, tried = _run(tmp_path, fail_zones="zone-a", zones="zone-a zone-b zone-c")

    assert tried == ["zone-a", "zone-b"], (
        f"expected to stop after the first success, but attempted {tried}. "
        f"Each attempt after a success is another billed GPU instance.")
    assert "zone-c" not in tried
    assert proc.returncode == 0


def test_keeps_trying_after_a_stockout(tmp_path):
    """The other half -- continuing after a *failure* is free and correct,
    so the fix must not have turned the loop into a single attempt."""
    proc, tried = _run(tmp_path, fail_zones="zone-a zone-b", zones="zone-a zone-b zone-c")

    assert tried == ["zone-a", "zone-b", "zone-c"]
    assert proc.returncode == 0


def test_all_zones_stocked_out_creates_nothing_and_fails(tmp_path):
    proc, tried = _run(tmp_path, fail_zones="zone-a zone-b", zones="zone-a zone-b")

    assert tried == ["zone-a", "zone-b"]
    assert proc.returncode != 0
    assert "nothing is billing" in (proc.stdout + proc.stderr)


def test_refuses_to_launch_when_an_instance_already_exists(tmp_path):
    """Structural backstop, independent of the loop: a second create is
    refused whatever caused it. On 2026-09-03 only the GPUS_ALL_REGIONS=1
    quota would have prevented triple billing, and a quota is not a plan."""
    proc, tried = _run(tmp_path, fail_zones="", zones="zone-a",
                       existing="attnbench-l4-compile-old  asia-south1-c  RUNNING")

    assert tried == [], "must not attempt a create while an instance exists"
    assert proc.returncode != 0
    assert "REFUSING TO LAUNCH" in (proc.stdout + proc.stderr)


def test_the_break_line_is_still_present():
    """Cheap guard against the fix being refactored away -- the behavioural
    tests above would catch it, but this names the exact line so a reader
    editing the loop sees why it matters."""
    src = SCRIPT.read_text()
    assert "break" in src and "do not remove" in src


@pytest.mark.parametrize("stale", ["attnbench-l4-validation-image-v2-20260902"])
def test_default_source_image_is_not_the_deleted_v2(stale):
    """v2 was deleted on 2026-09-03 once v3 superseded it and the GQA gate
    passed. A default pointing at it would fail closed at launch, wasting a
    session's opening minutes."""
    assert stale not in SCRIPT.read_text()
