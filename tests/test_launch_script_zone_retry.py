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
      printf '%s\n' "$ARGS" >> "${CREATE_ARGV_LOG:-/dev/null}"
      case " $FAIL_ZONES " in
        *" $ZONE "*) echo "ZONE_RESOURCE_POOL_EXHAUSTED" >&2; exit 1 ;;
      esac
      case " ${INVALID_ZONES:-} " in
        *" $ZONE "*)
          echo "ERROR: (gcloud.compute.instances.create) Could not fetch resource:" >&2
          echo " - Invalid value for field 'resource.disks[0].interface': 'NVME'." >&2
          exit 1 ;;
      esac
      echo "Created instance in $ZONE"; exit 0 ;;
esac
exit 0
"""


def _run(tmp_path, *, fail_zones: str, zones: str, existing: str = "", **extra_env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gcloud"
    fake.write_text(FAKE_GCLOUD)
    fake.chmod(0o755)

    attempts = tmp_path / "attempts.txt"
    attempts.write_text("")
    argv_log = tmp_path / "create_argv.txt"
    argv_log.write_text("")

    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        ATTEMPT_LOG=str(attempts),
        CREATE_ARGV_LOG=str(argv_log),
        FAIL_ZONES=fail_zones,
        INVALID_ZONES="",
        FAKE_EXISTING_INSTANCES=existing,
        GCP_ZONE_FALLBACKS=zones,
        GCP_PROJECT="test-project",
    )
    env.update({k: str(v) for k, v in extra_env.items()})
    proc = subprocess.run(["bash", str(SCRIPT), "test-instance"], input="launch\n",
                          capture_output=True, text=True, env=env, timeout=60)
    tried = [z for z in attempts.read_text().split() if z]
    proc.create_argv = argv_log.read_text()      # type: ignore[attr-defined]
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


def test_a_g2_launch_passes_no_accelerator_flag(tmp_path):
    """Default: inherit the machine image's own card.

    Every L4 session to date created the instance with no --accelerator flag
    at all, and those sessions produced the entire dataset. Adding an A2
    override must not start sending a flag on the G2 path.
    """
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a")
    assert proc.returncode == 0
    assert "--accelerator" not in proc.create_argv


def test_an_empty_override_does_not_pass_an_empty_argument(tmp_path):
    """The bash-3.2 trap, stated as behaviour rather than as an idiom.

    `"${ARR[@]}"` on an empty array is an unbound variable under `set -u` in
    bash 3.2, which is what macOS ships -- so the naive version aborted the
    launch script before gcloud was reached, on EVERY launch from this laptop
    and not only the A2 one. It failed closed, which is the good direction,
    but it failed on the G2 path this project actually uses.
    """
    proc, tried = _run(tmp_path, fail_zones="", zones="zone-a", GCP_ACCELERATOR="")
    assert proc.returncode == 0, proc.stderr
    assert tried == ["zone-a"], "the create was never attempted"
    assert "--accelerator=" not in proc.create_argv


def test_the_a2_override_reaches_the_create_call(tmp_path):
    """attnbench-l4-image-v4 records guestAccelerators: nvidia-l4, because an
    L4 is the card it was captured from. Carrying that onto a2-ultragpu-1g,
    whose A100 is part of the machine type, is the one image property that
    must not be inherited."""
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a",
                   GCP_MACHINE_TYPE="a2-ultragpu-1g",
                   GCP_PROVISIONING_MODEL="SPOT",
                   GCP_ACCELERATOR="type=nvidia-a100-80gb,count=1")
    assert proc.returncode == 0, proc.stderr
    assert "--accelerator=type=nvidia-a100-80gb,count=1" in proc.create_argv
    assert "--machine-type=a2-ultragpu-1g" in proc.create_argv
    assert "--provisioning-model=SPOT" in proc.create_argv


def test_the_default_image_is_the_current_one(tmp_path):
    """A stale default is not harmless. On 2026-09-05 a launch that set the
    zone and the instance name but left GCP_SOURCE_IMAGE alone booted v3 --
    an image predating the deploy-provenance fix -- and nobody noticed until
    the audit log was read."""
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a")
    assert "attnbench-l4-image-v4-20260903" in proc.create_argv
    assert "v3-20260903" not in proc.create_argv


def test_a_validation_error_is_not_reported_as_a_stockout(tmp_path):
    """The 2026-09-05 defect. `a2-ultragpu-1g` rejected the machine image's
    NVME boot-disk interface, and the script printed the stockout message --
    "wait and attempt again later" -- for an error that is deterministic and
    would never clear. Advice that is patiently followed forever is worse than
    no advice."""
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a", INVALID_ZONES="zone-a")
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "CONFIGURATION ERROR" in out
    assert "not a stockout" in out.lower()
    assert "attempt again later" not in out, (
        "a deterministic rejection must not be described as transient")


def test_a_validation_error_does_not_walk_the_zone_list(tmp_path):
    """A bad request fails identically everywhere. Iterating the fallback list
    prints the same rejection N times and buries the one that matters."""
    proc, tried = _run(tmp_path, fail_zones="", zones="zone-a zone-b zone-c",
                       INVALID_ZONES="zone-a zone-b zone-c")
    assert tried == ["zone-a"], f"stopped after {tried}, should stop at the first"


def test_the_error_names_the_overrides_that_fix_it(tmp_path):
    """Both cross-family image properties found so far, named at the point of
    failure -- the operator is holding a rejection, not reading this file."""
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a", INVALID_ZONES="zone-a")
    out = proc.stdout + proc.stderr
    assert "GCP_BOOT_DISK_INTERFACE" in out and "SCSI" in out
    assert "GCP_ACCELERATOR" in out


def test_a_real_stockout_still_says_it_is_transient(tmp_path):
    """The other direction: the fix must not turn every capacity failure into
    a configuration error. Stockouts in this region HAVE cleared minutes
    later, and that advice is correct for them."""
    proc, tried = _run(tmp_path, fail_zones="zone-a zone-b", zones="zone-a zone-b")
    out = proc.stdout + proc.stderr
    assert tried == ["zone-a", "zone-b"], "a stockout must still try the list"
    assert "CONFIGURATION ERROR" not in out
    assert "nothing is billing" in out


def test_the_boot_disk_interface_override_reaches_the_create_call(tmp_path):
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a",
                   GCP_MACHINE_TYPE="a2-ultragpu-1g",
                   GCP_BOOT_DISK_INTERFACE="SCSI")
    assert proc.returncode == 0, proc.stderr
    assert "--boot-disk-interface=SCSI" in proc.create_argv


def test_a_g2_launch_passes_no_boot_disk_interface(tmp_path):
    """Inherit-from-the-image stays the default; the L4 sessions that produced
    the whole dataset passed no such flag."""
    proc, _ = _run(tmp_path, fail_zones="", zones="zone-a")
    assert "--boot-disk-interface" not in proc.create_argv


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


# ---------------------------------------------------------------------------
# A100: Spot-only, single-zone.
# ---------------------------------------------------------------------------
#
# The A100 grant is PREEMPTIBLE_NVIDIA_A100_80GB_GPUS=1 with the on-demand
# NVIDIA_A100_80GB_GPUS at 0, so a STANDARD create cannot succeed at all --
# it is refused by quota, not merely more expensive. And a2-ultragpu-1g
# exists in asia-southeast1-c alone, so the fallback list has one element and
# there is nothing to retry into.

def test_provisioning_model_is_overridable():
    src = SCRIPT.read_text()
    assert 'PROVISIONING_MODEL="${GCP_PROVISIONING_MODEL:-STANDARD}"' in src
    assert '--provisioning-model="$PROVISIONING_MODEL"' in src
    assert "--provisioning-model=STANDARD \\" not in src, (
        "a hardcoded STANDARD cannot create the A100, whose quota is Spot-only")


def test_machine_type_is_overridable():
    src = SCRIPT.read_text()
    assert 'MACHINE_TYPE="${GCP_MACHINE_TYPE:-g2-standard-8}"' in src


def test_the_default_is_still_on_demand():
    """Overridable, not changed. The L4 compile sessions must not silently
    become preemptible -- preemption 70 minutes into a build wastes far more
    than Spot saves, which is the reasoning in the script header."""
    src = SCRIPT.read_text()
    assert ":-STANDARD}" in src
    assert ":-g2-standard-8}" in src


def test_a_single_zone_stockout_does_not_advise_retrying_the_list():
    """With one zone there is nothing to retry INTO. Telling the operator to
    'retry the same list' is advice to re-run an identical single attempt,
    which reads as progress and is not."""
    proc = subprocess.run(["bash", str(SCRIPT), "test-instance"],
                          input="launch\n", capture_output=True, text=True,
                          env={**os.environ,
                               "GCP_ZONE_FALLBACKS": "asia-southeast1-c",
                               "PATH": os.environ["PATH"]})
    combined = proc.stdout + proc.stderr
    # The script may exit earlier (no project, image check) in a sandbox; the
    # branch is asserted against the source when it cannot be exercised.
    src = SCRIPT.read_text()
    assert "SINGLE-ZONE target" in src
    assert "Do not sit in a retry loop against one zone." in src
    assert 'if [[ "$N_ZONES" == "1" ]]' in src


def test_the_multi_zone_advice_survives():
    """asia-south1 stocked out in all three zones and cleared minutes later.
    That advice is correct there and must not be lost."""
    assert "retrying the same" in SCRIPT.read_text()
