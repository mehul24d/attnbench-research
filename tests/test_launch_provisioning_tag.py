"""Every create call tags the VM with the provisioning model it asks for (T2.20).

`provenance._cloud_identity` reads the tag back from the metadata server and
cross-checks it against `scheduling/preemptible`. The tag must come from the
create call that succeeded, so it is passed on that call with the same value
as its `--provisioning-model`. The A100 launcher's Spot fallback is a second
create call and must carry SPOT, not the FLEX_START of the first attempt
(researcher's decision, 25 Oct 2026).

Each launcher runs against a fake `gcloud` that records its arguments and
refuses every create, so the A100 script reaches its fallback and nothing
reaches GCP. `date` is faked to the last launch date, which the policy
allows.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from attnbench import provenance

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

FAKE_GCLOUD = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
  *"config get-value project"*) echo "test-project"; exit 0 ;;
  *"instances describe"*)       echo "ERROR: (gcloud.compute.instances.describe) The resource 'x' was not found" >&2; exit 1 ;;
  *"instances create"*)         echo "fake gcloud: create refused" >&2; exit 1 ;;
esac
exit 1
"""

FAKE_DATE = """#!/bin/bash
if [[ "$*" == "+%F" ]]; then echo {today}; else exec /bin/date "$@"; fi
"""

# (script, extra environment, the models its create calls must ask for, in order)
CASES = [
    ("gcp_launch_l4.sh", {}, ["STANDARD"]),
    ("gcp_launch_fa3_build.sh", {}, ["STANDARD"]),
    ("gcp_launch_h100.sh", {}, ["FLEX_START"]),
    ("gcp_launch_a100.sh", {}, ["FLEX_START", "SPOT"]),
]
# `gcp_launch_compile_session.sh` needs a fuller fake (machine images, zone
# retries); its tag is checked with that fake in
# `test_launch_script_zone_retry.py`.
CHECKED_ELSEWHERE = {"gcp_launch_compile_session.sh"}


def _last_date() -> str:
    m = re.search(r'^LAST_GCP_LAUNCH_DATE="(\d{4}-\d{2}-\d{2})"$',
                  (SCRIPTS / "_launch_policy.sh").read_text(), re.M)
    assert m
    return m.group(1)


def _creates(script: str, extra: dict, tmp_path: Path) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("gcloud", FAKE_GCLOUD),
                       ("date", FAKE_DATE.format(today=_last_date()))):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.touch()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
           "GCLOUD_LOG": str(log), "GCP_PROJECT": "test-project",
           "ATTNBENCH_GCP_PROJECT": "test-project",
           "GCP_RATE_INR_HR": "297.47", **extra}
    subprocess.run(["bash", str(SCRIPTS / script), "test-instance"],
                   input="launch\n", capture_output=True, text=True, env=env,
                   cwd=tmp_path, timeout=120)
    return [l for l in log.read_text().splitlines() if "instances create" in l]


def _flag(call: str, name: str) -> list[str]:
    return re.findall(rf"--{name}=(\S+)", call)


@pytest.mark.parametrize("script,extra,models", CASES, ids=[c[0] for c in CASES])
def test_each_create_is_tagged_with_its_own_model(script, extra, models, tmp_path):
    creates = _creates(script, extra, tmp_path)
    assert len(creates) == len(models), (script, creates)
    for call, model in zip(creates, models):
        assert _flag(call, "provisioning-model") == [model], call
        assert _flag(call, "metadata") == [f"{provenance.PROVISIONING_TAG}={model}"], call


def test_every_tag_value_is_one_provenance_reads():
    for _, _, models in CASES:
        for m in models:
            assert m in provenance.PROVISIONING_MODELS


def test_the_case_list_covers_every_launcher_that_measures():
    """A launcher added later without a case here would create untagged VMs,
    and every row from it would carry provisioning_model=None. The boot test
    creates a VM to check an image boots and measures nothing."""
    creating = {p.name for p in SCRIPTS.glob("*.sh")
                if "gcloud compute instances create" in p.read_text()}
    assert creating - {c[0] for c in CASES} - CHECKED_ELSEWHERE == {"gcp_boot_test_image.sh"}
