"""The H100 gate retry loop (`scripts/gcp_h100_gate_retry.sh`; T6.4, 31 Oct 2026).

The loop runs in a copy of the repository against fakes: `gcloud` (the
project pin, the instance list, ssh), `sleep` (recorded, returns at once) and
every script it calls (each records its arguments and the environment the
loop gave it). The fake launcher plays a script of outcomes, one per call:
`4` no capacity anywhere, `0:ZONE` a grant, any other number an error.
Nothing reaches GCP.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SHA = "1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a"
CALLED = ("gcp_launch_h100.sh", "gcp_verify_gcs_writable.sh", "gcp_deploy_source.sh",
          "gcp_preflight_instance.sh", "gcp_bootstrap_instance.sh", "gcp_session_elapsed.sh",
          "gcp_teardown_session.sh", "gcp_cleanup_check.sh")
ENV_SEEN = ("GCP_H100_PROVISIONING", "GCP_MAX_RUN", "GCP_HALT_MINUTES", "GCP_RATE_INR_HR",
            "GCP_ZONE_FALLBACKS", "GCP_CANCEL_AFTER_SECONDS", "GCP_IMAGE", "FA3_WHEEL_SHA256")

FAKE_SCRIPT = r"""#!/usr/bin/env bash
me="$(basename "$0")"
{ printf '%s' "$me $*"; for v in %ENV%; do printf ' %s=%s' "$v" "${!v:-}"; done; echo; } >> "$CALLS"
if [[ "$me" == gcp_launch_h100.sh ]]; then
  n=$(( $(cat "$FAKE_DIR/launches" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_DIR/launches"
  out="$(sed -n "${n}p" "$FAKE_DIR/launch_plan")"
  echo "launcher output for $ATTNBENCH_GCP_PROJECT"
  case "$out" in
    0:*) echo "Granted in ${out#0:} after 60s of waiting (VM status RUNNING). Billing now."; exit 0 ;;
    "")  echo "fake: no outcome left" >&2; exit 9 ;;
    *)   exit "$out" ;;
  esac
fi
exit 0
""".replace("%ENV%", " ".join(ENV_SEEN))

FAKE_GCLOUD = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GCLOUD_LOG"
case "$*" in
  "config get-value project") echo "$ATTNBENCH_GCP_PROJECT"; exit 0 ;;
  *"instances list"*)
    n=$(( $(cat "$FAKE_DIR/lists" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_DIR/lists"
    row="$(sed -n "${n}p" "$FAKE_DIR/list_plan" 2>/dev/null)"
    case "$row" in
      FAIL) echo "ERROR: projects/$ATTNBENCH_GCP_PROJECT: network" >&2; exit 1 ;;
      ""|NONE) exit 0 ;;
      *) printf '%s\n' "$row" | tr '|' '\n'; exit 0 ;;
    esac ;;
  *"compute ssh"*"--command=true"*) exit 0 ;;
  *"compute ssh"*) exit "${GATE_RC:-0}" ;;
esac
exit 0
"""

FAKE_SLEEP = """#!/bin/bash
echo "$1" >> "$FAKE_DIR/slept"
"""


class Loop:
    def __init__(self, tmp_path: Path, launches: list[str], lists: list[str] = (), **env):
        self.root = tmp_path / "repo"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("gcp_h100_gate_retry.sh", "_gcp_project.sh"):
            shutil.copy(REPO / "scripts" / name, self.root / "scripts" / name)
        for name in CALLED:
            (self.root / "scripts" / name).write_text(FAKE_SCRIPT)
        (self.root / ".gitignore").write_text("logs/\n")
        git = ["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(git[:3] + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "x"], check=True)
        self.fake = tmp_path / "fake"
        self.fake.mkdir()
        (self.fake / "launch_plan").write_text("".join(f"{x}\n" for x in launches))
        (self.fake / "list_plan").write_text("".join(f"{x}\n" for x in lists))
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, body in (("gcloud", FAKE_GCLOUD), ("sleep", FAKE_SLEEP)):
            (bin_dir / name).write_text(body)
            (bin_dir / name).chmod(0o755)
        self.calls, self.gcloud = tmp_path / "calls.log", tmp_path / "gcloud.log"
        self.log = tmp_path / "retry.log"
        self.env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
                    "FAKE_DIR": str(self.fake), "CALLS": str(self.calls),
                    "GCLOUD_LOG": str(self.gcloud), "H100_RETRY_LOG": str(self.log),
                    "ATTNBENCH_GCP_PROJECT": "secret-proj-123", "FA3_WHEEL_SHA256": SHA,
                    "H100_SPOT_RATE_INR_HR": "400", **{k: str(v) for k, v in env.items()}}

    def run(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.root / "scripts" / "gcp_h100_gate_retry.sh")],
                              capture_output=True, text=True, env=self.env, timeout=120)

    def called(self, script: str = "") -> list[str]:
        if not self.calls.exists():
            return []
        return [l for l in self.calls.read_text().splitlines() if l.startswith(script)]

    def slept(self) -> list[int]:
        p = self.fake / "slept"
        return [int(x) for x in p.read_text().split()] if p.exists() else []


def _env(call: str, name: str) -> str:
    return call.split(f" {name}=", 1)[1].split(" ", 1)[0]


def test_it_retries_until_a_grant_then_runs_the_gate_and_tears_down(tmp_path):
    lp = Loop(tmp_path, ["4", "4", "4", "4", "0:asia-southeast1-b"])
    p = lp.run()
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    launches = lp.called("gcp_launch_h100.sh")
    assert [_env(c, "GCP_H100_PROVISIONING") for c in launches] == \
        ["FLEX_START", "SPOT", "FLEX_START", "SPOT", "FLEX_START"]
    assert all(_env(c, "GCP_ZONE_FALLBACKS") == "us-central1-a" for c in launches)  # first word
    assert len(lp.slept()) == 2 and all(1700 <= s <= 1800 for s in lp.slept())
    order = [c.split()[0] for c in lp.called()][len(launches):]
    assert order == ["gcp_verify_gcs_writable.sh", "gcp_deploy_source.sh",
                     "gcp_preflight_instance.sh", "gcp_bootstrap_instance.sh",
                     "gcp_session_elapsed.sh", "gcp_teardown_session.sh", "gcp_cleanup_check.sh"]
    assert "--no-xattention" in lp.called("gcp_bootstrap_instance.sh")[0]
    assert "asia-southeast1-b" in lp.called("gcp_teardown_session.sh")[0]
    gate = [l for l in lp.gcloud.read_text().splitlines()
            if "compute ssh" in l and "--command=true" not in l]
    assert len(gate) == 1 and "t64_session.py run --gate-only" in gate[0] and SHA in gate[0]


def test_the_zones_are_all_three_and_each_flex_request_waits_five_minutes(tmp_path):
    lp = Loop(tmp_path, ["0:us-central1-a"])
    assert lp.run().returncode == 0
    (launch,) = lp.called("gcp_launch_h100.sh")
    assert "us-central1-a asia-southeast1-b asia-southeast1-c" in launch
    assert _env(launch, "GCP_CANCEL_AFTER_SECONDS") == "300"
    assert _env(launch, "GCP_HALT_MINUTES") == "40"
    assert _env(launch, "GCP_IMAGE") == "attnbench-env-v6-20260917"


def test_the_cap_is_the_hard_delete_at_each_models_rate(tmp_path):
    lp = Loop(tmp_path, ["4", "0:us-central1-a"])
    assert lp.run().returncode == 0
    flex, spot = lp.called("gcp_launch_h100.sh")
    assert _env(flex, "GCP_MAX_RUN") == "32m" and _env(flex, "GCP_RATE_INR_HR") == "463.47"
    # Spot's hard delete is 40 min, whatever its rate (rule, 1 Nov 2026)
    assert _env(spot, "GCP_MAX_RUN") == "40m" and _env(spot, "GCP_RATE_INR_HR") == "400"
    assert "400" in lp.called("gcp_session_elapsed.sh")[0]          # costed at the Spot rate


@pytest.mark.parametrize("rate, cap", [("668.04", 446), ("629.94", 420), ("525.01", 351)])
def test_spot_is_refused_where_forty_minutes_cost_more_than_350(tmp_path, rate, cap):
    lp = Loop(tmp_path, ["4", "0:us-central1-a"], H100_SPOT_RATE_INR_HR=rate)
    p = lp.run()
    assert p.returncode == 0
    assert (f"Spot refused: 40 min at Rs {rate}/h cost Rs {cap}, above the Rs 350 "
            f"Spot maximum") in p.stdout
    assert [_env(c, "GCP_H100_PROVISIONING") for c in lp.called("gcp_launch_h100.sh")] == \
        ["FLEX_START", "FLEX_START"]


def test_spot_at_525_costs_350_and_is_allowed(tmp_path):
    lp = Loop(tmp_path, ["4", "0:us-central1-a"], H100_SPOT_RATE_INR_HR="525")
    p = lp.run()
    assert p.returncode == 0
    assert "Spot Rs 350 = hard delete after 40 min" in p.stdout
    assert [_env(c, "GCP_H100_PROVISIONING") for c in lp.called("gcp_launch_h100.sh")] == \
        ["FLEX_START", "SPOT"]


@pytest.mark.parametrize("env, why", [({"H100_SPOT_DELETE_MINUTES": "39"}, "never under 40"),
                                      ({"H100_SPOT_CAP_MAX_INR": "351"}, "at most Rs 350")])
def test_the_spot_rule_cannot_be_loosened(tmp_path, env, why):
    p = Loop(tmp_path, ["0:us-central1-a"], **env).run()
    assert p.returncode != 0 and why in p.stdout


def test_spot_off_is_flex_only(tmp_path):
    lp = Loop(tmp_path, ["4", "0:us-central1-a"], H100_SPOT="0", H100_SPOT_RATE_INR_HR="")
    assert lp.run().returncode == 0
    assert {_env(c, "GCP_H100_PROVISIONING") for c in lp.called("gcp_launch_h100.sh")} == \
        {"FLEX_START"}


def test_a_round_is_skipped_while_another_attnbench_gpu_vm_exists(tmp_path):
    lp = Loop(tmp_path, ["0:us-central1-a"],
              lists=["attnbench-l4-s3-1 asia-northeast1-a g2-standard-8 RUNNING",
                     "attnbench-a100-x asia-southeast1-c a2-ultragpu-1g TERMINATED",
                     "attnbench-fa3-build us-central1-a c2-standard-30 RUNNING"],
              H100_SPOT="0")
    p = lp.run()
    assert p.returncode == 0
    assert len(lp.called("gcp_launch_h100.sh")) == 1        # rounds 1 and 2 skipped
    assert len(lp.slept()) == 2
    assert p.stdout.count("skipped; an attnbench GPU VM exists") == 2


def test_an_unreadable_instance_list_skips_the_round(tmp_path):
    lp = Loop(tmp_path, ["0:us-central1-a"], lists=["FAIL"], H100_SPOT="0")
    p = lp.run()
    assert p.returncode == 0 and "could not be read" in p.stdout
    assert len(lp.called("gcp_launch_h100.sh")) == 1


@pytest.mark.parametrize("code", ["1", "3"])
def test_a_launcher_error_stops_the_loop_without_a_teardown(tmp_path, code):
    lp = Loop(tmp_path, [code, "0:us-central1-a"])
    p = lp.run()
    assert p.returncode == 1 and f"exited {code}" in p.stdout
    assert len(lp.called("gcp_launch_h100.sh")) == 1
    assert not lp.called("gcp_teardown_session.sh") and not lp.slept()


def test_a_failed_gate_is_still_torn_down(tmp_path):
    lp = Loop(tmp_path, ["0:us-central1-a"], GATE_RC="1")
    p = lp.run()
    assert p.returncode != 0
    assert lp.called("gcp_teardown_session.sh") and lp.called("gcp_cleanup_check.sh")


def test_it_stops_after_its_rounds(tmp_path):
    lp = Loop(tmp_path, ["4"] * 10, H100_SPOT="0", H100_MAX_ROUNDS="3")
    p = lp.run()
    assert p.returncode == 1 and "3 rounds without a grant" in p.stdout
    assert len(lp.called("gcp_launch_h100.sh")) == 3


@pytest.mark.parametrize("env, why", [
    ({"FA3_WHEEL_SHA256": ""}, "FA3_WHEEL_SHA256"),
    ({"H100_SPOT_RATE_INR_HR": ""}, "H100_SPOT_RATE_INR_HR"),
    ({"H100_CAP_INR": "200"}, "the gate needs 30"),
    ({"ATTNBENCH_GCP_PROJECT": ""}, "ATTNBENCH_GCP_PROJECT is not set"),
])
def test_it_refuses_before_any_round(tmp_path, env, why):
    lp = Loop(tmp_path, ["0:us-central1-a"], **env)
    p = lp.run()
    assert p.returncode == 1 and why in p.stdout + p.stderr
    assert not lp.called()


def test_a_dirty_tree_is_refused(tmp_path):
    lp = Loop(tmp_path, ["0:us-central1-a"])
    (lp.root / "scripts" / "gcp_deploy_source.sh").write_text("# edited\n")
    p = lp.run()
    assert p.returncode == 1 and "dirty" in p.stdout
    assert not lp.called()


def test_the_project_is_never_printed_or_logged(tmp_path):
    lp = Loop(tmp_path, ["4", "4", "0:us-central1-a"], lists=["FAIL"])
    p = lp.run()
    assert p.returncode == 0
    for text in (p.stdout, p.stderr, lp.log.read_text()):
        assert "secret-proj-123" not in text
