"""scripts/gcp_bootstrap_instance.sh, run against a fake `gcloud`.

T1.1's two sessions copied the RULER data, installed the pinned wonderwords
and nltk, and installed XAttention by hand. The script does it in one
order, with the pin checks on both sides of the copy. These tests check the
order, that a local copy failing its pins is never shipped, that the
versions come from `ruler_data.PINNED_VERSIONS`, and that a missing marker
stops the run. They never touch GCP.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from attnbench.accuracy import ruler_data

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "gcp_bootstrap_instance.sh"
PIN = "pinned-proj-123"
PINS = " ".join(f"{k}=={v}" for k, v in sorted(ruler_data.PINNED_VERSIONS.items()))

# Answers each remote step with its marker. The instance's copy of the data
# verifies once the copy has been made, or from the start if FAKE_HAS_DATA
# is set; FAKE_FAIL names one marker that is withheld.
FAKE_GCLOUD = r"""#!/usr/bin/env bash
args="$*"; printf '%s\n' "${args//$'\n'/ }" >> "$GCLOUD_LOG"
say() { [[ "$1" == "${FAKE_FAIL:-}" ]] || echo "$1"; }
case "$*" in
  "config get-value project") echo "$FAKE_ACTIVE"; exit 0 ;;
  *"compute scp"*) touch "$STATE_COPIED"; exit 0 ;;
  *"compute ssh"*"PINS_OK"*)
     if [[ -e "$STATE_COPIED" || -n "${FAKE_HAS_DATA:-}" ]]; then say PINS_OK; else echo "missing"; fi ;;
  *"compute ssh"*"AREA_READY"*) say AREA_READY ;;
  *"compute ssh"*"PKGS_OK"*) say PKGS_OK ;;
  *"compute ssh"*"RULER_OK"*) say RULER_OK ;;
  *"compute ssh"*"XATTN_OK"*) say XATTN_OK ;;
  *"compute ssh"*"FA3_OK"*) say FA3_OK ;;
  *"compute ssh"*"TREE_CLEAN"*) say TREE_CLEAN ;;
esac
exit 0
"""

FAKE_PYTHON = r"""#!/usr/bin/env bash
case "$*" in
  *"len(r.PINNED_SHA256)"*) echo 3 ;;
  *"PINNED_VERSIONS"*) exec "$REAL_PYTHON" "$@" ;;
  *) cat >/dev/null; [[ -n "${FAKE_LOCAL_BAD:-}" ]] && exit 1; echo "$FAKE_RULER_DIR" ;;
esac
"""


def _run(tmp_path, *args, python="fake", **env_extra):
    import sys
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gcloud").write_text(FAKE_GCLOUD)
    (bin_dir / "gcloud").chmod(0o755)
    (bin_dir / "fake_python").write_text(FAKE_PYTHON)
    (bin_dir / "fake_python").chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.write_text("")
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GCP_", "ATTNBENCH_", "CLOUDSDK_", "FAKE_", "FA3_"))}
    env.update(PATH=os.pathsep.join([str(bin_dir), *real]), GCLOUD_LOG=str(log),
               FAKE_ACTIVE=PIN, ATTNBENCH_GCP_PROJECT=PIN,
               STATE_COPIED=str(tmp_path / "copied"), REAL_PYTHON=sys.executable,
               FAKE_RULER_DIR=str(tmp_path / "ruler"),
               ATTNBENCH_PYTHON=(str(bin_dir / "fake_python") if python == "fake" else python))
    env.update(env_extra)
    p = subprocess.run(["bash", str(SCRIPT), "inst", "zone", *args], cwd=REPO,
                       capture_output=True, text=True, env=env, timeout=120)
    calls = [c for c in log.read_text().splitlines() if c != "config get-value project"]
    return p, calls


def _step(calls, marker):
    return next(i for i, c in enumerate(calls) if marker in c)


def test_a_local_copy_that_fails_its_pins_is_never_shipped(tmp_path):
    p, calls = _run(tmp_path, FAKE_LOCAL_BAD="1")
    assert p.returncode == 1
    assert "local RULER data does not verify" in p.stderr
    assert calls == []


def test_the_real_pin_check_refuses_an_empty_data_directory(tmp_path):
    """With the real interpreter and an empty data directory, so the
    refusal comes from ruler_data's own SHA-256 check, not the fake."""
    import sys
    empty = tmp_path / "empty"
    empty.mkdir()
    p, calls = _run(tmp_path, python=sys.executable, ATTNBENCH_RULER_DATA=str(empty))
    assert p.returncode == 1 and calls == [], p.stderr[-800:]


def test_the_full_run_goes_in_order_with_the_pinned_versions(tmp_path):
    p, calls = _run(tmp_path)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    assert "BOOTSTRAP_OK inst" in p.stdout
    order = [_step(calls, "PINS_OK"), _step(calls, "AREA_READY"),
             _step(calls, "compute scp"), _step(calls, "PKGS_OK"),
             _step(calls, "RULER_OK"), _step(calls, "XATTN_OK"), _step(calls, "TREE_CLEAN")]
    assert order == sorted(order), calls
    # the copy is verified on the instance after it lands
    assert any("PINS_OK" in c for c in calls[_step(calls, "compute scp"):])
    pip = calls[_step(calls, "PKGS_OK")]
    assert f"pip install --user -q {PINS} " in pip, pip
    xattn = calls[_step(calls, "XATTN_OK")]
    assert "run_phase.sh bootstrap_xattention -- bash scripts/install_xattention.sh" in xattn
    assert all(f"--project={PIN}" in c for c in calls)


def test_a_copy_already_on_the_instance_that_verifies_is_not_copied_again(tmp_path):
    p, calls = _run(tmp_path, FAKE_HAS_DATA="1")
    assert p.returncode == 0, p.stderr[-1500:]
    assert not any("compute scp" in c or "AREA_READY" in c for c in calls)


@pytest.mark.parametrize("marker", ["AREA_READY", "PKGS_OK", "RULER_OK", "XATTN_OK", "TREE_CLEAN"])
def test_a_missing_marker_stops_the_run_there(marker, tmp_path):
    p, calls = _run(tmp_path, FAKE_FAIL=marker)
    assert p.returncode == 1
    assert "BOOTSTRAP_OK" not in p.stdout
    sequence = ["AREA_READY", "compute scp", "PKGS_OK", "RULER_OK", "XATTN_OK", "TREE_CLEAN"]
    for after in sequence[sequence.index(marker) + 1:]:
        assert not any(after in c for c in calls), (marker, after, calls)


def test_a_copy_that_fails_on_the_instance_stops_before_the_packages(tmp_path):
    p, calls = _run(tmp_path, FAKE_FAIL="PINS_OK")
    assert p.returncode == 1
    assert "does not verify on the instance" in p.stderr
    assert any("compute scp" in c for c in calls)
    assert not any("PKGS_OK" in c for c in calls)


def test_no_xattention_skips_only_the_install(tmp_path):
    p, calls = _run(tmp_path, "--no-xattention")
    assert p.returncode == 0, p.stderr[-1500:]
    assert not any("XATTN_OK" in c or "install_xattention" in c for c in calls)
    assert any("TREE_CLEAN" in c for c in calls)


def test_the_local_check_passes_on_the_real_data_when_it_is_present(tmp_path):
    """The success path with the real interpreter and the real pins, where
    this machine has the RULER data (the fresh-clone count skips it)."""
    import sys
    try:
        for name in ruler_data.PINNED_SHA256:
            ruler_data._verified(name)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no verified RULER data here ({type(e).__name__})")
    p, calls = _run(tmp_path, "--no-xattention", python=sys.executable,
                    **({"ATTNBENCH_RULER_DATA": os.environ["ATTNBENCH_RULER_DATA"]}
                       if "ATTNBENCH_RULER_DATA" in os.environ else {}))
    assert p.returncode == 0, p.stderr[-1500:]
    assert f"{len(ruler_data.PINNED_SHA256)} pinned files verified locally" in p.stdout
    assert str(ruler_data.data_dir()) not in p.stdout  # the local path is not printed


# --- the FA3 step (T6.4, 28 Oct 2026) ------------------------------------------------

SHA = "ab" * 32


def test_the_fa3_wheel_is_checked_after_xattention_and_before_the_tree_check(tmp_path):
    p, calls = _run(tmp_path, FA3_WHEEL_SHA256=SHA)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    fa3 = _step(calls, "FA3_OK")
    assert _step(calls, "XATTN_OK") < fa3 < _step(calls, "TREE_CLEAN")
    assert f"bash scripts/fa3_import_check.sh {SHA} " in calls[fa3]


def test_without_a_recorded_hash_the_fa3_step_is_skipped_and_says_so(tmp_path):
    p, calls = _run(tmp_path)
    assert p.returncode == 0, p.stderr[-1500:]
    assert not any("fa3_import_check" in c for c in calls)
    assert "FA3 skipped: FA3_WHEEL_SHA256 is not set" in p.stdout
    assert 'dense_set_incomplete="fa3:no wheel"' in p.stdout


@pytest.mark.parametrize("bad", ["abc", "AB" * 32, SHA + "0"])
def test_a_malformed_hash_refuses_before_anything_runs(bad, tmp_path):
    p, calls = _run(tmp_path, FA3_WHEEL_SHA256=bad)
    assert p.returncode == 1 and "not a sha256" in p.stderr
    assert calls == []


def test_a_failed_fa3_check_stops_before_the_tree_check(tmp_path):
    p, calls = _run(tmp_path, FA3_WHEEL_SHA256=SHA, FAKE_FAIL="FA3_OK")
    assert p.returncode == 1 and "FA3 import check" in p.stderr
    assert not any("TREE_CLEAN" in c for c in calls)
