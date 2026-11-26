"""scripts/aws_bootstrap_instance.sh against fake ssh, scp and a fake local
Python (adviser's rulings of 23 Nov 2026: the AWS bootstrap as a repository
script with a test, before the P3 launch)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aws_bootstrap_instance.sh"

FAKE_SSH = r'''#!/bin/bash
cmd="${@: -1}"
echo "SSH $cmd" >> "$CALLS"
m="$(grep -oE 'echo [A-Z_]+$' <<<"$cmd" | awk '{print $2}')"
if [[ "$m" == PINS_OK && -z "${FAKE_REMOTE_HAS_DATA:-}" ]]; then
  n=$(cat "$STATE" 2>/dev/null || echo 0); echo $((n + 1)) > "$STATE"
  [[ $n -eq 0 ]] && exit 1
fi
[[ " ${FAKE_FAIL:-} " == *" $m "* ]] && { echo "boom"; exit 1; }
echo "$m"
'''
FAKE_SCP = '#!/bin/bash\necho "SCP $*" >> "$CALLS"\n[[ -n "${FAKE_SCP_FAIL:-}" ]] && exit 1\nexit 0\n'
FAKE_PY = r'''#!/bin/bash
if [[ "$1" == - ]]; then cat >/dev/null; [[ -n "${FAKE_LOCAL_BAD:-}" ]] && exit 1; echo /fake/ruler_data; exit 0; fi
case "$2" in *PINNED_VERSIONS*) echo "nltk==3.9.1 wonderwords==2.2.0";; *LIGHTGBM*) echo 4.6.0;; esac
'''


def _run(tmp_path, **env):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    for name, body in (("ssh", FAKE_SSH), ("scp", FAKE_SCP), ("fakepy", FAKE_PY)):
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    calls = tmp_path / "calls.txt"
    p = subprocess.run(["bash", str(SCRIPT), "ubuntu@host"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "CALLS": str(calls),
                            "STATE": str(tmp_path / "state"), "AWS_SSH_KEY": "/fake/key",
                            "ATTNBENCH_PYTHON": str(b / "fakepy"), **env})
    lines = calls.read_text().splitlines() if calls.exists() else []
    return p, lines


def _markers(lines):
    out = []
    for ln in lines:
        if ln.startswith("SCP"):
            out.append("scp")
        else:
            out.append(ln.rsplit("echo ", 1)[-1])
    return out


def test_a_fresh_instance_gets_the_data_the_packages_and_every_check(tmp_path):
    p, lines = _run(tmp_path)
    assert p.returncode == 0 and p.stdout.strip().endswith("BOOTSTRAP_OK"), p.stdout + p.stderr
    assert _markers(lines) == ["PINS_OK", "AREA_READY", "scp", "PINS_OK", "PKGS_OK",
                               "RULER_OK", "TREE_CLEAN"]
    pkgs = next(ln for ln in lines if ln.endswith("PKGS_OK"))
    assert "/usr/bin/python3 -m pip install --user -q nltk==3.9.1 wonderwords==2.2.0 lightgbm==4.6.0" in pkgs
    scp = next(ln for ln in lines if ln.startswith("SCP"))
    assert "/fake/ruler_data ubuntu@host:~/.cache/attnbench/ruler_data" in scp and "-i /fake/key" in scp


def test_data_that_already_verifies_is_not_copied(tmp_path):
    p, lines = _run(tmp_path, FAKE_REMOTE_HAS_DATA="1")
    assert p.returncode == 0 and "scp" not in _markers(lines)


def test_a_bad_local_copy_is_refused_before_any_contact(tmp_path):
    p, lines = _run(tmp_path, FAKE_LOCAL_BAD="1")
    assert p.returncode == 1 and "REFUSING" in p.stderr and lines == []


@pytest.mark.parametrize("fail,ran", [
    ("AREA_READY", ["PINS_OK", "AREA_READY"]),
    ("PKGS_OK", ["PINS_OK", "AREA_READY", "scp", "PINS_OK", "PKGS_OK"]),
    ("RULER_OK", ["PINS_OK", "AREA_READY", "scp", "PINS_OK", "PKGS_OK", "RULER_OK"]),
    ("TREE_CLEAN", ["PINS_OK", "AREA_READY", "scp", "PINS_OK", "PKGS_OK", "RULER_OK", "TREE_CLEAN"]),
])
def test_a_missing_marker_stops_the_bootstrap(tmp_path, fail, ran):
    p, lines = _run(tmp_path, FAKE_FAIL=fail)
    assert p.returncode == 1 and "BOOTSTRAP_OK" not in p.stdout
    assert _markers(lines) == ran


def test_a_failed_copy_stops_before_the_packages(tmp_path):
    p, lines = _run(tmp_path, FAKE_SCP_FAIL="1")
    assert p.returncode == 1 and _markers(lines) == ["PINS_OK", "AREA_READY", "scp"]
