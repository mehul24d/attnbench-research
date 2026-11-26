"""The T2.20 preflight's accepted L4 hosts (adviser's rulings of 23 Nov 2026):
P1's GCP g2-standard-8 by default, AWS's g6.2xlarge for P3 when named, any
other name refused; and the file stays outside the timed-path pin."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t227_steps as S  # noqa: E402


def test_the_default_is_p1s_host(monkeypatch):
    monkeypatch.delenv(S.EXPECTED_HOST_ENV, raising=False)
    assert S.expected_host() == ("gcp", "g2-standard-8")


def test_p3s_host_is_accepted_when_named(monkeypatch):
    monkeypatch.setenv(S.EXPECTED_HOST_ENV, "aws:g6.2xlarge")
    assert S.expected_host() == ("aws", "g6.2xlarge")


@pytest.mark.parametrize("name", ["aws:g6.xlarge", "gcp:a2-highgpu-1g", "g6.2xlarge", "aws:"])
def test_any_other_host_is_refused(monkeypatch, name):
    monkeypatch.setenv(S.EXPECTED_HOST_ENV, name)
    with pytest.raises(S.StepFailed, match="not an accepted L4 host"):
        S.expected_host()


def test_the_preflight_reads_its_expectation_from_the_table():
    src = (ROOT / "scripts" / "t227_steps.py").read_text()
    body = src[src.index("def step_t21_t220"):src.index("densities = None", src.index("def step_t21_t220"))]
    assert "cloud, itype = expected_host()" in body
    assert '"cloud_provider": cloud, "instance_type": itype' in body
    assert '"cloud_provider": "gcp"' not in body


def test_the_file_is_outside_the_timed_path_pin():
    import timed_path
    assert "scripts/t227_steps.py" not in timed_path.pin_closure("HEAD")
