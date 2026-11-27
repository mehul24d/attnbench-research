"""The T2.20 preflight's accepted L4 hosts (adviser's rulings of 23 Nov 2026):
P1's GCP g2-standard-8 by default, AWS's g6.2xlarge for P3 when named, any
other name refused; P6's AWS p5.4xlarge with the H100 when named (26 Nov
2026); and the file stays outside the timed-path pin."""
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
    with pytest.raises(S.StepFailed, match="not an accepted host"):
        S.expected_host()


def test_p6s_host_is_accepted_when_named_and_requires_the_h100(monkeypatch):
    monkeypatch.setenv(S.EXPECTED_HOST_ENV, "aws:p5.4xlarge")
    assert S.expected_host() == ("aws", "p5.4xlarge")
    assert S.expected_gpu() == "H100"


@pytest.mark.parametrize("name", [None, "gcp:g2-standard-8", "aws:g6.2xlarge"])
def test_the_l4_hosts_require_the_l4(monkeypatch, name):
    if name is None:
        monkeypatch.delenv(S.EXPECTED_HOST_ENV, raising=False)
    else:
        monkeypatch.setenv(S.EXPECTED_HOST_ENV, name)
    assert S.expected_gpu() == "L4"


def test_the_preflight_reads_its_expectation_from_the_table():
    src = (ROOT / "scripts" / "t227_steps.py").read_text()
    body = src[src.index("def step_t21_t220"):src.index("densities = None", src.index("def step_t21_t220"))]
    assert "cloud, itype = expected_host()" in body
    assert '"cloud_provider": cloud, "instance_type": itype' in body
    assert '"cloud_provider": "gcp"' not in body
    assert "gpu = expected_gpu()" in body and "EXPECTED_GPU not in" not in body


def test_the_file_is_outside_the_timed_path_pin():
    import timed_path
    assert "scripts/t227_steps.py" not in timed_path.pin_closure("HEAD")


# --- P6 on Spot (the researcher's task of 27 Nov 2026): spot only on P6's host,
# only when the session names it; on-demand otherwise ------------------------------

@pytest.mark.parametrize("host", [None, "gcp:g2-standard-8", "aws:g6.2xlarge", "aws:p5.4xlarge"])
def test_the_default_provisioning_is_on_demand(monkeypatch, host):
    monkeypatch.delenv(S.EXPECTED_PROVISIONING_ENV, raising=False)
    if host is None:
        monkeypatch.delenv(S.EXPECTED_HOST_ENV, raising=False)
    else:
        monkeypatch.setenv(S.EXPECTED_HOST_ENV, host)
    assert S.expected_provisioning() == "on-demand"


def test_spot_is_accepted_on_p6s_host_when_named(monkeypatch):
    monkeypatch.setenv(S.EXPECTED_HOST_ENV, "aws:p5.4xlarge")
    monkeypatch.setenv(S.EXPECTED_PROVISIONING_ENV, "spot")
    assert S.expected_provisioning() == "spot"


@pytest.mark.parametrize("host,want", [(None, "spot"), ("aws:g6.2xlarge", "spot"),
                                       ("gcp:g2-standard-8", "spot"), ("aws:p5.4xlarge", "SPOT"),
                                       ("aws:p5.4xlarge", "dws-flex-start")])
def test_any_other_provisioning_is_refused(monkeypatch, host, want):
    if host is None:
        monkeypatch.delenv(S.EXPECTED_HOST_ENV, raising=False)
    else:
        monkeypatch.setenv(S.EXPECTED_HOST_ENV, host)
    monkeypatch.setenv(S.EXPECTED_PROVISIONING_ENV, want)
    with pytest.raises(S.StepFailed, match="is not accepted on"):
        S.expected_provisioning()


def test_the_preflight_reads_its_provisioning_from_the_rule():
    src = (ROOT / "scripts" / "t227_steps.py").read_text()
    body = src[src.index("def step_t21_t220"):src.index("densities = None", src.index("def step_t21_t220"))]
    assert '"provisioning_model": expected_provisioning(),' in body
