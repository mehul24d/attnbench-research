"""T2.20: microcode, instance, provisioning model and `sys.flags.optimize`
are in every provenance stamp.

The provisioning model comes from the tag the launch script's successful
create call wrote, cross-checked against GCP's `scheduling/preemptible`; any
disagreement or gap records None with the reason (researcher's decision, 25
Oct 2026). Every metadata read is faked: DMI files in tmp_path and a dict for
the metadata server. All CPU-only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench import provenance

REPO = Path(__file__).resolve().parents[1]

CPUINFO = """processor\t: 0
model name\t: Intel(R) Xeon(R) Platinum 8273CL CPU @ 2.20GHz
microcode\t: 0x5003604

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8273CL CPU @ 2.20GHz
microcode\t: 0x5003604
"""


def test_microcode_is_read_from_cpuinfo(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text(CPUINFO)
    assert provenance._cpu_microcode(str(f)) == "0x5003604"


def test_a_mixed_microcode_host_lists_every_revision(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text(CPUINFO.replace("microcode\t: 0x5003604", "microcode\t: 0x5003707", 1))
    got = provenance._cpu_microcode(str(f))
    assert got is not None and set(got.split(",")) == {"0x5003707", "0x5003604"}


def test_no_microcode_line_and_no_file_are_none(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nmodel name\t: x\n")
    assert provenance._cpu_microcode(str(f)) is None
    assert provenance._cpu_microcode(str(tmp_path / "absent")) is None


def test_the_cpu_probe_and_the_stamp_read_microcode_the_same_way(monkeypatch):
    """`host_cpu_probe builders` writes `cpu_microcode` on its rows and then
    merges the stamp, and `stamp_onto` raises on a disagreement, after the
    measurement and before the parquet is written."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import host_cpu_probe
    finally:
        sys.path.remove(str(REPO / "scripts"))
    df = pd.DataFrame([{"cpu_microcode": host_cpu_probe.cpu_identity().get("microcode")}])
    provenance.stamp_onto(df, {"cpu_microcode": provenance._cpu_microcode()})


# --- cloud identity -----------------------------------------------------------

def _dmi(tmp_path, vendor=None, product=None):
    d = tmp_path / "dmi"
    d.mkdir()
    if vendor is not None:
        (d / "sys_vendor").write_text(vendor + "\n")
    if product is not None:
        (d / "product_name").write_text(product + "\n")
    return str(d)


def _gce(tmp_path):
    return _dmi(tmp_path, "Google", "Google Compute Engine")


class FakeMetadata:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        return self.values.get(path)


def _md(tag=None, preemptible="FALSE",
        machine_type="projects/123456789012/machineTypes/g2-standard-8"):
    v = {"machine-type": machine_type, "scheduling/preemptible": preemptible}
    if tag is not None:
        v[f"attributes/{provenance.PROVISIONING_TAG}"] = tag
    return FakeMetadata(v)


def test_off_cloud_nothing_is_requested(tmp_path):
    fetch = FakeMetadata({})
    got = provenance._cloud_identity(str(tmp_path / "no-dmi"), fetch)
    assert fetch.calls == []
    assert got["cloud_provider"] is None and got["provisioning_model"] is None
    assert "no cloud instance" in got["provisioning_model_problem"]


@pytest.mark.parametrize("tag,pre,model", [
    ("STANDARD", "FALSE", "on-demand"),
    ("FLEX_START", "FALSE", "dws-flex-start"),
    ("SPOT", "TRUE", "spot"),
    ("RESERVED", "FALSE", "reserved"),
])
def test_an_agreeing_tag_gives_the_model(tmp_path, tag, pre, model):
    got = provenance._cloud_identity(_gce(tmp_path), _md(tag, pre))
    assert got["cloud_provider"] == "gcp"
    assert got["provisioning_model"] == model
    assert got["provisioning_model_problem"] is None


@pytest.mark.parametrize("tag,pre", [
    ("SPOT", "FALSE"),          # a Spot tag on a VM GCP says is not preemptible
    ("FLEX_START", "TRUE"),     # the A100 fallback's failure: Spot labelled Flex
    ("STANDARD", "TRUE"),
])
def test_a_disagreeing_tag_records_none_and_why(tmp_path, tag, pre):
    got = provenance._cloud_identity(_gce(tmp_path), _md(tag, pre))
    assert got["provisioning_model"] is None
    assert tag in got["provisioning_model_problem"]
    assert pre in got["provisioning_model_problem"]


def test_a_missing_tag_is_none_not_a_guess(tmp_path):
    got = provenance._cloud_identity(_gce(tmp_path), _md(None, "FALSE"))
    assert got["provisioning_model"] is None
    assert provenance.PROVISIONING_TAG in got["provisioning_model_problem"]


def test_an_unknown_tag_is_none(tmp_path):
    got = provenance._cloud_identity(_gce(tmp_path), _md("PREEMPTIBLE", "TRUE"))
    assert got["provisioning_model"] is None and "unknown" in got["provisioning_model_problem"]


def test_an_unreadable_preemptible_flag_is_none(tmp_path):
    got = provenance._cloud_identity(_gce(tmp_path), _md("STANDARD", None))
    assert got["provisioning_model"] is None
    assert "cross-checked" in got["provisioning_model_problem"]


@pytest.mark.parametrize("machine_type,generation", [
    ("projects/123456789012/machineTypes/g2-standard-8", "g2"),
    ("projects/123456789012/machineTypes/a2-ultragpu-1g", "a2"),
    ("projects/123456789012/machineTypes/a3-highgpu-1g", "a3"),
    ("projects/123456789012/machineTypes/n2d-standard-4", "n2d"),
    ("projects/123456789012/machineTypes/custom-8-32768", None),
])
def test_the_instance_type_and_its_series(tmp_path, machine_type, generation):
    got = provenance._cloud_identity(_gce(tmp_path),
                                     _md("STANDARD", "FALSE", machine_type))
    assert got["instance_type"] == machine_type.rsplit("/", 1)[-1]
    assert got["instance_generation"] == generation


def test_the_project_number_never_reaches_a_row(tmp_path):
    got = provenance._cloud_identity(_gce(tmp_path), _md("STANDARD", "FALSE"))
    assert not any("123456789012" in str(v) for v in got.values())


def test_aws_is_recognised_and_not_read(tmp_path):
    fetch = FakeMetadata({})
    got = provenance._cloud_identity(_dmi(tmp_path, "Amazon EC2", "g6.xlarge"), fetch)
    assert fetch.calls == []
    assert got["cloud_provider"] == "aws" and got["provisioning_model"] is None
    assert "T5.1" in got["provisioning_model_problem"]


# --- the stamp ------------------------------------------------------------------

T220_FIELDS = ("cpu_microcode", "cloud_provider", "instance_type",
               "instance_generation", "provisioning_model",
               "provisioning_model_problem", "python_optimize")


def test_capture_carries_every_t220_field():
    d = provenance.capture().to_dict()
    missing = [f for f in T220_FIELDS if f not in d]
    assert not missing
    assert d["python_optimize"] == sys.flags.optimize


def test_capture_takes_the_cloud_fields_from_the_reader(monkeypatch):
    fake = {"cloud_provider": "gcp", "instance_type": "g2-standard-8",
            "instance_generation": "g2", "provisioning_model": "on-demand",
            "provisioning_model_problem": None}
    monkeypatch.setattr(provenance, "_cloud_identity_cached", lambda: fake)
    d = provenance.capture().to_dict()
    assert {k: d[k] for k in fake} == fake


def test_python_optimize_is_observed_not_assumed():
    """Under `python -O` the stamp must say 1; a constant 0 would pass the
    in-process test above on every machine."""
    out = subprocess.run(
        [sys.executable, "-O", "-c",
         "from attnbench import provenance; print(provenance.capture().python_optimize)"],
        capture_output=True, text=True, cwd=REPO, timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "1"
