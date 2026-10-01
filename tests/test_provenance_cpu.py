"""The host CPU is in every provenance stamp (audit T1, 2026-10-01).

T1 asked which CPU each card's host had, and no stamp could answer: it was
rebuilt from instance metadata after the fact. These pin that the stamp now
answers it, on Linux (the rented hosts) and on macOS (the workstation).
"""

from __future__ import annotations

import os

from attnbench import provenance

GCP_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 85
model name\t: Intel(R) Xeon(R) Platinum 8273CL CPU @ 2.20GHz
stepping\t: 7

processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8273CL CPU @ 2.20GHz
"""


def test_the_linux_model_name_is_read_from_cpuinfo(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text(GCP_CPUINFO)
    assert provenance._cpu_model(str(f)) == "Intel(R) Xeon(R) Platinum 8273CL CPU @ 2.20GHz"


def test_a_missing_cpuinfo_falls_back_and_never_raises(tmp_path):
    got = provenance._cpu_model(str(tmp_path / "absent"))
    assert got is None or isinstance(got, str)


def test_a_cpuinfo_without_a_model_name_falls_back(tmp_path):
    """ARM Linux /proc/cpuinfo has no `model name` line; that must not be
    read as an empty string."""
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nBogoMIPS\t: 50.00\n")
    assert provenance._cpu_model(str(f)) != ""


def test_capture_stamps_the_cpu_of_this_machine():
    p = provenance.capture().to_dict()
    assert p["cpu_model"], "capture() produced no cpu_model on this machine"
    assert p["cpu_count"] == os.cpu_count()
