"""The suite record's GPU-host selection (adviser's ruling 1h dated 10 Nov 2026:
P2's and P4's cards pass --suite-record). The record failed 27 tests on the L4
that assume a CPU-only host (`docs/gpu_suite_failures_20261110.md`);
`scripts/gpu_suite_record.sh` deselects them from `tests/gpu_host_deselect.txt`.
A list entry that no longer names a test would deselect nothing and let the
test fail on the GPU host, so each entry is checked against the collection."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "tests" / "gpu_host_deselect.txt"
SCRIPT = ROOT / "scripts" / "gpu_suite_record.sh"


def entries() -> list[str]:
    return [l.strip() for l in LIST.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def test_the_list_holds_the_33_classified_failures_and_not_the_timed_region_file():
    ids = entries()
    assert len(ids) == len(set(ids)) == 33
    assert not any(i.startswith("tests/test_timed_region_setup.py") for i in ids)


def test_every_entry_names_a_collected_test():
    files = sorted({i.split("::")[0] for i in entries()})
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                        "-p", "no:cacheprovider", *files],
                       cwd=ROOT, capture_output=True, text=True)
    collected = {l.strip() for l in r.stdout.splitlines() if "::" in l}
    missing = [i for i in entries() if i not in collected]
    assert not missing, f"not collected (renamed or removed?): {missing}"


def test_the_script_passes_every_entry_to_pytest(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    shutil.copy(LIST, tmp_path / "tests" / LIST.name)
    log = tmp_path / "argv.txt"
    # run_phase.sh stand-in: write the command it was given, one argument a line
    (tmp_path / "scripts" / "run_phase.sh").write_text(
        f'shift 2\nfor a in "$@"; do echo "$a"; done >> "{log}"\necho --- >> "{log}"\n')
    r = subprocess.run(["/bin/bash", "scripts/gpu_suite_record.sh"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    suite = log.read_text().split("---\n")[0].splitlines()
    assert suite[:4] == ["python3", "-m", "pytest", "-q"]
    pairs = suite[4:]
    assert pairs[0::2] == ["--deselect"] * (len(pairs) // 2)
    assert pairs[1::2] == ["tests/test_timed_region_setup.py", *entries()]
