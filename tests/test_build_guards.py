"""The pre-flight build-memory guard.

The point of these tests is not that the arithmetic is right -- it is that
the guard actually REFUSES. A guard that computes correctly and then returns
0 anyway is worse than no guard, because it manufactures confidence. So the
impossible configurations are checked at three levels: the function raises,
the CLI exits non-zero, and the shell script that calls it aborts before
doing any work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from attnbench.build_guards import (
    CICC_GB_PER_PROCESS, SYSTEM_RESERVE_GB, BuildResourceError,
    check_build_memory, main, plan_build_memory)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# the multiplication that was missed twice
# ---------------------------------------------------------------------------

def test_concurrent_frontends_is_the_product_not_the_job_count():
    """The insight missed on 2026-09-02, twice, once with a second reader
    agreeing: --threads spawns a frontend per gencode target, so jobs and
    threads multiply."""
    plan = plan_build_memory(max_jobs=4, nvcc_threads=2, free_gb=100)
    assert plan.concurrent_frontends == 8
    assert plan.required_gb == 8 * CICC_GB_PER_PROCESS


def test_the_actual_failed_configuration_is_rejected():
    """MAX_JOBS=5 x NVCC_THREADS=2 on the 31 GB g2-standard-8 that OOM'd."""
    with pytest.raises(BuildResourceError):
        check_build_memory(max_jobs=5, nvcc_threads=2, free_gb=31)


def test_the_endorsed_but_still_wrong_configuration_is_also_rejected():
    """MAX_JOBS=4 with NVCC_THREADS left at 2 was proposed as the fix and
    independently agreed to. It is 8 frontends, ~40 GB, and would have OOM'd
    identically. Agreement is not verification -- this test is."""
    with pytest.raises(BuildResourceError):
        check_build_memory(max_jobs=4, nvcc_threads=2, free_gb=31)


def test_the_corrected_configuration_is_accepted():
    """MAX_JOBS=4 x NVCC_THREADS=1 = 4 frontends, ~20 GB, fits in 31 - 3."""
    plan = check_build_memory(max_jobs=4, nvcc_threads=1, free_gb=31)
    assert plan.fits and plan.concurrent_frontends == 4


def test_lowering_only_nvcc_threads_can_be_enough_and_the_error_says_so():
    with pytest.raises(BuildResourceError, match="MULTIPLY"):
        check_build_memory(max_jobs=5, nvcc_threads=2, free_gb=31)


# ---------------------------------------------------------------------------
# it refuses rather than warns
# ---------------------------------------------------------------------------

def test_impossible_configuration_raises_rather_than_returning():
    """A guard that computes and then proceeds is worse than none."""
    with pytest.raises(BuildResourceError):
        check_build_memory(max_jobs=64, nvcc_threads=8, free_gb=31)


def test_cli_exits_nonzero_on_an_impossible_configuration():
    """The shell script branches on this exit code -- if main() printed a
    warning and returned 0, the build would proceed into the OOM."""
    assert main(["--max-jobs", "64", "--nvcc-threads", "8", "--free-gb", "31"]) == 1


def test_cli_exits_zero_on_a_fitting_configuration():
    assert main(["--max-jobs", "4", "--nvcc-threads", "1", "--free-gb", "31"]) == 0


def test_cli_subprocess_exit_code_is_visible_to_a_shell():
    """Belt-and-braces: the shell sees a real non-zero status, not a python
    traceback swallowed into 0."""
    r = subprocess.run(
        [sys.executable, "-m", "attnbench.build_guards",
         "--max-jobs", "64", "--nvcc-threads", "8", "--free-gb", "31"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1
    assert "FATAL" in r.stdout


def test_a_machine_too_small_for_one_job_says_so_instead_of_clamping_to_one():
    """Reporting 'lower MAX_JOBS to 1' on a box where 1 job does not fit
    would send the operator into the same failure at a slower rate."""
    with pytest.raises(BuildResourceError, match="too small for even one"):
        check_build_memory(max_jobs=4, nvcc_threads=2, free_gb=SYSTEM_RESERVE_GB + 1)


# ---------------------------------------------------------------------------
# the shell script actually aborts
# ---------------------------------------------------------------------------

def test_build_script_aborts_before_doing_any_work_on_an_impossible_config():
    """End-to-end: run the real build script with an impossible MAX_JOBS and
    a forced free-memory value, and confirm it exits non-zero WITHOUT having
    started a download or a compile. The guard has to fire before the
    expensive part, not alongside it."""
    import os
    script = REPO / "scripts" / "build_flash_attn.sh"
    assert script.exists()
    env = dict(os.environ)
    env.update(MAX_JOBS="64", NVCC_THREADS="8", FA_FREE_GB_OVERRIDE="31")
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                       cwd=REPO, env=env, timeout=180)
    combined = r.stdout + r.stderr

    assert r.returncode != 0, "build script must refuse an impossible configuration"
    # Must fail for the RIGHT reason. Asserting only on the exit code would
    # pass if the script died on a missing ninja or toolchain instead, which
    # would leave the memory guard itself completely unexercised.
    assert "concurrent cicc" in combined, (
        f"aborted, but not via the memory guard -- output was: {combined[:500]}")
    assert "fetching" not in combined.lower(), (
        "guard fired too late -- the script had already started downloading")
