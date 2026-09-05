"""No test may reach a real network CLI.

On 2026-09-05 `test_a_single_zone_stockout_does_not_advise_retrying_the_list`
ran `scripts/gcp_launch_compile_session.sh` with the real `PATH`, no
`GCP_PROJECT`, `input="launch\n"` to clear the confirmation prompt, and
`GCP_ZONE_FALLBACKS=asia-southeast1-c`. It therefore resolved the live project
from `gcloud config get-value project` and **created a real, billable GPU
instance in GCP on every run** -- four times, ₹218, across a day spent
investigating where the instances were coming from.

It passed every time. Its three assertions all read `SCRIPT.read_text()`, so
nothing the subprocess did was ever checked. A test that cannot fail cannot
report anything, and this one had a side effect on a billing account.

Every OTHER test in that file stubs `gcloud` onto `PATH` itself. That is the
real lesson: **per-test stubbing is opt-in, and opt-in protection fails by
omission.** One test written without the stub is indistinguishable from one
written with it, right up until the invoice.

So the shim is session-scoped and applies to everything. A test that stubs a
tool itself still wins -- it prepends its own directory ahead of this one --
which means reaching a real network CLI is now something a test has to do
deliberately rather than something it can do by forgetting.

Deliberately broad: the same failure is available through `curl`, `ssh`,
`scp`, or any other CLI a shell script might reach for. `git` is NOT shimmed;
the deploy-guard tests build real fixture repos, and `git` is not a network
CLI in that usage (`push`/`fetch` against a tmp_path repo have nowhere to go).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Anything that can reach a network. Add to this list rather than carving out
# exceptions: a tool that needs to be real in one test can be stubbed by that
# test, which is a visible, local decision.
BLOCKED = (
    "gcloud", "gsutil", "bq",           # GCP
    "aws", "az",                        # other clouds
    "ssh", "scp", "sftp", "rsync",      # remote shells and copies
    "curl", "wget",                     # http
    "kubectl", "docker",                # orchestration
    "nvidia-smi",                       # not networked, but not present in CI
)

_SHIM = """#!/bin/bash
# Installed by tests/conftest.py. See that file.
echo "BLOCKED IN TESTS: '{name}' was invoked by a test." >&2
echo "  argv: $*" >&2
echo "  A test reached a real network CLI. If this call is intended, stub it" >&2
echo "  in the test by putting a fake '{name}' first on PATH -- do not remove" >&2
echo "  the guard. A test that shells out to production can bill real money;" >&2
echo "  that is what happened on 2026-09-05 (see tests/conftest.py)." >&2
if [ -n "${{ATTNBENCH_BLOCKED_CALL_LOG:-}}" ]; then
  echo "{name} $*" >> "$ATTNBENCH_BLOCKED_CALL_LOG"
fi
exit 127
"""


@pytest.fixture(scope="session", autouse=True)
def block_network_clis(tmp_path_factory):
    """Prepend a directory of refusing shims to PATH for the whole session."""
    shim_dir = tmp_path_factory.mktemp("blocked-clis")
    for name in BLOCKED:
        p = shim_dir / name
        p.write_text(_SHIM.format(name=name))
        p.chmod(0o755)

    old = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{old}"
    try:
        yield shim_dir
    finally:
        os.environ["PATH"] = old


@pytest.fixture
def blocked_call_log(tmp_path, monkeypatch):
    """Opt-in: record what the shims refused, so a test can assert on it."""
    log = tmp_path / "blocked_calls.txt"
    log.write_text("")
    monkeypatch.setenv("ATTNBENCH_BLOCKED_CALL_LOG", str(log))
    return log
