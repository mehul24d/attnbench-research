"""The instance startup script, rendered from the launch script and executed.

Two things run on every boot and must both be right: the hard-cap shutdown,
and the derived-cache clear. The cache clear exists because a machine image
preserves whatever the captured instance had on disk -- including caches whose
purpose is to make expensive work free.

On 2026-09-03 the v3 image carried `results/accuracy/score_cache` from the
session that captured it, and the next session's *timed* scoring pass loaded a
file instead of computing: `0.007 s, 9041 TFLOPS` for a phase that really
takes 12 s. It was caught only because 9041 TFLOPS is absurd on its face; at a
plausible magnitude the same failure is silent, and the Stage 3 estimate built
on it would simply have been hours low.

These tests render the startup script the way the launch script does, then
actually run its cache-clearing loop against a fake home layout -- because a
boot step nobody has watched work is a guess.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gcp_launch_compile_session.sh"


def _render(cap_minutes: int = 180) -> str:
    """Expand the launch script's startup-script heredoc exactly as the
    launch script would, without creating anything."""
    body = subprocess.run(
        ["awk", '/^cat > "\\$STARTUP_SCRIPT" <<EOF$/{f=1;next} /^EOF$/{f=0} f',
         str(SCRIPT)],
        capture_output=True, text=True, check=True).stdout
    rendered = subprocess.run(
        ["bash", "-c", f'CAP_MINUTES={cap_minutes}; cat <<EOF\n{body}\nEOF'],
        capture_output=True, text=True, check=True)
    return rendered.stdout


def test_hard_cap_is_expanded_at_launch_time():
    """CAP_MINUTES must be baked in, since the instance has no idea what
    deadline was agreed."""
    assert "shutdown -h +180" in _render(180)
    assert "shutdown -h +240" in _render(240)


def test_cache_path_variable_is_NOT_expanded_at_launch_time():
    """$CACHE is the loop's own variable and must reach the instance intact.

    The heredoc is unquoted so CAP_MINUTES expands; that same expansion would
    silently blank $CACHE without the backslash escape, turning the guard into
    `rm -rf ""` -- a loop that runs, reports nothing, and clears nothing.
    """
    rendered = _render()
    assert 'rm -rf "$CACHE"' in rendered
    assert 'rm -rf ""' not in rendered


def test_the_cache_clearing_loop_actually_removes_a_stale_cache(tmp_path):
    """Execute the real loop against a fake /home layout."""
    home = tmp_path / "home"
    proj = home / "someuser" / "attnbench_scaffold" / "results"
    cache = proj / "accuracy" / "score_cache"
    cache.mkdir(parents=True)
    (cache / "677315e93956df57.pt").write_bytes(b"stale scores")

    # a measurement output that must SURVIVE
    (proj / "batch_scaling").mkdir(parents=True)
    survivor = proj / "batch_scaling" / "batch_scaling.parquet"
    survivor.write_bytes(b"measurements")

    loop = _extract_cache_loop(_render()).replace("/home/*", f"{home}/*")
    subprocess.run(["bash", "-c", loop], check=True,
                   capture_output=True, text=True)

    assert not cache.exists(), "stale score cache must be removed on boot"
    assert survivor.exists(), (
        "measurement outputs must NEVER be deleted -- an image that quietly "
        "discarded a session's product would be worse than the bug this fixes")
    assert survivor.read_bytes() == b"measurements"


def test_the_loop_is_harmless_when_no_cache_exists(tmp_path):
    """Booting a fresh image with no cache must not error, or every launch
    fails on a machine that was never contaminated."""
    home = tmp_path / "home"
    (home / "someuser").mkdir(parents=True)
    loop = _extract_cache_loop(_render()).replace("/home/*", f"{home}/*")
    proc = subprocess.run(["bash", "-c", loop], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def _extract_cache_loop(rendered: str) -> str:
    """The `for CACHE in ...; do ... done` block, with `logger` stubbed --
    logger writes to syslog and is not present in every test environment."""
    m = re.search(r"^for CACHE in .*?^done$", rendered, re.M | re.S)
    assert m, "cache-clearing loop not found in the rendered startup script"
    return "logger() { :; }\n" + m.group(0)


def test_extraction_finds_a_real_loop():
    """Guards against the regex silently matching nothing, which would make
    every test above pass by executing an empty string."""
    loop = _extract_cache_loop(_render())
    assert "score_cache" in loop and "rm -rf" in loop
