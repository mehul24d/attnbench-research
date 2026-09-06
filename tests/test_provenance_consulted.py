"""A provenance field that no gate reads fails exactly as a missing one does.

On 2026-09-04 a 4576-row result set -- a complete Stage 0 and the project's
first complete Stage 1 -- was stamped with a commit 18 behind the code that
produced it. Source had been untarred over the instance's checkout, so `.git`
still described the machine image and `provenance.capture()` faithfully
reported the image's commit.

The part worth keeping: **the mechanism worked.** `git_dirty=True` was
recorded correctly on every one of those rows. The stamp announced its own
unreliability and nothing was listening -- `load_stage1_pass_set` read
`git_commit` and no other field, so the table would have cleared the Stage 2
commit gate while describing different code.

That is the tenth instance in docs/silent_failure_patterns.md, and the
generalisation is what these tests enforce:

  1. every Provenance field is classified as GATED (some gate reads it) or
     RECORDED (kept for the record, consulted by nobody, deliberately) -- so
     a new field forces the decision instead of defaulting to decorative; and
  2. every GATED field is actually read somewhere outside provenance.py,
     asserted against the source. A set-membership declaration that drifts
     from reality would be the same failure one level up.

All CPU-only.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pandas as pd
import pytest

from attnbench import provenance
from attnbench.provenance import (GATED_FIELDS, RECORDED_FIELDS, Provenance,
                                  stamp_integrity_problems)

REPO = Path(__file__).resolve().parents[1]

# Modules that gate or join on provenance. A GATED field must appear in at
# least one of these; provenance.py itself does not count, since writing a
# field is precisely what it does regardless.
GATE_MODULES = (
    REPO / "attnbench" / "sweep.py",
    REPO / "attnbench" / "analysis" / "cross_arch.py",
    REPO / "attnbench" / "analysis" / "canary.py",
)


def _all_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(Provenance)}


def test_every_field_is_classified():
    """The check that makes the next field a decision rather than a default."""
    unclassified = _all_fields() - GATED_FIELDS - RECORDED_FIELDS
    assert not unclassified, (
        f"provenance fields {sorted(unclassified)} are in neither "
        f"GATED_FIELDS nor RECORDED_FIELDS. Decide which: a field no gate "
        f"reads is decorative, and decorative fields have already cost this "
        f"project a 4576-row table")


def test_no_field_is_classified_twice():
    assert not (GATED_FIELDS & RECORDED_FIELDS)


def test_the_classification_describes_real_fields():
    """A typo in either set would silently exempt the real field from the
    coverage check above."""
    stale = (GATED_FIELDS | RECORDED_FIELDS) - _all_fields()
    assert not stale, f"classified fields that do not exist: {sorted(stale)}"


@pytest.mark.parametrize("field", sorted(GATED_FIELDS))
def test_a_gated_field_is_actually_read_by_a_gate(field):
    """Membership in GATED_FIELDS is a claim about other code. Asserted, not
    trusted -- an out-of-date declaration is the same failure as an unread
    field, one level of indirection away."""
    hits = [m.name for m in GATE_MODULES if field in m.read_text()]
    assert hits, (
        f"{field!r} is declared GATED but appears in none of "
        f"{[m.name for m in GATE_MODULES]}. Either wire it into a gate or "
        f"move it to RECORDED_FIELDS and say why")


def test_git_dirty_is_gated():
    """The specific regression. Named on its own so that moving it back to
    RECORDED is a deliberate act with a failing test attached."""
    assert "git_dirty" in GATED_FIELDS


# ---------------------------------------------------------------------------
# stamp_integrity_problems
# ---------------------------------------------------------------------------

CLEAN = {"git_commit": "a" * 40, "git_dirty": False}


def test_a_clean_stamp_has_no_problems():
    assert stamp_integrity_problems(CLEAN) == []


def test_a_dirty_stamp_is_a_problem():
    problems = stamp_integrity_problems({**CLEAN, "git_dirty": True})
    assert len(problems) == 1
    assert "git_dirty" in problems[0]


def test_a_missing_commit_is_a_problem():
    assert stamp_integrity_problems({"git_commit": None, "git_dirty": False})


def test_the_literal_string_HEAD_is_rejected():
    """Instance 3's shape: `git rev-parse HEAD` echoes its argument to stdout
    when HEAD does not resolve, so a naive capture stores "HEAD" as a commit --
    populated, identical on every row, and verifying nothing."""
    assert stamp_integrity_problems({"git_commit": "HEAD", "git_dirty": False})


def test_a_dirty_stamp_with_a_valid_commit_is_still_a_problem():
    """The exact 2026-09-04 shape, and the reason a commit check alone could
    not catch it: the SHA is real, well-formed, and in this repo's history."""
    problems = stamp_integrity_problems(
        {"git_commit": "b6ed63bc6e61c11efb272867ba11e7849be8cb23",
         "git_dirty": True})
    assert problems, "a well-formed commit made the dirty tree invisible"


def test_it_reads_a_pandas_row():
    """Where the check will actually run. `v is True` fails on the
    numpy.bool_ a pandas column hands back, so a version of this that passed
    every dict test would have missed every parquet row it exists to catch --
    which is the whole failure mode, one layer down."""
    row = pd.DataFrame([{**CLEAN, "git_dirty": True}]).iloc[0]
    assert stamp_integrity_problems(row)


def test_a_numpy_bool_is_recognised():
    import numpy as np
    assert stamp_integrity_problems({**CLEAN, "git_dirty": np.bool_(True)})
    assert stamp_integrity_problems({**CLEAN, "git_dirty": np.bool_(False)}) == []


def test_a_missing_dirty_flag_does_not_read_as_dirty():
    """The opposite error, and the more expensive one. `bool(nan)` is True, so
    a naive truth test would report every row that predates the field as
    dirty and refuse a whole prior dataset. A check that fires on absence is
    as useless as one that never fires."""
    assert stamp_integrity_problems({**CLEAN, "git_dirty": float("nan")}) == []
    assert stamp_integrity_problems({"git_commit": "a" * 40}) == []


def test_capture_records_both_gated_git_fields():
    p = provenance.capture()
    d = p.to_dict()
    assert "git_commit" in d and "git_dirty" in d


# ---------------------------------------------------------------------------
# lock_clocks must report what happened, not what was attempted
# ---------------------------------------------------------------------------

def _fake_nvidia_smi(tmp_path, *, lgc_exit: int, lgc_stdout: str = ""):
    """A stand-in nvidia-smi on PATH. The real one prints its PERMISSION
    ERROR to stdout and exits 4, which is the whole trap."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    smi = bin_dir / "nvidia-smi"
    smi.write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do case "$a" in\n'
        '  -lgc) printf "%s" "' + lgc_stdout + f'"; exit {lgc_exit} ;;\n'
        '  --query-gpu=clocks.max.sm) echo 2040; exit 0 ;;\n'
        "esac; done\n"
        "exit 0\n")
    smi.chmod(0o755)
    return bin_dir


def test_lock_clocks_reports_failure_when_the_lock_is_refused(tmp_path, monkeypatch):
    """The measured failure, 2026-09-06 on an L4: returned True, clock
    unchanged at 2040 MHz. nvidia-smi -lgc exits 4 and prints its permission
    error to STDOUT, and `_sh` returns stdout-or-None without ever looking at
    the exit code -- so non-empty stdout read as success.

    Worse than the other two stdout-as-outcome bugs because of its direction:
    it reports a control as ESTABLISHED when it is absent, so every row it
    stamps overstates how well the run was controlled.
    """
    from attnbench import provenance
    bin_dir = _fake_nvidia_smi(
        tmp_path, lgc_exit=4,
        lgc_stdout="The current user does not have permission to change clocks")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert provenance.lock_clocks() is False


def test_lock_clocks_reports_success_when_the_lock_is_granted(tmp_path, monkeypatch):
    """The other half -- the fix must not have turned it into a constant
    False, which would be the same defect pointing the other way."""
    from attnbench import provenance
    bin_dir = _fake_nvidia_smi(tmp_path, lgc_exit=0,
                               lgc_stdout="GPU clocks set to ...\nAll done.")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert provenance.lock_clocks() is True


def test_lock_clocks_reports_success_on_silent_success(tmp_path, monkeypatch):
    """Exit 0 with EMPTY stdout is still success. The old implementation got
    this wrong too, in the opposite direction -- it is the same conflation."""
    from attnbench import provenance
    bin_dir = _fake_nvidia_smi(tmp_path, lgc_exit=0, lgc_stdout="")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert provenance.lock_clocks() is True


def test_lock_clocks_escalates_to_passwordless_sudo(tmp_path, monkeypatch):
    """Clock control needs root; the measurement must not run as root, or
    results/ ends up root-owned and the next segment cannot append to it.

    The escalation lives inside lock_clocks so the caller neither runs the
    whole job as root nor ASSERTS the lock state on a command line -- an
    asserted control being exactly what this function was fixed for.
    """
    from attnbench import provenance
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    (bin_dir / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "smi $*" >> {calls}\n'
        'for a in "$@"; do case "$a" in\n'
        '  -lgc) echo "permission denied"; exit 4 ;;\n'
        '  --query-gpu=clocks.max.sm) echo 2040; exit 0 ;;\n'
        "esac; done\nexit 0\n")
    (bin_dir / "sudo").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "sudo $*" >> {calls}\n'
        '[ "$1" = "-n" ] && shift\n'
        'for a in "$@"; do case "$a" in -lgc) echo "All done."; exit 0 ;; esac; done\n'
        "exit 0\n")
    for f in ("nvidia-smi", "sudo"):
        (bin_dir / f).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    assert provenance.lock_clocks() is True
    log = calls.read_text()
    assert "smi -lgc" in log, "must try unprivileged first"
    assert "sudo -n nvidia-smi -lgc" in log, "must escalate when refused"


def test_lock_clocks_stays_false_when_sudo_is_refused_too(tmp_path, monkeypatch):
    """sudo -n never prompts. No rights, or a password required, must come
    back as False rather than hanging or claiming success."""
    from attnbench import provenance
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do case "$a" in\n'
        '  -lgc) echo "permission denied"; exit 4 ;;\n'
        '  --query-gpu=clocks.max.sm) echo 2040; exit 0 ;;\n'
        "esac; done\nexit 0\n")
    (bin_dir / "sudo").write_text(
        "#!/usr/bin/env bash\necho 'sudo: a password is required' >&2\nexit 1\n")
    for f in ("nvidia-smi", "sudo"):
        (bin_dir / f).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    assert provenance.lock_clocks() is False
