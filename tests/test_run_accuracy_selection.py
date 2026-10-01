"""What `scripts/run_accuracy.py` plans: tasks, bands and n per band.

Commit 40118f7 added --t4-pilot-tasks as a second `if` whose `else` reset
the task list to the whole grid, so --tasks and a bare --t4-dense-pilot were
silently overwritten (`--tasks vt` planned 19500 cells instead of 6500), and
set the pilot's n twice, the second write (5) always winning. Neither banked
T4 run was affected: the probe ran at 822a4dd, before the change, and the
selected pilot passed --t4-pilot-tasks and --n-per-length 50 explicitly. The
last two tests pin those two commands to what they ran.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_run_accuracy_sel", Path(__file__).resolve().parents[1] / "scripts" / "run_accuracy.py")
ra = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ra)

GRID_TASKS = ("niah_single", "niah_multikey", "vt")
GRID_LENS = {2048: 300, 4096: 300, 8192: 300, 16384: 300, 32768: 100}
CANDIDATES = ("niah_multikey_1", "niah_multivalue", "niah_multiquery", "qa_1", "qa_2")
SELECTED = "niah_multivalue,niah_multiquery,qa_1"


def tasks(tasks=None, pilot=False, subset=None):
    return ra.select_tasks(grid_tasks=GRID_TASKS, tasks=tasks,
                           t4_dense_pilot=pilot, t4_pilot_tasks=subset)


def lens(seq_lens=None, n=None, pilot=False, subset=None):
    return ra.select_seq_lens(grid_seq_lens=GRID_LENS, seq_lens=seq_lens,
                              n_per_length=n, t4_dense_pilot=pilot,
                              t4_pilot_tasks=subset)


def test_tasks_flag_is_not_overwritten():
    assert tasks("vt") == ("vt",)
    assert tasks("vt, niah_single") == ("vt", "niah_single")


def test_no_flag_is_the_whole_grid():
    assert tasks() == GRID_TASKS
    assert lens() == GRID_LENS


def test_bare_pilot_plans_the_five_candidates():
    assert tasks(pilot=True) == CANDIDATES == ra.T4_DENSE_PILOT_TASKS


def test_pilot_subset_must_come_from_the_candidates():
    assert tasks(pilot=True, subset=SELECTED) == ("niah_multivalue", "niah_multiquery", "qa_1")
    with pytest.raises(SystemExit):
        tasks(pilot=True, subset="niah_single")
    with pytest.raises(SystemExit):
        tasks(pilot=True, subset=" , ")


def test_unknown_grid_task_refuses():
    with pytest.raises(SystemExit):
        tasks("niah_multivalue")


def test_pilot_defaults_are_the_preregistered_sizes():
    assert lens(pilot=True) == {16384: 5, 32768: 5}
    assert lens(pilot=True, subset=SELECTED) == {16384: 50, 32768: 50}
    assert lens("16384", pilot=True) == {16384: 5}


def test_n_per_length_is_bounded_by_the_grid_and_the_probe():
    assert lens("8192", n=10) == {8192: 10}
    with pytest.raises(SystemExit):
        lens("32768", n=101)                       # grid n there is 100
    with pytest.raises(SystemExit):
        lens(n=0)
    with pytest.raises(SystemExit):
        lens(pilot=True, n=50)                     # 50 is for the selected subset
    with pytest.raises(SystemExit):
        lens("1024")                               # a band the grid did not plan


def test_the_banked_probe_command_plans_what_it_ran():
    """822a4dd probe: `--t4-dense-pilot`, 5 tasks x 2 bands x 5 = 50 rows."""
    t, l = tasks(pilot=True), lens(pilot=True)
    assert len(t) * sum(l.values()) == 50


def test_the_banked_selected_pilot_command_plans_what_it_ran():
    """40118f7 pilot: the doc's command, 3 tasks x 2 bands x 50 = 300 rows."""
    t = tasks(pilot=True, subset=SELECTED)
    l = lens(n=50, pilot=True, subset=SELECTED)
    assert len(t) * sum(l.values()) == 300
    assert l == lens(pilot=True, subset=SELECTED)     # the default agrees
