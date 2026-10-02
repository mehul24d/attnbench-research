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


# --- the sparse pilot -----------------------------------------------------

def test_sparse_pilot_tasks_bands_and_n():
    assert ra.select_tasks(grid_tasks=GRID_TASKS, tasks=None, t4_dense_pilot=False,
                           t4_pilot_tasks=None, t4_sparse_pilot=True) == (
        "qa_1", "niah_multivalue", "niah_multiquery")
    assert ra.select_seq_lens(grid_seq_lens=GRID_LENS, seq_lens=None, n_per_length=None,
                              t4_dense_pilot=False, t4_pilot_tasks=None,
                              t4_sparse_pilot=True) == {16384: 100, 32768: 100}
    assert ra.sparse_pilot_n("qa_1", None) == 100
    assert ra.sparse_pilot_n("niah_multiquery", None) == 50
    assert ra.sparse_pilot_n("qa_1", 2) == ra.sparse_pilot_n("niah_multivalue", 2) == 2
    assert ra.sparse_pilot_n("niah_multivalue", 80) == 50       # never more
    with pytest.raises(SystemExit):
        ra.select_seq_lens(grid_seq_lens=GRID_LENS, seq_lens="8192", n_per_length=None,
                           t4_dense_pilot=False, t4_pilot_tasks=None, t4_sparse_pilot=True)
    with pytest.raises(SystemExit):
        ra.select_seq_lens(grid_seq_lens=GRID_LENS, seq_lens=None, n_per_length=101,
                           t4_dense_pilot=False, t4_pilot_tasks=None, t4_sparse_pilot=True)


def _dry_run(monkeypatch, capsys, tmp_path, *argv):
    """main() on the real grid with only the RULER example builder stubbed
    (its data is not committed), so the printed plan is the real one."""
    import sys
    from attnbench.accuracy.ruler import RulerExample

    def fake_build(grid, seed, *, count_tokens, tasks, seq_lens):
        return {(t, b): [RulerExample(task=t, example_id=f"{t}_{b}_{i}", context="c",
                                      question="", answer=["a"], context_length=b,
                                      sizing="approximate")
                         for i in range(n)]
                for t in tasks for b, n in seq_lens.items()}

    monkeypatch.setattr(ra, "build_examples_by_task_length", fake_build)
    monkeypatch.setattr(sys, "argv", ["run_accuracy.py", "--dry-run",
                                      "--out", str(tmp_path), *argv])
    ra.main()
    out = capsys.readouterr().out
    total = int(next(l for l in out.splitlines() if l.startswith("total cells")).split(":")[1])
    backends = next(l for l in out.splitlines() if l.startswith("backends")).split(":")[1]
    return total, [b.strip() for b in backends.split(",")]


def test_sparse_pilot_oracle_run_plans_dense_plus_three_sparsities(monkeypatch, capsys, tmp_path):
    # examples per band: qa_1 100 + 50 + 50 = 200, two bands = 400
    total, backends = _dry_run(monkeypatch, capsys, tmp_path, "--t4-sparse-pilot")
    assert backends == ["sdpa_flash", "block_sparse"]
    assert total == 400 * (1 + 3)


def test_sparse_pilot_inline_run_and_canary(monkeypatch, capsys, tmp_path):
    total, backends = _dry_run(monkeypatch, capsys, tmp_path, "--t4-sparse-pilot",
                               "--score-source", "minference_meanpool_inline",
                               "--only-backends", "block_sparse")
    assert (total, backends) == (400 * 3, ["block_sparse"])
    total, _ = _dry_run(monkeypatch, capsys, tmp_path / "c", "--t4-sparse-pilot",
                        "--n-per-length", "2")
    assert total == 3 * 2 * 2 * 4                 # tasks x bands x n x arms


def test_sparse_pilot_refuses_what_the_plan_does_not_contain(monkeypatch, capsys, tmp_path):
    for extra in (["--score-source", "minference_meanpool"], ["--tasks", "vt"],
                  ["--t4-dense-pilot"], ["--mask-source", "importance_randfree"]):
        with pytest.raises(SystemExit):
            _dry_run(monkeypatch, capsys, tmp_path, "--t4-sparse-pilot", *extra)


# --- the XAttention phase ------------------------------------------------

def test_xattn_pilot_plans_dense_and_one_threshold(monkeypatch, capsys, tmp_path):
    total, backends = _dry_run(monkeypatch, capsys, tmp_path, "--t4-xattn-pilot",
                               "--xattn-threshold", "0.95")
    assert backends == ["sdpa_flash", "xattention"]
    assert total == 400 * 2
    total, backends = _dry_run(monkeypatch, capsys, tmp_path / "d", "--t4-xattn-pilot",
                               "--only-backends", "sdpa_flash")
    assert (total, backends) == (400, ["sdpa_flash"])
    total, backends = _dry_run(monkeypatch, capsys, tmp_path / "x", "--t4-xattn-pilot",
                               "--xattn-threshold", "0.8", "--only-backends", "xattention",
                               "--n-per-length", "2")
    assert (total, backends) == (3 * 2 * 2, ["xattention"])


def test_xattn_pilot_refuses_what_the_plan_does_not_contain(monkeypatch, capsys, tmp_path):
    for argv in (["--t4-xattn-pilot"],                               # no threshold
                 ["--t4-xattn-pilot", "--xattn-threshold", "0.85"],  # off the grid
                 ["--t4-xattn-pilot", "--xattn-threshold", "0.9", "--t4-sparse-pilot"],
                 ["--t4-xattn-pilot", "--xattn-threshold", "0.9", "--tasks", "vt"],
                 ["--t4-xattn-pilot", "--xattn-threshold", "0.9", "--sparsities", "0.5"],
                 ["--t4-xattn-pilot", "--xattn-threshold", "0.9",
                  "--score-source", "minference_meanpool_inline"],
                 ["--xattn-threshold", "0.9"]):                       # without the pilot
        with pytest.raises(SystemExit):
            _dry_run(monkeypatch, capsys, tmp_path, *argv)
