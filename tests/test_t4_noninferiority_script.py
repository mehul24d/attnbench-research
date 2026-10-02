"""scripts/run_t4_noninferiority.py on synthetic pilot parquets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench.accuracy import t4_pilot

_SPEC = importlib.util.spec_from_file_location(
    "_t4_ni", Path(__file__).resolve().parents[1] / "scripts" / "run_t4_noninferiority.py")
ni_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ni_script)

ORACLE, INLINE = t4_pilot.SPARSE_PILOT_SCORE_SOURCES


def _rows(losses: dict, *, dirty=False, commit="c" * 40):
    """Dense always right; each (score_source, sparsity) arm wrong on its
    first `losses[...]` examples of every (task, band)."""
    rows = []
    for task in t4_pilot.SPARSE_PILOT_TASKS:
        for band in t4_pilot.PILOT_BANDS:
            n = t4_pilot.SPARSE_PILOT_N[task]
            for i in range(n):
                base = dict(task=task, example_id=f"{task}_{band}_{i}",
                            context_length=band - 5, git_commit=commit, git_dirty=dirty)
                rows.append({**base, "backend": "sdpa_flash",
                             "backend_role": "dense_reference", "score_source": None,
                             "sparsity": None, "correct": True})
                for (src, s), k in losses.items():
                    rows.append({**base, "backend": "block_sparse",
                                 "backend_role": "block_sparse", "score_source": src,
                                 "sparsity": s, "correct": i >= k})
    return pd.DataFrame(rows)


def _run(tmp_path, df, *extra):
    p = tmp_path / "in.parquet"
    df.to_parquet(p)
    out = tmp_path / "out"
    old = sys.argv
    sys.argv = ["x", str(p), "--out", str(out), *extra]
    try:
        ni_script.main()
    finally:
        sys.argv = old
    return pd.read_parquet(out / "t4_noninferiority.parquet")


def test_fixed_sequence_stops_at_the_first_failure(tmp_path):
    # inline: 0 losses at 0.5, 30 at 0.75 (fails), 0 at 0.9 (passes, but unreached)
    losses = {(ORACLE, 0.5): 0, (ORACLE, 0.75): 0, (ORACLE, 0.9): 0,
              (INLINE, 0.5): 0, (INLINE, 0.75): 30, (INLINE, 0.9): 0}
    res = _run(tmp_path, _rows(losses)).set_index(["task", "band", "score_source", "sparsity"])
    q = res.loc[("qa_1", 16384, INLINE)]
    assert list(q.tested) == [True, True, False]
    assert list(q.non_inferior) == [True, False, True]
    assert list(q.claim) == [True, False, False]          # 0.9 passes and is still no claim
    assert res.loc[("qa_1", 16384, ORACLE)].claim.all()
    assert res.loc[("niah_multiquery", 32768, ORACLE)].tier.unique().tolist() == ["observational"]
    assert set(res.tier) == {"primary", "secondary", "observational"}


def test_n50_tasks_cannot_certify_where_qa1_can(tmp_path):
    """The reason qa_1 runs at n=100: one loss each, and only qa_1 clears 10."""
    losses = {(ORACLE, 0.5): 1, (ORACLE, 0.75): 1, (ORACLE, 0.9): 1}
    res = _run(tmp_path, _rows(losses)).set_index(["task", "band", "score_source", "sparsity"])
    assert res.loc[("qa_1", 16384, ORACLE, 0.5)].non_inferior
    assert not res.loc[("niah_multivalue", 16384, ORACLE, 0.5)].non_inferior


@pytest.mark.parametrize("bad", ["dirty", "commits", "task", "arm"])
def test_refusals(tmp_path, bad):
    df = _rows({(ORACLE, 0.5): 0})
    if bad == "dirty":
        df.loc[0, "git_dirty"] = True
    elif bad == "commits":
        df.loc[0, "git_commit"] = "d" * 40
    elif bad == "task":
        df.loc[df.index[-1], "task"] = "vt"
    else:
        df.loc[df.backend == "block_sparse", "score_source"] = "minference_meanpool"
    with pytest.raises(SystemExit):
        _run(tmp_path, df)


def test_printed_table_keeps_the_grid_sparsities(tmp_path, capsys):
    """A blanket round(1) printed 0.75 as 0.8 on the 2026-10-02 pilot table."""
    losses = {(ORACLE, s): 0 for s in t4_pilot.SPARSITY_SEQUENCE}
    _run(tmp_path, _rows(losses))
    out = capsys.readouterr().out
    assert " 0.75 " in out and " 0.8 " not in out


XATTN = t4_pilot.XATTN_SCORE_SOURCE


def _xattn_rows(losses: dict):
    """Dense always right; XAttention at threshold tau wrong on its first
    `losses[tau]` examples of every (task, band), with density 1 - tau/2."""
    rows = []
    for task in t4_pilot.SPARSE_PILOT_TASKS:
        for band in t4_pilot.PILOT_BANDS:
            for i in range(t4_pilot.SPARSE_PILOT_N[task]):
                base = dict(task=task, example_id=f"{task}_{band}_{i}",
                            context_length=band - 5, git_commit="c" * 40, git_dirty=False)
                rows.append({**base, "backend": "sdpa_flash", "backend_role": "dense_reference",
                             "score_source": None, "sparsity": None, "xattn_threshold": None,
                             "realised_density": None, "correct": True})
                for tau, k in losses.items():
                    rows.append({**base, "backend": "xattention", "backend_role": "block_sparse",
                                 "score_source": XATTN, "sparsity": None,
                                 "xattn_threshold": tau, "realised_density": 1 - tau / 2,
                                 "correct": i >= k})
    return pd.DataFrame(rows)


def test_xattention_thresholds_run_least_aggressive_first(tmp_path):
    # 0.95 passes, 0.9 fails, 0.8 would pass but is never reached
    res = _run(tmp_path, _xattn_rows({0.95: 0, 0.9: 30, 0.8: 0}))
    q = res[(res.task == "qa_1") & (res.band == 16384)].sort_values("seq_pos")
    assert list(q.xattn_threshold) == [0.95, 0.9, 0.8]
    assert q.sparsity.isna().all()
    assert list(q.tested) == [True, True, False]
    assert list(q.claim) == [True, False, False]
    assert list(q.mean_realised_density.round(3)) == [0.525, 0.55, 0.6]


def test_an_xattention_row_without_a_threshold_is_refused(tmp_path):
    df = _xattn_rows({0.95: 0})
    df.loc[df.backend == "xattention", "xattn_threshold"] = None
    with pytest.raises(ValueError, match="no xattn_threshold"):
        _run(tmp_path, df)
