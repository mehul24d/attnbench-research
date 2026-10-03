"""The `split` column on recall rows, the selection script's refusal, and the
firewall in front of the evaluation analysis (pre-registration sec. 5, 7.2)."""
from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from attnbench.analysis import frontier_recall as fr
from attnbench.analysis.frontier_prereg import SplitLeak


def _row(index=1000, split="selection", d_nom=0.25, recall_norm=0.97, arm="MP", band=16384, **kw):
    base = dict(model="Qwen/Qwen2.5-1.5B-Instruct", task="qa_1", band=band,
                example_id=f"qa_1_{band}_{index}", split=split, arm=arm,
                mask_selector="per_head", block_size=128, d_nom=d_nom, recall_raw=0.8,
                recall_norm=recall_norm, realised_density=0.27, git_commit="abc", git_dirty=False)
    base.update(kw)
    return fr.RecallRow(**base)


def _selection(recalls):
    return fr.to_frame([_row(index=1000 + i, d_nom=d, recall_norm=r)
                        for d, r in recalls.items() for i in range(4)])


@pytest.mark.parametrize("index,split", [(0, "t4_replication"), (99, "t4_replication"),
                                         (1000, "selection"), (1031, "selection"),
                                         (2000, "calibration"), (3000, "evaluation"),
                                         (3299, "evaluation")])
def test_split_of(index, split):
    assert fr.split_of(f"qa_1_16384_{index}") == split


@pytest.mark.parametrize("index", [100, 999, 1032, 2048, 3300])
def test_an_index_in_no_split_is_refused(index):
    with pytest.raises(SplitLeak):
        fr.split_of(f"qa_1_16384_{index}")


def test_a_row_cannot_be_relabelled_into_another_split():
    with pytest.raises(SplitLeak):
        _row(index=3000, split="selection")
    with pytest.raises(ValueError):
        _row(split="test")
    with pytest.raises(ValueError):
        _row(mask_selector="per-head")
    with pytest.raises(ValueError):
        _row(git_dirty=True)
    with pytest.raises(TypeError):
        fr.RecallRow(model="m", task="qa_1", band=1, example_id="qa_1_1_1000", arm="MP",
                     mask_selector="per_head", block_size=128, d_nom=0.25, recall_raw=1,
                     recall_norm=1, realised_density=1, git_commit="a", git_dirty=False)


def test_every_recall_row_carries_split_and_mask_selector():
    assert {"split", "mask_selector"} <= set(fr.RECALL_COLUMNS)
    assert list(fr.to_frame([_row()]).columns) == list(fr.RECALL_COLUMNS)


def test_budget_selection_picks_the_smallest_qualifying_budget():
    sel = fr.select_budgets(_selection({0.50: 0.99, 0.25: 0.96, 0.10: 0.951, 0.05: 0.94}))
    assert sel["MP@16384"]["b_star"] == 0.10 and sel["MP@16384"]["qualified"]
    none = fr.select_budgets(_selection({0.50: 0.9, 0.25: 0.8, 0.10: 0.7, 0.05: 0.6}))
    assert none["MP@16384"]["b_star"] == 0.50 and not none["MP@16384"]["qualified"]
    edge = fr.select_budgets(_selection({0.50: 0.99, 0.25: 0.95, 0.10: 0.9499, 0.05: 0.9}))
    assert edge["MP@16384"]["b_star"] == 0.25


def test_budget_selection_refuses_any_row_that_is_not_selection():
    frame = pd.concat([_selection({d: 0.99 for d in fr.D_NOMS}),
                       fr.to_frame([_row(index=3000, split="evaluation")])])
    with pytest.raises(SplitLeak, match="selection split only"):
        fr.select_budgets(frame)


def test_budget_selection_refuses_rows_without_a_split_or_with_a_wrong_one():
    frame = _selection({d: 0.99 for d in fr.D_NOMS})
    with pytest.raises(SplitLeak, match="lack"):
        fr.select_budgets(frame.drop(columns=["split"]))
    blank = frame.copy()
    blank.loc[blank.index[0], "split"] = None
    with pytest.raises(SplitLeak):
        fr.select_budgets(blank)
    relabelled = frame.copy()
    relabelled["example_id"] = relabelled["example_id"].str.replace("_10", "_30", regex=False)
    with pytest.raises(SplitLeak):
        fr.select_budgets(relabelled)        # evaluation examples wearing the selection label


def test_budget_selection_refuses_a_missing_budget_level():
    with pytest.raises(SplitLeak, match="no rows at d_nom"):
        fr.select_budgets(_selection({0.50: 0.99, 0.25: 0.99}))


def _repo(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "README").write_text("x\n")
    git("add", "README")
    git("commit", "-q", "-m", "init")
    return git


def test_the_evaluation_analysis_is_refused_until_the_selection_is_committed(tmp_path):
    git = _repo(tmp_path)
    path = tmp_path / "budget_selection.json"
    frame = pd.concat([_selection({d: 0.99 for d in fr.D_NOMS}),
                       fr.to_frame([_row(index=3000, split="evaluation")])])
    with pytest.raises(SplitLeak, match="does not exist"):
        fr.evaluation_frame(frame, path, repo=tmp_path)
    sel = fr.select_budgets(frame[frame["split"] == "selection"])
    fr.write_budget_selection(path, sel, git_commit="abc")
    with pytest.raises(SplitLeak, match="not committed at HEAD"):
        fr.evaluation_frame(frame, path, repo=tmp_path)
    git("add", "budget_selection.json")
    git("commit", "-q", "-m", "selection")
    out = fr.evaluation_frame(frame, path, repo=tmp_path)
    assert set(out["split"]) == {"evaluation"} and len(out) == 1
    assert out.attrs["budget_selection"]["selection"] == sel
    # Edited after the commit, with or without a consistent digest: refused.
    original = path.read_text()
    path.write_text(original.replace('"b_star": 0.05', '"b_star": 0.5'))
    with pytest.raises(SplitLeak, match="differs from the copy at HEAD"):
        fr.evaluation_frame(frame, path, repo=tmp_path)
    git("commit", "-q", "-am", "tampered")
    with pytest.raises(SplitLeak, match="digest"):
        fr.evaluation_frame(frame, path, repo=tmp_path)
