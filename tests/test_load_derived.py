"""provenance.load_derived: the analysis stamp is written by eight tools and,
until 2026-09-19, read by none. These pin the four refusals, and the pass."""
import pandas as pd
import pytest

from attnbench.provenance import UnverifiableDerivedInput, load_derived

TOOL = "scripts/run_pareto.py"


def _write(tmp_path, **cols):
    base = {"x": [1.0, 2.0], "analysis_tool": [TOOL, TOOL],
            "analysis_git_commit": ["a" * 40, "a" * 40],
            "analysis_git_dirty": [False, False]}
    base.update(cols)
    p = tmp_path / "d.parquet"
    pd.DataFrame(base).to_parquet(p)
    return p


def test_clean_single_commit_from_expected_tool_passes(tmp_path):
    d = load_derived(_write(tmp_path), expect_tool=TOOL)
    assert list(d.x) == [1.0, 2.0]


def test_missing_stamp_is_refused(tmp_path):
    p = tmp_path / "d.parquet"
    pd.DataFrame({"x": [1.0]}).to_parquet(p)
    with pytest.raises(UnverifiableDerivedInput, match="no analysis stamp"):
        load_derived(p)


def test_dirty_tree_is_refused_even_on_one_row(tmp_path):
    with pytest.raises(UnverifiableDerivedInput, match="dirty tree"):
        load_derived(_write(tmp_path, analysis_git_dirty=[False, True]))


def test_mixed_commits_are_refused(tmp_path):
    with pytest.raises(UnverifiableDerivedInput, match="2 commits"):
        load_derived(_write(tmp_path, analysis_git_commit=["a" * 40, "b" * 40]))


def test_right_shape_wrong_tool_is_refused(tmp_path):
    """The case that type-checks: a decision-map table where a Pareto table
    is expected has plausible columns and the wrong meaning."""
    with pytest.raises(UnverifiableDerivedInput, match="expected"):
        load_derived(_write(tmp_path), expect_tool="scripts/run_decision_map.py")


def test_escape_hatches_are_explicit(tmp_path):
    p = _write(tmp_path, analysis_git_dirty=[True, True],
               analysis_git_commit=["a" * 40, "b" * 40])
    assert len(load_derived(p, allow_dirty=True, allow_mixed_commits=True)) == 2
