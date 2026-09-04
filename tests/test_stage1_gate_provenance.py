"""The Stage 2 gate refuses a Stage 1 table whose stamp means nothing. CPU-only.

Companion to tests/test_provenance_consulted.py, which asserts the field is
classified as gated. This asserts the gate actually refuses -- the two are
different claims, and only the pair is worth anything.

The distinction under test: a WRONG commit and a MEANINGLESS commit are
different failures. `Stage1CommitError` already covered the first. The second
passes every commit check by construction, because the commit is real and
matches; it is the working tree that does not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.sweep import (Stage1CommitError, Stage1ProvenanceError,
                             explain_stage1_rejections, load_stage1_pass_set)

COMMIT = "b6ed63bc6e61c11efb272867ba11e7849be8cb23"


def _table(tmp_path, **overrides):
    row = dict(backend="fa2", config_key="abc123", passed=True,
               check_kind="exact", git_commit=COMMIT, git_dirty=False)
    row.update(overrides)
    p = tmp_path / "correctness.parquet"
    pd.DataFrame([row]).to_parquet(p)
    return p


def test_a_clean_table_still_loads(tmp_path):
    p = _table(tmp_path)
    assert load_stage1_pass_set(p, at_commit=COMMIT) == {("fa2", "abc123")}


def test_a_dirty_table_is_refused(tmp_path):
    p = _table(tmp_path, git_dirty=True)
    with pytest.raises(Stage1ProvenanceError, match="dirty working tree"):
        load_stage1_pass_set(p, at_commit=COMMIT)


def test_the_dirty_refusal_survives_a_matching_commit(tmp_path):
    """The 2026-09-04 shape exactly. The commit asked for is the commit
    recorded, so every commit check passes -- and the code that ran was 18
    commits ahead of it."""
    p = _table(tmp_path, git_dirty=True)
    with pytest.raises(Stage1ProvenanceError):
        load_stage1_pass_set(p, at_commit=COMMIT)


def test_dirty_is_checked_before_the_commit_filter(tmp_path):
    """Order matters. Filtering to the requested commit first and checking
    dirtiness after would hand back a set that satisfied the commit check
    while describing other code -- the dirty rows would simply have been
    filtered out of view first if they carried a different commit, and kept
    silently if they carried this one."""
    p = _table(tmp_path, git_dirty=True, git_commit="c" * 40)
    with pytest.raises(Stage1ProvenanceError):
        load_stage1_pass_set(p, at_commit=COMMIT)


def test_a_wrong_commit_is_still_its_own_error(tmp_path):
    """The pre-existing check must not be swallowed by the new one."""
    p = _table(tmp_path, git_commit="c" * 40)
    with pytest.raises(Stage1CommitError):
        load_stage1_pass_set(p, at_commit=COMMIT)


def test_allow_dirty_is_an_explicit_opt_in(tmp_path):
    """Local development needs a hatch. It is off by default, and taking it
    is a visible act at the call site rather than a config default."""
    p = _table(tmp_path, git_dirty=True)
    assert load_stage1_pass_set(p, at_commit=COMMIT,
                                allow_dirty=True) == {("fa2", "abc123")}


def test_a_table_without_the_column_still_loads(tmp_path):
    """Rows predating the field must not be refused: absence of evidence is
    not a dirty tree, and refusing them would delete prior datasets."""
    p = tmp_path / "c.parquet"
    pd.DataFrame([dict(backend="fa2", config_key="abc123", passed=True,
                       git_commit=COMMIT)]).to_parquet(p)
    assert load_stage1_pass_set(p, at_commit=COMMIT) == {("fa2", "abc123")}


def test_the_rejection_explainer_names_dirtiness_specifically(tmp_path):
    """A refusal nobody can act on costs a session. The explainer must say
    DIRTY, not repeat the commit story -- they have different fixes."""
    from attnbench.config import AttnConfig
    from attnbench.sweep import SweepCell

    p = _table(tmp_path, git_dirty=True)
    cfg = AttnConfig(seq_len=64, batch=1, n_heads_q=4, n_heads_kv=4,
                     head_dim=64, dtype="float32")
    cells = [SweepCell(backend_name="fa2", cfg=cfg)]
    out = explain_stage1_rejections(cells, p)
    assert "DIRTY PROVENANCE" in out["fa2"]
