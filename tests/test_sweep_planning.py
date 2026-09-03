"""sweep.py's preconditions and dry-run/real-run parity. Pure functions of
(cells, stage1 pass set, done keys, exclusivity status) -- no GPU touched.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.config import AttnConfig, SweepGrid
from attnbench.sweep import (SweepCell, build_cells, load_done_keys,
                              load_stage1_pass_set, plan, run_sweep)
from attnbench.timing import Measurement


def _cfg(seq_len: int = 128) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=2, n_heads_kv=2, head_dim=64)


def test_missing_stage1_row_is_a_rejection_not_a_pass():
    """The precondition this project's config-hash mistake makes concrete:
    right after a hash-forking field change, every prior row is missing,
    not failing -- and missing must gate exactly like failing does."""
    cell = SweepCell(cfg=_cfg(), backend_name="fake")
    decisions = plan([cell], stage1_passed=set(), done_keys=set(),
                      host="h", exclusivity_status="exclusive")
    assert decisions[0].action == "reject_stage1"


def test_stage1_pass_allows_run():
    cell = SweepCell(cfg=_cfg(), backend_name="fake")
    decisions = plan([cell], stage1_passed={("fake", cell.cfg.key())},
                      done_keys=set(), host="h", exclusivity_status="exclusive")
    assert decisions[0].action == "run"


@pytest.mark.parametrize("status", ["contaminated", "unverifiable"])
def test_non_exclusive_rejects_even_with_stage1_pass(status):
    cell = SweepCell(cfg=_cfg(), backend_name="fake")
    decisions = plan([cell], stage1_passed={("fake", cell.cfg.key())},
                      done_keys=set(), host="h", exclusivity_status=status)
    assert decisions[0].action == "reject_exclusivity"


def test_already_done_cell_is_not_reevaluated_against_current_preconditions():
    """A cell that already ran and was checkpointed stays 'done' even if
    Stage 1 or exclusivity would reject it under today's state -- what
    happened when it ran is the fact of record, not something to re-judge
    against possibly-stale current preconditions."""
    cell = SweepCell(cfg=_cfg(), backend_name="fake")
    done = {(cell.cfg.key(), "fake", "h")}
    decisions = plan([cell], stage1_passed=set(), done_keys=done,
                      host="h", exclusivity_status="contaminated")
    assert decisions[0].action == "skip_done"


def test_load_stage1_pass_set_missing_file_is_empty():
    assert load_stage1_pass_set("/nonexistent/path.parquet") == set()


def test_load_done_keys_missing_checkpoint_is_empty():
    assert load_done_keys("/nonexistent/path.parquet") == set()


def test_dry_run_report_matches_what_a_real_run_would_do(tmp_path):
    """The point of --dry-run is that its counts are not a separate
    approximation of the real gating logic -- running for real afterward
    does exactly what the dry run predicted."""
    cells = [SweepCell(cfg=_cfg(128), backend_name="fake"),
             SweepCell(cfg=_cfg(256), backend_name="fake")]
    stage1_path = tmp_path / "correctness.parquet"
    pd.DataFrame([{"backend": "fake", "config_key": cells[0].cfg.key(), "passed": True}]
                 ).to_parquet(stage1_path)

    def fake_measure(backend, cfg, mask):
        return Measurement(ok=True, status="ok", latency_ms_p50=1.0)

    dry = run_sweep(cells, out_dir=tmp_path / "dry", stage1_path=stage1_path,
                     backend_lookup={"fake": None}, measure_fn=fake_measure,
                     exclusivity_check=lambda: ("exclusive", ""), dry_run=True)
    assert dry.run == 1
    assert dry.reject_stage1 == 1
    assert not (tmp_path / "dry" / "sweep.parquet").exists()

    real = run_sweep(cells, out_dir=tmp_path / "real", stage1_path=stage1_path,
                      backend_lookup={"fake": None}, measure_fn=fake_measure,
                      exclusivity_check=lambda: ("exclusive", ""), dry_run=False)
    assert real.run == dry.run
    assert real.reject_stage1 == dry.reject_stage1

    df = pd.read_parquet(tmp_path / "real" / "sweep.parquet")
    assert len(df) == real.run


def test_build_cells_rejects_importance_mask_source():
    with pytest.raises(ValueError):
        build_cells(SweepGrid(), backends=[], mask_source="importance")


# ---------------------------------------------------------------------------
# Stage 1 passes are evidence about the code that produced them.
# ---------------------------------------------------------------------------

def test_stage1_passes_from_another_commit_are_refused(tmp_path):
    """A correctness pass recorded before a semantics change does not
    describe the code the sweep is about to time.

    The concrete case: the 2026-09-03 `to_dense_bool` fix changed what a
    causal block-sparse mask means, so every block_sparse pass recorded
    before it describes a different function.
    """
    import pandas as pd
    from attnbench.sweep import Stage1CommitError, load_stage1_pass_set

    p = tmp_path / "correctness.parquet"
    pd.DataFrame([dict(backend="flex", config_key="c1", passed=True,
                       git_commit="a" * 40)]).to_parquet(p, index=False)

    assert load_stage1_pass_set(p, at_commit="a" * 40) == {("flex", "c1")}

    with pytest.raises(Stage1CommitError, match="none recorded at the current"):
        load_stage1_pass_set(p, at_commit="b" * 40)

    # unchecked remains possible, and is what the old behaviour did
    assert load_stage1_pass_set(p) == {("flex", "c1")}


def test_stage1_table_without_a_commit_column_is_refused_when_checking(tmp_path):
    import pandas as pd
    from attnbench.sweep import Stage1CommitError, load_stage1_pass_set

    p = tmp_path / "correctness.parquet"
    pd.DataFrame([dict(backend="flex", config_key="c1", passed=True)]).to_parquet(
        p, index=False)
    with pytest.raises(Stage1CommitError, match="no git_commit column"):
        load_stage1_pass_set(p, at_commit="a" * 40)
