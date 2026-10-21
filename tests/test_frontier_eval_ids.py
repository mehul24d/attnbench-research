"""The evaluation split against the banked files (estimator-frontier
pre-registration sec. 5; added 2026-10-03).

The evaluation split must use example indices no banked run has used, for
that task at any band, because RULER QA picks its question by index alone.
`frontier_prereg.BANKED_INDEX_RANGES` is the registry; these tests re-derive
it from the parquets on disk. The pure half is in
`tests/test_frontier_prereg_plan.py` and runs everywhere.

Also here: the six L4-A100 cells of H1 are out of sample, which needs no
banked estimator timing from any card but the L4.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from attnbench.analysis import frontier_prereg as fp

RESULTS = Path(__file__).resolve().parents[1] / "results"
_ID = re.compile(r"^(.*)_(\d+)_(\d+)$")


def _banked() -> dict:
    """(task, band) -> set of example indices, over every parquet on disk
    that has an `example_id` column."""
    out: dict = {}
    for p in sorted(RESULTS.rglob("*.parquet")):
        if not p.is_file() or "example_id" not in pq.ParquetFile(p).schema.names:
            continue
        for eid in pd.read_parquet(p, columns=["example_id"]).example_id.dropna().unique():
            m = _ID.match(str(eid))
            assert m, f"{p}: unparsed example id {eid!r}"
            out.setdefault((m.group(1), int(m.group(2))), set()).add(int(m.group(3)))
    return out


def test_no_evaluation_id_appears_in_any_banked_parquet():
    banked = _banked()
    if not banked:
        pytest.skip("banked results not present")
    evaluation = set(fp.evaluation_indices(300))
    for (task, band), idx in banked.items():
        both = sorted(evaluation & idx)
        assert not both, f"evaluation indices {both[:3]} are banked for {task} at {band}"
        assert (task, band) in fp.BANKED_INDEX_RANGES, (task, band)
        lo, hi = fp.BANKED_INDEX_RANGES[(task, band)]
        assert lo <= min(idx) and max(idx) <= hi, (task, band, min(idx), max(idx))


def test_the_registry_is_the_whole_banked_tree():
    if not (RESULTS / "stage3_32768").is_dir():
        pytest.skip("banked results not present")
    banked = {k: (min(v), max(v)) for k, v in _banked().items()}
    assert banked == fp.BANKED_INDEX_RANGES


def test_no_estimator_timing_is_banked_off_the_l4():
    """H1's six L4-A100 cells are out of sample only if the estimator's cost
    was never measured on an A100 or H100 (sec. 3, H1)."""
    files = sorted(RESULTS.rglob("estimator_cost*.parquet"))
    if not files:
        pytest.skip("banked results not present")
    for p in files:
        assert set(pd.read_parquet(p, columns=["gpu_name"]).gpu_name) == {"NVIDIA L4"}, p
