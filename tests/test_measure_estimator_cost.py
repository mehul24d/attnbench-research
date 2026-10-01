"""scripts/measure_estimator_cost.py: shape of what it banks, on CPU.

The numbers mean nothing here (perf_counter on a CPU, toy shapes); what is
checked is that every component is either measured or recorded as
unavailable with a reason, never silently missing, and that the summary only
pairs components measured at the same (seq_len, sparsity).
"""

import importlib.util
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "_est_cost", Path(__file__).resolve().parents[1] / "scripts" / "measure_estimator_cost.py")
mec = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mec)

GEOM = dict(layers=2, heads_q=4, heads_kv=2, head_dim=8)


def _rows():
    return pd.DataFrame(mec.measure(seq_lens=[64, 128], sparsities=[0.5, 0.75],
                                    thresholds=[0.9], block_size=16, geometry=GEOM,
                                    device="cpu", dtype="float32", warmup=1, reps=3))


def test_every_component_is_measured_or_says_why_not():
    df = _rows()
    for comp, n in (("meanpool", 2), ("dense", 2), ("mask_build", 4),
                    ("block_sparse", 4), ("xattn_estimate", 2)):
        got = df[df.component == comp]
        assert len(got) == n, comp
        assert (got.available | (got.reason.str.len() > 0)).all(), comp
    assert df[df.component.isin(["meanpool", "mask_build"])].available.all()
    built = df[df.component == "mask_build"]
    assert (built.block_density > 0).all() and (built.block_density < 1).all()
    # density falls as sparsity rises, at each length
    for s, g in built.groupby("seq_len"):
        assert g.sort_values("sparsity").block_density.is_monotonic_decreasing


def test_summary_pairs_only_complete_cells():
    df = _rows()
    # a fabricated complete cell: the summary must use exactly these numbers
    fake = pd.DataFrame([
        dict(component="dense", seq_len=999, sparsity=None, median_ms=10.0, available=True),
        dict(component="meanpool", seq_len=999, sparsity=None, median_ms=0.5, available=True),
        dict(component="mask_build", seq_len=999, sparsity=0.5, median_ms=0.25, available=True),
        dict(component="block_sparse", seq_len=999, sparsity=0.5, median_ms=6.0, available=True),
    ])
    out = mec.summarise(pd.concat([df, fake], ignore_index=True), layers=28)
    row = out[out.seq_len == 999].iloc[0]
    assert row.saving_ms == 4.0 and row.estimator_ms == 0.75
    assert row.estimator_over_saving == 0.75 / 4.0
    assert row.net_saving_per_forward_ms == 28 * 3.25
