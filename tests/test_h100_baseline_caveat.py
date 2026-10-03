"""The H100 baseline-strength caveat in `docs/limitations.md` (2026-10-03),
recomputed from the parquets it cites.

The caveat labels the H100 in-model speedups' baseline strength
**unmeasured**, because no banked H100 file times a dense kernel other than
`sdpa_flash` at the model's (12, 2) geometry. The test checks that premise
and every figure the caveat states:

- the H100 (32, 8) dense comparison, from the Stage 2 sweep;
- the A100 cuDNN/flash ratio at 8192 at both geometries, which is why the
  (32, 8) data cannot size the in-model effect.

Two of the inputs are not tracked in git, so in a clone without them this
module skips. Nothing is read at collection time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
H100_SWEEP = REPO / "results/h100_20260916_stage2/results/sweep/sweep.parquet"
A100_SWEEP_32_8 = REPO / "results/a100/sweep_a100.parquet"
A100_SWEEP_12_2 = REPO / "results/s7_sweep_hl122/sweep.parquet"
H100_E2E = REPO / "results/s12_h100_vec_endtoend/vec_endtoend.parquet"
LIMITATIONS = REPO / "docs/limitations.md"
INPUTS = (H100_SWEEP, A100_SWEEP_32_8, A100_SWEEP_12_2, H100_E2E)

pytestmark = pytest.mark.skipif(not all(p.exists() for p in INPUTS),
                                reason="banked parquets not present")

FIGURES = ("h100 16384 fastest", "h100 16384 flash", "h100 16384 ratio",
           "h100 32768 fastest", "h100 32768 flash", "h100 32768 ratio",
           "h100 4096 cudnn", "h100 4096 flash", "h100 4096 ratio",
           "a100 32x8 cudnn", "a100 32x8 flash", "a100 32x8 ratio",
           "a100 12x2 cudnn", "a100 12x2 flash", "a100 12x2 ratio")


def _dense(path, hq, hkv):
    d = pd.read_parquet(path)
    d = d[(d["mask"] == "causal") & (d.pass_kind == "fwd") & (d.n_heads_q == hq)
          & (d.n_heads_kv == hkv) & (d.batch == 1) & (d.status == "ok")]
    assert not d.duplicated(["seq_len", "backend"]).any(), path
    return d.pivot_table(index="seq_len", columns="backend",
                         values="latency_ms_p50", aggfunc="first")


def derived() -> dict[str, str]:
    h = _dense(H100_SWEEP, 32, 8)
    a32 = _dense(A100_SWEEP_32_8, 32, 8)
    a12 = _dense(A100_SWEEP_12_2, 12, 2)
    out = {}
    for band in (16384, 32768):
        row = h.loc[band].dropna()
        fastest = row.idxmin()
        assert fastest == "flex", (band, fastest)
        out[f"h100 {band} fastest"] = f"{row[fastest]:.3f}"
        out[f"h100 {band} flash"] = f"{row['sdpa_flash']:.3f}"
        out[f"h100 {band} ratio"] = f"{row['sdpa_flash'] / row[fastest]:.2f}×"
    for key, tab, band in (("h100 4096", h, 4096), ("a100 32x8", a32, 8192),
                           ("a100 12x2", a12, 8192)):
        cud, fl = tab.loc[band, "sdpa_cudnn"], tab.loc[band, "sdpa_flash"]
        out[f"{key} cudnn"] = f"{cud:.3f}"
        out[f"{key} flash"] = f"{fl:.3f}"
        out[f"{key} ratio"] = f"{cud / fl:.{2 if key.startswith('h100') else 3}f}×"
    return out


def caveat_text() -> str:
    blocks = re.findall(r"> \*\*H100 baseline strength: unmeasured.*?(?=\n\n[^>])",
                        LIMITATIONS.read_text(), re.S)
    assert len(blocks) == 1, f"expected one H100 caveat block, found {len(blocks)}"
    return blocks[0]


def test_the_figure_list_is_what_derived_produces():
    assert set(derived()) == set(FIGURES)


@pytest.mark.parametrize("name", FIGURES)
def test_every_caveat_figure_resolves_to_the_parquets(name):
    value = derived()[name]
    assert value in caveat_text(), f"{name} = {value} is not stated in the caveat"


def test_no_banked_h100_dense_kernel_at_the_model_geometry():
    """The premise of "unmeasured": every banked H100 parquet with geometry
    columns has no causal (12, 2) row, and the in-model end-to-end file
    timed sdpa_flash as its only dense kernel."""
    for p in REPO.glob("results/h100_*/**/*.parquet"):
        d = pd.read_parquet(p)
        if {"n_heads_q", "n_heads_kv", "mask"} <= set(d.columns):
            assert not ((d.n_heads_q == 12) & (d.n_heads_kv == 2)
                        & (d["mask"] == "causal")).any(), p
    e2e = pd.read_parquet(H100_E2E)
    assert set(e2e.backend) == {"sdpa_flash", "block_sparse"}


def test_the_geometry_changes_the_8192_ranking_on_the_a100():
    d = derived()
    assert float(d["a100 32x8 ratio"].rstrip("×")) > 1 > float(d["a100 12x2 ratio"].rstrip("×"))
