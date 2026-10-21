"""The A100 baseline-strength caveat in `docs/claims.md` (2026-10-03),
recomputed from the two parquets it cites.

Every figure in the caveat is derived from data here, not restated:

- the per-layer dense kernels at `(12,2)`, from `results/s7_sweep_hl122`;
- the vectorised end-to-end prefills, from `results/s8_vec_endtoend`.

If either the doc or the data moves, this fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "results/s7_sweep_hl122/sweep.parquet"
E2E = REPO / "results/s8_vec_endtoend/vec_endtoend.parquet"
CLAIMS = REPO / "docs/claims.md"
LAYERS = 28
FASTEST = {8192: "sdpa_cudnn", 16384: "fa2"}
SPARSITIES = (0.5, 0.75, 0.9)

pytestmark = pytest.mark.skipif(not (SWEEP.exists() and E2E.exists()),
                                reason="banked parquets not present")


def _kernel(sweep, backend, band, sparsity=None):
    x = sweep[(sweep.pass_kind == "fwd") & (sweep.backend == backend)
              & (sweep.seq_len == band)]
    if sparsity is not None:
        x = x[x.sparsity == sparsity]
    assert len(x) == 1, (backend, band, sparsity, len(x))
    return float(x.latency_ms_p50.iloc[0])


def derived() -> dict[str, str]:
    """Every figure the caveat states, formatted as the caveat prints it."""
    sweep, e2e = pd.read_parquet(SWEEP), pd.read_parquet(E2E)
    e2e = e2e[e2e.builder == "vectorised"]
    out = {}
    for band, alt in FASTEST.items():
        flash, fast = _kernel(sweep, "sdpa_flash", band), _kernel(sweep, alt, band)
        out[f"{band} alt ms"] = f"{fast:.3f}"
        out[f"{band} flash ms"] = f"{flash:.3f}"
        out[f"{band} pct faster"] = f"{100 * (1 - fast / flash):.1f}%"
        dense = float(e2e[(e2e.band == band) & (e2e.backend == "sdpa_flash")]
                      .prefill_ms_mean.iloc[0])
        adj = dense - LAYERS * (flash - fast)
        out[f"{band} dense e2e"] = f"{dense:.1f}"
        out[f"{band} adj e2e"] = f"{adj:.1f}"
        for sp in SPARSITIES:
            bsa = _kernel(sweep, "block_sparse", band, sp)
            arm = float(e2e[(e2e.band == band) & (e2e.backend == "block_sparse")
                            & (e2e.sparsity == sp)].prefill_ms_mean.iloc[0])
            out[f"{band} {sp} kernel vs alt"] = f"{fast / bsa:.3f}×"
            out[f"{band} {sp} e2e est"] = f"{adj / arm:.3f}×"
    return out


def caveat_text() -> str:
    text = CLAIMS.read_text()
    blocks = re.findall(r"> \*\*Baseline-strength caveat.*?(?=\n\n[^>])", text, re.S)
    assert len(blocks) == 2, f"expected the two caveat blocks, found {len(blocks)}"
    return "\n".join(blocks)


@pytest.mark.parametrize("name", sorted(derived()))
def test_every_caveat_figure_resolves_to_the_parquets(name):
    value = derived()[name]
    assert value in caveat_text(), f"{name} = {value} is not stated in the caveat"


def test_the_8192_sign_flip_is_real():
    """The caveat's one qualitative claim: at 8192 against cuDNN, the
    vectorised end-to-end estimate is below 1 at every sparsity."""
    d = derived()
    assert all(float(d[f"8192 {sp} e2e est"].rstrip("×")) < 1 for sp in SPARSITIES)
