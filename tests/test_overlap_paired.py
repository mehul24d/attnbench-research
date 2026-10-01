"""Why the L4 did not pay for mask construction: the paired 2026-10-01 measurement.

**What this pins.** Audit T1 asked why block-sparse prefill wins on an L4 and
loses on an A100 with the same reference mask builder. The 2026-10-01 sessions
answered it with one measurement on each card's own host:

  - **standalone**, `scripts/host_cpu_probe.py builders`: the builder timed
    alone on the host CPU. Both hosts are the same SKU (live `cpuPlatform`
    Intel Cascade Lake on all three instances), and the cost agrees within 3%.
  - **in the model**, `scripts/run_vectorised_endtoend.py`: reference minus
    vectorised end-to-end prefill, same process, same scores -- what the
    forward actually pays.

The A100's forward pays 13-88% of the standalone cost; the L4's pays ~0, except
at 32768/0.50 where building one layer's mask takes as long as the GPU's whole
per-layer work. Kernel launches are asynchronous, so the CPU builds the next
mask while the GPU runs work already queued; the slower card queues more. The
L4/A100 contrast is a host-device balance effect, not a host difference and not
the attention kernel.

**Which run supplies which L4 cell.** The first L4 run timed builder-major, and
its dense control drifted +44 ms at 8192 and +38 ms at 32768 between the two
blocks, so its in-model cells there are good to about that. Its 16384 cells
produced builder pairs agreeing to 0.001 SE -- against 4-213 SE everywhere else
in the run -- and are superseded by the 16384 rerun, which is interleaved per
rep and banks its samples. The A100 run is interleaved throughout.

**Force-committed**, like `s7_sweep_hl122` and `s8_vec_endtoend`, so this runs
in a clone: the conclusion is load-bearing for the paper, and a test that skips
everywhere but one workstation is decoration (instances #52, #53).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
A100_SA = RESULTS / "s12_a100_hostcpu" / "builders.parquet"
A100_IN = RESULTS / "s12_a100_vec_endtoend" / "vec_endtoend.parquet"
L4_SA = RESULTS / "s12_l4_hostcpu" / "builders.parquet"
L4_IN = RESULTS / "s12_l4_vec_endtoend" / "vec_endtoend.parquet"
L4_SA_2 = RESULTS / "s12_l4_rerun16k_hostcpu" / "builders.parquet"
L4_IN_16K = RESULTS / "s12_l4_rerun16k_vec_endtoend" / "vec_endtoend.parquet"
NEEDED = (A100_SA, A100_IN, L4_SA, L4_IN, L4_SA_2, L4_IN_16K)
MISSING = [str(p.relative_to(REPO)) for p in NEEDED if not p.exists()]
LIMITATIONS = REPO / "docs" / "limitations.md"

needs_banked = pytest.mark.skipif(
    bool(MISSING), reason=f"2026-10-01 paired inputs not in this checkout: {MISSING}")

BANDS = (8192, 16384, 32768)
SPARSITIES = (0.5, 0.75, 0.9)
L4_DRIFT_MS = 44.4   # first L4 run's dense-control drift at 8192, builder-major

# limitations.md's paired table, to 0.1 ms. (band, sparsity) ->
# (A100 standalone, L4 standalone, A100 in-model, L4 in-model)
TABLE = {
    (8192, 0.5): (254.4, 254.9, 156.9, -24.5),
    (8192, 0.75): (167.8, 168.5, 70.5, -11.1),
    (8192, 0.9): (113.1, 114.5, 15.1, 10.7),
    (16384, 0.5): (893.2, 895.3, 684.3, 17.5),
    (16384, 0.75): (544.4, 542.3, 328.6, 8.3),
    (16384, 0.9): (328.0, 330.6, 110.4, 1.6),
    (32768, 0.5): (3414.7, 3384.0, 3012.0, 655.7),
    (32768, 0.75): (1977.9, 1961.0, 1553.6, 49.5),
    (32768, 0.9): (1093.8, 1100.4, 656.6, 22.5),
}
# A100 32768, dense / sparse prefill, vectorised builder.
A100_32K_VEC_SPEEDUP = (1.250, 1.481, 1.666)


def _pd():
    return pytest.importorskip("pandas")


def _one(df, **eq):
    for col, val in eq.items():
        df = df[(df[col] - val).abs() < 1e-9] if isinstance(val, float) else df[df[col] == val]
    assert len(df) == 1, f"expected one row for {eq}, got {len(df)}"
    return df.iloc[0]


def standalone(path: Path, band: int, s: float) -> float:
    d = _pd().read_parquet(path)
    ref = _one(d, band=band, builder="reference", sparsity=s).ms_per_forward_p50
    vec = _one(d, band=band, builder="vectorised", sparsity=s).ms_per_forward_p50
    return float(ref - vec)


def in_model(path: Path, band: int, s: float) -> float:
    d = _pd().read_parquet(path)
    row = lambda bu: _one(d, band=band, builder=bu, backend="block_sparse",
                          sparsity=s).prefill_ms_mean
    return float(row("reference") - row("vectorised"))


def l4_in_model(band: int, s: float) -> float:
    return in_model(L4_IN_16K if band == 16384 else L4_IN, band, s)


@needs_banked
def test_the_table_is_derived_from_the_parquets():
    for (band, s), want in TABLE.items():
        got = (standalone(A100_SA, band, s), standalone(L4_SA, band, s),
               in_model(A100_IN, band, s), l4_in_model(band, s))
        assert got == pytest.approx(want, abs=0.05), (band, s, got)


@needs_banked
def test_the_hosts_build_masks_at_the_same_speed():
    """Three hosts, one SKU: the standalone cost agrees within 3% in every
    cell (worst 2.7%, 8192/0.90). Anything that explains the in-model
    difference must leave this intact."""
    for band in BANDS:
        for s in SPARSITIES:
            costs = [standalone(p, band, s) for p in (A100_SA, L4_SA, L4_SA_2)]
            assert max(costs) / min(costs) < 1.03, (band, s, costs)


@needs_banked
def test_the_a100_forward_pays_and_the_l4_forward_hides():
    for band in BANDS:
        for s in SPARSITIES:
            a100 = in_model(A100_IN, band, s) / standalone(A100_SA, band, s)
            l4 = l4_in_model(band, s) / standalone(L4_SA, band, s)
            assert a100 > 0.10, (band, s, a100)
            if (band, s) == (32768, 0.5):
                # The one cell where one layer's build (~122 ms) equals the
                # L4's GPU time per layer (~121 ms): not enough queued work.
                assert 0.10 < l4 < 0.30, l4
            else:
                assert abs(l4) < 0.10, (band, s, l4)
            # Separation is claimed only where the A100's in-model cost
            # clears the builder-major L4 run's own drift (44 ms). At
            # 8192/0.90 it is 15.1 ms, and that cell separates nothing.
            if in_model(A100_IN, band, s) > L4_DRIFT_MS:
                assert a100 > l4 + 0.25, (band, s, a100, l4)


@needs_banked
def test_the_interleaved_runs_have_a_still_dense_control():
    """The reason the A100 and 16384-rerun cells are quoted to 0.1 ms and the
    builder-major L4 cells are not."""
    d = _pd().read_parquet(A100_IN)
    for band in BANDS:
        ref = _one(d, band=band, builder="reference", backend="sdpa_flash").prefill_ms_mean
        vec = _one(d, band=band, builder="vectorised", backend="sdpa_flash").prefill_ms_mean
        assert abs(vec - ref) < 1.0, (band, ref, vec)
    for p in (A100_IN, L4_IN_16K):
        assert set(_pd().read_parquet(p).interleave) == {"per_rep_alternating"}


@needs_banked
def test_the_16384_coincidence_does_not_recur():
    """First L4 run: dense and 0.75 builder pairs agreed to 0.001 SE. The
    interleaved rerun's pairs are ordinary -- here from the banked samples."""
    np = pytest.importorskip("numpy")
    d = _pd().read_parquet(L4_IN_16K)
    for backend, s in (("sdpa_flash", None), ("block_sparse", 0.75)):
        sel = d[d.backend == backend] if s is None else d[(d.backend == backend)
                                                           & ((d.sparsity - s).abs() < 1e-9)]
        x = np.array(sel[sel.builder == "reference"].prefill_ms_samples.iloc[0])
        y = np.array(sel[sel.builder == "vectorised"].prefill_ms_samples.iloc[0])
        se = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 0.5
        assert abs(x.mean() - y.mean()) / se > 0.1, (backend, s)


@needs_banked
def test_a100_32768_with_the_vectorised_builder():
    d = _pd().read_parquet(A100_IN)
    d = d[(d.band == 32768) & (d.builder == "vectorised")]
    dense = _one(d, backend="sdpa_flash").prefill_ms_mean
    got = tuple(float(dense / _one(d, backend="block_sparse", sparsity=s).prefill_ms_mean)
                for s in SPARSITIES)
    assert got == pytest.approx(A100_32K_VEC_SPEEDUP, abs=5e-4), got


def test_limitations_states_the_paired_table():
    """Runs in any checkout: the doc must carry what the banked test derives."""
    text = LIMITATIONS.read_text()
    # The doc's negatives use U+2212, like the rest of limitations.md.
    figures = [f"{x:.1f}".replace("-", "\u2212") for row in TABLE.values() for x in row]
    figures += [f"{x:.3f}" for x in A100_32K_VEC_SPEEDUP]
    missing = sorted({f for f in figures if f not in text})
    assert not missing, f"limitations.md no longer states {missing}"
