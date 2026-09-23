"""S11's figures, derived from the banked parquets rather than restated.

**Why this file exists.** S11 was closed on 2026-09-21 from the figures
`scripts/measure_mask_h2d_tax.py` printed in its headline, not from the columns
it banked. The run produced four candidate per-call figures and a resolution
floor; the disposition used one candidate, never compared anything to the
floor, and took one of its three denominators from a table `claims.md`
supersedes 20 lines later. All three errors ran in the same direction, so
nothing downstream disagreed and the closure survived a full audit pass.

That is the shape this repository already has a name for: a claim no consumer
can contradict has to be checked against its own inputs. So the inputs are the
test. Every percentage in the S11 row of `docs/audit_register.md` and in
`docs/limitations.md`'s timed-region section is recomputed here from the
parquets and asserted to match, and the script's own pre-registered stop
condition is replayed against the banked rows so it cannot quietly stop firing.

**Gated on the files it reads, not on a count.** Two of the five inputs are
force-committed (`s7_sweep_hl122`, `s8_vec_endtoend`) and three are not, so this
skips in a clone and runs on a working tree -- see instance #52 and #53 for why
the guard names the files instead of counting them.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

S11_A100 = RESULTS / "s11_mask_h2d_a100" / "mask_h2d.parquet"
S11_L4 = RESULTS / "s11_mask_h2d_l4" / "mask_h2d.parquet"
SEG1 = RESULTS / "stage2" / "segment_20260903_seg1" / "segment.parquet"
S7_SWEEP = RESULTS / "s7_sweep_hl122" / "sweep.parquet"      # tracked
S8_VEC = RESULTS / "s8_vec_endtoend" / "vec_endtoend.parquet"  # tracked

NEEDED = (S11_A100, S11_L4, SEG1, S7_SWEEP, S8_VEC)
MISSING = [str(p.relative_to(REPO)) for p in NEEDED if not p.exists()]

REGISTER = REPO / "docs" / "audit_register.md"
LIMITATIONS = REPO / "docs" / "limitations.md"

needs_banked = pytest.mark.skipif(
    bool(MISSING), reason=f"S11 inputs not in this checkout: {MISSING}")


def _script():
    """The measurement script's own analysis, imported for replay. Importing it
    must not need CUDA -- `analyse()` is pure for exactly that reason."""
    spec = importlib.util.spec_from_file_location(
        "_s11", REPO / "scripts" / "measure_mask_h2d_tax.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s11"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(path: Path) -> list[dict]:
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(path).sort_values("band")
    return df.to_dict("records")


# --- anti-vacuity -----------------------------------------------------------

@needs_banked
def test_the_inputs_are_all_present_and_nonempty():
    """If any of these came back empty every check below would pass on nothing."""
    for p in NEEDED:
        assert p.stat().st_size > 0, p
    for path in (S11_A100, S11_L4):
        rows = _rows(path)
        assert len(rows) == 5, f"{path} has {len(rows)} bands, expected 5"
        assert [r["band"] for r in rows] == [1024, 2048, 4096, 8192, 16384]


# --- the stop condition the first closure did not apply ---------------------

@needs_banked
@pytest.mark.parametrize("path,card", [(S11_A100, "A100"), (S11_L4, "L4")])
def test_the_method_does_not_resolve_the_copy_below_16384(path, card):
    """The pre-registered stop condition, replayed. If this ever stops firing
    the floor has changed and the constants must be re-derived, not reused."""
    m = _script()
    rows = _rows(path)
    a = m.analyse(rows, float(rows[0]["floor_p50_us"]))
    assert a["unresolvable_bands"] == [1024, 2048, 4096, 8192], (
        f"{card}: expected the copy to be unresolvable below 16384, got "
        f"{a['unresolvable_bands']}. The 2026-09-21 closure published the "
        f"1024 figure as a measured constant when 99% of it was harness.")
    assert a["floor_share"][1024] > 0.95, (
        f"{card}: the floor is {100 * a['floor_share'][1024]:.1f}% of the 1024 "
        f"figure; it was 99.1% (A100) / 98.8% (L4) when this was written")


@needs_banked
@pytest.mark.parametrize("path,card,lo", [(S11_A100, "A100", 15.0),
                                          (S11_L4, "L4", 14.0)])
def test_the_structural_verdict_is_size_sensitive_not_launch_dominated(path, card, lo):
    """The conclusion that replaced "launch-dominated". Net of the floor the
    marginal cost rises by more than an order of magnitude, which is the
    script's own falsifier for "roughly constant per call"."""
    m = _script()
    rows = _rows(path)
    a = m.analyse(rows, float(rows[0]["floor_p50_us"]))
    assert a["launch_dominated"] is False, a["verdict"]
    assert a["marginal_ratio"] > lo, (
        f"{card}: marginal ratio {a['marginal_ratio']:.1f}x; it was 19.3x "
        f"(A100) / 16.7x (L4). The raw p50 ratio is {a['raw_ratio']:.2f}x, "
        f"which is what made this look flat.")
    assert a["raw_ratio"] < 1.5, (
        "the raw ratio is no longer flat, so the floor no longer dominates and "
        "this test's framing needs revisiting rather than its threshold")


@needs_banked
@pytest.mark.parametrize("path,card", [(S11_A100, "A100"), (S11_L4, "L4")])
def test_the_throughput_form_is_wholly_inside_the_assumed_range(path, card):
    """The cross-check the first closure did not report. The register claimed
    both constants contradicted the assumed 15-30 us range; the same run's
    throughput form sits inside it at every band on both cards."""
    m = _script()
    rows = _rows(path)
    a = m.analyse(rows, float(rows[0]["floor_p50_us"]))
    thru = a["candidates"]["copy_throughput"]
    assert thru["wholly_inside"], (
        f"{card}: throughput form {thru['min']:.2f}-{thru['max']:.2f} us is no "
        f"longer wholly inside {m.ASSUMED_LO}-{m.ASSUMED_HI}")
    lat = a["candidates"]["copy_latency"]
    assert not lat["wholly_inside"] and lat["bands_inside"] >= 3, (
        f"{card}: the latency form has {lat['bands_inside']}/5 bands inside. "
        f"The register said A100 was 'at or above the assumed upper end at "
        f"every band'; three of five are below 30 us.")


# --- the denominators, and the host/geometry pairing -----------------------

@needs_banked
def test_the_1024_denominator_is_an_L4_row():
    """The pairing the 2026-09-21 pass got right, pinned so it stays right."""
    pd = pytest.importorskip("pandas")
    d = pd.read_parquet(SEG1)
    bs = d[(d.backend == "block_sparse") & d.latency_ms_p50.notna()]
    assert not bs.empty, "no block_sparse rows with a p50 in this segment"

    # Derived as "the fastest Stage 2 cell", which is what the register claims,
    # rather than by a (seq_len, sparsity, block_size) selector -- that selector
    # matches six rows here because the segment sweeps `batch` as well.
    fastest = bs.loc[bs.latency_ms_p50.idxmin()]
    assert round(float(fastest.latency_ms_p50), 3) == 0.801
    assert int(fastest.seq_len) == 1024
    assert float(fastest.sparsity) == 0.75
    assert int(fastest.block_size) == 128
    assert "L4" in str(fastest.gpu_name), (
        f"the fastest Stage 2 cell is now a {fastest.gpu_name} row, so the "
        f"1024 cell no longer takes the L4 constant")


@needs_banked
def test_the_8192_and_16384_denominators_are_the_real_head_geometry():
    """The defect that reopened S11: the 8192 cell used 1.969 ms, an A100 row
    from the (32,8) table `claims.md` supersedes, while the 16384 cells used
    the (12,2) geometry the model actually has. One sentence, two geometries.
    """
    pd = pytest.importorskip("pandas")
    d = pd.read_parquet(S7_SWEEP)
    assert set(zip(d.n_heads_q, d.n_heads_kv, d.head_dim)) == {(12, 2, 128)}, (
        "s7_sweep_hl122 is no longer the (12,2) sweep")
    bs = d[(d.backend == "block_sparse") & (d.block_size == 128)]
    got = {(int(r.seq_len), float(r.sparsity)): round(float(r.latency_ms_p50), 3)
           for r in bs.itertuples()}
    assert got[(8192, 0.5)] == 1.544
    assert got[(8192, 0.75)] == 1.230
    assert got[(8192, 0.9)] == 1.133
    assert got[(16384, 0.5)] == 3.373
    assert got[(16384, 0.75)] == 2.266
    assert got[(16384, 0.9)] == 1.795
    assert 1.969 not in got.values(), (
        "1.969 ms now appears in the real-geometry sweep, which would make the "
        "withdrawn 8192 denominator legitimate again -- re-derive S11")


@needs_banked
def test_the_end_to_end_denominators_are_the_banked_vectorised_rows():
    pd = pytest.importorskip("pandas")
    d = pd.read_parquet(S8_VEC)
    v = d[d.builder == "vectorised"]
    bs = sorted(round(float(x), 1)
                for x in v[v.backend == "block_sparse"].prefill_ms_mean
                if float(x) > 300)
    assert bs == [340.4, 363.4, 400.4], bs
    dense = [round(float(x), 1) for x in v[v.backend == "sdpa_flash"].prefill_ms_mean
             if float(x) > 300]
    assert dense == [436.5], dense
    # and the ratios the tax is a fraction OF
    ratios = sorted(round(436.479227 / m, 3) for m in (400.350147, 363.440243,
                                                       340.355859))
    assert ratios == [1.090, 1.201, 1.282], ratios


# --- the documented percentages, recomputed --------------------------------

def _pct(constant_us: float, denominator_ms: float) -> float:
    return 100.0 * (constant_us / 1000.0) / denominator_ms


@needs_banked
def test_the_documented_per_kernel_call_figures_are_the_derivation():
    """Upper bound = the full call, latency form, against the real-geometry
    denominators. These are the numbers the register now states."""
    a = {r["band"]: r for r in _rows(S11_A100)}
    l4 = {r["band"]: r for r in _rows(S11_L4)}
    full_a = lambda b: float(a[b]["full_call_p50_us"])      # noqa: E731
    full_l = lambda b: float(l4[b]["full_call_p50_us"])     # noqa: E731

    assert round(_pct(full_l(1024), 0.801), 2) == 4.90
    assert [round(_pct(full_a(8192), m), 2)
            for m in (1.544192, 1.229824, 1.132544)] == [2.87, 3.61, 3.92]
    assert [round(_pct(full_a(16384), m), 2)
            for m in (3.373056, 2.266112, 1.795072)] == [1.43, 2.12, 2.68]

    # lower bounds, from the throughput form
    thru_a = lambda b: float(a[b]["batched_per_call_us"])    # noqa: E731
    assert round(_pct(float(l4[1024]["batched_per_call_us"]), 0.801), 2) == 1.89
    assert [round(_pct(thru_a(8192), m), 2)
            for m in (1.544192, 1.229824, 1.132544)] == [1.23, 1.55, 1.68]
    assert [round(_pct(thru_a(16384), m), 2)
            for m in (3.373056, 2.266112, 1.795072)] == [0.68, 1.01, 1.28]

    text = REGISTER.read_text()
    for stated in ("**4.90%**", "**2.87 / 3.61 / 3.92%**",
                   "**1.43 / 2.12 / 2.68%**", "1.89%", "1.23-1.68%",
                   "0.68-1.28%"):
        assert stated in text, (
            f"the register no longer states {stated}; it and this derivation "
            f"have to move together")


@needs_banked
def test_the_documented_end_to_end_figures_are_the_derivation():
    """28 copies per prefill, one per layer, against the vectorised A100 16384
    rows. This is the bound the 1.282x floor claim rests on."""
    a = {r["band"]: r for r in _rows(S11_A100)}
    LAYERS = 28
    upper = [round(_pct(LAYERS * float(a[16384]["full_call_p50_us"]), m), 3)
             for m in (400.350147, 363.440243, 340.355859)]
    lower = [round(_pct(LAYERS * float(a[16384]["batched_per_call_us"]), m), 3)
             for m in (400.350147, 363.440243, 340.355859)]
    assert upper == [0.336, 0.371, 0.396], upper
    assert lower == [0.160, 0.176, 0.188], lower

    text = REGISTER.read_text()
    assert "**0.336 / 0.371 / 0.396%**" in text
    assert "0.160 / 0.176 / 0.188%" in text
    assert "at most ~0.40%" in text, (
        "the register no longer states the ~0.40% bound; max(upper) is "
        f"{max(upper)}%")
    # the floor claim must be stated as a bound that the derivation supports
    assert max(upper) < 0.40, f"max upper bound is {max(upper)}%, not under 0.40"


@needs_banked
def test_limitations_states_the_same_bracket_as_the_register():
    """F10's propagation. The first closure's figures were consistent across
    both files -- consistently the wrong column. Consistency is not enough on
    its own, so both are checked against the parquets, not against each other.
    """
    a = {r["band"]: r for r in _rows(S11_A100)}
    l4 = {r["band"]: r for r in _rows(S11_L4)}
    text = LIMITATIONS.read_text()

    for rows, card in ((a, "A100"), (l4, "L4")):
        vals = [float(r["full_call_p50_us"]) for r in rows.values()]
        span = f"{min(vals):.2f}–{max(vals):.2f} µs"
        assert span in text, f"{card}: limitations.md does not state {span}"

    for floor, card in ((float(a[1024]["floor_p50_us"]), "A100"),
                        (float(l4[1024]["floor_p50_us"]), "L4")):
        assert f"{floor:.3f} µs" in text, (
            f"{card}: the {floor:.3f} us harness floor is not stated in "
            f"limitations.md, and it is the reason the constant is a bracket")

    assert "1.4–4.9% per kernel call" in text
    assert "0.34–0.40% end-to-end" in text
    assert "19.3× (A100) / 16.7× (L4)" in text


@needs_banked
def test_neither_doc_still_quotes_the_copy_as_the_measured_constant():
    """The withdrawn FRAMING, not the withdrawn digits.

    The digits may and do appear as history, inside `(was "...")` corrections;
    whether such an occurrence is adequately marked is
    `test_no_stale_figures.py`'s job and the registry in
    `docs/withdrawn_figures.md` carries the entries. What this asserts is
    narrower and is not covered there: that no live sentence still presents the
    bare-copy figure as the per-call magnitude.
    """
    def _without_history(text: str) -> str:
        """Drop italic `*( ... )*` correction spans and double-quoted spans.

        A withdrawn figure quoted inside its own correction is the CORRECT way
        to state one -- `test_no_stale_figures.py` has a test saying so -- so
        searching the raw text would flag the fix as the defect. What is left
        after stripping those is the document's live voice.
        """
        text = re.sub(r"\*\((?:.|\n)*?\)\*", " ", text)
        return re.sub(r"\"(?:[^\"\n]|\n)*?\"", " ", text)

    live_phrasings = (
        "p50 29.7-34.6 us per call",      # limitations.md's old summary
        "p50 26.5-31.7 us per call",
        "a fraction of a percent (0.24-3.3%",
        "floor understated by at most ~0.28%",
    )
    for path in (REGISTER, LIMITATIONS):
        text = _without_history(path.read_text())
        for bad in live_phrasings:
            assert bad not in text, (
                f"{path.name} still states {bad!r} as a live figure; it is the "
                f"bare `.to()` (or a percentage derived from it), not "
                f"`to_block_sparse_attn_mask()`")
    # and the replacement framing is present
    assert "bracket, not a constant" in LIMITATIONS.read_text()
