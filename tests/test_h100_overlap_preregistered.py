"""The pre-registered H100 overlap test, scored from the banked parquets.

`docs/h100_overlap_preregistration.md` fixed three predictions (commit
`597a7a9`) before the H100 session ran. This file scores them exactly as
written -- thresholds copied, not tuned -- and asserts the outcome the docs
report, including the prediction that FAILED. A test that only pinned the
passes would let the failure be rounded away later.

Outcome, 2026-10-01: P1 9/9 (pass needs 7), P2 7/9 (needs 9: fails at
8192/0.90 and 32768/0.75), P3 9/9.

Force-committed with the other `results/s12_*` evidence, so it runs in a
clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import test_overlap_paired as paired

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
H100_SA = RESULTS / "s12_h100_hostcpu" / "builders.parquet"
H100_IN = RESULTS / "s12_h100_vec_endtoend" / "vec_endtoend.parquet"
PREREG = REPO / "docs" / "h100_overlap_preregistration.md"
LIMITATIONS = REPO / "docs" / "limitations.md"
NEEDED = (H100_SA, H100_IN, paired.A100_SA, paired.A100_IN)
MISSING = [str(p.relative_to(REPO)) for p in NEEDED if not p.exists()]
needs_banked = pytest.mark.skipif(
    bool(MISSING), reason=f"H100 inputs not in this checkout: {MISSING}")

ALPHA_LO, ALPHA_HI, SLACK = 0.45, 0.85, 0.10     # copied from the pre-registration
P1_PASS_AT = 7
# The A100 reference values the pre-registration printed, (r, exposed share).
A100_REF = {
    (8192, 0.5): (1.30, 0.62), (8192, 0.75): (0.91, 0.42), (8192, 0.9): (0.63, 0.13),
    (16384, 0.5): (2.25, 0.77), (16384, 0.75): (1.51, 0.60), (16384, 0.9): (0.97, 0.34),
    (32768, 0.5): (3.87, 0.88), (32768, 0.75): (2.66, 0.79), (32768, 0.9): (1.65, 0.60),
}
# What limitations.md reports, to 0.1 ms / 0.01: (standalone, in model, r, share)
H100_TABLE = {
    (8192, 0.5): (117.7, 73.3, 1.37, 0.62),
    (8192, 0.75): (76.6, 38.7, 0.96, 0.51),
    (8192, 0.9): (47.1, 16.8, 0.61, 0.36),
    (16384, 0.5): (408.4, 331.4, 2.32, 0.81),
    (16384, 0.75): (247.5, 176.4, 1.59, 0.71),
    (16384, 0.9): (153.9, 88.5, 1.08, 0.57),
    (32768, 0.5): (1571.5, 1313.7, 3.86, 0.84),
    (32768, 0.75): (935.2, 685.0, 2.84, 0.73),
    (32768, 0.9): (502.7, 329.8, 1.78, 0.66),
}
P2_FAILS = {(8192, 0.9), (32768, 0.75)}


def cell(band: int, s: float) -> tuple[float, float, float, float]:
    pd = pytest.importorskip("pandas")
    sa = paired.standalone(H100_SA, band, s)
    inm = paired.in_model(H100_IN, band, s)
    d = pd.read_parquet(H100_IN)
    gpu = float(paired._one(d, band=band, builder="vectorised", backend="block_sparse",
                            sparsity=s).prefill_ms_mean) / 28
    return sa, inm, sa / 28 / gpu, inm / sa


@needs_banked
def test_the_reported_table_is_derived():
    for key, want in H100_TABLE.items():
        sa, inm, r, share = cell(*key)
        assert (sa, inm) == pytest.approx(want[:2], abs=0.05), key
        assert (r, share) == pytest.approx(want[2:], abs=0.005), key


@needs_banked
def test_the_a100_reference_values_are_what_the_preregistration_printed():
    """P2 compares against these; they must be the banked A100 values, not
    numbers retyped later."""
    pd = pytest.importorskip("pandas")
    d = pd.read_parquet(paired.A100_IN)
    for (band, s), (r_ref, share_ref) in A100_REF.items():
        sa = paired.standalone(paired.A100_SA, band, s)
        gpu = float(paired._one(d, band=band, builder="vectorised",
                                backend="block_sparse", sparsity=s).prefill_ms_mean) / 28
        assert sa / 28 / gpu == pytest.approx(r_ref, abs=0.005), (band, s)
        assert paired.in_model(paired.A100_IN, band, s) / sa == pytest.approx(share_ref, abs=0.005)


@needs_banked
def test_p1_passes_nine_of_nine():
    inside = 0
    for key in H100_TABLE:
        _, _, r, share = cell(*key)
        lo = max(0.0, 1 - ALPHA_HI / r) - SLACK
        hi = max(0.0, 1 - ALPHA_LO / r) + SLACK
        inside += lo <= share <= hi
    assert inside == 9 and inside >= P1_PASS_AT


@needs_banked
def test_p2_fails_in_exactly_the_two_reported_cells():
    """'In every cell' was the pre-registered bar. It is not met, and the
    docs say so; this pins which cells, so the failure cannot quietly move."""
    fails = set()
    for key, (r_a, share_a) in A100_REF.items():
        _, _, r, share = cell(*key)
        ok = share > share_a if r > r_a else share <= share_a
        if not ok:
            fails.add(key)
    assert fails == P2_FAILS


@needs_banked
def test_p3_sapphire_rapids_builds_faster_everywhere():
    for key in H100_TABLE:
        assert paired.standalone(H100_SA, *key) < paired.standalone(paired.A100_SA, *key)


def test_the_docs_state_the_outcome_including_the_failure():
    text = PREREG.read_text() + LIMITATIONS.read_text()
    for phrase in ("P1", "9 of 9", "P2", "7 of 9", "FAILED"):
        assert phrase in text, phrase
    lim = LIMITATIONS.read_text()
    missing = [f"{x:.1f}" for row in H100_TABLE.values() for x in row[:2]
               if f"{x:.1f}" not in lim]
    assert not missing, f"limitations.md no longer states {missing}"
