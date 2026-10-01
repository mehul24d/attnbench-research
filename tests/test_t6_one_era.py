"""Audit T2 / T3 / T6 (2026-10-01): the figures the fixes rest on, derived.

**T6 -- are 1.373x and 0.475x one experiment?** No. The L4 row is era 1
(pre-sink, `3421f89`) and the A100 row era 2 (forced sink, `56c5fff`). The
2026-10-01 sessions timed the same reference builder on both cards in era 3,
from one script, so the cross-card contrast can be read in one era. This file
pins both pairs, their eras, and the claim the docs draw from them: no cell
moves 3% or more between eras, and the sign of the contrast does not change.

**T3 -- what 1.24x is.** A whole-model prefill ratio at L4 16384/0.9 from the
session-4 sizing probe, whose log is not under version control. The banked
Stage 5 cells at the same configuration are what make it checkable.

**T2 -- the oracle's level is the implementation's.** Scoring at 32768 is
~11x a dense prefill.

The Stage 5 phases parquets were force-committed with this file (2026-10-01),
like `s12_*`: a test that skips in every clone is decoration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CLAIMS = REPO / "docs" / "claims.md"
WRITEUP = REPO / "docs" / "writeup_input.md"

L4_E1_32K = RESULTS / "stage5_32768" / "phases.parquet"
L4_E1_16K = RESULTS / "stage5_ols" / "phases.parquet"
L4_E1_16K_REPL = RESULTS / "l4_20260916_stage5_replicate" / "results" / "stage5_16384" / "phases.parquet"
A100_E2_32K = RESULTS / "a100_stage5" / "stage5_32768" / "phases.parquet"
A100_E2_16K = RESULTS / "a100_stage5" / "stage5_16384" / "phases.parquet"
L4_E3_32K = RESULTS / "s12_l4_vec_endtoend" / "vec_endtoend.parquet"
L4_E3_16K = RESULTS / "s12_l4_rerun16k_vec_endtoend" / "vec_endtoend.parquet"
A100_E3 = RESULTS / "s12_a100_vec_endtoend" / "vec_endtoend.parquet"
NEEDED = (L4_E1_32K, L4_E1_16K, L4_E1_16K_REPL, A100_E2_32K, A100_E2_16K,
          L4_E3_32K, L4_E3_16K, A100_E3)
MISSING = [str(p.relative_to(REPO)) for p in NEEDED if not p.exists()]
needs_banked = pytest.mark.skipif(bool(MISSING), reason=f"not in this checkout: {MISSING}")

SPARSITIES = (0.5, 0.75, 0.9)
# (card, band) -> (file, era, dense/sparse prefill at 0.50 / 0.75 / 0.90)
CROSS_ERA = {
    ("L4", 32768): (L4_E1_32K, 1, (1.014, 1.373, 1.475)),
    ("L4", 16384): (L4_E1_16K, 1, (1.108, 1.194, 1.258)),
    ("A100", 32768): (A100_E2_32K, 2, (0.281, 0.475, 0.817)),
    ("A100", 16384): (A100_E2_16K, 2, (0.394, 0.615, 0.951)),
}
ONE_ERA = {
    ("L4", 32768): (L4_E3_32K, (1.003, 1.344, 1.433)),
    ("L4", 16384): (L4_E3_16K, (1.097, 1.182, 1.241)),
    ("A100", 32768): (A100_E3, (0.283, 0.480, 0.836)),
    ("A100", 16384): (A100_E3, (0.403, 0.633, 0.972)),
}
ERA_SHIFT_BOUND = 0.03


def _pd():
    return pytest.importorskip("pandas")


def _close(col, x):
    return (col - x).abs() < 1e-9


def stage5_ratios(path: Path, band: int) -> tuple[float, ...]:
    d = _pd().read_parquet(path)
    d = d[(d.phase == "prefill") & (d.context_length == band)]
    dense = d[d.backend == "sdpa_flash"].ms_mean.item()
    return tuple(dense / d[(d.backend == "block_sparse") & _close(d.sparsity, s)].ms_mean.item()
                 for s in SPARSITIES)


def s12_ratios(path: Path, band: int) -> tuple[float, ...]:
    d = _pd().read_parquet(path)
    d = d[(d.band == band) & (d.builder == "reference")]
    dense = d[d.backend == "sdpa_flash"].prefill_ms_mean.item()
    return tuple(dense / d[(d.backend == "block_sparse") & _close(d.sparsity, s)].prefill_ms_mean.item()
                 for s in SPARSITIES)


def _era(path: Path) -> int:
    from attnbench.analysis import eras
    commits = set(_pd().read_parquet(path).git_commit)
    assert len(commits) == 1, (path, commits)
    try:
        return eras.era_from_git(next(iter(commits))[:7])
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"git history unavailable: {exc}")


@needs_banked
def test_the_cross_era_figures_are_what_the_docs_quote():
    for (card, band), (path, _, want) in CROSS_ERA.items():
        assert stage5_ratios(path, band) == pytest.approx(want, abs=5e-4), (card, band)


@needs_banked
def test_the_one_era_figures_are_what_the_docs_quote():
    for (card, band), (path, want) in ONE_ERA.items():
        assert s12_ratios(path, band) == pytest.approx(want, abs=5e-4), (card, band)


@needs_banked
def test_the_original_pair_really_is_two_eras_and_the_new_one_is_one():
    for key, (path, era, _) in CROSS_ERA.items():
        assert _era(path) == era, key
    assert {_era(p) for p, _ in ONE_ERA.values()} == {3}


@needs_banked
def test_the_mask_rule_does_not_carry_the_reversal():
    """Every cell within 3% of its cross-era counterpart, and the sign of the
    L4/A100 contrast is the same in one era as across two."""
    for key, (path, _) in ONE_ERA.items():
        old = stage5_ratios(CROSS_ERA[key][0], key[1])
        new = s12_ratios(path, key[1])
        for o, n in zip(old, new):
            assert abs(n / o - 1) < ERA_SHIFT_BOUND, (key, o, n)
    for band in (16384, 32768):
        assert all(r >= 1.0 for r in s12_ratios(ONE_ERA[("L4", band)][0], band))
        assert all(r < 1.0 for r in s12_ratios(A100_E3, band))


@needs_banked
def test_1_24_is_the_l4_16384_0_9_prefill_ratio_wherever_it_was_banked():
    """T3: era 1, its replicate, and era 3 all land within 2.5% of 1.244
    (1.566 / 1.259 s, the session-4 probe)."""
    probe = 1.566 / 1.259
    got = (stage5_ratios(L4_E1_16K, 16384)[2], stage5_ratios(L4_E1_16K_REPL, 16384)[2],
           s12_ratios(L4_E3_16K, 16384)[2])
    assert got == pytest.approx((1.258, 1.271, 1.241), abs=5e-4)
    assert all(abs(g / probe - 1) < 0.025 for g in got), got


@needs_banked
def test_the_oracle_scoring_pass_is_eleven_dense_prefills_at_32768():
    """T2: the level of the 35x is the implementation's."""
    d = _pd().read_parquet(L4_E1_32K)
    scoring = d[d.phase == "scoring"].ms_mean.item()
    dense = d[(d.phase == "prefill") & (d.backend == "sdpa_flash")].ms_mean.item()
    assert scoring == pytest.approx(44652, abs=1) and dense == pytest.approx(4018, abs=1)
    assert round(scoring / dense) == 11


def test_the_docs_state_the_one_era_pair_and_the_scope_of_1_24():
    """Runs in any checkout."""
    claims, writeup = CLAIMS.read_text(), WRITEUP.read_text()
    for (card, band), (_, want) in ONE_ERA.items():
        for x in want:
            assert f"{x:.3f}" in claims, (card, band, x)
    flat = lambda t: " ".join(t.split())
    # claims.md states the pair twice (the one-era paragraph and the
    # hardware-conditional row), the write-up once; each copy is pinned.
    assert flat(claims).count("1.344× against 0.480×") == 2
    assert flat(writeup).count("1.344× against 0.480×") == 1
    for text in (claims, writeup):
        assert "whole-model prefill" in text and "1.258×" in text and "1.241×" in text
