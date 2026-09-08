"""The decode-kernel confound correction (Stage 5 -> Stage 6).

CPU-only. Everything here is arithmetic over measured phases; the measuring
is Stage 5's job and is tested separately.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.analysis import decode_confound as dc
from attnbench.analysis.decode_confound import (
    PhaseModel, _dominates, correct, movement, phase_model_from)
from attnbench.analysis import pareto as pareto_mod


def _phases():
    """Two bands, dense + one sparsity, shaped like a real phases.parquet."""
    rows = []
    for band, dense_pf, sp_pf, dense_dec, sp_dec in (
            (2048, 139.2, 146.3, 34.80, 42.95),
            (8192, 694.9, 652.6, 34.16, 55.59)):
        rows += [
            dict(context_length=band, sparsity=None, phase="prefill",
                 backend="sdpa_flash", ms_mean=dense_pf),
            dict(context_length=band, sparsity=0.75, phase="prefill",
                 backend="block_sparse", ms_mean=sp_pf),
            dict(context_length=band, sparsity=None, phase="decode_step",
                 backend="sdpa_flash", ms_mean=dense_dec),
            dict(context_length=band, sparsity=0.75, phase="decode_step",
                 backend="block_sparse", ms_mean=sp_dec),
        ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The penalty itself
# --------------------------------------------------------------------------

def test_dense_arm_pays_no_decode_penalty_because_it_decodes_through_itself():
    m = phase_model_from(_phases())
    assert m.decode_penalty(8192, None) == 0.0


def test_penalty_is_the_gap_between_the_two_dense_kernels():
    """sdpa_math (the fallback) minus sdpa_flash (what dense uses)."""
    m = phase_model_from(_phases())
    assert m.decode_penalty(8192, 0.75) == pytest.approx(55.59 - 34.16)
    assert m.decode_penalty(2048, 0.75) == pytest.approx(42.95 - 34.80)


def test_penalty_grows_with_context_which_is_why_8192_was_worth_measuring():
    m = phase_model_from(_phases())
    assert m.decode_penalty(8192, 0.75) > 2 * m.decode_penalty(2048, 0.75)


def test_normalization_changes_only_prefill():
    """Sparsity is prefill-only. If a normalized total ever differed between
    arms by more than their prefill difference, the model would be crediting
    sparsity for something it does not touch."""
    m = phase_model_from(_phases())
    n = 30.0
    diff = m.normalized_total(8192, None, n) - m.normalized_total(8192, 0.75, n)
    assert diff == pytest.approx(694.9 - 652.6)


def test_phase_model_refuses_a_prefill_with_no_decode_step():
    p = _phases()
    p = p[~((p.phase == "decode_step") & (p.sparsity == 0.75))]
    with pytest.raises(ValueError, match="decode_step"):
        phase_model_from(p)


def test_phase_model_refuses_a_band_with_no_dense_decode_to_normalize_against():
    """Both dense rows for the band removed, so the earlier prefill/decode
    pairing guard has nothing to complain about and this one is the guard
    actually under test."""
    p = _phases()
    p = p[~((p.context_length == 8192) & (p.sparsity.isna()))]
    with pytest.raises(ValueError, match="nothing to normalize"):
        phase_model_from(p)


# --------------------------------------------------------------------------
# The dominance rule is COPIED from pareto. Assert the copy, don't trust it.
# --------------------------------------------------------------------------

def test_local_dominance_rule_agrees_with_paretos_on_every_ordering():
    """`_dominates` here is a scalar restatement of `pareto._dominates`. Two
    definitions of one rule is exactly the single-source-of-truth hazard this
    project keeps meeting (#22, #23), so the two are checked against each
    other rather than maintained in parallel by hope."""
    from attnbench.analysis.pareto import OperatingPoint
    vals = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (0.5, 3.0)]
    for a_lat, a_m in vals:
        for b_lat, b_m in vals:
            def op(lat, m):
                return OperatingPoint(
                    task="t", context_length=2048, epsilon=1.0,
                    backend="x", sparsity=None, latency_ms=lat, margin=m,
                    directional=False)
            assert _dominates(a_lat, a_m, b_lat, b_m) == \
                pareto_mod._dominates(op(a_lat, a_m), op(b_lat, b_m))


# --------------------------------------------------------------------------
# The correction end to end
# --------------------------------------------------------------------------

def _pareto_frame(band=8192):
    return pd.DataFrame([
        dict(task="t", context_length=band, epsilon=1.0, backend="sdpa_flash",
             sparsity=float("nan"), latency_ms=1140.0, margin=1.0,
             is_dense_reference=True, dominated_by_dense=False),
        dict(task="t", context_length=band, epsilon=1.0, backend="block_sparse",
             sparsity=0.75, latency_ms=1360.0, margin=1.0,
             is_dense_reference=False, dominated_by_dense=True),
    ])


def test_a_point_can_leave_the_dominated_set_under_the_correction():
    n = {("sdpa_flash", "t", 8192, None): 14.0,
         ("block_sparse", "t", 8192, 0.75): 14.0}
    pts = correct(_pareto_frame(), _phases(), n)
    sparse = [p for p in pts if not p.is_dense_reference][0]
    assert sparse.dominated_measured is True
    assert sparse.dominated_normalized is False
    assert sparse.normalized_ms < sparse.measured_ms


def test_the_measured_value_is_carried_beside_the_corrected_ones():
    """A corrected number that travels without what it was corrected from is
    a number someone will quote as a measurement."""
    n = {("sdpa_flash", "t", 8192, None): 14.0,
         ("block_sparse", "t", 8192, 0.75): 14.0}
    df = dc.to_dataframe(correct(_pareto_frame(), _phases(), n))
    for col in ("measured_ms", "decode_corrected_ms", "normalized_ms",
                "n_generated_own", "n_generated_common"):
        assert col in df.columns


def test_unequal_generation_length_can_move_a_point_the_OTHER_way():
    """The `vt` case. An arm that generates fewer tokens finishes sooner for
    reasons unrelated to attention speed, so correcting ONLY the decode
    kernel frees points that normalization then re-dominates. A correction
    that could only ever free points would be one nobody checked."""
    # Band 2048, where the fixture's sparse prefill (146.3) is SLOWER than
    # dense (139.2) -- so once both arms are held to the same length there is
    # nothing left to make the sparse point look fast.
    frame = _pareto_frame(band=2048)
    frame.loc[1, "latency_ms"] = 1100.0        # looks faster than dense...
    frame.loc[1, "dominated_by_dense"] = False
    n = {("sdpa_flash", "t", 2048, None): 38.8,   # ...but only because it
         ("block_sparse", "t", 2048, 0.75): 20.0}  # generated half as many
    pts = correct(frame, _phases(), n)
    sparse = [p for p in pts if not p.is_dense_reference][0]
    assert sparse.dominated_measured is False
    assert sparse.dominated_normalized is True
    assert movement(pts)["newly_dominated"] == [sparse]


def test_normalization_uses_the_dense_arms_length_for_both():
    n = {("sdpa_flash", "t", 8192, None): 38.8,
         ("block_sparse", "t", 8192, 0.75): 20.0}
    pts = correct(_pareto_frame(), _phases(), n)
    assert {p.n_generated_common for p in pts} == {38.8}
    assert {p.n_generated_own for p in pts} == {38.8, 20.0}


def test_correct_refuses_a_cell_with_no_dense_reference():
    frame = _pareto_frame()
    frame = frame[~frame.is_dense_reference]
    n = {("block_sparse", "t", 8192, 0.75): 14.0}
    with pytest.raises(ValueError, match="no dense reference"):
        correct(frame, _phases(), n)
